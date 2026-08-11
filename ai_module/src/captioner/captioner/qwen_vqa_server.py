#!/usr/bin/env python3
"""Persistent Qwen-VL VQA server (loads weights once).

ROS topics
  /qwen_vqa/status   (std_msgs/String, TRANSIENT_LOCAL)  "loading" | "ready" | "error:…"
  /qwen_vqa/request  (std_msgs/String JSON)
      {"id": "<uuid>", "image": "/abs/path.png", "question": "How many…"}
      {"id": "<uuid>", "image": null, "question": "…", "mode": "freeform"}
      {"id": "<uuid>", "images": ["/a.png", "/b.png"], "question": "How many…"}
      mode: "numerical" (default) | "freeform" (no integer-only wrapper)
  /qwen_vqa/response (std_msgs/String JSON)
      {"id": "…", "answer": "4", "number": 4, "error": null, "seconds": 0.5}

`images` shows the model several views of one scene in a single call, which is what
lets a counting question draw on more than the one best view that happens to rank
first. `image` is the one-view spelling of the same thing and stays supported: most
callers send exactly one crop and should not have to wrap it in a list.

Requests are served by a single worker thread: one GPU, one generate() at a time.
Extra requests queue up to QUEUE_DEPTH and are rejected with error="busy" beyond
that, so a client republishing at 1 Hz cannot grow the backlog without bound.

Keep this node running; ask with `qwen_vqa_ask` / `just vqa-ask`.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional

import imageio.v3 as iio
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from captioner.models.captioning import load_qwen_backend
from captioner.paths import secure_image_path
from captioner.qwen_vqa_protocol import (
    REQUEST_TOPIC,
    RESPONSE_TOPIC,
    STATUS_TOPIC,
    parse_vqa_request,
)

# Tiny RGB placeholder so VL backends accept text-only extract / planning prompts.
_BLANK_RGB = np.zeros((64, 64, 3), dtype=np.uint8)

# Deep enough to absorb a burst from the reasoner (extract + answer + attribute
# captions), shallow enough that a stuck client is rejected rather than queued.
QUEUE_DEPTH = 16


class QwenVQAServer(Node):
    def __init__(self):
        super().__init__("qwen_vqa_server")

        self.declare_parameter("captioning_model", "qwen3vl")
        self.declare_parameter("quantization", "int4")
        self.declare_parameter("model_id", "")
        self.declare_parameter("max_new_tokens", 32)
        self.declare_parameter("max_pixels", 1280 * 28 * 28)

        captioning_model = self.get_parameter("captioning_model").value
        quantization = self.get_parameter("quantization").value
        model_id = self.get_parameter("model_id").value or None
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        max_pixels = int(self.get_parameter("max_pixels").value)

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, status_qos)
        self.response_pub = self.create_publisher(String, RESPONSE_TOPIC, 10)

        self._publish_status("loading")
        self.get_logger().info(
            f"Loading {captioning_model} (quantization={quantization})…")
        t0 = time.time()
        try:
            self.model = load_qwen_backend(
                captioning_model,
                quantization=quantization,
                model_id=model_id,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                max_pixels=max_pixels,
            )
        except Exception as exc:
            # Without this, a load failure (OOM, missing offline checkpoint) kills
            # the node while `qwen_vqa_wait_ready` blocks for its full timeout.
            self.get_logger().error(f"Model load failed: {type(exc).__name__}: {exc}")
            self._publish_status(f"error:{type(exc).__name__}")
            # Let the transient-local sample reach subscribers before we die.
            rclpy.spin_once(self, timeout_sec=0.5)
            raise

        self.max_new_tokens = max_new_tokens

        # One worker: the GPU serialises generate() anyway, and a thread per
        # request would grow without bound while they all waited their turn.
        self._requests: queue.Queue[str] = queue.Queue(maxsize=QUEUE_DEPTH)
        self._worker = threading.Thread(
            target=self._worker_loop, name="vqa_worker", daemon=True)
        self._worker.start()

        self.create_subscription(String, REQUEST_TOPIC, self._on_request, 10)

        self.get_logger().info(
            f"Ready: {self.model.model_id} in {time.time() - t0:.1f}s "
            f"(ask on {REQUEST_TOPIC})")
        self._publish_status("ready")

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))

    def _publish_response(self, payload: dict):
        self.response_pub.publish(String(data=json.dumps(payload)))

    def _publish_error(self, req_id: Optional[str], error: str):
        self._publish_response({
            "id": req_id,
            "answer": None,
            "number": None,
            "error": error,
            "seconds": 0.0,
        })

    def _on_request(self, msg: String):
        try:
            self._requests.put_nowait(msg.data)
        except queue.Full:
            # Recover the id if we can, so the client fails fast instead of
            # waiting out its own timeout.
            req_id = None
            try:
                req_id = json.loads(msg.data).get("id")
            except (json.JSONDecodeError, AttributeError):
                pass
            self.get_logger().warn(f"request queue full ({QUEUE_DEPTH}); rejecting id={req_id}")
            self._publish_error(req_id, "busy")

    def _worker_loop(self):
        while rclpy.ok():
            try:
                raw = self._requests.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle_request(raw)
            finally:
                self._requests.task_done()

    def _handle_request(self, raw: str):
        req_id = None
        try:
            req_id, question, image_paths, freeform, max_new_tokens = \
                parse_vqa_request(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"rejected request: {exc}")
            self._publish_error(req_id, f"invalid request: {exc}")
            return

        try:
            images = []
            for image_path in image_paths:
                path = secure_image_path(image_path)
                image = iio.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                images.append(image)

            budget = max_new_tokens or self.max_new_tokens
            t0 = time.time()
            if len(images) > 1:
                # One prompt, several views. Kept off answer_questions because that
                # one pairs N images with N questions for batching, so handing it a
                # list here would ask the same question of each view separately and
                # return N answers that then have to be reconciled.
                raw_answer = self.model.answer_multi_image(
                    images, question,
                    max_new_tokens=budget,
                    freeform=freeform,
                )
            else:
                answers = self.model.answer_questions(
                    [images[0] if images else _BLANK_RGB], [question],
                    max_new_tokens=budget,
                    freeform=freeform,
                )
                raw_answer = answers[0]
            elapsed = time.time() - t0

            number = self.model.extract_integer(raw_answer)
            self._publish_response({
                "id": req_id,
                "answer": raw_answer,
                "number": number,
                "error": None,
                "seconds": round(elapsed, 3),
            })
            self.get_logger().info(
                f"id={req_id} views={len(images)} number={number} ({elapsed:.2f}s)")
        except Exception as exc:  # noqa: BLE001 — one bad request must not kill the server
            # Report the exception type only: the message can carry filesystem
            # paths, and requests arrive from other processes.
            self.get_logger().error(
                f"VQA failed ({type(exc).__name__}: {exc}) id={req_id}")
            self._publish_error(req_id, type(exc).__name__)


def main(args: Optional[list] = None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = QwenVQAServer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
