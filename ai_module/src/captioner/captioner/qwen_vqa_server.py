#!/usr/bin/env python3
"""Persistent Qwen-VL VQA server (loads weights once).

ROS topics
  /qwen_vqa/status   (std_msgs/String, TRANSIENT_LOCAL)  "loading" | "ready" | "error:…"
  /qwen_vqa/request  (std_msgs/String JSON)
      {"id": "<uuid>", "image": "/abs/path.png", "question": "How many…"}
  /qwen_vqa/response (std_msgs/String JSON)
      {"id": "…", "answer": "4", "number": 4, "error": null, "seconds": 0.5}

Keep this node running; ask with `qwen_vqa_ask` / `just vqa-ask`.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import imageio.v3 as iio
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from captioner.models.captioning import load_qwen_backend
from captioner.qwen_vqa_topics import REQUEST_TOPIC, RESPONSE_TOPIC, STATUS_TOPIC

def _allowed_roots() -> tuple[Path, ...]:
    """Roots visible via compose mounts (data/, ai_module, ${HOME}:${HOME})."""
    candidates = [
        Path("/data/workspace"),
        Path("/home/docker"),
        Path.home(),
    ]
    # Host HOME is bind-mounted at the same absolute path; container HOME is
    # usually /home/docker, so also accept other /home/<user> mounts that exist.
    home = Path("/home")
    if home.is_dir():
        candidates.extend(p for p in home.iterdir() if p.is_dir())
    return tuple(p.resolve() for p in candidates if p.exists())


def _secure_image_path(user_path: str) -> Path:
    decoded = unquote(user_path)
    if ".." in Path(decoded).parts:
        raise ValueError("Path traversal rejected")
    path = Path(decoded).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    allowed = any(
        str(path) == str(root) or str(path).startswith(str(root) + "/")
        for root in _allowed_roots()
    )
    if not allowed:
        raise PermissionError(f"Image path not under allowed mounts: {path}")
    return path


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
        self.create_subscription(String, REQUEST_TOPIC, self._on_request, 10)

        self._publish_status("loading")
        self.get_logger().info(
            f"Loading {captioning_model} (quantization={quantization})…")
        t0 = time.time()
        self.model = load_qwen_backend(
            captioning_model,
            quantization=quantization,
            model_id=model_id,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            max_pixels=max_pixels,
        )
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self.get_logger().info(
            f"Ready: {self.model.model_id} in {time.time() - t0:.1f}s "
            f"(ask on {REQUEST_TOPIC})")
        self._publish_status("ready")

    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))

    def _on_request(self, msg: String):
        threading.Thread(
            target=self._handle_request, args=(msg.data,), daemon=True
        ).start()

    def _handle_request(self, raw: str):
        req_id = None
        try:
            payload = json.loads(raw)
            req_id = payload.get("id")
            image_path = payload["image"]
            question = payload["question"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._publish_response({
                "id": req_id,
                "answer": None,
                "number": None,
                "error": "invalid request JSON (need id, image, question)",
                "seconds": 0.0,
            })
            return

        try:
            path = _secure_image_path(image_path)
            image = iio.imread(str(path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")

            with self._lock:
                t0 = time.time()
                answers = self.model.answer_questions(
                    [image], [question], max_new_tokens=self.max_new_tokens)
                elapsed = time.time() - t0

            raw_answer = answers[0]
            number = self.model.extract_integer(raw_answer)
            self._publish_response({
                "id": req_id,
                "answer": raw_answer,
                "number": number,
                "error": None,
                "seconds": round(elapsed, 3),
            })
            self.get_logger().info(
                f"id={req_id} number={number} ({elapsed:.2f}s)")
        except Exception as exc:  # noqa: BLE001 — keep server alive
            self.get_logger().error(
                f"VQA failed ({type(exc).__name__}) id={req_id}")
            self._publish_response({
                "id": req_id,
                "answer": None,
                "number": None,
                "error": type(exc).__name__,
                "seconds": 0.0,
            })

    def _publish_response(self, payload: dict):
        self.response_pub.publish(String(data=json.dumps(payload)))


def main(args: Optional[list] = None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = QwenVQAServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
