#!/usr/bin/env python3
"""Answer numerical challenge questions from the live camera image.

Subscribes:
  /challenge_question  (std_msgs/String)
  /camera/image        (sensor_msgs/Image)
  /qwen_vqa/{status,response}

Publishes:
  /numerical_response  (std_msgs/Int32)
  /qwen_vqa/request

Inference runs in the shared `qwen_vqa_server` process rather than here. Loading
the checkpoint in-process too would put a second full copy of the weights on the
GPU whenever the category-1 flow (which requires the server) is also running.
Start the server first — `just vqa-up`.

Only handles questions classified as numerical ("How many…", "Count…"); find and
instruction questions are simply ignored here.

NOT part of `smart_vlm.launch` — run it by hand:

    ros2 run smart_vlm qwen_numerical

It is the SAM-independent counterpart to `numerical_reasoner`, which is the head the
launch and the eval harness actually use. That one answers from SAM best-view crops; this
one asks Qwen about the whole 360° panorama, so it still produces something when the
detector finds nothing at all.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from captioner.qwen_vqa_protocol import REQUEST_TOPIC, RESPONSE_TOPIC, STATUS_TOPIC
from captioner.ros_utils import wait_for_subscriber
from smart_vlm.question import QuestionType, classify


def image_msg_to_rgb(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to HxWx3 uint8 RGB without cv_bridge."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    encoding = (msg.encoding or "").lower()

    if encoding in ("rgb8", "bgr8"):
        img = arr.reshape(msg.height, msg.width, 3)
        if encoding == "bgr8":
            return img[:, :, ::-1].copy()
        return img.copy()

    if encoding in ("rgba8", "bgra8"):
        img = arr.reshape(msg.height, msg.width, 4)[:, :, :3]
        if encoding == "bgra8":
            return img[:, :, ::-1].copy()
        return img.copy()

    # Unity / captioner path often packs 3-channel bytes; treat as BGR then flip
    # to RGB to match captioning_node.handle_rgb (.flip((-1))).
    if msg.step >= msg.width * 3:
        img = arr.reshape(msg.height, msg.width, -1)[:, :, :3]
        return img[:, :, ::-1].copy()

    raise ValueError(f"Unsupported image encoding={msg.encoding!r}")


class QwenNumericalNode(Node):
    def __init__(self):
        super().__init__("qwen_numerical")

        self.declare_parameter("camera_topic", "/camera/image")
        self.declare_parameter("max_new_tokens", 16)
        self.declare_parameter("image_wait_s", 5.0)
        self.declare_parameter("vqa_timeout_s", 120.0)
        self.declare_parameter("server_wait_s", 600.0)
        # Must be inside a mount the VQA server is allowed to read (captioner.paths).
        self.declare_parameter("scratch_dir", "/data/runs/qwen_numerical")

        camera_topic = self.get_parameter("camera_topic").value
        self.max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        self.image_wait_s = float(self.get_parameter("image_wait_s").value)
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        self.server_wait_s = float(self.get_parameter("server_wait_s").value)
        self.scratch_dir = Path(str(self.get_parameter("scratch_dir").value))

        self._latest_rgb: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()
        self._answered = False
        self._busy = False
        self._question: Optional[str] = None

        self._vqa_lock = threading.Lock()
        self._vqa_wait_id: Optional[str] = None
        self._vqa_response: Optional[dict] = None
        self._vqa_event = threading.Event()
        self._server_status: Optional[str] = None

        latch_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        sensor_cb = MutuallyExclusiveCallbackGroup()
        question_cb = MutuallyExclusiveCallbackGroup()
        vqa_cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Image, camera_topic, self._on_image, 1, callback_group=sensor_cb)
        self.create_subscription(
            String, "/challenge_question", self._on_question, 10,
            callback_group=question_cb)
        self.create_subscription(
            String, RESPONSE_TOPIC, self._on_vqa_response, 10, callback_group=vqa_cb)
        self.create_subscription(
            String, STATUS_TOPIC, self._on_vqa_status, latch_qos, callback_group=vqa_cb)

        self.pub_int = self.create_publisher(Int32, "/numerical_response", 10)
        self.pub_vqa_req = self.create_publisher(String, REQUEST_TOPIC, 10)

        self.get_logger().info(
            "qwen_numerical ready — waiting for How many / Count questions "
            f"(inference via {REQUEST_TOPIC}; start the server with `just vqa-up`)")

    # -- VQA server plumbing --------------------------------------------------

    def _on_vqa_status(self, msg: String):
        self._server_status = (msg.data or "").strip()

    def _on_vqa_response(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._vqa_lock:
            if self._vqa_wait_id is None or payload.get("id") != self._vqa_wait_id:
                return
            self._vqa_response = payload
            self._vqa_event.set()

    def _wait_for_server(self) -> None:
        if self._server_status == "ready":
            return
        deadline = time.time() + self.server_wait_s
        while time.time() < deadline and rclpy.ok():
            if self._server_status == "ready":
                return
            if self._server_status and self._server_status.startswith("error"):
                raise RuntimeError(f"qwen_vqa_server failed: {self._server_status}")
            time.sleep(0.2)
        raise TimeoutError(
            f"qwen_vqa_server not ready within {self.server_wait_s:.0f}s "
            f"(status={self._server_status!r}); start it with `just vqa-up`")

    def _ask_vqa(self, question: str, image_path: Path) -> dict:
        req_id = str(uuid.uuid4())
        with self._vqa_lock:
            self._vqa_wait_id = req_id
            self._vqa_response = None
            self._vqa_event.clear()

        wait_for_subscriber(self.pub_vqa_req)
        self.pub_vqa_req.publish(String(data=json.dumps({
            "id": req_id,
            "image": str(image_path),
            "question": question,
            "max_new_tokens": self.max_new_tokens,
            "mode": "numerical",
        })))

        if not self._vqa_event.wait(timeout=self.vqa_timeout_s):
            raise TimeoutError(f"No VQA response within {self.vqa_timeout_s:.0f}s")

        with self._vqa_lock:
            response = self._vqa_response or {}
            self._vqa_wait_id = None
        if response.get("error"):
            raise RuntimeError(f"VQA error: {response['error']}")
        return response

    # -- question handling ----------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            rgb = image_msg_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        with self._image_lock:
            self._latest_rgb = rgb

    def _on_question(self, msg: String):
        # Eval repeats the question at 1 Hz — answer only once.
        if self._answered or self._busy or self._question is not None:
            return
        if classify(msg.data) is not QuestionType.NUMERICAL:
            return

        self._question = msg.data
        self._busy = True
        self.get_logger().info(f"NUMERICAL QUESTION: {self._question}")
        threading.Thread(target=self._answer_worker, daemon=True).start()

    def _wait_for_image(self) -> Optional[np.ndarray]:
        deadline = time.time() + self.image_wait_s
        while time.time() < deadline and rclpy.ok():
            with self._image_lock:
                if self._latest_rgb is not None:
                    return self._latest_rgb.copy()
            time.sleep(0.05)
        return None

    def _answer_worker(self):
        frame_path = None
        try:
            self._wait_for_server()

            image = self._wait_for_image()
            if image is None:
                self.get_logger().error(
                    "No /camera/image within wait window; not publishing")
                return

            # The server reads images from disk, so hand off via a file under a
            # mount it is allowed to open rather than over the topic.
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            frame_path = self.scratch_dir / f"frame_{uuid.uuid4().hex[:8]}.png"
            iio.imwrite(str(frame_path), image)

            t0 = time.time()
            response = self._ask_vqa(self._question, frame_path)
            elapsed = time.time() - t0

            number = response.get("number")
            if number is None:
                self.get_logger().error(
                    f"Could not parse an integer from model reply "
                    f"{response.get('answer')!r} after {elapsed:.1f}s")
                return

            self.pub_int.publish(Int32(data=int(number)))
            self._answered = True
            self.get_logger().info(
                f"Published /numerical_response={number} in {elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001 — keep the node alive
            self.get_logger().error(
                f"Qwen numerical inference failed: {type(exc).__name__}: {exc}")
        finally:
            if frame_path is not None:
                try:
                    frame_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = QwenNumericalNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
