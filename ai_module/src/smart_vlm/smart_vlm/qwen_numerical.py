#!/usr/bin/env python3
"""Answer numerical challenge questions with Qwen-VL on the live camera image.

Subscribes:
  /challenge_question  (std_msgs/String)
  /camera/image        (sensor_msgs/Image)

Publishes:
  /numerical_response  (std_msgs/Int32)

Only handles questions classified as numerical ("How many…", "Count…").
Find / instruction questions are left to dummy_vlm (or later reasoner heads).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from captioner.models.captioning import load_qwen_backend
from smart_vlm.supervisor import classify


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

        self.declare_parameter("captioning_model", "qwen3vl")
        self.declare_parameter("quantization", "int4")
        self.declare_parameter("model_id", "")
        self.declare_parameter("camera_topic", "/camera/image")
        self.declare_parameter("max_new_tokens", 16)
        self.declare_parameter("max_pixels", 1280 * 28 * 28)
        self.declare_parameter("image_wait_s", 5.0)

        captioning_model = self.get_parameter("captioning_model").value
        quantization = self.get_parameter("quantization").value
        model_id = self.get_parameter("model_id").value or None
        camera_topic = self.get_parameter("camera_topic").value
        self.max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        max_pixels = int(self.get_parameter("max_pixels").value)
        self.image_wait_s = float(self.get_parameter("image_wait_s").value)

        self.get_logger().info(
            f"Loading {captioning_model} (quantization={quantization})…")
        load_t0 = time.time()
        self.model = load_qwen_backend(
            captioning_model,
            quantization=quantization,
            model_id=model_id,
            batch_size=1,
            max_new_tokens=self.max_new_tokens,
            max_pixels=max_pixels,
        )
        self.get_logger().info(
            f"Loaded {self.model.model_id} in {time.time() - load_t0:.1f}s")

        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._image_lock = threading.Lock()
        self._answered = False
        self._busy = False
        self._question: Optional[str] = None

        sensor_cb = MutuallyExclusiveCallbackGroup()
        question_cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Image, camera_topic, self._on_image, 1, callback_group=sensor_cb)
        self.create_subscription(
            String, "/challenge_question", self._on_question, 10,
            callback_group=question_cb)
        self.pub_int = self.create_publisher(Int32, "/numerical_response", 10)

        self.get_logger().info(
            "qwen_numerical ready — waiting for How many / Count questions")

    def _on_image(self, msg: Image):
        try:
            rgb = image_msg_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return
        with self._image_lock:
            self._latest_rgb = rgb
            self._latest_stamp = msg.header.stamp

    def _on_question(self, msg: String):
        # Eval repeats the question at 1 Hz — answer only once.
        if self._answered or self._busy or self._question is not None:
            return
        if classify(msg.data) != "numerical":
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
        try:
            image = self._wait_for_image()
            if image is None:
                self.get_logger().error(
                    "No /camera/image within wait window; not publishing")
                return

            t0 = time.time()
            numbers = self.model.answer_numerical(
                [image], [self._question], max_new_tokens=self.max_new_tokens)
            number = numbers[0]
            elapsed = time.time() - t0

            if number is None:
                self.get_logger().error(
                    f"Could not parse an integer from model reply "
                    f"after {elapsed:.1f}s")
                return

            self.pub_int.publish(Int32(data=int(number)))
            self._answered = True
            self.get_logger().info(
                f"Published /numerical_response={number} in {elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001 — keep the node alive
            # Avoid leaking internals to clients; log a short operational error.
            self.get_logger().error(f"Qwen numerical inference failed: {type(exc).__name__}")
        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = QwenNumericalNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
