"""Unit tests for the cv_bridge-free sensor_msgs/Image -> RGB conversion.

qwen_numerical imports rclpy at module scope, so this file skips itself when ROS
is absent rather than failing collection on a bare host.
"""
from __future__ import annotations

import numpy as np
import pytest

qwen_numerical = pytest.importorskip(
    "smart_vlm.qwen_numerical", reason="needs rclpy / sensor_msgs (run in container)")
image_msg_to_rgb = qwen_numerical.image_msg_to_rgb


class FakeImage:
    """Minimal stand-in for sensor_msgs/Image (only the fields we read)."""

    def __init__(self, array: np.ndarray, encoding: str, step: int | None = None):
        self.height, self.width = array.shape[:2]
        self.encoding = encoding
        self.data = array.tobytes()
        self.step = step if step is not None else array.shape[1] * array.shape[2]


def _gradient(channels: int) -> np.ndarray:
    return np.arange(4 * 5 * channels, dtype=np.uint8).reshape(4, 5, channels)


def test_rgb8_passes_through():
    src = _gradient(3)
    np.testing.assert_array_equal(image_msg_to_rgb(FakeImage(src, "rgb8")), src)


def test_bgr8_is_flipped_to_rgb():
    src = _gradient(3)
    np.testing.assert_array_equal(
        image_msg_to_rgb(FakeImage(src, "bgr8")), src[:, :, ::-1])


def test_rgba8_drops_alpha():
    src = _gradient(4)
    np.testing.assert_array_equal(
        image_msg_to_rgb(FakeImage(src, "rgba8")), src[:, :, :3])


def test_bgra8_drops_alpha_and_flips():
    src = _gradient(4)
    np.testing.assert_array_equal(
        image_msg_to_rgb(FakeImage(src, "bgra8")), src[:, :, :3][:, :, ::-1])


def test_unknown_encoding_with_wide_step_falls_back_to_bgr():
    # Unity publishes 3-channel bytes under a non-standard encoding name.
    src = _gradient(3)
    np.testing.assert_array_equal(
        image_msg_to_rgb(FakeImage(src, "unity_rgb")), src[:, :, ::-1])


def test_unsupported_encoding_raises():
    src = np.zeros((4, 5, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image encoding"):
        image_msg_to_rgb(FakeImage(src, "mono8", step=5))


def test_output_is_owned_not_a_view_of_the_message_buffer():
    # np.frombuffer over msg.data is read-only; a view would break downstream
    # writes and keep the whole message alive.
    src = _gradient(3)
    out = image_msg_to_rgb(FakeImage(src, "rgb8"))
    assert out.flags.writeable
