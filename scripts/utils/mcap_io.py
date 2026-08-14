#!/usr/bin/env python3
"""Read a scene bag's camera/lidar/odometry without docker, ROS2, rclpy or rosbag2_py.

``scripts/eval/object_visibility.py`` reads these same three topics from inside the
ai_module container because it goes through ``rosbag2_py`` + ``cv_bridge`` +
``sensor_msgs_py``. None of that is actually needed: the bag is stored as plain CDR
messages in an mcap file (``pip install mcap mcap-ros2-support`` decodes it directly,
no ROS install required), and ``Image``/``PointCloud2`` are small, fixed-layout
messages that are simpler to unpack by hand than to pull in a whole message-conversion
stack for. The camera projection and pose-interpolation math the container path uses
is itself pure numpy/scipy -- ``ai_module/src/sam_mapper`` -- so importing it directly
from the source tree gets the exact deployed camera model with nothing extra installed.

    import utils.mcap_io as mcap_io
    track = mcap_io.read_track("data/bags/arabic_room/arabic_room_0.mcap")
    rot, trans = track.pose_at(stamp)
    pixels = mcap_io.project(world_points, rot, trans)
"""
from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Pure numpy/scipy modules, no ai_module colcon build needed -- see docstrings there.
sys.path.insert(0, str(REPO / "ai_module" / "src" / "sam_mapper"))
from sam_mapper import cloud_image_fusion, frame_sync  # noqa: E402

scan2pixels_mecanum_sim = cloud_image_fusion.scan2pixels_mecanum_sim
gather_cloud = frame_sync.gather_cloud
interpolate_odom = frame_sync.interpolate_odom

CLOUD_TOPIC = "/registered_scan"
ODOM_TOPIC = "/state_estimation"
IMAGE_TOPIC = "/camera/image"

__all__ = [
    "scan2pixels_mecanum_sim", "gather_cloud", "interpolate_odom", "project",
    "Track", "read_track", "read_image_near",
]


def project(points_world: np.ndarray, rot_b2w: np.ndarray, trans_b2w: np.ndarray) -> np.ndarray:
    """World points -> (col, row, horizontal range) on the robot's panorama.

    Unclipped on purpose, same as object_visibility.py's project(): a row outside
    [0, height) is itself the answer to "could the camera see it".
    """
    body = (points_world - trans_b2w) @ rot_b2w
    return scan2pixels_mecanum_sim(body, clip=False)


def _stamp_s(ros_msg) -> float:
    h = ros_msg.header
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _decode_cloud_xyz(msg) -> np.ndarray:
    """PointCloud2 -> (N,3) float32 xyz, reading the fields/point_step it declares
    rather than assuming a fixed layout (this bag's lidar also carries `intensity`,
    which we don't need here)."""
    offsets = {f.name: (f.offset, f.datatype) for f in msg.fields}
    missing = [a for a in ("x", "y", "z") if a not in offsets]
    if missing:
        raise ValueError(f"PointCloud2 is missing field(s) {missing}: has {list(offsets)}")
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)[: n * msg.point_step]
    raw = raw.reshape(n, msg.point_step)
    dtype = np.dtype(">f4") if msg.is_bigendian else np.dtype("<f4")
    cols = []
    for axis in ("x", "y", "z"):
        offset, datatype = offsets[axis]
        if datatype != 7:  # sensor_msgs/PointField.FLOAT32
            raise ValueError(f"field {axis!r} is not float32 (datatype={datatype})")
        cols.append(raw[:, offset:offset + 4].copy().view(dtype)[:, 0])
    return np.stack(cols, axis=1).astype(np.float32)


def _decode_image_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 uint8, BGR (cv2's native order)."""
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = arr[: msg.height * msg.step].reshape(msg.height, msg.step)[:, : msg.width * 3]
    arr = arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


class Track:
    """One scene bag's odometry (interpolatable) and lidar (windowable), read once."""

    def __init__(self, odom_stack, odom_stamps, clouds, cloud_stamps):
        self._odom_pose_stack = [
            {"position": p, "orientation": q, "linear_velocity": [0.0] * 3,
             "angular_velocity": [0.0] * 3}
            for p, q in odom_stack
        ]
        self.odom_stamps = odom_stamps
        self.clouds = clouds
        self.cloud_stamps = cloud_stamps

    def pose_at(self, stamp: float):
        """(rot_b2w 3x3, trans_b2w (3,)) at `stamp`, or (None, None) if odometry
        does not bracket it."""
        from scipy.spatial.transform import Rotation

        odom, status = frame_sync.interpolate_odom(self._odom_pose_stack, self.odom_stamps, stamp)
        if status != frame_sync.OK:
            return None, None
        return Rotation.from_quat(odom["orientation"]).as_matrix(), np.asarray(odom["position"], float)

    def cloud_near(self, stamp: float, before: float = 0.5, after: float = 0.1):
        """Lidar returns (world frame) in the mapper's own fusion window around `stamp`."""
        return frame_sync.gather_cloud(self.clouds, self.cloud_stamps, stamp, before, after)


def read_track(bag_path: str | Path) -> Track:
    """One pass over the mcap for /state_estimation + /registered_scan."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    odom_stack, odom_stamps, clouds, cloud_stamps = [], [], [], []
    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, ros_msg in reader.iter_decoded_messages(
            topics=[ODOM_TOPIC, CLOUD_TOPIC]
        ):
            if channel.topic == ODOM_TOPIC:
                p, q = ros_msg.pose.pose.position, ros_msg.pose.pose.orientation
                odom_stack.append(([p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
                odom_stamps.append(_stamp_s(ros_msg))
            else:
                clouds.append(_decode_cloud_xyz(ros_msg))
                cloud_stamps.append(_stamp_s(ros_msg))
    return Track(odom_stack, odom_stamps, clouds, cloud_stamps)


def read_image_near(bag_path: str | Path, stamps: Iterable[float],
                    tol: float = 0.2) -> dict[float, np.ndarray]:
    """Decode only the /camera/image frames nearest each of `stamps`.

    A caller that wants a handful of objects' best views should not have to decode all
    ~375 frames of a scene's panorama to get them.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    wanted = sorted(set(stamps))
    if not wanted:
        return {}
    best: dict[float, tuple[float, np.ndarray]] = {}
    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, ros_msg in reader.iter_decoded_messages(topics=[IMAGE_TOPIC]):
            stamp = _stamp_s(ros_msg)
            idx = min(bisect_left(wanted, stamp), len(wanted) - 1)
            for target in wanted[max(0, idx - 1):idx + 2]:
                delta = abs(target - stamp)
                if delta <= tol and (target not in best or delta < best[target][0]):
                    best[target] = (delta, _decode_image_bgr(ros_msg))
    return {stamp: image for stamp, (_delta, image) in best.items()}
