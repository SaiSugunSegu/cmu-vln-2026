"""Dump /camera/image frames from a rosbag to numbered PNGs.

Feeds the offline backend probe (`python -m sam_mapper.sam3_backend --frames ...`),
which needs frames in capture order but no ROS running.

    python -m sam_mapper.tools.dump_frames --bag /data/bags/scene_0 --out /data/frames
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    """Open a bag given either its directory or a specific .mcap file."""
    if os.path.isdir(bag_path):
        mcaps = sorted(glob.glob(os.path.join(bag_path, "*.mcap")))
        uri = mcaps[0] if mcaps else bag_path
    else:
        uri = bag_path

    storage_id = "mcap" if uri.endswith(".mcap") else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    return reader


_WRITABLE_HINT = (
    "Writable in the container: /tmp/... (container-local) or /data/bags/... "
    "(bind-mounted to <repo>/bags). /data itself is root-owned — only /data/bags is "
    "mounted and chmodded."
)


def ensure_writable(path: str) -> None:
    """Fail once and clearly, rather than letting cv2 warn per frame."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as err:
        raise SystemExit(f"cannot create {path}: {err}\n{_WRITABLE_HINT}") from err

    probe = os.path.join(path, ".write_probe")
    try:
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError as err:
        raise SystemExit(f"{path} is not writable: {err}\n{_WRITABLE_HINT}") from err


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="bag directory or .mcap file")
    parser.add_argument("--out", default="/tmp/frames", help="output directory for PNGs")
    parser.add_argument("--topic", default="/camera/image")
    parser.add_argument("--limit", type=int, default=40, help="max frames to write")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = parser.parse_args(argv)

    ensure_writable(args.out)
    reader = open_reader(args.bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    bridge = CvBridge()
    seen = written = 0
    shape = None
    while reader.has_next() and written < args.limit:
        _topic, data, stamp_ns = reader.read_next()
        seen += 1
        if (seen - 1) % args.stride:
            continue
        image = bridge.imgmsg_to_cv2(deserialize_message(data, Image), desired_encoding="bgr8")
        # Zero-padded index keeps lexical sort == capture order, which the probe relies on.
        path = os.path.join(args.out, f"frame_{written:05d}_{stamp_ns}.png")
        # imwrite reports failure by RETURN VALUE, not by raising. Unchecked, this loop
        # happily reports success after writing nothing at all.
        if not cv2.imwrite(path, image):
            raise SystemExit(f"failed to write {path}\n{_WRITABLE_HINT}")
        shape = image.shape
        written += 1

    if written == 0:
        raise SystemExit(f"no messages on {args.topic} in {args.bag}")
    print(f"wrote {written} frames ({shape[1]}x{shape[0]}) to {args.out}")


if __name__ == "__main__":
    main()
