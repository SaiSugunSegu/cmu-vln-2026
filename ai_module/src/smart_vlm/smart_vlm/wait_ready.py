#!/usr/bin/env python3
"""Block until a latched pipeline gate fires, then exit — a shell-callable barrier.

Used by bag_replay.launch to hold playback until smart_vlm publishes /pipeline/armed,
so the bag never streams past a detector that is still loading weights or still holding
the config's placeholder prompts.

`ros2 topic echo --once` cannot do this job: its subscriber is VOLATILE, so it misses an
already-latched sample and hangs forever if the gate fired before it started.

  ros2 run smart_vlm wait_ready --topic /pipeline/armed --timeout 300

Exits 0 when the gate fires, 1 on timeout.
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def wait_for(topic: str, timeout_s: float) -> bool:
    node = Node("wait_ready")
    seen: list[str] = []
    # Must match the publisher's durability, or the latched sample is never delivered.
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    node.create_subscription(String, topic, lambda msg: seen.append(msg.data), qos)

    deadline = time.monotonic() + timeout_s
    try:
        while not seen and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        if seen:
            node.get_logger().info(f"{topic}: {seen[0]}")
        else:
            node.get_logger().error(f"timed out after {timeout_s:.0f}s waiting for {topic}")
        return bool(seen)
    finally:
        node.destroy_node()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/pipeline/armed")
    parser.add_argument("--timeout", type=float, default=300.0)
    # ros2 run appends --ros-args; parse_known_args keeps it out of our way.
    args, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init(args=ros_args or None)
    try:
        ok = wait_for(args.topic, args.timeout)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
