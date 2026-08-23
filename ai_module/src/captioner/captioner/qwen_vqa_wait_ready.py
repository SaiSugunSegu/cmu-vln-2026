#!/usr/bin/env python3
"""Block until /qwen_vqa/status == ready (used by `just vqa-up`)."""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from captioner.qwen_vqa_protocol import STATUS_TOPIC


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = rclpy.create_node("qwen_vqa_wait_ready")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    box = {"status": None}

    def _cb(msg: String):
        box["status"] = msg.data

    node.create_subscription(String, STATUS_TOPIC, _cb, qos)
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            status = box["status"]
            if status == "ready":
                print("VQA server ready.")
                return 0
            if status and str(status).startswith("error"):
                print(status, file=sys.stderr)
                return 1
        print(
            f"Timed out waiting for {STATUS_TOPIC}=ready",
            file=sys.stderr,
        )
        print(
            "See: docker exec iros2026_ai_module tail -n 50 /tmp/qwen_vqa_server.log",
            file=sys.stderr,
        )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
