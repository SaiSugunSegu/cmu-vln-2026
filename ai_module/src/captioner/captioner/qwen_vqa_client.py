#!/usr/bin/env python3
"""Ask a question of the persistent qwen_vqa_server over ROS topics."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from captioner.qwen_vqa_topics import REQUEST_TOPIC, RESPONSE_TOPIC


class VQAClient(Node):
    def __init__(self):
        super().__init__("qwen_vqa_client")
        self._response: Optional[dict] = None
        self._req_id = str(uuid.uuid4())
        self.create_subscription(String, RESPONSE_TOPIC, self._on_response, 10)
        self.pub = self.create_publisher(String, REQUEST_TOPIC, 10)

    def _on_response(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if payload.get("id") == self._req_id:
            self._response = payload

    def ask(self, image: str, question: str, timeout_s: float) -> dict:
        # Give the subscription time to connect before publishing.
        deadline_connect = time.time() + 2.0
        while time.time() < deadline_connect and self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)

        req = {"id": self._req_id, "image": image, "question": question}
        self.pub.publish(String(data=json.dumps(req)))

        deadline = time.time() + timeout_s
        while self._response is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        if self._response is None:
            raise TimeoutError(
                f"No response on {RESPONSE_TOPIC} within {timeout_s:.0f}s "
                f"(is qwen_vqa_server running?)")
        return self._response


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(
        description="Ask the persistent Qwen VQA server (model already loaded).")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--image", "-i", required=True,
                        help="Absolute path visible inside the AI container.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--raw", action="store_true",
                        help="Print raw model text instead of parsed integer.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full response JSON.")
    args = parser.parse_args(argv)

    rclpy.init()
    try:
        client = VQAClient()
        try:
            result = client.ask(args.image, args.question, args.timeout)
        finally:
            client.destroy_node()
    finally:
        rclpy.shutdown()

    if result.get("error"):
        print(result["error"], file=sys.stderr)
        if args.json:
            print(json.dumps(result, indent=2))
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.raw or result.get("number") is None:
        print(result.get("answer") or "")
    else:
        print(result["number"])

    if result.get("number") is None and not args.raw and not args.json:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
