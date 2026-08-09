#!/usr/bin/env python3
"""Ask a question of the persistent qwen_vqa_server over ROS topics."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from captioner.crop_dir import list_crop_images, question_from_crop_dir
from captioner.paths import secure_image_path, secure_path
from captioner.qwen_vqa_topics import REQUEST_TOPIC, RESPONSE_TOPIC
from captioner.ros_utils import wait_for_subscriber


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

    def ask(
            self,
            question: str,
            timeout_s: float,
            *,
            image: Optional[str] = None,
            images: Optional[list[str]] = None,
            instruction: Optional[str] = None,
            mode: str = "numerical",
            ) -> dict:
        wait_for_subscriber(
            self.pub, spin=lambda t: rclpy.spin_once(self, timeout_sec=t))

        req: dict = {"id": self._req_id, "question": question, "mode": mode}
        if images:
            req["images"] = list(images)
        else:
            req["image"] = image
        if instruction and str(instruction).strip():
            req["instruction"] = str(instruction).strip()
        self.pub.publish(String(data=json.dumps(req)))

        deadline = time.time() + timeout_s
        while self._response is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        if self._response is None:
            raise TimeoutError(
                f"No response on {RESPONSE_TOPIC} within {timeout_s:.0f}s "
                f"(is qwen_vqa_server running? try `just vqa-status`)")
        return self._response


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the persistent Qwen VQA server (model already loaded).")
    parser.add_argument("--question", "-q", default=None,
                        help="Question text (optional with --crop-dir).")
    parser.add_argument("--image", "-i", action="append", default=[],
                        help="Absolute path visible inside the AI container. "
                             "Repeat for multi-image context.")
    parser.add_argument(
        "--crop-dir", default=None,
        help="Eval-crops run directory under /data: load best_rank*.png "
             "(or annotated/ with --annotated) and derive the question from "
             "the folder name.")
    parser.add_argument(
        "--annotated", action="store_true",
        help="With --crop-dir, use annotated/*.png overlays instead of clean crops.")
    parser.add_argument(
        "--instruction", "-I", default=None,
        help="Extra textual guidance added to the prompt "
             "(e.g. counting rules across views).")
    parser.add_argument(
        "--freeform", action="store_true",
        help="Do not wrap with the integer-only numerical prompt.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--raw", action="store_true",
                        help="Print raw model text instead of parsed integer.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full response JSON.")
    args = parser.parse_args(argv)

    image_paths: list[str]
    question: str

    if args.crop_dir:
        if args.image:
            parser.error("Pass either --crop-dir or --image, not both.")
        crop_root = Path(args.crop_dir).expanduser().resolve()
        if not crop_root.is_dir():
            print(f"crop-dir is not a directory: {crop_root}", file=sys.stderr)
            return 1
        crop_dir = secure_path(str(crop_root), roots=(crop_root,))
        # Confine image reads to this crop run (root or annotated/).
        image_paths = [
            str(secure_image_path(str(p), roots=(crop_dir,)))
            for p in list_crop_images(crop_dir, annotated=args.annotated)
        ]
        question = args.question or question_from_crop_dir(crop_dir)
    else:
        if not args.image:
            parser.error("at least one --image / -i is required (or use --crop-dir)")
        if not args.question or not str(args.question).strip():
            parser.error("--question is required when not using --crop-dir")
        image_paths = list(args.image)
        question = args.question

    mode = "freeform" if args.freeform else "numerical"
    ask_kwargs: dict = {
        "instruction": args.instruction,
        "mode": mode,
    }
    if len(image_paths) == 1:
        ask_kwargs["image"] = image_paths[0]
    else:
        ask_kwargs["images"] = image_paths

    rclpy.init()
    try:
        client = VQAClient()
        try:
            result = client.ask(question, args.timeout, **ask_kwargs)
        except TimeoutError as exc:
            # This is a CLI — a traceback adds nothing the message doesn't say.
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            client.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    if result.get("error"):
        print(result["error"], file=sys.stderr)
        if args.json:
            print(json.dumps(result, indent=2))
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.raw or args.freeform or result.get("number") is None:
        print(result.get("answer") or "")
    else:
        print(result["number"])

    # Exit 2 means "answered, but the reply held no integer" — distinct from a
    # failure (1), so scripted numeric questions can tell the two apart.
    if (result.get("number") is None and not args.raw and not args.json
            and not args.freeform):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
