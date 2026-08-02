#!/usr/bin/env python3
"""Category-1 (numerical) reasoner: Qwen extract → SAM prompts → best-view VQA.

Challenge-facing topics:
  /challenge_question   (sub)  std_msgs/String   — latch first; eval repeats at 1 Hz
  /numerical_response   (pub)  std_msgs/Int32    — once per question

Internal orchestration:
  /sam3/set_prompts     (pub)  JSON {prompts, run_id}
  /sam3/prompts_ack     (sub)
  /sam3/best_view_dir   (sub, latched)
  /pipeline/explore_done (sub) — bag driver signals exploration finished
  /qwen_vqa/{request,response,status}

Phases per question: extract targets → set SAM prompts → wait explore_done →
answer from best_rank1 image (+ manifest) → caption instance attributes into manifest.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import cv2
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from smart_vlm.category1_utils import (
    extract_integer,
    heuristic_targets,
    parse_target_list,
)
from smart_vlm.supervisor import classify

EXTRACT_PROMPT = (
    "Return ONLY a JSON list of short object noun phrases that a detector should "
    "look for in order to answer the question. Include EVERY referenced object "
    "(counted objects and landmarks like jars/tables). No colors or adjectives. "
    "Example: [\"glass\", \"arabic jar\"].\n"
    "Question: {question}"
)

ATTRIBUTE_PROMPT = (
    "Describe the {label} in this image using color, material, and shape. "
    "Reply in one short sentence starting with \"The {label} is\"."
)

_SAFE_RUN_ID = re.compile(r"[^a-zA-Z0-9._-]+")


def _secure_under_roots(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = path.expanduser().resolve()
    if ".." in Path(unquote(str(path))).parts:
        raise ValueError("Path traversal rejected")
    allowed = any(
        str(resolved) == str(root) or str(resolved).startswith(str(root) + "/")
        for root in roots
    )
    if not allowed:
        raise PermissionError(f"Path not under allowed mounts: {resolved}")
    return resolved


def _allowed_roots() -> tuple[Path, ...]:
    candidates = [
        Path("/data/bags"),
        Path("/data/workspace"),
        Path("/home/docker"),
        Path.home(),
    ]
    home = Path("/home")
    if home.is_dir():
        candidates.extend(p for p in home.iterdir() if p.is_dir())
    return tuple(p.resolve() for p in candidates if p.exists())


class Category1Reasoner(Node):
    PHASE_IDLE = "idle"
    PHASE_EXTRACT = "extract"
    PHASE_PROMPTS = "prompts"
    PHASE_EXPLORE = "explore"
    PHASE_ANSWER = "answer"
    PHASE_DONE = "done"

    def __init__(self):
        super().__init__("category1_reasoner")

        self.declare_parameter("run_id_prefix", "cat1")
        self.declare_parameter("vqa_timeout_s", 120.0)
        self.declare_parameter("prompts_ack_timeout_s", 30.0)
        self.declare_parameter("attribute_max_instances", 12)

        self.run_id_prefix = str(self.get_parameter("run_id_prefix").value)
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        self.prompts_ack_timeout_s = float(self.get_parameter("prompts_ack_timeout_s").value)
        self.attribute_max_instances = int(self.get_parameter("attribute_max_instances").value)

        self._lock = threading.Lock()
        self._phase = self.PHASE_IDLE
        self._question: Optional[str] = None
        self._run_id: Optional[str] = None
        self._extracted: list[str] = []
        self._best_view_dir: Optional[str] = None
        self._pending_ack_run_id: Optional[str] = None
        self._ack_event = threading.Event()
        self._ack_payload: Optional[dict] = None

        self._vqa_lock = threading.Lock()
        self._vqa_wait_id: Optional[str] = None
        self._vqa_response: Optional[dict] = None
        self._vqa_event = threading.Event()
        self._qwen_ready = False

        latch_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        cb = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, "/challenge_question", self._on_question, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/pipeline/explore_done", self._on_explore_done, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/sam3/prompts_ack", self._on_prompts_ack, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/sam3/best_view_dir", self._on_best_view_dir, latch_qos,
                                 callback_group=cb)
        self.create_subscription(String, "/qwen_vqa/response", self._on_vqa_response, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/qwen_vqa/status", self._on_vqa_status, latch_qos,
                                 callback_group=cb)

        self.pub_prompts = self.create_publisher(String, "/sam3/set_prompts", 10)
        self.pub_vqa_req = self.create_publisher(String, "/qwen_vqa/request", 10)
        self.pub_answer = self.create_publisher(Int32, "/numerical_response", 10)
        self.pub_status = self.create_publisher(String, "/category1_reasoner/status", latch_qos)

        self._publish_status("idle")
        self.get_logger().info(
            "category1_reasoner ready — waiting for How many / Count questions")

    def _publish_status(self, text: str) -> None:
        self.pub_status.publish(String(data=text))

    def _on_vqa_status(self, msg: String) -> None:
        self._qwen_ready = msg.data.strip() == "ready"

    def _on_best_view_dir(self, msg: String) -> None:
        with self._lock:
            self._best_view_dir = msg.data

    def _on_prompts_ack(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            expected = self._pending_ack_run_id
        if expected is not None and payload.get("run_id") != expected:
            return
        self._ack_payload = payload
        self._ack_event.set()

    def _on_vqa_response(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._vqa_lock:
            if self._vqa_wait_id is None or payload.get("id") != self._vqa_wait_id:
                return
            self._vqa_response = payload
            self._vqa_event.set()

    def _on_question(self, msg: String) -> None:
        # Eval repeats at 1 Hz — take the first unanswered question only.
        with self._lock:
            if self._phase != self.PHASE_IDLE:
                return
            if classify(msg.data) != "numerical":
                return
            self._phase = self.PHASE_EXTRACT
            self._question = msg.data.strip()
            qid = uuid.uuid4().hex[:8]
            safe_q = _SAFE_RUN_ID.sub("_", self._question[:48]).strip("_")
            self._run_id = f"{self.run_id_prefix}_{qid}_{safe_q}"[:80]
            self._extracted = []
            self._ack_payload = None
            self._ack_event.clear()
            self._pending_ack_run_id = None

        self.get_logger().info(f"QUESTION: {self._question}")
        self._publish_status("extract")
        threading.Thread(target=self._run_extract_and_set_prompts, daemon=True).start()

    def _on_explore_done(self, msg: String) -> None:
        with self._lock:
            if self._phase != self.PHASE_EXPLORE:
                return
            self._phase = self.PHASE_ANSWER
            question = self._question
            run_id = self._run_id
            best_dir = self._best_view_dir
            extracted = list(self._extracted)

        self.get_logger().info(f"explore_done ({msg.data or 'ok'}) — answering")
        self._publish_status("answer")
        threading.Thread(
            target=self._run_answer,
            args=(question, run_id, best_dir, extracted),
            daemon=True,
        ).start()

    def _ask_vqa(
        self,
        question: str,
        image: Optional[str],
        timeout_s: float,
        max_new_tokens: int = 64,
        *,
        mode: str = "numerical",
    ) -> dict:
        if not self._qwen_ready:
            # Brief wait for latched status if we just started.
            deadline = time.time() + min(30.0, timeout_s)
            while not self._qwen_ready and time.time() < deadline and rclpy.ok():
                time.sleep(0.1)
        if not self._qwen_ready:
            raise RuntimeError("qwen_vqa_server not ready")

        req_id = str(uuid.uuid4())
        payload = {
            "id": req_id,
            "image": image,
            "question": question,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
        }
        with self._vqa_lock:
            self._vqa_wait_id = req_id
            self._vqa_response = None
            self._vqa_event.clear()

        # Give the subscription time to connect before publishing.
        deadline_connect = time.time() + 2.0
        while time.time() < deadline_connect and self.pub_vqa_req.get_subscription_count() == 0:
            time.sleep(0.05)

        self.pub_vqa_req.publish(String(data=json.dumps(payload)))
        if not self._vqa_event.wait(timeout=timeout_s):
            raise TimeoutError(f"No VQA response within {timeout_s:.0f}s")

        with self._vqa_lock:
            response = self._vqa_response or {}
            self._vqa_wait_id = None
        if response.get("error"):
            raise RuntimeError(f"VQA error: {response['error']}")
        return response

    def _run_extract_and_set_prompts(self) -> None:
        try:
            with self._lock:
                question = self._question
                run_id = self._run_id

            prompt = EXTRACT_PROMPT.format(question=question)
            response = self._ask_vqa(
                prompt, image=None, timeout_s=self.vqa_timeout_s,
                max_new_tokens=128, mode="freeform")
            targets = parse_target_list(response.get("answer") or "")
            if not targets:
                # Fallback: crude noun guess from "How many X ..."
                targets = heuristic_targets(question)
                self.get_logger().warn(
                    f"extract parse failed ({response.get('answer')!r}); "
                    f"heuristic targets={targets}")
            if not targets:
                raise RuntimeError("Could not extract target objects from question")

            with self._lock:
                self._extracted = targets
                self._phase = self.PHASE_PROMPTS
                self._pending_ack_run_id = run_id
                self._ack_event.clear()
                self._ack_payload = None

            self._publish_status("prompts")
            self.get_logger().info(f"extracted targets: {targets}")

            set_payload = {"prompts": targets, "run_id": run_id}
            deadline_connect = time.time() + 2.0
            while (time.time() < deadline_connect
                   and self.pub_prompts.get_subscription_count() == 0):
                time.sleep(0.05)
            self.pub_prompts.publish(String(data=json.dumps(set_payload)))

            if not self._ack_event.wait(timeout=self.prompts_ack_timeout_s):
                raise TimeoutError("No /sam3/prompts_ack within timeout")
            ack = self._ack_payload or {}
            if not ack.get("ok", False):
                raise RuntimeError(f"set_prompts failed: {ack.get('error')}")

            with self._lock:
                if ack.get("run_dir"):
                    self._best_view_dir = ack["run_dir"]
                self._phase = self.PHASE_EXPLORE
                self._pending_ack_run_id = None

            self._publish_status("explore")
            self.get_logger().info(
                f"prompts applied; waiting for /pipeline/explore_done "
                f"(run_dir={ack.get('run_dir')})")
        except Exception as exc:  # noqa: BLE001 — keep node alive
            self.get_logger().error(f"extract/set_prompts failed: {type(exc).__name__}: {exc}")
            with self._lock:
                self._phase = self.PHASE_IDLE
                self._question = None
                self._run_id = None
                self._pending_ack_run_id = None
            self._publish_status("error")

    def _run_answer(
        self,
        question: Optional[str],
        run_id: Optional[str],
        best_dir: Optional[str],
        extracted: list[str],
    ) -> None:
        try:
            if not question:
                raise RuntimeError("No question latched")
            run_dir = self._resolve_best_view_dir(best_dir)
            manifest_path = run_dir / "manifest.json"
            manifest: dict = {"targets": extracted, "selected": []}
            if manifest_path.is_file():
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)

            selected = manifest.get("selected") or []
            image_path = None
            if selected:
                candidate = run_dir / selected[0]["file"]
                if candidate.is_file():
                    image_path = candidate

            if image_path is None:
                # No detections / empty best-view run — still answer so the harness
                # does not hang; score will mark incorrect.
                self.get_logger().error(
                    f"No best-view image in {run_dir}; publishing 0")
                self.pub_answer.publish(Int32(data=0))
                manifest["question"] = question
                manifest["extracted_targets"] = extracted
                manifest["predicted_answer"] = 0
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)
                    handle.write("\n")
            else:
                # mode=numerical wraps with the integer-only template in the VQA server.
                response = self._ask_vqa(
                    question,
                    image=str(image_path),
                    timeout_s=self.vqa_timeout_s,
                    max_new_tokens=32,
                    mode="numerical",
                )
                number = response.get("number")
                if number is None:
                    number = extract_integer(response.get("answer"))
                if number is None:
                    raise RuntimeError("Could not parse integer answer from VQA")

                self.pub_answer.publish(Int32(data=int(number)))
                self.get_logger().info(f"Published /numerical_response={number}")

                self._append_attributes(
                    manifest, run_dir, question, extracted, int(number))
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)
                    handle.write("\n")

            with self._lock:
                self._phase = self.PHASE_DONE
            self._publish_status("done")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"answer failed: {type(exc).__name__}: {exc}")
            with self._lock:
                self._phase = self.PHASE_IDLE
                self._question = None
                self._run_id = None
            self._publish_status("error")
            return

        # Soft reset so the next question can start (bag harness relaunches per Q).
        with self._lock:
            self._phase = self.PHASE_IDLE
            self._question = None
            self._run_id = None
            self._extracted = []
        self._publish_status("idle")

    def _resolve_best_view_dir(self, best_dir: Optional[str]) -> Path:
        if not best_dir:
            raise RuntimeError("No /sam3/best_view_dir available")
        return _secure_under_roots(Path(best_dir), _allowed_roots())

    def _append_attributes(
        self,
        manifest: dict,
        run_dir: Path,
        question: str,
        extracted: list[str],
        predicted: int,
    ) -> None:
        manifest["question"] = question
        manifest["extracted_targets"] = extracted
        manifest["predicted_answer"] = predicted

        roots = _allowed_roots()
        count = 0
        for entry in manifest.get("selected") or []:
            if count >= self.attribute_max_instances:
                break
            image_path = _secure_under_roots(run_dir / entry["file"], roots)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            for inst in entry.get("instances") or []:
                if count >= self.attribute_max_instances:
                    break
                label = str(inst.get("label") or "object")
                bbox = inst.get("bbox") or [0, 0, image.shape[1], image.shape[0]]
                x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = image[y0:y1, x0:x1]
                crop_path = run_dir / f"_attr_crop_{entry.get('rank', 0)}_{inst.get('track_id')}.png"
                cv2.imwrite(str(crop_path), crop)
                try:
                    attr_q = ATTRIBUTE_PROMPT.format(label=label)
                    resp = self._ask_vqa(
                        attr_q,
                        image=str(crop_path),
                        timeout_s=self.vqa_timeout_s,
                        max_new_tokens=64,
                        mode="freeform",
                    )
                    inst["attributes"] = (resp.get("answer") or "").strip()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        f"attribute caption failed for {label}: {type(exc).__name__}")
                    inst["attributes"] = ""
                finally:
                    try:
                        crop_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                count += 1


def main(args=None):
    rclpy.init(args=args)
    node = Category1Reasoner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
