#!/usr/bin/env python3
"""Category-1 (numerical) reasoner: extract targets → SAM prompts → best-view count.

Challenge-facing topics:
  /challenge_question   (sub)  std_msgs/String   — latch first; eval repeats at 1 Hz
  /numerical_response   (pub)  std_msgs/Int32    — once per question

Internal orchestration:
  /gt_target_objects    (sub, latched) — benchmark targets; skips the extract step
  /sam3/set_prompts     (pub)  JSON {prompts, run_id} — the ONLY thing that arms sam_node
  /sam3/best_view_dir   (sub, latched)
  /pipeline/explore_done (sub) — smart_vlm signals exploration finished
  /qwen_vqa/{request,response,status} — the local backend's transport only

Phases per question: extract targets → set SAM prompts → wait explore_done →
answer from the top `max_context_views` best-view images (+ manifest).

`crops_only:=true` stops after saving the crops and publishes a placeholder answer.
That is how the best-view cache is built: extraction still runs on a real model, so SAM
is armed exactly as it would be on a scored run, and `cat1_bench` replays the counting
step over the saved images afterwards, once per model instead of once per sweep.

Both model steps go through `captioner.vlm_backends`, so the same pipeline runs against
a hosted model over its OpenAI-compatible endpoint (`backend:=cloud`, the scored path,
provider chosen by VLM_PROVIDER) or the resident Qwen server (`backend:=local`, which
costs nothing and is the development loop).

The answer is published before the manifest is written: the harness tears this pipeline
down the instant it sees /numerical_response, so anything queued after the publish may
never finish.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

from typing import NamedTuple

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from captioner.paths import secure_path
from captioner.qwen_vqa_protocol import vqa_image_fields
from captioner.ros_utils import wait_for_subscriber
from captioner.ros_utils import shutdown_guard
from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import VLM_BACKEND
from captioner.vlm_backends.schemas import CountAnswer, TargetList
from sam_mapper.best_view import sanitize_run_id
from smart_vlm.numerical_utils import (
    ANSWER_SYSTEM,
    EXTRACT_SYSTEM,
    clean_targets,
    heuristic_targets,
    select_context_views,
)
from smart_vlm.question import QuestionType, classify

# Only for a run with no run_id of its own: a random id keeps two manual runs of the
# same question from writing over each other, at the cost of being unfindable later.
ADHOC_RUN_PREFIX = "num_reasoner"


class TargetChoice(NamedTuple):
    """How this question's SAM prompts were arrived at, kept for the manifest.

    Wrong prompts are the failure the crops cannot recover from, and the count they
    produce looks like an ordinary miss. Recording the model's untouched reply next to
    what SAM was finally armed with is what separates "the model named the wrong
    objects" from "the cleanup dropped the right ones" when reading a bad result back.
    """
    prompts: list[str]                 # exactly what went to /sam3/set_prompts
    source: str                        # "gt" | the backend's name | "heuristic"
    extract_reply: list[str] | None    # the model's raw list, before clean_targets


class NumericalReasoner(Node):
    PHASE_IDLE = "idle"
    PHASE_EXTRACT = "extract"
    PHASE_PROMPTS = "prompts"
    PHASE_EXPLORE = "explore"
    PHASE_ANSWER = "answer"

    def __init__(self):
        super().__init__("numerical_reasoner")

        # Where this question's crops go, under the crops root. The eval harness passes
        # "<sweep>/<scene>/<question id>-<question>", which is what makes a run findable
        # afterwards; empty falls back to a name with a random id in it, which is fine
        # for a one-off manual run.
        self.declare_parameter("run_id", "")
        # Produce the crops and stop: extract targets, arm SAM, then publish a
        # placeholder answer instead of paying a model to count. This is how the
        # best-view cache is built; cat1_bench does the counting later, once per model.
        self.declare_parameter("crops_only", False)
        self.declare_parameter("vqa_timeout_s", 120.0)
        # local | cloud | auto (auto defers to VLM_BACKEND in the environment).
        self.declare_parameter("backend", "auto")
        # How many best-view ranks to show the model at once. Each view costs a full
        # image's worth of visual tokens, so this trades latency for the recall of
        # objects that rank 1 happens not to frame.
        self.declare_parameter("max_context_views", 3)

        self.fixed_run_id = str(self.get_parameter("run_id").value).strip()
        self.crops_only = bool(self.get_parameter("crops_only").value)
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        backend_param = str(self.get_parameter("backend").value).strip().lower()
        self.backend_name = (
            VLM_BACKEND if backend_param in ("", "auto") else backend_param)
        self.max_context_views = max(1, int(self.get_parameter("max_context_views").value))

        self._lock = threading.Lock()
        self._phase = self.PHASE_IDLE
        self._question: Optional[str] = None
        self._current_gt_targets: Optional[list[str]] = None
        self._active_gt_targets: Optional[list[str]] = None
        self._run_id: Optional[str] = None
        self._targets: Optional[TargetChoice] = None
        self._crop_dir: Optional[str] = None

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
        self.create_subscription(String, "/gt_target_objects", self._on_gt_targets, latch_qos,
                                 callback_group=cb)
        self.create_subscription(String, "/challenge_question", self._on_question, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/pipeline/explore_done", self._on_explore_done, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/sam3/best_view_dir", self._on_crop_dir, latch_qos,
                                 callback_group=cb)
        self.create_subscription(String, "/qwen_vqa/response", self._on_vqa_response, 10,
                                 callback_group=cb)
        self.create_subscription(String, "/qwen_vqa/status", self._on_vqa_status, latch_qos,
                                 callback_group=cb)

        self.pub_prompts = self.create_publisher(String, "/sam3/set_prompts", 10)
        self.pub_vqa_req = self.create_publisher(String, "/qwen_vqa/request", 10)
        self.pub_answer = self.create_publisher(Int32, "/numerical_response", 10)
        self.pub_status = self.create_publisher(String, "/numerical_reasoner/status", latch_qos)

        self.backend = make_backend(
            self.backend_name,
            ask_vqa=self._ask_vqa_text,
            log=self.get_logger().info,
        )

        self._publish_status("idle")
        self.get_logger().info(
            f"numerical_reasoner ready (backend={self.backend.name}, "
            f"max_context_views={self.max_context_views}"
            f"{', crops_only' if self.crops_only else ''}) — waiting for "
            "How many / Count questions")

    def _publish_status(self, text: str) -> None:
        # Worker threads call this during teardown, after SIGINT has already invalidated
        # the context — publishing then raises RCLError and kills the thread noisily.
        if not rclpy.ok():
            return
        self.pub_status.publish(String(data=text))

    def _on_gt_targets(self, msg: String) -> None:
        try:
            targets = json.loads(msg.data)
            with self._lock:
                self._current_gt_targets = targets if targets else None
        except json.JSONDecodeError:
            pass

    def _on_vqa_status(self, msg: String) -> None:
        self._qwen_ready = msg.data.strip() == "ready"

    def _on_crop_dir(self, msg: String) -> None:
        with self._lock:
            self._crop_dir = msg.data

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
            
            q_text = msg.data.strip()
            
            if classify(q_text) is not QuestionType.NUMERICAL:
                return
            self._phase = self.PHASE_EXTRACT
            self._question = q_text
            self._active_gt_targets = self._current_gt_targets
            # Same sanitiser sam_mapper applies when it turns run_id into a
            # directory name, so the path we log matches the one it creates.
            if self.fixed_run_id:
                self._run_id = sanitize_run_id(self.fixed_run_id)
            else:
                safe_q = sanitize_run_id(self._question[:48], fallback="q")
                self._run_id = f"{ADHOC_RUN_PREFIX}_{uuid.uuid4().hex[:8]}_{safe_q}"[:80]
            self._targets = None

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
            best_dir = self._crop_dir
            targets = self._targets

        self.get_logger().info(f"explore_done ({msg.data or 'ok'}) — answering")
        self._publish_status("answer")
        threading.Thread(
            target=self._run_answer,
            args=(question, run_id, best_dir, targets),
            daemon=True,
        ).start()

    def _ask_vqa_text(
        self,
        question: str,
        images: Sequence[str] = (),
        max_new_tokens: int = 64,
        mode: str = "freeform",
    ) -> str:
        """The transport the local backend runs on: one round-trip, answer text out."""
        timeout_s = self.vqa_timeout_s
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
            "question": question,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            **vqa_image_fields(images),
        }
        with self._vqa_lock:
            self._vqa_wait_id = req_id
            self._vqa_response = None
            self._vqa_event.clear()

        # On a MultiThreadedExecutor our callbacks keep running, so this can sleep.
        wait_for_subscriber(self.pub_vqa_req)

        self.pub_vqa_req.publish(String(data=json.dumps(payload)))
        if not self._vqa_event.wait(timeout=timeout_s):
            raise TimeoutError(f"No VQA response within {timeout_s:.0f}s")

        with self._vqa_lock:
            response = self._vqa_response or {}
            self._vqa_wait_id = None
        if response.get("error"):
            raise RuntimeError(f"VQA error: {response['error']}")
        return response.get("answer") or ""

    def _extract_targets(self, question: str) -> TargetChoice:
        """The nouns SAM gets armed with. Never raises: an empty list is caught upstream."""
        reply: list[str] | None = None
        try:
            # User turn is the raw query, matching language_planner's extract_chain.
            result = self.backend.ask(
                EXTRACT_SYSTEM, question, [], TargetList, lite=True)
            reply = list(result.targets)
            targets = clean_targets(result.targets)
            if targets:
                return TargetChoice(targets, self.backend.name, reply)
            self.get_logger().warn(f"extract returned nothing usable: {reply!r}")
        except VLMError as exc:
            self.get_logger().warn(f"extract failed: {exc}")

        # Crude noun guess from "How many X ...". Worse than the model, but it keeps
        # a question answerable instead of failing the whole run on one bad reply.
        targets = heuristic_targets(question)
        self.get_logger().warn(f"falling back to heuristic targets={targets}")
        return TargetChoice(targets, "heuristic", reply)

    def _run_extract_and_set_prompts(self) -> None:
        try:
            with self._lock:
                question = self._question
                run_id = self._run_id
                gt_targets = self._active_gt_targets

            if gt_targets is not None:
                choice = TargetChoice(gt_targets, "gt", None)
            else:
                choice = self._extract_targets(question)

            if not choice.prompts:
                raise RuntimeError("Could not extract target objects from question")

            with self._lock:
                self._targets = choice
                self._phase = self.PHASE_PROMPTS

            self._publish_status("prompts")
            self.get_logger().info(
                f"extracted targets ({choice.source}): {choice.prompts}")

            # Always publish, GT path included: sam_node boots unarmed, so this is the
            # only thing that ever gives it prompts. It also gives both paths a per-run
            # run_id, so /sam3/best_view_dir points at this question's own crop dir
            # instead of silently reusing SAM's boot-time default.
            set_payload = {"prompts": choice.prompts, "run_id": run_id}
            wait_for_subscriber(self.pub_prompts)
            self.pub_prompts.publish(String(data=json.dumps(set_payload)))

            with self._lock:
                self._phase = self.PHASE_EXPLORE

            self._publish_status("explore")
            self.get_logger().info(
                f"prompts applied; waiting for /pipeline/explore_done "
                f"(and final best_view crops)")
        except Exception as exc:  # noqa: BLE001 — keep node alive
            self.get_logger().error(f"extract/set_prompts failed: {type(exc).__name__}: {exc}")
            with self._lock:
                self._phase = self.PHASE_IDLE
                self._question = None
                self._run_id = None
            self._publish_status("error")

    def _run_answer(
        self,
        question: Optional[str],
        run_id: Optional[str],
        best_dir: Optional[str],
        targets: Optional[TargetChoice],
    ) -> None:
        try:
            if not question:
                raise RuntimeError("No question latched")
            run_dir = self._resolve_crop_dir(best_dir)
            manifest_path = run_dir / "manifest.json"
            manifest: dict = {"selected": []}
            if manifest_path.is_file():
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)

            image_paths = select_context_views(run_dir, manifest, self.max_context_views)

            if not image_paths:
                # No detections / empty best-view run — still answer so the harness
                # does not hang; score will mark incorrect.
                self.get_logger().error(
                    f"No best-view image in {run_dir}; publishing 0")
                answer = 0
            elif self.crops_only:
                # The crops are the product of this run. Answering would cost a call per
                # question for a number the report is about to record as meaningless.
                self.get_logger().info(
                    f"crops_only: {len(image_paths)} view(s) saved to {run_dir}, "
                    "publishing a placeholder answer")
                answer = 0
                reason = None
            else:
                answer, reason = self._count_from_views(question, image_paths)

            # Record first, publish second. The harness tears the pipeline down the
            # moment it sees /numerical_response, and it gets there before a file write
            # does — publishing first cost us the manifest on every single question.
            # Question first, then how SAM came to be looking for what it looked for.
            # This file is read by hand at least as often as by code, and those are the
            # things you need before any of the geometry below makes sense.
            manifest = {
                "question": question,
                "sam_prompts": targets.prompts if targets else [],
                "target_source": targets.source if targets else "unknown",
                "extract_reply": targets.extract_reply if targets else None,
                **{k: v for k, v in manifest.items()
                   if k not in ("question", "sam_prompts", "target_source",
                                "extract_reply")},
            }
            # None rather than the placeholder 0: a cached directory that claims an
            # answer nobody computed is a trap for whoever reads it next.
            manifest["predicted_answer"] = None if self.crops_only else answer
            # The model's own account of the count. None under crops_only for the same
            # reason predicted_answer is: no answering call was made, so claiming a
            # rationale would dress a cache run up as a scored one.
            manifest["answer_reason"] = reason
            manifest["context_views"] = [p.name for p in image_paths]
            manifest["backend"] = self.backend.name
            manifest["crops_only"] = self.crops_only
            self._save_manifest(manifest_path, manifest)

            self.pub_answer.publish(Int32(data=answer))
            self.get_logger().info(f"Published /numerical_response={answer}")

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"answer failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")
            return

        # Soft reset so the next question can start (bag harness relaunches per Q).
        self._reset_to_idle("idle")

    def _count_from_views(self, question: str,
                          image_paths: list[Path]) -> tuple[int, str]:
        """Ask the backend for a count over every view at once.

        Returns the reason alongside the count so the manifest can keep it. The schema
        asks for it first precisely because it makes the model enumerate rather than
        guess (see CountAnswer), which makes it the one field that says whether a wrong
        number came from bad perception or bad reasoning — worth more than the log line
        it used to end up in.
        """
        result = self.backend.ask(ANSWER_SYSTEM, question, image_paths, CountAnswer)
        self.get_logger().info(
            f"count={result.count} from {len(image_paths)} view(s): {result.reason}")
        return max(0, int(result.count)), result.reason

    def _reset_to_idle(self, status: str) -> None:
        """Drop the current question so the next one is accepted."""
        with self._lock:
            self._phase = self.PHASE_IDLE
            self._question = None
            self._run_id = None
            self._targets = None
        self._publish_status(status)

    def _resolve_crop_dir(self, best_dir: Optional[str]) -> Path:
        if not best_dir:
            raise RuntimeError("No /sam3/best_view_dir available")
        return secure_path(best_dir)

    @staticmethod
    def _save_manifest(manifest_path: Path, manifest: dict) -> None:
        """Write atomically — a reader must never see a half-serialised manifest."""
        tmp = manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        tmp.replace(manifest_path)


def main(args=None):
    rclpy.init(args=args)
    node = NumericalReasoner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        with shutdown_guard():
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
