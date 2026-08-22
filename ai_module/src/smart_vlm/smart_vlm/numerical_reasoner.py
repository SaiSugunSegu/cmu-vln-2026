#!/usr/bin/env python3
"""Category-1 (numerical) reasoner: extract targets → SAM prompts → best-view count.

Challenge-facing topics:
  /challenge_question   (sub)  std_msgs/String   — latch first; eval repeats at 1 Hz
  /numerical_response   (pub)  std_msgs/Int32    — once per question

Internal orchestration:
  /gt_target_objects    (sub, latched) — benchmark targets; skips the extract step
  /sam3/set_prompts     (pub)  JSON {prompts, run_id} — arms sam_node for this question
  /sam3/best_view_dir   (sub, latched)
  /pipeline/explore_done (sub) — smart_vlm signals exploration finished
  /qwen_vqa/{request,response,status} — the local backend's transport only

`crops_only:=true` saves the crops and publishes a placeholder answer so `cat1_bench`
can replay the count later. A scored run must never set it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import NamedTuple, Optional

import rclpy
from std_msgs.msg import Int32, String

from captioner.paths import secure_path
from captioner.vlm_backends.constants import VIEW_SOURCE
from captioner.vlm_backends.schemas import CountAnswer
from smart_vlm.numerical_utils import ANSWER_SYSTEM, heuristic_targets, select_context_views
from smart_vlm.question import QuestionType
from smart_vlm.reasoner_common import ReasonerNode, save_manifest, spin_reasoner


class TargetChoice(NamedTuple):
    """How this question's SAM prompts were arrived at, kept for the manifest."""
    prompts: list[str]
    source: str
    extract_reply: list[str] | None


class NumericalReasoner(ReasonerNode):
    QUESTION_TYPE = QuestionType.NUMERICAL
    STATUS_TOPIC = "/numerical_reasoner/status"
    ADHOC_PREFIX = "num_reasoner"

    def __init__(self):
        super().__init__("numerical_reasoner", extra_params={
            "crops_only": False,
            "max_context_views": 3,
        })
        self.crops_only = bool(self.get_parameter("crops_only").value)
        self.max_context_views = max(1, int(self.get_parameter("max_context_views").value))
        self.pub_answer = self.create_publisher(Int32, "/numerical_response", 10)
        self.get_logger().info(
            f"numerical_reasoner ready (backend={self.backend.name}, "
            f"extract_backend={self.extract_backend.name}, "
            f"view_source={VIEW_SOURCE}, "
            f"max_context_views={self.max_context_views}"
            f"{', crops_only' if self.crops_only else ''}) — waiting for "
            "How many / Count questions")

    def _heuristic_targets(self, question: str) -> list[str]:
        return heuristic_targets(question)

    def _begin_answer(self, explore_msg: String, snap: dict) -> None:
        self.get_logger().info(f"explore_done ({explore_msg.data or 'ok'}) — answering")
        self._publish_status("answer")
        targets = TargetChoice(
            list(snap["prompts"] or []), snap["source"], snap["extract_reply"])
        threading.Thread(
            target=self._run_answer,
            args=(snap["question"], snap["crop_dir"], targets),
            daemon=True,
        ).start()

    def _run_answer(
        self,
        question: Optional[str],
        best_dir: Optional[str],
        targets: TargetChoice,
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
            reason = None
            if not image_paths:
                self.get_logger().error(f"No best-view image in {run_dir}; publishing 0")
                answer = 0
            elif self.crops_only:
                self.get_logger().info(
                    f"crops_only: {len(image_paths)} view(s) saved to {run_dir}, "
                    "publishing a placeholder answer")
                answer = 0
            else:
                answer, reason = self._count_from_views(question, image_paths)

            # Record first, publish second: the harness tears the pipeline down the
            # moment it sees /numerical_response.
            manifest = {
                "question": question,
                "sam_prompts": targets.prompts,
                "target_source": targets.source,
                "extract_reply": targets.extract_reply,
                **{k: v for k, v in manifest.items()
                   if k not in ("question", "sam_prompts", "target_source",
                                "extract_reply")},
            }
            manifest["predicted_answer"] = None if self.crops_only else answer
            manifest["answer_reason"] = reason
            manifest["context_views"] = [p.name for p in image_paths]
            manifest["backend"] = self.backend.name
            manifest["extract_backend"] = self.extract_backend.name
            manifest["view_source"] = VIEW_SOURCE
            manifest["crops_only"] = self.crops_only
            save_manifest(manifest_path, manifest)

            self.pub_answer.publish(Int32(data=answer))
            self.get_logger().info(f"Published /numerical_response={answer}")

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"answer failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")
            return

        self._reset_to_idle("idle")

    def _count_from_views(self, question: str,
                          image_paths: list[Path]) -> tuple[int, str]:
        result = self.backend.ask(ANSWER_SYSTEM, question, image_paths, CountAnswer)
        self.get_logger().info(
            f"count={result.count} from {len(image_paths)} view(s): {result.reason}")
        return max(0, int(result.count)), result.reason

    def _resolve_crop_dir(self, best_dir: Optional[str]) -> Path:
        if not best_dir:
            raise RuntimeError("No /sam3/best_view_dir available")
        return secure_path(best_dir)


def main(args=None):
    rclpy.init(args=args)
    spin_reasoner(NumericalReasoner())


if __name__ == "__main__":
    main()
