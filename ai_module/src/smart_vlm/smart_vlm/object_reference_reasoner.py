#!/usr/bin/env python3
"""Category-2 (object reference) reasoner: extract targets → SAM prompts → pick one object.

Challenge-facing topics:
  /challenge_question       (sub)  std_msgs/String            — latch first; eval repeats at 1 Hz
  /selected_object_marker   (pub)  visualization_msgs/Marker  — once per question

Internal orchestration:
  /gt_target_objects        (sub, latched) — benchmark targets; skips the extract step
  /sam3/set_prompts         (pub)  JSON {prompts, run_id} — arms sam_node for this question
  /sam3/best_view_dir       (sub, latched) — where the crops and obj_map.json land
  /pipeline/explore_done    (sub)  — smart_vlm signals exploration finished

The answer is a CUBE Marker from `obj_map.json` (3D IoU against the ground-truth box).
`cat2_utils.select_object` is shared with the offline bench. `crops_only:=true` publishes
a placeholder Marker so the cache can be built; a scored run must never set it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional, Sequence

import rclpy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from captioner.paths import secure_path
from captioner.ros_utils import wait_for_subscriber
from captioner.vlm_backends.constants import VIEW_SOURCE
from sam_mapper.challenge_marker import payload_from_map_object
from sam_mapper.ros_markers import create_selected_object_marker
from smart_vlm.cat2_utils import (
    SOLVER_AVAILABLE,
    Selection,
    heuristic_targets,
    marked_views,
    naive_from_raw,
    select_object,
    solver_status,
)
from smart_vlm.question import QuestionType
from smart_vlm.reasoner_common import (
    ReasonerNode,
    read_obj_map,
    save_manifest,
    spin_reasoner,
)


class ObjectReferenceReasoner(ReasonerNode):
    QUESTION_TYPE = QuestionType.OBJECT_REFERENCE
    STATUS_TOPIC = "/object_reference/status"
    ADHOC_PREFIX = "objref_reasoner"

    def __init__(self):
        super().__init__("object_reference_reasoner", extra_params={
            "crops_only": False,
            "max_context_views": 3,
            "mode": "hybrid",
        })
        self.crops_only = bool(self.get_parameter("crops_only").value)
        self.max_context_views = max(1, int(self.get_parameter("max_context_views").value))
        self.mode = str(self.get_parameter("mode").value).strip().lower() or "hybrid"
        self.pub_answer = self.create_publisher(Marker, "/selected_object_marker", 10)
        if not SOLVER_AVAILABLE:
            self.get_logger().error(solver_status())
        self.get_logger().info(
            f"object_reference_reasoner ready (backend={self.backend.name}, "
            f"extract_backend={self.extract_backend.name}, mode={self.mode}, "
            f"view_source={VIEW_SOURCE}, "
            f"max_context_views={self.max_context_views}"
            f"{', crops_only' if self.crops_only else ''}) — waiting for Find / The questions")

    def _heuristic_targets(self, question: str) -> list[str]:
        return heuristic_targets(question)

    def _begin_answer(self, explore_msg: String, snap: dict) -> None:
        self.get_logger().info(
            f"explore_done ({explore_msg.data or 'ok'}) — choosing an object")
        self._publish_status("answer")
        threading.Thread(
            target=self._run_answer,
            args=(snap["question"], snap["crop_dir"], snap["prompts"], snap["source"]),
            daemon=True,
        ).start()

    def _run_answer(self, question: Optional[str], best_dir: Optional[str],
                    prompts: Optional[list[str]], source: str) -> None:
        try:
            if not question:
                raise RuntimeError("No question latched")
            run_dir = secure_path(best_dir) if best_dir else None
            if run_dir is None:
                raise RuntimeError("No /sam3/best_view_dir available")

            manifest_path = run_dir / "manifest.json"
            manifest: dict = {"selected": []}
            if manifest_path.is_file():
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)

            raw_map = read_obj_map(run_dir, self.get_logger().error)
            if self.crops_only:
                selection = Selection(
                    None, "crops_only",
                    "crops_only: the crops and map are the product", [], [], 0)
            else:
                selection = self._choose(
                    question, run_dir, manifest,
                    raw_map, prompts or manifest.get("sam_prompts") or [])

            save_manifest(manifest_path, {
                "question": question,
                "sam_prompts": prompts or [],
                "target_source": source,
                **{k: v for k, v in manifest.items()
                   if k not in ("question", "sam_prompts", "target_source")},
                "predicted_object_id": selection.object_id,
                "selection_source": selection.source,
                "selection_reason": selection.reason,
                "selection_candidates": selection.candidates,
                "selection_trace": selection.trace,
                "vlm_calls": selection.vlm_calls,
                "mode": self.mode,
                "backend": self.backend.name,
                "extract_backend": self.extract_backend.name,
                "view_source": VIEW_SOURCE,
                "crops_only": self.crops_only,
                "n_map_objects": len(raw_map),
            })

            if self.crops_only:
                self._publish_placeholder()
            else:
                self._publish_answer(selection, raw_map)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"answer failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")
            return

        self._reset_to_idle("idle")

    def _choose(self, question: str, run_dir: Path, manifest: dict, raw_map: dict,
                prompts: Sequence[str]) -> Selection:
        if not raw_map:
            return Selection(
                None, "none", f"no obj_map.json objects in {run_dir}", [], [], 0)
        if not SOLVER_AVAILABLE:
            oid, why = naive_from_raw(raw_map)
            return Selection(oid, "naive", why, [oid] if oid else [], [solver_status()], 0)

        import utils.objmap as objmap

        objects = objmap.load_obj_map(run_dir / "obj_map.json", prompts)
        selection = select_object(
            question,
            objects,
            mode=self.mode,
            ask=self.backend.ask,
            views_for=lambda ids, labels: marked_views(
                run_dir, manifest, ids, self.max_context_views, labels),
            log=self.get_logger().warn,
        )
        self.get_logger().info(
            f"selected {selection.object_id} via {selection.source} "
            f"from {len(selection.candidates)} candidate(s): {selection.reason}")
        return selection

    def _publish_answer(self, selection: Selection, raw_map: dict) -> None:
        entry = raw_map.get(str(selection.object_id)) if selection.object_id else None
        if entry is None:
            self.get_logger().error(
                f"no map entry for the chosen id {selection.object_id!r} — publishing nothing")
            return
        stamp = self.get_clock().now().seconds_nanoseconds()
        marker = create_selected_object_marker(
            payload_from_map_object(entry),
            marker_id=int(selection.object_id)
            if str(selection.object_id).lstrip("-").isdigit() else 0,
            seconds=int(stamp[0]),
            nanoseconds=int(stamp[1]),
        )
        wait_for_subscriber(self.pub_answer)
        self.pub_answer.publish(marker)
        self.get_logger().info(
            f"Published /selected_object_marker id={marker.id} text={marker.text!r} "
            f"at ({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f}, "
            f"{marker.pose.position.z:.2f})")

    def _publish_placeholder(self) -> None:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "placeholder"
        marker.id = -1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1e-3
        marker.text = "crops_only"
        wait_for_subscriber(self.pub_answer)
        self.pub_answer.publish(marker)
        self.get_logger().info("crops_only: published a placeholder marker")


def main(args=None):
    rclpy.init(args=args)
    spin_reasoner(ObjectReferenceReasoner())


if __name__ == "__main__":
    main()
