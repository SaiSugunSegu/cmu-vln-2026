#!/usr/bin/env python3
"""Category-2 (object reference) reasoner: extract targets → SAM prompts → pick one object.

Challenge-facing topics:
  /challenge_question       (sub)  std_msgs/String            — latch first; eval repeats at 1 Hz
  /selected_object_marker   (pub)  visualization_msgs/Marker  — once per question

Internal orchestration:
  /gt_target_objects        (sub, latched) — benchmark targets; skips the extract step
  /sam3/set_prompts         (pub)  JSON {prompts, run_id} — the ONLY thing that arms sam_node
  /sam3/best_view_dir       (sub, latched) — where the crops and obj_map.json land
  /pipeline/explore_done    (sub)  — smart_vlm signals exploration finished

The answer is a box, not a word: the score is twice the 3D IoU between the Marker we publish
and the ground-truth box (`scripts/eval/score.py`). map_node has already fused this question's
detections into `obj_map.json` beside the crops, one entry per tracked instance, so answering
is choosing one of its keys and republishing that entry's box as a CUBE. Nothing here invents
geometry — a fabricated box cannot overlap anything, and its centre doubles as a navigation
waypoint.

Which key to choose is `cat2_utils.select_object`, shared with the offline bench so the two
cannot drift. The default `hybrid` mode spends a model call only where the geometry is not
decisive.

`crops_only:=true` stops after saving the crops and publishes a placeholder Marker (ns
`placeholder`, id -1) instead of choosing. That is how the cache is built: extraction still
runs on a real model, so SAM is armed exactly as on a scored run, and the bench replays the
selection step over the saved crops and maps afterwards, once per configuration instead of
once per sweep. A scored run must never set it.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from captioner.paths import secure_path
from captioner.qwen_vqa_protocol import vqa_image_fields
from captioner.ros_utils import wait_for_subscriber
from captioner.ros_utils import shutdown_guard
from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import VLM_BACKEND
from captioner.vlm_backends.schemas import TargetList
from sam_mapper.best_view import sanitize_run_id
from sam_mapper.challenge_marker import payload_from_map_object
from sam_mapper.ros_markers import create_selected_object_marker
from smart_vlm.cat2_utils import (
    EXTRACT_SYSTEM,
    SOLVER_AVAILABLE,
    Selection,
    heuristic_targets,
    marked_views,
    naive_from_raw,
    select_object,
    solver_status,
)
from smart_vlm.numerical_utils import clean_targets
from smart_vlm.question import QuestionType, classify

ADHOC_RUN_PREFIX = "objref_reasoner"

# map_node rewrites obj_map.json on every map publish, so by explore_done it is normally
# already there. Normally, not always: a scene where the last fusion lands just after the
# gate leaves the file a moment behind, and giving up instantly would throw away a whole
# question over a few hundred milliseconds.
MAP_WAIT_S = 5.0
MAP_POLL_S = 0.25


class ObjectReferenceReasoner(Node):
    PHASE_IDLE = "idle"
    PHASE_EXTRACT = "extract"
    PHASE_PROMPTS = "prompts"
    PHASE_EXPLORE = "explore"
    PHASE_ANSWER = "answer"

    def __init__(self):
        super().__init__("object_reference_reasoner")

        self.declare_parameter("run_id", "")
        self.declare_parameter("crops_only", False)
        self.declare_parameter("vqa_timeout_s", 120.0)
        self.declare_parameter("backend", "auto")
        self.declare_parameter("max_context_views", 3)
        # naive | solver | vlm | hybrid — see cat2_utils.select_object.
        self.declare_parameter("mode", "hybrid")

        self.fixed_run_id = str(self.get_parameter("run_id").value).strip()
        self.crops_only = bool(self.get_parameter("crops_only").value)
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        backend_param = str(self.get_parameter("backend").value).strip().lower()
        self.backend_name = VLM_BACKEND if backend_param in ("", "auto") else backend_param
        self.max_context_views = max(1, int(self.get_parameter("max_context_views").value))
        self.mode = str(self.get_parameter("mode").value).strip().lower() or "hybrid"

        self._lock = threading.Lock()
        self._phase = self.PHASE_IDLE
        self._question: Optional[str] = None
        self._current_gt_targets: Optional[list[str]] = None
        self._active_gt_targets: Optional[list[str]] = None
        self._run_id: Optional[str] = None
        self._prompts: Optional[list[str]] = None
        self._target_source = "unknown"
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
        self.pub_answer = self.create_publisher(Marker, "/selected_object_marker", 10)
        self.pub_status = self.create_publisher(String, "/object_reference/status", latch_qos)

        self.backend = make_backend(
            self.backend_name, ask_vqa=self._ask_vqa_text, log=self.get_logger().info)

        self._publish_status("idle")
        if not SOLVER_AVAILABLE:
            self.get_logger().error(solver_status())
        self.get_logger().info(
            f"object_reference_reasoner ready (backend={self.backend.name}, mode={self.mode}, "
            f"max_context_views={self.max_context_views}"
            f"{', crops_only' if self.crops_only else ''}) — waiting for Find / The questions")

    def _publish_status(self, text: str) -> None:
        if not rclpy.ok():
            return
        self.pub_status.publish(String(data=text))

    # ---- callbacks -------------------------------------------------------

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
        with self._lock:
            if self._phase != self.PHASE_IDLE:
                return
            q_text = msg.data.strip()
            if classify(q_text) is not QuestionType.OBJECT_REFERENCE:
                return
            self._phase = self.PHASE_EXTRACT
            self._question = q_text
            self._active_gt_targets = self._current_gt_targets
            if self.fixed_run_id:
                self._run_id = sanitize_run_id(self.fixed_run_id)
            else:
                safe_q = sanitize_run_id(self._question[:48], fallback="q")
                self._run_id = f"{ADHOC_RUN_PREFIX}_{uuid.uuid4().hex[:8]}_{safe_q}"[:80]
            self._prompts = None

        self.get_logger().info(f"QUESTION: {self._question}")
        self._publish_status("extract")
        threading.Thread(target=self._run_extract_and_set_prompts, daemon=True).start()

    def _on_explore_done(self, msg: String) -> None:
        with self._lock:
            if self._phase != self.PHASE_EXPLORE:
                return
            self._phase = self.PHASE_ANSWER
            question, run_dir = self._question, self._crop_dir
            prompts, source = self._prompts, self._target_source

        self.get_logger().info(f"explore_done ({msg.data or 'ok'}) — choosing an object")
        self._publish_status("answer")
        threading.Thread(target=self._run_answer,
                         args=(question, run_dir, prompts, source), daemon=True).start()

    # ---- model transport -------------------------------------------------

    def _ask_vqa_text(self, question: str, images: Sequence[str] = (),
                      max_new_tokens: int = 64, mode: str = "freeform") -> str:
        """The transport the local backend runs on: one round-trip, answer text out."""
        timeout_s = self.vqa_timeout_s
        if not self._qwen_ready:
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

    # ---- phases ----------------------------------------------------------

    def _extract_targets(self, question: str) -> tuple[list[str], str]:
        """The nouns SAM gets armed with, and where they came from."""
        try:
            # User turn is the raw query, matching language_planner's extract_chain.
            result = self.backend.ask(
                EXTRACT_SYSTEM, question, [], TargetList, lite=True)
            if targets := clean_targets(result.targets):
                return targets, self.backend.name
            self.get_logger().warn(f"extract returned nothing usable: {list(result.targets)!r}")
        except VLMError as exc:
            self.get_logger().warn(f"extract failed: {exc}")

        targets = heuristic_targets(question)
        self.get_logger().warn(f"falling back to parsed targets={targets}")
        return targets, "heuristic"

    def _run_extract_and_set_prompts(self) -> None:
        try:
            with self._lock:
                question, run_id = self._question, self._run_id
                gt_targets = self._active_gt_targets

            if gt_targets is not None:
                prompts, source = list(gt_targets), "gt"
            else:
                prompts, source = self._extract_targets(question)
            if not prompts:
                raise RuntimeError("Could not extract target objects from question")

            with self._lock:
                self._prompts, self._target_source = prompts, source
                self._phase = self.PHASE_PROMPTS

            self._publish_status("prompts")
            self.get_logger().info(f"extracted targets ({source}): {prompts}")

            wait_for_subscriber(self.pub_prompts)
            self.pub_prompts.publish(
                String(data=json.dumps({"prompts": prompts, "run_id": run_id})))

            with self._lock:
                self._phase = self.PHASE_EXPLORE
            self._publish_status("explore")
            self.get_logger().info("prompts applied; waiting for /pipeline/explore_done")
        except Exception as exc:  # noqa: BLE001 — keep the node alive
            self.get_logger().error(f"extract/set_prompts failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")

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

            raw_map = self._read_obj_map(run_dir)
            if self.crops_only:
                selection = Selection(None, "crops_only",
                                      "crops_only: the crops and map are the product", [],
                                      [], 0)
            else:
                selection = self._choose(question, run_dir, manifest, raw_map,
                                         prompts or manifest.get("sam_prompts") or [])

            # Record first, publish second. The harness tears the pipeline down the moment it
            # sees an answer on the topic, and it gets there before a file write does — the
            # counting reasoner lost its manifest on every question until it was ordered this
            # way round, and the manifest is the only record of why this object was chosen.
            self._save_manifest(manifest_path, {
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
        """Which map entry answers the question."""
        if not raw_map:
            return Selection(None, "none", f"no obj_map.json objects in {run_dir}", [], [], 0)
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

    def _read_obj_map(self, run_dir: Path) -> dict:
        deadline = time.monotonic() + MAP_WAIT_S
        path = run_dir / "obj_map.json"
        while time.monotonic() < deadline and rclpy.ok():
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle) or {}
                except json.JSONDecodeError:
                    # map_node writes atomically, so this is a partial read at worst —
                    # retry rather than treat it as an empty map.
                    pass
            time.sleep(MAP_POLL_S)
        self.get_logger().error(f"no readable obj_map.json in {run_dir} after {MAP_WAIT_S:.0f}s")
        return {}

    # ---- publishing ------------------------------------------------------

    def _publish_answer(self, selection: Selection, raw_map: dict) -> None:
        entry = raw_map.get(str(selection.object_id)) if selection.object_id else None
        if entry is None:
            # Nothing to publish: every box we could claim would be one we made up, and a
            # fabricated marker both scores zero and sends the robot somewhere wrong.
            self.get_logger().error(
                f"no map entry for the chosen id {selection.object_id!r} — publishing nothing")
            return
        stamp = self.get_clock().now().seconds_nanoseconds()
        marker = create_selected_object_marker(
            payload_from_map_object(entry),
            # The track id, so a scorer or a log can tell WHICH object was claimed and not
            # merely where the box was.
            marker_id=int(selection.object_id) if str(selection.object_id).lstrip("-").isdigit()
            else 0,
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
        """A marker that says "no answer was computed", so a cache sweep still has a gate.

        Namespaced `placeholder` with a negative id: the harness keys on those to record no
        prediction, and nothing else in the system consumes this topic. A scored run never
        takes this path.
        """
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

    def _reset_to_idle(self, status: str) -> None:
        with self._lock:
            self._phase = self.PHASE_IDLE
            self._question = None
            self._run_id = None
            self._prompts = None
        self._publish_status(status)

    @staticmethod
    def _save_manifest(manifest_path: Path, manifest: dict) -> None:
        tmp = manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        tmp.replace(manifest_path)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectReferenceReasoner()
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
