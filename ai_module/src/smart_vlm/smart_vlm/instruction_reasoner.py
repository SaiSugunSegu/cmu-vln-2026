#!/usr/bin/env python3
"""Category-3 (instruction-following) reasoner: extract targets → arm SAM → ordered waypoints.

Challenge-facing topics:
  /challenge_question        (sub)  std_msgs/String
  /way_point_with_heading    (pub)  geometry_msgs/Pose2D  — one pose at a time, after explore
  /state_estimation          (sub)  nav_msgs/Odometry     — arrival gate

Internal:
  /sam3/set_prompts          (pub)  JSON {prompts, run_id} — arms sam_node so TARE can start
  /start_exploration         (pub)  Bool(false) — releases the waypoint channel after explore
  /pipeline/explore_done     (sub)
  /sam3/best_view_dir        (sub, latched)
  /instruction_reasoner/status (pub, latched) — "answered" so the supervisor T-30 stays quiet

Without this node, instruction questions never arm SAM, TARE never leaves the start
gate, and the robot sits still for the full 10-minute budget.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from captioner.paths import secure_path
from captioner.qwen_vqa_protocol import vqa_image_fields
from captioner.ros_utils import shutdown_guard, wait_for_subscriber
from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import EXTRACT_BACKEND, VLM_BACKEND
from captioner.vlm_backends.schemas import TargetList
from sam_mapper.best_view import sanitize_run_id
from smart_vlm.cat2_utils import EXTRACT_SYSTEM
from smart_vlm.cat3_utils import heuristic_targets, waypoints_from_map
from smart_vlm.numerical_utils import clean_targets
from smart_vlm.question import QuestionType, classify

ADHOC_RUN_PREFIX = "instr_reasoner"
MAP_WAIT_S = 5.0
MAP_POLL_S = 0.25


class InstructionReasoner(Node):
    PHASE_IDLE = "idle"
    PHASE_EXTRACT = "extract"
    PHASE_PROMPTS = "prompts"
    PHASE_EXPLORE = "explore"
    PHASE_EXECUTE = "execute"

    def __init__(self):
        super().__init__("instruction_reasoner")

        self.declare_parameter("run_id", "")
        self.declare_parameter("vqa_timeout_s", 120.0)
        self.declare_parameter("backend", "auto")
        self.declare_parameter("reach_m", 1.5)
        self.declare_parameter("waypoint_timeout_s", 60.0)

        self.fixed_run_id = str(self.get_parameter("run_id").value).strip()
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        backend_param = str(self.get_parameter("backend").value).strip().lower()
        self.backend_name = VLM_BACKEND if backend_param in ("", "auto") else backend_param
        self.reach_m = float(self.get_parameter("reach_m").value)
        self.waypoint_timeout_s = float(self.get_parameter("waypoint_timeout_s").value)

        self._lock = threading.Lock()
        self._phase = self.PHASE_IDLE
        self._question: Optional[str] = None
        self._current_gt_targets: Optional[list[str]] = None
        self._active_gt_targets: Optional[list[str]] = None
        self._run_id: Optional[str] = None
        self._prompts: Optional[list[str]] = None
        self._target_source = "unknown"
        self._crop_dir: Optional[str] = None
        self._odom_xy: Optional[tuple[float, float]] = None

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
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10,
                                 callback_group=cb)

        self.pub_prompts = self.create_publisher(String, "/sam3/set_prompts", 10)
        self.pub_vqa_req = self.create_publisher(String, "/qwen_vqa/request", 10)
        self.pub_waypoint = self.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self.pub_start = self.create_publisher(Bool, "/start_exploration", latch_qos)
        self.pub_status = self.create_publisher(String, "/instruction_reasoner/status", latch_qos)

        self.extract_backend_name = (
            self.backend_name if EXTRACT_BACKEND == "auto" else EXTRACT_BACKEND)
        self.extract_backend = make_backend(
            self.extract_backend_name,
            ask_vqa=self._ask_vqa_text,
            log=self.get_logger().info,
        )

        self._publish_status("idle")
        self.get_logger().info(
            f"instruction_reasoner ready (extract_backend={self.extract_backend.name}) "
            "— waiting for Take / Avoid / Go questions")

    def _publish_status(self, text: str) -> None:
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

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        with self._lock:
            self._odom_xy = (float(p.x), float(p.y))

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
            if classify(q_text) is not QuestionType.INSTRUCTION_FOLLOWING:
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
            self._phase = self.PHASE_EXECUTE
            run_dir, prompts = self._crop_dir, self._prompts

        self.get_logger().info(f"explore_done ({msg.data or 'ok'}) — executing waypoints")
        self._publish_status("execute")
        threading.Thread(target=self._run_execute, args=(run_dir, prompts), daemon=True).start()

    def _ask_vqa_text(self, question: str, images: Sequence[str] = (),
                      max_new_tokens: int = 64, mode: str = "freeform") -> str:
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

    def _extract_targets(self, question: str) -> tuple[list[str], str]:
        try:
            result = self.extract_backend.ask(
                EXTRACT_SYSTEM, question, [], TargetList, lite=True)
            if targets := clean_targets(result.targets):
                return targets, self.extract_backend.name
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

    def _run_execute(self, best_dir: Optional[str], prompts: Optional[list[str]]) -> None:
        try:
            # Hold TARE so its coverage waypoints do not fight the instruction sequence.
            self.pub_start.publish(Bool(data=False))

            raw_map = self._read_obj_map(best_dir)
            waypoints = waypoints_from_map(raw_map, prompts or [])
            if not waypoints:
                self.get_logger().error(
                    "no matching objects in the map — publishing nothing, T-30 may guess")
                self._reset_to_idle("error")
                return

            self._publish_status("answered")
            for i, wp in enumerate(waypoints):
                self.get_logger().info(
                    f"waypoint {i + 1}/{len(waypoints)}: {wp['label']} "
                    f"({wp['x']:.2f}, {wp['y']:.2f})")
                self._drive_to(wp["x"], wp["y"])
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"execute failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")
            return

        self._reset_to_idle("idle")

    def _drive_to(self, x: float, y: float) -> None:
        msg = Pose2D(x=float(x), y=float(y), theta=0.0)
        wait_for_subscriber(self.pub_waypoint)
        deadline = time.monotonic() + self.waypoint_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self.pub_waypoint.publish(msg)
            with self._lock:
                here = self._odom_xy
            if here is not None:
                if math.hypot(here[0] - x, here[1] - y) <= self.reach_m:
                    return
            time.sleep(0.5)
        self.get_logger().warn(
            f"waypoint ({x:.2f}, {y:.2f}) not reached in {self.waypoint_timeout_s:.0f}s — "
            "advancing")

    def _read_obj_map(self, best_dir: Optional[str]) -> dict:
        if not best_dir:
            return {}
        try:
            run_dir = secure_path(best_dir)
        except (ValueError, PermissionError, RuntimeError) as exc:
            self.get_logger().error(f"best_view_dir rejected: {exc}")
            return {}
        deadline = time.monotonic() + MAP_WAIT_S
        path = Path(run_dir) / "obj_map.json"
        while time.monotonic() < deadline and rclpy.ok():
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle) or {}
                except json.JSONDecodeError:
                    pass
            time.sleep(MAP_POLL_S)
        self.get_logger().error(f"no readable obj_map.json in {run_dir} after {MAP_WAIT_S:.0f}s")
        return {}

    def _reset_to_idle(self, status: str) -> None:
        with self._lock:
            self._phase = self.PHASE_IDLE
            self._question = None
            self._run_id = None
            self._prompts = None
        self._publish_status(status)


def main(args=None):
    rclpy.init(args=args)
    node = InstructionReasoner()
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
