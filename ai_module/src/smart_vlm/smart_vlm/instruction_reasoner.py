#!/usr/bin/env python3
"""Category-3 reasoner: extract targets → arm SAM → ordered /way_point_with_heading."""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from captioner.ros_utils import wait_for_subscriber
from smart_vlm.cat3_utils import heuristic_targets, waypoints_from_map
from smart_vlm.question import QuestionType
from smart_vlm.reasoner_common import (
    ReasonerNode,
    read_obj_map,
    spin_reasoner,
)


class InstructionReasoner(ReasonerNode):
    QUESTION_TYPE = QuestionType.INSTRUCTION_FOLLOWING
    STATUS_TOPIC = "/instruction_reasoner/status"
    ADHOC_PREFIX = "instr_reasoner"
    ANSWER_BACKEND = False

    def __init__(self):
        super().__init__("instruction_reasoner", extra_params={
            "reach_m": 1.5,
            "waypoint_timeout_s": 60.0,
        })
        self.reach_m = float(self.get_parameter("reach_m").value)
        self.waypoint_timeout_s = float(self.get_parameter("waypoint_timeout_s").value)
        self._odom_xy: Optional[tuple[float, float]] = None

        self.create_subscription(
            Odometry, "/state_estimation", self._on_odom, 10, callback_group=self._cb)
        self.pub_waypoint = self.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self.pub_start = self.create_publisher(Bool, "/start_exploration", self._qos)
        self.get_logger().info(
            f"instruction_reasoner ready (extract={self.extract_backend.name})")

    def _heuristic_targets(self, question: str) -> list[str]:
        return heuristic_targets(question)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        with self._lock:
            self._odom_xy = (float(p.x), float(p.y))

    def _begin_answer(self, explore_msg: String, snap: dict) -> None:
        self.get_logger().info(f"explore_done ({explore_msg.data or 'ok'})")
        self._publish_status("execute")
        threading.Thread(
            target=self._execute,
            args=(snap["crop_dir"], snap["prompts"]),
            daemon=True,
        ).start()

    def _execute(self, best_dir: Optional[str], prompts: Optional[list[str]]) -> None:
        try:
            self.pub_start.publish(Bool(data=False))
            waypoints = waypoints_from_map(
                read_obj_map(best_dir, self.get_logger().error), prompts or [])
            if not waypoints:
                self.get_logger().error("no matching objects in the map")
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
            if here is not None and math.hypot(here[0] - x, here[1] - y) <= self.reach_m:
                return
            time.sleep(0.5)
        self.get_logger().warn(
            f"waypoint ({x:.2f}, {y:.2f}) not reached in {self.waypoint_timeout_s:.0f}s")


def main(args=None):
    rclpy.init(args=args)
    spin_reasoner(InstructionReasoner())


if __name__ == "__main__":
    main()
