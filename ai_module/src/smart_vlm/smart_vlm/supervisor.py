#!/usr/bin/env python3
"""smart_vlm supervisor — the brain-stem of the AI module.

v0 responsibilities (today):
  - subscribe to the 6 allowed topics, verify they're alive, log rates
  - receive /challenge_question, classify type, log
  - own the mission clock (question received -> T-90s handoff -> T-30s fallback)
  - exploration control hooks (start/stop TARE, later: reobserve)

Later: decision gate, scene-graph queries, answer heads move in here / alongside.
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

ALLOWED_TOPICS = {
    "/camera/image": Image,
    "/registered_scan": PointCloud2,
    "/sensor_scan": PointCloud2,
    "/terrain_map": PointCloud2,
    "/terrain_map_ext": PointCloud2,
    "/state_estimation": Odometry,
}

QUESTION_BUDGET_S = 600.0
ANSWER_MODE_S = 90.0      # T-90: stop exploring, start answering
FALLBACK_S = 30.0         # T-30: publish best guess no matter what


def classify(q: str) -> str:
    ql = q.lower().strip()
    if ql.startswith(("how many", "count")):
        return "numerical"
    if ql.startswith(("find", "the ")):
        return "object_reference"
    return "instruction_following"


class Supervisor(Node):
    def __init__(self):
        super().__init__("smart_vlm_supervisor")
        self.question = None
        self.qtype = None
        self.t_question = None
        self.msg_counts = {t: 0 for t in ALLOWED_TOPICS}
        self.t0 = time.time()

        for topic, mtype in ALLOWED_TOPICS.items():
            self.create_subscription(mtype, topic, self._counter(topic), 10)
        self.create_subscription(String, "/challenge_question", self._on_question, 10)

        # outputs (owned by supervisor so there is exactly ONE writer per topic)
        self.pub_int = self.create_publisher(Int32, "/numerical_response", 10)
        self.pub_marker = self.create_publisher(Marker, "/selected_object_marker", 10)
        self.pub_wp = self.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self.pub_status = self.create_publisher(String, "/smart_vlm/status", 10)

        self.create_timer(5.0, self._heartbeat)
        self.create_timer(1.0, self._clock_check)
        self.get_logger().info("smart_vlm supervisor up — waiting for topics + question")

    def _counter(self, topic):
        def cb(_msg):
            self.msg_counts[topic] += 1
        return cb

    def _on_question(self, msg):
        if self.question is not None:
            return  # eval node repeats at 1 Hz — take the first
        self.question = msg.data
        self.qtype = classify(msg.data)
        self.t_question = time.time()
        self.get_logger().info(f"QUESTION [{self.qtype}]: {self.question}")
        self._start_exploration()

    # ---- exploration control hooks -------------------------------------
    def _start_exploration(self):
        """TARE is launched alongside (see smart_vlm.launch). If it needs a
        kick-off (the stack's 'Resume Navigation to Goal'), publish it here.
        TODO(after submodule init): confirm TARE's start mechanism in
        system_simulation_with_exploration_planner.sh — likely automatic on
        launch, else a boundary/initial waypoint publish."""
        self._status("exploring")

    def _stop_exploration(self):
        """TODO: gate TARE output (lifecycle/kill or stop republishing its
        waypoints once we own /way_point_with_heading via the mux)."""
        self._status("answer_mode")

    # ---- clocks ---------------------------------------------------------
    def _elapsed(self):
        return time.time() - self.t_question if self.t_question else 0.0

    def _clock_check(self):
        if self.t_question is None:
            return
        left = QUESTION_BUDGET_S - self._elapsed()
        if left <= FALLBACK_S:
            self._fallback_answer()
        elif left <= ANSWER_MODE_S:
            self._stop_exploration()

    def _fallback_answer(self):
        """Always answer *something* — partial credit beats silence."""
        if self.qtype == "numerical":
            self.pub_int.publish(Int32(data=2))  # TODO: modal count from instances
        # TODO object_reference: best label-match instance marker
        # TODO instruction: publish goal-object waypoint
        self._status("fallback_answered")

    # ---- diagnostics -----------------------------------------------------
    def _heartbeat(self):
        dt = time.time() - self.t0
        rates = {t: round(c / dt, 1) for t, c in self.msg_counts.items()}
        dead = [t for t, c in self.msg_counts.items() if c == 0]
        self.get_logger().info(f"rates(hz)={rates}")
        if dead and dt > 15:
            self.get_logger().warn(f"NO DATA on: {dead}")

    def _status(self, s):
        self.pub_status.publish(String(data=s))


def main():
    rclpy.init()
    rclpy.spin(Supervisor())


if __name__ == "__main__":
    main()
