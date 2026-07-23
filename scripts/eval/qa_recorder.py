#!/usr/bin/env python3
"""Per-question recorder. Runs INSIDE the ai_module container.

Mimics the challenge_evaluation_node: waits for the system to be ready,
publishes one question at 1 Hz on /challenge_question, records everything the
AI module answers, plus the executed trajectory, until timeout or completion.

Usage:
  python3 qa_recorder.py --question "How many pillows are on the bed?" \
      --qtype numerical --scene hotel_room_1 --qid n0 \
      --timeout 600 --out /tmp/eval_results/hotel_room_1_n0.json
"""
import argparse, json, math, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker


class QARecorder(Node):
    def __init__(self, args):
        super().__init__("qa_recorder")
        self.args = args
        self.t_start = None          # set when system is ready
        self.t_answer = None
        self.numerical = None
        self.marker = None
        self.waypoints = []          # [(t, x, y, theta)]
        self.trajectory = []         # [(t, x, y)] from /state_estimation, ~2 Hz
        self._last_traj_t = 0.0
        self._system_ready = False

        self.q_pub = self.create_publisher(String, "/challenge_question", 10)
        self.create_subscription(PointCloud2, "/registered_scan", self._on_scan, 10)
        self.create_subscription(Int32, "/numerical_response", self._on_int, 10)
        self.create_subscription(Marker, "/selected_object_marker", self._on_marker, 10)
        self.create_subscription(Pose2D, "/way_point_with_heading", self._on_wp, 10)
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 50)
        self.create_timer(1.0, self._tick)  # 1 Hz question publish, like eval node

    # -- readiness: first registered scan means sim + state estimation are up
    def _on_scan(self, _msg):
        if not self._system_ready:
            self._system_ready = True
            self.t_start = time.time()
            self.get_logger().info("System ready — clock started, publishing question")

    def _tick(self):
        if not self._system_ready:
            return
        self.q_pub.publish(String(data=self.args.question))
        if self._done():
            self._finish(timed_out=False)
        elif time.time() - self.t_start > self.args.timeout:
            self._finish(timed_out=True)

    def _now(self):
        return round(time.time() - self.t_start, 2) if self.t_start else None

    def _on_int(self, msg):
        if self.numerical is None:
            self.numerical, self.t_answer = int(msg.data), self._now()

    def _on_marker(self, msg):
        self.marker = {
            "label": msg.text,
            "center": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            "size": [msg.scale.x, msg.scale.y, msg.scale.z],
        }
        self.t_answer = self._now()

    def _on_wp(self, msg):
        self.waypoints.append((self._now(), msg.x, msg.y, msg.theta))
        self.t_answer = self._now()   # instruction answers "end" at last waypoint reached

    def _on_odom(self, msg):
        if not self._system_ready:
            return
        t = time.time() - self.t_start
        if t - self._last_traj_t >= 0.5:
            p = msg.pose.pose.position
            self.trajectory.append((round(t, 2), round(p.x, 3), round(p.y, 3)))
            self._last_traj_t = t

    def _done(self):
        qt = self.args.qtype
        if qt == "numerical":
            return self.numerical is not None
        if qt == "object_reference":
            # marker + settle time (module may refine); stop 20 s after last change
            return self.marker is not None and self._now() - (self.t_answer or 0) > 20
        if qt == "instruction_following":
            # done when robot idle 30 s after last waypoint and >=1 waypoint sent
            if not self.waypoints:
                return False
            if self._now() - self.waypoints[-1][0] < 30:
                return False
            return self._robot_idle(15.0)
        return False

    def _robot_idle(self, window_s):
        pts = [p for p in self.trajectory if p[0] > (self._now() or 0) - window_s]
        if len(pts) < 2:
            return False
        d = math.dist(pts[0][1:], pts[-1][1:])
        return d < 0.2

    def _finish(self, timed_out):
        out = {
            "scene": self.args.scene, "qid": self.args.qid, "qtype": self.args.qtype,
            "question": self.args.question, "timed_out": timed_out,
            "t_answer_s": self.t_answer, "t_total_s": self._now(),
            "numerical": self.numerical, "marker": self.marker,
            "waypoints": self.waypoints, "trajectory": self.trajectory,
        }
        with open(self.args.out, "w") as f:
            json.dump(out, f, indent=1)
        self.get_logger().info(f"Wrote {self.args.out} (timed_out={timed_out})")
        raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--qtype", required=True,
                    choices=["numerical", "object_reference", "instruction_following"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--qid", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = QARecorder(args)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
