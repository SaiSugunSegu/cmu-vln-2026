#!/usr/bin/env python3
"""smart_vlm supervisor — the mission clock and gatekeeper of the AI module.

One supervisor lives exactly one question: `smart_vlm.launch` is a disposable unit
(sam_node + supervisor + reasoner + scene source) that the harness relaunches per
question, matching the challenge's rule that the system is restarted for each language
command. That is why the budget below starts at node construction and why nothing here
needs an in-process reset path.

It supervises by publishing gates, never by managing processes:

  /pipeline/ready   (latched)  models loaded — the harness may publish the question
  /pipeline/armed   (latched)  SAM holds this question's real prompts — release the
                               scene source. This is what stops the bag from playing
                               past a detector that is still loading weights, and what
                               guarantees SAM never infers on the config's placeholder
                               prompts.
  /pipeline/explore_done       exploration is closed — reasoner, answer now.

Plus the safety net the score depends on: at T-30 it publishes a best guess rather than
letting a question go unanswered.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from smart_vlm.mission_clock import MissionBudget, MissionClock
from smart_vlm.question import QuestionType, classify, question_text

# The only inputs allowed at test time (README 'System Outputs'). Names only: liveness
# comes from count_publishers(), so we never pay to deserialize 10 Hz panoramas and
# 5 Hz point clouds just to prove they are arriving. sam_node's own heartbeat already
# reports /camera/image frame counts.
ALLOWED_TOPICS = (
    "/camera/image",
    "/registered_scan",
    "/sensor_scan",
    "/terrain_map",
    "/terrain_map_ext",
    "/state_estimation",
)

# sam_node reports this while it holds weights but no prompts (sam_node wait_for_prompts).
SAM_READY_STATES = frozenset({"ready", "awaiting_prompts"})


def _latched(depth: int = 1) -> QoSProfile:
    """Transient-local QoS, so a subscriber that joins late still sees the last sample."""
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


class SmartVLM(Node):

    HEARTBEAT_S = 5.0
    TICK_S = 1.0

    def __init__(self) -> None:
        super().__init__("smart_vlm_node")

        budget = MissionBudget(
            question_budget_s=self._param("question_budget_s", 600.0),
            explore_timeout_s=self._param("explore_timeout_s", 120.0),
            answer_reserve_s=self._param("answer_reserve_s", 90.0),
            fallback_reserve_s=self._param("fallback_reserve_s", 30.0),
        )
        self.fallback_count = int(self._param("fallback_count", 1))
        self.ready_timeout_s = self._param("ready_timeout_s", 300.0)
        self.require_vqa_ready = bool(self._param("require_vqa_ready", True))
        self.idle_stop_s = self._param("idle_stop_s", 5.0)
        self.startup_grace_s = self._param("startup_grace_s", 15.0)

        # time.monotonic throughout: these are durations, and wall time can step.
        self.clock = MissionClock(budget, time.monotonic())

        self.question: Optional[str] = None
        self.qtype: Optional[QuestionType] = None
        self.sam_status: Optional[str] = None
        self.vqa_status: Optional[str] = None
        self.armed_prompts: Optional[list[str]] = None
        self.answered = False
        self.fallback_published = False
        self.ready_published = False
        self.armed_published = False
        self.explore_done_published = False
        self.odom_count = 0
        self.last_odom_t: Optional[float] = None

        # -- inputs ---------------------------------------------------------
        self.create_subscription(String, "/challenge_question", self._on_question, 10)
        self.create_subscription(String, "/sam3/status", self._on_sam_status, _latched())
        self.create_subscription(String, "/qwen_vqa/status", self._on_vqa_status, _latched())
        self.create_subscription(String, "/sam3/prompts_ack", self._on_prompts_ack, 10)
        # The one allowed topic we actually subscribe to: Odometry is tiny, and its
        # arrival (then its absence) is how we detect the scene starting and finishing.
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10)
        # Answer heads publish these; we only watch, so the T-30 net stays quiet when
        # a real answer already went out.
        self.create_subscription(Int32, "/numerical_response", self._on_numerical, 10)
        self.create_subscription(Marker, "/selected_object_marker", self._on_marker, 10)

        # -- outputs --------------------------------------------------------
        self.pub_ready = self.create_publisher(String, "/pipeline/ready", _latched())
        self.pub_armed = self.create_publisher(String, "/pipeline/armed", _latched())
        self.pub_explore_done = self.create_publisher(String, "/pipeline/explore_done", 10)
        self.pub_status = self.create_publisher(String, "/smart_vlm/status", _latched())
        # Second writer on /numerical_response by design: numerical_reasoner owns the
        # real answer, this one only fires as the T-30 fallback.
        self.pub_answer = self.create_publisher(Int32, "/numerical_response", 10)

        self.create_timer(self.TICK_S, self._tick)
        self.create_timer(self.HEARTBEAT_S, self._heartbeat)
        self._publish_status()
        self.get_logger().info(
            f"smart_vlm supervisor up — budget {budget.question_budget_s:.0f}s "
            f"(explore {budget.explore_timeout_s:.0f}s from first data, "
            f"answer at T-{budget.answer_reserve_s:.0f}s, "
            f"fallback at T-{budget.fallback_reserve_s:.0f}s)"
        )

    def _param(self, name: str, default: float | bool):
        self.declare_parameter(name, default)
        value = self.get_parameter(name).value
        return value if isinstance(default, bool) else float(value)

    # ---- readiness -------------------------------------------------------

    def _on_sam_status(self, msg: String) -> None:
        self.sam_status = (msg.data or "").strip()
        self._check_ready()

    def _on_vqa_status(self, msg: String) -> None:
        self.vqa_status = (msg.data or "").strip()
        self._check_ready()

    def _check_ready(self, degraded: bool = False) -> None:
        """Announce the pipeline once every model that must be up, is up."""
        if self.ready_published:
            return
        sam_ok = self.sam_status in SAM_READY_STATES
        vqa_ok = self.vqa_status == "ready" or not self.require_vqa_ready
        if not degraded and not (sam_ok and vqa_ok):
            return

        self.ready_published = True
        warmup_s = time.monotonic() - self.clock.t0
        payload = {
            "sam": self.sam_status,
            "vqa": self.vqa_status,
            "degraded": degraded,
            "warmup_s": round(warmup_s, 1),
        }
        self.pub_ready.publish(String(data=json.dumps(payload)))
        # Worth logging loudly: this is the model-load tax the challenge charges us
        # against the same 10-minute budget as exploration.
        log = self.get_logger().warn if degraded else self.get_logger().info
        log(f"pipeline ready after {warmup_s:.1f}s (sam={self.sam_status} vqa={self.vqa_status})"
            + (" — DEGRADED, readiness timed out" if degraded else ""))
        self._publish_status()

    def _on_prompts_ack(self, msg: String) -> None:
        """SAM has this question's real prompts — release the scene source.

        Until this fires the detector is either loading weights or holding the config's
        placeholder prompts, so any frame it saw would be wasted.
        """
        if self.armed_published:
            return
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("could not parse /sam3/prompts_ack")
            return
        if not ack.get("ok"):
            self.get_logger().error(f"SAM rejected prompts: {ack.get('error')}")
            return

        self.armed_published = True
        self.armed_prompts = list(ack.get("prompts") or [])
        payload = {
            "prompts": self.armed_prompts,
            "run_dir": ack.get("run_dir"),
            "elapsed_s": round(self.clock.elapsed(time.monotonic()), 1),
        }
        self.pub_armed.publish(String(data=json.dumps(payload)))
        self.get_logger().info(
            f"SAM armed with {self.armed_prompts} -> {ack.get('run_dir')}; "
            f"releasing the scene source")
        self._publish_status()

    # ---- question --------------------------------------------------------

    def _on_question(self, msg: String) -> None:
        if self.question is not None:
            return  # the eval node repeats at 1 Hz — take the first
        self.question = question_text(msg.data)
        self.qtype = classify(msg.data)
        self.get_logger().info(f"QUESTION [{self.qtype.value}]: {self.question}")
        self._publish_status()

    # ---- answers we observe ---------------------------------------------

    def _on_numerical(self, msg: Int32) -> None:
        if self.fallback_published:
            return  # our own fallback echoing back
        self.answered = True
        self.get_logger().info(f"observed /numerical_response={msg.data}")
        self._publish_status()

    def _on_marker(self, _msg: Marker) -> None:
        self.answered = True

    # ---- scene progress --------------------------------------------------

    def _on_odom(self, _msg: Odometry) -> None:
        now = time.monotonic()
        self.odom_count += 1
        self.last_odom_t = now
        if not self.clock.exploring_started:
            self.clock.mark_exploring(now)
            self.get_logger().info(
                f"sensor data flowing — exploring until "
                f"T+{self.clock.explore_deadline - self.clock.t0:.0f}s")
            self._publish_status()

    def _scene_finished(self, now: float) -> bool:
        """A bag that has played out stops publishing odometry; the sim never does.

        The sim streams /state_estimation at 100-200 Hz for as long as it runs, so this
        only ever fires on a finished replay — where stopping early is worth real points
        (README 'Timing': finishing early earns a tie-break bonus).
        """
        if not self.clock.exploring_started or self.last_odom_t is None:
            return False
        return (now - self.last_odom_t) >= self.idle_stop_s

    # ---- clock -----------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()

        if not self.ready_published and self.clock.elapsed(now) >= self.ready_timeout_s:
            self._check_ready(degraded=True)

        if self._scene_finished(now):
            self._finish_exploration("bag_end", now)
        elif now >= self.clock.explore_deadline:
            reason = "budget" if self.clock.explore_deadline >= self.clock.answer_deadline \
                else "timeout"
            self._finish_exploration(reason, now)

        if now >= self.clock.fallback_deadline:
            self._publish_fallback()

    def _finish_exploration(self, reason: str, now: float) -> None:
        """Close exploration exactly once and hand the question to the reasoner."""
        if self.explore_done_published:
            return
        self.explore_done_published = True
        elapsed = round(self.clock.elapsed(now), 1)
        self.pub_explore_done.publish(String(data=json.dumps(
            {"reason": reason, "elapsed_s": elapsed, "scene_done": True})))
        self.get_logger().info(
            f"exploration closed after {elapsed}s ({reason}) — /pipeline/explore_done sent")
        self._publish_status()

    def _publish_fallback(self) -> None:
        """Answer *something* at T-30 — partial credit beats silence."""
        if self.answered or self.fallback_published:
            return
        self.fallback_published = True

        if self.qtype is QuestionType.NUMERICAL:
            self.pub_answer.publish(Int32(data=self.fallback_count))
            self.get_logger().error(
                f"FALLBACK: no answer by T-{self.clock.budget.fallback_reserve_s:.0f}s — "
                f"publishing /numerical_response={self.fallback_count} as a guess")
        elif self.qtype is None:
            self.get_logger().error("FALLBACK: budget spent and no question ever arrived")
        else:
            # A fabricated marker scores ~0 on overlap anyway, and its centre is used as
            # a navigation waypoint (README 'System Inputs') — so a wrong one actively
            # drives the robot somewhere wrong. Silence is the better failure.
            # TODO(M4): emit the best label-matched instance instead of nothing.
            self.get_logger().error(
                f"FALLBACK: no answer for a {self.qtype.value} question and no guess "
                f"is available — publishing nothing")
        self._publish_status()

    # ---- diagnostics -----------------------------------------------------

    def _heartbeat(self) -> None:
        now = time.monotonic()
        elapsed = self.clock.elapsed(now)
        # count_publishers is a graph query, not a data subscription: it tells us a topic
        # has a source without decoding a single message off it.
        dead = [t for t in ALLOWED_TOPICS if self.count_publishers(t) == 0]
        if dead and elapsed > self.startup_grace_s:
            self.get_logger().warn(f"NO PUBLISHER on: {dead}")

        odom_hz = self.odom_count / elapsed if elapsed > 0 else 0.0
        self.get_logger().info(
            f"{self.clock.phase_at(now).value} | t+{elapsed:.0f}s | "
            f"odom {odom_hz:.0f} Hz | sam={self.sam_status} vqa={self.vqa_status} | "
            f"answered={self.answered}")
        self._publish_status()

    def _publish_status(self) -> None:
        now = time.monotonic()
        payload = {
            **self.clock.snapshot(now),
            "question": self.question,
            "qtype": self.qtype.value if self.qtype else None,
            "sam": self.sam_status,
            "vqa": self.vqa_status,
            "ready": self.ready_published,
            "armed": self.armed_published,
            "armed_prompts": self.armed_prompts,
            "explore_done": self.explore_done_published,
            "answered": self.answered,
            "fallback": self.fallback_published,
        }
        self.pub_status.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = SmartVLM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
