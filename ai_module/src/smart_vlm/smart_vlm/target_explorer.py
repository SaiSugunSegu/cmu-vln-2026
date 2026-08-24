#!/usr/bin/env python3
"""target_explorer — turn "which side of this object have I never looked from" into
"go stand here", until every reachable side of every target has been seen.

Exploration runs in TWO phases, gated by `tare_explore_priority` (see target_coverage's
`default_params`). Phase one hands the room to TARE's own frontier exploration and publishes
no viewpoints at all, so the tour is solved over the whole grid; it ends when TARE reports
finished or `tare_explore_max_time` elapses. Phase two is everything below, unchanged --
targets steer the tour. Discovery comes first because a target the robot never drove past is
one no amount of re-viewing will find; with the flag off, targets compete from the first
second, which is how this node behaved before.

The ROS shell only. Every rule lives in `target_coverage`, which imports no rclpy and is
unit-tested under `just test`; this file is subscriptions, a publish timer, and one file
write. Read target_coverage's module docstring for why the coverage model looks the way it
does — in short: an object the robot never circled has no centroid, so it is ABSENT from
obj_map.json rather than merely loosely boxed, and the answer path cannot name it at all.

Nothing here checks traversability, on purpose. These are desiderata; TARE snaps each to its
nearest *candidate* viewpoint — collision-free, in line of sight and graph-connected by
construction — and reports back what it could not place, which is what marks a sector
blocked.

  /sam3/set_prompts                      (sub)  the question's target labels
  /sam3/prompts_ack                      (sub)  this run's output directory
  /exploration/object_targets            (sub)  every tracked object, published or not, + bins
  /exploration/target_viewpoint_feedback (sub)  TARE's accepted / unreachable verdicts
  /state_estimation                      (sub)  robot position: routing order AND arrival
  /pipeline/explore_done                 (sub)  stop emitting, print the close-out
  /exploration/target_viewpoints         (pub)  PoseArray of standing positions, best first
  /exploration/target_preempt            (pub)  may targets outrank frontier exploration
  /exploration/target_coverage_done      (pub)  every target covered — exploration may stop
  /exploration/target_status             (pub)  per-sector progress; a heartbeat, not a side effect
"""
from __future__ import annotations

import json
import os
import time

import rclpy

from captioner.ros_utils import shutdown_guard
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

# The ONE place a prompt becomes a class name. map_node names every object with it, so
# comparing raw prompts against map labels silently fails on any multi-word class: the map
# calls it `pottedplant`, the question asks for `potted plant`, no goal is ever built and the
# label reads as "never found" for the whole run. Measured on the 13-scene sim sweep: 9 of 13
# questions carried a multi-word target and office_1 (both targets multi-word) requested zero
# viewpoints across its entire 150 s window.
from sam_mapper.detections import default_label

from smart_vlm.target_coverage import (TARE_DEFAULTS, WAYPOINT_XY_RADIUS_M, CoverageModel,
                                       default_params, validate_params)

#: Written next to the run's obj_map.json and manifest.json. Without it the only trace of what
#: coverage a run achieved is launch stdout, which the eval orchestrator does not capture —
#: so a sweep leaves no evidence of whether exploration or perception was the limiting factor.
COVERAGE_FILE = "target_coverage.json"


class TargetExplorer(Node):

    def __init__(self):
        super().__init__("target_explorer")

        self.defaults = default_params()
        for name, value in self.defaults.items():
            self.declare_parameter(name, value)
        # Which tare_planner scenario yaml is in force. Same value smart_vlm.launch hands
        # explore.launch, so both halves read the same file.
        self.declare_parameter("tare_scenario", "cmu_challenge")

        params = {k: self.get_parameter(k).value for k in self.defaults}
        params.update(self._tare_contract())
        validate_params(params)
        self.model = CoverageModel(params)
        self.robot = None
        self.run_dir = None
        self.run_id = None
        self.tare_finished = False
        self.stopped = False
        self._last_preempt: bool | None = None
        self._complete_since: float | None = None
        self._coverage_done = False
        self._last_report_t = 0.0
        #: Phase 1 of two: TARE sweeps the room on frontier exploration alone, then targets
        #: take over. Latched false once it ends -- see `_tare_phase_active`.
        self._tare_phase = bool(self.p("tare_explore_priority"))
        self._tare_phase_started: float | None = None

        self.create_subscription(String, "/sam3/set_prompts", self._on_prompts, 10)
        self.create_subscription(String, "/sam3/prompts_ack", self._on_ack, 10)
        self.create_subscription(String, "/exploration/object_targets", self._on_objects, 2)
        self.create_subscription(
            String, "/exploration/target_viewpoint_feedback", self._on_feedback, 2)
        self.create_subscription(Odometry, "/state_estimation", self._on_odom, 10)
        self.create_subscription(String, "/pipeline/explore_done", self._on_done, 10)
        # Read only to put TARE's own verdict in the report next to ours. The supervisor owns
        # the decision; recording both here is what makes a run say which half was the laggard.
        self.create_subscription(Bool, "/exploration_finish", self._on_tare_finished, 2)

        self.pub_viewpoints = self.create_publisher(
            PoseArray, "/exploration/target_viewpoints", 2)
        self.pub_status = self.create_publisher(String, "/exploration/target_status", 2)
        # Tells TARE when targets may preempt frontier exploration. Held false until every
        # target label has an instance: an undiscovered label can be found only by ordinary
        # exploration, and preempt restricts the global tour to subspaces holding a target.
        self.pub_preempt = self.create_publisher(Bool, "/exploration/target_preempt", 2)
        # Lets the supervisor close exploration the moment coverage is done rather than at the
        # end of the window. Finishing early is worth real points (README 'Timing').
        self.pub_coverage_done = self.create_publisher(
            Bool, "/exploration/target_coverage_done", 2)

        self.create_timer(1.0 / max(float(self.p("publish_hz")), 0.1), self._publish)
        self.get_logger().info("target_explorer up — waiting for /sam3/set_prompts")

    def p(self, name):
        return self.get_parameter(name).value

    def _tare_contract(self) -> dict:
        """Read the two TARE numbers this node has to agree with, from TARE's own yaml.

        `kMaxTargetViewPointNum` and `kTargetViewPointSnapMaxDist` are the planner's
        contract: send more poses than the first and the tail is silently unheard; assume
        the wrong second and both the arrival radius and the position-quantisation argument
        are wrong. Copying them into Python meant three places to change and no way to notice
        when they drifted -- the C++ default for the snap distance had already been sitting at
        1.5 against the yaml's 0.5.

        Read from the installed share directory, which is where the launch also gets it, so
        this is the same file the planner is running on. If it cannot be read, keep the
        declared fallbacks and say so loudly: guessing silently is what this removes.
        """
        scenario = self.p("tare_scenario")
        try:
            from ament_index_python.packages import get_package_share_directory
            import yaml

            # No "config" component: tare_planner's CMakeLists installs `config/` WITH a
            # trailing slash, which copies the contents to share/tare_planner/ rather than
            # share/tare_planner/config/. explore.launch already reads it from there, and
            # this is the same file. Getting it wrong is silent -- the fallback below happens
            # to match today's yaml, so the only symptom was 53 warnings in one sweep and a
            # contract that would drift the moment either value changed.
            path = os.path.join(get_package_share_directory("tare_planner"),
                                f"{scenario}.yaml")
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            knobs = loaded["tare_planner_node"]["ros__parameters"]
            contract = {
                "max_viewpoints": int(knobs["kMaxTargetViewPointNum"]),
                "max_viewpoints_cap": int(knobs["kMaxTargetViewPointNum"]),
                "snap_max_dist_m": float(knobs["kTargetViewPointSnapMaxDist"]),
            }
            contract["arrival_radius_m"] = (contract["snap_max_dist_m"]
                                            + WAYPOINT_XY_RADIUS_M)
            self.get_logger().info(
                f"TARE contract from {scenario}.yaml: "
                f"max_viewpoints={contract['max_viewpoints']}, "
                f"snap={contract['snap_max_dist_m']:.2f} m, "
                f"arrival_radius={contract['arrival_radius_m']:.2f} m")
            return contract
        except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warning(
                f"could not read tare_planner/{scenario}.yaml ({exc}) — falling back to "
                f"{TARE_DEFAULTS}; verify these still match the planner")
            return {"max_viewpoints_cap": TARE_DEFAULTS["max_target_viewpoints"]}

    # -- inputs ---------------------------------------------------------------

    def _on_prompts(self, msg: String) -> None:
        try:
            prompts = json.loads(msg.data).get("prompts") or []
        except (ValueError, AttributeError):
            self.get_logger().warning("un-parseable /sam3/set_prompts, ignoring")
            return
        labels = {default_label(str(x)) for x in prompts if str(x).strip()}
        if self.model.set_targets(labels):
            self._rearm(f"targets: {sorted(self.model.targets)}")

    def _on_ack(self, msg: String) -> None:
        """The run directory to report into, and the authoritative re-arm signal.

        `run_id` rather than the prompt set: map_node drops its entire map on a new run_id
        (`_reset_map`), and two consecutive questions can share prompts — "how many chairs are
        by the window" then "find the chair by the window". Keying the reset on labels would
        leave this node holding goals built from a map that no longer exists.
        """
        try:
            ack = json.loads(msg.data)
        except (ValueError, AttributeError):
            return
        if ack.get("run_dir"):
            self.run_dir = str(ack["run_dir"])
        run_id = ack.get("run_id")
        if run_id is not None and run_id != self.run_id:
            self.run_id = run_id
            self.model.reset_map_state()          # keep this question's labels, drop the map
            self._rearm(f"run {run_id}: map reset, goals cleared")

    def _rearm(self, message: str) -> None:
        """Clear every latch. One process can serve several questions, and without this the
        node is dead after the first."""
        self.stopped, self._last_preempt = False, None
        self._complete_since, self._coverage_done = None, False
        self.tare_finished = False
        self._tare_phase = bool(self.p("tare_explore_priority"))
        self._tare_phase_started = None
        self.get_logger().info(message)

    def _on_tare_finished(self, msg: Bool) -> None:
        if msg.data and not self.tare_finished:
            self.tare_finished = True
            self.get_logger().info("TARE reports exploration finished")

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.robot = (p.x, p.y, p.z)
        # Arrival is checked here, not on the publish timer: odometry runs at 100-200 Hz and
        # the robot can pass through a viewpoint and leave again between two 2 Hz ticks.
        self.model.note_position(self.robot)

    def _on_done(self, _msg: String) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.pub_viewpoints.publish(PoseArray())   # TARE reverts to stock on the empty array
        self.pub_preempt.publish(Bool(data=False))
        self._write_report(final=True)
        self.get_logger().info(f"explore_done — {self.model.close_out()}")

    def _on_objects(self, msg: String) -> None:
        if self.stopped or not self.model.targets or self.robot is None:
            return
        try:
            objects = json.loads(msg.data).get("objects") or []
        except (ValueError, AttributeError):
            return
        self.model.ingest(objects, self.robot, time.monotonic())

    def _on_feedback(self, msg: String) -> None:
        """Hand TARE's verdicts to the model — accepted, unreachable, and the silent `far`."""
        try:
            payload = json.loads(msg.data)
        except (ValueError, AttributeError):
            return
        if not isinstance(payload, dict):
            return
        for key in self.model.note_feedback(payload):
            goal = self.model.goals[key]
            self.get_logger().info(
                f"'{goal.label}' refused — {self.model.sector_states(goal)}",
                throttle_duration_sec=5.0)

    # -- output ---------------------------------------------------------------

    def _tare_phase_active(self, now: float) -> bool:
        """Is TARE still doing the global sweep, before targets are allowed to steer it?

        Latched: it can end, never restart. Two ways out, whichever comes first --

          * TARE reports `/exploration_finish`, i.e. it believes the room is explored;
          * `tare_explore_max_time` elapses, which is the one that actually fires. Across a
            measured 15-scene sweep TARE never raised the finish signal once, so without a cap
            the target phase would simply never begin.

        The clock starts when the question's targets land, not at node construction: this
        timer runs while the robot drives, and anchoring it earlier would spend the global
        phase on model loading, before the scene is even released.
        """
        if not self._tare_phase:
            return False
        if not self.model.targets:
            return True                      # not armed yet; nothing to steer with anyway
        if self._tare_phase_started is None:
            self._tare_phase_started = now
            self.get_logger().info(
                f"global exploration first — TARE has the room to itself for up to "
                f"{float(self.p('tare_explore_max_time')):.0f}s before targets take over")
        elapsed = now - self._tare_phase_started
        limit = float(self.p("tare_explore_max_time"))
        if self.tare_finished or elapsed >= limit:
            self._tare_phase = False
            self.get_logger().info(
                f"global exploration ended after {elapsed:.0f}s "
                f"({'TARE finished' if self.tare_finished else f'{limit:.0f}s cap'}) "
                f"— target-driven exploration takes over")
            return False
        return True

    def _publish(self) -> None:
        if self.stopped or self.robot is None:
            return
        now = time.monotonic()

        if self._tare_phase_active(now):
            # An empty PoseArray is what reverts TARE to stock frontier exploration -- with no
            # priority cells, `grid_world.cpp`'s `priority_only` is false whatever preempt
            # says, so the tour is solved over the whole grid. Preempt goes out false as well
            # so nothing is left latched on when the phase ends. Coverage still accrues from
            # whatever the sweep happens to drive past; it is only the STEERING that waits.
            self.pub_viewpoints.publish(PoseArray())
            self.pub_preempt.publish(Bool(data=False))
            self._publish_coverage_done(now)
            self.pub_status.publish(String(data=json.dumps(self.model.status(now))))
            self._write_report()
            return

        requests = self.model.requests(self.robot, now)

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for x, y, _key, _sector, _bin in requests:
            pose = Pose()
            pose.position.x, pose.position.y = x, y
            # Must be the robot's own z: InLocalPlanningHorizon rejects anything more than
            # 0.4 m off it, and kUseTerrainHeight is false so every candidate viewpoint sits
            # at exactly this height.
            pose.position.z = self.robot[2]
            # Orientation unused: the camera is a 360 equirect panorama, so which way the
            # robot faces does not change what it sees. Identity, not a computed heading.
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_viewpoints.publish(msg)

        self._publish_preempt(now)
        self._publish_coverage_done(now)
        self.pub_status.publish(String(data=json.dumps(self.model.status(now))))
        self._write_report()

        if requests:
            committed = self.model.committed
            self.get_logger().info(
                f"{len(requests)} viewpoint(s) for {len(self.model.pending())} pending goal(s); "
                f"committed='{committed[0] if committed else '-'}'",
                throttle_duration_sec=5.0)

    def _publish_preempt(self, now: float) -> None:
        preempt = self.model.preempt(now)
        if preempt != self._last_preempt:
            self.get_logger().info(
                f"target preempt {'ON' if preempt else 'off'} — "
                f"{len(self.model.pending())} pending, found: {sorted(self.model.found_labels())}, "
                f"still unseen: {sorted(self.model.targets - self.model.found_labels())}")
            self._last_preempt = preempt
        self.pub_preempt.publish(Bool(data=preempt))

    def _publish_coverage_done(self, now: float) -> None:
        """Latch coverage-complete only once it has held for a full perception round trip.

        map_node publishes at ~2.2 Hz and TARE's feedback at ~1 Hz, so a single frame in which
        regularization happens to drop an object would otherwise end exploration early and
        permanently — the supervisor's explore_done is one-shot.
        """
        if self._coverage_done:
            self.pub_coverage_done.publish(Bool(data=True))
            return
        if not self.model.coverage_complete():
            self._complete_since = None
            self.pub_coverage_done.publish(Bool(data=False))
            return
        if self._complete_since is None:
            self._complete_since = now
        if (now - self._complete_since) < float(self.p("coverage_hold_s")):
            self.pub_coverage_done.publish(Bool(data=False))
            return
        self._coverage_done = True
        self.get_logger().info(f"target coverage complete — {self.model.close_out()}")
        self.pub_coverage_done.publish(Bool(data=True))

    def _write_report(self, final: bool = False) -> None:
        """Atomically, so a kill mid-write leaves the previous complete report — the same
        rule map_node uses for obj_map.json.

        Rate-limited well below the publish timer: this exists so an interrupted run still has
        a usable report, not as a trace log, and the final write on explore_done is the one
        anything reads.
        """
        if not self.run_dir:
            return
        now = time.monotonic()
        if not final and (now - self._last_report_t) < 2.0:
            return
        self._last_report_t = now
        payload = {
            "final": final,
            # TARE's half of the stop condition, recorded alongside ours: a run that ended on
            # the timeout is diagnosed by which of these two was still false.
            "tare_finished": self.tare_finished,
            "target_coverage_done": self._coverage_done,
            # Which of the two exploration phases this run was in when the report was
            # written. A sweep that covered nothing reads very differently if it spent the
            # whole window in the global phase.
            "phase": "global" if self._tare_phase else "target",
            "summary": self.model.summary(),
            "status": self.model.status(time.monotonic()),
        }
        path = os.path.join(self.run_dir, COVERAGE_FILE)
        # Plain open(), not tempfile.NamedTemporaryFile: that creates 0600, and os.replace
        # carries the mode across, so this file landed -rw------- next to obj_map.json and
        # manifest.json at 0644 and could not be read from the host at all. The same
        # temp-plus-replace shape every other writer here uses (map_node.write_obj_map,
        # report_utils.write, NumericalReasoner._save_manifest) goes through open() and
        # inherits the umask like its neighbours.
        tmp = path + ".tmp"
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(tmp, path)
        except OSError as exc:
            self.get_logger().warning(f"could not write {path}: {exc}",
                                      throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = TargetExplorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        with shutdown_guard():
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
