#!/usr/bin/env python3
"""Category-3 reasoner: extract targets -> arm SAM -> VLM route -> ordered Pose2D.

The answer to an instruction-following command is the path the robot drives, not a message, so
this node's product is a sequence on /way_point_with_heading and the pose it finishes in. The
route itself comes from one model call: the question, the robot's whole 3D map as a table, and
the best-view frames with each mapped object tagged by its map id. Nothing here ranks
candidates or evaluates a spatial relation -- see `cat3_utils` for why that is deliberate.

What this file owns is everything between the reply and the wheels, and most of it exists
because of what the base autonomy does with a waypoint (waypointConverter.cpp):

  * within 5 m it REPLACES our waypoint, and not with the nearest traversable point: it
    minimises ||p - waypoint|| + 0.5 * ||p - vehicle|| over ground at least 0.75 m from any
    obstacle, a compromise pulled halfway back toward the robot. Aimed at a point inside the
    furniture it names -- which is what an object centre, or a midpoint between two, usually is
    -- that lands well short: 1.73 m short, measured. So we snap onto traversable ground
    ourselves before publishing (`cat3_utils.snap_to_traversable`), and because that weight is
    below 1 the converter then passes our point through untouched;
  * it latches arrival against that adjusted point and publishes the residual to our original
    on /way_point_reached, which is a better arrival test than our own distance;
  * every republish resets that latch, so republishing is required while driving and must stop
    once we have arrived, or the robot creeps off the pose the goal is scored on.
"""
from __future__ import annotations

import json
import math
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from std_msgs.msg import Bool, Float32, String

from captioner.paths import secure_path
from captioner.vlm_backends import constants as vlm_constants
from captioner.vlm_backends.constants import (SILHOUETTE_POLL_S, SILHOUETTE_WAIT_S,
                                              is_silhouette, view_dir)
from captioner.ros_utils import wait_for_subscriber
from captioner.prompts import get_route_plan_prompt
from smart_vlm.cat3_utils import (
    ARRIVAL_RADIUS_M,
    MAX_SNAP_M,
    MAX_TRIES,
    OBSTACLE_CLEARANCE_M,
    SETTLE_RADIUS_M,
    SETTLE_S,
    TRY_DURATION_S,
    Waypoint,
    fallback_route,
    heuristic_targets,
    map_table,
    parse_route,
    plan_payload,
    route_summary,
    snap_to_traversable,
    usable_objects,
)
from smart_vlm.mission_clock import MissionBudget, MissionClock, budget_for
from smart_vlm.question import QuestionType
from smart_vlm.traversable_area import TraversableArea
from smart_vlm.reasoner_common import (
    ReasonerNode,
    read_obj_map,
    save_manifest,
    spin_reasoner,
)

ROUTE_SYSTEM = get_route_plan_prompt()

#: Publish rate while driving. Fast enough that the converter's arrival latch is refreshed
#: promptly, slow enough that it still gets to run its traversability adjustment between
#: messages -- that adjustment is reset by every republish.
DRIVE_HZ = 2.0

#: The terrain analysis reading we are allowed at test time. The challenge withdrew the
#: traversable area this year, so this 20 m cloud is the only view of the floor we get.
#: Both terrain clouds, folded into one grid. The 20 m one is the reach; the 5 m one is the
#: quality. terrain_analysis runs `scanVoxelSize` 0.05 against ext's 0.10 and, crucially,
#: `noDecayDis = 1.75` -- its near field does NOT decay, while ext runs `noDecayDis = 0.0` and
#: is a rolling four-second window. So the near cloud is the strongest evidence we ever get for
#: floor the robot actually drove past, and it was going unused. Last-write-wins in
#: TraversableArea.add already prefers whichever arrived most recently, which is the right rule
#: when the two disagree: the closer reading is built from denser support.
TERRAIN_TOPICS = ("/terrain_map_ext", "/terrain_map")

#: How much one message from each source is worth when cells vote on being floor
#: (`TraversableArea.add`). /terrain_map is the better witness over the ground the two share:
#: `scanVoxelSize` 0.05 against ext's 0.10, and `noDecayDis = 1.75`, so its near field is a
#: real map rather than ext's four-second rolling window. Weighting it higher means ground the
#: robot actually drove past is not talked out of being floor by grazing returns from 15 m.
TERRAIN_WEIGHTS = {"/terrain_map": 2, "/terrain_map_ext": 1}

#: Readings further than this from the robot are discarded. /terrain_map_ext is 20 m wide by
#: construction, so anything past it is bad data rather than distant floor -- and bad floor is
#: worse than none, because the snap will aim at it. See TraversableArea.add.
TERRAIN_MAX_RANGE_M = 21.0

#: Splits that cloud into floor and obstacle. This is the base autonomy's own
#: `obstacleHeightThre`, so our idea of an obstacle matches the converter's.
OBSTACLE_HEIGHT_M = 0.05

#: Resolution of the accumulated floor grid, and so the quantisation floor on how precisely the
#: snap can aim (half a cell diagonal, ~0.07 m). It matches the source rather than beating it:
#: terrain analysis voxelises the scan at `scanVoxelSize` = 0.1, so a finer grid would mark a
#: lattice with unmarked holes between the points instead of a filled region -- no more accurate,
#: and four times the transform cost.
TERRAIN_CELL_M = 0.10

#: Half-width of that grid at startup, in metres. It grows if a reading lands outside, so this
#: only decides how often that happens -- a 40 m square costs 0.64 MB and covers every scene.
TERRAIN_HALF_SPAN_M = 20.0

#: Terrain fold rate cap. 0 means take every cloud, which is the default and the point.
#:
#: This was 1.0, on the reasoning that "the cloud is 20 m wide and the robot moves under 1 m/s,
#: so consecutive clouds overlap almost entirely". That holds for a robot driving in a straight
#: line across open floor. It does not hold for one turning: each cloud is a different line of
#: sight, ext decays everything after 4 s, and a dropped cloud is floor that is never seen
#: again. Measured on hotel_room_1 Q04 the robot ended 2.89 m from its own target with the
#: floor there still UNKNOWN, which is what a 2.82 m snap was reaching around.
#:
#: Scattering costs 0.11 ms per 10k points and touches only this message's cells, so there is
#: nothing to throttle away from. Kept as a knob purely so the claim stays testable.
TERRAIN_HZ = 0.0

#: How far the robot must travel before a leg re-chooses where to aim. The floor near a distant
#: waypoint usually does not exist when the leg starts -- terrain analysis runs `decayTime = 4.0`
#: with `noDecayDis = 0.0`, so /terrain_map_ext only ever holds the last few seconds of line of
#: sight -- and a snap taken then has nothing to snap to. Driving a metre closer is what makes
#: the ground visible, so that is when we look again. At cruise this fires under 1 Hz.
RESNAP_MOVE_M = 1.0



class InstructionReasoner(ReasonerNode):
    QUESTION_TYPE = QuestionType.INSTRUCTION_FOLLOWING
    STATUS_TOPIC = "/instruction_reasoner/status"
    ADHOC_PREFIX = "instr_reasoner"
    # True, or `self.backend` is never built and there is no model to plan the route with.
    ANSWER_BACKEND = True

    def __init__(self):
        super().__init__("instruction_reasoner", extra_params={
            # Stop driving here, leaving room to settle at the goal and be torn down.
            "route_reserve_s": 45.0,
            "question_budget_s": 600.0,
            # One try publishes the waypoint for this long; three tries, then the route
            # moves on. Worst case per waypoint is max_tries * try_duration_s + settle_s.
            "try_duration_s": TRY_DURATION_S,
            "max_tries": MAX_TRIES,
            "settle_radius_m": SETTLE_RADIUS_M,
            "settle_s": SETTLE_S,
            # Snapping a waypoint onto traversable ground before publishing it. The clearance
            # is the converter's own, so a point we pick is a point it accepts unchanged.
            "arrival_m": ARRIVAL_RADIUS_M,
            "snap_clearance_m": OBSTACLE_CLEARANCE_M,
            "traversable_cell_m": TERRAIN_CELL_M,
            "traversable_half_span_m": TERRAIN_HALF_SPAN_M,
            "max_snap_m": MAX_SNAP_M,
            # map_node writes obj_map.json on its own cycle; the supervisor allows 8 s of grace
            # after explore_done, so waiting only reasoner_common's default 5 s can miss a map
            # that was about to land.
            "map_wait_s": 15.0,
            "max_views": 3,
        })
        self.route_reserve_s = float(self.get_parameter("route_reserve_s").value)
        self.try_duration_s = float(self.get_parameter("try_duration_s").value)
        self.max_tries = max(1, int(self.get_parameter("max_tries").value))
        self.settle_radius_m = float(self.get_parameter("settle_radius_m").value)
        self.settle_s = float(self.get_parameter("settle_s").value)
        self.arrival_m = float(self.get_parameter("arrival_m").value)
        self.snap_clearance_m = float(self.get_parameter("snap_clearance_m").value)
        self.traversable_cell_m = float(self.get_parameter("traversable_cell_m").value)
        self.traversable_half_span_m = float(self.get_parameter("traversable_half_span_m").value)
        self.max_snap_m = float(self.get_parameter("max_snap_m").value)
        self.map_wait_s = float(self.get_parameter("map_wait_s").value)
        self.max_views = max(1, int(self.get_parameter("max_views").value))

        # Same shape and the same t0 as the supervisor's, so "how long is left" means one thing
        # in both processes without either publishing it.
        self.clock = MissionClock(
            budget_for(3, MissionBudget(
                question_budget_s=float(self.get_parameter("question_budget_s").value))),
            time.monotonic())

        self._odom_xy: Optional[tuple[float, float]] = None
        self._reached_residual: Optional[float] = None
        # Built from the first terrain message onward -- which is well before the route exists,
        # because this node is launched with the stack and explores alongside it. By the time a
        # waypoint needs snapping, the floor the robot drove past is already in here.
        self._area = TraversableArea(cell_m=self.traversable_cell_m,
                                     half_span_m=self.traversable_half_span_m)
        self._terrain_at = 0.0

        self.create_subscription(
            Odometry, "/state_estimation", self._on_odom, 10, callback_group=self._cb)
        # The converter's own arrival verdict, carrying the residual distance to the waypoint we
        # actually asked for. More reliable than our odometry check, because it knows where it
        # re-targeted us to.
        self.create_subscription(
            Float32, "/way_point_reached", self._on_reached, 10, callback_group=self._cb)
        # The floor, so a waypoint can be put somewhere the robot can stand before it is
        # published rather than after -- see `snap_to_traversable`. On its OWN callback group:
        # folding a cloud in costs tens of milliseconds, and `self._cb` is mutually exclusive,
        # so sharing it would stall the odometry the drive loop steers by.
        self._terrain_cb = MutuallyExclusiveCallbackGroup()
        for topic in TERRAIN_TOPICS:
            self.create_subscription(
                PointCloud2, topic,
                partial(self._on_terrain, weight=TERRAIN_WEIGHTS.get(topic, 1)),
                1, callback_group=self._terrain_cb)
        self.pub_waypoint = self.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self.pub_start = self.create_publisher(Bool, "/start_exploration", self._qos)
        self.get_logger().info(
            f"instruction_reasoner ready (backend={self.backend.name}, "
            f"extract={self.extract_backend.name}, "
            f"route reserve {self.route_reserve_s:.0f}s)")

    # ---- inputs ----------------------------------------------------------

    def _heuristic_targets(self, question: str) -> list[str]:
        return heuristic_targets(question)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        with self._lock:
            self._odom_xy = (float(p.x), float(p.y))

    def _on_reached(self, msg: Float32) -> None:
        with self._lock:
            self._reached_residual = float(msg.data)

    def _here(self) -> Optional[tuple[float, float]]:
        with self._lock:
            return self._odom_xy

    def _on_terrain(self, msg: PointCloud2, weight: int = 1) -> None:
        """Fold one terrain cloud into the traversable area.

        Terrain analysis reports height above the local ground plane as `intensity`, and the
        base autonomy calls anything above `obstacleHeightThre` an obstacle. Splitting on the
        same number means our idea of floor is the same as the stack's.

        The cloud is 20 m wide and follows the robot, so it has to be accumulated: by the time
        the route is driven, the interesting floor is wherever the robot explored, not wherever
        it happens to be standing.
        """
        now = time.monotonic()
        if TERRAIN_HZ > 0.0 and now - self._terrain_at < 1.0 / TERRAIN_HZ:
            return
        self._terrain_at = now

        try:
            points = read_points_numpy(msg, field_names=("x", "y", "intensity"), skip_nans=True)
        except Exception as exc:                      # a malformed cloud must not stop the drive
            self.get_logger().warn(f"terrain cloud unreadable: {exc}")
            return
        if not len(points):
            return

        # Scattering into the grid is sub-millisecond and touches only the cells this message
        # covers, so unlike the accumulate-and-deduplicate it replaced there is nothing here
        # worth doing outside the lock.
        here = self._here()
        with self._lock:
            self._area.add(points[:, :2], points[:, 2] >= OBSTACLE_HEIGHT_M,
                           origin=here, max_range_m=TERRAIN_MAX_RANGE_M, weight=weight)

    def _snap(self, wp: Waypoint) -> tuple[tuple[float, float], float]:
        """Where to actually publish for `wp`, and how far that is from what the model asked.

        The model's coordinate is never rewritten -- it is what the route is scored against and
        what `instruction_plan.json` records. This is only where the robot is aimed.
        """
        with self._lock:
            target = snap_to_traversable(
                (wp.x, wp.y), self._area,
                clearance_m=self.snap_clearance_m,
                max_snap_m=self.max_snap_m)
        return target, math.dist((wp.x, wp.y), target)

    # ---- answering -------------------------------------------------------

    def _begin_answer(self, explore_msg: String, snap: dict) -> None:
        self.get_logger().info(f"explore_done ({explore_msg.data or 'ok'}) — planning a route")
        threading.Thread(target=self._run_answer, args=(snap,), daemon=True).start()

    def _run_answer(self, snap: dict) -> None:
        try:
            self._hold_exploration()
            run_dir = secure_path(snap["crop_dir"]) if snap.get("crop_dir") else None
            raw_map = read_obj_map(run_dir, self.get_logger().error, wait_s=self.map_wait_s)
            objects = usable_objects(raw_map)
            if not objects:
                raise RuntimeError("the 3D map holds no usable objects")

            route, payload = self._plan(snap, run_dir, objects)
            if run_dir is not None:
                self._record(run_dir, payload)
            if not route:
                raise RuntimeError("no drivable route")

            self.get_logger().info(f"route: {route_summary(route)}")
            # Before driving, not after: this latches the supervisor's `answered` flag, which is
            # the only thing suppressing its T-30 fallback waypoint. That fallback picks an
            # object naively and would pull the robot off the pose the goal is scored on. The
            # eval harness knows this means "I have a plan", not "I have arrived" -- it ends the
            # question on the execute -> idle transition instead.
            self._publish_status("answered")
            self._publish_status("execute")
            self._drive(route, run_dir)
        except Exception as exc:  # noqa: BLE001 — one bad question must not kill the node
            self.get_logger().error(f"execute failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")
            return
        self._reset_to_idle("idle")

    def _views(self, run_dir: Path) -> list[Path]:
        """The images this run sends the model, named by vqa.yaml's `view_source_instruction_following`.

        Nothing is drawn here. sam_node already writes every overlay we want -- the silhouette
        copies carry mask outlines and `label [map id]` captions keyed to obj_map.json, in the
        right coordinates for whichever geometry they belong to. Re-deriving them from the
        manifest raced sam_node's finalize pass and produced boxes from a manifest written
        after they were drawn.

        Category 3 wants `full_silhouette`: a route is planned across a whole room, and an ROI
        crop shows one corner of it. Falls back to the cropped equivalent when
        `save_full_views` is off, so a mis-set pair of configs costs resolution, not the call.
        """
        manifest_path = run_dir / "manifest.json"
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                selected = (json.load(handle) or {}).get("selected") or []
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"no best-view manifest ({exc}) — sending the table alone")
            return []

        source = vlm_constants.VIEW_SOURCE_INSTRUCTION_FOLLOWING
        subdir = view_dir(source)
        deadline = time.monotonic() + (SILHOUETTE_WAIT_S if is_silhouette(source) else 0.0)
        out: list[Path] = []
        for entry in selected[:self.max_views]:
            name = entry.get("file")
            if not name:
                continue
            plain = secure_path(run_dir / name)
            wanted = secure_path(run_dir / subdir / name) if subdir else plain
            # A silhouette is written by the finalize pass that /pipeline/explore_done also
            # starts, so arriving a few hundred ms early is normal rather than a fault.
            while is_silhouette(source) and not wanted.is_file() and time.monotonic() < deadline:
                time.sleep(SILHOUETTE_POLL_S)
            if wanted.is_file():
                out.append(wanted)
            elif plain.is_file():
                self.get_logger().warn(
                    f"{name}: no {subdir or 'crop'} copy — sending the plain crop instead")
                out.append(plain)
        self.get_logger().info(f"views ({source}): {[p.name for p in out]}")
        return out

    def _plan(self, snap: dict, run_dir: Optional[Path],
              objects: dict) -> tuple[list[Waypoint], dict]:
        """One model call, then bookkeeping. Falls back rather than losing the question."""
        from captioner.vlm_backends.schemas import RoutePlan

        question = snap.get("question") or ""
        here = self._here()
        table = map_table(objects, here)
        images = self._views(run_dir) if run_dir else []

        user = (f"Command: {question}\n\n"
                f"Objects the robot mapped:\n{table}\n\n"
                f"The images show these objects outlined and tagged with the same ids.\n"
                "Reply with the route: the places to drive to, in order, goal last.")
        try:
            reply = self.backend.ask(ROUTE_SYSTEM, user, images, RoutePlan)
            route, trace = parse_route(reply, objects, self.get_logger().warn,
                                       reach_m=self.settle_radius_m)
            source = self.backend.name
        except Exception as exc:  # noqa: BLE001 — a model failure must not lose the question
            self.get_logger().error(
                f"route call failed ({type(exc).__name__}: {exc}) — "
                "falling back to the prompt objects in order")
            reply, trace, source = None, [f"model call failed: {type(exc).__name__}: {exc}"], \
                "fallback"
            route = []
        if not route:
            route = fallback_route(objects, snap.get("prompts") or [])
            source = "fallback"
            trace = list(trace) + ["empty route — using the prompt objects in order"]

        return route, plan_payload(
            question, objects, route, table=table, images=[str(p) for p in images],
            reply=reply, trace=trace, source=source, robot_xy=here,
            view_source=vlm_constants.VIEW_SOURCE_INSTRUCTION_FOLLOWING)

    def _record(self, run_dir: Path, payload: dict) -> None:
        """Write the plan and its rationale BEFORE the robot moves.

        The harness SIGINTs the whole pipeline as soon as the question ends, so anything not on
        disk by then is gone. `answer_reason` is the key eval_orchestrator.answer_rationale
        reads, so the report carries the model's own account of the route.
        """
        try:
            with open(run_dir / "instruction_plan.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        except OSError as exc:
            self.get_logger().warn(f"could not write instruction_plan.json: {exc}")

        manifest_path = run_dir / "manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle) or {}
            except (OSError, json.JSONDecodeError):
                manifest = {}
        manifest["answer_reason"] = payload.get("summary")
        manifest["plan_source"] = payload.get("plan_source")
        manifest["route"] = payload.get("route")
        try:
            save_manifest(manifest_path, manifest)
        except OSError as exc:
            self.get_logger().warn(f"could not update manifest.json: {exc}")

    def _record_drive(self, run_dir: Path, log: list[dict]) -> None:
        """Append what the drive actually did to instruction_plan.json.

        The plan half of that file already showed the model reasoning well; what it could not
        show was whether the robot went there. Reading a score then meant rebuilding the
        comparison by hand from the report's trajectory. Per leg: reached or not, the closest
        the robot ever got, seconds spent, which stall rungs fired.
        """
        path = run_dir / "instruction_plan.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"could not re-read {path.name} to record the drive: {exc}")
            return
        payload["drive"] = log
        with self._lock:
            payload["traversable"] = self._area.counts()
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        except OSError as exc:
            self.get_logger().warn(f"could not record the drive: {exc}")
        self._record_traversable(run_dir)

    def _record_traversable(self, run_dir: Path) -> None:
        """Dump the floor grid beside the plan, so the snap can be looked at afterwards.

        `free_cells` alone cannot answer the question that matters. hotel_room_1 Q04 recorded
        4776 of them -- a healthy-looking number -- while the floor beside its target had never
        been observed at all, which is what made the snap reach 2.82 m. The difference between
        "blocked" and "never seen" is invisible in a count and obvious in a picture, and they
        want opposite fixes: one is furniture, the other is unexplored.

        Compressed uint8: a 40 m grid at 0.10 m is tens of kilobytes.
        """
        with self._lock:
            snap = self._area.snapshot()
        try:
            np.savez_compressed(run_dir / "traversable_area.npz", **snap)
        except OSError as exc:
            self.get_logger().warn(f"could not record the traversable area: {exc}")

    # ---- driving ---------------------------------------------------------

    def _hold_exploration(self) -> None:
        """Take /way_point_with_heading from TARE. Repeated: it emits after the first pause."""
        self.pub_start.publish(Bool(data=False))

    def _publish(self, x: float, y: float) -> None:
        # theta is ignored: the deployed waypoint_converter runs yawConfig -1, which holds the
        # current heading on arrival. The README says to neglect heading this year.
        self.pub_waypoint.publish(Pose2D(x=float(x), y=float(y), theta=0.0))

    def _drive(self, route: list[Waypoint], run_dir: Optional[Path]) -> None:
        wait_for_subscriber(self.pub_waypoint)
        deadline = self.clock.hard_deadline - self.route_reserve_s
        log: list[dict] = []
        for wp in route:
            if time.monotonic() >= deadline and wp.role != "goal":
                self.get_logger().warn(
                    "CAT3 ROUTE DEADLINE — skipping to the goal with what time is left")
                log.append({"role": wp.role, "why": wp.why, "skipped": "route deadline"})
                continue
            log.append(self._drive_to(wp))
        # Nothing republishes after this: TARE is held and we have stopped, so the pose the goal
        # is scored on is the pose the robot keeps.
        goal = next((leg for leg in reversed(log) if leg.get("role") == "goal"), None)
        if goal is not None:
            self.get_logger().info(
                f"CAT3 GOAL HELD final={goal.get('final_m')}m "
                f"closest={goal.get('closest_m')}m "
                f"reach={goal.get('reach_m')}m reached={goal.get('reached')}")
        if run_dir is not None:
            self._record_drive(run_dir, log)

    def _drive_to(self, wp: Waypoint) -> dict:
        """Publish the waypoint in tries until the robot is near enough, or the tries run out.

        The waypoint is snapped onto ground the robot can stand on, and re-snapped every
        `RESNAP_MOVE_M` of travel: the floor near a distant target is not in /terrain_map_ext
        when the leg starts, so the first snap often has no candidates and passes the model's
        raw point through. Driving toward it is what reveals the ground. A measured goal leg
        started 6 m out with nothing to snap to and sat 2.30 m short for 45.7 s; a metre of
        travel would have given it a target.

        Two arrival tests, because one cannot always be won. The distance to the MODEL's point
        is what the scorer grades, so it is tested first against `reach_m` -- but it is bounded
        below by `snap_m` plus the converter's 0.3 m stop deadband, and a leg whose snap moved
        1.4 m can never enter a 1.5 m circle however well it drives. So arriving within
        `arrival_m` of the point we actually published also ends the leg. That second test is
        gated on the snap having fired: on a passthrough the published point IS the model's
        point, the two tests collapse into one, and a wedged leg would report success.

        A try publishes for `try_duration_s`; on arrival it stops publishing, lets the robot
        settle for `settle_s`, and the leg is done. Otherwise the next try starts. After
        `max_tries` the route moves on -- and at the goal there is nothing to move on to, so
        the robot simply stops where it is and is scored on that pose.

        Republishing IS the retry: every message resets the waypoint converter's arrival latch
        and makes it re-pick a traversable point. That is why a try needs nothing cleverer.
        This replaced a nudge/retarget/unwedge ladder that a measured run showed doing nothing
        -- a wedged goal ran 13 recovery attempts over 149 s while its distance moved 6 cm.

        Returns what happened, for `instruction_plan.json`: a plan nobody can check against the
        drive is only half a record.
        """
        goal = (wp.x, wp.y)
        target, snap_m = goal, 0.0
        snapped_at: Optional[tuple[float, float]] = None
        snaps = 0
        started = time.monotonic()
        best, best_pose = math.inf, None
        period = 1.0 / DRIVE_HZ

        self.get_logger().info(
            f"waypoint {wp.role} ({wp.x:.2f}, {wp.y:.2f}) reach<={wp.reach_m:.2f}m "
            f"{self.max_tries} x {self.try_duration_s:.0f}s"
            f"{f' — {wp.why}' if wp.why else ''}")

        for attempt in range(1, self.max_tries + 1):
            try_end = time.monotonic() + self.try_duration_s
            while rclpy.ok() and time.monotonic() < try_end:
                here = self._here()

                # Re-aim before publishing, once the robot has travelled far enough that the
                # terrain around the waypoint may have come into view since the last look.
                if here is not None and (snapped_at is None
                                         or math.dist(here, snapped_at) >= RESNAP_MOVE_M):
                    snapped_at = here
                    fresh, fresh_m = self._snap(wp)
                    if snaps == 0 or fresh != target:
                        moved = (f" -> ({fresh[0]:.2f}, {fresh[1]:.2f}) snap {fresh_m:.2f}m"
                                 if fresh_m else " -> unchanged (no traversable reading)")
                        self.get_logger().info(f"aim {wp.role} ({wp.x:.2f}, {wp.y:.2f}){moved}")
                    target, snap_m = fresh, fresh_m
                    snaps += 1

                self._publish(*target)
                time.sleep(period)

                here = self._here()
                if here is None:
                    continue
                # `best` tracks the MODEL's waypoint, never the snapped one: that is the point
                # the scorer grades, and a snap that moved us 1 m has not brought us 1 m closer
                # to it.
                to_goal = math.dist(here, goal)
                if to_goal < best:
                    best, best_pose = to_goal, here

                arrived = None
                if to_goal <= wp.reach_m:
                    arrived = "goal"
                elif snap_m > 0.0 and math.dist(here, target) <= self.arrival_m:
                    arrived = "target"
                if arrived:
                    # Stop publishing and let the converter finish its own approach. Another
                    # message here would reset its arrival latch and restore cruise speed,
                    # creeping the robot off the pose the goal is scored on.
                    reason = (f"within {wp.reach_m:.2f}m of {wp.role}"
                              if arrived == "goal" else
                              f"at the point we aimed ({snap_m:.2f}m off {wp.role})")
                    self.get_logger().info(
                        f"{reason} at {to_goal:.2f}m on try {attempt}/{self.max_tries} "
                        f"— settling {self.settle_s:.0f}s")
                    best, best_pose = self._settle(goal, best, best_pose)
                    return self._leg(wp, True, best, best_pose,
                                     time.monotonic() - started, attempt, target, snap_m,
                                     snaps, arrived)
            self.get_logger().warn(
                f"try {attempt}/{self.max_tries} for {wp.role} ({wp.x:.2f}, {wp.y:.2f}) "
                f"ended at {best:.2f}m")

        self.get_logger().warn(
            f"{wp.role} ({wp.x:.2f}, {wp.y:.2f}) skipped after {self.max_tries} tries "
            f"(closest {best:.2f}m)")
        return self._leg(wp, False, best, best_pose, time.monotonic() - started,
                         self.max_tries, target, snap_m, snaps, None)

    def _settle(self, target: tuple[float, float], best: float,
                best_pose: Optional[tuple[float, float]]
                ) -> tuple[float, Optional[tuple[float, float]]]:
        """Hold still, publishing nothing, so the converter can complete its approach.

        It keeps watching while it waits. An earlier version slept blind and reported the
        distance from before the wait, which understated how close the robot got by a median
        0.43 m across a full sweep -- one leg recorded 1.85 m having ended 0.21 m out. Anything
        tuned against that number was tuned against a fiction.
        """
        deadline = time.monotonic() + max(0.0, self.settle_s)
        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(0.25)
            here = self._here()
            if here is None:
                continue
            distance = math.dist(here, target)
            if distance < best:
                best, best_pose = distance, here
        return best, best_pose

    def _leg(self, wp: Waypoint, reached: bool, best: float,
             best_pose: Optional[tuple[float, float]], elapsed: float,
             tries: int, published: tuple[float, float], snap_m: float,
             snaps: int, arrived_at: Optional[str]) -> dict:
        here = self._here()
        return {
            "role": wp.role, "why": wp.why,
            "waypoint": [round(wp.x, 3), round(wp.y, 3)],
            # Where the robot was last aimed. Equal to `waypoint` when the model's point was
            # already traversable, or when no terrain had been seen near it.
            "published": [round(v, 3) for v in published],
            "snap_m": round(snap_m, 3),
            # How many times the leg re-chose where to aim. Without it there is no telling a
            # re-snap that fired and found nothing from one that never triggered.
            "snaps": snaps,
            # Floor cells known when the leg ended. A `snap_m` of 0 with this near zero means
            # terrain never arrived; with it large, it means none arrived near THAT waypoint.
            "free_cells": self._area.counts()["free"],
            "reach_m": round(wp.reach_m, 2),
            "reached": reached,
            # Which test ended the leg: "goal" (inside the scored circle) or "target" (at the
            # point we published, the best ground available). None when the tries ran out.
            "arrived_at": arrived_at,
            "tries": tries,
            "closest_m": None if best == math.inf else round(best, 3),
            "closest_pose": None if best_pose is None else [round(v, 3) for v in best_pose],
            "final_pose": None if here is None else [round(v, 3) for v in here],
            # What the goal is scored on: where the robot ENDED, not the best it ever managed.
            "final_m": None if here is None else round(math.dist(here, (wp.x, wp.y)), 3),
            # The same pose against where we aimed it -- how well the drive tracked, separated
            # from how far the snap had to move. `final_m` conflates the two.
            "final_target_m": None if here is None else round(math.dist(here, published), 3),
            "seconds": round(elapsed, 1),
        }


def main(args=None):
    rclpy.init(args=args)
    spin_reasoner(InstructionReasoner())


if __name__ == "__main__":
    main()
