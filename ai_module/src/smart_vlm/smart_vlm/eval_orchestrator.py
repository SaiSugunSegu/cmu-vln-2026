#!/usr/bin/env python3
"""End-to-end evaluation: every scene, every question, clean slate each time.

The challenge relaunches the whole system for each language command so nothing carries
over between questions (README "Evaluation"). This harness does the same: per question it
spawns the pipeline as its own process group, drives it over topics, records the
result, then SIGINTs the group. `scene_source` decides WHICH launch that is, rather than
passing a flag into one: `smart_vlm.launch` is the submission artifact and owns no scene
source, so bag runs go through the eval-only `eval_bag.launch` wrapper instead. A sim run
spawns the submission launch verbatim; it cannot switch Unity scenes itself, so
`scripts/eval/run_sim_sweep.py` drives it one scene at a time. Model loading therefore lands inside the measured budget,
exactly as it will on the real evaluation — `time_taken_s` in the report is honest.

Per question:

  1. publish /gt_target_objects            (latched, so the reasoner sees it whenever it starts)
  2. spawn  eval_bag.launch bag:=<scene>    (scene_source:=bag, the default)
            smart_vlm.launch                 (scene_source:=sim — the submission launch
                                              verbatim, fed by a sim someone else started)
  3. wait   /pipeline/ready                 models loaded
  4. publish /challenge_question @ 1 Hz     (the eval node's own cadence)
  5. wait   /pipeline/armed                 SAM holds this question's prompts; the scene
                                            is released — the bag plays, or TARE drives
  6. wait   the answer topic for the category (see ANSWER_TOPIC)
  7. record, tear the group down, wait for the ROS graph to drain, next

`category:=2` drives the object-reference questions instead: same pipeline, same gates, but
the answer arrives as a Marker on /selected_object_marker and is graded on overlap rather
than equality — twice the axis-aligned 3D IoU against the answer object's box, which is the
challenge's own formula (`scripts/eval/score.py`). The row keeps the marker's own id as
`predicted_object_id`, so a wrong answer can be read as the wrong object rather than merely
a low score, and `correct` means the pick landed on the answer object at all.

`category:=3` is instruction following, and it is the one that does not wait for a message.
The challenge grades "the actual trajectory followed by the robot" (README.md, "Question
Types and Initial Scoring"), so this node records /state_estimation for the whole question --
exploration included, because the challenge times and scores both together -- and scores the
path against the instruction's ordered constraints with scripts/eval/score.py, out of 6. The
question ends when the reasoner's route ends, not when it announces one; see
EvalOrchestratorNode._cat3_done for why those are different moments.

Waiting on /pipeline/armed rather than /sam3/status is what makes the non-GT path work:
SAM's status is `awaiting_prompts` from boot, so it says nothing about whether this
question's targets ever reached it.

  ros2 run smart_vlm eval_orchestrator --ros-args -p scene:=all -p question_limit:=0

Where inference runs is `vlm_backend` in config/vqa.yaml (imported as VLM_BACKEND). The
report records that value so a local sweep and a cloud sweep stay comparable.

Which image of a best-view crop the reasoner is shown (crop | silhouette, default
silhouette) is set once in `config/vqa.yaml`'s `view_source`, not per-invocation here —
see captioner/vlm_backends/constants.py. `eval-cat1`, `eval-cat2`, `cache-cat1`,
`cache-cat2` and the sim sweeps all read the same value that way.

Crops are saved under `<crops root>/<run_id>/<scene>/<question id>-<question>` and every
row records the `best_view_dir` they went to, which turns the report into a cache index:
`cat1_bench` (category 1) and `cat2_bench` (category 2) replay the answering step over
those saved crops in minutes instead of hours. Build one with `crops_only:=true`, which runs
target extraction and the whole perception half but skips the answering call.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import rclpy
import yaml
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from captioner.ros_utils import shutdown_guard
from captioner.vlm_backends.constants import VIEW_SOURCE, VLM_BACKEND
from smart_vlm.cat3_utils import synthetic_trajectory
from smart_vlm.report_utils import iou3d, previous_results, summarise, write_report

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

#: The sensors a question cannot survive without. Odometry is deliberately NOT here: on a
#: measured 15-scene sweep the failure mode was odometry flowing at 500 samples while
#: /camera/image and /registered_scan delivered nothing at all, so odometry is exactly the
#: signal that does not notice.
SENSOR_TOPICS = ("/camera/image", "/registered_scan")

#: How long every sensor may be silent before a question is abandoned. The scan is 5 Hz and
#: the camera 10 Hz, so half a minute of nothing is not a hiccup, it is a dead sim.
SENSOR_SILENCE_S = 30.0

#: How often the liveness test runs while a question is in flight.
SENSOR_PROBE_S = 30.0

#: run_orchestration returns this when a scene lost its sensors mid-question, so the sweep can
#: tell "restart the sim and try this scene again" from "the run failed on its own merits".
EXIT_SENSORS_LOST = 3


class SensorsLost(RuntimeError):
    """The sim stopped delivering camera or lidar while a question was running.

    Its own type because the response is different in kind: every other failure scores the
    question, and this one must NOT -- a question that never received a frame measures the
    infrastructure, and scoring it 0 understates the system. `run_sim_sweep` restarts the sim
    and re-runs the scene; a second failure skips it, writing no rows.
    """


# What "answered" means, per category. Category 3 has no answer MESSAGE — the challenge
# grades "the actual trajectory followed by the robot" — so this names the topic the response
# is delivered on, which is what a timeout should complain about.
ANSWER_TOPIC = {1: "/numerical_response", 2: "/selected_object_marker",
                3: "/way_point_with_heading"}

# Category 3, from README.md "Question Types and Initial Scoring".
#: Odometry sampling period. Matches scripts/eval/qa_recorder.py so a trajectory recorded
#: here and one recorded there are the same kind of object.
TRAJ_SAMPLE_S = 0.5
#: "Each question has a total time limit of 10 minutes for exploration and question answering
#: combined." Exceeding it "incurs a penalty"; the penalty is not specified, so this only
#: flags the row rather than adjusting the score.
QUESTION_TIME_LIMIT_S = 600.0
#: Full marks. Used only to report `correct`, the way HIT_IOU is used for category 2.
CAT3_MAX_SCORE = 6.0
#: Fallback completion, from qa_recorder._done: the planner sent at least one waypoint, has
#: sent nothing since, and the robot has stopped. Only reached when the reasoner never
#: returns to idle — a crash mid-route, say — so it is a backstop, not the normal path.
CAT3_IDLE_AFTER_WAYPOINT_S = 30.0
CAT3_IDLE_WINDOW_S = 15.0
CAT3_IDLE_MOVED_M = 0.2

# An overlap this small is not a hit, whatever it scores: two same-class objects standing
# side by side clip each other's boxes. Used only to split "picked the right object" from
# "scored a little" in the report — the score itself is the raw overlap, ungated.
HIT_IOU = 0.25


def log(message: str, *, err: bool = False) -> None:
    print(f"[orchestrator] {message}", file=sys.stderr if err else sys.stdout, flush=True)


class EvalOrchestratorNode(Node):
    """Long-lived driver. Only this node survives across questions."""

    def __init__(self) -> None:
        super().__init__("eval_orchestrator")

        self.scene = str(self._param("scene", "all"))
        # 1 = counting (an Int32), 2 = object reference (a Marker), 3 = instruction
        # following (the driven path). One sweep runs one category: they are graded on
        # different scales -- /1, /2, /6 -- and a report mixing them would have to explain
        # what its accuracy meant.
        self.category = int(self._param("category", 1))
        if self.category not in ANSWER_TOPIC:
            raise ValueError(f"category must be 1, 2 or 3, got {self.category}")
        # Where the six allowed topics come from. bag: replay a recording from
        # bags_dir, which is why a bag sweep can walk every scene by itself. sim: the
        # live Unity sim in the OTHER container, which this process cannot start,
        # stop or re-scene — scripts/eval/run_sim_sweep.py owns that loop and calls
        # this one scene at a time.
        self.scene_source = str(self._param("scene_source", "bag")).strip().lower()
        if self.scene_source not in ("bag", "sim"):
            raise ValueError(
                f"scene_source must be 'bag' or 'sim', got {self.scene_source!r}")
        self.report_file = str(self._param("report_file", "/data/runs/challenge_report.json"))
        # Top level of the crop layout, so one sweep's crops sit together and the next
        # sweep cannot overwrite them question by question. Defaults to the report's own
        # name because the report is the index into these directories: keeping the two
        # named alike is what stops a cache from being paired with the wrong crops.
        self.run_id = str(self._param("run_id", "")).strip() or Path(self.report_file).stem
        self.question_limit = int(self._param("question_limit", 0))
        # Pin one benchmark id (e.g. Q01). When set, question_limit is ignored.
        self.question_id = str(self._param("question_id", "")).strip()
        self.target_source = str(self._param("target_source", "gt"))
        # A real double, like every other numeric param here. ROS matches override types
        # against the declared one, so pass it with a decimal point: speed:=2.0, not 2.
        self.speed = float(self._param("speed", 1.0))
        self.sam_config = str(self._param("sam_config", "sam3_mecanum_sim.yaml"))
        self.resolved_backend = VLM_BACKEND
        # Build the best-view cache: run the expensive perception half and skip the
        # counting call. Accuracy in the resulting report is meaningless by design —
        # it exists to record where each question's crops went.
        self.crops_only = bool(self._param("crops_only", False))
        # How the object-reference reasoner chooses (category 2 only); see cat2_utils.
        self.cat2_mode = str(self._param("cat2_mode", "hybrid"))
        # Skip questions the report beside these crops already covers. A sweep of every
        # scene runs for hours, so an interruption in hour five must not mean starting
        # over. Off by default: a scored run has to answer every question itself.
        self.resume = bool(self._param("resume", False))
        # Extend an existing report instead of replacing it. One sweep of the live sim is
        # several orchestrator runs — the Unity scene can only be changed from the host,
        # so run_sim_sweep.py invokes us once per scene — and without this each run would
        # overwrite the last, leaving only the final scene. With it, every scene lands in
        # ONE report whose summary (and per_scene block) covers the whole sweep, exactly
        # as a single-process bag sweep produces.
        self.append = bool(self._param("append", False))
        self.benchmark_dir = Path(str(self._param("benchmark_dir", "/data/benchmark")))
        self.bags_dir = Path(str(self._param("bags_dir", "/data/bags")))
        # Phase budgets. Generous, because a first run may download weights; a phase that
        # blows its budget fails one question rather than stalling the sweep.
        self.ready_timeout_s = float(self._param("ready_timeout_s", 420.0))
        self.armed_timeout_s = float(self._param("armed_timeout_s", 300.0))
        self.answer_timeout_s = float(self._param("answer_timeout_s", 600.0))
        self.teardown_timeout_s = float(self._param("teardown_timeout_s", 45.0))

        self.pipeline_ready = False
        self.pipeline_armed = False
        self.armed_prompts: Optional[list] = None
        self.predicted: Optional[int] = None
        self.marker: Optional[dict] = None
        self.sam_status: Optional[str] = None
        self.best_view_dir: Optional[str] = None
        # -- category 3 ------------------------------------------------------
        #: (t, x, y) from /state_estimation at TRAJ_SAMPLE_S. THE graded artifact: the
        #: challenge scores "the actual trajectory followed by the robot", and it times
        #: exploration and answering together, so this runs from the first sample of the
        #: question rather than from the moment the planner starts driving.
        self.trajectory: list[tuple] = []
        self._last_traj_t = -TRAJ_SAMPLE_S
        self._t0: Optional[float] = None
        #: (t, x, y, theta) from /way_point_with_heading. Not scored -- it is the mechanism,
        #: not the response -- but it separates "the planner emitted nothing" from "the
        #: planner emitted a bad route", which look identical in the trajectory alone.
        self.waypoints: list[tuple] = []
        #: Latest /instruction_reasoner/status, and whether `execute` has been seen. The
        #: reasoner publishes `idle` once at startup too, so the completion test is a
        #: TRANSITION, not a value.
        self.instr_status: Optional[str] = None
        self._instr_executing = False
        #: Elapsed seconds at the `execute` transition — the exploration/route boundary.
        self.execute_at: Optional[float] = None

        #: monotonic time the last message arrived on each sensor topic. None until one
        #: does, which is why the liveness test only arms once the scene is playing.
        self._sensor_seen: dict[str, Optional[float]] = {t: None for t in SENSOR_TOPICS}

        self.pub_question = self.create_publisher(String, "/challenge_question", 10)
        self.pub_gt_targets = self.create_publisher(String, "/gt_target_objects", LATCHED)
        self.create_subscription(String, "/pipeline/ready", self._on_ready, LATCHED)
        self.create_subscription(String, "/pipeline/armed", self._on_armed, LATCHED)
        self.create_subscription(String, "/sam3/status", self._on_sam_status, LATCHED)
        self.create_subscription(Int32, "/numerical_response", self._on_answer, 10)
        self.create_subscription(Marker, "/selected_object_marker", self._on_marker, 10)
        # Recorded per question so the crops this run wrote stay addressable afterwards:
        # the directory name carries a random run id, so nothing else can reconstruct
        # which one belongs to which question. cat1_bench replays from exactly this.
        self.create_subscription(String, "/sam3/best_view_dir", self._on_best_view_dir,
                                 LATCHED)
        # Best effort and depth 1: this is a heartbeat, not data. Reliable QoS would not
        # match the sim's sensor publishers, and a deep queue would hold frames we discard.
        heartbeat = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, "/camera/image",
                                 lambda _m: self._on_sensor("/camera/image"), heartbeat)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 lambda _m: self._on_sensor("/registered_scan"), heartbeat)
        if self.category == 3:
            self.create_subscription(Odometry, "/state_estimation", self._on_odom, 50)
            self.create_subscription(Pose2D, "/way_point_with_heading", self._on_waypoint, 10)
            self.create_subscription(String, "/instruction_reasoner/status",
                                     self._on_instr_status, LATCHED)

    def _param(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    # -- callbacks ---------------------------------------------------------

    def _on_ready(self, _msg: String) -> None:
        self.pipeline_ready = True

    def _on_armed(self, msg: String) -> None:
        self.pipeline_armed = True
        try:
            self.armed_prompts = json.loads(msg.data).get("prompts")
        except (json.JSONDecodeError, AttributeError):
            self.armed_prompts = None

    def _on_sam_status(self, msg: String) -> None:
        self.sam_status = (msg.data or "").strip()

    def _on_answer(self, msg: Int32) -> None:
        self.predicted = int(msg.data)

    def _on_marker(self, msg: Marker) -> None:
        """A category-2 answer, flattened to what grading needs.

        Read exactly as the challenge scorer reads it — pose position and scale, orientation
        ignored — so a marker that scores here scores there. `id` is the reasoner's own map
        track id, kept so a miss can be attributed to the wrong object rather than to a bad
        box, and `ns` distinguishes a real answer from the `crops_only` placeholder.
        """
        self.marker = {
            "ns": msg.ns,
            "id": int(msg.id),
            "center": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            "size": [msg.scale.x, msg.scale.y, msg.scale.z],
            "text": msg.text,
            "placeholder": msg.ns == "placeholder" or msg.id < 0,
        }

    def _on_best_view_dir(self, msg: String) -> None:
        self.best_view_dir = (msg.data or "").strip() or None

    # -- category 3 --------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        if elapsed - self._last_traj_t < TRAJ_SAMPLE_S:
            return
        self._last_traj_t = elapsed
        p = msg.pose.pose.position
        self.trajectory.append((round(elapsed, 2), round(p.x, 3), round(p.y, 3)))

    def _on_waypoint(self, msg: Pose2D) -> None:
        t = 0.0 if self._t0 is None else time.monotonic() - self._t0
        self.waypoints.append((round(t, 2), round(msg.x, 3), round(msg.y, 3),
                               round(msg.theta, 3)))

    def _on_instr_status(self, msg: String) -> None:
        self.instr_status = (msg.data or "").strip()
        if self.instr_status == "execute" and not self._instr_executing:
            self._instr_executing = True
            # When the reasoner took /way_point_with_heading from TARE, in the trajectory's
            # own timebase. Everything before it is exploration -- the scorer counts that
            # ground (a constraint satisfied while exploring still scores), but a reader must
            # not mistake it for the route: one run drove 48 m exploring and 4.7 m following
            # the plan, and a single line through both says nothing about either.
            self.execute_at = (None if self._t0 is None
                               else round(time.monotonic() - self._t0, 2))

    def _cat3_done(self) -> bool:
        """The route is finished — the robot has stopped where it meant to stop.

        NOT the reasoner's own "answered": instruction_reasoner._execute publishes that
        BEFORE its driving loop, so it means "I have a plan", not "I have arrived". Since
        score_instruction takes the goal from trajectory[-1], ending the question there would
        grade the pre-execution pose and miss the goal on every question.

        `idle` is published at reasoner startup as well, so the test is the transition out of
        `execute`, not the value on its own.
        """
        if self._instr_executing and self.instr_status in ("idle", "error"):
            return True
        # Backstop for a reasoner that never returns to idle (crashed mid-route). Same rule
        # as qa_recorder._done: it sent a waypoint, has sent none since, and has stopped.
        if not self.waypoints or not self.trajectory:
            return False
        now = self.trajectory[-1][0]
        if now - self.waypoints[-1][0] < CAT3_IDLE_AFTER_WAYPOINT_S:
            return False
        recent = [p for p in self.trajectory if p[0] > now - CAT3_IDLE_WINDOW_S]
        if len(recent) < 2:
            return False
        return math.dist(recent[0][1:], recent[-1][1:]) < CAT3_IDLE_MOVED_M

    def answered(self) -> bool:
        if self.category == 3:
            return self._cat3_done()
        return self.marker is not None if self.category == 2 else self.predicted is not None

    def reset_for_question(self) -> None:
        self.pipeline_ready = False
        self.pipeline_armed = False
        self.armed_prompts = None
        self.predicted = None
        self.marker = None
        self.sam_status = None
        self.best_view_dir = None
        self.trajectory = []
        self.waypoints = []
        self._last_traj_t = -TRAJ_SAMPLE_S
        self._t0 = None
        self.instr_status = None
        self._instr_executing = False
        self.execute_at = None

    def _on_sensor(self, topic: str) -> None:
        self._sensor_seen[topic] = time.monotonic()

    def reset_sensor_watch(self) -> None:
        """Forget what arrived before this question. Called when the scene starts playing."""
        self._sensor_seen = {t: None for t in SENSOR_TOPICS}

    def sensors_silent(self, max_age_s: float, since: float) -> list[str]:
        """Which sensors have delivered nothing recently. Empty means the sim is feeding us.

        A topic that has NEVER produced is judged against `since` -- the moment the scene
        started playing -- so a sim that comes up dead is caught as well as one that dies
        partway. Without that, "never seen" would look the same as "seen a moment ago".
        """
        now = time.monotonic()
        stale = []
        for topic, seen in self._sensor_seen.items():
            age = now - (seen if seen is not None else since)
            if age >= max_age_s:
                stale.append(topic)
        return stale

    # -- helpers -----------------------------------------------------------

    def spin_until(
        self,
        predicate: Callable[[], bool],
        timeout_s: float,
        proc: Optional[subprocess.Popen] = None,
    ) -> bool:
        """Pump callbacks until `predicate` holds, the deadline passes, or `proc` dies.

        The `proc` check matters: without it a launch that crashes on startup burns the
        whole phase budget before anyone notices.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            if proc is not None and proc.poll() is not None:
                log(f"pipeline exited early with code {proc.returncode}", err=True)
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return predicate()

    def graph_is_clear(self) -> bool:
        """True once the previous pipeline's publishers have left the ROS graph."""
        return all(self.count_publishers(t) == 0
                   for t in ("/sam3/status", "/pipeline/ready", "/smart_vlm/status"))


# -- process control --------------------------------------------------------

def spawn_pipeline(node: EvalOrchestratorNode, scene: str, run_id: str) -> subprocess.Popen:
    # The scene source is a different LAUNCH FILE, not a flag: smart_vlm.launch is the
    # submission artifact and knows nothing about bags, so offline runs go through the
    # eval-only wrapper that adds one. A sim run therefore spawns exactly what the
    # organizers will.
    if node.scene_source == "bag":
        cmd = [
            "ros2", "launch", "smart_vlm", "eval_bag.launch",
            f"bag:={scene}",
            f"speed:={node.speed}",
        ]
    else:
        cmd = ["ros2", "launch", "smart_vlm", "smart_vlm.launch"]
    cmd += [
        f"sam_config:={node.sam_config}",
        f"run_id:={run_id}",
        f"crops_only:={'true' if node.crops_only else 'false'}",
        f"cat2_mode:={node.cat2_mode}",
    ]
    log(f"launching: {' '.join(cmd)}")
    # Own process group, so one killpg reclaims every node it started — including the
    # GPU-resident detector. Nothing in the pipeline detaches itself from this group.
    return subprocess.Popen(cmd, start_new_session=True)


def teardown(node: EvalOrchestratorNode, proc: Optional[subprocess.Popen]) -> None:
    """SIGINT the group, escalate if it hangs, then wait for the graph to drain.

    The graph wait is the part the clean-slate guarantee actually rests on: a process
    that has exited may still be discoverable for a moment, and a stale publisher would
    make the next question's readiness gate fire instantly against a dead pipeline.
    """
    if proc is not None and proc.poll() is None:
        log("tearing down pipeline ...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=node.teardown_timeout_s)
        except subprocess.TimeoutExpired:
            log("pipeline ignored SIGINT — sending SIGKILL", err=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=15.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        except ProcessLookupError:
            pass

    if not node.spin_until(node.graph_is_clear, timeout_s=30.0):
        log("ROS graph still shows pipeline publishers after teardown", err=True)


# -- benchmark discovery ----------------------------------------------------

def has_bag(scene_dir: Path) -> bool:
    """A directory counts only if there is something in it ros2 bag play can open.

    Several scenes ship a metadata-only stub directory with no recording. Treating the
    directory itself as proof of a bag let those scenes through, and each of their
    questions then burned the full warmup timeout waiting for /camera/image.
    """
    return any(scene_dir.glob("*.mcap")) or (scene_dir / "metadata.yaml").is_file()


def available_scenes(node: EvalOrchestratorNode) -> set[str]:
    """Scenes we can actually replay: already on disk, or fetchable via scenes.yaml."""
    scenes = {p.name for p in node.bags_dir.iterdir() if p.is_dir() and has_bag(p)} \
        if node.bags_dir.is_dir() else set()
    manifest = node.bags_dir / "scenes.yaml"
    if manifest.is_file():
        try:
            with open(manifest, "r", encoding="utf-8") as handle:
                scenes |= set((yaml.safe_load(handle) or {}).get("scenes") or {})
        except (OSError, yaml.YAMLError) as err:
            log(f"could not read {manifest}: {err}", err=True)
    return scenes


def discover_questions(node: EvalOrchestratorNode) -> Iterator[tuple[str, dict]]:
    """Flatten the benchmark into one (scene, question) stream, in a stable order."""
    if not node.benchmark_dir.is_dir():
        raise FileNotFoundError(f"benchmark directory {node.benchmark_dir} not found")

    folder = f"category_{node.category}"
    if node.scene and node.scene != "all":
        scenes = [node.scene]
    elif node.scene_source == "sim":
        # Unity holds one scene per launch and this process cannot swap it — the mesh
        # lives in the other container's image. Sweeping "all" here would score every
        # scene's questions against whichever scene happens to be loaded, and score
        # them plausibly, so fail loudly instead. scripts/eval/run_sim_sweep.py is the
        # thing that can walk scenes: it re-meshes and restarts the sim, then calls
        # this driver once per scene.
        raise ValueError(
            "scene_source:=sim needs an explicit scene (Unity holds one per launch); "
            "use scripts/eval/run_sim_sweep.py, or `just eval-cat1-sim`, to sweep")
    else:
        scenes = sorted(p.name for p in node.benchmark_dir.iterdir()
                        if (p / folder).is_dir())

    # Only bags can be missing from disk; the sim supplies whatever is loaded.
    playable = available_scenes(node) if node.scene_source == "bag" else set(scenes)
    for scene in scenes:
        qa_file = (node.benchmark_dir / scene / folder
                   / f"{scene}_category{node.category}_qa.json")
        if not qa_file.is_file():
            log(f"skipping {scene}: no QA file at {qa_file}", err=True)
            continue
        if scene not in playable:
            log(f"skipping {scene}: no bag on disk and none listed in scenes.yaml", err=True)
            continue
        with open(qa_file, "r", encoding="utf-8") as handle:
            questions = json.load(handle).get("questions") or []
        if node.question_id:
            questions = [q for q in questions if str(q.get("id", "")) == node.question_id]
            if not questions:
                log(f"skipping {scene}: no question {node.question_id} in {qa_file}",
                    err=True)
                continue
        elif node.question_limit > 0:
            questions = questions[:node.question_limit]
        for entry in questions:
            yield scene, entry


def answer_rationale(best_view_dir: Optional[str], category: int) -> tuple:
    """The model's own explanation of this answer, from the run directory's manifest.

    Read from disk rather than a topic because the answer topics carry only the answer —
    an Int32 or a Marker — and there is nowhere in them to put a sentence. Both reasoners
    already write the manifest BEFORE publishing (see the "Record first, publish second"
    comments), so by the time a row is built, after teardown, the file is there.

    The two categories name the field differently because they explain different
    decisions: cat1 explains a count, cat2 explains which candidate object it picked.
    Both surface as `reason` in the report so one column means one thing.

    Anything missing yields (None, None): a question that failed before the reasoner ran
    has no manifest, and losing the row over an absent explanation would be a bad trade.
    """
    if not best_view_dir:
        return None, None
    manifest_path = Path(best_view_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        log(f"could not read {manifest_path} for the rationale: {err}", err=True)
        return None, None
    if not isinstance(manifest, dict):
        return None, None
    key = "selection_reason" if category == 2 else "answer_reason"
    views = manifest.get("context_views")
    return manifest.get(key), (len(views) if isinstance(views, list) else None)


def coverage_summary(best_view_dir: Optional[str]) -> Optional[dict]:
    """What target_explorer achieved, from the run directory's target_coverage.json.

    Read here for the same reason `answer_rationale` is: the score alone cannot tell you
    whether a zero was selection, perception, or the robot simply never walking round the
    object. `labels_unseen` and `goals_unpublished` separate those three, and without this
    the only trace is launch stdout, which nothing captures.

    Missing yields None — a question that failed before arming has no coverage report, and
    losing the row over that would be a bad trade.
    """
    if not best_view_dir:
        return None
    path = Path(best_view_dir) / "target_coverage.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        log(f"could not read {path} for the coverage summary: {err}", err=True)
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    return summary if isinstance(summary, dict) else None


# -- one question -----------------------------------------------------------

_SCORE_MODULE = None


def load_scorer():
    """`scripts/eval/score.py`, imported by path — it is a script, not an installed module.

    Same trick as verify_category3.py::load_scorer, and for the same reason: score_instruction
    must be ONE implementation. `just cat3-verify` gates the ground truth at 6/6 with it, so a
    copy here could drift and we would be grading against a scorer the GT was never checked
    with. scripts/ is bind-mounted read-only into this container (see docker/compose_gpu.yml).
    """
    global _SCORE_MODULE
    if _SCORE_MODULE is None:
        # Container layout first, then a repo checkout, so the same code works if this is
        # ever driven from the host. Mirrors score_map3d.py's two-layout resolution.
        repo = Path(__file__).resolve().parents[4]
        for candidate in (Path("/home/docker/scripts/eval/score.py"),
                          repo / "scripts" / "eval" / "score.py"):
            path = candidate
            if path.is_file():
                spec = importlib.util.spec_from_file_location("cat3_score", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _SCORE_MODULE = module
                break
        else:
            raise FileNotFoundError(
                "scripts/eval/score.py not found — category 3 cannot be graded. It is "
                "bind-mounted at /home/docker/scripts; check docker/compose_gpu.yml")
    return _SCORE_MODULE


def grade_instruction(node: EvalOrchestratorNode, entry: dict, elapsed: float) -> dict:
    """Category 3: score the path the robot actually drove.

    README.md, "Question Types and Initial Scoring": *"The score will be calculated based on
    the actual trajectory followed by the robot based on whether it follows the path
    constraints in the command and in the correct order."* So the trajectory is the graded
    artifact and the waypoint list is only the mechanism — recorded, but not scored.

    The whole run is scored, exploration included, because the challenge times and scores
    exploration and answering together.
    """
    gt = entry.get("gt") or {}
    goal = (gt.get("goal") or {})
    row: dict = {
        "gt": goal.get("phrase") or goal.get("label"),
        "gt_label": goal.get("label"),
        "gt_object_ids": goal.get("object_ids"),
        "predicted": None,
        "score": 0.0,
        "note": "no trajectory",
        "correct": False,
        "trajectory": node.trajectory,
        "waypoints": node.waypoints,
        # Where the trajectory stops being exploration and starts being the driven route.
        "drive_started_s": node.execute_at,
        "n_constraints": len(gt.get("pass_near") or []) + (1 if goal else 0),
        # "Exceeding the time limit for a certain question incurs a penalty on the initial
        # score" — the penalty is unspecified, so flag the row and leave the score alone.
        "over_time": elapsed > QUESTION_TIME_LIMIT_S,
    }
    if not node.trajectory:
        return row
    score, note = load_scorer().score_instruction({"trajectory": node.trajectory}, gt)
    row.update({
        # Where the robot actually finished — what the goal constraint is scored on.
        "predicted": [node.trajectory[-1][1], node.trajectory[-1][2]],
        "score": round(float(score), 4),
        "note": note,
        "correct": float(score) >= CAT3_MAX_SCORE,
    })
    row.update(plan_quality(node, gt, float(score)))
    row.update(drive_phase(node, gt, float(score)))
    return row


def drive_phase(node: EvalOrchestratorNode, gt: dict, score: float) -> dict:
    """What the ROUTE-FOLLOWING part of the run scored, with exploration excluded.

    `score` grades the whole trajectory, and a pass constraint is credited anywhere on it --
    so 150 s of wandering can satisfy constraints the planned route never goes near. That is
    not hypothetical: one measured question scored 3.0 against a plan worth 1.5, having missed
    its goal by 5.75 m. Its extra point came from the exploration path, not from driving.

    So `drive_loss` on its own conflates "the drive failed" with "exploration got lucky", and
    it is the number used to decide whether driving is worth working on. Grading the same
    trajectory from `execute_at` onward separates them. Both are reported; neither replaces
    the official score, which is and stays the whole trajectory.

    Returns {} when there is no execute transition to cut at -- a question that never planned.
    """
    if node.execute_at is None or not node.trajectory:
        return {}
    driven = [p for p in node.trajectory if p[0] >= node.execute_at]
    if len(driven) < 2:
        return {}
    drive_score, _ = load_scorer().score_instruction({"trajectory": driven}, gt)
    return {
        "drive_phase_score": round(float(drive_score), 4),
        # What the whole-run score owes to the exploration path. Large means the number
        # above is flattered by wandering rather than earned by the route.
        "explore_credit": round(float(score) - float(drive_score), 4),
        "explore_m": round(_path_length(node.trajectory, end=node.execute_at), 1),
        "drive_m": round(_path_length(driven), 1),
    }


def _path_length(traj: list, end: Optional[float] = None) -> float:
    pts = [p for p in traj if end is None or p[0] <= end]
    return sum(math.dist(pts[i][1:3], pts[i + 1][1:3]) for i in range(len(pts) - 1))


def plan_quality(node: EvalOrchestratorNode, gt: dict, score: float) -> dict:
    """What the ROUTE was worth, separately from whether the robot managed to drive it.

    A bare score cannot tell a plan that was wrong from a plan that was right and undriven —
    both read `goal missed`. Scoring a perfect drive of the model's own waypoints splits them:
    across one 21-question sweep the plans were worth 80/114 while the runs scored 59/114, so
    grounding and driving were costing about the same and neither was the obvious culprit.

    Uses the same `load_scorer()` as the real grade, so the ceiling and the score are always
    computed by one implementation. Returns {} when the run recorded no plan — an older report,
    or a question that died before the reasoner wrote one.
    """
    run_dir = node.best_view_dir
    if not run_dir:
        return {}
    try:
        with open(Path(run_dir) / "instruction_plan.json", "r", encoding="utf-8") as handle:
            route = (json.load(handle) or {}).get("route") or []
    except (OSError, json.JSONDecodeError):
        return {}
    if not route:
        return {}

    start = (node.trajectory[0][1], node.trajectory[0][2]) if node.trajectory else None
    plan_score, _ = load_scorer().score_instruction(
        {"trajectory": synthetic_trajectory(start, route)}, gt)

    out = {"plan_score": round(float(plan_score), 4),
           "drive_loss": round(float(plan_score) - score, 4)}
    # How far the planned STOP is from the thing the goal constraint is measured against --
    # the single most diagnostic number for grounding, since the goal is the one constraint
    # scored on where the robot ends.
    goal_wp = next((w for w in reversed(route) if w.get("role") == "goal"), None)
    centre = ((gt.get("goal") or {}).get("center") or [])[:2]
    if goal_wp is not None and len(centre) == 2:
        out["goal_error_m"] = round(
            math.dist((goal_wp.get("x", 0.0), goal_wp.get("y", 0.0)), centre), 3)
    return out


def grade(node: EvalOrchestratorNode, entry: dict, elapsed: float = 0.0) -> dict:
    """The answer-shaped half of a result row: what came back, and what it was worth.

    Category 1 is equality against an integer. Category 2 is twice the axis-aligned 3D IoU
    between the Marker and the answer object's box, exactly as `scripts/eval/score.py`
    computes it, and is reported alongside the ids of both objects: an overlap of zero means
    something quite different when the right object was chosen and the box was wrong.
    Category 3 is the driven path against the instruction's constraints — see
    grade_instruction.
    """
    if node.category == 3:
        return grade_instruction(node, entry, elapsed)
    if node.category == 1:
        gt = int(entry["answer"])
        return {"gt": gt, "predicted": node.predicted, "correct": node.predicted == gt}

    answer = entry["answer"]
    marker = node.marker
    row: dict = {
        "gt": answer["object_id"],
        "gt_label": answer.get("label"),
        "predicted": None,
        "predicted_object_id": None,
        "iou": 0.0,
        "score": 0.0,
        "correct": False,
        "marker": marker,
    }
    if marker is None or marker["placeholder"]:
        return row

    overlap = iou3d(marker["center"], marker["size"], answer["center"], answer["size"])
    row.update({
        "predicted": marker["text"] or marker["id"],
        # The marker id is the reasoner's map track id, which is NOT comparable with the
        # benchmark's object id — they index different maps. It identifies which candidate
        # was chosen, for reading a manifest back; only the overlap grades it.
        "predicted_object_id": marker["id"],
        "iou": round(overlap, 4),
        "score": round(2.0 * overlap, 4),
        "correct": overlap >= HIT_IOU,
    })
    return row


def run_question(node: EvalOrchestratorNode, scene: str, entry: dict) -> dict:
    qid = entry["id"]
    question = entry["question"]
    if node.category == 1:
        gt_answer = int(entry["answer"])
    elif node.category == 3:
        # No single answer object: the goal constraint is what the run is judged to have
        # reached, so name it. `entry["answer"]` does not exist for this category.
        goal = (entry.get("gt") or {}).get("goal") or {}
        gt_answer = goal.get("phrase") or goal.get("label") or "?"
    else:
        gt_answer = f"{entry['answer'].get('label')}#{entry['answer']['object_id']}"

    log(f"=== {scene} {qid} ===")
    log(f"Q: {question}   (GT: {gt_answer})")

    node.reset_for_question()
    proc: Optional[subprocess.Popen] = None
    error: Optional[str] = None
    t_start = time.monotonic()

    try:
        # 1. Latched, so the reasoner picks it up whenever it finishes starting. An
        #    empty list means "extract the targets with Qwen instead".
        targets = entry.get("target_objects", []) if node.target_source == "gt" else []
        node.pub_gt_targets.publish(String(data=json.dumps(targets)))

        # <sweep>/<scene>/<question id>-<question>, so a directory says what it holds
        # without opening its manifest. sam_mapper scrubs each level; the question is
        # capped only to stay clear of the filesystem's per-component length limit.
        proc = spawn_pipeline(node, scene, f"{node.run_id}/{scene}/{qid}-{question[:80]}")

        # 2. Models up.
        log("waiting for /pipeline/ready (SAM weights ~60s) ...")
        if not node.spin_until(lambda: node.pipeline_ready, node.ready_timeout_s, proc):
            raise TimeoutError(f"pipeline not ready within {node.ready_timeout_s:.0f}s")

        # 3. Ask at 1 Hz, the cadence the real challenge_evaluation_node uses, until SAM
        #    confirms it holds this question's prompts. Only then does the bag play.
        log("ready — publishing the question at 1 Hz ...")
        msg = String(data=question)
        last_pub = 0.0

        def ask_and_check(flag: Callable[[], bool]) -> Callable[[], bool]:
            def predicate() -> bool:
                nonlocal last_pub
                now = time.monotonic()
                if now - last_pub >= 1.0:
                    node.pub_question.publish(msg)
                    last_pub = now
                return flag()
            return predicate

        if not node.spin_until(ask_and_check(lambda: node.pipeline_armed),
                               node.armed_timeout_s, proc):
            raise TimeoutError(
                f"SAM was never armed within {node.armed_timeout_s:.0f}s "
                f"(target extraction or /sam3/set_prompts failed)")
        log(f"SAM armed with {node.armed_prompts} — the scene is now playing")
        node.reset_sensor_watch()
        playing_since = time.monotonic()

        # 4. The answer. Keep republishing: the reasoner may still be starting up.
        #    Watched, because the sim can pass the sweep's pre-flight probe and then stop
        #    feeding: on a measured 15-scene sweep four scenes ran their whole window on
        #    odometry alone, with zero camera frames and zero clouds, and scored 0 for it.
        last_probe = [time.monotonic()]

        def answered_with_live_sensors() -> bool:
            now = time.monotonic()
            if now - last_probe[0] >= SENSOR_PROBE_S:
                last_probe[0] = now
                silent = node.sensors_silent(SENSOR_SILENCE_S, playing_since)
                if silent:
                    raise SensorsLost(
                        f"no data on {silent} for {SENSOR_SILENCE_S:.0f}s "
                        f"while {ANSWER_TOPIC[node.category]} was awaited")
            return node.answered()

        if not node.spin_until(ask_and_check(answered_with_live_sensors),
                               node.answer_timeout_s, proc):
            raise TimeoutError(f"no answer on {ANSWER_TOPIC[node.category]} within "
                               f"{node.answer_timeout_s:.0f}s")

    except SensorsLost:
        # NOT scored, and not caught here: the scene has to be abandoned and retried, and a
        # row for it would record an infrastructure failure as a system failure.
        teardown(node, proc)
        raise
    except Exception as exc:  # noqa: BLE001 — one bad question must not end the sweep
        error = f"{type(exc).__name__}: {exc}"
        log(error, err=True)
    finally:
        teardown(node, proc)

    elapsed = time.monotonic() - t_start
    outcome = grade(node, entry, elapsed)
    reason, n_views = answer_rationale(node.best_view_dir, node.category)
    scale = {2: "/2", 3: "/6"}.get(node.category)
    log(f"result: predicted={outcome['predicted']} gt={gt_answer} "
        f"correct={outcome['correct']}"
        + (f" score={outcome['score']:.2f}{scale}" if scale else "")
        + (f" ({outcome['note']})" if node.category == 3 else "")
        + f" time={elapsed:.1f}s"
        + (" OVER TIME LIMIT" if outcome.get("over_time") else ""))

    return {
        "scene": scene,
        "id": qid,
        "question": question,
        "category": node.category,
        **outcome,
        "time_taken_s": round(elapsed, 2),
        # Which world answered. A bag row and a sim row are not comparable — same
        # question, different trajectory and different exploration — so the report has
        # to say which it was rather than leaving it to the filename.
        "scene_source": node.scene_source,
        "target_source": node.target_source,
        "vlm_backend": node.resolved_backend,
        "view_source": VIEW_SOURCE,
        "prompts": node.armed_prompts,
        "best_view_dir": node.best_view_dir,
        # Why the model answered as it did, and over how many views. A bare number tells
        # you a row is wrong; these tell you whether perception or reasoning was at fault.
        "reason": reason,
        "n_context_views": n_views,
        # Did exploration deliver what the answer path needed? `labels_unseen` non-empty
        # means SAM never found a target at all; `goals_unpublished` means it was found but
        # never circled enough to earn a centroid, so it is absent from obj_map.json.
        "target_coverage": coverage_summary(node.best_view_dir),
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def cached_rows(report_path: Path, category: int = 1) -> dict[tuple[str, str], dict]:
    """Rows of an earlier report we can adopt instead of replaying the scene.

    A row only counts if its crops are still on disk and its question text still matches
    the benchmark, so an edited QA file or a half-written directory re-runs rather than
    quietly answering from the wrong images.

    Category 2 additionally needs the 3D map: the crops alone cannot answer a reference
    question, so a row whose `obj_map.json` never landed is a hole in the cache and has to be
    replayed rather than counted as done. Category 3's equivalent is the trajectory: it IS
    the answer, so a row without one recorded nothing worth keeping.
    """
    if not report_path.is_file():
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle).get("results") or []
    except (OSError, json.JSONDecodeError) as err:
        log(f"could not read {report_path} to resume: {err}", err=True)
        return {}

    keep: dict[tuple[str, str], dict] = {}
    for row in rows:
        crop_dir = row.get("best_view_dir")
        if not crop_dir or row.get("error") or not row.get("question"):
            continue
        if not any(Path(crop_dir).glob("best_rank*.png")):
            continue
        if category == 2 and not (Path(crop_dir) / "obj_map.json").is_file():
            continue
        if category == 3 and not row.get("trajectory"):
            continue
        keep[(row.get("scene"), row.get("id"))] = row
    return keep


# -- driver -----------------------------------------------------------------

def run_orchestration(node: EvalOrchestratorNode) -> int:
    report_path = Path(node.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    questions = list(discover_questions(node))
    if not questions:
        log("no questions to run — check scene/benchmark_dir", err=True)
        return 1
    scenes = sorted({s for s, _ in questions})
    log(f"category {node.category} from the {node.scene_source}: {len(questions)} "
        f"question(s) across {len(scenes)} scene(s): {scenes}")

    # So a cache report cannot be mistaken for a scored one: with crops_only every
    # prediction is a placeholder, and the accuracy beside it means nothing.
    extra = {"crops_only": node.crops_only, "category": node.category,
             "scene_source": node.scene_source, "view_source": VIEW_SOURCE}
    if node.category == 2:
        extra["cat2_mode"] = node.cat2_mode

    cached = cached_rows(report_path, node.category) if node.resume else {}
    if cached:
        log(f"resume: {len(cached)} question(s) already have crops — keeping their rows")

    results: list[dict] = previous_results(report_path) if node.append else []
    if results:
        log(f"append: carrying {len(results)} row(s) already in {report_path}")
    interrupted = False
    sensors_lost = False
    try:
        for scene, entry in questions:
            row = cached.get((scene, entry["id"]))
            if row and row["question"] == entry["question"]:
                log(f"=== {scene} {entry['id']} === already cached, skipping")
                results.append({**row, "resumed": True})
            else:
                results.append(run_question(node, scene, entry))
            write_report(report_path, results, extra)
    except SensorsLost as exc:
        # Abandon the SCENE, not just the question: the sim is what died, and every
        # remaining question here would run blind too. No row is written for any of them --
        # see SensorsLost. run_sim_sweep restarts the sim and re-runs the scene.
        sensors_lost = True
        log(f"sensors lost — abandoning this scene without scoring it: {exc}", err=True)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)

    if results:
        write_report(report_path, results, extra)
    summary = summarise(results)
    log(f"done: {summary['correct']}/{summary['total_run']} correct "
        f"(accuracy {summary['accuracy']}, errors {summary['errors']}"
        + (f", score {summary['total_score']}/{summary['max_score']}"
           if "total_score" in summary else "")
        + f") -> {report_path}")
    if sensors_lost:
        return EXIT_SENSORS_LOST
    return 130 if interrupted else 0


def main(args=None) -> None:
    ret = 1
    rclpy.init(args=args)
    node = EvalOrchestratorNode()
    try:
        ret = run_orchestration(node)
    except Exception as exc:  # noqa: BLE001 — report the real failure, not a NameError
        log(f"fatal: {type(exc).__name__}: {exc}", err=True)
    finally:
        with shutdown_guard():
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
