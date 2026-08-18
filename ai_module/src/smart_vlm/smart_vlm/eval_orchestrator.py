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

Waiting on /pipeline/armed rather than /sam3/status is what makes the non-GT path work:
SAM's status is `awaiting_prompts` from boot, so it says nothing about whether this
question's targets ever reached it.

  ros2 run smart_vlm eval_orchestrator --ros-args -p scene:=all -p question_limit:=0

`vlm_backend` picks where the reasoner runs inference (a hosted model, whichever
VLM_PROVIDER names, or the resident local Qwen). The report format is identical either
way, so a local sweep and a cloud sweep are directly comparable.

Crops are saved under `<crops root>/<run_id>/<scene>/<question id>-<question>` and every
row records the `best_view_dir` they went to, which turns the report into a cache index:
`cat1_bench` (category 1) and `cat2_bench` (category 2) replay the answering step over
those saved crops in minutes instead of hours. Build one with `crops_only:=true`, which runs
target extraction and the whole perception half but skips the answering call.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from captioner.ros_utils import shutdown_guard
from captioner.vlm_backends.constants import VLM_BACKEND
from smart_vlm.report_utils import iou3d, previous_results, summarise, write_report

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

# What "answered" means, per category.
ANSWER_TOPIC = {1: "/numerical_response", 2: "/selected_object_marker"}

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
        # 1 = counting (an Int32), 2 = object reference (a Marker). One sweep runs one
        # category: they are graded differently, and a report mixing the two would have to
        # explain what its accuracy meant.
        self.category = int(self._param("category", 1))
        if self.category not in ANSWER_TOPIC:
            raise ValueError(f"category must be 1 or 2, got {self.category}")
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
        self.target_source = str(self._param("target_source", "gt"))
        # A real double, like every other numeric param here. ROS matches override types
        # against the declared one, so pass it with a decimal point: speed:=2.0, not 2.
        self.speed = float(self._param("speed", 1.0))
        self.sam_config = str(self._param("sam_config", "sam3_mecanum_sim.yaml"))
        # cloud | local; "auto" leaves it on $VLM_BACKEND. Not "": rcl rejects an empty
        # parameter override and launch rejects a bare `name:=`, so the sentinel has to
        # be a real word to survive both hops. The resolved name is what decides whether
        # the pipeline has to wait for the local VQA server.
        self.vlm_backend = str(self._param("vlm_backend", "auto"))
        self.resolved_backend = (
            VLM_BACKEND if self.vlm_backend in ("", "auto") else self.vlm_backend)
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

    def answered(self) -> bool:
        return self.marker is not None if self.category == 2 else self.predicted is not None

    def reset_for_question(self) -> None:
        self.pipeline_ready = False
        self.pipeline_armed = False
        self.armed_prompts = None
        self.predicted = None
        self.marker = None
        self.sam_status = None
        self.best_view_dir = None

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
        f"vlm_backend:={node.vlm_backend}",
        f"run_id:={run_id}",
        f"crops_only:={'true' if node.crops_only else 'false'}",
        f"cat2_mode:={node.cat2_mode}",
        # The launch derives this from vlm_backend, but only we know what "auto"
        # resolved to against the environment — so pin it. true makes the pipeline start
        # its own Qwen server per question, which is why a local sweep needs no
        # `just vqa-up`; require_vqa_ready follows from it inside the launch.
        f"local_vqa:={'true' if node.resolved_backend == 'local' else 'false'}",
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
        if node.question_limit > 0:
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


# -- one question -----------------------------------------------------------

def grade(node: EvalOrchestratorNode, entry: dict) -> dict:
    """The answer-shaped half of a result row: what came back, and what it was worth.

    Category 1 is equality against an integer. Category 2 is twice the axis-aligned 3D IoU
    between the Marker and the answer object's box, exactly as `scripts/eval/score.py`
    computes it, and is reported alongside the ids of both objects: an overlap of zero means
    something quite different when the right object was chosen and the box was wrong.
    """
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
    gt_answer = (int(entry["answer"]) if node.category == 1
                 else f"{entry['answer'].get('label')}#{entry['answer']['object_id']}")

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

        # 4. The answer. Keep republishing: the reasoner may still be starting up.
        if not node.spin_until(ask_and_check(node.answered),
                               node.answer_timeout_s, proc):
            raise TimeoutError(f"no answer on {ANSWER_TOPIC[node.category]} within "
                               f"{node.answer_timeout_s:.0f}s")

    except Exception as exc:  # noqa: BLE001 — one bad question must not end the sweep
        error = f"{type(exc).__name__}: {exc}"
        log(error, err=True)
    finally:
        teardown(node, proc)

    elapsed = time.monotonic() - t_start
    outcome = grade(node, entry)
    reason, n_views = answer_rationale(node.best_view_dir, node.category)
    log(f"result: predicted={outcome['predicted']} gt={gt_answer} "
        f"correct={outcome['correct']}"
        + (f" score={outcome['score']:.2f}/2" if node.category == 2 else "")
        + f" time={elapsed:.1f}s")

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
        "prompts": node.armed_prompts,
        "best_view_dir": node.best_view_dir,
        # Why the model answered as it did, and over how many views. A bare number tells
        # you a row is wrong; these tell you whether perception or reasoning was at fault.
        "reason": reason,
        "n_context_views": n_views,
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
    replayed rather than counted as done.
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
             "scene_source": node.scene_source}
    if node.category == 2:
        extra["cat2_mode"] = node.cat2_mode

    cached = cached_rows(report_path, node.category) if node.resume else {}
    if cached:
        log(f"resume: {len(cached)} question(s) already have crops — keeping their rows")

    results: list[dict] = previous_results(report_path) if node.append else []
    if results:
        log(f"append: carrying {len(results)} row(s) already in {report_path}")
    interrupted = False
    try:
        for scene, entry in questions:
            row = cached.get((scene, entry["id"]))
            if row and row["question"] == entry["question"]:
                log(f"=== {scene} {entry['id']} === already cached, skipping")
                results.append({**row, "resumed": True})
            else:
                results.append(run_question(node, scene, entry))
            write_report(report_path, results, extra)
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
