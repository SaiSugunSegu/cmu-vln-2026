#!/usr/bin/env python3
"""Persistent cache builder for category-1/2 best-view crops (runs INSIDE iros2026_ai_module).

This IS `just cache-cat1` / `just cache-cat2`. The problem it replaces: driving
`eval_orchestrator --crops_only` directly (what those two recipes used to do, before this
script existed) relaunches the ENTIRE pipeline — SAM 3 weight load (~60 s), the whole ROS
graph, and a `speed:=0.1` (10x slowed) bag replay — for EVERY question, because that
mirrors how the real challenge (and the SCORED eval) must behave: a clean slate per
question, nothing carried over, honest per-question timing. See eval_orchestrator.py's
own docstring and `justfile`'s `eval-cat1`/`eval-cat2` comments (still relaunch-per-
question, since that IS the scored eval's requirement) — this script is cache-only.

Cache building has none of that requirement — `crops_only:=true` never answers, so there
is nothing for cross-question state to bias, and the two node modules already know how to
run this way: sam_node re-arms in place on every `/sam3/set_prompts` (see its
`_on_set_prompts`), and map_node explicitly drops and rebuilds its map on the same signal
("Nodes that live one question (eval_orchestrator relaunches everything) never hit this;
the persistent bench loop re-arms in place and always would." — map_node.py). This script
is that persistent bench loop, generalised from the existing manual 4-terminal
`run_cat1_bag_bench.py` flow to (a) drive either category, (b) launch SAM/VQA/map/reasoner
itself instead of needing them started by hand in separate terminals, and (c) write a
report in the exact `eval_orchestrator` {summary, results} shape, so `resume`,
`cat1_bench --cache`, and `cat2_bench --cache` all keep working unmodified.

Per question, only the CHEAP part is repeated: a bag replay + a SAM 3 session reset
(`/sam3/set_prompts`) + (category 2) a map rebuild. The weight loads happen exactly once
for the whole sweep, independent of GPU — this is a software/scheduling fix, not a
hardware one, and helps identically on an H200 or the deployment 4090.

--speed is a SEPARATE knob from the above and is NOT freed up by removing the relaunch:
sam_node keeps only the latest frame (`_take_frame`, "latest-frame-wins") and silently
drops anything that arrives before the worker consumes it, so a bag played faster than
SAM 3 can process just produces sparser, worse crops with no error. Measured live (its
own `frame N: ... | dropped D/I` log) on an idle H200 with a 3-object-prompt question,
the busiest scene (arabic_room, ~7.4 Hz native /camera/image) held zero steady-state
drops at speed 0.15 and accumulated a growing backlog at 0.3 — that bound comes from
SAM 3's per-frame cost against the bag's camera rate, not from anything this script
changes. --speed still defaults to 0.1 (eval_orchestrator's own value) for that reason;
raise it only after confirming `dropped` stays flat on your own GPU and scenes.

  just cache-cat1 [scene] [limit] [backend] [target_source]
  just cache-cat2 [scene] [limit] [backend] [target_source] [mode]

or directly:

  python3 scripts/eval/cache_bag_bench.py --category 1 --scene all \\
      --backend cloud --target-source vlm --cache /data/runs/views_cache.json --resume
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

# Reused verbatim from eval_orchestrator so "how many were correct", "is this question
# already cached", and "which scenes/questions exist" can never disagree between the slow
# (relaunch) and fast (persistent) cache builders. Safe to import: module import has no
# rclpy.init()/node side effects — discover_questions/available_scenes/grade only read
# plain attributes off whatever object they're given (SweepConfig below, not a real Node).
from smart_vlm.eval_orchestrator import ANSWER_TOPIC, cached_rows, discover_questions, grade
from smart_vlm.report_utils import summarise, write_report

from captioner.qwen_vqa_protocol import STATUS_TOPIC as VQA_STATUS_TOPIC
from captioner.vlm_backends.constants import VLM_BACKEND

_CONTAINER_ROOTS = ("/data", "/home/docker")


def _require_container_path(path: Path, flag: str) -> Path:
    text = str(path)
    if not any(text == root or text.startswith(root + "/") for root in _CONTAINER_ROOTS):
        raise SystemExit(
            f"{flag}={text} is not a container path. This script runs inside "
            f"iros2026_ai_module; use a path under {' or '.join(_CONTAINER_ROOTS)}."
        )
    return path


def log(message: str, *, err: bool = False) -> None:
    print(f"[cache-bench] {message}", file=sys.stderr if err else sys.stdout, flush=True)


@dataclass
class SweepConfig:
    """The subset of EvalOrchestratorNode's attributes that
    discover_questions/available_scenes/cached_rows/grade actually read — kept as a plain
    dataclass (no rclpy Node) so those functions can be reused without dragging in a
    second ROS node identity for what is otherwise a pure bookkeeping question."""
    category: int
    scene: str
    question_limit: int
    benchmark_dir: Path
    bags_dir: Path
    predicted: Optional[int] = None
    marker: Optional[dict] = None


# -- persistent process management -------------------------------------------------

@dataclass
class Managed:
    """One long-lived subprocess this script owns for the whole sweep."""
    name: str
    proc: subprocess.Popen


def spawn(name: str, cmd: list[str]) -> Managed:
    log(f"launching {name}: {' '.join(cmd)}")
    # Own process group per node, mirroring eval_orchestrator.spawn_pipeline, so a
    # teardown can killpg each one independently of the driver's own process group.
    proc = subprocess.Popen(cmd, start_new_session=True)
    return Managed(name, proc)


def teardown_all(managed: list[Managed], timeout_s: float = 20.0) -> None:
    for m in managed:
        if m.proc.poll() is not None:
            continue
        log(f"stopping {m.name} ...")
        try:
            os.killpg(os.getpgid(m.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + timeout_s
    for m in managed:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            m.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log(f"{m.name} ignored SIGINT — sending SIGKILL", err=True)
            try:
                os.killpg(os.getpgid(m.proc.pid), signal.SIGKILL)
                m.proc.wait(timeout=10.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass


def play_bag(scene: str, bags_dir: str, speed: float) -> None:
    """Single-pass bag play, blocking. No wait_for_armed: the driver below already
    gates on /sam3/status==ready before ever calling this, exactly as run_cat1_bag_bench
    does — this only replaces per-question model loads, not this proven ordering."""
    cmd = (
        "source /home/docker/ai_module/install/setup.bash && "
        "ros2 launch smart_vlm bag_replay.launch "
        f"scene:={shlex.quote(scene)} "
        f"bags_dir:={shlex.quote(bags_dir)} "
        f"speed:={float(speed)} loop:=false"
    )
    log(f"playing bag scene={scene} speed={speed}")
    proc = subprocess.run(["bash", "-lc", cmd], check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"bag play exited with code {proc.returncode}")


# -- the driver node ----------------------------------------------------------------

class CacheBenchDriver(Node):
    """Generalisation of run_cat1_bag_bench.py's Cat1BagDriver to either category.

    Lives for the WHOLE sweep (unlike eval_orchestrator's node, which lives one process
    per question only because the pipeline around it is relaunched). Each question only
    resets this node's own small bit of per-question state before publishing the next one.
    """

    def __init__(self, category: int):
        super().__init__("cache_bag_bench_driver")
        self.category = category

        self._ack: Optional[dict] = None
        self._ack_event = threading.Event()
        self._predicted: Optional[int] = None
        self._marker: Optional[dict] = None
        self._answer_event = threading.Event()
        self._crop_dir: Optional[str] = None
        self._tick_on = False
        self._question = ""
        self._sam_status: Optional[str] = None
        self._vqa_status: Optional[str] = None
        self._reasoner_status_seen = False

        latch = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.q_pub = self.create_publisher(String, "/challenge_question", 10)
        self.gt_pub = self.create_publisher(String, "/gt_target_objects", latch)
        self.done_pub = self.create_publisher(String, "/pipeline/explore_done", 10)

        self.create_subscription(String, "/sam3/prompts_ack", self._on_ack, 10)
        self.create_subscription(String, "/sam3/best_view_dir", self._on_dir, latch)
        self.create_subscription(String, "/sam3/status", self._on_sam_status, latch)
        self.create_subscription(String, VQA_STATUS_TOPIC, self._on_vqa_status, latch)
        reasoner_status_topic = ("/numerical_reasoner/status" if category == 1
                                 else "/object_reference/status")
        self.create_subscription(String, reasoner_status_topic, self._on_reasoner_status,
                                 latch)
        self.create_subscription(Int32, "/numerical_response", self._on_answer_cat1, 10)
        self.create_subscription(Marker, "/selected_object_marker", self._on_answer_cat2, 10)
        self.create_timer(1.0, self._tick)

    # -- properties consumed by eval_orchestrator.grade() ------------------
    @property
    def predicted(self):
        return self._predicted

    @property
    def marker(self):
        return self._marker

    @property
    def best_view_dir(self) -> Optional[str]:
        return self._crop_dir

    # -- callbacks -----------------------------------------------------------
    def _tick(self):
        if self._tick_on:
            self.q_pub.publish(String(data=self._question))

    def _on_sam_status(self, msg: String):
        self._sam_status = (msg.data or "").strip()

    def _on_vqa_status(self, msg: String):
        self._vqa_status = (msg.data or "").strip()

    def _on_reasoner_status(self, _msg: String):
        self._reasoner_status_seen = True

    def _on_ack(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if self._ack_event.is_set() or not payload.get("ok") or not self._tick_on:
            return
        self._ack = payload
        if payload.get("run_dir"):
            self._crop_dir = payload["run_dir"]
        self._ack_event.set()

    def _on_dir(self, msg: String):
        self._crop_dir = msg.data

    def _on_answer_cat1(self, msg: Int32):
        if self.category == 1 and not self._answer_event.is_set():
            self._predicted = int(msg.data)
            self._answer_event.set()

    def _on_answer_cat2(self, msg: Marker):
        if self.category != 2 or self._answer_event.is_set():
            return
        # Same shape as EvalOrchestratorNode._on_marker, so grade() reads it identically.
        self._marker = {
            "ns": msg.ns,
            "id": int(msg.id),
            "center": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            "size": [msg.scale.x, msg.scale.y, msg.scale.z],
            "text": msg.text,
            "placeholder": msg.ns == "placeholder" or msg.id < 0,
        }
        self._answer_event.set()

    # -- per-question lifecycle ----------------------------------------------
    def reset_for_question(self, question: str) -> None:
        self._question = question
        self._ack = None
        self._ack_event.clear()
        self._predicted = None
        self._marker = None
        self._answer_event.clear()

    def start_question(self) -> None:
        self._tick_on = True
        self.q_pub.publish(String(data=self._question))

    def stop_question(self) -> None:
        self._tick_on = False

    def spin_for(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)

    def _spin_until(self, predicate: Callable[[], bool], timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return predicate()

    def wait_sam_ready(self, timeout_s: float, *, context: str = "SAM") -> None:
        if not self._spin_until(lambda: self._sam_status == "ready", timeout_s):
            raise TimeoutError(
                f"/sam3/status not ready within {timeout_s:.0f}s "
                f"(last={self._sam_status!r}) — {context}")

    def wait_vqa_ready(self, timeout_s: float) -> None:
        if not self._spin_until(lambda: self._vqa_status == "ready", timeout_s):
            raise TimeoutError(
                f"/qwen_vqa/status not ready within {timeout_s:.0f}s "
                f"(last={self._vqa_status!r})")

    def wait_reasoner_up(self, timeout_s: float) -> None:
        if not self._spin_until(lambda: self._reasoner_status_seen, timeout_s):
            raise TimeoutError(f"reasoner never published its status within {timeout_s:.0f}s")

    def wait_ack(self, timeout_s: float) -> dict:
        if not self._spin_until(lambda: self._ack_event.is_set(), timeout_s):
            raise TimeoutError(f"no /sam3/prompts_ack within {timeout_s:.0f}s")
        return self._ack or {}

    def signal_explore_done(self, run_id: Optional[str]) -> None:
        self.done_pub.publish(String(data=json.dumps({
            "run_id": run_id, "scene_done": True,
        })))

    def wait_answer(self, timeout_s: float) -> None:
        if not self._spin_until(lambda: self._answer_event.is_set(), timeout_s):
            topic = ANSWER_TOPIC[self.category]
            raise TimeoutError(f"no answer on {topic} within {timeout_s:.0f}s")


# -- one question ---------------------------------------------------------------------

def run_question(driver: CacheBenchDriver, cfg: SweepConfig, scene: str, entry: dict,
                 args) -> dict:
    qid = entry["id"]
    question = entry["question"]
    log(f"=== {scene} {qid} === {question}")

    driver.reset_for_question(question)
    t_start = time.monotonic()
    error: Optional[str] = None
    ack: dict = {}
    try:
        targets = entry.get("target_objects", []) if args.target_source == "gt" else []
        driver.gt_pub.publish(String(data=json.dumps(targets)))

        driver.wait_sam_ready(args.sam_ready_timeout, context=f"{scene} {qid} (idle)")
        driver.start_question()
        ack = driver.wait_ack(args.ack_timeout)
        driver.stop_question()

        driver.wait_sam_ready(args.sam_ready_timeout,
                              context=f"{scene} {qid} (prompts applied)")
        if args.pre_bag_settle > 0:
            driver.spin_for(args.pre_bag_settle)

        play_bag(scene, args.bags_dir, args.speed)

        if args.post_bag_wait > 0:
            driver.spin_for(args.post_bag_wait)

        driver.signal_explore_done(ack.get("run_id"))
        driver.wait_answer(args.answer_timeout)
    except Exception as exc:  # noqa: BLE001 — one bad question must not end the sweep
        error = f"{type(exc).__name__}: {exc}"
        log(error, err=True)
    finally:
        driver.stop_question()

    elapsed = time.monotonic() - t_start
    cfg.predicted = driver.predicted
    cfg.marker = driver.marker
    outcome = grade(cfg, entry)
    best_view_dir = ack.get("run_dir") or driver.best_view_dir
    log(f"result: best_view_dir={best_view_dir} time={elapsed:.1f}s"
        + (f" error={error}" if error else ""))

    return {
        "scene": scene,
        "id": qid,
        "question": question,
        "category": cfg.category,
        **outcome,
        "time_taken_s": round(elapsed, 2),
        "target_source": args.target_source,
        "vlm_backend": args.resolved_backend,
        "prompts": ack.get("prompts"),
        "best_view_dir": best_view_dir,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# -- persistent pipeline (spawned once, kept up for the whole sweep) ------------------

def spawn_pipeline(args) -> list[Managed]:
    managed: list[Managed] = []
    if args.resolved_backend == "local":
        managed.append(spawn("qwen_vqa_server", [
            "ros2", "launch", "captioner", "vqa_server.launch",
            f"model:={args.vqa_model}", f"quantization:={args.vqa_quantization}",
        ]))
    managed.append(spawn("sam_node", [
        "ros2", "launch", "sam_mapper", "sam_node.launch",
        f"config:={args.sam_config}", "wait_for_prompts:=true",
    ]))
    if args.category == 2:
        managed.append(spawn("map_node", [
            "ros2", "launch", "sam_mapper", "map_node.launch",
            f"config:={args.sam_config}",
        ]))
    reasoner_exec = "numerical_reasoner" if args.category == 1 else "object_reference_reasoner"
    reasoner_cmd = [
        "ros2", "run", "smart_vlm", reasoner_exec, "--ros-args",
        "-p", "crops_only:=true",
        "-p", f"backend:={args.backend}",
        "-p", f"max_context_views:={args.max_context_views}",
        "-p", f"vqa_timeout_s:={args.vqa_timeout_s}",
    ]
    if args.category == 2:
        reasoner_cmd += ["-p", f"mode:={args.cat2_mode}"]
    managed.append(spawn(reasoner_exec, reasoner_cmd))
    return managed


def wait_pipeline_ready(driver: CacheBenchDriver, args) -> None:
    log("waiting for SAM 3 weights (~60s, longer on first download) ...")
    driver.wait_sam_ready(args.ready_timeout, context="startup")
    if args.resolved_backend == "local":
        log("waiting for the local Qwen VQA server ...")
        driver.wait_vqa_ready(args.ready_timeout)
    log("waiting for the reasoner to come up ...")
    driver.wait_reasoner_up(args.ready_timeout)
    log("pipeline ready")


# -- driver -----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", type=int, choices=(1, 2), required=True)
    ap.add_argument("--scene", default="all")
    ap.add_argument("--limit", type=int, default=0, dest="question_limit",
                    help="max questions PER SCENE (0 = all)")
    ap.add_argument("--target-source", choices=("gt", "vlm"), default="vlm")
    ap.add_argument("--speed", type=float, default=0.1,
                    help="bag playback rate. Bounded by SAM 3's own per-frame throughput"
                         " against the bag's native camera rate (measured 3.5-7.4 Hz"
                         " across scenes) — sam_node's 'latest-frame-wins' policy means a"
                         " speed that outruns SAM silently drops most frames rather than"
                         " erroring. This has nothing to do with the relaunch cost this"
                         " script removes: raising it must be justified by sam_node's own"
                         " 'frame N: ... | dropped D/I' log staying flat, not by there"
                         " being no relaunch left to hide behind. Defaults to the same"
                         " 0.1 eval_orchestrator itself uses (eval-cat1/eval-cat2)")
    ap.add_argument("--backend", default="auto", help="cloud | local | auto ($VLM_BACKEND)")
    ap.add_argument("--cat2-mode", default="hybrid")
    ap.add_argument("--cache", default="/data/runs/views_cache.json", dest="report_file")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--sam-config", default="sam3_mecanum_sim.yaml")
    ap.add_argument("--benchmark-dir", default="/data/benchmark")
    ap.add_argument("--bags-dir", default="/data/bags")
    ap.add_argument("--max-context-views", type=int, default=3)
    ap.add_argument("--vqa-model", default="qwen3vl")
    ap.add_argument("--vqa-quantization", default="int4")
    ap.add_argument("--ready-timeout", type=float, default=420.0,
                    help="seconds to wait for the persistent pipeline to come up ONCE")
    ap.add_argument("--sam-ready-timeout", type=float, default=180.0)
    ap.add_argument("--ack-timeout", type=float, default=180.0)
    ap.add_argument("--answer-timeout", type=float, default=180.0)
    ap.add_argument("--vqa-timeout-s", type=float, default=180.0)
    ap.add_argument("--pre-bag-settle", type=float, default=1.0)
    ap.add_argument("--post-bag-wait", type=float, default=45.0,
                    help="seconds to let SAM (+ map_node, category 2) drain its frame "
                         "backlog before /pipeline/explore_done")
    args = ap.parse_args(argv)

    args.resolved_backend = VLM_BACKEND if args.backend in ("", "auto") else args.backend

    report_path = _require_container_path(Path(args.report_file), "--cache")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = SweepConfig(
        category=args.category,
        scene=args.scene,
        question_limit=args.question_limit,
        benchmark_dir=Path(args.benchmark_dir),
        bags_dir=Path(args.bags_dir),
    )
    questions = list(discover_questions(cfg))
    if not questions:
        log("no questions to run — check scene/benchmark_dir", err=True)
        return 1
    scenes = sorted({s for s, _ in questions})
    log(f"category {args.category}: {len(questions)} question(s) across "
        f"{len(scenes)} scene(s): {scenes}")

    cached = cached_rows(report_path, args.category) if args.resume else {}
    if cached:
        log(f"resume: {len(cached)} question(s) already have crops — keeping their rows")

    def extra() -> dict:
        e = {"crops_only": True, "category": args.category}
        if args.category == 2:
            e["cat2_mode"] = args.cat2_mode
        return e

    managed = spawn_pipeline(args)
    results: list[dict] = []
    interrupted = False
    fatal: Optional[str] = None
    rclpy.init()
    driver = CacheBenchDriver(args.category)
    try:
        wait_pipeline_ready(driver, args)
        for scene, entry in questions:
            row = cached.get((scene, entry["id"]))
            if row and row["question"] == entry["question"]:
                log(f"=== {scene} {entry['id']} === already cached, skipping")
                results.append({**row, "resumed": True})
            else:
                results.append(run_question(driver, cfg, scene, entry, args))
            write_report(report_path, results, extra())
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)
    except Exception as exc:  # noqa: BLE001 — still tear down and write what we have
        fatal = f"{type(exc).__name__}: {exc}"
        log(f"fatal: {fatal}", err=True)
    finally:
        driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        teardown_all(managed)

    if results:
        write_report(report_path, results, extra())
    summary = summarise(results)
    log(f"done: {summary['total_run']} run, {summary['errors']} error(s) -> {report_path}")
    if fatal:
        return 1
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
