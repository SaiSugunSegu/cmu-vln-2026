#!/usr/bin/env python3
"""End-to-end category-1 evaluation: every scene, every question, clean slate each time.

The challenge relaunches the whole system for each language command so nothing carries
over between questions (README "Evaluation"). This harness does the same: per question it
spawns `smart_vlm.launch` as its own process group, drives it over topics, records the
result, then SIGINTs the group. Model loading therefore lands inside the measured budget,
exactly as it will on the real evaluation — `time_taken_s` in the report is honest.

Per question:

  1. publish /gt_target_objects            (latched, so the reasoner sees it whenever it starts)
  2. spawn  smart_vlm.launch use_bag:=true bag:=<scene>
  3. wait   /pipeline/ready                 models loaded
  4. publish /challenge_question @ 1 Hz     (the eval node's own cadence)
  5. wait   /pipeline/armed                 SAM holds this question's prompts; the bag is released
  6. wait   /numerical_response             the answer
  7. record, tear the group down, wait for the ROS graph to drain, next

Waiting on /pipeline/armed rather than /sam3/status is what makes the non-GT path work:
SAM's status is `awaiting_prompts` from boot, so it says nothing about whether this
question's targets ever reached it.

  ros2 run smart_vlm eval_orchestrator --ros-args -p scene:=all -p question_limit:=0
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


def log(message: str, *, err: bool = False) -> None:
    print(f"[orchestrator] {message}", file=sys.stderr if err else sys.stdout, flush=True)


class EvalOrchestratorNode(Node):
    """Long-lived driver. Only this node survives across questions."""

    def __init__(self) -> None:
        super().__init__("eval_orchestrator")

        self.scene = str(self._param("scene", "all"))
        self.report_file = str(self._param("report_file", "/data/runs/challenge_report.json"))
        self.question_limit = int(self._param("question_limit", 0))
        self.target_source = str(self._param("target_source", "gt"))
        # A real double, like every other numeric param here. ROS matches override types
        # against the declared one, so pass it with a decimal point: speed:=2.0, not 2.
        self.speed = float(self._param("speed", 1.0))
        self.sam_config = str(self._param("sam_config", "sam3_mecanum_sim.yaml"))
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
        self.sam_status: Optional[str] = None

        self.pub_question = self.create_publisher(String, "/challenge_question", 10)
        self.pub_gt_targets = self.create_publisher(String, "/gt_target_objects", LATCHED)
        self.create_subscription(String, "/pipeline/ready", self._on_ready, LATCHED)
        self.create_subscription(String, "/pipeline/armed", self._on_armed, LATCHED)
        self.create_subscription(String, "/sam3/status", self._on_sam_status, LATCHED)
        self.create_subscription(Int32, "/numerical_response", self._on_answer, 10)

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

    def reset_for_question(self) -> None:
        self.pipeline_ready = False
        self.pipeline_armed = False
        self.armed_prompts = None
        self.predicted = None
        self.sam_status = None

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

def spawn_pipeline(node: EvalOrchestratorNode, scene: str) -> subprocess.Popen:
    cmd = [
        "ros2", "launch", "smart_vlm", "smart_vlm.launch",
        "use_bag:=true",
        f"bag:={scene}",
        f"speed:={node.speed}",
        f"sam_config:={node.sam_config}",
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

def available_scenes(node: EvalOrchestratorNode) -> set[str]:
    """Scenes we can actually replay: already on disk, or fetchable via scenes.yaml."""
    scenes = {p.name for p in node.bags_dir.iterdir() if p.is_dir()} \
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

    if node.scene and node.scene != "all":
        scenes = [node.scene]
    else:
        scenes = sorted(p.name for p in node.benchmark_dir.iterdir()
                        if (p / "category_1").is_dir())

    playable = available_scenes(node)
    for scene in scenes:
        qa_file = node.benchmark_dir / scene / "category_1" / f"{scene}_category1_qa.json"
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


# -- one question -----------------------------------------------------------

def run_question(node: EvalOrchestratorNode, scene: str, entry: dict) -> dict:
    qid = entry["id"]
    question = entry["question"]
    gt_answer = int(entry["answer"])

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

        proc = spawn_pipeline(node, scene)

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
        if not node.spin_until(ask_and_check(lambda: node.predicted is not None),
                               node.answer_timeout_s, proc):
            raise TimeoutError(f"no answer within {node.answer_timeout_s:.0f}s")

    except Exception as exc:  # noqa: BLE001 — one bad question must not end the sweep
        error = f"{type(exc).__name__}: {exc}"
        log(error, err=True)
    finally:
        teardown(node, proc)

    elapsed = time.monotonic() - t_start
    predicted = node.predicted
    correct = predicted is not None and predicted == gt_answer
    log(f"result: predicted={predicted} gt={gt_answer} correct={correct} "
        f"time={elapsed:.1f}s")

    return {
        "scene": scene,
        "id": qid,
        "question": question,
        "gt": gt_answer,
        "predicted": predicted,
        "correct": correct,
        "time_taken_s": round(elapsed, 2),
        "target_source": node.target_source,
        "prompts": node.armed_prompts,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# -- report -----------------------------------------------------------------

def summarise(results: list[dict]) -> dict[str, Any]:
    def accuracy(rows: list[dict]) -> float:
        return round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else 0.0

    scenes = sorted({r["scene"] for r in results})
    timed = [r["time_taken_s"] for r in results if r["error"] is None]
    return {
        "total_run": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy(results),
        "errors": sum(1 for r in results if r["error"]),
        "mean_time_s": round(sum(timed) / len(timed), 2) if timed else None,
        "per_scene": {
            scene: {
                "run": len([r for r in results if r["scene"] == scene]),
                "correct": sum(1 for r in results if r["scene"] == scene and r["correct"]),
                "accuracy": accuracy([r for r in results if r["scene"] == scene]),
            }
            for scene in scenes
        },
    }


def write_report(path: Path, results: list[dict]) -> None:
    """Rewrite the whole report after every question, so an interrupted sweep is usable."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"summary": summarise(results), "results": results}, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)  # atomic, so the file is never observed half-written


# -- driver -----------------------------------------------------------------

def run_orchestration(node: EvalOrchestratorNode) -> int:
    report_path = Path(node.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    questions = list(discover_questions(node))
    if not questions:
        log("no questions to run — check scene/benchmark_dir", err=True)
        return 1
    scenes = sorted({s for s, _ in questions})
    log(f"{len(questions)} question(s) across {len(scenes)} scene(s): {scenes}")

    results: list[dict] = []
    interrupted = False
    try:
        for scene, entry in questions:
            results.append(run_question(node, scene, entry))
            write_report(report_path, results)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)

    if results:
        write_report(report_path, results)
    summary = summarise(results)
    log(f"done: {summary['correct']}/{summary['total_run']} correct "
        f"(accuracy {summary['accuracy']}, errors {summary['errors']}) -> {report_path}")
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
