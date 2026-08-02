#!/usr/bin/env python3
"""Category-1 bag benchmark driver (runs INSIDE iros2026_ai_module).

Challenge-like loop per QA entry:
  0. Wait for /sam3/status == ready (SAM weights loaded)
  1. Publish /challenge_question at 1 Hz
  2. Wait for /sam3/prompts_ack (reasoner extracted targets → SAM)
  3. Wait again for /sam3/status == ready (prompts applied)
  4. Play the scene bag once (exploration) — only after SAM is ready
  5. Publish /pipeline/explore_done
  6. Wait for /numerical_response
  7. Score against GT answer; write result JSON

Prerequisites (separate terminals):
  just vqa-up
  just run-sam
  just cat1-reasoner

Usage (from host via just, or inside the container):
  python3 scripts/eval/run_cat1_bag_bench.py \
    --qa /data/workspace/benchmark/arabic_room/category_1/arabic_room_category1_qa.json \
    --scene arabic_room \
    --out /data/workspace/runs/cat1_arabic_room \
    --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String


def _rewrite_host_data_path(path: Path) -> Path:
    """Map host repo data/ paths to the container /data/workspace mount when present."""
    text = str(path)
    if text.startswith("/data/"):
        return path
    # Common host layout: <repo>/data/...
    parts = path.resolve().parts if path.exists() or path.is_absolute() else path.parts
    if "data" in parts:
        idx = parts.index("data")
        rel = Path(*parts[idx + 1 :])
        candidate = Path("/data/workspace") / rel
        if candidate.exists() or not path.exists():
            return candidate
    return path


class Cat1BagDriver(Node):
    def __init__(self, question: str, run_id: str, ack_timeout: float, answer_timeout: float):
        super().__init__("cat1_bag_driver")
        self.question = question
        self.run_id = run_id
        self.ack_timeout = ack_timeout
        self.answer_timeout = answer_timeout

        self._ack: dict | None = None
        self._ack_event = threading.Event()
        self._answer: int | None = None
        self._answer_event = threading.Event()
        self._best_view_dir: str | None = None
        self._tick_on = False
        self._sam_status: str | None = None

        latch = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.q_pub = self.create_publisher(String, "/challenge_question", 10)
        self.done_pub = self.create_publisher(String, "/pipeline/explore_done", 10)
        self.create_subscription(String, "/sam3/prompts_ack", self._on_ack, 10)
        self.create_subscription(String, "/sam3/best_view_dir", self._on_dir, latch)
        self.create_subscription(String, "/sam3/status", self._on_sam_status, latch)
        self.create_subscription(Int32, "/numerical_response", self._on_answer, 10)
        self.create_timer(1.0, self._tick)

    def _tick(self):
        if self._tick_on:
            self.q_pub.publish(String(data=self.question))

    def _on_sam_status(self, msg: String):
        self._sam_status = (msg.data or "").strip()

    def _on_ack(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # Reasoner assigns its own run_id; accept the first successful ack after we
        # started publishing this question.
        if self._ack_event.is_set() or not payload.get("ok"):
            return
        if not self._tick_on:
            return
        self._ack = payload
        if payload.get("run_dir"):
            self._best_view_dir = payload["run_dir"]
        self._ack_event.set()

    def _on_dir(self, msg: String):
        self._best_view_dir = msg.data

    def _on_answer(self, msg: Int32):
        if self._answer is None:
            self._answer = int(msg.data)
            self._answer_event.set()

    def start_question(self):
        self._tick_on = True
        self.q_pub.publish(String(data=self.question))

    def stop_question(self):
        self._tick_on = False

    def wait_sam_ready(self, timeout_s: float, *, context: str = "SAM") -> None:
        """Block until latched /sam3/status is 'ready' (weights loaded / prompts applied)."""
        deadline = time.time() + timeout_s
        last_log = 0.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._sam_status == "ready":
                print(f"[cat1-bench] {context} ready (/sam3/status=ready)", flush=True)
                return
            now = time.time()
            if now - last_log >= 5.0:
                print(
                    f"[cat1-bench] waiting for {context} "
                    f"(status={self._sam_status!r})…",
                    flush=True,
                )
                last_log = now
        raise TimeoutError(
            f"/sam3/status not ready within {timeout_s:.0f}s "
            f"(last={self._sam_status!r}); is `just run-sam` up?"
        )

    def wait_ack(self) -> dict:
        deadline = time.time() + self.ack_timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._ack_event.is_set():
                return self._ack or {}
        raise TimeoutError(f"No /sam3/prompts_ack within {self.ack_timeout:.0f}s")

    def signal_explore_done(self):
        self.done_pub.publish(String(data=json.dumps({
            "run_id": self.run_id,
            "scene_done": True,
        })))

    def wait_answer(self) -> int:
        deadline = time.time() + self.answer_timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._answer_event.is_set() and self._answer is not None:
                return self._answer
        raise TimeoutError(f"No /numerical_response within {self.answer_timeout:.0f}s")


def play_bag(scene: str, bags_dir: str, speed: float) -> None:
    """Single-pass bag play via the existing launch file."""
    cmd = (
        f"source /home/docker/ai_module/install/setup.bash && "
        f"ros2 launch smart_vlm bag_replay.launch "
        f"scene:={scene} bags_dir:={bags_dir} speed:={speed} loop:=false"
    )
    print(f"[cat1-bench] playing bag scene={scene} speed={speed}", flush=True)
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bag play exited with code {proc.returncode}")


def run_one(args, entry: dict, scene: str) -> dict:
    qid = entry["id"]
    question = entry["question"]
    gt = int(entry["answer"])
    run_id = f"{scene}_{qid}"

    print(f"\n[cat1-bench] === {qid}: {question} (gt={gt}) ===", flush=True)

    rclpy.init()
    driver = Cat1BagDriver(
        question=question,
        run_id=run_id,
        ack_timeout=args.ack_timeout,
        answer_timeout=args.answer_timeout,
    )
    result = {
        "id": qid,
        "question": question,
        "gt": gt,
        "predicted": None,
        "extracted_targets": entry.get("target_objects"),
        "best_view_dir": None,
        "correct": False,
        "error": None,
        "prompts_ack": None,
    }
    try:
        # Spin briefly so pubs/subs connect (incl. latched /sam3/status).
        for _ in range(20):
            rclpy.spin_once(driver, timeout_sec=0.05)

        # 1) SAM must already be up before we ask / extract / play.
        driver.wait_sam_ready(args.sam_ready_timeout, context="SAM (startup)")

        driver.start_question()
        ack = driver.wait_ack()
        # Stop 1 Hz republish after ack so a failed answer cannot re-latch the
        # same question while we still hold this driver instance.
        driver.stop_question()
        result["prompts_ack"] = {
            "prompts": ack.get("prompts"),
            "labels": ack.get("labels"),
            "run_dir": ack.get("run_dir"),
            "run_id": ack.get("run_id"),
        }
        result["extracted_targets"] = ack.get("prompts") or entry.get("target_objects")
        result["gt_target_objects"] = entry.get("target_objects")
        result["best_view_dir"] = ack.get("run_dir") or driver._best_view_dir
        print(f"[cat1-bench] prompts ack: {ack.get('prompts')} -> {ack.get('run_dir')}",
              flush=True)

        # 2) Prompts applied — wait until status is ready again, then start bag.
        driver.wait_sam_ready(args.sam_ready_timeout, context="SAM (prompts applied)")
        if args.pre_bag_settle > 0:
            print(f"[cat1-bench] pre-bag settle {args.pre_bag_settle:.1f}s", flush=True)
            deadline = time.time() + args.pre_bag_settle
            while time.time() < deadline:
                rclpy.spin_once(driver, timeout_sec=0.1)

        play_bag(scene, args.bags_dir, args.speed)

        # Bag play exits when messages are published, but sam_node still has a
        # backlog (drops most frames, processes ~2-4 Hz). Wait so best-views flush.
        if args.post_bag_wait > 0:
            print(f"[cat1-bench] waiting {args.post_bag_wait:.0f}s for SAM to drain",
                  flush=True)
            deadline = time.time() + args.post_bag_wait
            while time.time() < deadline:
                rclpy.spin_once(driver, timeout_sec=0.2)

        driver.signal_explore_done()
        print("[cat1-bench] explore_done published; waiting for answer", flush=True)
        predicted = driver.wait_answer()
        result["predicted"] = predicted
        result["correct"] = predicted == gt
        result["best_view_dir"] = driver._best_view_dir or result["best_view_dir"]
        print(
            f"[cat1-bench] {qid}: predicted={predicted} gt={gt} "
            f"{'OK' if result['correct'] else 'WRONG'}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[cat1-bench] {qid} FAILED: {result['error']}", flush=True)
    finally:
        driver.stop_question()
        driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Small gap so reasoner resets to idle before the next question.
        time.sleep(args.settle)

    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Category-1 bag benchmark (challenge-like ROS).")
    ap.add_argument("--qa", type=Path, required=True,
                    help="Path to <scene>_category1_qa.json")
    ap.add_argument("--scene", default=None,
                    help="Bag scene name (default: QA file 'scene' field)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for results")
    ap.add_argument("--bags-dir", default="/data/bags")
    ap.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    ap.add_argument("--ids", nargs="*", help="Subset of question ids, e.g. Q01 Q02")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ack-timeout", type=float, default=180.0)
    ap.add_argument("--answer-timeout", type=float, default=180.0)
    ap.add_argument("--sam-ready-timeout", type=float, default=600.0,
                    help="Max seconds to wait for /sam3/status=ready before bag play")
    ap.add_argument("--pre-bag-settle", type=float, default=1.0,
                    help="Extra seconds after SAM ready before starting bag play")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="Seconds to wait between questions")
    ap.add_argument("--post-bag-wait", type=float, default=45.0,
                    help="Seconds to let SAM drain frames after bag play exits "
                         "before publishing /pipeline/explore_done")
    args = ap.parse_args(argv)

    qa_path = _rewrite_host_data_path(args.qa)
    out_dir = _rewrite_host_data_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(qa_path, "r", encoding="utf-8") as handle:
        qa = json.load(handle)

    scene = args.scene or qa.get("scene")
    if not scene:
        print("[cat1-bench] --scene required when QA file has no scene field", file=sys.stderr)
        return 2

    questions = qa.get("questions") or []
    if args.ids:
        want = set(args.ids)
        questions = [q for q in questions if q.get("id") in want]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    print(f"[cat1-bench] scene={scene} n={len(questions)} out={out_dir}", flush=True)
    print("[cat1-bench] order: wait SAM ready → set prompts → wait SAM ready → bag",
          flush=True)

    results = []
    for entry in questions:
        res = run_one(args, entry, scene)
        results.append(res)
        with open(out_dir / f"{scene}_{entry['id']}.json", "w", encoding="utf-8") as handle:
            json.dump(res, handle, indent=2)
            handle.write("\n")

    n_ok = sum(1 for r in results if r.get("correct"))
    n_ans = sum(1 for r in results if r.get("predicted") is not None)
    summary = {
        "scene": scene,
        "total": len(results),
        "answered": n_ans,
        "correct": n_ok,
        "accuracy": (n_ok / len(results)) if results else 0.0,
        "results": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"\n[cat1-bench] DONE accuracy={summary['accuracy']:.3f} "
        f"({n_ok}/{len(results)} correct, {n_ans} answered)",
        flush=True,
    )
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    # Avoid unused import warnings when linting without ROS env.
    _ = os.environ
    raise SystemExit(main())
