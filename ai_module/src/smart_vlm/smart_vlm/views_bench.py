#!/usr/bin/env python3
"""Benchmark a VLM over best views a previous sweep already saved.

TARE exploration and SAM detection are the expensive half of `eval_orchestrator`: a
full sweep is hours, and almost all of it is spent producing the same crops again.
Those crops are already on disk under /data/crops/<sweep>/<scene>/<question>, and a
sweep records which directory belongs to which question, so the answering step is replayed
on its own — minutes per model instead of hours, and the perception input is
byte-identical across models, which is the only way an A/B says anything about the model.

    # once: the expensive half. Extraction still runs on a real model, so SAM is armed
    # exactly as it would be on a scored run; only the counting call is skipped.
    ros2 run smart_vlm eval_orchestrator --ros-args -p crops_only:=true \\
        -p target_source:=vlm -p vlm_backend:=cloud \\
        -p report_file:=/data/runs/views_cache.json

    # then, per model, as often as you like
    ros2 run smart_vlm views_bench --cache /data/runs/views_cache.json \\
        --report /data/runs/bench_gemini.json

The model is whatever the environment selects (VLM_PROVIDER / VLM_MODEL, see
captioner/vlm_backends/constants.py), so comparing two providers is two runs of this
with different variables and different --report paths.

What this does NOT cover: target extraction and everything downstream of it. The cached
crops were produced from one particular set of SAM prompts, so a model that would have
extracted different targets cannot be measured here — that needs a full sweep.

Any sweep's report works as the cache index, not just a `crops_only` one: every run of
eval_orchestrator saves its crops under the same layout and records the directory.
`crops_only` only skips paying for an answer nobody asked for.

Cloud backends only. The local Qwen path answers over ROS topics, which is exactly the
graph this script exists to avoid; benchmark it with a normal `backend:=local` sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from captioner.paths import secure_path
from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import MODEL_NAME, VLM_PROVIDER
from captioner.vlm_backends.schemas import CountAnswer
from smart_vlm.numerical_utils import ANSWER_SYSTEM, select_context_views
from smart_vlm.report_utils import summarise, write_report


def log(message: str, *, err: bool = False) -> None:
    print(f"[views_bench] {message}", file=sys.stderr if err else sys.stdout, flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the counting step over cached best views.")
    parser.add_argument(
        "--cache", default="/data/runs/views_cache.json",
        help="report from any earlier sweep; its rows supply best_view_dir "
             "(default: %(default)s)")
    parser.add_argument(
        "--report", default="/data/runs/views_bench_report.json",
        help="where to write this run's report (default: %(default)s)")
    parser.add_argument(
        "--views", type=int, default=3,
        help="how many best-view ranks to answer from, capped by what was cached "
             "(default: %(default)s)")
    parser.add_argument("--scene", default="all", help="restrict to one scene")
    parser.add_argument(
        "--limit", type=int, default=0, help="first N questions per scene, 0 = all")
    return parser.parse_args(argv)


def load_cache(path: Path, scene: str, limit: int) -> list[dict]:
    """The cached questions to replay, in report order, filtered like the sweep is."""
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle).get("results") or []

    if scene and scene != "all":
        rows = [r for r in rows if r.get("scene") == scene]
    if limit > 0:
        kept: list[dict] = []
        seen: dict[str, int] = {}
        for row in rows:
            name = row.get("scene", "")
            if seen.get(name, 0) >= limit:
                continue
            seen[name] = seen.get(name, 0) + 1
            kept.append(row)
        rows = kept
    return rows


def answer_one(backend, row: dict, views: int) -> dict:
    """Score one cached question. Never raises: a bad row is one error, not a lost run."""
    started = time.monotonic()
    predicted: Optional[int] = None
    error: Optional[str] = None
    images: list[Path] = []

    try:
        run_dir = row.get("best_view_dir")
        if not run_dir:
            raise FileNotFoundError(
                "row has no best_view_dir — it predates the cache index, so re-run the "
                "sweep that produced it")
        # From a JSON file rather than the pipeline, so treated as untrusted input.
        resolved = secure_path(run_dir)
        manifest_path = resolved / "manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

        images = select_context_views(resolved, manifest, views)
        if not images:
            # SAM found nothing for this question. The live reasoner publishes 0 here,
            # so the bench has to as well or the two disagree on the same input.
            log(f"{row['scene']} {row['id']}: no cached views in {resolved} — scoring 0")
            predicted = 0
        else:
            result = backend.ask(ANSWER_SYSTEM, row["question"], images, CountAnswer)
            predicted = max(0, int(result.count))
            log(f"{row['scene']} {row['id']}: count={predicted} "
                f"from {len(images)} view(s): {result.reason}")
    except (VLMError, OSError, ValueError, PermissionError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(f"{row.get('scene')} {row.get('id')}: {error}", err=True)

    gt = row.get("gt")
    return {
        "scene": row.get("scene"),
        "id": row.get("id"),
        "question": row.get("question"),
        "gt": gt,
        "predicted": predicted,
        "correct": predicted is not None and predicted == gt,
        "time_taken_s": round(time.monotonic() - started, 2),
        # Carried over so a bench report explains itself without the cache next to it.
        "target_source": row.get("target_source"),
        "prompts": row.get("prompts"),
        "best_view_dir": row.get("best_view_dir"),
        "views_used": [p.name for p in images],
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cache_path = Path(args.cache)
    if not cache_path.is_file():
        log(f"no cache at {cache_path} — build one with `just cache-views`", err=True)
        sys.exit(1)

    rows = load_cache(cache_path, args.scene, args.limit)
    if not rows:
        log(f"no rows in {cache_path} for scene={args.scene}", err=True)
        sys.exit(1)

    # Always cloud: the local backend answers over ROS topics, which is the graph this
    # script exists to avoid, so a backend flag here would have exactly one valid value.
    backend = make_backend("cloud", log=log)
    log(f"{len(rows)} cached question(s), {args.views} view(s) each, backend "
        f"{backend.name}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"backend": backend.name, "model": MODEL_NAME, "provider": VLM_PROVIDER,
             "views": args.views, "cache": str(cache_path)}

    results: list[dict] = []
    interrupted = False
    try:
        for row in rows:
            results.append(answer_one(backend, row, args.views))
            # After every question: a rate-limited run that dies halfway still leaves
            # usable numbers behind.
            write_report(report_path, results, extra)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)

    if results:
        write_report(report_path, results, extra)
    summary = summarise(results)
    log(f"done: {summary['correct']}/{summary['total_run']} correct "
        f"(accuracy {summary['accuracy']}, errors {summary['errors']}) -> {report_path}")
    sys.exit(130 if interrupted else 0)


if __name__ == "__main__":
    main()
