#!/usr/bin/env python3
"""Score target extraction against each question's `target_objects`, text only.

Cat1 and cat2 arm SAM with nouns taken from the question. This replays that extract
call — the same `EXTRACT_SYSTEM` and raw-question user turn the live reasoners use —
against every QA file, and compares the cleaned list to the benchmark's
`target_objects`. No bag, no SAM, no crops: one cheap `lite=True` call per question.

    ros2 run smart_vlm extract_bench --category all --report /data/runs/extract_bench.json
    ros2 run smart_vlm extract_bench --backend local --category 1 --scene arabic_room --limit 2

`exact` / `correct` is set equality after `clean_targets`. `coverage` is the fraction
of GT nouns the model still named, allowing "trash can" to cover "black trash can"
(the extract prompt drops color and pattern). `precision` is the fraction of predicted
nouns that cover at least one GT entry.

`--backend` defaults to cloud (VLM_PROVIDER / VLM_MODEL_LITE in .env; extract is a
`lite=True` call). `--backend local` answers over the resident qwen_vqa_server instead
(`just vqa-up` must already be running). Comparing two backends is two runs with
different --report paths.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import MODEL_NAME, MODEL_NAME_LITE, VLM_PROVIDER
from captioner.vlm_backends.schemas import TargetList
from smart_vlm.numerical_utils import EXTRACT_SYSTEM, clean_targets
from smart_vlm.report_utils import summarise, write_report

CATEGORIES = (1, 2)


def log(message: str, *, err: bool = False) -> None:
    print(f"[extract_bench] {message}", file=sys.stderr if err else sys.stdout, flush=True)


def squash_label(text: str) -> str:
    """Letters and digits only, lowercased — so 'Potted Plant' and 'pottedplant' match."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def noun_covers(pred: str, gt: str) -> bool:
    """True when pred and gt name the same object, ignoring spaces and color-style extras."""
    sp, sg = squash_label(pred), squash_label(gt)
    return bool(sp and sg and (sp == sg or sp in sg or sg in sp))


def score_extraction(pred: list[str], gt: list[str]) -> dict[str, Any]:
    """Compare a predicted noun list to GT `target_objects`. Both sides are cleaned first."""
    pred_n = clean_targets(pred)
    gt_n = clean_targets(gt)
    hit = [g for g in gt_n if any(noun_covers(p, g) for p in pred_n)]
    extra = [p for p in pred_n if not any(noun_covers(p, g) for g in gt_n)]
    missing = [g for g in gt_n if g not in hit]
    covered_pred = sum(1 for p in pred_n if any(noun_covers(p, g) for g in gt_n))
    return {
        "pred": pred_n,
        "gt": gt_n,
        "hit": hit,
        "extra": extra,
        "missing": missing,
        "exact": set(pred_n) == set(gt_n),
        "coverage": (len(hit) / len(gt_n)) if gt_n else 1.0,
        "precision": (covered_pred / len(pred_n)) if pred_n else 1.0,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score target extraction against benchmark target_objects (text only).")
    parser.add_argument(
        "--category", default="all", choices=("1", "2", "all"),
        help="1, 2, or both (default: %(default)s)")
    parser.add_argument("--scene", default="all", help="restrict to one scene")
    parser.add_argument(
        "--limit", type=int, default=0, help="first N questions per scene, 0 = all")
    parser.add_argument(
        "--benchmark-dir", default="/data/benchmark",
        help="QA files live here (default: %(default)s)")
    parser.add_argument(
        "--report", default="/data/runs/extract_bench_report.json",
        help="where to write this run's report (default: %(default)s)")
    parser.add_argument(
        "--backend", default="cloud", choices=("cloud", "local"),
        help="where the extract call runs; local needs `just vqa-up` already up "
             "(default: %(default)s)")
    return parser.parse_args(argv)


def categories_from_arg(value: str) -> tuple[int, ...]:
    return CATEGORIES if value == "all" else (int(value),)


def load_questions(
    benchmark_dir: Path, categories: tuple[int, ...], scene: str, limit: int,
) -> list[tuple[int, str, dict]]:
    """(category, scene, question) in benchmark order, filtered like a sweep filters."""
    out: list[tuple[int, str, dict]] = []
    for cat in categories:
        folder = f"category_{cat}"
        scenes = [scene] if scene and scene != "all" else sorted(
            p.name for p in benchmark_dir.iterdir() if (p / folder).is_dir())
        for name in scenes:
            qa_file = benchmark_dir / name / folder / f"{name}_category{cat}_qa.json"
            if not qa_file.is_file():
                log(f"skipping {name} cat{cat}: no QA file at {qa_file}", err=True)
                continue
            with open(qa_file, "r", encoding="utf-8") as handle:
                questions = json.load(handle).get("questions") or []
            chosen = questions[:limit] if limit > 0 else questions
            out.extend((cat, name, q) for q in chosen)
    return out


def extract_one(backend, category: int, scene: str, entry: dict) -> dict:
    """Score one question. Never raises: a bad row is one error, not a lost run."""
    started = time.monotonic()
    question = entry.get("question") or ""
    gt = entry.get("target_objects") or []
    raw: list[str] | None = None
    error: Optional[str] = None

    try:
        result = backend.ask(EXTRACT_SYSTEM, question, [], TargetList, lite=True)
        raw = list(result.targets)
        scored = score_extraction(result.targets, gt)
    except (VLMError, OSError, ValueError, KeyError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        scored = score_extraction([], gt)
        log(f"cat{category} {scene} {entry.get('id')}: {error}", err=True)
    else:
        flag = "OK" if scored["exact"] else "miss"
        log(f"{flag} cat{category} {scene} {entry.get('id')}: "
            f"pred={scored['pred']} gt={scored['gt']}")

    return {
        "scene": scene,
        "id": entry.get("id"),
        "question": question,
        "category": category,
        "gt": scored["gt"],
        "pred": scored["pred"],
        "extract_reply": raw,
        "hit": scored["hit"],
        "extra": scored["extra"],
        "missing": scored["missing"],
        "exact": scored["exact"],
        "coverage": round(scored["coverage"], 4),
        "precision": round(scored["precision"], 4),
        "correct": scored["exact"] and error is None,
        "time_taken_s": round(time.monotonic() - started, 2),
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _agg(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "exact": 0.0, "mean_coverage": 0.0, "mean_precision": 0.0}
    return {
        "n": n,
        "exact": round(sum(1 for r in rows if r["exact"]) / n, 4),
        "mean_coverage": round(sum(r["coverage"] for r in rows) / n, 4),
        "mean_precision": round(sum(r["precision"] for r in rows) / n, 4),
    }


def extract_extras(results: list[dict], args, backend) -> dict[str, Any]:
    if args.backend == "cloud":
        model, provider = (MODEL_NAME_LITE or MODEL_NAME), VLM_PROVIDER
    else:
        model, provider = "qwen_vqa_server (local)", "local"
    return {
        "kind": "extract",
        "category": args.category,
        "backend": args.backend,
        "model": model,
        "provider": provider,
        "backend_name": backend.name,
        "extract": {
            "all": _agg(results),
            "cat1": _agg([r for r in results if r["category"] == 1]),
            "cat2": _agg([r for r in results if r["category"] == 2]),
        },
    }


def build_backend(args):
    """(backend, vqa_client). vqa_client is only set for local and must be closed."""
    if args.backend == "local":
        from captioner.vlm_backends.qwen_ros_client import LocalVqaClient
        vqa_client = LocalVqaClient(log=log)
        return make_backend("local", ask_vqa=vqa_client.ask_vqa, log=log), vqa_client
    return make_backend("cloud", log=log), None


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_dir():
        log(f"no benchmark at {benchmark_dir}", err=True)
        sys.exit(1)

    questions = load_questions(
        benchmark_dir, categories_from_arg(args.category), args.scene, args.limit)
    if not questions:
        log(f"no questions for category={args.category} scene={args.scene}", err=True)
        sys.exit(1)

    backend, vqa_client = build_backend(args)
    log(f"{len(questions)} question(s), category={args.category}, backend {backend.name}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    extra = extract_extras([], args, backend)

    results: list[dict] = []
    interrupted = False
    try:
        for category, scene, entry in questions:
            results.append(extract_one(backend, category, scene, entry))
            extra = extract_extras(results, args, backend)
            write_report(report_path, results, extra)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)
    finally:
        if vqa_client is not None:
            vqa_client.close()

    if results:
        extra = extract_extras(results, args, backend)
        write_report(report_path, results, extra)
    summary = summarise(results)
    extract = extra.get("extract", {}).get("all", {})
    log(f"done: exact {summary['correct']}/{summary['total_run']} "
        f"(coverage {extract.get('mean_coverage')}, "
        f"precision {extract.get('mean_precision')}, "
        f"errors {summary['errors']}) -> {report_path}")
    sys.exit(130 if interrupted else 0)


if __name__ == "__main__":
    main()
