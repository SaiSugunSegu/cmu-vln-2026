#!/usr/bin/env python3
"""Benchmark category-2 object selection over maps and crops a previous sweep already saved.

The category-2 half of what `cat1_bench` does for category 1, and for the same reason:
exploration and detection are the expensive half of a sweep — hours — while the selection step
is milliseconds to seconds. Replaying selection alone against a fixed cache is the only way an
A/B between two policies says anything, because the perception input is byte-identical and a
difference in score is therefore a difference in reasoning.

    # once: the expensive half (~5.5 h for the 122 questions across 13 scenes)
    ros2 run smart_vlm eval_orchestrator --ros-args -p category:=2 -p crops_only:=true \\
        -p report_file:=/data/runs/cat2_cache.json -p resume:=true

    # then, per mode, as often as you like
    ros2 run smart_vlm cat2_bench --mode hybrid --report /data/runs/cat2_hybrid.json

Four modes, and the gap between them is the whole point:

    naive   largest instance of the class the question names. The floor.
    solver  the benchmark's own spatial predicates over the map. No model call.
    vlm     a model choosing from the candidate table and the marked crops.
    hybrid  solver where the geometry is decisive, model where it is not. What ships.

Every row also carries `ceiling_score` — twice the best IoU any selector could reach against
these boxes, which needs no selection to compute — so one report already separates the three
ways a question is lost: the object is not in the map (`found` false), the wrong candidate was
chosen (`score` far below `ceiling_score`), or the box is loose (`ceiling_score` itself is low).
On the first cache measured, the ceiling was 0.31 of 2, i.e. selection was not the bottleneck.

`loss_bucket` turns that reading into one word per row and `loss_attribution` counts them, so
a change can be checked against the bucket it was supposed to empty: box sizing should move
`box_near_miss`, and a prompt should move `wrong_pick_vlm` while leaving `perception` alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from captioner.paths import secure_path
from captioner.vlm_backends import make_backend
from captioner.vlm_backends.constants import MODEL_NAME, VIEW_SOURCE, VLM_PROVIDER
from smart_vlm.cat2_utils import (
    SOLVER_AVAILABLE,
    marked_views,
    naive_pick,
    select_object,
    solver_status,
)
from smart_vlm.report_utils import summarise, write_report

MODES = ("naive", "solver", "vlm", "hybrid")

# Same threshold the orchestrator reports `correct` with: below this the boxes are touching,
# not describing the same object.
HIT_IOU = 0.25


def log(message: str, *, err: bool = False) -> None:
    print(f"[cat2_bench] {message}", file=sys.stderr if err else sys.stdout, flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay category-2 object selection over cached maps and crops.")
    parser.add_argument("--mode", default="hybrid", choices=MODES,
                        help="selection policy (default: %(default)s)")
    parser.add_argument("--cache", default="/data/runs/cat2_cache.json",
                        help="report from an earlier category-2 sweep; supplies each "
                             "question's crops and map (default: %(default)s)")
    parser.add_argument("--report", default="/data/runs/cat2_bench_report.json",
                        help="where to write this run's report (default: %(default)s)")
    parser.add_argument("--benchmark-dir", default="/data/benchmark",
                        help="category_2 QA lives here (default: %(default)s)")
    parser.add_argument("--scene", default="all", help="restrict to one scene")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N questions per scene, 0 = all")
    parser.add_argument("--views", type=int, default=3,
                        help="best-view ranks shown to the model (default: %(default)s)")
    parser.add_argument("--score-uncached", action="store_true",
                        help="score questions the cache does not cover as zeros instead of "
                             "skipping them. A partial cache otherwise reports a mean over "
                             "the questions it could actually replay")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- inputs


def load_index(path: Path) -> dict[tuple[str, str], dict]:
    """(scene, question id) -> the sweep row that says where its crops and map landed."""
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle).get("results") or []
    return {(r.get("scene"), r.get("id")): r for r in rows if r.get("scene")}


def load_questions(benchmark_dir: Path, scene: str, limit: int) -> list[tuple[str, dict]]:
    """Every category-2 question, in benchmark order, filtered like a sweep filters."""
    scenes = [scene] if scene and scene != "all" else sorted(
        p.name for p in benchmark_dir.iterdir() if (p / "category_2").is_dir())
    out: list[tuple[str, dict]] = []
    for name in scenes:
        qa_file = benchmark_dir / name / "category_2" / f"{name}_category2_qa.json"
        if not qa_file.is_file():
            log(f"skipping {name}: no QA file at {qa_file}", err=True)
            continue
        with open(qa_file, "r", encoding="utf-8") as handle:
            questions = json.load(handle).get("questions") or []
        out.extend((name, q) for q in (questions[:limit] if limit > 0 else questions))
    return out


def gt_object(entry: dict):
    """The answer as an `Obj`, so one IoU function serves the answer and the ground truth."""
    import numpy as np
    from utils.geometry import Obj

    answer = entry["answer"]
    return Obj(str(answer["object_id"]), str(answer.get("region") or "0"), {
        "bbox": np.asarray(answer["bbox_corners"], dtype=float),
        "center": answer["center"],
        "size": answer["size"],
        "raw_label": answer.get("raw_label") or answer.get("label") or "",
        "nyu_label": answer.get("nyu_label") or answer.get("label") or "",
        "color_labels": answer.get("colors") or [],
    })


def cached_map(scene: str, entry: dict, row: Optional[dict]) -> tuple[dict, Path, dict]:
    """This question's map, its run directory and its manifest: (objects, run_dir, manifest)."""
    import utils.objmap as objmap

    if not row or not row.get("best_view_dir"):
        raise FileNotFoundError(
            f"{scene} {entry['id']} is not in the cache — build it with "
            f"`just cache-cat2` (eval_orchestrator category:=2 crops_only:=true)")
    run_dir = secure_path(row["best_view_dir"])
    map_path = run_dir / "obj_map.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"no obj_map.json in {run_dir}")

    manifest: dict = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    prompts = row.get("prompts") or manifest.get("sam_prompts") or entry.get("target_objects")
    return objmap.load_obj_map(map_path, prompts or []), run_dir, manifest


# ---------------------------------------------------------------- one question


def score_one(backend, scene: str, entry: dict, row: Optional[dict], args) -> dict:
    """Answer one question and grade it. Never raises: a bad row is one error, not a run."""
    import utils.objmap as objmap

    started = time.monotonic()
    answer = gt_object(entry)
    out: dict[str, Any] = {
        "scene": scene,
        "id": entry["id"],
        "question": entry["question"],
        "category": 2,
        "relation": entry.get("relation") or "",
        "gt": entry["answer"]["object_id"],
        "gt_label": entry["answer"].get("label"),
        "found": False,
        "n_map_objects": 0,
        "n_candidates": 0,
        "predicted_object_id": None,
        "predicted_label": None,
        "iou": 0.0,
        "score": 0.0,
        "ceiling_iou": 0.0,
        "ceiling_score": 0.0,
        "ceiling_object_id": None,
        "correct": False,
        "selection_source": None,
        "reason": None,
        # Provenance for `attribution`: which ids the choice was made from, which of them
        # the ceiling belongs to, and what the geometry would have answered alone. Without
        # these a lost question cannot be told apart from a lost *ranking* after the fact.
        "candidate_ids": [],
        "solver_first": None,
        "ceiling_rank": None,
        "loss": None,
        "vlm_calls": 0,
        "views_used": [],
        "best_view_dir": (row or {}).get("best_view_dir"),
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        objects, run_dir, manifest = cached_map(scene, entry, row)
        out["n_map_objects"] = len(objects)

        # The ceiling, computed before anything is chosen: the best any selector could do
        # against these boxes. Class-blind, because overlap is what scores — a prediction that
        # lands on the answer's box while calling it something else has still pointed at the
        # right thing.
        best_id, best_iou = None, 0.0
        for oid, obj in objects.items():
            iou = objmap.aabb_iou(obj, answer)
            if iou > best_iou:
                best_id, best_iou = oid, iou
        out.update({"ceiling_object_id": best_id, "ceiling_iou": round(best_iou, 4),
                    "ceiling_score": round(2.0 * best_iou, 4),
                    # "found" is about perception, so it asks whether the answer is in the
                    # map at all — a candidate that overlaps it and shares its class.
                    "found": bool(best_id is not None and best_iou >= HIT_IOU
                                  and objmap.same_class(objects[best_id], answer))})

        if not objects:
            raise ValueError("the map holds no objects")

        views: list[Path] = []

        def views_for(ids: list[str], labels: dict[str, str]) -> list[Path]:
            # wait_s=0: this is a replay over a cache nothing is still writing to, so a
            # missing silhouette can only ever time out — waiting would just cost every
            # question SILHOUETTE_WAIT_S for nothing.
            rendered = marked_views(run_dir, manifest, ids, args.views, labels, wait_s=0.0)
            # Recorded here rather than returned, because only this closure knows which
            # images the model was actually shown.
            views.extend(rendered)
            return rendered

        if args.mode == "naive":
            chosen, reason = naive_pick(entry["question"], objects)
            source, candidates = "naive", [chosen] if chosen else []
        else:
            selection = select_object(
                entry["question"], objects,
                mode=args.mode,
                ask=backend.ask if backend is not None else None,
                views_for=views_for,
                log=lambda m: log(m, err=True),
            )
            chosen, source, reason = (selection.object_id, selection.source, selection.reason)
            candidates = selection.candidates
            out["vlm_calls"] = selection.vlm_calls

        ranked = [str(c) for c in candidates]
        out.update({"selection_source": source, "reason": reason,
                    "n_candidates": len(candidates),
                    "candidate_ids": ranked,
                    "solver_first": ranked[0] if ranked else None,
                    "ceiling_rank": (ranked.index(str(best_id)) + 1
                                     if best_id is not None and str(best_id) in ranked
                                     else None),
                    "views_used": [p.name for p in views]})
        if chosen is not None and chosen in objects:
            picked = objects[chosen]
            overlap = objmap.aabb_iou(picked, answer)
            out.update({
                "predicted_object_id": chosen,
                "predicted_label": picked.display,
                "iou": round(overlap, 4),
                "score": round(2.0 * overlap, 4),
                "correct": overlap >= HIT_IOU,
            })
        out["loss"] = loss_bucket(out)
        log(f"{scene} {entry['id']} [{out['relation'] or '-'}]: "
            f"{out['predicted_label']}#{chosen} via {source} "
            f"-> {out['score']:.2f}/2 (ceiling {out['ceiling_score']:.2f}) — {reason}")
    except Exception as exc:  # noqa: BLE001 — one bad question must not end the run
        out["error"] = f"{type(exc).__name__}: {exc}"
        log(f"{scene} {entry['id']}: {out['error']}", err=True)

    out["time_taken_s"] = round(time.monotonic() - started, 2)
    return out


# ---------------------------------------------------------------- aggregates


#: Ordered because the buckets are a cascade, not a partition of independent causes: a
#: question whose map never held the object cannot also have been mis-ranked, so the first
#: bucket that applies is the one that actually cost the point.
LOSS_BUCKETS = ("hit", "perception", "box_near_miss", "shortlist_miss",
                "wrong_pick_solver", "wrong_pick_vlm", "no_pick", "error")


def loss_bucket(row: dict) -> str:
    """Why this question did not score, in one word.

    The split the plan turns on is the first two buckets against the rest: only the lower
    ones are reachable by changing how an object is *chosen*, so a reasoning change that
    moves the `perception` count is measuring noise rather than itself. `box_near_miss` is
    kept apart from `perception` because the two want opposite fixes — something in the map
    does overlap the answer and the box is merely too loose to clear the threshold, which is
    a sizing bug, not a missed detection.
    """
    if row.get("error"):
        return "error"
    if row.get("correct"):
        return "hit"
    ceiling = float(row.get("ceiling_iou") or 0.0)
    if ceiling <= 0.0:
        return "perception"
    if ceiling < HIT_IOU:
        return "box_near_miss"
    if row.get("predicted_object_id") is None:
        return "no_pick"
    if row.get("ceiling_rank") is None:
        return "shortlist_miss"
    # The shortlist held the answer and the ranking was rejected: blame whichever stage
    # made the final call, so a prompt change is scored against the picks it can move.
    return "wrong_pick_vlm" if row.get("selection_source") == "vlm" else "wrong_pick_solver"


def loss_attribution(results: list[dict]) -> dict[str, int]:
    """How many questions each bucket cost, in cascade order and omitting empty ones."""
    counts = {name: 0 for name in LOSS_BUCKETS}
    for row in results:
        counts[loss_bucket(row)] += 1
    return {name: n for name, n in counts.items() if n}


def per_relation(results: list[dict]) -> dict[str, dict]:
    """Mean score and hit rate by relation, worst first.

    The split is what turns a single number into a decision: a policy that gains overall while
    losing on `between` is a different thing from one that gains everywhere.
    """
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(str(row.get("relation") or "-"), []).append(row)
    out = {
        name: {
            "n": len(rows),
            "hits": sum(1 for r in rows if r["correct"]),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 3),
            "mean_ceiling": round(sum(r["ceiling_score"] for r in rows) / len(rows), 3),
        }
        for name, rows in groups.items()
    }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["mean_score"]))


def cat2_extras(results: list[dict], args) -> dict[str, Any]:
    n = len(results) or 1
    found = [r for r in results if r["found"]]
    return {
        "mode": args.mode,
        "model": MODEL_NAME if args.mode in ("vlm", "hybrid") else None,
        "provider": VLM_PROVIDER if args.mode in ("vlm", "hybrid") else None,
        "view_source": VIEW_SOURCE,
        "views": args.views,
        # The map contains the answer this often: the hard ceiling on everything below.
        "found_rate": round(len(found) / n, 4),
        "mean_ceiling_score": round(sum(r["ceiling_score"] for r in results) / n, 4),
        # Of the questions whose answer IS in the map, how often the right one was chosen.
        # This is the number a change to the selection policy moves.
        "selection_accuracy": round(
            sum(1 for r in found if r["correct"]) / (len(found) or 1), 4),
        "total_vlm_calls": sum(int(r.get("vlm_calls") or 0) for r in results),
        # Where the other (n - correct) questions went. Two runs with the same mean score
        # and different buckets are two different problems.
        "loss_attribution": loss_attribution(results),
        "per_relation": per_relation(results),
        "cache": args.cache,
        "skipped_uncached": getattr(args, "skipped_uncached", 0),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not SOLVER_AVAILABLE:
        log(solver_status(), err=True)
        sys.exit(1)

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_dir():
        log(f"no benchmark at {benchmark_dir}", err=True)
        sys.exit(1)

    cache_path = Path(args.cache)
    if not cache_path.is_file():
        log(f"no cache at {cache_path} — build one with `just cache-cat2`", err=True)
        sys.exit(1)

    questions = load_questions(benchmark_dir, args.scene, args.limit)
    if not questions:
        log(f"no category-2 questions for scene={args.scene}", err=True)
        sys.exit(1)
    index = load_index(cache_path)

    # A cache built for a subset of scenes covers a subset of the questions, and scoring the
    # rest as zeros reports the cache's coverage rather than the policy's accuracy — on a
    # 40-of-122 cache it divides every mean by three. They are dropped rather than counted,
    # and the count is carried into the summary so a thin cache stays visible.
    if not args.score_uncached:
        covered = [(s, e) for s, e in questions if (s, e["id"]) in index]
        args.skipped_uncached = len(questions) - len(covered)
        if args.skipped_uncached:
            log(f"{args.skipped_uncached} question(s) not in the cache — skipping "
                "(pass --score-uncached to score them as zeros)")
        questions = covered
        if not questions:
            log(f"the cache at {cache_path} covers none of these questions", err=True)
            sys.exit(1)

    # Cloud only, and only when a mode actually asks a model: the local Qwen answers over
    # ROS topics, which is the graph this script exists to avoid.
    backend = make_backend("cloud", log=log) if args.mode in ("vlm", "hybrid") else None
    log(f"{len(questions)} question(s), mode={args.mode}, view_source={VIEW_SOURCE}"
        + (f", backend {backend.name}" if backend else ""))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    interrupted = False
    try:
        for scene, entry in questions:
            results.append(
                score_one(backend, scene, entry, index.get((scene, entry["id"])), args))
            # After every question: a rate-limited run that dies halfway still leaves usable
            # numbers behind.
            write_report(report_path, results, cat2_extras(results, args))
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted — writing the partial report", err=True)

    if results:
        write_report(report_path, results, cat2_extras(results, args))
    summary = {**summarise(results), **cat2_extras(results, args)}
    log(f"done: score {summary['total_score']}/{summary['max_score']} "
        f"(mean {summary['mean_score']}/2, ceiling {summary['mean_ceiling_score']}/2), "
        f"{summary['correct']}/{summary['total_run']} on the right object, "
        f"found {summary['found_rate']}, selection {summary['selection_accuracy']}, "
        f"{summary['total_vlm_calls']} model call(s) -> {report_path}")
    log("  loss: " + ", ".join(f"{name} {n}"
                               for name, n in summary["loss_attribution"].items()))
    for name, stats in summary["per_relation"].items():
        log(f"  {name:<10} n={stats['n']:<3} mean {stats['mean_score']:.2f}/2 "
            f"(ceiling {stats['mean_ceiling']:.2f}) hits {stats['hits']}/{stats['n']}")
    sys.exit(130 if interrupted else 0)


if __name__ == "__main__":
    main()
