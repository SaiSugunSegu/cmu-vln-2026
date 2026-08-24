"""Scoring and report writing, shared by the live sweep and the offline views bench.

Both produce the same {summary, results} document. That is the whole point of keeping
this in one place: the reason to run `cat1_bench` is to compare its accuracy against a
full `eval_orchestrator` sweep, and two implementations of "how many were correct" would
eventually disagree about an error row or a None prediction and make the comparison a
lie.

No ROS imports, so the bench does not need a graph and the tests do not need rclpy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def iou3d(center_a, size_a, center_b, size_b) -> float:
    """Axis-aligned 3D IoU of two centre+size boxes.

    The challenge's own formula, transcribed from `scripts/eval/score.py`: a category-2
    answer is scored as twice this, computed from the Marker's `pose.position` and `scale`
    against the ground-truth object's `center` and `size` — orientation is read from neither
    side. `scripts/utils/objmap.py:aabb_iou` is the same arithmetic over the benchmark's `Obj`
    type, for the host-side tools that cannot import this package; if one changes, both must.
    """
    inter = 1.0
    for axis in range(3):
        a_lo, a_hi = center_a[axis] - size_a[axis] / 2.0, center_a[axis] + size_a[axis] / 2.0
        b_lo, b_hi = center_b[axis] - size_b[axis] / 2.0, center_b[axis] + size_b[axis] / 2.0
        overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
        if overlap <= 0.0:
            return 0.0
        inter *= overlap
    vol_a = size_a[0] * size_a[1] * size_a[2]
    vol_b = size_b[0] * size_b[1] * size_b[2]
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0.0 else 0.0


#: Per-question maximum, from README.md "Question Types and Initial Scoring": numerical /1,
#: object reference /2, instruction following /6. Read from each row's own category rather
#: than fixed, or a category-3 sweep reports itself against the wrong denominator.
PER_QUESTION_MAX = {1: 1.0, 2: 2.0, 3: 6.0}


def _scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if isinstance(r.get("score"), (int, float))]


def _available(rows: list[dict]) -> float:
    return sum(PER_QUESTION_MAX.get(int(r.get("category", 2)), 2.0) for r in rows)


def summarise(results: list[dict]) -> dict[str, Any]:
    def accuracy(rows: list[dict]) -> float:
        """The share of the available points these rows earned.

        NOT the share of perfect answers. Categories 2 and 3 award partial credit, and
        category 3 is out of 6 with `correct` reserved for a flawless 6.0 -- so counting only
        perfect rows reported `accuracy: 0.0` for a sweep that had in fact earned 7 of its 12
        points. A number that reads as total failure for a run doing better than half the job
        is worse than no number.

        `correct` is still reported beside this and still means the strict thing, so nothing
        is lost: one field for "how many were flawless", one for "how much did we score".

        Rows with no `score` at all fall back to the strict count. That is category 1, whose
        answer is simply right or wrong -- the two definitions agree there anyway.
        """
        if not rows:
            return 0.0
        scored = _scored(rows)
        if not scored:
            return round(sum(1 for r in rows if r["correct"]) / len(rows), 4)
        available = _available(scored)
        if available <= 0.0:
            return 0.0
        return round(sum(float(r["score"]) for r in scored) / available, 4)

    scenes = sorted({r["scene"] for r in results})
    # Timings from failed rows are the duration of a timeout, not of an answer, so they
    # would drag the mean toward the phase budget rather than describe the model.
    timed = [r["time_taken_s"] for r in results if r.get("error") is None]
    # Category 2 is graded on overlap, not on identity, so "how many were correct" describes
    # it only partly: a right object with a loose box and a wrong object that happens to
    # overlap both count as neither hit nor miss. Rows that carry a `score` therefore also
    # get the mean of it, which is the number the challenge actually awards.
    scored_rows = _scored(results)
    scored = [float(r["score"]) for r in scored_rows]
    graded = {
        "mean_score": round(sum(scored) / len(scored), 4),
        "total_score": round(sum(scored), 2),
        "max_score": round(_available(scored_rows), 2),
    } if scored else {}
    # What the plans were worth if driven perfectly (category 3 only, see
    # eval_orchestrator.plan_quality). Reported beside the real total because the gap between
    # them is the question worth asking: a planning problem and a driving problem look
    # identical in `total_score` alone.
    planned = [float(r["plan_score"]) for r in results
               if isinstance(r.get("plan_score"), (int, float))]
    if planned:
        graded["total_plan_score"] = round(sum(planned), 2)
        graded["total_drive_loss"] = round(
            sum(float(r.get("drive_loss") or 0.0) for r in results
                if isinstance(r.get("plan_score"), (int, float))), 2)
    return {
        "total_run": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy(results),
        "errors": sum(1 for r in results if r.get("error")),
        "mean_time_s": round(sum(timed) / len(timed), 2) if timed else None,
        **graded,
        "per_scene": {
            scene: {
                "run": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "accuracy": accuracy(rows),
                # The two numbers the fraction above is made of, so a scene that scored
                # badly can be told from one that ran only a question or two.
                **({"score": round(sum(float(r["score"]) for r in _scored(rows)), 2),
                    "max_score": round(_available(_scored(rows)), 2)}
                   if _scored(rows) else {}),
            }
            for scene in scenes
            for rows in [[r for r in results if r["scene"] == scene]]
        },
    }


def previous_results(report_path: Path) -> list[dict]:
    """Rows already in a report, so a later run can extend rather than replace them.

    Unlike cached_rows this filters nothing: the caller wants the sweep's history intact,
    including failed rows, because a report that silently dropped them would overstate
    accuracy. A missing or unreadable file is an empty history, not an error — the first
    scene of a sweep runs with no report there yet.
    """
    if not report_path.is_file():
        return []
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle).get("results") or []
    except (OSError, json.JSONDecodeError) as err:
        print(f"[report] could not read {report_path} to append: {err}")
        return []
    return [r for r in rows if isinstance(r, dict)]


def write_report(path: Path, results: list[dict], extra: dict | None = None) -> None:
    """Rewrite the whole report after every question, so an interrupted run is usable.

    `extra` is merged into the summary — the bench uses it to record which model and how
    many views produced these numbers, without which two bench reports are
    indistinguishable.
    """
    summary = summarise(results)
    if extra:
        summary.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)  # atomic, so the file is never observed half-written
