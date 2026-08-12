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


def summarise(results: list[dict]) -> dict[str, Any]:
    def accuracy(rows: list[dict]) -> float:
        return round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else 0.0

    scenes = sorted({r["scene"] for r in results})
    # Timings from failed rows are the duration of a timeout, not of an answer, so they
    # would drag the mean toward the phase budget rather than describe the model.
    timed = [r["time_taken_s"] for r in results if r.get("error") is None]
    # Category 2 is graded on overlap, not on identity, so "how many were correct" describes
    # it only partly: a right object with a loose box and a wrong object that happens to
    # overlap both count as neither hit nor miss. Rows that carry a `score` therefore also
    # get the mean of it, which is the number the challenge actually awards.
    scored = [float(r["score"]) for r in results
              if isinstance(r.get("score"), (int, float))]
    graded = {
        "mean_score": round(sum(scored) / len(scored), 4),
        "total_score": round(sum(scored), 2),
        "max_score": round(2.0 * len(results), 2),
    } if scored else {}
    return {
        "total_run": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy(results),
        "errors": sum(1 for r in results if r.get("error")),
        "mean_time_s": round(sum(timed) / len(timed), 2) if timed else None,
        **graded,
        "per_scene": {
            scene: {
                "run": len([r for r in results if r["scene"] == scene]),
                "correct": sum(1 for r in results if r["scene"] == scene and r["correct"]),
                "accuracy": accuracy([r for r in results if r["scene"] == scene]),
            }
            for scene in scenes
        },
    }


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
