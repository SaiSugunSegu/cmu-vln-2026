"""Scoring and report writing, shared by the live sweep and the offline views bench.

Both produce the same {summary, results} document. That is the whole point of keeping
this in one place: the reason to run `views_bench` is to compare its accuracy against a
full `eval_orchestrator` sweep, and two implementations of "how many were correct" would
eventually disagree about an error row or a None prediction and make the comparison a
lie.

No ROS imports, so the bench does not need a graph and the tests do not need rclpy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarise(results: list[dict]) -> dict[str, Any]:
    def accuracy(rows: list[dict]) -> float:
        return round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else 0.0

    scenes = sorted({r["scene"] for r in results})
    # Timings from failed rows are the duration of a timeout, not of an answer, so they
    # would drag the mean toward the phase budget rather than describe the model.
    timed = [r["time_taken_s"] for r in results if r.get("error") is None]
    return {
        "total_run": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": accuracy(results),
        "errors": sum(1 for r in results if r.get("error")),
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
