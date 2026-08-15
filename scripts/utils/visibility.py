#!/usr/bin/env python3
"""The visibility gate: has the robot's camera actually seen this object?

IRef-VLA is annotated from the Unity model, so its boxes include geometry no robot on the
floor of that room ever imaged -- recessed downlights a 0.85 m camera passes under, the far
side of a wall, a book that covers 270 px^2 of a 1920x640 panorama. A category-2 question
about one of those has a perfectly good 3D answer and no way to earn it from the sensors,
which measures the benchmark rather than the system.

``scripts/eval/object_visibility.py`` measures what was seen, per scene, by projecting every
box into the robot's own camera frames and requiring lidar returns off the object inside the
image. This module is the read side: the generator refuses candidates whose target or anchors
are not in the report, and the verifier re-checks that nothing slipped through.

The reports are tracked in git (``data/benchmark/<scene>/visibility/``) precisely so the gate
does not need the bags -- ``just gen-cat2`` runs from a fresh clone, ``just visibility``
regenerates the reports when the camera model or the thresholds change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "data" / "benchmark"


def report_path(scene: str) -> Path:
    return BENCHMARK / scene / "visibility" / f"{scene}_visibility.json"


def load_visibility(scene: str) -> dict[str, Any] | None:
    """The scene's visibility report, or None when it has not been measured yet."""
    path = report_path(scene)
    if not path.exists():
        return None
    return json.loads(path.read_text())


class Visibility:
    """Per-object verdicts for one scene, with a no-op mode for un-measured scenes.

    Un-measured is not the same as invisible: without a report the honest thing is to let
    every candidate through and say so, rather than to silently empty the benchmark.
    """

    def __init__(self, report: dict[str, Any] | None, hidden: dict[str, str] | None = None):
        self.report = report
        self.objects: dict[str, Any] = (report or {}).get("objects", {})
        self.active = report is not None
        # Objects a human overruled the measurement on. The gate answers "did the sensors
        # reach it", which is not quite "can anything in the image tell it apart from what it
        # lies on": a flat dark object on a dark shelf returns lidar as well as anything and
        # is invisible in the frame. That residual is a review judgement, and this is where
        # it lands -- keyed by object, because dropping one phrasing only brings the object
        # back under the next one.
        self.hidden_by_review: dict[str, str] = {str(k): v for k, v in (hidden or {}).items()}

    def __bool__(self) -> bool:
        return self.active

    def visible(self, oid: str) -> bool:
        if str(oid) in self.hidden_by_review:
            return False
        if not self.active:
            return True
        return bool(self.objects.get(str(oid), {}).get("visible"))

    def reason(self, oid: str) -> str:
        if note := self.hidden_by_review.get(str(oid)):
            return f"hidden by review: {note}"
        if not self.active:
            return "unmeasured"
        entry = self.objects.get(str(oid))
        if entry is None:
            return "not_in_report"
        return entry.get("reason") or "visible"

    def view(self, oid: str) -> dict[str, Any] | None:
        """What the best frame of this object looked like, for the question record."""
        entry = self.objects.get(str(oid)) or {}
        best = entry.get("best_view")
        if not self.visible(oid) or not best:
            return None
        view = {
            "frames_visible": entry.get("frames_visible"),
            "distance_m": best.get("distance_m"),
            "px_area": best.get("px_area"),
            "occlusion": best.get("occlusion"),
            "elevation_deg": best.get("elevation_deg"),
            "stamp": best.get("stamp"),
        }
        if image := entry.get("view_image"):
            view["image"] = image
        return view

    def quality(self, oid: str) -> float:
        """How well the robot saw it, for ranking candidates that all pass the gate."""
        best = (self.objects.get(str(oid)) or {}).get("best_view") or {}
        return float(best.get("px_area", 0.0)) * (1.0 - float(best.get("occlusion", 0.0)))

    def hidden(self, oids: list[str]) -> list[tuple[str, str]]:
        """(object_id, reason) for each of `oids` the robot never resolved."""
        return [(str(o), self.reason(o)) for o in oids if not self.visible(o)]
