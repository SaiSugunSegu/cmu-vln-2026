"""Pure-python tests for instruction waypoint selection."""
from __future__ import annotations

from smart_vlm.cat3_utils import (
    center_xy,
    heuristic_targets,
    label_matches,
    waypoints_from_map,
)


def _box(label: str, x: float, y: float, volume: float = 1.0) -> dict:
    side = volume ** (1.0 / 3.0)
    return {
        "label": label,
        "bbox3d": {"center": [x, y, 0.5], "extent": [side, side, side]},
    }


def test_label_matches_ignores_spaces_and_containment():
    assert label_matches("potted plant", "pottedplant")
    assert label_matches("chair", "dining chair")
    assert not label_matches("chair", "table")
    assert not label_matches("", "chair")


def test_center_xy_reads_bbox_midpoint():
    assert center_xy(_box("chair", 1.5, -2.0)) == (1.5, -2.0)
    assert center_xy({}) is None


def test_waypoints_follow_prompt_order_and_skip_repeats():
    raw = {
        "1": _box("chair", 0.0, 0.0, 1.0),
        "2": _box("chair", 4.0, 0.0, 8.0),
        "3": _box("fridge", 1.0, 3.0, 2.0),
    }
    wps = waypoints_from_map(raw, ["chair", "fridge", "chair"])
    assert [w["object_id"] for w in wps] == ["2", "3", "1"]
    assert (wps[0]["x"], wps[0]["y"]) == (4.0, 0.0)
    assert wps[1]["label"] == "fridge"


def test_waypoints_skip_unmatched_prompts():
    raw = {"1": _box("sofa", 2.0, 2.0)}
    assert waypoints_from_map(raw, ["window", "lamp"]) == []


def test_heuristic_targets_keeps_object_nouns():
    assert "window" in heuristic_targets("Take the path near the window to the fridge")
    assert "fridge" in heuristic_targets("Take the path near the window to the fridge")
    assert "take" not in heuristic_targets("Take the path near the window to the fridge")
