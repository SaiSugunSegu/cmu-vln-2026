"""Pure-python tests for numerical helpers (no ROS / GPU)."""
from __future__ import annotations

from smart_vlm.numerical_utils import (
    clean_targets,
    extract_integer,
    heuristic_targets,
)


def test_clean_targets_normalises_and_dedupes():
    assert clean_targets(["Glass", " glass ", "Arabic Jar"]) == ["glass", "arabic jar"]


def test_clean_targets_rejects_digits_and_blanks():
    """A structured reply can still carry the numerical wrapper's stray "0"."""
    assert clean_targets(["0", "", "  ", "chair"]) == ["chair"]


def test_extract_integer():
    assert extract_integer("There are 3 glasses.") == 3
    assert extract_integer("no number") is None


def test_heuristic_targets():
    assert heuristic_targets("How many carpets are in the scene?") == ["carpet"]
    assert heuristic_targets("How many glasses are near the arabic jar?")[0] == "glass"
