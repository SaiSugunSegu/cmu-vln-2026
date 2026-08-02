"""Pure-python tests for category1 helpers (no ROS / GPU)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_vlm.category1_utils import (  # noqa: E402
    extract_integer,
    heuristic_targets,
    parse_target_list,
)


def test_parse_target_list_json():
    assert parse_target_list('["carpet", "arabic jar"]') == ["carpet", "arabic jar"]


def test_parse_target_list_embedded():
    text = 'Sure, here you go:\n["glass", "coffee pot"]\n'
    assert parse_target_list(text) == ["glass", "coffee pot"]


def test_parse_target_list_rejects_digits():
    assert parse_target_list('["0"]') == []
    assert parse_target_list("[0]") == []


def test_extract_integer():
    assert extract_integer("There are 3 glasses.") == 3
    assert extract_integer("no number") is None


def test_heuristic_targets():
    assert heuristic_targets("How many carpets are in the scene?") == ["carpet"]
    assert heuristic_targets("How many glasses are near the arabic jar?")[0] == "glass"
