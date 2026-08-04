"""Unit tests for run-id sanitising shared with smart_vlm's category-1 reasoner.

Like the other sam_mapper tests, this imports best_view, which pulls in cv2 —
run it inside the container (`just test`), not on a bare host.
"""
from __future__ import annotations

from sam_mapper.best_view import sanitize_run_id


def test_passes_through_safe_characters():
    assert sanitize_run_id("cat1_ab12_arabic_room") == "cat1_ab12_arabic_room"


def test_collapses_unsafe_runs_to_one_underscore():
    assert sanitize_run_id("How many glasses?") == "How_many_glasses"


def test_strips_path_separators():
    # A run id reaches us over ROS and becomes a directory name.
    assert "/" not in sanitize_run_id("../../etc/passwd")


def test_trims_leading_and_trailing_underscores():
    assert sanitize_run_id("  spaced  ") == "spaced"


def test_falls_back_when_nothing_survives():
    assert sanitize_run_id("???") == "run"
    assert sanitize_run_id("???", fallback="q") == "q"
