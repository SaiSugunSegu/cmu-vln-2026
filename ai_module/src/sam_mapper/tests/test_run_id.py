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


def test_keeps_a_slash_as_a_directory_level():
    """The eval sweep's <scene>/<question id> layout depends on this."""
    assert sanitize_run_id("arabic_room/Q01") == "arabic_room/Q01"


def test_sanitises_each_level_independently():
    assert sanitize_run_id("arabic room/How many?") == "arabic_room/How_many"


def test_cannot_walk_out_of_the_output_directory():
    """A run id reaches us over ROS and is joined onto the crops root.

    Dots survive the character filter, so `..` has to be dropped as a component or the
    result would climb a level per pair — the one case where allowing separators would
    otherwise be worse than stripping them.
    """
    result = sanitize_run_id("../../etc/passwd")
    assert ".." not in result.split("/")
    assert not result.startswith("/")

    assert not sanitize_run_id("/absolute/path").startswith("/")
    assert sanitize_run_id("..") == "run"


def test_trims_leading_and_trailing_underscores():
    assert sanitize_run_id("  spaced  ") == "spaced"


def test_falls_back_when_nothing_survives():
    assert sanitize_run_id("???") == "run"
    assert sanitize_run_id("???", fallback="q") == "q"
    assert sanitize_run_id("//") == "run"
