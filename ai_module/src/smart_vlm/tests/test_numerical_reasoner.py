"""Pure-python tests for numerical helpers (no ROS / GPU)."""
from __future__ import annotations

import json

import pytest

from smart_vlm.numerical_utils import (
    clean_targets,
    extract_integer,
    heuristic_targets,
    select_context_views,
)
from smart_vlm.report_utils import summarise, write_report


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


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A crop directory that secure_path will accept.

    The real allowed roots are the container's /data and /home/docker mounts, and a
    pytest tmp_path is under neither, so the confinement has to be pointed at the
    fixture rather than switched off — a test that bypassed it would not exercise the
    same code path the reasoner takes.
    """
    from captioner import paths

    monkeypatch.setattr(paths, "ALLOWED_ROOTS", (tmp_path.resolve(),))
    for rank in (1, 2, 3):
        (tmp_path / f"best_rank{rank}_sofa.png").write_bytes(b"png")
    return tmp_path


def _manifest(*names):
    return {"selected": [{"rank": i, "file": n} for i, n in enumerate(names, start=1)]}


def test_select_context_views_takes_the_top_ranks_in_order(run_dir):
    views = select_context_views(
        run_dir, _manifest("best_rank1_sofa.png", "best_rank2_sofa.png",
                           "best_rank3_sofa.png"), 2)
    assert [p.name for p in views] == ["best_rank1_sofa.png", "best_rank2_sofa.png"]


def test_select_context_views_skips_a_rank_whose_image_is_gone(run_dir):
    """The manifest is rewritten on every flush, so it can name a file mid-delete."""
    views = select_context_views(
        run_dir, _manifest("best_rank1_sofa.png", "vanished.png"), 3)
    assert [p.name for p in views] == ["best_rank1_sofa.png"]


def test_select_context_views_handles_an_empty_selection(run_dir):
    """SAM found nothing: the caller answers 0 rather than crashing."""
    assert select_context_views(run_dir, {"selected": []}, 3) == []
    assert select_context_views(run_dir, {}, 3) == []


def test_select_context_views_rejects_a_traversing_filename(run_dir):
    with pytest.raises(ValueError):
        select_context_views(run_dir, _manifest("../escape.png"), 1)


def test_summary_counts_errors_separately_from_wrong_answers():
    results = [
        {"scene": "a", "correct": True, "time_taken_s": 2.0, "error": None},
        {"scene": "a", "correct": False, "time_taken_s": 4.0, "error": None},
        {"scene": "b", "correct": False, "time_taken_s": 90.0, "error": "TimeoutError: x"},
    ]
    summary = summarise(results)
    assert summary["correct"] == 1
    assert summary["accuracy"] == round(1 / 3, 4)
    assert summary["errors"] == 1
    # The timed-out row would otherwise pull the mean toward the phase budget.
    assert summary["mean_time_s"] == 3.0
    assert summary["per_scene"]["b"]["accuracy"] == 0.0


def test_summary_of_nothing_does_not_divide_by_zero():
    assert summarise([])["accuracy"] == 0.0
    assert summarise([])["mean_time_s"] is None


def test_write_report_merges_extra_into_the_summary(tmp_path):
    path = tmp_path / "bench.json"
    write_report(path, [{"scene": "a", "correct": True, "time_taken_s": 1.0,
                         "error": None}], {"model": "some-model", "views": 3})
    written = json.loads(path.read_text())
    assert written["summary"]["model"] == "some-model"
    assert written["summary"]["correct"] == 1
    assert len(written["results"]) == 1
