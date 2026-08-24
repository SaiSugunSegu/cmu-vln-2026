"""Unit tests for the loss cascade: which single cause a lost question is charged to."""
from __future__ import annotations

from smart_vlm.cat2_bench import HIT_IOU, loss_attribution, loss_bucket


def _row(**over):
    """A row as score_one writes it, defaulted to a clean hit."""
    return {
        "error": None,
        "correct": True,
        "ceiling_iou": 0.6,
        "ceiling_object_id": "3",
        "predicted_object_id": "3",
        "ceiling_rank": 1,
        "selection_source": "solver",
        **over,
    }


def test_a_hit_is_not_a_loss():
    assert loss_bucket(_row()) == "hit"


def test_nothing_in_the_map_overlaps_the_answer():
    assert loss_bucket(_row(correct=False, ceiling_iou=0.0,
                            ceiling_object_id=None)) == "perception"


def test_the_object_is_mapped_but_its_box_is_too_loose():
    # The A-i case: overlap exists, so detection worked and only the sizing lost the point.
    assert loss_bucket(_row(correct=False, ceiling_iou=HIT_IOU / 2)) == "box_near_miss"


def test_the_shortlist_never_offered_the_answer():
    assert loss_bucket(_row(correct=False, predicted_object_id="9",
                            ceiling_rank=None)) == "shortlist_miss"


def test_a_reachable_answer_the_model_ranked_past():
    assert loss_bucket(_row(correct=False, predicted_object_id="9", ceiling_rank=2,
                            selection_source="vlm")) == "wrong_pick_vlm"


def test_a_reachable_answer_the_geometry_ranked_past():
    assert loss_bucket(_row(correct=False, predicted_object_id="9",
                            ceiling_rank=2)) == "wrong_pick_solver"


def test_no_candidate_was_chosen_at_all():
    assert loss_bucket(_row(correct=False, predicted_object_id=None)) == "no_pick"


def test_an_error_outranks_every_other_cause():
    # Including `correct`, which a row that raised never got the chance to set truthfully.
    assert loss_bucket(_row(error="ValueError: boom", correct=False,
                            ceiling_iou=0.0)) == "error"


def test_attribution_counts_every_row_once_and_drops_empty_buckets():
    rows = [_row(), _row(), _row(correct=False, ceiling_iou=0.0, ceiling_object_id=None)]

    counts = loss_attribution(rows)

    assert counts == {"hit": 2, "perception": 1}
    assert sum(counts.values()) == len(rows)
