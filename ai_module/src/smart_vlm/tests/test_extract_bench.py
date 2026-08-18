"""Pure-python tests for extract_bench scoring (no ROS / GPU / network)."""
from __future__ import annotations

from smart_vlm.extract_bench import (
    categories_from_arg,
    noun_covers,
    parse_args,
    score_extraction,
    squash_label,
)


def test_squash_label_drops_spaces_and_case():
    assert squash_label("Potted Plant") == "pottedplant"
    assert squash_label("trash can") == "trashcan"


def test_noun_covers_matches_color_stripped_gt():
    """The extract prompt drops color, so 'trash can' still covers 'black trash can'."""
    assert noun_covers("trash can", "black trash can")
    assert noun_covers("pillow", "pillow with black stripes")
    assert not noun_covers("chair", "table")


def test_score_extraction_exact_set_match():
    scored = score_extraction(["Pillow", "couch"], ["couch", "pillow"])
    assert scored["exact"] is True
    assert scored["coverage"] == 1.0
    assert scored["precision"] == 1.0
    assert scored["extra"] == []
    assert scored["missing"] == []


def test_score_extraction_coverage_when_color_is_dropped():
    scored = score_extraction(["trash can"], ["black trash can", "blue trash can"])
    assert scored["exact"] is False
    assert scored["coverage"] == 1.0
    assert scored["missing"] == []
    assert scored["pred"] == ["trash can"]


def test_score_extraction_missing_and_extra():
    scored = score_extraction(["chair", "lamp"], ["chair", "desk"])
    assert scored["exact"] is False
    assert scored["hit"] == ["chair"]
    assert scored["extra"] == ["lamp"]
    assert scored["missing"] == ["desk"]
    assert scored["coverage"] == 0.5
    assert scored["precision"] == 0.5


def test_score_extraction_empty_gt_is_full_coverage():
    scored = score_extraction([], [])
    assert scored["exact"] is True
    assert scored["coverage"] == 1.0
    assert scored["precision"] == 1.0


def test_categories_from_arg():
    assert categories_from_arg("all") == (1, 2)
    assert categories_from_arg("1") == (1,)
    assert categories_from_arg("2") == (2,)


def test_parse_args_backend_defaults_to_cloud():
    args = parse_args([])
    assert args.backend == "cloud"


def test_parse_args_backend_local():
    args = parse_args(["--backend", "local", "--category", "1"])
    assert args.backend == "local"
    assert args.category == "1"
