"""Class folding and legacy box un-padding, the two halves of what makes a map's boxes usable.

Imports `utils.*` the way `cat2_utils` does, so a container without `scripts/` bind-mounted
skips rather than fails — the same degradation `SOLVER_AVAILABLE` describes.
"""
from __future__ import annotations

import json

import pytest

from smart_vlm.cat2_utils import SOLVER_AVAILABLE

pytestmark = pytest.mark.skipif(not SOLVER_AVAILABLE, reason="scripts/utils not importable")

if SOLVER_AVAILABLE:
    import utils.objmap as objmap
    from utils.geometry import norm_class


# ---------------------------------------------------------------- class folding


@pytest.mark.parametrize("plural, singular", [
    # The two the rule adds over the hand-written alias entries.
    ("flowers", "flower"),
    ("drawers", "drawer"),
    # Previously an alias entry each; the rule has to keep covering them.
    ("books", "book"),
    ("curtains", "curtain"),
    ("windows", "window"),
    ("shelves", "shelf"),
    ("boxes", "box"),
    ("dishes", "dish"),
])
def test_a_plural_folds_onto_its_singular(plural, singular):
    assert norm_class(plural) == norm_class(singular)


@pytest.mark.parametrize("word", ["glass", "dress", "mattress", "cactus", "tennis", "bus"])
def test_a_word_merely_ending_in_s_is_left_alone(word):
    # Stripping one of these would invent a class ("glas") that matches nothing.
    assert norm_class(word) == word


def test_a_plural_still_reaches_its_synonym_group():
    # "couches" -> "couch" -> the group's canonical, in one call rather than two rules.
    assert norm_class("couches") == norm_class("sofa")


def test_folding_does_not_merge_two_real_classes():
    for a, b in [("table", "cabinet"), ("chair", "stool"), ("bowl", "vase"),
                 ("book", "shelf"), ("picture", "window")]:
        assert norm_class(a) != norm_class(b)


# ---------------------------------------------------------------- legacy boxes


def test_unpadding_recovers_the_extent_the_clamp_would_have_written():
    voxel = 0.05
    # A book: raw span 0.20 x 0.15 x 0.03, published by the old rule as span + voxel.
    span = [0.20, 0.15, 0.03]
    legacy = [v + voxel for v in span]

    recovered = objmap.unpad_legacy_extent(legacy, voxel)

    assert recovered == pytest.approx([max(v, voxel) for v in span])


def test_unpadding_never_returns_a_degenerate_extent():
    # A one-voxel-thick window: the old rule wrote exactly `voxel` on the thin axis, so
    # subtracting it lands on zero. The floor is what stops a zero-volume candidate.
    assert objmap.unpad_legacy_extent([0.05, 0.60, 0.80], 0.05)[0] == pytest.approx(0.05)


def _write_map(tmp_path, schema=None):
    entry = {"label": "book", "id": [3], "center": [0.0, 0.0, 0.0],
             "bbox3d": {"center": [0.0, 0.0, 0.0], "extent": [0.25, 0.20, 0.08],
                        "rotation": [0.0, 0.0, 0.0, 1.0]}}
    raw = {"3": entry}
    if schema is not None:
        raw["_schema"] = schema
    path = tmp_path / "obj_map.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_a_map_with_no_schema_marker_is_treated_as_padded(tmp_path):
    obj = objmap.load_obj_map(_write_map(tmp_path))["3"]

    assert list(obj.size) == pytest.approx([0.20, 0.15, 0.05])


def test_a_map_written_by_the_current_mapper_is_left_alone(tmp_path):
    path = _write_map(tmp_path, schema={"box_extent": "voxel_floor", "voxel_size": 0.05})

    obj = objmap.load_obj_map(path)["3"]

    assert list(obj.size) == pytest.approx([0.25, 0.20, 0.08])


def test_the_schema_entry_is_not_loaded_as_an_object(tmp_path):
    path = _write_map(tmp_path, schema={"box_extent": "voxel_floor", "voxel_size": 0.05})

    assert list(objmap.load_obj_map(path)) == ["3"]
