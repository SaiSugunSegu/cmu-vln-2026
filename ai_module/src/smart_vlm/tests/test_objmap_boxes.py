"""Class folding: singularising and synonym-merging a detector's raw labels.

Imports `utils.*` the way `cat2_utils` does, so a container without `scripts/` bind-mounted
skips rather than fails — the same degradation `SOLVER_AVAILABLE` describes.
"""
from __future__ import annotations

import pytest

from smart_vlm.cat2_utils import SOLVER_AVAILABLE

pytestmark = pytest.mark.skipif(not SOLVER_AVAILABLE, reason="scripts/utils not importable")

if SOLVER_AVAILABLE:
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
