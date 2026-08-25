"""The anchor guard: an object the question names to locate its target is not the target.

Three of the 40 measured losses answered with an anchor — a folding screen to a bowl
question, a tv to a map question, a display ledge to a flowers question — and all three
came through the fallback that runs when nothing in the map matches the head noun. A prompt
rule cannot be a constraint, so the rule is here.
"""
from __future__ import annotations

import pytest

from smart_vlm.cat2_utils import SOLVER_AVAILABLE, answerable, naive_pick

pytestmark = pytest.mark.skipif(not SOLVER_AVAILABLE, reason="scripts/utils not importable")

if SOLVER_AVAILABLE:
    import utils.objmap as objmap


def obj(oid, label, centre, size):
    return objmap.map_entry_to_obj(str(oid), {
        "label": label,
        "bbox3d": {"center": list(centre), "extent": list(size),
                   "rotation": [0.0, 0.0, 0.0, 1.0]},
    }, {})


@pytest.fixture
def bowl_scene():
    """chinese_room Q01, reduced: no bowl was ever mapped, and the screen is the biggest box.

    The size ordering is the point. `naive_pick` ranks by volume, so with nothing of the
    right class in the map the anchor wins on its own merits unless it is excluded.
    """
    return {o.id: o for o in [
        obj(1, "table", (0.0, 0.0, 0.35), (1.20, 0.80, 0.70)),
        obj(2, "folding screen", (2.0, 0.0, 0.90), (0.40, 1.90, 1.80)),
        obj(3, "stool", (1.0, 1.0, 0.22), (0.35, 0.35, 0.45)),
    ]}


QUESTION = "Find the bowl on the table closest to the folding screen."


def test_the_fallback_does_not_answer_with_an_anchor(bowl_scene):
    picked = objmap.shortlist(QUESTION, list(bowl_scene.values()))
    anchors = {a.id for a in picked["all_anchors"]}
    assert anchors == {"1", "2"}, "both hops' landmarks have to be anchors, not just the first"

    chosen, why = naive_pick(QUESTION, bowl_scene, exclude=anchors)

    assert chosen == "3"          # the stool: the largest box that is not an anchor
    assert "anchor" in why


def test_without_the_guard_the_anchor_is_exactly_what_wins(bowl_scene):
    # Pins the defect the guard removes, so a regression is visible rather than inferred.
    chosen, _ = naive_pick(QUESTION, bowl_scene)

    assert chosen == "2"          # the folding screen, an anchor of the question


def test_the_fallback_no_longer_claims_a_class_it_did_not_find(bowl_scene):
    # It used to report "largest bowl in the map" having matched no bowl at all.
    _, why = naive_pick(QUESTION, bowl_scene, exclude={"1", "2"})

    assert "largest bowl" not in why


def test_a_class_match_still_beats_the_size_ranking(bowl_scene):
    scene = dict(bowl_scene)
    small_bowl = obj(4, "bowl", (0.0, 0.0, 0.75), (0.18, 0.18, 0.08))
    scene[small_bowl.id] = small_bowl

    chosen, why = naive_pick(QUESTION, scene, exclude={"1", "2"})

    assert chosen == "4"
    assert "largest bowl" in why


def test_a_map_of_nothing_but_anchors_answers_nothing(bowl_scene):
    chosen, why = naive_pick(QUESTION, bowl_scene, exclude=set(bowl_scene))

    assert chosen is None
    assert "anchor" in why


def test_an_anchor_among_the_candidates_is_dropped(bowl_scene):
    candidates = list(bowl_scene.values())

    kept = answerable(candidates, {"2"})

    assert [o.id for o in kept] == ["1", "3"]


def test_dropping_every_candidate_is_worse_than_keeping_a_bad_one(bowl_scene):
    # No box scores zero exactly as a wrong box does, so an empty shortlist buys nothing.
    candidates = list(bowl_scene.values())

    kept = answerable(candidates, {o.id for o in candidates})

    assert [o.id for o in kept] == [o.id for o in candidates]
