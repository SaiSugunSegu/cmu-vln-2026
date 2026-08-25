"""What the model is shown: the candidate table's roles, header and units.

Every assertion here stands for one of the three wrong-class answers in the 40-question
diagnosis, where an anchor's id was offered inside a candidate's fact rows with nothing
saying it was not answerable.

Imports `utils.*` the way `cat2_utils` does, so a container without `scripts/` bind-mounted
skips rather than fails.
"""
from __future__ import annotations

import pytest

from smart_vlm.cat2_utils import SOLVER_AVAILABLE

pytestmark = pytest.mark.skipif(not SOLVER_AVAILABLE, reason="scripts/utils not importable")

if SOLVER_AVAILABLE:
    import utils.objmap as objmap


def obj(oid, label, centre, size):
    """One map entry as an `Obj`, spelled the way a run with recorded prompts spells it."""
    return objmap.map_entry_to_obj(str(oid), {
        "label": label,
        "bbox3d": {"center": list(centre), "extent": list(size),
                   "rotation": [0.0, 0.0, 0.0, 1.0]},
    }, {})


def table_for(question, objects):
    """`shortlist` then `candidate_table`, wired as `select_object` wires them."""
    picked = objmap.shortlist(question, objects)
    return picked, objmap.candidate_table(
        picked["candidates"], picked["anchor_groups"], picked["relation"], limit=12,
        head=picked["head"], anchors=picked["all_anchors"],
        unmatched_anchors=picked["anchors_unmatched"])


def section(text, name):
    """The lines under one heading, up to the next blank line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(name))
    out = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        out.append(line)
    return out


@pytest.fixture
def shelf_scene():
    """A shelf with two vases on it, plus a cabinet the questions never mention."""
    return [
        obj(1, "shelf", (0.0, 0.0, 1.00), (1.60, 0.30, 0.05)),
        obj(2, "vase", (-0.50, 0.0, 1.15), (0.20, 0.20, 0.25)),
        obj(3, "vase", (0.50, 0.0, 1.15), (0.18, 0.18, 0.22)),
        obj(4, "cabinet", (3.00, 2.00, 0.45), (1.20, 0.50, 0.90)),
    ]


def test_the_head_noun_the_answer_must_be_is_stated(shelf_scene):
    _, text = table_for("Find the vase on the shelf.", shelf_scene)

    assert "The answer is a 'vase'." in text


def test_candidates_and_anchors_are_separate_sections(shelf_scene):
    # The three wrong-class answers all named an object that appears only as an anchor.
    _, text = table_for("Find the vase on the shelf.", shelf_scene)

    candidates = section(text, "CANDIDATES")
    anchors = section(text, "ANCHORS")
    assert [line.split()[0] for line in candidates if line.startswith("[")] == ["[2]", "[3]"]
    assert [line.split()[0] for line in anchors] == ["[1]"]
    assert "Never the answer" in text


def test_units_are_stated_once_rather_than_per_number(shelf_scene):
    _, text = table_for("Find the vase on the shelf.", shelf_scene)

    assert "All lengths are in metres." in text
    facts = [line for line in text.splitlines() if line.startswith("    to ")]
    assert facts, "the candidates should carry fact rows"
    assert not any(" m," in line or line.endswith(" m") for line in facts)


def test_the_precomputed_relation_flags_survive_the_rewrite(shelf_scene):
    # The flags are the whole reason the table exists: the model must never do the geometry.
    _, text = table_for("Find the vase on the shelf.", shelf_scene)

    assert "holds: on" in text


def test_an_anchor_the_map_does_not_hold_is_named(shelf_scene):
    # arabic_room Q01 lost its whole chain to an undetected `book` without ever saying so.
    _, text = table_for("Find the vase closest to the book on the shelf.", shelf_scene)

    assert "NOT DETECTED" in text
    assert "'book'" in text


def test_an_anchor_from_a_deeper_hop_is_listed_too(shelf_scene):
    """`anchor_groups` stops at the first hop, so the second hop's landmark was nameless.

    chinese_room Q01 answered `foldingscreen` to a bowl question, and the screen is named
    in the second hop of "the bowl on the table closest to the folding screen".
    """
    scene = shelf_scene + [obj(5, "folding screen", (2.0, 0.0, 0.85), (0.40, 1.80, 1.70))]

    picked, text = table_for(
        "Find the vase on the shelf closest to the folding screen.", scene)

    assert {a.id for a in picked["all_anchors"]} == {"1", "5"}
    assert [line.split()[0] for line in section(text, "ANCHORS")] == ["[1]", "[5]"]


def test_a_question_with_no_candidate_says_so_instead_of_heading_an_empty_list(shelf_scene):
    _, text = table_for("Find the bowl on the shelf.", shelf_scene)

    assert "CANDIDATES: none" in text
    assert "bowl" in text


def test_an_object_matched_by_two_phrases_is_listed_once(shelf_scene):
    picked, text = table_for("Find the vase between the shelf and the shelf.", shelf_scene)

    anchors = section(text, "ANCHORS")
    assert len(anchors) == len({line.split()[0] for line in anchors})
