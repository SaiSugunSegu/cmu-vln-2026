"""Unit tests for the SAM 3 -> ObjMapper detections translation.

Pure numpy/python — no GPU, no model, no ROS. Run with:
    python -m pytest ai_module/src/sam_mapper/tests/test_detections.py
"""
import numpy as np
import pytest

from sam_mapper.detections import PromptTable, default_label, to_detections
from sam_mapper.sam3_backend import Sam3FrameResult


OBJECTS = [
    {"prompt": "chair", "instance": True},
    {"prompt": "table", "instance": True},
    {"prompt": "potted plant", "instance": True},
    {"prompt": "wall", "instance": False},
    {"prompt": "floor", "instance": False},
]


def make_result(object_ids, scores, prompt_to_obj_ids, h=4, w=6):
    n = len(object_ids)
    return Sam3FrameResult(
        object_ids=np.array(object_ids, dtype=int),
        scores=np.array(scores, dtype=float),
        boxes=np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1)),
        masks=np.ones((n, h, w), dtype=bool),
        prompt_to_obj_ids=prompt_to_obj_ids,
    )


def test_default_label_strips_spaces_to_match_dimension_priors():
    # semantic_mapping's DIMENSION_PRIORS spells these without spaces.
    assert default_label("potted plant") == "pottedplant"
    assert default_label("fire extinguisher") == "fireextinguisher"
    assert default_label("Chair") == "chair"


def test_explicit_label_merges_prompts_into_one_class():
    table = PromptTable([
        {"prompt": "sofa", "instance": True},
        {"prompt": "loveseat", "instance": True, "label": "sofa"},
    ])
    assert {s.label for s in table.specs} == {"sofa"}


def test_duplicate_prompts_rejected():
    with pytest.raises(ValueError, match="duplicate prompts"):
        PromptTable([{"prompt": "chair"}, {"prompt": "chair"}])


def test_empty_objects_rejected():
    with pytest.raises(ValueError, match="empty"):
        PromptTable([])


def test_background_ids_are_negative_and_stable():
    table = PromptTable(OBJECTS)
    assert table.background_ids == {"wall": -1, "floor": -2}
    # Rebuilding must not renumber — ids have to survive across frames.
    assert PromptTable(OBJECTS).background_ids == table.background_ids


def test_background_ids_one_per_label_not_per_prompt():
    table = PromptTable([
        {"prompt": "wall", "instance": False},
        {"prompt": "white wall", "instance": False, "label": "wall"},
        {"prompt": "floor", "instance": False},
    ])
    assert table.background_ids == {"wall": -1, "floor": -2}


def test_label_template_shape_matches_objmapper_expectation():
    # ObjMapper.__init__ reads val["is_instance"] then replaces val with val["prompts"].
    template = PromptTable(OBJECTS).label_template()
    assert template["chair"] == {"is_instance": True, "prompts": ["chair"]}
    assert template["wall"]["is_instance"] is False
    for entry in template.values():
        assert set(entry) == {"is_instance", "prompts"}


def test_instance_ids_pass_through_and_background_is_renumbered():
    table = PromptTable(OBJECTS)
    result = make_result(
        object_ids=[7, 12, 3],
        scores=[0.9, 0.8, 0.7],
        prompt_to_obj_ids={"chair": [7], "table": [12], "wall": [3]},
    )
    det = to_detections(result, table)

    by_label = dict(zip(det["labels"], det["ids"]))
    assert by_label["chair"] == 7        # SAM 3 id preserved
    assert by_label["table"] == 12
    assert by_label["wall"] == -1        # renumbered onto the negative convention
    assert det["masks"].shape == (3, 4, 6)
    assert det["bboxes"].shape == (3, 4)


def test_multi_prompt_claim_resolves_to_highest_score():
    table = PromptTable(OBJECTS)
    # Object 5 is claimed by both "chair" and "table"; chair scored higher.
    result = make_result(
        object_ids=[5],
        scores=[0.91],
        prompt_to_obj_ids={"chair": [5], "table": [5]},
    )
    det = to_detections(result, table)
    assert len(det["ids"]) == 1
    assert det["labels"][0] == "chair"


def test_unclaimed_objects_are_dropped():
    table = PromptTable(OBJECTS)
    result = make_result(
        object_ids=[1, 2],
        scores=[0.9, 0.9],
        prompt_to_obj_ids={"chair": [1]},   # object 2 claimed by nobody
    )
    det = to_detections(result, table)
    assert det["ids"].tolist() == [1]


def test_prompt_outside_config_is_dropped():
    table = PromptTable(OBJECTS)
    result = make_result(
        object_ids=[1],
        scores=[0.9],
        prompt_to_obj_ids={"giraffe": [1]},
    )
    assert len(to_detections(result, table)["ids"]) == 0


def test_empty_frame_returns_well_formed_empty_dict():
    table = PromptTable(OBJECTS)
    det = to_detections(Sam3FrameResult.empty(4, 6), table)
    assert set(det) == {"bboxes", "confidences", "labels", "ids", "masks"}
    assert len(det["ids"]) == 0
    assert det["bboxes"].shape == (0, 4)


def test_dict_keys_and_dtypes_match_objmapper_contract():
    table = PromptTable(OBJECTS)
    det = to_detections(
        make_result([7], [0.9], {"chair": [7]}), table,
    )
    assert set(det) == {"bboxes", "confidences", "labels", "ids", "masks"}
    assert det["bboxes"].dtype == float
    assert det["confidences"].dtype == float
    assert det["ids"].dtype == int
    assert det["masks"].dtype == bool
    # update_map zips these five in lockstep — lengths must agree.
    assert len({len(det[k]) for k in det}) == 1
