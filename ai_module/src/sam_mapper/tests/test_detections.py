"""Unit tests for the SAM 3 -> ObjMapper detections translation.

Pure numpy/python — no GPU, no model, no ROS. Run with:
    python -m pytest ai_module/src/sam_mapper/tests/test_detections.py
"""
import numpy as np
import pytest

from sam_mapper.detections import (PromptTable, build_id_map, default_label,
                                   encode_instance_id, to_detections)
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


def test_dropped_rows_select_the_correct_masks_and_boxes():
    """to_detections selects surviving rows out of the backend arrays instead of appending
    per-object copies. Every other test here uses identical masks and boxes for all rows, so
    a row-selection bug would sail straight through them — this one gives each row a
    distinguishable value."""
    table = PromptTable(OBJECTS)
    result = make_result(
        object_ids=[10, 11, 12],
        scores=[0.5, 0.6, 0.7],
        prompt_to_obj_ids={"chair": [10], "table": [12]},   # 11 unclaimed -> dropped
    )
    result.boxes[:] = np.arange(12, dtype=float).reshape(3, 4)
    for row in range(3):
        result.masks[row] = False
        result.masks[row, row, 0] = True                     # row r marks pixel (r, 0)

    det = to_detections(result, table)

    assert det["ids"].tolist() == [10, 12]
    assert det["confidences"].tolist() == [0.5, 0.7]
    assert det["bboxes"].tolist() == [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]
    assert det["masks"][0][0, 0] and not det["masks"][0][1, 0]
    assert det["masks"][1][2, 0] and not det["masks"][1][1, 0]


def test_keep_all_path_matches_the_selecting_path():
    """When nothing is dropped, masks pass through without a copy. The values must be
    identical to what the row-selecting branch would produce."""
    table = PromptTable(OBJECTS)
    result = make_result([1, 2], [0.9, 0.8], {"chair": [1], "table": [2]})
    result.masks[0, 0, 0] = False

    det = to_detections(result, table)
    assert np.array_equal(det["masks"], result.masks)
    assert det["masks"].dtype == bool


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


def _id_map_reference(ids, masks, height, width):
    """The per-object loop build_id_map replaces. The vectorised version must agree with it
    pixel for pixel, tie-breaks included."""
    id_map = np.zeros((height, width), dtype=np.uint16)
    for obj_id, mask in zip(ids, masks):
        id_map[mask] = encode_instance_id(int(obj_id))
    return id_map


def test_build_id_map_matches_the_per_object_loop():
    rng = np.random.default_rng(0)
    ids = np.array([0, 5, -1, 12], dtype=int)          # instance ids and one background id
    masks = rng.random((4, 12, 20)) > 0.6              # deliberately overlapping
    assert np.array_equal(build_id_map(ids, masks, 12, 20),
                          _id_map_reference(ids, masks, 12, 20))


def test_build_id_map_gives_overlaps_to_the_last_entry():
    """The loop's semantics: later entries overwrite earlier ones. Losing this would silently
    reassign shared pixels to a different object in map_node."""
    ids = np.array([3, 7], dtype=int)
    masks = np.ones((2, 2, 2), dtype=bool)             # total overlap
    assert (build_id_map(ids, masks, 2, 2) == encode_instance_id(7)).all()


def test_build_id_map_leaves_unmasked_pixels_at_zero():
    ids = np.array([4], dtype=int)
    masks = np.zeros((1, 3, 3), dtype=bool)
    masks[0, 1, 1] = True
    id_map = build_id_map(ids, masks, 3, 3)
    assert id_map[1, 1] == encode_instance_id(4)
    assert id_map.sum() == encode_instance_id(4)       # 0 elsewhere == "no detection"


def test_build_id_map_with_no_detections_is_all_zero():
    id_map = build_id_map(np.zeros(0, dtype=int), np.zeros((0, 0, 0), dtype=bool), 5, 6)
    assert id_map.shape == (5, 6) and id_map.dtype == np.uint16 and not id_map.any()


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
