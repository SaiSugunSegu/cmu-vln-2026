"""MultiConceptMixin: the merge and its concept attribution.

Exercised against a stand-in base class, so no checkpoint, no GPU and no `sam3` install —
only torch, which the mixin uses for tensor ops.

The single-caption case earns its own test because its failure mode is invisible: the mixin
short-circuits, `_prompt` never gets attached, `mc_obj_concept` stays empty, every object
lands in the unlabelled bucket, and `to_detections` drops the lot. A one-target question then
runs SAM at full cost and publishes ZERO detections, with no error anywhere.

    python -m pytest ai_module/src/sam_mapper/tests/test_sam31_merge.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="mixin does tensor ops")

from sam_mapper.sam31_backend import MultiConceptMixin  # noqa: E402

QUERIES = 5


class FakeDetector:
    """Shaped like Sam3MultiplexBase's detection half: det_out batched over CAPTIONS."""

    def __init__(self, scores):
        self.scores = torch.tensor(scores, dtype=torch.float32)   # (captions, queries)

    def run_backbone_and_detection(self, **kwargs):
        n_captions, n_queries = self.scores.shape
        det_out = {
            "scores": self.scores,
            "bbox": torch.zeros(n_captions, n_queries, 4),
            "mask": torch.zeros(n_captions, n_queries, 2, 2),
        }
        return det_out, self.scores > 0.5

    def run_tracker_update_planning_phase(self, **kwargs):
        return dict(self.plan), {}


def merged(scores, max_dets=128):
    cls = type("Merged", (MultiConceptMixin, FakeDetector), {})
    model = cls(scores)
    model.mc_max_dets = max_dets
    model.mc_reset()
    return model


def test_single_caption_still_attaches_prompt():
    """Regression: without `_prompt` a one-prompt run publishes nothing at all."""
    model = merged([[0.9, 0.2, 0.7, 0.1, 0.6]])
    model.mc_concepts = ["chair"]
    det_out, keep = model.run_backbone_and_detection()

    assert "_prompt" in det_out, "single-caption path must still carry attribution"
    assert det_out["_prompt"].shape == keep.shape
    assert int(det_out["_prompt"].sum()) == 0, "one caption -> every row is prompt 0"


def test_single_caption_is_otherwise_untouched():
    """The merge must not filter or reorder when there is nothing to merge."""
    scores = [[0.9, 0.2, 0.7, 0.1, 0.6]]
    det_out, keep = merged(scores).run_backbone_and_detection()
    assert det_out["scores"].shape == (1, QUERIES)
    assert keep.tolist() == [[True, False, True, False, True]]


def test_multi_caption_concatenates_positives_of_every_caption():
    model = merged([[0.9, 0.1, 0.1, 0.1, 0.1],     # chair: 1 positive
                    [0.8, 0.7, 0.1, 0.1, 0.1],     # table: 2 positives
                    [0.1, 0.1, 0.1, 0.1, 0.1]])    # tv:    none
    det_out, keep = model.run_backbone_and_detection()

    assert det_out["scores"].shape == (1, 3), "batch collapses to 1, queries concatenate"
    assert bool(keep.all()), "only positives survive, so every kept row is True"
    # Sorted by score descending: chair 0.9, table 0.8, table 0.7
    assert det_out["_prompt"].reshape(-1).tolist() == [0, 1, 1]


def test_max_dets_caps_by_score_not_by_caption_order():
    model = merged([[0.6, 0.6, 0.6, 0.6, 0.6],
                    [0.99, 0.98, 0.1, 0.1, 0.1]], max_dets=2)
    det_out, _ = model.run_backbone_and_detection()
    assert det_out["scores"].reshape(-1).tolist() == pytest.approx([0.99, 0.98])
    assert det_out["_prompt"].reshape(-1).tolist() == [1, 1]


def test_no_positives_yields_an_empty_detection_set():
    """Upstream handles a zero-detection frame; the merge must produce one cleanly."""
    det_out, keep = merged([[0.1] * QUERIES, [0.2] * QUERIES]).run_backbone_and_detection()
    assert det_out["scores"].shape == (1, 0)
    assert keep.shape == (1, 0)


def test_attribution_maps_new_objects_to_their_concept():
    model = merged([[0.9, 0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.7, 0.1, 0.1, 0.1]])
    model.mc_concepts = ["chair", "table"]
    det_out, _ = model.run_backbone_and_detection()
    # The planning phase sees det_out AFTER the [0] squeeze upstream applies.
    squeezed = {k: v[0] for k, v in det_out.items()}
    model.plan = {"new_det_fa_inds": np.array([0, 2]),
                  "new_det_obj_ids": np.array([11, 12])}

    model.run_tracker_update_planning_phase(det_out=squeezed)

    assert model.mc_obj_concept == {11: "chair", 12: "table"}


def test_attribution_survives_the_upstream_permutation():
    """Upstream permutes det_out with a generic index_select over every key; `_prompt` must
    move with the rows or objects get labelled with another concept's caption."""
    model = merged([[0.9, 0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.7, 0.1, 0.1, 0.1]])
    model.mc_concepts = ["chair", "table"]
    det_out, _ = model.run_backbone_and_detection()

    order = torch.tensor([2, 0, 1])                      # any permutation
    permuted = {k: torch.index_select(v[0], 0, order) for k, v in det_out.items()}
    model.plan = {"new_det_fa_inds": np.array([0]),      # was row 2 -> table
                  "new_det_obj_ids": np.array([7])}

    model.run_tracker_update_planning_phase(det_out=permuted)

    assert model.mc_obj_concept == {7: "table"}
