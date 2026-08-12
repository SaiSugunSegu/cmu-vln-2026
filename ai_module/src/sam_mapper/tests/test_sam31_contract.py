"""Pins the facebookresearch/sam3 internals Sam31Backend depends on.

`sam3` is a pinned third-party package (Dockerfile SAM31_SHA), not our code, and the backend
reaches past its public demo API in three ways. Each one fails silently rather than loudly if
upstream moves, so each gets a test:

  * MERGE      MultiConceptMixin overrides two PUBLIC methods by MRO. If either is renamed,
               `super()` still resolves but our override is never called — the model quietly
               reverts to single-concept and 5 of 6 prompts stop producing objects.
  * STREAMING  we drive `_run_single_frame_inference` / `_postprocess_output` per frame
               instead of `propagate_in_video`. A changed signature is an immediate crash;
               a changed DEFAULT is worse — see the filters test.
  * FILTERS    `_postprocess_output` hides removed / suppressed / unconfirmed masklets only
               when passed those ids. We once omitted them and shipped ~0.9 extra phantom
               objects per frame with no error anywhere.

Everything here is inspection only — no checkpoint, no GPU, no model built — so it runs in
CI in under a second. Skipped wholesale when `sam3` is not installed.

    python -m pytest ai_module/src/sam_mapper/tests/test_sam31_contract.py
"""
import inspect

import pytest

sam3 = pytest.importorskip("sam3", reason="facebookresearch/sam3 not installed")

from sam3.model.sam3_multiplex_base import Sam3MultiplexBase          # noqa: E402
from sam3.model.sam3_multiplex_tracking import (                      # noqa: E402
    Sam3MultiplexTracking,
    Sam3MultiplexTrackingWithInteractivity,
)

from sam_mapper.sam31_backend import Sam31Backend                     # noqa: E402

# What build_sam3_multiplex_video_predictor actually returns. Test THIS, not the base: it
# overrides init_state / reset_state / propagate_in_video / add_prompt, so attribute lookups
# through the MRO are what production runs. It does NOT override the per-frame primitives we
# call, and that fact is itself worth pinning.
Built = Sam3MultiplexTrackingWithInteractivity


def params(func) -> set:
    return set(inspect.signature(func).parameters)


# -- MERGE: the two methods MultiConceptMixin overrides --------------------------------

@pytest.mark.parametrize("name", ["run_backbone_and_detection",
                                  "run_tracker_update_planning_phase"])
def test_overridden_methods_are_public_and_present(name):
    """A rename turns our override into dead code that never runs — no exception, just
    single-concept behaviour returning."""
    assert hasattr(Sam3MultiplexBase, name), (
        f"{name} is gone from Sam3MultiplexBase; MultiConceptMixin's override would be "
        f"silently dead and the merge would stop happening")
    assert not name.startswith("_"), "override target must stay public for MRO to be honest"


def test_detection_batch_assert_still_exists():
    """The whole reason the merge is needed: upstream squeezes the caption batch and asserts
    it is 1. If this assert disappears upstream may have added native multi-concept support,
    and the merge should be re-evaluated rather than kept."""
    source = inspect.getsource(Sam3MultiplexBase._det_track_one_frame_impl)
    assert "pos_pred_mask.shape[0] == 1" in source
    assert "det_out[k][0] for k in det_out" in source


def test_det_out_is_permuted_generically():
    """Attribution rides on this: we add a `_prompt` key to det_out and rely on the
    permutation being a loop over ALL keys. If upstream hardcodes bbox/mask/scores, `_prompt`
    stops tracking its rows and every object gets the wrong concept label."""
    source = inspect.getsource(Sam3MultiplexBase._det_track_one_frame_impl)
    assert "index_select(det_out[k], dim=0, index=pos_pred_mask_idx)" in source
    assert "for k in det_out" in source


def test_update_plan_exposes_new_object_attribution():
    """obj_id -> concept is recovered from these two keys, paired positionally."""
    source = inspect.getsource(Sam3MultiplexBase.run_tracker_update_planning_phase)
    assert '"new_det_fa_inds"' in source
    assert '"new_det_obj_ids"' in source


# -- STREAMING: the per-frame primitives we call instead of propagate_in_video ----------

def test_single_frame_primitive_signature():
    got = params(Built._run_single_frame_inference)
    assert {"inference_state", "frame_idx", "reverse"} <= got


def test_propagate_is_just_a_loop_over_the_primitive():
    """Our streaming loop is only legitimate because propagate_in_video does the same thing
    per frame. If that stops being true, the offline and streaming paths have diverged."""
    assert "_run_single_frame_inference" in inspect.getsource(
        Sam3MultiplexTracking.propagate_in_video)


@pytest.mark.parametrize("name", ["_postprocess_output", "_init_backbone_out",
                                  "_construct_initial_input_batch", "init_state",
                                  "_run_single_frame_inference"])
def test_private_helpers_still_exist(name):
    assert hasattr(Built, name)


@pytest.mark.parametrize("name", ["_run_single_frame_inference", "_postprocess_output",
                                  "_construct_initial_input_batch"])
def test_per_frame_primitives_are_not_overridden_by_the_interactivity_subclass(name):
    """The built class overrides propagate_in_video with an interactive variant that passes
    only `suppressed_obj_ids` (:2473). If it ever starts overriding the primitives we call,
    our streaming path silently changes semantics under us."""
    assert getattr(Built, name) is getattr(Sam3MultiplexTracking, name)


def test_preallocation_helper_takes_images():
    """The backend reuses one preallocated frame buffer across sessions by re-laying the
    per-frame scaffolding over it; that needs (inference_state, images)."""
    assert {"inference_state", "images"} <= params(Built._construct_initial_input_batch)


# -- FILTERS: the output-suppression contract we got wrong once ------------------------

def test_postprocess_accepts_the_three_suppression_lists():
    """Without these, removed / suppressed / unconfirmed masklets are emitted as real
    objects. They default to None, so omitting them is silent."""
    assert {"removed_obj_ids", "suppressed_obj_ids", "unconfirmed_obj_ids"} <= params(Built._postprocess_output)


def test_single_frame_output_carries_the_suppression_ids():
    """`_run_single_frame_inference` is where we read them from."""
    source = inspect.getsource(Built._run_single_frame_inference)
    for key in ("removed_obj_ids", "suppressed_obj_ids", "unconfirmed_obj_ids"):
        assert f'"{key}"' in source


def test_cached_frame_outputs_is_still_unevicted():
    """The backend pops cached_frame_outputs[t] every frame because upstream never does — at
    ~18 MB/frame that is 4.7 GB over one 256-frame session. If upstream starts evicting, our
    pop becomes redundant (harmless); if the dict is renamed, the leak returns silently."""
    source = inspect.getsource(Built._cache_frame_outputs)
    assert 'inference_state["cached_frame_outputs"][frame_idx]' in source


# -- CONFIG: every knob we expose must exist upstream ----------------------------------

def test_every_tunable_is_a_real_constructor_parameter():
    """A typo or an upstream rename in TUNABLES is otherwise invisible: _apply_override logs
    and moves on, so the yaml key silently does nothing."""
    accepted = params(Sam3MultiplexBase.__init__)
    unknown = sorted(k for k in Sam31Backend.TUNABLES if k not in accepted)
    assert not unknown, f"TUNABLES not accepted by Sam3MultiplexBase.__init__: {unknown}"


def test_batched_grounding_switch_still_exists():
    """We force it off — batching over FRAMES needs future frames we do not have."""
    assert "use_batched_grounding" in params(Sam3MultiplexBase.__init__)


def test_tracking_bounds_is_how_prefetch_is_suppressed():
    """The backend writes feature_cache['tracking_bounds'] to collapse the detector's valid
    range onto the current frame, stopping the one-frame lookahead prefetch."""
    source = inspect.getsource(Sam3MultiplexBase.run_backbone_and_detection)
    assert '"tracking_bounds"' in source
    assert "max_frame_num_to_track" in source
