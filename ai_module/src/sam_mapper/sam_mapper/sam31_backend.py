"""Native SAM 3.1 (Object Multiplex) behind the same protocol as Sam3Backend.

Measured against SAM 3 on 40 livingroom_1 frames, 6 concepts, L4 (scripts/eval/sam31_probe.py):

    stage            SAM 3    SAM 3.1
    encoder/backbone   179        221     3.1 pays +42 ms every frame
    detection          147        255     3.1 is 1.7x heavier per concept
    tracker            537         70     3.1 wins 7.7x  <- the whole reason to switch
    TOTAL              920        557     1.65x
    mask agreement            0.985 of SAM 3's objects at IoU >= 0.5

Why this file exists at all: upstream is single-concept. `add_prompt` holds one caption and
calls reset_state on a second call, and `_det_track_one_frame_impl` asserts the detection
batch is 1 (sam3_multiplex_base.py:541-543). Running one session per concept costs 3.1x. The
mixin below is the missing merge -- N captions through ONE backbone pass, their positives
concatenated into one detection set the concept-agnostic tracker accepts unchanged.
"""
from __future__ import annotations

import os
import time

import numpy as np

from sam_mapper.profiling import StageTimer
from sam_mapper.sam3_backend import Sam3FrameResult

# Same seams the probe measures, so a production profile is comparable to the bench.
SAM31_STAGES = (
    ("run_backbone_and_detection", "detect(+backbone)"),
    ("run_tracker_propagation", "tracker_propagate"),
    ("run_tracker_update_planning_phase", "tracker_plan"),
    ("run_tracker_update_execution_phase", "tracker_execute"),
    ("build_outputs", "build_outputs"),
)


class MultiConceptMixin:
    """N concepts -> one backbone pass -> one tracker.

    Both methods below are PUBLIC upstream methods, so this is an ordinary MRO override, not
    a monkeypatch: mixed in ahead of the model's class, `super()` reaches the original. The
    two things it does:

    * merge -- upstream's detector already batches over captions, but the tracker stage
      asserts that batch is 1. Concatenating each caption's positives along the query axis
      is the whole fix; the tracker tracks masklets and does not know what a caption is, and
      its other input (the SAM 2 backbone feature) is batch-1 no matter how many captions
      ride along, because _get_img_feats forwards the backbone on unique img_ids only.
    * attribution -- `det_out` is permuted by a generic index_select over `for k in det_out`
      (sam3_multiplex_base.py:549-551), so an extra `_prompt` column survives the permutation
      and new_det_fa_inds -> new_det_obj_ids (:1410-1412) turns it into obj_id -> concept.
      ObjMapper needs that to label objects.
    """

    #: set by the backend before any frame is processed
    mc_concepts: list[str] = []
    mc_max_dets: int = 128

    def mc_reset(self) -> None:
        self.mc_obj_concept: dict[int, str] = {}

    def run_backbone_and_detection(self, *args, **kwargs):
        import torch

        det_out, pos = super().run_backbone_and_detection(*args, **kwargs)
        if pos.shape[0] == 1:
            # One caption: nothing to merge, but `_prompt` must still be attached or
            # attribution stays empty, every object falls into the unlabelled bucket, and
            # to_detections() drops the lot — a one-target question would publish NOTHING
            # while paying full inference cost.
            det_out["_prompt"] = torch.zeros_like(pos, dtype=torch.long)
            return det_out, pos
        idx = pos.nonzero(as_tuple=False)           # (caption, query) pairs above threshold
        scores = det_out["scores"][idx[:, 0], idx[:, 1]]
        idx = idx[scores.argsort(descending=True)[: self.mc_max_dets]]
        merged = {k: v[idx[:, 0], idx[:, 1]].unsqueeze(0) for k, v in det_out.items()}
        merged["_prompt"] = idx[:, 0].unsqueeze(0)
        return merged, torch.ones_like(merged["_prompt"], dtype=torch.bool)

    def run_tracker_update_planning_phase(self, *args, **kwargs):
        import torch

        plan, metadata = super().run_tracker_update_planning_phase(*args, **kwargs)
        det_out = kwargs.get("det_out")
        prompts = det_out.get("_prompt") if det_out is not None else None
        fa_inds, obj_ids = plan.get("new_det_fa_inds"), plan.get("new_det_obj_ids")
        if prompts is not None and fa_inds is not None and len(fa_inds):
            sel = torch.as_tensor(np.asarray(fa_inds), dtype=torch.long,
                                  device=prompts.device)
            for obj_id, pid in zip(np.asarray(obj_ids).tolist(), prompts[sel].tolist()):
                self.mc_obj_concept[int(obj_id)] = self.mc_concepts[int(pid)]
        return plan, metadata


def quiet() -> None:
    """Silence sam3's own console output. Must run BEFORE `import sam3`.

    Two separate sources, neither reachable by ordinary logging config:
      * get_logger() sets each logger's level at import and sets propagate=False
        (logger.py:41-56), so raising a parent logger does nothing — but it reads $LOG_LEVEL.
      * the builder print()s every missing key of both state-dict loads (model_builder.py
        :809, :1062, :1225) — thousands of lines. Those are captured in build_predictor.
    """
    import logging

    os.environ.setdefault("LOG_LEVEL", "WARNING")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if name.startswith("sam3") and isinstance(logger, logging.Logger):
            logger.setLevel(logging.WARNING)


def build_predictor(checkpoint: str, *, max_num_objects: int = 32,
                    multiplex_count: int = 16, use_fa3: bool = False,
                    compile: bool = False, log=print):
    """Build the multiplex predictor and audit the non-strict state-dict load.

    The builder loads twice (model_builder.py:1120 then :1222): first into the bare tracker
    BEFORE the sam3_model.->detector. key remap, which misses ~900 keys and means nothing,
    then into the assembled model. Only the second is worth judging — so hold the builder's
    own key dumps and replay them only if the audit finds a real problem.
    """
    import contextlib
    import io

    import torch

    quiet()
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    captured: list[tuple[str, list, list]] = []
    original = torch.nn.Module.load_state_dict

    def recording(self, state_dict, *args, **kwargs):
        result = original(self, state_dict, *args, **kwargs)
        try:
            captured.append((type(self).__name__,
                             list(result.missing_keys), list(result.unexpected_keys)))
        except AttributeError:            # strict=True returns None on some versions
            pass
        return result

    noise = io.StringIO()
    torch.nn.Module.load_state_dict = recording
    try:
        with contextlib.redirect_stdout(noise):
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint,
                # FA3 is Hopper-only (sm_90); both deploy targets are Ada sm_89. The
                # else-branch is a clean SDPA path (model_misc.py:397-410).
                use_fa3=use_fa3, compile=compile, warm_up=False,
                max_num_objects=max_num_objects, multiplex_count=multiplex_count,
            )
    finally:
        torch.nn.Module.load_state_dict = original

    # freqs_cis* are register_buffers recomputed in __init__ (vitdet.py:552-555), not weights.
    benign = ("freqs_cis_real", "freqs_cis_imag", "freqs_cis")
    if captured:
        name, missing, unexpected = captured[-1]
        rest = [k for k in missing if not any(k.endswith(b) for b in benign)]
        if rest or unexpected:
            log(f"[sam31] !! {len(rest)} randomly initialised + {len(unexpected)} dropped "
                f"in {name} — results are NOT trustworthy")
            for key in rest[:8]:
                log(f"[sam31]   !! RANDOM-INITIALISED: {key}")
            for key in unexpected[:8]:
                log(f"[sam31]   !! UNEXPECTED (dropped): {key}")
            log(noise.getvalue())
        else:
            log(f"[sam31] state dict OK — {name} loaded every tensor it needs "
                f"({len(missing)} benign RoPE buffers)")

    # Upstream bug: Sam3BasePredictor.start_session always passes offload_state_to_cpu, but
    # no init_state in the multiplex chain accepts it. Filter by the real signature so
    # offload_video_to_cpu (which IS supported) keeps working.
    import inspect

    orig_init_state = predictor.model.init_state
    accepted = set(inspect.signature(orig_init_state).parameters)

    def patched_init_state(*args, **kwargs):
        for key in [k for k in kwargs if k not in accepted]:
            kwargs.pop(key)
        return orig_init_state(*args, **kwargs)

    predictor.model.init_state = patched_init_state
    return predictor


def find_checkpoint(explicit: str | None) -> str:
    """build_sam3_multiplex_video_predictor passes load_from_HF=False, so the .pt path must
    be explicit. Falls back to whatever `just hf-fetch sam3.1` put in the local cache."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"sam3.checkpoint {explicit} does not exist")
        return explicit

    from huggingface_hub import snapshot_download

    root = snapshot_download("facebook/sam3.1")
    candidates = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"no .pt in {root} — contents: {sorted(os.listdir(root))}")
    return sorted(candidates)[0]


class Sam31Backend:
    """Drop-in for Sam3Backend: set_prompts / process_frame / reset / release + .timer.

    Streaming means no lookahead, so three upstream mechanisms are off by construction:
    hotstart, masklet confirmation, and batched grounding (which batches over FRAMES). The
    SAM 3 path already runs without hotstart, so this is not a regression.
    """

    # Every accuracy knob the multiplex model exposes (sam3_multiplex_base.py:205-258), minus
    # the ones that cannot work here. Plain attributes on the built model — unlike the
    # transformers backend there is no config object to keep in sync. Leave a key unset in
    # yaml and the builder's own demo defaults stand (model_builder.py:1167-1199), which are
    # what the published 3.1 numbers were produced with.
    #
    # NOT exposed, and why:
    #   o2o_matching_masklets_enable  upstream asserts it off (sam3_video_base.py:186)
    #   use_batched_grounding         batches over FRAMES; streaming has none
    #   image_only_det_thresh         image-only inputs; we are always video
    TUNABLES = (
        # detection
        "score_threshold_detection",     # keep gate (0.4)
        "new_det_thresh",                # spawn gate (0.65) — the one that decides whether
                                         # a concept ever becomes an object at all
        "det_nms_thresh", "det_nms_use_iom",          # 0.1 / True
        "suppress_det_close_to_boundary",             # True
        # detection <-> track association
        "assoc_iou_thresh",              # 0.1, loose: "is this detection already tracked"
        "trk_assoc_iou_thresh",          # 0.5, strict: "is this masklet unmatched"
        # masklet lifetime
        "max_trk_keep_alive", "min_trk_keep_alive", "init_trk_keep_alive",
        "decrease_trk_keep_alive_for_empty_masklets",
        # duplicate/occlusion suppression — all frame-local, so all usable streaming
        "suppress_overlapping_based_on_recent_occlusion_threshold",   # 0.7
        "allow_unoccluded_to_suppress",
        # reconditioning: re-seed a drifting masklet from a high-confidence detection.
        # Frame-local, and the single most valuable accuracy mechanism we CAN keep.
        "recondition_every_nth_frame",   # 16
        "use_iom_recondition", "iom_thresh_recondition", "iou_thresh_recondition",
        "reconstruction_bbox_iou_thresh", "reconstruction_bbox_det_score",
        # masklet confirmation: hide a masklet until it is detected on N consecutive frames.
        # Upstream reads the status from N-1 frames AHEAD; we read it now, which is stricter.
        "masklet_confirmation_enable", "masklet_confirmation_consecutive_det_thresh",
        # hotstart: the OUTPUT buffering needs lookahead and is gone, but the removal
        # heuristics it gates run per frame inside the planning phase, so these still bite.
        "hotstart_delay", "hotstart_unmatch_thresh", "hotstart_dup_thresh",
        "suppress_unmatched_only_within_hotstart",
        # mask cleanup
        "fill_hole_area", "sprinkle_removal_area",
    )

    def __init__(self, cfg: dict, log=print, profile: bool = False):
        import torch
        self._torch = torch
        self.cfg = cfg
        self.log = log
        self.device = cfg.get("device", "cuda")

        tuning = cfg.get("native_sam31") or {}
        checkpoint = find_checkpoint(tuning.get("checkpoint"))
        self.log(f"[sam31] checkpoint: {checkpoint}")
        predictor = build_predictor(
            checkpoint,
            # builder default 16 clips a 20-object scene
            max_num_objects=int(tuning.get("max_num_objects", 32)),
            multiplex_count=int(tuning.get("multiplex_count", 16)),
            use_fa3=bool(tuning.get("use_fa3", False)),
            compile=bool(tuning.get("compile", False)),
            log=log,
        )
        self.model = predictor.model

        # MRO override, not assignment: super() in the mixin reaches the original method and
        # there is no half-patched state to restore.
        self.model.__class__ = type("Sam3MultiplexMerged",
                                    (MultiConceptMixin, type(self.model)), {})
        self.model.mc_max_dets = int(tuning.get("max_dets", 128))
        self.model.mc_reset()

        # Batching over FRAMES needs future frames; we have exactly one.
        self.model.use_batched_grounding = False

        # Thresholds are read ONLY from the `native_sam31:` sub-block, never from the shared
        # `sam3:` level. The two models do not score the same caption alike — measured:
        # "pot" clears SAM 3's 0.8 spawn gate but not SAM 3.1's lower 0.65 — so inheriting
        # SAM 3's hand-tuned 0.7/0.8 would silently delete whole concepts here.
        for key in self.TUNABLES:
            if (value := tuning.get(key)) is not None:
                self._apply_override(key, value)
        self._log_effective()
        self._warn_ignored_shared_keys(cfg)

        self.image_size = int(getattr(self.model, "image_size", 1008))
        self.session_frames = int(tuning.get("session_frames", 256))
        self._mean, self._std = self._normalisation()
        # Allocated once and reused by every session: rebuilding it per session would stall
        # the node for seconds at each rollover.
        self._buffer = torch.zeros(self.session_frames, 3, self.image_size,
                                   self.image_size, dtype=torch.float16,
                                   device=self.device)

        self.timer = StageTimer(enabled=profile or bool(cfg.get("profile", False)),
                                torch_module=torch)
        if self.timer.enabled:
            self.timer.attach(self.model, stages=SAM31_STAGES)
            if self.timer.missing:
                self.log(f"[sam31] !! stages not found: {self.timer.missing}")

        self.prompts: list[str] = []
        self._id_base = 0            # keeps ids monotonic across internal session rollovers
        self._max_emitted = -1
        self._new_session()

    # -- setup -------------------------------------------------------------

    def _bf16(self):
        """The autocast the model assumes is always on.

        Sam3MultiplexBase.__init__ does `self.bf16_context.__enter__()` and comments "keep
        using for the entire model process" (sam3_multiplex_base.py:171-172) — but
        torch.autocast is THREAD-LOCAL. sam_node builds the backend on the main thread and
        calls process_frame on its worker thread, where that context does not exist.

        The failure is not a clean one. `use_early_fusion=True` means set_prompts (main
        thread, autocast on) encodes text to bf16, those language features are fused into the
        ViT trunk, and the trunk's fp32 weights then meet bf16 activations on the worker
        thread with no autocast to reconcile them:
            RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
        Entering it per call is what upstream's own request handler does
        (sam3_base_predictor.py:205), and re-entering an active autocast is a no-op.
        """
        return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)

    def _normalisation(self):
        """Match io_utils.load_resource_as_video_frames exactly — a different mean/std is a
        silent accuracy regression, not a crash."""
        import torch

        mean = torch.tensor(getattr(self.model, "image_mean", (0.5, 0.5, 0.5)),
                            dtype=torch.float16, device=self.device)[:, None, None]
        std = torch.tensor(getattr(self.model, "image_std", (0.5, 0.5, 0.5)),
                           dtype=torch.float16, device=self.device)[:, None, None]
        return mean, std

    def _apply_override(self, key: str, value) -> None:
        if hasattr(self.model, key):
            setattr(self.model, key, value)
        else:
            self.log(f"[sam31] config key '{key}' not on the model — ignored "
                     f"(renamed upstream?)")

    # Shared `sam3:` keys this backend does not read. Silence here is dangerous: someone
    # A/Bs image_size 672 vs 1008, gets identical timings, and concludes it does not matter.
    IGNORED_SHARED_KEYS = ("image_size", "dtype", "model_id", "attn_implementation",
                           "max_vision_features_cache_size")

    def _warn_ignored_shared_keys(self, cfg: dict) -> None:
        present = [k for k in self.IGNORED_SHARED_KEYS if cfg.get(k) is not None]
        if present:
            self.log(f"[sam31] ignoring hf_sam3-only keys {present} — this backend reads "
                     f"image size from the checkpoint and tuning from `native_sam31:`")

    def _log_effective(self) -> None:
        """A dead threshold shows up only as changed object counts, so print what runs."""
        pairs = [f"{k}={getattr(self.model, k)}" for k in self.TUNABLES
                 if hasattr(self.model, k)]
        self.log(f"[sam31] effective: {', '.join(pairs)}")

    # -- protocol ----------------------------------------------------------

    def set_prompts(self, prompts: list[str]) -> None:
        self.prompts = list(prompts)
        self.reset()

    def release(self) -> None:
        self.state = None

    def reset(self) -> None:
        """Full restart: ids go back to zero, matching what callers expect from a reset
        (sam_node zeroes its own id_offset in _on_set_prompts)."""
        self._id_base = 0
        self._max_emitted = -1
        self._new_session()

    def _new_session(self) -> None:
        self.state = None
        self._t = 0
        self._removed: set[int] = set()
        self.model.mc_reset()
        self.model.mc_concepts = list(self.prompts)

    def _roll_session(self, height: int, width: int) -> None:
        """Roll to a new session WITHOUT restarting the id space.

        The protocol's guarantee is that object ids are stable for the life of the backend —
        Sam3Backend keeps one transformers session for the whole run, so ids never repeat.
        Here the frame buffer and every per-frame list are sized at init, so a long run must
        start a new session, and the model's ids restart from zero when it does. sam_node
        does NOT renumber for us: its id_offset moves only on a bag time jump
        (sam_node.py:302). So the offset is ours to carry, or the next session's "chair 0"
        would be merged by map_node into the previous session's "chair 0" — two physical
        objects fused into one 3D instance.
        """
        self.log(f"[sam31] session full at {self.session_frames} frames — rolling over, "
                 f"ids continue from {self._max_emitted + 1}")
        self._id_base = self._max_emitted + 1
        self._new_session()
        self._open_session(height, width)

    def _open_session(self, height: int, width: int) -> None:
        """Every concept live at once: one text batch, an N-wide find stage per frame,
        N-wide empty geometry. Deliberately not add_prompt — that keeps a single caption and
        calls reset_state (sam3_multiplex_tracking.py:1694-1704)."""
        import torch
        from PIL import Image

        with torch.inference_mode(), self._bf16():
            # One throwaway frame: init_state only needs it to build the scaffolding, which
            # _construct_initial_input_batch then re-lays over the preallocated buffer.
            blank = Image.new("RGB", (width, height))
            state = self.model.init_state(resource_path=[blank])
            state["num_frames"] = self.session_frames
            self.model._construct_initial_input_batch(state, self._buffer)
            state["orig_height"], state["orig_width"] = height, width
            state["is_image_only"] = False

            n = len(self.prompts)
            batch = state["input_batch"]
            batch.find_text_batch[:] = list(self.prompts)
            base, ids = batch.find_inputs[0], list(range(n))
            for i in range(len(batch.find_inputs)):
                batch.find_inputs[i] = _widen(base, i, ids)
            state["constants"]["empty_geometric_prompt"] = _empty_geo(state, n)
            state["backbone_out"] = self.model._init_backbone_out(state)  # text ONCE
        self.state = state

    def process_frame(self, rgb: np.ndarray) -> Sam3FrameResult:
        """rgb: (H, W, 3) uint8, RGB order."""
        import torch

        height, width = rgb.shape[:2]
        if not self.prompts:
            return Sam3FrameResult.empty(height, width)
        if self.state is None:
            self._open_session(height, width)
        elif (height, width) != (self.state["orig_height"], self.state["orig_width"]):
            # Masks come back at the session's resolution but boxes are scaled by the
            # current frame's, so a mid-run resolution change would desync them.
            self.log(f"[sam31] frame size changed to {width}x{height} — rolling over")
            self._roll_session(height, width)
        elif self._t >= self.session_frames:
            # The frame buffer and every per-frame list are sized at init; wrapping would
            # replay slots that already have outputs.
            self._roll_session(height, width)

        frame_start = time.perf_counter()
        timer = self.timer
        t = self._t
        with torch.inference_mode(), self._bf16():
            with timer.stage("preprocess"):
                buf = self.state["input_batch"].img_batch
                # upstream's own both-shapes idiom (necks.py:235)
                getattr(buf, "tensors", buf)[t] = self._preprocess(rgb)
                # Collapse the detector's valid range onto this frame so it cannot prefetch
                # t+1 (sam3_multiplex_detector.py:467-493). Streaming has no t+1.
                self.state["feature_cache"]["tracking_bounds"] = {
                    "max_frame_num_to_track": 1,
                    "propagate_in_video_start_frame_idx": t,
                }
            out = self.model._run_single_frame_inference(self.state, t, reverse=False)
            with timer.stage("postprocess"):
                # These three lists are NOT optional. propagate_in_video passes them
                # (sam3_multiplex_tracking.py:446-468) and without them we emit objects
                # upstream hides: removed by hotstart, suppressed by overlap/occlusion, and
                # unconfirmed by masklet confirmation.
                #   removed   -- cumulative: an object removed stays removed, which is what
                #                propagate_in_video's running hotstart_removed_obj_ids does.
                #   unconfirmed -- upstream reads the status from `consecutive_det_thresh-1`
                #                frames AHEAD. Streaming has no ahead, so we use the current
                #                frame: strictly more conservative (an object stays hidden
                #                until confirmed) at the cost of a few frames of latency.
                self._removed.update(out.get("removed_obj_ids") or ())
                processed = self.model._postprocess_output(
                    self.state, out,
                    removed_obj_ids=sorted(self._removed),
                    suppressed_obj_ids=out.get("suppressed_obj_ids"),
                    unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"),
                )
            # cached_frame_outputs[t] holds full-res masks and is NEVER evicted upstream
            # (sam3_multiplex_tracking.py:1239). At ~18 MB/frame that is 4.7 GB over one
            # 256-frame session. Safe to drop ONLY because the two readers —
            # _build_sam2_output and the interactivity path's
            # fetch_and_process_single_frame_results (:2275) — are both reached via
            # propagate_in_video, which we never call.
            self.state["cached_frame_outputs"].pop(t, None)

        with timer.stage("to_numpy"):
            result = self._to_result(processed, height, width)
        self._t += 1
        timer.end_frame((time.perf_counter() - frame_start) * 1000.0,
                        len(result), len(self.prompts))
        return result

    def _preprocess(self, rgb: np.ndarray):
        """PIL resize, /255, normalise, float16 — byte-identical to the offline loader the
        0.985 mask agreement was measured with. ~1.5% of a frame; matching it matters more."""
        import torch
        from PIL import Image

        arr = np.asarray(Image.fromarray(rgb).resize((self.image_size, self.image_size)),
                         dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).to(device=self.device,
                                                        dtype=torch.float16)
        return (img - self._mean) / self._std

    def _to_result(self, processed: dict, height: int, width: int) -> Sam3FrameResult:
        object_ids = processed.get("out_obj_ids")
        if object_ids is None or len(object_ids) == 0:
            return Sam3FrameResult.empty(height, width)

        # Raw model ids restart at zero every session; _id_base keeps the backend's own id
        # space monotonic across rollovers. Attribution is keyed on the RAW id.
        raw_ids = np.asarray(object_ids).astype(int).reshape(-1)
        object_ids = raw_ids + self._id_base
        if object_ids.size:
            self._max_emitted = max(self._max_emitted, int(object_ids.max()))
        # Upstream gives normalised xywh (x_min, y_min, w, h); the protocol wants absolute
        # xyxy in original pixels.
        xywh = np.asarray(processed["out_boxes_xywh"], dtype=float).reshape(-1, 4)
        boxes = np.column_stack([
            xywh[:, 0] * width, xywh[:, 1] * height,
            (xywh[:, 0] + xywh[:, 2]) * width, (xywh[:, 1] + xywh[:, 3]) * height])

        concept_of = self.model.mc_obj_concept
        prompt_to_obj_ids: dict[str, list[int]] = {p: [] for p in self.prompts}
        for raw_id, obj_id in zip(raw_ids, object_ids):
            prompt_to_obj_ids.setdefault(concept_of.get(int(raw_id), ""), []).append(
                int(obj_id))
        prompt_to_obj_ids.pop("", None)          # objects born before attribution existed

        # NOTE: out_probs is the SPAWN-frame detection score ("first frame detection score",
        # sam3_multiplex_tracking.py:646), frozen for the life of the masklet — unlike
        # hf_sam3, which supplies a fresh score every frame. best_view.py gates on
        # mask_area * confidence, so under this backend a track that spawned marginally can
        # never rise above min_instance_score and one that degrades never falls below it.
        return Sam3FrameResult(
            object_ids=object_ids,
            scores=np.asarray(processed["out_probs"], dtype=float).reshape(-1),
            boxes=boxes,
            masks=np.asarray(processed["out_binary_masks"]).astype(bool),
            prompt_to_obj_ids=prompt_to_obj_ids,
        )


def _empty_geo(state, batch: int):
    """The stock empty_geometric_prompt is bs=1 (sam3_multiplex_tracking.py:161-168); with B
    captions the geometry encoder's grid must be batch B or grid_sampler mismatches. Always
    EMPTY — text prompts only, no geometry."""
    import torch

    template = state["constants"]["empty_geometric_prompt"]
    dev = state["device"]
    return type(template)(
        box_embeddings=torch.zeros(0, batch, 4, device=dev),
        box_mask=torch.zeros(batch, 0, device=dev, dtype=torch.bool),
        box_labels=torch.zeros(0, batch, device=dev, dtype=torch.long),
        point_embeddings=torch.zeros(0, batch, 2, device=dev),
        point_mask=torch.zeros(batch, 0, device=dev, dtype=torch.bool),
        point_labels=torch.zeros(0, batch, device=dev, dtype=torch.long),
    )


def _widen(stage, img_id: int, text_ids: list):
    """One FindStage carrying B=len(text_ids) queries: same image, several captions.
    text_ids and img_ids are parallel batches (sam3_image.py:180-184)."""
    import copy

    import torch

    wide = copy.copy(stage)
    n = len(text_ids)
    dev = stage.text_ids.device
    wide.img_ids = torch.full((n,), img_id, dtype=torch.long, device=dev)
    wide.text_ids = torch.tensor(text_ids, dtype=torch.long, device=dev)
    if getattr(stage, "img_ids_np", None) is not None:
        wide.img_ids_np = np.full((n,), img_id)
    for field in ("input_boxes", "input_boxes_mask", "input_boxes_label", "input_points",
                  "input_points_mask", "input_boxes_before_embed",
                  "input_points_before_embed"):
        value = getattr(stage, field, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == 1:
            setattr(wide, field, value.repeat(n, *([1] * (value.dim() - 1))))
    return wide
