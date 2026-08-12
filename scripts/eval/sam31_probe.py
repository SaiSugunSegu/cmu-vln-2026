#!/usr/bin/env python3
"""SAM 3.1 (Object Multiplex) streaming profile — all concepts batched, detect + track per frame.

Replicates sam_node's shape: one frame arrives, detections come out, tracker memory lives in
`inference_state`. Not `propagate_in_video` — that is an offline-video path and gets lookahead
(batched grounding over 16 frames, hotstart) we will never have at 10 Hz.

Established by probing facebookresearch/sam3 @ 96914d2:

  * `add_prompt` holds ONE caption and calls reset_state on a second call. That is the demo
    predictor's wiring, not the model — `find_text_batch` is an arbitrary caption list and any
    slot is selectable via `text_ids`.
  * text_ids and img_ids are PARALLEL BATCHES (sam3_image.py:180-184). img_ids=[t]*N with
    text_ids=[0..N-1] is N concepts against ONE image, and `_get_img_feats` forwards the
    backbone on unique img_ids only (sam3_image.py:139-165), so it costs ONE backbone pass.
  * That batch of N never reached the tracker: `_det_track_one_frame_impl` asserts the
    detection batch is 1 and squeezes it (sam3_multiplex_base.py:541-543). MultiConcept below
    is the missing merge — concat every concept's positives along the query axis.
  * The detector prefetches one frame ahead (sam3_multiplex_detector.py:479-493). --lookahead
    keeps it; the default collapses the valid range per frame so nothing future is touched.

    python3 /home/docker/scripts/eval/sam31_probe.py --bench \\
        --concepts "sofa,pillow,chair,table,pot,tv"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# One audited builder, shared with the production backend, so the bench and the node load
# the model identically — including silencing sam3's thousands of lines of key dumps.
from sam_mapper.sam31_backend import build_predictor, quiet


# SAM 3.1's per-frame seams. run_backbone_and_detection folds SAM 3's vision_encoder +
# detection into one call; the rest mirror profiling.MODEL_STAGES.
SAM31_STAGES = (
    ("run_backbone_and_detection", "detect(+backbone)"),
    ("run_tracker_propagation", "tracker_propagate"),
    ("run_tracker_update_planning_phase", "tracker_plan"),
    ("run_tracker_update_execution_phase", "tracker_execute"),
    ("build_outputs", "build_outputs"),
)

# Nested inside detect(+backbone), so these OVERLAP the table above: grounding wraps the
# backbone (forward_grounding -> _encode_prompt -> _get_img_feats -> forward_image), and the
# grounding head alone is grounding minus backbone.
SAM31_SUBSTAGES = (
    ("detector.backbone.forward_image", "  .. backbone(trunk+3 necks)"),
    ("detector.forward_video_grounding_multigpu", "  .. grounding(incl. backbone)"),
)


def find_checkpoint(explicit: str | None) -> str:
    """Locate sam3.1_multiplex.pt, but insist it is already cached — a benchmark should not
    silently spend minutes downloading 3.5 GB. The backend's own find_checkpoint will fetch."""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"[sam31] --checkpoint {explicit} does not exist")
        return explicit

    from huggingface_hub import snapshot_download

    try:
        root = snapshot_download("facebook/sam3.1", local_files_only=True)
    except Exception as err:  # noqa: BLE001
        raise SystemExit(
            f"[sam31] facebook/sam3.1 not in the local HF cache ({type(err).__name__}).\n"
            f"        Fetch it first:  just hf-fetch sam3.1") from err

    candidates = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
    if not candidates:
        raise SystemExit(f"[sam31] no .pt in {root} — contents: {sorted(os.listdir(root))}")
    return sorted(candidates)[0]


def load_pil_frames(frames_dir: str, limit: int):
    """resource_path accepts a list of PIL Images (io_utils.py:44) — no JPEG folder needed."""
    import glob

    from PIL import Image

    paths = sorted(p for p in glob.glob(os.path.join(frames_dir, "*"))
                   if p.lower().endswith((".png", ".jpg", ".jpeg")))
    if not paths:
        raise SystemExit(f"[sam31] no images in {frames_dir}")
    if limit:
        paths = paths[:limit]
    return [Image.open(p).convert("RGB") for p in paths]


def start(predictor, frames):
    return predictor.handle_request(
        request=dict(type="start_session", resource_path=frames))["session_id"]


def _empty_geo(state, batch: int):
    """The stock empty_geometric_prompt is bs=1 (sam3_multiplex_tracking.py:161); with B
    captions the geometry encoder's grid must be batch B or grid_sampler mismatches.
    Always EMPTY — text prompts only, no geometry."""
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
    """One FindStage carrying B=len(text_ids) queries: same image, several captions."""
    import copy

    import numpy as np
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
        val = getattr(stage, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] == 1:
            setattr(wide, field, val.repeat(n, *([1] * (val.dim() - 1))))
    return wide


class MultiConcept:
    """The merge that lets N concepts share one backbone pass AND one tracker.

    Detection already batches over captions, but `_det_track_one_frame_impl` asserts that
    batch is 1 and squeezes it (sam3_multiplex_base.py:541-543), so only one concept ever
    reached the tracker. Concatenating each concept's positives along the query axis is the
    whole fix: the tracker is concept-agnostic (it tracks masklets) and its other input, the
    SAM 2 backbone feature, is batch-1 regardless of caption count.

    Attribution survives because `det_out` is permuted with a generic index_select over
    `for k in det_out` (:549-551) — an extra `_prompt` column rides along, and
    new_det_fa_inds -> new_det_obj_ids (:1410-1412) turns it into obj_id -> concept.
    """

    def __init__(self, model, concepts, max_dets: int):
        self.model, self.concepts, self.max_dets = model, list(concepts), max_dets
        self.obj_concept: dict[int, str] = {}
        n = len(self.concepts)
        self.dets_per_concept = [0] * n
        # Why a detection never became a masklet, one cause each, per concept.
        self.rej_score = [0] * n        # kept by score_threshold_detection, below new_det_thresh
        self.rej_overlap = [0] * n      # IoU >= assoc_iou_thresh with a masklet of ANY concept
        self.born = [0] * n
        self.max_score = [0.0] * n      # how far a concept is from the spawn gate
        self.dropped_by_limit = 0
        self._orig: dict[str, object] = {}

    def install(self) -> "MultiConcept":
        for key, attr in (("detect", "run_backbone_and_detection"),
                          ("plan", "run_tracker_update_planning_phase")):
            self._orig[key] = getattr(self.model, attr)
        self.model.run_backbone_and_detection = self._detect
        self.model.run_tracker_update_planning_phase = self._plan
        return self

    def remove(self) -> None:
        for key, attr in (("detect", "run_backbone_and_detection"),
                          ("plan", "run_tracker_update_planning_phase")):
            if key in self._orig:
                setattr(self.model, attr, self._orig[key])
        self._orig.clear()

    def _detect(self, **kwargs):
        import torch

        det_out, pos = self._orig["detect"](**kwargs)
        if pos.shape[0] == 1:                       # single caption — nothing to merge
            return det_out, pos

        idx = pos.nonzero(as_tuple=False)           # (caption, query) pairs above threshold
        scores = det_out["scores"][idx[:, 0], idx[:, 1]]
        idx = idx[scores.argsort(descending=True)[: self.max_dets]]
        for i in range(len(self.concepts)):
            self.dets_per_concept[i] += int((idx[:, 0] == i).sum())
        merged = {k: v[idx[:, 0], idx[:, 1]].unsqueeze(0) for k, v in det_out.items()}
        merged["_prompt"] = idx[:, 0].unsqueeze(0)
        return merged, torch.ones_like(merged["_prompt"], dtype=torch.bool)

    def _plan(self, **kwargs):
        import numpy as np
        import torch

        prompts = kwargs["det_out"].get("_prompt")
        if prompts is not None:
            self._audit(kwargs, prompts)
        plan, metadata = self._orig["plan"](**kwargs)
        self.dropped_by_limit += int(plan.get("num_obj_dropped_due_to_limit", 0) or 0)
        fa_inds, obj_ids = plan.get("new_det_fa_inds"), plan.get("new_det_obj_ids")
        if prompts is not None and fa_inds is not None and len(fa_inds):
            sel = torch.as_tensor(np.asarray(fa_inds), dtype=torch.long,
                                  device=prompts.device)
            for obj_id, pid in zip(np.asarray(obj_ids).tolist(), prompts[sel].tolist()):
                self.obj_concept[int(obj_id)] = self.concepts[int(pid)]
                self.born[int(pid)] += 1
        return plan, metadata

    def _audit(self, kwargs, prompts) -> None:
        """Replay the two gates in _associate_det_trk_compilable (sam3_video_base.py:208-212)
        so each rejected detection is charged to one cause. Diagnostic only — the real
        association still runs untouched."""
        import torch
        import torch.nn.functional as F
        from sam3.perflib.masks_ops import mask_iou

        model = self.model
        det_masks = kwargs["det_out"]["mask"]
        scores = kwargs["det_out"]["scores"].float()
        trk_masks = kwargs["tracker_low_res_masks_global"]
        above = scores >= model.new_det_thresh

        overlaps = torch.zeros_like(above)
        if trk_masks is not None and trk_masks.size(0) and det_masks.size(0):
            det_bin, trk_bin = det_masks > 0, trk_masks > 0
            if det_bin.shape[-2:] != trk_bin.shape[-2:]:
                trk_bin = F.interpolate(trk_bin.unsqueeze(1).float(),
                                        size=det_bin.shape[-2:], mode="nearest"
                                        ).squeeze(1) > 0
            overlaps = (mask_iou(det_bin, trk_bin) >= model.assoc_iou_thresh).any(dim=1)

        for i in range(len(self.concepts)):
            mine = prompts == i
            self.rej_score[i] += int((mine & ~above).sum())
            self.rej_overlap[i] += int((mine & above & overlaps).sum())
            if bool(mine.any()):
                self.max_score[i] = max(self.max_score[i], float(scores[mine].max()))


def open_stream(predictor, frames, concepts):
    """A session with every concept live: one text batch, an N-wide find stage per frame,
    N-wide empty geometry. Deliberately not add_prompt — that keeps one caption and resets."""
    import torch

    model = predictor.model
    sid = start(predictor, frames)
    state = predictor._all_inference_states[sid]["state"]
    n = len(concepts)
    with torch.inference_mode():
        batch = state["input_batch"]
        batch.find_text_batch[:] = list(concepts)
        base, ids = batch.find_inputs[0], list(range(n))
        for i in range(len(batch.find_inputs)):
            batch.find_inputs[i] = _widen(base, i, ids)
        state["constants"]["empty_geometric_prompt"] = _empty_geo(state, n)
        state["backbone_out"] = model._init_backbone_out(state)   # text encoded ONCE
    return sid, state


def save_frame_masks(out_dir: str, index: int, res: dict) -> None:
    """Same archive layout sam3_backend._save_frame_masks writes, so sam3_backend's
    compare_masks can score SAM 3.1 against a SAM 3 reference run."""
    import numpy as np

    masks = res["out_binary_masks"].astype(bool)
    np.savez_compressed(
        os.path.join(out_dir, f"frame_{index:05d}.npz"),
        object_ids=res["out_obj_ids"], scores=res["out_probs"],
        boxes=res["out_boxes_xywh"],
        mask_shape=np.asarray(masks.shape, dtype=np.int64),
        masks_packed=(np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
                      if masks.size else np.zeros((0, 0), dtype=np.uint8)),
    )


def overlap_pairs(out, obj_concept, stride: int = 4):
    """Pairwise IoU and IoM among a frame's masklets, taken BEFORE _postprocess_output forces
    them disjoint (sam3_tracking_predictor.py:1349-1369 hands every pixel to one object, which
    would hide a duplicate as an eaten mask rather than an overlap).

    IoM matters as much as IoU: a fragment sitting inside a larger mask has low IoU but IoM
    near 1, and that is exactly what object splitting looks like.
    """
    import torch

    id_to_mask = out.get("obj_id_to_mask") or {}
    ids = sorted(id_to_mask)
    if len(ids) < 2:
        return []
    flat = torch.cat([id_to_mask[i].reshape(1, -1)[:, ::stride] for i in ids]).float()
    inter = flat @ flat.t()
    area = torch.diagonal(inter)
    union = area[:, None] + area[None, :] - inter
    smaller = torch.minimum(area[:, None], area[None, :])
    zero = torch.zeros_like(inter)
    iou = torch.where(union > 0, inter / union.clamp_min(1e-9), zero).cpu()
    iom = torch.where(smaller > 0, inter / smaller.clamp_min(1e-9), zero).cpu()
    return [(float(iou[a, b]), float(iom[a, b]),
             obj_concept.get(int(ids[a]), "?"), obj_concept.get(int(ids[b]), "?"))
            for a in range(len(ids)) for b in range(a + 1, len(ids))]


def report_overlaps(pairs, n_frames: int, log=print, iou_thresh=0.5, iom_thresh=0.7) -> None:
    """Which of the three explanations for the extra objects/frame holds: cross-concept
    duplicates, same-concept fragmentation, or genuine extra recall."""
    same_iou = cross_iou = same_iom = cross_iom = 0
    offenders: dict[tuple, int] = {}
    for iou, iom, ca, cb in pairs:
        same = ca == cb
        if iou >= iou_thresh:
            same_iou += same
            cross_iou += not same
        if iom >= iom_thresh:
            same_iom += same
            cross_iom += not same
            key = (ca, cb) if same else tuple(sorted((ca, cb)))
            offenders[key] = offenders.get(key, 0) + 1
    nf = max(n_frames, 1)
    # Without these two lines a genuine "nothing overlaps" is indistinguishable from a probe
    # that silently examined nothing.
    log(f"   {len(pairs)} mask pairs examined ({len(pairs) / nf:.0f}/frame), "
        f"max IoU {max((p[0] for p in pairs), default=0.0):.3f}, "
        f"max IoM {max((p[1] for p in pairs), default=0.0):.3f}")
    if not pairs:
        log("   !! no pairs — check failed to read obj_id_to_mask, result is NOT a negative")
        return
    log(f"   duplicates  IoU>={iou_thresh}   same-concept {same_iou / nf:5.2f}/frame   "
        f"cross-concept {cross_iou / nf:5.2f}/frame")
    log(f"   contained   IoM>={iom_thresh}   same-concept {same_iom / nf:5.2f}/frame   "
        f"cross-concept {cross_iom / nf:5.2f}/frame")
    for (ca, cb), n in sorted(offenders.items(), key=lambda kv: -kv[1])[:5]:
        log(f"      {ca} + {cb}: {n / nf:.2f}/frame")
    if not (same_iou or cross_iou or same_iom or cross_iom):
        log("   -> no overlapping masklets: the extra objects are genuine detections")


def apply_thresholds(model, score: float | None, spawn: float | None, log=print) -> None:
    """SAM 3.1's builder defaults (0.4 keep / 0.65 spawn) are NOT our SAM 3 yaml's (0.7/0.8),
    and the two models do not score the same caption alike — so these are calibrated against
    SAM 3's masks, not copied from it."""
    for attr, value in (("score_threshold_detection", score), ("new_det_thresh", spawn)):
        if value is not None:
            log(f"[sam31] {attr}: {getattr(model, attr)} -> {value}")
            setattr(model, attr, value)


def probe_necks(predictor, state, repeats: int = 10, log=print):
    """Price the interactive neck. sam3_multiplex_base.py:770 computes it every frame under
    the comment "We do not need the interaction features every frame"; it is only consumed
    when a masklet is born. The trunk runs once and three cloned FPN necks decode it
    (necks.py:213-216), so the saving is one neck, not one third of detect.

    Isolated: same image, forward the vision backbone with and without that branch. The model
    holds a process-wide bf16 autocast (sam3_multiplex_base.py:171-172), so this matches the
    live path's precision without entering one here.
    """
    import numpy as np
    import torch

    backbone = predictor.model.detector.backbone.vision_backbone
    img_batch = state["input_batch"].img_batch
    image = (img_batch[0:1] if isinstance(img_batch, torch.Tensor)
             else img_batch[0].unsqueeze(0))
    image = image.to(dtype=torch.float32, device=predictor.model.device)

    def timed(**flags) -> float:
        with torch.inference_mode():
            backbone.forward(image, **flags)                    # warm
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                backbone.forward(image, **flags)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / repeats

    full = timed(need_sam3_out=True, need_interactive_out=True, need_propagation_out=True)
    lean = timed(need_sam3_out=True, need_interactive_out=False, need_propagation_out=True)
    log(f"   trunk + 3 necks        {full:7.1f} ms")
    log(f"   trunk + 2 necks        {lean:7.1f} ms   (interactive branch off)")
    log(f"   -> interactive neck    {full - lean:7.1f} ms/frame recoverable")
    return full, lean


def stream_profile(predictor, frames, concepts, max_dets, lookahead, save_masks=None,
                   log=print):
    """One frame in -> detect(all concepts) + track -> out, as sam_node receives /camera/image."""
    import torch
    from sam_mapper.profiling import StageTimer

    model = predictor.model
    model.use_batched_grounding = False       # batching over FRAMES needs future frames
    sid, state = open_stream(predictor, frames, concepts)
    merge = MultiConcept(model, concepts, max_dets).install()
    timer = StageTimer(enabled=True, torch_module=torch).attach(
        model, stages=SAM31_STAGES + SAM31_SUBSTAGES)
    if timer.missing:
        log(f"[sam31] !! stages not found: {timer.missing}")

    per_frame_ms, counts, pairs = [], [], []
    removed: set[int] = set()          # cumulative, as propagate_in_video keeps it
    with torch.inference_mode():
        for t in range(len(frames)):
            if not lookahead:
                # Collapse the detector's valid range onto this frame so it cannot prefetch
                # t+1 (sam3_multiplex_detector.py:467-480). Streaming has no t+1.
                state["feature_cache"]["tracking_bounds"] = {
                    "max_frame_num_to_track": 1,
                    "propagate_in_video_start_frame_idx": t,
                }
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model._run_single_frame_inference(state, t, reverse=False)
            # Not optional: propagate_in_video passes these three
            # (sam3_multiplex_tracking.py:446-468). Without them the output includes objects
            # upstream hides — removed by hotstart, suppressed by overlap/occlusion,
            # unconfirmed by masklet confirmation.
            removed.update(out.get("removed_obj_ids") or ())
            res = model._postprocess_output(
                state, out, removed_obj_ids=sorted(removed),
                suppressed_obj_ids=out.get("suppressed_obj_ids"),
                unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"))
            torch.cuda.synchronize()
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
            obj_ids = res.get("out_obj_ids")
            counts.append(0 if obj_ids is None else int(len(obj_ids)))
            timer.end_frame(per_frame_ms[-1], counts[-1], len(concepts))
            # After the clock stops — diagnostics must not land in the timings.
            pairs.extend(overlap_pairs(out, merge.obj_concept))
            if save_masks:
                save_frame_masks(save_masks, t, res)

    timer.detach()
    merge.remove()
    log("\n INTERACTIVE NECK (isolated, same image)")
    necks = probe_necks(predictor, state, log=log)
    predictor.handle_request(request=dict(type="close_session", session_id=sid))
    return timer.summary(), per_frame_ms, counts, merge, necks, pairs


def bench(predictor, frames, concepts, max_dets, lookahead, save_masks=None,
          baseline=None, sam3_ms=None, log=print) -> None:
    import numpy as np
    from sam_mapper.profiling import format_gpu_state, format_thermal_drift, gpu_state

    if save_masks:
        os.makedirs(save_masks, exist_ok=True)
    before = gpu_state()
    log("=" * 78)
    log(f" SAM 3.1 STREAMING — {len(frames)} frames, {len(concepts)} concepts batched")
    log(f" {', '.join(concepts)}")
    log(f" one frame in -> detect(all concepts, ONE backbone pass) + track -> out")
    log(f" lookahead: {'ON (offline-like)' if lookahead else 'OFF (true streaming)'}")
    log(f" {format_gpu_state(before).strip()}")
    log("=" * 78)

    summary, per_frame_ms, counts, merge, necks, pairs = stream_profile(
        predictor, frames, concepts, max_dets, lookahead, save_masks, log)

    stages = summary["stages"]
    nested = {k: v for k, v in stages.items() if k.startswith("  ")}
    flat = {k: v for k, v in stages.items() if not k.startswith("  ")}
    detect = flat.get("detect(+backbone)", 0.0)
    tracker = sum(v for k, v in flat.items() if k.startswith(("tracker", "build")))
    total = float(np.median(per_frame_ms))
    n_obj = float(np.mean(counts))

    log("\n PER FRAME")
    for label, ms in sorted(flat.items(), key=lambda kv: -kv[1]):
        if ms >= 0.1:
            log(f"   {label:<28} {ms:7.1f} ms  {100.0 * ms / max(total, 1e-9):5.1f}%")
    for label, ms in sorted(nested.items(), key=lambda kv: -kv[1]):
        log(f"   {label:<28} {ms:7.1f} ms")
    backbone = nested.get("  .. backbone(trunk+3 necks)")
    grounding = nested.get("  .. grounding(incl. backbone)")
    if backbone is not None and grounding is not None:
        log(f"   {'  .. grounding head only':<28} {grounding - backbone:7.1f} ms")
    log(f"   {'-' * 50}")
    log(f"   {'detect':<28} {detect:7.1f} ms")
    log(f"   {'tracker':<28} {tracker:7.1f} ms")
    log(f"   {'TOTAL':<28} {total:7.1f} ms   ({1000.0 / max(total, 1e-9):.2f} Hz)")
    log(f"   {'objects tracked':<28} {n_obj:7.1f}")

    # Attribution proves the merge kept concepts apart, and charges every detection that did
    # NOT become a masklet to one cause. `overlap` is the cross-concept one: a detection
    # suppressed because it overlaps a masklet that some OTHER concept owns.
    nf = max(len(per_frame_ms), 1)
    log(f"\n PER CONCEPT   (score gate {predictor.model.score_threshold_detection} keep / "
        f"{predictor.model.new_det_thresh} spawn, assoc IoU {predictor.model.assoc_iou_thresh})")
    log(f"   {'concept':<12} {'masklets':>8} {'dets/fr':>8} {'rej:score':>10} "
        f"{'rej:overlap':>12} {'max score':>10}")
    spawn = predictor.model.new_det_thresh
    for i, concept in enumerate(concepts):
        flag = "  <- never reaches the spawn gate" if merge.max_score[i] < spawn else ""
        log(f"   {concept:<12} {merge.born[i]:8d} "
            f"{merge.dets_per_concept[i] / nf:8.1f} "
            f"{merge.rej_score[i] / nf:10.1f} {merge.rej_overlap[i] / nf:12.1f} "
            f"{merge.max_score[i]:10.3f}{flag}")
    if merge.dropped_by_limit:
        log(f"   !! {merge.dropped_by_limit} objects dropped by the max_num_objects limit")

    log("\n OVERLAPPING MASKLETS   (pre-postprocess, before masks are forced disjoint)")
    report_overlaps(pairs, len(per_frame_ms), log)

    if sam3_ms:
        sam3, basis = sam3_ms, "measured"
    else:
        # Fitted at P=2,3 / N=2.0,12.2 — P=6 is an extrapolation and reads ~25% low against
        # the measured run. Pass --sam3-ms once you have a real baseline.
        sam3, basis = 175 + 30 * len(concepts) + 37 + 20.4 * n_obj, "EXTRAPOLATED"
    log(f"\n SAM 3, {len(concepts)} concepts, {n_obj:.1f} obj: {sam3:.0f} ms [{basis}]")
    log(f"   speedup                       {sam3 / max(total, 1e-9):8.2f}x")
    saved = necks[0] - necks[1]
    log(f"   without the interactive neck  {sam3 / max(total - saved, 1e-9):8.2f}x   "
        f"({total - saved:.0f} ms, {1000.0 / max(total - saved, 1e-9):.2f} Hz)")

    log("\n Stage figures are MEDIANS: a stage that only fires when a masklet is born"
        "\n (tracker_execute) reads ~0 because most frames create none.")
    if drift := format_thermal_drift(before, gpu_state()):
        log(drift)

    if baseline:
        from sam_mapper.sam3_backend import compare_masks

        log("\n ACCURACY vs SAM 3")
        compare_masks(baseline, save_masks, log=log)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", default="/data/bags/_frames")
    parser.add_argument("--concepts", default="sofa,pillow,chair,table,pot,tv",
                        help="comma-separated text prompts, all batched into one pass")
    parser.add_argument("--limit", type=int, default=0, help="0 = every frame")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-num-objects", type=int, default=32,
                        help="builder default 16 clips a 20-object scene")
    parser.add_argument("--multiplex-count", type=int, default=16, help="objects per bucket")
    parser.add_argument("--max-dets", type=int, default=128,
                        help="cap on merged detections per frame, highest score first")
    parser.add_argument("--lookahead", action="store_true",
                        help="keep the detector's 1-frame prefetch (offline-like, not our case)")
    parser.add_argument("--score-thresh", type=float, default=None,
                        help="override score_threshold_detection (3.1 default 0.4)")
    parser.add_argument("--new-det-thresh", type=float, default=None,
                        help="override new_det_thresh, the masklet spawn gate (3.1 default 0.65)")
    parser.add_argument("--save-masks", default=None,
                        help="write per-frame mask archives here, for --compare-baseline")
    parser.add_argument("--compare-baseline",
                        help="a SAM 3 --save-masks directory to score mask IoU against")
    parser.add_argument("--sam3-ms", type=float, default=None,
                        help="measured SAM 3 ms/frame for the same frames and concepts; "
                             "without it the comparison is an extrapolated fit")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args(argv)

    if not args.bench:
        parser.error("pass --bench")
    if args.compare_baseline and not args.save_masks:
        parser.error("--compare-baseline needs --save-masks to compare against")

    quiet()                                  # must precede `import sam3`
    try:
        import sam3  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            f"[sam31] the `sam3` package is not installed ({err}).\n"
            f"        See ai_module/docker/Dockerfile — pinned, --no-deps, plus einops/"
            f"iopath/pycocotools.") from err

    checkpoint = find_checkpoint(args.checkpoint)
    print(f"[sam31] checkpoint: {checkpoint} ({os.path.getsize(checkpoint) / 1e9:.2f} GB)")
    frames = load_pil_frames(args.frames, args.limit)
    print(f"[sam31] {len(frames)} frames from {args.frames} "
          f"({frames[0].size[0]}x{frames[0].size[1]})")

    predictor = build_predictor(checkpoint,
                                max_num_objects=args.max_num_objects,
                                multiplex_count=args.multiplex_count)
    apply_thresholds(predictor.model, args.score_thresh, args.new_det_thresh)
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    bench(predictor, frames, concepts, args.max_dets, args.lookahead,
          args.save_masks, args.compare_baseline, args.sam3_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
