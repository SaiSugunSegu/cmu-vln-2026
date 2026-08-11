"""SAM 3 video backend: text prompts in, tracked masks out.

Two rules govern how SAM 3 is driven, and both are load-bearing:

  1. ONE session, ALL prompts — vision features are shared across prompts, so N classes
     cost ~1 forward pass, not N.
  2. ONE session for the whole run — object ids are only stable within a session, and
     those ids are what the 3D mapper associates on.

Frames may be dropped (the tracker is memory-based, so that just lowers the effective
tracking rate) but must arrive in order.

Standalone probe:
    python -m sam_mapper.sam3_backend --frames DIR --config CFG.yaml --profile
    python -m sam_mapper.sam3_backend --frames DIR --config CFG.yaml --sweep-image-size
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from sam_mapper.profiling import StageTimer

# Non-square input does not work: a reshape downstream of the ViT hardcodes a square
# token grid, so [672,2016] dies with "shape '[1, 72, 72, -1]' is invalid". The panorama
# therefore stays squashed into a square, and the only lever is which square. Valid sizes
# must divide by patch_size 14 AND give a grid divisible by window_size 24 -> {24,48,72}.
IMAGE_SIZE_PRESETS = {
    "square_1008": [1008, 1008],   # 5184 tokens, model default
    "square_672": [672, 672],      # 2304 tokens, 0.44x
    "square_336": [336, 336],      # 576 tokens, 0.11x
}


@dataclass
class Sam3FrameResult:
    """One frame of SAM 3 output.

    object_ids:        (N,) int   — stable across frames within a session
    scores:            (N,) float
    boxes:             (N,4) xyxy, absolute pixels in the original image
    masks:             (N,H,W) bool, at original resolution
    prompt_to_obj_ids: which prompt produced which ids
    """
    object_ids: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    masks: np.ndarray
    prompt_to_obj_ids: dict[str, list[int]]

    @classmethod
    def empty(cls, height: int, width: int) -> "Sam3FrameResult":
        return cls(np.zeros(0, int), np.zeros(0, float), np.zeros((0, 4), float),
                   np.zeros((0, height, width), bool), {})

    def __len__(self) -> int:
        return int(self.object_ids.shape[0])


class Sam3Backend:
    """transformers Sam3VideoModel in streaming mode.

    Streaming disables SAM 3's hotstart heuristics (they need future frames), so this
    yields more duplicate tracks than pre-loaded inference. Tolerable because
    ObjMapper's world-space same-label merge folds duplicates into one 3D instance.
    """

    def __init__(self, cfg: dict, log=print, profile: bool = False):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self._torch = torch
        self.log = log
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

        model_id = cfg.get("model_id", "facebook/sam3")
        image_size = cfg.get("image_size")

        kwargs = {"dtype": self.dtype}
        if image_size is not None:
            from transformers import Sam3VideoConfig
            config = Sam3VideoConfig.from_pretrained(model_id)
            # Top-level property: fans out to BOTH detector_config and tracker_config.
            # Setting detector_config.image_size alone leaves the tracker's prompt-encoder
            # grid hardcoded at 1008/14=72, so it can't reshape a smaller feature map.
            config.image_size = int(image_size[0])
            kwargs["config"] = config

        self.model = self._load(Sam3VideoModel, model_id, kwargs)
        self.model.to(self.device)
        self.model.eval()

        proc_kwargs = {}
        if image_size is not None:
            proc_kwargs["size"] = {"height": int(image_size[0]), "width": int(image_size[1])}
        self.processor = Sam3VideoProcessor.from_pretrained(model_id, **proc_kwargs)

        # fill_hole_area <= 0 makes fill_holes_in_mask_scores return early — the supported
        # off-switch for hole filling / sprinkle removal, and the A/B control for cv-utils.
        tunables = ("score_threshold_detection", "new_det_thresh", "det_nms_thresh",
                    "recondition_every_nth_frame", "max_trk_keep_alive",
                    "min_trk_keep_alive", "hotstart_delay", "fill_hole_area")
        for key in tunables:
            if (value := cfg.get(key)) is None:
                continue
            self._apply_override(key, value)
        self.log_effective_config(tunables)

        # What _load accepted is the REQUEST, not what runs: transformers can satisfy a
        # flash_attention_2 request from the kernels hub with no flash_attn package
        # installed. One line at startup, so nobody has to guess again.
        from sam_mapper.profiling import effective_attention
        self.log(f"[sam3] attn effective: {effective_attention(self.model)}")

        # Off by default: every stage boundary is a cuda synchronize, which is exactly what
        # makes the numbers attributable and exactly why production must not pay for it.
        self.timer = StageTimer(enabled=profile or bool(cfg.get("profile", False)),
                                torch_module=torch)
        if self.timer.enabled:
            self.timer.attach(self.model)

        self.prompts: list[str] = []
        self.session = None

    # Flash attention is deliberately absent. Measured on SAM 3
    # kernels-community/flash-attn2 returns 0.0 objects/frame vs sdpa's 2.0 — Sam3Attention
    # falls back to SDPA for its relative-position cross-attention anyway. transformers 5.15
    # silently substitutes the hub kernel for a `flash_attention_2` request, so leaving it in
    # an automatic chain would quietly yield a zero-detection detector. Request via --attn to
    # re-measure. FA3 is Hopper-only (sm_90); both deploy targets are Ada sm_89.
    ATTN_FALLBACKS = ("sdpa", "eager")

    def _load(self, model_cls, model_id, kwargs):
        """Try the requested attention backend, degrade rather than die."""
        requested = self.cfg.get("attn_implementation", "flash_attention_2")
        errors = []
        for attn in [requested] + [a for a in self.ATTN_FALLBACKS if a != requested]:
            try:
                model = model_cls.from_pretrained(model_id, attn_implementation=attn, **kwargs)
                self.log(f"[sam3] attn requested: {attn}"
                         + (f" ('{requested}' unavailable: {errors[0]})" if attn != requested
                            else ""))
                return model
            except Exception as err:            # noqa: BLE001 - degrade, then report
                errors.append(f"{attn}: {type(err).__name__}: {err}")
        raise RuntimeError(f"could not load {model_id} with any attention backend; "
                           + " | ".join(errors))

    def _apply_override(self, key: str, value) -> None:
        """Apply one knob to BOTH the config and the model instance.

        Setting only `model.config` is a silent no-op: Sam3VideoModel.__init__ copies each
        knob onto the module and the forward path reads the copy (verified in transformers
        5.15 — `self.config.<key>` is never read). The module attribute is what runs; the
        config attribute is what a save/from_pretrained round-trip would carry.
        """
        found = False
        for target in (self.model.config, self.model):
            if hasattr(target, key):
                setattr(target, key, value)
                found = True
        if not found:
            self.log(f"[sam3] config key '{key}' exists on neither the model nor its config "
                     f"— ignored (renamed upstream?)")

    def log_effective_config(self, keys) -> None:
        """What the model will actually use — insurance against the override path
        regressing again, since a dead threshold shows up only as changed counts."""
        pairs = [f"{k}={getattr(self.model, k)}" for k in keys if hasattr(self.model, k)]
        if pairs:
            self.log(f"[sam3] effective: {', '.join(pairs)}")

    def set_prompts(self, prompts: list[str]) -> None:
        self.prompts = list(prompts)
        self.reset()

    def release(self) -> None:
        """Drop the current session and everything it holds.
        """
        self.session = None

    def reset(self) -> None:
        """Start a fresh session. Object ids restart, so callers must renumber."""
        self.session = self.processor.init_video_session(
            inference_device=self.device, processing_device="cpu",
            video_storage_device="cpu", dtype=self.dtype,
            max_vision_features_cache_size=self.cfg.get("max_vision_features_cache_size", 1),
        )
        if self.prompts:
            self.processor.add_text_prompt(self.session, self.prompts)

    def process_frame(self, rgb: np.ndarray) -> Sam3FrameResult:
        """rgb: (H, W, 3) uint8, RGB order."""
        if self.session is None:
            self.reset()

        height, width = rgb.shape[:2]
        timer = self.timer
        frame_start = time.perf_counter()

        with timer.stage("preprocess"):
            inputs = self.processor(images=rgb, device=self.device,
                                    return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            # Model stages are timed from inside, by StageTimer's instance wrappers.
            outputs = self.model(inference_session=self.session,
                                 frame=inputs.pixel_values[0], reverse=False)
            with timer.stage("postprocess"):
                processed = self.processor.postprocess_outputs(
                    self.session, outputs, original_sizes=inputs.original_sizes)
        with timer.stage("to_numpy"):
            result = self._to_result(processed, height, width)

        timer.end_frame((time.perf_counter() - frame_start) * 1000.0,
                        len(result), len(self.prompts))
        return result

    @staticmethod
    def _to_result(processed: dict, height: int, width: int) -> Sam3FrameResult:
        object_ids = processed.get("object_ids")
        if object_ids is None or len(object_ids) == 0:
            return Sam3FrameResult.empty(height, width)

        def to_numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        # Masks are the large payload: (N, 640, 1920) is 1.2 MB per object per frame.
        # Casting to bool ON DEVICE makes the host transfer move an already-final array;
        # `.cpu().numpy().astype(bool)` allocated a second full-size host copy.
        masks = processed["masks"]
        if hasattr(masks, "detach"):
            import torch

            masks = masks.detach()
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.to(dtype=torch.bool).contiguous().cpu().numpy()
        else:
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.astype(bool, copy=False)

        return Sam3FrameResult(
            object_ids=to_numpy(object_ids).astype(int).reshape(-1),
            scores=to_numpy(processed["scores"]).astype(float).reshape(-1),
            boxes=to_numpy(processed["boxes"]).astype(float).reshape(-1, 4),
            masks=masks,
            prompt_to_obj_ids={str(k): [int(i) for i in v]
                               for k, v in (processed.get("prompt_to_obj_ids") or {}).items()},
        )


# -- standalone probe ---------------------------------------------------------


def report_device(log=print) -> None:
    """Slow frames can mean no GPU, wrong GPU, or emulated bf16 — timing alone cannot tell."""
    import torch

    log(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("  !! running on CPU — explains multi-second frames on its own")
        return
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    log(f"  {props.name} (compute {props.major}.{props.minor}, {props.total_memory / 1e9:.0f} GB)")
    if props.major < 8:
        log("  !! pre-Ampere: bfloat16 is emulated and slow — try dtype: float16")

    # Reported every run: a throttled card rescales the whole stage table, and it drifts
    # between runs, so back-to-back A/B rows get slower purely by execution order.
    from sam_mapper.profiling import format_gpu_state, gpu_state

    log(format_gpu_state(gpu_state()))
    report_mask_kernel(log)


def report_mask_kernel(log=print) -> None:
    """Is the cv-utils kernel live? Its absence is invisible: mask NMS is skipped entirely
    and hole filling returns dummy counts that never trigger, so output just gets worse."""
    try:
        from transformers.models.sam3_video import modeling_sam3_video as m

        m._load_cv_utils_kernel_once()
        if getattr(m, "cv_utils_kernel", None):
            log("  cv-utils kernel: loaded (NMS, hole filling, sprinkle removal active)")
        else:
            log("  !! cv-utils kernel NOT loaded — mask NMS and post-processing skipped, "
                "silently.\n"
                "     Usually a `kernels` version outside transformers' window, or no build "
                "variant\n"
                "     for this torch: see docker/requirements_captioner.txt.")
    except Exception as err:                     # noqa: BLE001 — a probe must not die here
        log(f"  cv-utils kernel: could not determine ({type(err).__name__}: {err})")


def _load_frames(frames_dir: str) -> tuple[list[np.ndarray], list[str]]:
    """Returns (frames, paths) — paths kept so annotated output can reuse the same filenames."""
    import glob
    import os

    import cv2

    paths = sorted(p for p in glob.glob(os.path.join(frames_dir, "*"))
                   if p.lower().endswith((".png", ".jpg", ".jpeg")))
    if not paths:
        raise SystemExit(f"no images found in {frames_dir}")
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    return frames, paths


def _log_verbose_2d(log, i, detections):
    """Every 2D detection for one frame — mirrors sam_node's verbose_objects (2D half only;
    no lidar here, so no 3D boxes to print)."""
    log(f"  frame {i}: {len(detections['ids'])} detections")
    for label, score, obj_id, bbox in zip(detections["labels"], detections["confidences"],
                                          detections["ids"], detections["bboxes"]):
        x0, y0, x1, y1 = (round(float(v)) for v in bbox)
        log(f"    2D  {label:<14} id={obj_id:<4} score={score:.2f} bbox=({x0},{y0},{x1},{y1})")


def _run(frames, cfg, label, log=print, verbose=False, save_annotated=None, paths=None,
         profile=False, save_masks=None):
    """save_annotated: directory to write annotate.annotate_frame() overlays to, or None.
    save_masks: directory to write per-frame mask archives to, for --compare-baseline."""
    import os

    from sam_mapper.detections import PromptTable, to_detections

    backend = Sam3Backend(cfg, log=log, profile=profile)
    table = PromptTable(cfg["_objects"])
    backend.set_prompts(table.prompts)

    if save_annotated:
        import cv2

        from sam_mapper.annotate import annotate_frame
        os.makedirs(save_annotated, exist_ok=True)
    if save_masks:
        os.makedirs(save_masks, exist_ok=True)

    from sam_mapper.profiling import format_thermal_drift, gpu_state

    gpu_before = gpu_state()
    per_frame_ids, times, counts = [], [], []
    for i, frame in enumerate(frames):
        start = time.perf_counter()
        result = backend.process_frame(frame)
        if i > 0:                                  # skip warm-up
            times.append((time.perf_counter() - start) * 1000.0)
        per_frame_ids.append(set(result.object_ids.tolist()))
        counts.append(len(result))

        detections = to_detections(result, table) if (verbose or save_annotated) else None
        if verbose:
            _log_verbose_2d(log, i, detections)
        elif i == 0:
            hits = {k: len(v) for k, v in result.prompt_to_obj_ids.items() if v}
            log(f"  frame 0: {len(result)} objects, {hits}")

        if save_annotated:
            name = os.path.basename(paths[i]) if paths else f"frame_{i:05d}.png"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)          # SAM3 wants RGB; cv2 writes BGR
            cv2.imwrite(os.path.join(save_annotated, name), annotate_frame(bgr, detections))
        if save_masks:
            _save_frame_masks(save_masks, i, result)

    if save_annotated:
        log(f"  saved {len(frames)} annotated frames to {save_annotated}")
    if save_masks:
        log(f"  saved {len(frames)} mask archives to {save_masks}")

    # Ids present in EVERY frame are stably tracked — the property that replaces
    # ByteTrack. Near zero here means tracking is not working.
    persistent = set.intersection(*per_frame_ids) if per_frame_ids else set()
    ms = float(np.median(times)) if times else float("nan")
    objects = float(np.mean(counts))
    log(f"  {label}: {ms:.0f} ms/frame ({1000.0 / ms:.2f} Hz), {objects:.1f} objects "
        f"({ms / max(objects, 1):.0f} ms/object), {len(persistent)} ids in all {len(frames)} frames")

    gpu_after = gpu_state()
    if drift := format_thermal_drift(gpu_before, gpu_after):
        log(drift)

    row = {"label": label, "ms": ms, "objects": objects, "persistent_ids": len(persistent),
           "gpu_before": gpu_before, "gpu_after": gpu_after}
    if profile:
        from sam_mapper.profiling import (format_per_object_fit, format_summary)

        summary = backend.timer.summary()
        fit = backend.timer.per_object_fit()
        log(format_summary(summary, title=f"stage breakdown — {label}"))
        log(format_per_object_fit(fit))
        row["summary"] = summary
        row["per_object_fit"] = fit
    return row


def _torch_profile(frames, cfg, out_dir, log=print, warmup=2, active=3):
    """torch.profiler over a few frames — catches costs that sit on no named seam (a hidden
    host sync, an unfused elementwise storm, a CPU loop inside one call)."""
    import os

    import torch
    from torch.profiler import ProfilerActivity, profile

    from sam_mapper.detections import PromptTable

    backend = Sam3Backend(cfg, log=log)
    backend.set_prompts(PromptTable(cfg["_objects"]).prompts)

    for frame in frames[:warmup]:                    # get weights/kernels resident first
        backend.process_frame(frame)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        for frame in frames[warmup:warmup + active]:
            backend.process_frame(frame)

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    log(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    os.makedirs(out_dir, exist_ok=True)
    trace = os.path.join(out_dir, "sam3_trace.json")
    prof.export_chrome_trace(trace)
    log(f"  chrome trace: {trace}  (open in chrome://tracing or perfetto.dev)")


# -- mask archives + baseline comparison --------------------------------------
# The quality gate: a speed change can degrade masks while ms/frame improves.

# IoU is computed on masks strided by this. Full-res pairwise IoU is ~1.3 G element-ops per
# frame; stride 4 makes it a 76800-wide matmul and moves the number in the third decimal.
IOU_STRIDE = 4


def _save_frame_masks(out_dir: str, index: int, result: "Sam3FrameResult") -> None:
    """One compressed archive per frame; masks bit-packed (40 MB raw -> 5 MB -> less after
    deflate, being mostly zeros)."""
    import os

    masks = result.masks
    np.savez_compressed(
        os.path.join(out_dir, f"frame_{index:05d}.npz"),
        object_ids=result.object_ids, scores=result.scores, boxes=result.boxes,
        mask_shape=np.asarray(masks.shape, dtype=np.int64),
        masks_packed=np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
        if masks.size else np.zeros((0, 0), dtype=np.uint8),
    )


def _load_frame_masks(path: str) -> np.ndarray:
    """-> masks strided to IOU_STRIDE. Only ever loaded to compare against a reference."""
    data = np.load(path)
    shape = tuple(int(v) for v in data["mask_shape"])
    if shape[0] == 0:
        return np.zeros((0, 0), dtype=bool)
    flat = np.unpackbits(data["masks_packed"], axis=1, count=int(np.prod(shape[1:])))
    return flat.reshape(shape).astype(bool)[:, ::IOU_STRIDE, ::IOU_STRIDE]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two stacks of masks, as a matmul: intersection = A @ B.T."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    flat_a = a.reshape(a.shape[0], -1).astype(np.float32)
    flat_b = b.reshape(b.shape[0], -1).astype(np.float32)
    inter = flat_a @ flat_b.T
    area_a = flat_a.sum(axis=1)[:, None]
    area_b = flat_b.sum(axis=1)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def compare_masks(baseline_dir: str, candidate_dir: str, log=print) -> dict:
    """Per-frame agreement vs a reference run. Matched greedily by IoU, not by object id —
    ids are only meaningful within one session and these are two."""
    import glob
    import os

    base_paths = sorted(glob.glob(os.path.join(baseline_dir, "frame_*.npz")))
    cand_paths = sorted(glob.glob(os.path.join(candidate_dir, "frame_*.npz")))
    if not base_paths or not cand_paths:
        raise SystemExit(f"no mask archives in {baseline_dir} or {candidate_dir}")
    if len(base_paths) != len(cand_paths):
        log(f"  !! frame counts differ ({len(base_paths)} vs {len(cand_paths)}); "
            f"comparing the first {min(len(base_paths), len(cand_paths))}")

    best_ious, matched_50, n_base, n_cand = [], [], [], []
    for base_path, cand_path in zip(base_paths, cand_paths):
        base_masks = _load_frame_masks(base_path)
        cand_masks = _load_frame_masks(cand_path)
        n_base.append(base_masks.shape[0])
        n_cand.append(cand_masks.shape[0])
        if base_masks.shape[0] == 0:
            continue
        # Best candidate per baseline object — recall-flavoured, so emitting MORE objects
        # cannot inflate it.
        best = _iou_matrix(base_masks, cand_masks).max(axis=1) if cand_masks.shape[0] \
            else np.zeros(base_masks.shape[0])
        best_ious.append(float(best.mean()))
        matched_50.append(float((best >= 0.5).mean()))

    stats = {
        "frames": len(best_ious),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "frac_iou_50": float(np.mean(matched_50)) if matched_50 else float("nan"),
        "baseline_objects": float(np.mean(n_base)) if n_base else 0.0,
        "candidate_objects": float(np.mean(n_cand)) if n_cand else 0.0,
    }
    log(f"\n  vs baseline {baseline_dir}:")
    log(f"    objects/frame     {stats['baseline_objects']:.1f} -> "
        f"{stats['candidate_objects']:.1f}")
    log(f"    mean best IoU     {stats['mean_best_iou']:.3f}   "
        f"(per baseline object, greedy match, stride {IOU_STRIDE})")
    log(f"    frac IoU >= 0.50  {stats['frac_iou_50']:.3f}")
    return stats


def _table(rows):
    # degC/MHz are what the row STARTED at. A sweep runs presets back to back, so on a card
    # that heats up ms/frame rises with execution order regardless of treatment.
    print("\n preset          ms/frame     Hz   objects  ms/obj  stable-ids   degC    MHz")
    for r in rows:
        ms, objects = r["ms"], r["objects"]
        hz = 1000.0 / ms if ms == ms else float("nan")
        per = ms / max(objects, 1) if ms == ms and objects == objects else float("nan")
        gpu = r.get("gpu_before") or {}
        temp = f"{gpu['temp_c']:.0f}" if gpu else "-"
        mhz = f"{gpu['sm_mhz']:.0f}" if gpu else "-"
        print(f" {r['label']:<14} {ms:8.0f} {hz:6.2f} {objects:9.1f} {per:7.0f} "
              f"{r['persistent_ids']:11d} {temp:>6} {mhz:>6}")

    temps = [r["gpu_before"]["temp_c"] for r in rows if r.get("gpu_before")]
    if len(temps) >= 2 and max(temps) - min(temps) >= 5:
        print(f"\n !! rows started {min(temps):.0f}-{max(temps):.0f}C apart — compare "
              f"objects and stable-ids\n    (thermally invariant); the ms column is not an "
              f"A/B until temperatures match.")


def _run_in_subprocess(base_argv: list[str], label: str, extra: list[str]) -> dict | None:
    """One sweep point in a fresh interpreter.

    In-process reloads died on the SECOND model load (docs/M2_perception.md 3.6). A
    subprocess also gives each preset a clean CUDA allocator.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.json")
        argv = [sys.executable, "-m", "sam_mapper.sam3_backend", *base_argv, *extra,
                "--label", label, "--child-json", out]
        print(f"\n=== {label} ===  ({' '.join(extra) or 'as configured'})")
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0 or not os.path.exists(out):
            print(f"  {label} FAILED: child exited {proc.returncode}")
            return None
        with open(out) as handle:
            return json.load(handle)


def main(argv=None):
    import argparse
    import os

    import yaml

    parser = argparse.ArgumentParser(description="Probe SAM 3 on a folder of frames.")
    parser.add_argument("--frames", required=True, help="directory of frames, in order")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="max frames; 0 (default) = every frame in the directory")
    parser.add_argument("--sweep-image-size", action="store_true",
                        help="time every IMAGE_SIZE_PRESETS entry, one subprocess each")
    parser.add_argument("--prompts",
                        help="comma-separated prompts; ones absent from the config are added "
                             "as instance objects, matching /sam3/set_prompts")
    parser.add_argument("--dtype", help="override sam3.dtype")
    parser.add_argument("--verbose", action="store_true",
                        help="print every 2D detection (label/id/score/bbox) for every frame, "
                             "not just frame 0 — no lidar here, so 2D only")
    parser.add_argument("--save-annotated", action="store_true",
                        help="write mask/box/label overlays to <frames>/annotated/ for visual "
                             "validation — reuses sam_mapper.annotate (the /annotated_image code)")
    parser.add_argument("--profile", action="store_true",
                        help="per-stage breakdown + a ms-vs-object-count fit")
    parser.add_argument("--torch-profile", action="store_true",
                        help="torch.profiler over 3 frames + chrome trace; use when "
                             "--profile shows a large off-seam residual")
    parser.add_argument("--image-size", help="override sam3.image_size, e.g. 672 or 672x672")
    parser.add_argument("--model-id",
                        help="override sam3.model_id — A/B a local checkpoint dir without "
                             "writing a second config")
    parser.add_argument("--attn",
                        help="override sam3.attn_implementation, e.g. sdpa or "
                             "kernels-community/flash-attn2")
    parser.add_argument("--fill-hole-area", type=int,
                        help="override sam3.fill_hole_area; 0 disables cv-utils hole "
                             "filling and sprinkle removal")
    parser.add_argument("--save-masks", help="write per-frame mask archives here, for a "
                                             "later --compare-baseline")
    parser.add_argument("--compare-baseline",
                        help="reference mask dir; reports objects/frame, mean best IoU "
                             "and frac IoU>=0.5 against it")
    # Sweep plumbing: each preset re-invokes this module as a child (see
    # _run_in_subprocess), and the child reports back through --child-json.
    parser.add_argument("--label", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    full_cfg = yaml.safe_load(open(args.config))
    report_device()
    frames, paths = _load_frames(args.frames)
    available = len(frames)
    if args.limit:
        frames, paths = frames[: args.limit], paths[: args.limit]
    print(f"loaded {len(frames)}/{available} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    save_annotated = os.path.join(args.frames, "annotated") if args.save_annotated else None

    base = dict(full_cfg["sam3"])
    base["_objects"] = full_cfg["objects"]
    if args.dtype:
        base["dtype"] = args.dtype
    if args.image_size:
        side = [int(v) for v in args.image_size.lower().split("x")]
        base["image_size"] = side * 2 if len(side) == 1 else side
    if args.model_id:
        base["model_id"] = args.model_id
    if args.attn:
        base["attn_implementation"] = args.attn
    if args.fill_hole_area is not None:
        base["fill_hole_area"] = args.fill_hole_area
    if args.prompts:
        wanted = [p.strip() for p in args.prompts.split(",") if p.strip()]
        by_prompt = {o["prompt"]: o for o in full_cfg["objects"]}
        # Unlisted prompts are ADDED as instance objects, matching what
        # sam_node._on_set_prompts does with a question's targets.
        base["_objects"] = [by_prompt.get(p, {"prompt": p, "instance": True}) for p in wanted]
        if added := [p for p in wanted if p not in by_prompt]:
            print(f"  prompts not in config, added as instance objects "
                  f"(matches /sam3/set_prompts): {added}")

    if args.torch_profile:
        _torch_profile(frames, base, args.frames)
        return

    if args.sweep_image_size:
        # One subprocess per preset; --image-size is what each overrides, and --sweep-*
        # itself is dropped or every child would recurse.
        inherited = ["--frames", args.frames, "--config", args.config,
                     "--limit", str(args.limit)]
        for flag, value in (("--dtype", args.dtype), ("--model-id", args.model_id),
                            ("--attn", args.attn), ("--prompts", args.prompts),
                            ("--fill-hole-area", None if args.fill_hole_area is None
                             else str(args.fill_hole_area))):
            if value:
                inherited += [flag, value]
        for flag, on in (("--verbose", args.verbose), ("--save-annotated", args.save_annotated),
                         ("--profile", args.profile)):
            if on:
                inherited.append(flag)
        rows = []
        for name, size in IMAGE_SIZE_PRESETS.items():
            extra = ["--image-size", str(size[0])]
            if args.save_masks:
                # Per-preset dir, else presets overwrite each other's archives.
                extra += ["--save-masks", os.path.join(args.save_masks, name)]
            if (row := _run_in_subprocess(inherited, name, extra)) is not None:
                rows.append(row)
        _table(rows)
        return

    # -- single run -----------------------------------------------------------
    label = args.label or str(base.get("image_size", "default"))
    row = _run(frames, base, label, verbose=args.verbose, save_annotated=save_annotated,
               paths=paths, profile=args.profile, save_masks=args.save_masks)

    if args.compare_baseline:
        if not args.save_masks:
            raise SystemExit("--compare-baseline needs --save-masks to write this run's masks")
        row["comparison"] = compare_masks(args.compare_baseline, args.save_masks)

    if args.child_json:
        import json

        with open(args.child_json, "w") as handle:
            json.dump(row, handle)


if __name__ == "__main__":
    main()
