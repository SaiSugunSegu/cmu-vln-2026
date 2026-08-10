"""SAM 3 video backend: text prompts in, tracked masks out.

Two rules govern how SAM 3 is driven, and both are load-bearing:

  1. ONE session, ALL prompts — vision features are shared across prompts, so N classes
     cost ~1 forward pass, not N.
  2. ONE session for the whole run — object ids are only stable within a session, and
     those ids are what the 3D mapper associates on.

Frames may be dropped (the tracker is memory-based, so that just lowers the effective
tracking rate) but must arrive in order.

Standalone probe:
    python -m sam_mapper.sam3_backend --frames DIR --config CFG.yaml [--sweep-image-size] [--verbose]
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

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

    def __init__(self, cfg: dict, log=print):
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

        for key in ("score_threshold_detection", "new_det_thresh", "det_nms_thresh",
                    "recondition_every_nth_frame", "max_trk_keep_alive",
                    "min_trk_keep_alive", "hotstart_delay"):
            if (value := cfg.get(key)) is not None and hasattr(self.model.config, key):
                setattr(self.model.config, key, value)

        self.prompts: list[str] = []
        self.session = None

    def _load(self, model_cls, model_id, kwargs):
        """Try the requested attention backend, degrade rather than die."""
        requested = self.cfg.get("attn_implementation", "flash_attention_2")
        errors = []
        for attn in [requested] + [a for a in ("sdpa", "eager") if a != requested]:
            try:
                model = model_cls.from_pretrained(model_id, attn_implementation=attn, **kwargs)
                self.log(f"[sam3] attn: {attn}"
                         + (f" ('{requested}' unavailable: {errors[0]})" if attn != requested
                            else ""))
                return model
            except Exception as err:            # noqa: BLE001 - degrade, then report
                errors.append(f"{attn}: {type(err).__name__}: {err}")
        raise RuntimeError(f"could not load {model_id} with any attention backend; "
                           + " | ".join(errors))

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
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(inference_session=self.session,
                                 frame=inputs.pixel_values[0], reverse=False)
            processed = self.processor.postprocess_outputs(
                self.session, outputs, original_sizes=inputs.original_sizes)
        return self._to_result(processed, height, width)

    @staticmethod
    def _to_result(processed: dict, height: int, width: int) -> Sam3FrameResult:
        object_ids = processed.get("object_ids")
        if object_ids is None or len(object_ids) == 0:
            return Sam3FrameResult.empty(height, width)

        def to_numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        masks = to_numpy(processed["masks"])
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return Sam3FrameResult(
            object_ids=to_numpy(object_ids).astype(int).reshape(-1),
            scores=to_numpy(processed["scores"]).astype(float).reshape(-1),
            boxes=to_numpy(processed["boxes"]).astype(float).reshape(-1, 4),
            masks=masks.astype(bool),
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


def _run(frames, cfg, label, log=print, verbose=False, save_annotated=None, paths=None):
    """save_annotated: directory to write annotate.annotate_frame() overlays to, or None."""
    import os

    from sam_mapper.detections import PromptTable, to_detections

    backend = Sam3Backend(cfg, log=log)
    table = PromptTable(cfg["_objects"])
    backend.set_prompts(table.prompts)

    if save_annotated:
        import cv2

        from sam_mapper.annotate import annotate_frame
        os.makedirs(save_annotated, exist_ok=True)

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

    if save_annotated:
        log(f"  saved {len(frames)} annotated frames to {save_annotated}")

    # Ids present in EVERY frame are stably tracked — the property that replaces
    # ByteTrack. Near zero here means tracking is not working.
    persistent = set.intersection(*per_frame_ids) if per_frame_ids else set()
    ms = float(np.median(times)) if times else float("nan")
    objects = float(np.mean(counts))
    log(f"  {label}: {ms:.0f} ms/frame ({1000.0 / ms:.2f} Hz), {objects:.1f} objects "
        f"({ms / max(objects, 1):.0f} ms/object), {len(persistent)} ids in all {len(frames)} frames")
    return {"label": label, "ms": ms, "objects": objects, "persistent_ids": len(persistent)}


def _table(rows):
    print("\n preset          ms/frame     Hz   objects  ms/obj  stable-ids")
    for r in rows:
        ms, objects = r["ms"], r["objects"]
        hz = 1000.0 / ms if ms == ms else float("nan")
        per = ms / max(objects, 1) if ms == ms and objects == objects else float("nan")
        print(f" {r['label']:<14} {ms:8.0f} {hz:6.2f} {objects:9.1f} {per:7.0f} "
              f"{r['persistent_ids']:11d}")


def main(argv=None):
    import argparse
    import copy
    import os

    import yaml

    parser = argparse.ArgumentParser(description="Probe SAM 3 on a folder of frames.")
    parser.add_argument("--frames", required=True, help="directory of frames, in order")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--sweep-image-size", action="store_true")
    parser.add_argument("--sweep-prompts", action="store_true",
                        help="all prompts vs instances-only vs 3 classes — tests whether "
                             "runtime scales with object count")
    parser.add_argument("--prompts", help="comma-separated prompt subset")
    parser.add_argument("--dtype", help="override sam3.dtype")
    parser.add_argument("--verbose", action="store_true",
                        help="print every 2D detection (label/id/score/bbox) for every frame, "
                             "not just frame 0 — no lidar here, so 2D only")
    parser.add_argument("--save-annotated", action="store_true",
                        help="write mask/box/label overlays to <frames>/annotated/ for visual "
                             "validation — reuses sam_mapper.annotate (the /annotated_image code)")
    args = parser.parse_args(argv)

    full_cfg = yaml.safe_load(open(args.config))
    report_device()
    frames, paths = _load_frames(args.frames)
    frames, paths = frames[: args.limit], paths[: args.limit]
    print(f"loaded {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
    save_annotated = os.path.join(args.frames, "annotated") if args.save_annotated else None

    base = dict(full_cfg["sam3"])
    base["_objects"] = full_cfg["objects"]
    if args.dtype:
        base["dtype"] = args.dtype
    if args.prompts:
        wanted = {p.strip() for p in args.prompts.split(",") if p.strip()}
        base["_objects"] = [o for o in full_cfg["objects"] if o["prompt"] in wanted]
        if missing := wanted - {o["prompt"] for o in base["_objects"]}:
            raise SystemExit(f"prompts not in config: {sorted(missing)}")

    if args.sweep_prompts:
        objects = full_cfg["objects"]
        instances = [o for o in objects if o.get("instance", True)]
        subsets = {
            f"all_{len(objects)}": objects,
            f"instances_{len(instances)}": instances,
            "three": [o for o in objects if o["prompt"] in ("chair", "table", "sofa")],
        }
    elif args.sweep_image_size:
        subsets = None
    else:
        _run(frames, base, str(base.get("image_size", "default")), verbose=args.verbose,
             save_annotated=save_annotated, paths=paths)
        return

    rows = []
    if subsets is not None:
        for name, objects in subsets.items():
            cfg = copy.deepcopy(base)
            cfg["_objects"] = objects
            print(f"\n=== {name}: {[o['prompt'] for o in objects]} ===")
            try:
                rows.append(_run(frames, cfg, name, verbose=args.verbose,
                                 save_annotated=save_annotated, paths=paths))
            except Exception as err:                          # noqa: BLE001 — sweep must not abort
                print(f"  {name} FAILED: {type(err).__name__}: {err}")
    else:
        for name, image_size in IMAGE_SIZE_PRESETS.items():
            cfg = copy.deepcopy(base)
            cfg["image_size"] = image_size
            print(f"\n=== {name}: {image_size} ===")
            try:
                rows.append(_run(frames, cfg, name, verbose=args.verbose,
                                 save_annotated=save_annotated, paths=paths))
            except Exception as err:                          # noqa: BLE001
                print(f"  {name} FAILED: {type(err).__name__}: {err}")
    _table(rows)


if __name__ == "__main__":
    main()
