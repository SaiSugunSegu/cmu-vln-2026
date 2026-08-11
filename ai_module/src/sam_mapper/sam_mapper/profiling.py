"""Per-stage timing for a SAM 3 frame.

`Sam3VideoModel._det_track_one_frame` calls seven named methods in sequence; StageTimer
wraps them on the model INSTANCE (no transformers fork) and `detach()` restores them.

Every boundary is a `torch.cuda.synchronize()` — that is what makes the numbers
attributable, and why this stays behind `--profile` / `runtime.profile`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps

import numpy as np

# (attribute path on the model, report label), in call order. A path that no longer resolves
# is reported in `missing` rather than raising — transformers renames things.
MODEL_STAGES = (
    ("detector_model.get_vision_features", "vision_encoder"),   # the image_size lever
    ("run_detection", "detection"),                             # loops per PROMPT
    ("get_vision_features_for_tracker", "tracker_neck"),
    ("run_tracker_propagation", "tracker_propagate"),           # the SAM 3.1 lever
    ("run_tracker_update_planning_phase", "tracker_plan"),
    ("run_tracker_update_execution_phase", "tracker_execute"),
    ("build_outputs", "build_outputs"),
)

# Timed by Sam3Backend around the model call.
BACKEND_STAGES = ("preprocess", "postprocess", "to_numpy")

STAGE_ORDER = tuple(label for _, label in MODEL_STAGES) + BACKEND_STAGES


def _resolve(root, path: str):
    """'a.b.c' -> (owner_of_c, 'c', bound_c), or None if any hop is missing."""
    owner = root
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    name = parts[-1]
    attr = getattr(owner, name, None)
    return None if attr is None else (owner, name, attr)


class StageTimer:
    """Per-stage wall-clock ms, one record per frame. Disabled instances are a true no-op."""

    # A stage smaller than this cannot be fitted — its range is below the timing jitter, so
    # least squares returns a large slope with a nonsense intercept.
    FIT_FLOOR_FRACTION = 0.01
    FIT_FLOOR_MS = 0.5
    FIT_MIN_R2 = 0.5

    def __init__(self, enabled: bool = True, torch_module=None):
        self.enabled = bool(enabled)
        self._torch = torch_module
        self._patched: list[tuple[object, str, object]] = []
        self._current: dict[str, float] = {}
        self.frames: list[dict] = []
        self.missing: list[str] = []
        self._held = False              # an outer frame() owns the boundary
        self._pending = (float("nan"), 0, 0)
        # Resolved once: _sync runs twice per stage and the answer cannot change mid-run.
        self._needs_sync = bool(torch_module is not None and torch_module.cuda.is_available())

    # -- lifecycle ---------------------------------------------------------

    def attach(self, model) -> "StageTimer":
        if not self.enabled or self._patched:
            return self
        if self._torch is None:
            # Optional: without torch there is no CUDA queue to drain, which is also what
            # lets the unit tests run on a host with no torch.
            try:
                import torch

                self._torch = torch
                self._needs_sync = torch.cuda.is_available()
            except ImportError:
                pass
        for path, label in MODEL_STAGES:
            found = _resolve(model, path)
            if found is None:
                self.missing.append(path)
                continue
            owner, name, original = found
            setattr(owner, name, self._wrap(label, original))
            self._patched.append((owner, name, original))
        return self

    def detach(self) -> None:
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()

    def _wrap(self, label: str, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with self.stage(label):
                return fn(*args, **kwargs)
        return wrapper

    # -- measurement -------------------------------------------------------

    def _sync(self) -> None:
        if self._needs_sync:
            self._torch.cuda.synchronize()

    @contextmanager
    def stage(self, label: str):
        """Time a named stage. Re-entering a label within a frame accumulates."""
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._current[label] = (self._current.get(label, 0.0)
                                    + (time.perf_counter() - start) * 1000.0)

    @contextmanager
    def frame(self):
        """Own the frame boundary from outside process_frame.

        sam_node's post-SAM stages run after process_frame returns; without this they land
        in the NEXT frame's record and the whole table is off by one.
        """
        if not self.enabled:
            yield
            return
        self._held = True
        self._pending = (float("nan"), 0, 0)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._held = False
            _, n_objects, n_prompts = self._pending
            self.end_frame((time.perf_counter() - start) * 1000.0, n_objects, n_prompts)

    def end_frame(self, total_ms: float, n_objects: int, n_prompts: int) -> None:
        """Close the frame. `total_ms` is the caller's wall clock, checked against the sum."""
        if not self.enabled:
            return
        if self._held:
            self._pending = (total_ms, n_objects, n_prompts)
            return
        record = dict(self._current)
        record["_total"] = float(total_ms)
        record["_objects"] = int(n_objects)
        record["_prompts"] = int(n_prompts)
        self.frames.append(record)
        self._current = {}

    # -- reporting ---------------------------------------------------------

    def summary(self, skip_first: int = 1) -> dict:
        """Median ms per stage. Median not mean: early frames legitimately do more work
        (every object is new) and one such frame moves a mean enough to reorder the table."""
        frames = self.frames[skip_first:]
        if not frames:
            return {"frames": 0, "stages": {}, "total_ms": float("nan"),
                    "accounted_ms": float("nan"), "objects": float("nan"),
                    "prompts": 0, "missing": list(self.missing)}

        labels = [s for s in STAGE_ORDER if any(s in f for f in frames)]
        labels += sorted({k for f in frames for k in f
                          if not k.startswith("_") and k not in STAGE_ORDER})
        stages = {label: float(np.median([f.get(label, 0.0) for f in frames]))
                  for label in labels}
        return {
            "frames": len(frames),
            "stages": stages,
            "total_ms": float(np.median([f["_total"] for f in frames])),
            "accounted_ms": float(sum(stages.values())),
            "objects": float(np.mean([f["_objects"] for f in frames])),
            "prompts": int(frames[-1]["_prompts"]),
            "missing": list(self.missing),
        }

    def per_object_fit(self, skip_first: int = 1) -> dict:
        """Least squares of each stage against object count: ms ~ fixed + per_object * N.

        Only a stage with a real per-object slope is something SAM 3.1's Object Multiplex
        could share work across. The fit is only as good as the object-count spread the run
        produced, so `counts` reports the actual range.
        """
        frames = self.frames[skip_first:]
        counts = np.array([f["_objects"] for f in frames], dtype=float)
        spread = int(len(set(counts.tolist())))
        observed = {"spread": spread,
                    "counts": (int(counts.min()), int(counts.max())) if len(counts) else (0, 0)}
        if len(frames) < 3 or spread < 2:
            return {**observed, "fits": {}, "skipped": []}

        summary = self.summary(skip_first)
        floor = max(self.FIT_FLOOR_MS, self.FIT_FLOOR_FRACTION * summary["total_ms"])
        design = np.stack([np.ones_like(counts), counts], axis=1)
        fits, skipped = {}, []
        for label, median_ms in summary["stages"].items():
            if median_ms < floor:
                skipped.append(label)
                continue
            values = np.array([f.get(label, 0.0) for f in frames], dtype=float)
            (fixed, per_object), *_ = np.linalg.lstsq(design, values, rcond=None)
            residual = values - design @ np.array([fixed, per_object])
            centred = values - values.mean()
            r2 = 1.0 - (residual @ residual) / (centred @ centred) if centred.any() else 0.0
            fits[label] = {"fixed_ms": float(fixed), "per_object_ms": float(per_object),
                           "r2": float(r2), "reliable": bool(r2 >= self.FIT_MIN_R2)}
        return {**observed, "fits": fits, "skipped": skipped, "floor_ms": float(floor)}


def format_summary(summary: dict, title: str = "") -> str:
    """The stage table. A large `residual` means a cost is off-seam — use --torch-profile."""
    if not summary["frames"]:
        return "  (no frames profiled)"

    total = summary["total_ms"]
    lines = []
    if title:
        lines.append(f"\n=== {title} ===")
    lines.append(f"  median over {summary['frames']} frames (frame 0 dropped as warm-up), "
                 f"{summary['objects']:.1f} objects/frame, {summary['prompts']} prompts")
    lines.append("")
    lines.append("  stage                    ms      %      <- ms is PER FRAME, not a total")
    for label, ms in sorted(summary["stages"].items(), key=lambda kv: -kv[1]):
        share = 100.0 * ms / total if total else float("nan")
        lines.append(f"  {label:<20} {ms:7.1f} {share:6.1f}")
    residual = total - summary["accounted_ms"]
    lines.append("  " + "-" * 34)
    lines.append(f"  {'accounted':<20} {summary['accounted_ms']:7.1f} "
                 f"{100.0 * summary['accounted_ms'] / total if total else float('nan'):6.1f}")
    lines.append(f"  {'residual (off-seam)':<20} {residual:7.1f} "
                 f"{100.0 * residual / total if total else float('nan'):6.1f}")
    lines.append(f"  {'TOTAL':<20} {total:7.1f} {100.0:6.1f}   "
                 f"({1000.0 / total if total else float('nan'):.2f} Hz)")
    if abs(residual) > 0.1 * total:
        lines.append("  !! >10% off-seam — a cost is hiding; re-run with --torch-profile")
    if summary["missing"]:
        lines.append(f"  !! stages not found on this model: {summary['missing']}")
    return "\n".join(lines)


def format_per_object_fit(fit: dict) -> str:
    low, high = fit.get("counts", (0, 0))
    if not fit["fits"]:
        return (f"\n  (object-count fit needs >=3 frames with >=2 distinct counts; "
                f"saw {fit['spread']} distinct in {low}-{high} objects)")

    lines = ["", f"  ms ~ fixed + per_object * N, over {low}-{high} objects/frame",
             "  stage                 fixed ms   ms/object      r2",
             "  " + "-" * 51]
    for label, f in sorted(fit["fits"].items(), key=lambda kv: -kv[1]["per_object_ms"]):
        flag = "" if f["reliable"] else "  (low r2 — noise)"
        lines.append(f"  {label:<20} {f['fixed_ms']:9.1f} {f['per_object_ms']:11.2f} "
                     f"{f['r2']:7.2f}{flag}")
    if fit.get("skipped"):
        lines.append(f"\n  not fitted (median below {fit['floor_ms']:.1f} ms): "
                     f"{', '.join(fit['skipped'])}")
    if high - low < 5:
        lines.append(f"\n  !! object count only spans {low}-{high} — too narrow to separate "
                     f"fixed from\n     per-object cost. Re-run with more prompts.")
    return "\n".join(lines)


def gpu_state() -> dict:
    """Temperature / SM clock / power via nvidia-smi (already in the image; the torch
    equivalents need nvidia-ml-py, which is not). {} if unavailable."""
    import subprocess

    query = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        temp, sm, sm_max, power, limit = (float(f) for f in out.split(",")[:5])
    except Exception:                          # noqa: BLE001 — never break a run over this
        return {}
    return {"temp_c": temp, "sm_mhz": sm, "sm_max_mhz": sm_max,
            "power_w": power, "power_limit_w": limit}


def format_gpu_state(state: dict) -> str:
    """A throttled card rescales every timing, so back-to-back A/B rows get slower purely by
    execution order."""
    if not state:
        return "  gpu: state unavailable"
    ratio = state["sm_mhz"] / state["sm_max_mhz"] if state["sm_max_mhz"] else 1.0
    line = (f"  gpu: {state['temp_c']:.0f}C, SM {state['sm_mhz']:.0f}/"
            f"{state['sm_max_mhz']:.0f} MHz ({ratio:.0%}), "
            f"{state['power_w']:.0f}/{state['power_limit_w']:.0f} W")
    if state["temp_c"] >= 80:
        line += "\n  !! hot card — timings are not comparable across runs until it cools"
    return line


def format_thermal_drift(before: dict, after: dict) -> str:
    """Did the card slow down DURING the run? Then its own frames were not measured under
    constant conditions."""
    if not before or not after:
        return ""
    d_clock = after["sm_mhz"] - before["sm_mhz"]
    line = (f"  gpu drift: {before['temp_c']:.0f}C -> {after['temp_c']:.0f}C, "
            f"SM {before['sm_mhz']:.0f} -> {after['sm_mhz']:.0f} MHz ({d_clock:+.0f})")
    if d_clock <= -100:
        line += "\n  !! clocks fell mid-run — treat this as a lower bound"
    return line


def effective_attention(model) -> str:
    """What attention actually runs. The requested implementation tells you nothing:
    transformers substitutes a kernels-hub build when the pip package is absent."""
    declared = getattr(getattr(model, "config", None), "_attn_implementation", None)
    sub = [getattr(c, "config", None) for c in getattr(model, "children", lambda: [])()]
    nested = sorted({getattr(c, "_attn_implementation", None) for c in sub if c is not None}
                    - {None})
    impl_classes = sorted({type(m).__name__ for m in model.modules()
                           if "attention" in type(m).__name__.lower()})[:4]
    parts = [f"declared={declared!r}"]
    if nested and nested != [declared]:
        parts.append(f"submodels={nested}")
    if impl_classes:
        parts.append(f"modules={impl_classes}")
    return "  ".join(parts)
