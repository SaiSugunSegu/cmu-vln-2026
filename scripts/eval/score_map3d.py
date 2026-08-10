#!/usr/bin/env python3
"""Score a replayed 3D map against IRef-VLA ground truth.

Reports in priority order: is the right object in the map, where is it, how big is it —
plus the category-2 marker score computed on the Marker we would actually publish.

    just map3d-score [<run-id>] [--audit-labels]"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Runs in BOTH places, and the data root differs: the repo checkout on the host, /data in
# the container (where scripts/ is bind-mounted at /home/docker/scripts, so a repo-relative
# path resolves to the wrong place). Detect rather than require the caller to know.
#
# Prefer running this in the container — `just map3d-score` does — so that replay and score
# write /data/runs as the same uid. Mixing uids in one bind-mounted tree is what caused the
# scorer to fail on a directory replay had just created.
DATA_ROOT = os.environ.get("MAP3D_DATA_ROOT") or (
    os.path.join(REPO, "data") if os.path.isdir(os.path.join(REPO, "data")) else "/data")

# challenge_marker is pure numpy, so it imports with or without ROS. Importing it rather
# than re-deriving the marker geometry here is the point: the bench must score the SAME
# payload the node publishes, or it is benchmarking a fiction.
_SAM_MAPPER = os.path.join(REPO, "ai_module", "src", "sam_mapper")
if not os.path.isdir(_SAM_MAPPER):                      # container layout
    _SAM_MAPPER = "/home/docker/ai_module/src/sam_mapper"
sys.path.insert(0, _SAM_MAPPER)
from sam_mapper.box_geometry import (  # noqa: E402
    area as _area, box_iou_poly as _box_iou_poly, ccw as _ccw, clip as _clip,
    rect_from_center_extent as _rect_from_center_extent)
from sam_mapper.challenge_marker import (  # noqa: E402
    marker_payload, quaternion_to_yaw)

IOU_LEVELS = (0.25, 0.50)
SMALL_MAX_DIM = 0.5        # split point for the small/large error breakdown

# How far from the robot path a GT object must be to count as plausibly observable.
# Mirrors mapping.range_filter.max_distance: beyond it the mapper discards points by
# construction, so counting those objects as misses would grade M2 for M1's coverage.
OBSERVABLE_RANGE_M = 5.0

# Genuine synonyms between a SAM 3 prompt and a VLA-3D raw_label. Kept deliberately SHORT:
# every entry here inflates recall if it is wrong, and the exact-match path below already
# covers the common case. Extend only from --audit-labels output, never speculatively.
SYNONYMS = {
    "sofa": {"sofa", "couch"},
    "couch": {"couch", "sofa"},
    "picture": {"picture", "painting", "photo", "poster", "framed record"},
    "painting": {"painting", "picture", "photo", "poster"},
    "carpet": {"carpet", "rug"},
    "potted plant": {"potted plant", "pottedplant", "plant"},
    "nightstand": {"nightstand", "night stand", "bedside table"},
    "night stand": {"night stand", "nightstand", "bedside table"},
    "bedside table": {"bedside table", "nightstand", "night stand"},
    "tv": {"tv", "television"},
    "book": {"book", "books"},
    "curtain": {"curtain", "curtains"},
    "window": {"window", "windows"},
}


# ---------------------------------------------------------------- geometry


def obb_iou(a: "Box", b: "Box") -> float:
    """

Exact 3D IoU for two gravity-aligned boxes. Thin adapter over box_geometry."""
    return _box_iou_poly(a.footprint, a.z0, a.z1, a.volume,
                         b.footprint, b.z0, b.z1, b.volume)


def aabb_iou(a: "Box", b: "Box") -> float:
    """

IoU of the two boxes' world-axis-aligned envelopes — what score.py compares."""
    lo = np.maximum(a.aabb_lo, b.aabb_lo)
    hi = np.minimum(a.aabb_hi, b.aabb_hi)
    d = hi - lo
    if np.any(d <= 0):
        return 0.0
    inter = float(np.prod(d))
    va = float(np.prod(a.aabb_hi - a.aabb_lo))
    vb = float(np.prod(b.aabb_hi - b.aabb_lo))
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Box:
    label: str
    center: np.ndarray
    footprint: np.ndarray      # (4,2) XY corners
    z0: float
    z1: float
    obj_id: str = ""
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    #: map_node publishes TWO points and they differ whenever voxels are unevenly
    #: distributed — i.e. always, for a one-sided view. `center` above is bbox3d.center (the
    #: box midpoint); this is the diversity-weighted voxel centroid. Reported side by side
    #: so the choice of which to publish is made on evidence.
    voxel_centroid: np.ndarray | None = None
    n_ids: int = 1

    @property
    def volume(self) -> float:
        return _area(self.footprint) * max(self.z1 - self.z0, 0.0)

    @property
    def aabb_lo(self) -> np.ndarray:
        return np.array([self.footprint[:, 0].min(), self.footprint[:, 1].min(), self.z0])

    @property
    def aabb_hi(self) -> np.ndarray:
        return np.array([self.footprint[:, 0].max(), self.footprint[:, 1].max(), self.z1])

    @property
    def max_dim(self) -> float:
        return float(np.max(self.size)) if self.size.any() else 0.0


# ---------------------------------------------------------------- loading


def load_gt(scene: str) -> list[Box]:
    hits = glob.glob(os.path.join(DATA_ROOT, "bags", scene, "iref_vla_metadata",
                                  "*_object_data.json"))
    if not hits:
        return []
    boxes = []
    for region in json.load(open(hits[0])).values():
        for obj in region.values():
            corners = np.asarray(obj["bbox"], dtype=float)
            # Corners 0-3 are the top face in order, so they already ARE the rotated
            # footprint — no need to recover a heading and risk its convention.
            boxes.append(Box(
                label=(obj.get("raw_label") or "").strip().lower(),
                center=np.asarray(obj["center"], dtype=float),
                footprint=corners[0:4, :2],
                z0=float(corners[:, 2].min()),
                z1=float(corners[:, 2].max()),
                obj_id=str(obj.get("object_id")),
                size=np.asarray(obj["size"], dtype=float),
            ))
    return boxes


def load_pred(path: str) -> tuple[list[Box], dict]:
    report = json.load(open(path))
    boxes = []
    for oid, obj in report.get("objects", {}).items():
        bbox = obj["bbox3d"]
        center = np.asarray(bbox["center"], dtype=float)
        extent = np.asarray(bbox["extent"], dtype=float)
        yaw = quaternion_to_yaw(bbox.get("rotation", [0, 0, 0, 1]))
        raw_centroid = obj.get("center")
        boxes.append(Box(
            label=str(obj.get("label", "")).strip().lower(),
            center=center,
            footprint=_rect_from_center_extent(center, extent, yaw),
            z0=float(center[2] - extent[2] / 2.0),
            z1=float(center[2] + extent[2] / 2.0),
            obj_id=str(oid),
            size=extent,
            yaw=yaw,
            voxel_centroid=np.asarray(raw_centroid, dtype=float) if raw_centroid else None,
            n_ids=len(obj.get("id", [oid])),
        ))
    return boxes, report


def question_target_ids(scene: str) -> set[str]:
    """GT object ids the questions actually ask about, per scene.

    Scoring every GT object grades us on furniture nobody asks about. None means "no question
    file" (score everything); an empty set means "asked about nothing"."""
    ids: set[str] = set()
    for path in glob.glob(os.path.join(DATA_ROOT, "benchmark", scene, "category_1",
                                       "*_qa.json")):
        data = json.load(open(path))
        for q in (data.get("questions", []) if isinstance(data, dict) else data):
            for oid in (q.get("object_ids") or []):
                ids.add(str(oid))
    return ids


# ---------------------------------------------------------------- labels


def build_label_map(prompts: list[str]) -> dict[str, set[str]]:
    """

Predicted label -> the set of GT raw_labels that should count as a match.

    Predicted labels are default_label(prompt) — the prompt lowercased with spaces
    stripped ('potted plant' -> 'pottedplant'). So the mapping back is mechanical, and
    the only judgement is the short SYNONYMS table above.
    """
    out: dict[str, set[str]] = {}
    for prompt in prompts:
        p = prompt.strip().lower()
        key = p.replace(" ", "")
        accepted = set(SYNONYMS.get(p, {p}))
        accepted.add(p)
        out.setdefault(key, set()).update(accepted)
    return out


def compatible(pred: Box, gt: Box, label_map: dict[str, set[str]]) -> bool:
    return gt.label in label_map.get(pred.label, {pred.label})


# ---------------------------------------------------------------- scoring


def associate(preds: list[Box], gts: list[Box], label_map, iou_fn, thresh: float):
    """

Greedy matching by descending IoU within label-compatible pairs.

    Returns (matches, dup_count). A prediction whose best GT is already taken is a
    DUPLICATE, not simply a false positive — that distinction is the whole point of the
    duplicate-rate metric, and a plain FP count would hide it.
    """
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if not compatible(p, g, label_map):
                continue
            iou = iou_fn(p, g)
            if iou >= thresh:
                pairs.append((iou, pi, gi))
    pairs.sort(key=lambda t: -t[0])

    matched_p, matched_g, matches, dups = set(), set(), [], 0
    for iou, pi, gi in pairs:
        if pi in matched_p:
            continue
        if gi in matched_g:
            dups += 1                       # this prediction wants an already-claimed GT
            matched_p.add(pi)
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        matches.append((pi, gi, iou))
    return matches, dups, matched_g


def box_from_payload(payload: dict, honour_orientation: bool) -> Box:
    """Rebuild a Box from a published Marker payload, the way a scorer would read it.

    `honour_orientation=False` reproduces score.py, which treats pose.orientation as identity
    and compares axis-aligned envelopes."""
    center = np.asarray(payload["position"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    yaw = quaternion_to_yaw(payload["orientation"]) if honour_orientation else 0.0
    return Box(
        label=payload.get("text", ""),
        center=center,
        footprint=_rect_from_center_extent(center, scale, yaw),
        z0=float(center[2] - scale[2] / 2.0),
        z1=float(center[2] + scale[2] / 2.0),
        size=scale,
    )


def gt_as_axis_aligned(gt: Box) -> Box:
    """

GT rebuilt as centre +- size/2, discarding its heading.

    This is what scripts/eval/score.py compares against, and what a scorer reading the
    published gt.json (centre + size) would use. It is NOT the same as gt's true footprint
    whenever the object is rotated, which is ~45% of them.
    """
    return Box(label=gt.label, center=gt.center,
               footprint=_rect_from_center_extent(gt.center, gt.size, 0.0),
               z0=float(gt.center[2] - gt.size[2] / 2.0),
               z1=float(gt.center[2] + gt.size[2] / 2.0),
               obj_id=gt.obj_id, size=gt.size)


def challenge_bench(preds, askable, label_map, targets) -> dict:
    """Score category 2 as the Marker we would publish, in priority order.

    Selection is ORACLE (best label-compatible candidate per target), so this is the ceiling a
    reasoner works under, not a prediction. `naive_oriented` picks by largest volume instead;
    the gap between them is what duplicates cost."""
    rows = []
    for g in askable:
        if targets and g.obj_id not in targets:
            continue
        candidates = [p for p in preds if compatible(p, g, label_map)]
        if not candidates:
            rows.append({"gt_id": g.obj_id, "label": g.label, "oriented": 0.0,
                         "axis_aligned": 0.0, "found": False})
            continue

        best = max(candidates, key=lambda p: obb_iou(p, g))
        payload = marker_payload(best.center, best.size, yaw=best.yaw, label=best.label)

        # NAIVE selection, alongside the oracle. cat2's oracle picks the best-matching
        # candidate, so producing MORE objects makes it easier — measured, a change that
        # took duplicate_rate 0.029 -> 0.189 "raised" cat2 by 0.139 while precision fell.
        # M4 will not have an oracle. Largest-volume is what an uninformed selector does,
        # and the gap between the two columns is what duplicates actually cost.
        naive = max(candidates, key=lambda p: p.volume)
        naive_payload = marker_payload(naive.center, naive.size, yaw=naive.yaw,
                                       label=naive.label)

        rows.append({
            "gt_id": g.obj_id,
            "label": g.label,
            "oriented": 2.0 * obb_iou(box_from_payload(payload, True), g),
            "axis_aligned": 2.0 * aabb_iou(box_from_payload(payload, False),
                                           gt_as_axis_aligned(g)),
            "naive_oriented": 2.0 * obb_iou(box_from_payload(naive_payload, True), g),
            # Priority order is: right object, then where it is, then how big it is. IoU
            # conflates the last two and ignores the middle one, so report the centre
            # distances directly. Both of the points map_node publishes are scored, because
            # they differ whenever an object was seen from one side.
            "center_err": float(np.linalg.norm(best.center - g.center)),
            "centroid_err": (float(np.linalg.norm(best.voxel_centroid - g.center))
                             if best.voxel_centroid is not None else None),
            "n_candidates": len(candidates),
            "found": True,
        })

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def median(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.median(vals)), 3) if vals else None

    found = [r for r in rows if r["found"]]
    return {
        "n_targets": len(rows),
        "n_found": sum(r["found"] for r in rows),
        # -- priority 1: is the right object in the map at all? ------------------------
        "found_rate": round(len(found) / len(rows), 3) if rows else None,
        # -- priority 2: where is it? (metres, over the found targets) -----------------
        "center_err_median": median("center_err"),
        "center_err_p90": (round(float(np.percentile([r["center_err"] for r in found], 90)), 3)
                           if found else None),
        "centroid_err_median": median("centroid_err"),
        # -- priority 3: how big is it? ------------------------------------------------
        "mean_score_oriented_of_2": mean("oriented"),
        "mean_score_axis_aligned_of_2": mean("axis_aligned"),
        # -- what duplicates cost: oracle minus naive selection ------------------------
        "mean_score_naive_selection_of_2": mean("naive_oriented"),
        "mean_candidates_per_target": mean("n_candidates"),
        # Publishing the wireframe puts all geometry in marker.points and leaves pose at
        # identity with scale.x as a line width — every pose+scale reader scores it 0.
        "mean_score_if_wireframe_of_2": 0.0,
        "per_target": rows,
    }


def observable(gt: Box, trajectory: np.ndarray) -> bool:
    if trajectory.size == 0:
        return True
    return bool(np.min(np.linalg.norm(trajectory - gt.center, axis=1)) <= OBSERVABLE_RANGE_M)


def score_scene(scene: str, pred_path: str) -> dict:
    preds, report = load_pred(pred_path)
    gts = load_gt(scene)
    if not gts:
        return {"scene": scene, "error": "no GT metadata"}

    prompts = report.get("source", {}).get("prompts", [])
    label_map = build_label_map(prompts)
    trajectory = np.asarray(report.get("trajectory", []), dtype=float)

    # Only GT whose label is in the prompt set can possibly be found. Scoring against all
    # 85 objects when 16 classes were prompted would report a recall governed by the
    # prompt set, not by the mapper.
    askable = [g for g in gts if any(g.label in acc for acc in label_map.values())]
    targets = question_target_ids(scene)

    out = {
        "scene": scene,
        "n_pred": len(preds),
        "n_gt_total": len(gts),
        "n_gt_askable": len(askable),
        "n_gt_observable": sum(observable(g, trajectory) for g in askable),
        "n_question_targets": len(targets),
        "frames_processed": report.get("frames_processed"),
        "update_map_ms": report.get("update_map_ms"),
        "map_digest": report.get("map_digest"),
    }

    for name, iou_fn in (("aabb", aabb_iou), ("obb", obb_iou)):
        for thresh in IOU_LEVELS:
            matches, dups, _matched_g = associate(preds, askable, label_map, iou_fn, thresh)
            tp = len(matches)
            precision = tp / len(preds) if preds else 0.0
            recall = tp / len(askable) if askable else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            obs = [i for i, g in enumerate(askable) if observable(g, trajectory)]
            rec_obs = (len({gi for _, gi, _ in matches} & set(obs)) / len(obs)) if obs else 0.0
            tgt = [i for i, g in enumerate(askable) if g.obj_id in targets]
            rec_tgt = (len({gi for _, gi, _ in matches} & set(tgt)) / len(tgt)) if tgt else None

            key = f"{name}@{thresh:g}"
            out[key] = {
                "tp": tp,
                "precision": round(precision, 3),
                "recall_askable": round(recall, 3),
                "recall_observable": round(rec_obs, 3),
                "recall_question_targets": round(rec_tgt, 3) if rec_tgt is not None else None,
                "f1": round(f1, 3),
                "duplicate_rate": round(dups / max(tp + dups, 1), 3),
                "mean_iou": round(float(np.mean([m[2] for m in matches])), 3) if matches else 0.0,
            }

    # Over-merge is NOT visible in the association above: greedy matching gives each
    # prediction at most one GT, so counting matched GT can never reveal that one box
    # swallowed two chairs. Measure it directly — a prediction that geometrically contains
    # the centres of two or more label-compatible GT objects has fused them.
    swallowed = []
    for p in preds:
        lo, hi = p.aabb_lo, p.aabb_hi
        inside = sum(1 for g in askable
                     if compatible(p, g, label_map) and np.all(g.center >= lo) and np.all(g.center <= hi))
        swallowed.append(inside)
    out["over_merge"] = {
        "rate": round(sum(1 for n in swallowed if n >= 2) / len(preds), 3) if preds else 0.0,
        "max_gt_in_one_prediction": max(swallowed, default=0),
        "gt_absorbed_by_multi_boxes": sum(n for n in swallowed if n >= 2),
    }

    # Threshold-free box quality. Precision/recall at a fixed IoU are step functions: on
    # the first 20-frame sample EVERY prediction scored 0 at 0.25 (best actual IoUs were
    # 0.068 and 0.216), so a fix that moved 0.07 -> 0.24 would look like no change at all.
    # These two move continuously and are what the Tier-1/2 A/Bs should be read against.
    #   per_gt      - includes GT with no compatible prediction, so it mixes in recall
    #   per_matched - only GT that some prediction claims, isolating pure box quality
    best_per_gt, best_matched = [], []
    for g in askable:
        cands = [obb_iou(p, g) for p in preds if compatible(p, g, label_map)]
        best = max(cands, default=0.0)
        best_per_gt.append(best)
        if cands:
            best_matched.append(best)
    near = sum(1 for v in best_per_gt if 0.1 <= v < 0.25)
    out["box_quality"] = {
        "mean_best_iou_per_gt": round(float(np.mean(best_per_gt)), 4) if best_per_gt else 0.0,
        "mean_best_iou_per_claimed_gt": round(float(np.mean(best_matched)), 4) if best_matched else 0.0,
        "n_claimed_gt": len(best_matched),
        "near_misses_0.1_to_0.25": near,
    }

    # Error breakdown uses the loosest threshold: a box can only have a meaningful error
    # if it was matched at all, and 0.25 keeps the badly-sized-but-correct objects in.
    matches, _, _ = associate(preds, askable, label_map, aabb_iou, 0.25)
    small, large = [], []
    for pi, gi, _iou in matches:
        p, g = preds[pi], askable[gi]
        row = {
            "center_err": float(np.linalg.norm(p.center - g.center)),
            "extent_err": np.abs(np.sort(p.size)[::-1] - np.sort(g.size)[::-1]).tolist(),
        }
        (small if g.max_dim < SMALL_MAX_DIM else large).append(row)

    def summarise(rows):
        if not rows:
            return None
        ce = np.array([r["center_err"] for r in rows])
        ee = np.array([r["extent_err"] for r in rows])
        return {
            "n": len(rows),
            "center_err_median": round(float(np.median(ce)), 3),
            "center_err_p90": round(float(np.percentile(ce, 90)), 3),
            "extent_err_median": [round(float(v), 3) for v in np.median(ee, axis=0)],
        }

    out["error_small"] = summarise(small)
    out["error_large"] = summarise(large)

    out["challenge_object_reference"] = challenge_bench(preds, askable, label_map, targets)

    out["unmatched_pred_labels"] = sorted(
        {p.label for p in preds} - {p.label for p in preds
                                    if any(compatible(p, g, label_map) for g in gts)})
    return out


# ---------------------------------------------------------------- reporting


def audit_labels(scene: str, pred_path: str) -> None:
    preds, report = load_pred(pred_path)
    gts = load_gt(scene)
    prompts = report.get("source", {}).get("prompts", [])
    label_map = build_label_map(prompts)
    gt_labels = {g.label for g in gts}

    print(f"\n=== {scene} label audit ===")
    unresolved = sorted(p for prompt_key, accepted in label_map.items()
                        for p in [prompt_key] if not (accepted & gt_labels))
    if unresolved:
        print("  prompts matching NO GT label (typo, or genuinely absent from the scene):")
        for key in unresolved:
            print(f"    {key:<20} accepts {sorted(label_map[key])}")
    produced = {p.label for p in preds}
    unmapped = sorted(l for l in produced if not (label_map.get(l, set()) & gt_labels))
    if unmapped:
        print("  PREDICTED labels that can never match GT — these are scored as pure FPs:")
        for lab in unmapped:
            print(f"    {lab}")
    if not unresolved and not unmapped:
        print("  every prompt and predicted label resolves to a GT label")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(DATA_ROOT, "runs", "map3d"))
    ap.add_argument("--run", default=None, help="run id (default: most recent)")
    ap.add_argument("--scene", default="all")
    ap.add_argument("--audit-labels", action="store_true")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args(argv)

    # MAP3D_RUN_ID is set by the justfile from the host git sha — the SAME value
    # replay_map3d.py keys its output directory on. Using it here makes replay and score an
    # explicitly matched pair. The previous "newest directory by mtime" heuristic silently
    # scored the wrong run: rewriting a scene file does not update its directory's mtime,
    # but writing summary.json into an older run directory does, so the stale run looked
    # newer and the score reprinted the previous numbers as if they were the new ones.
    run = args.run or os.environ.get("MAP3D_RUN_ID")
    if not run:
        runs = [d for d in glob.glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(d)]
        if not runs:
            raise SystemExit(f"no runs under {args.runs_dir} — run replay_map3d.py first")
        # Fall back to the newest scene REPORT, not the newest directory.
        run = os.path.basename(os.path.dirname(max(
            (p for d in runs for p in glob.glob(os.path.join(d, "*.json"))
             if os.path.basename(p) != "summary.json"),
            key=os.path.getmtime, default=runs[-1] + "/x.json")))
    run_dir = os.path.join(args.runs_dir, run)

    paths = sorted(glob.glob(os.path.join(run_dir, "*.json")))
    paths = [p for p in paths if os.path.basename(p) != "summary.json"]
    if args.scene != "all":
        paths = [p for p in paths if os.path.basename(p) == f"{args.scene}.json"]
    if not paths:
        raise SystemExit(f"no scene reports in {run_dir}")

    results = []
    for path in paths:
        scene = os.path.splitext(os.path.basename(path))[0]
        if args.audit_labels:
            audit_labels(scene, path)
        results.append(score_scene(scene, path))

    if args.audit_labels:
        return 0

    print(f"run {run}   ({len(results)} scene(s))\n")
    hdr = (f"{'scene':<18}{'pred':>5}{'gt':>5}{'P@.25':>7}{'R@.25':>7}"
           f"{'R_obs':>7}{'dup':>6}{'bestIoU':>9}{'claimed':>9}{'near':>6}{'cat2/2':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['scene']:<18}{r['error']}")
            continue
        a, q = r["aabb@0.25"], r["box_quality"]
        orc = r["challenge_object_reference"]["mean_score_axis_aligned_of_2"]
        print(f"{r['scene']:<18}{r['n_pred']:>5}{r['n_gt_askable']:>5}"
              f"{a['precision']:>7.2f}{a['recall_askable']:>7.2f}"
              f"{a['recall_observable']:>7.2f}{a['duplicate_rate']:>6.2f}"
              f"{q['mean_best_iou_per_gt']:>9.3f}"
              f"{q['mean_best_iou_per_claimed_gt']:>9.3f}"
              f"{q['near_misses_0.1_to_0.25']:>6}"
              f"{(f'{orc:.2f}' if orc is not None else '  -- '):>8}")

    ok = [r for r in results if "error" not in r]
    if ok:
        print("\naggregate (mean over scenes, AABB@0.25 unless noted):")
        for key, path in (("precision", ("aabb@0.25", "precision")),
                          ("recall_askable", ("aabb@0.25", "recall_askable")),
                          ("recall_observable", ("aabb@0.25", "recall_observable")),
                          ("duplicate_rate", ("aabb@0.25", "duplicate_rate")),
                          ("over_merge_rate", ("over_merge", "rate")),
                          ("mean_iou", ("aabb@0.25", "mean_iou")),
                          ("recall @0.50", ("aabb@0.5", "recall_askable")),
                          ("recall OBB@0.25", ("obb@0.25", "recall_askable")),
                          # threshold-free — read A/Bs against these two
                          ("best_iou_per_gt", ("box_quality", "mean_best_iou_per_gt")),
                          ("best_iou_claimed", ("box_quality", "mean_best_iou_per_claimed_gt"))):
            vals = [r[path[0]][path[1]] for r in ok if r.get(path[0])]
            if vals:
                print(f"  {key:<20} {np.mean(vals):.3f}")
        for band in ("error_small", "error_large"):
            rows = [r[band] for r in ok if r.get(band)]
            if rows:
                print(f"  {band:<20} n={sum(x['n'] for x in rows)}  "
                      f"center_err_median={np.mean([x['center_err_median'] for x in rows]):.3f} m")

        # The number that converts to actual challenge points. Oracle selection, so it is
        # the ceiling M4 will be working under, not a prediction of the final score.
        cb = [r["challenge_object_reference"] for r in ok]
        n_t = sum(c["n_targets"] for c in cb)
        if n_t:
            def wmean(key):
                """

Target-weighted mean, skipping scenes that have no value for `key`.

                The denominator must be the weight of the CONTRIBUTING scenes, not n_t: a
                scene where no target was found reports None for the error medians, and
                dividing those by the full target count would quietly deflate them.
                """
                pairs = [(c[key], c["n_targets"]) for c in cb if c.get(key) is not None]
                total = sum(w for _, w in pairs)
                return sum(v * w for v, w in pairs) / total if total else float("nan")
            missing = n_t - sum(c["n_found"] for c in cb)
            print(f"\ncategory-2, {n_t} question targets — in priority order:")
            print(f"  1. RIGHT OBJECT   found            {(n_t - missing) / n_t:.1%}"
                  f"   ({missing}/{n_t} with no candidate at all)")
            print(f"  2. WHERE  centre err  median       {wmean('center_err_median'):.3f} m"
                  f"   p90 {wmean('center_err_p90'):.3f} m")
            print(f"           voxel-centroid median     {wmean('centroid_err_median'):.3f} m"
                  f"   <- the other point map_node publishes")
            print(f"  3. HOW BIG  score of 2.0           {wmean('mean_score_oriented_of_2'):.3f}"
                  f"   (ignoring orientation {wmean('mean_score_axis_aligned_of_2'):.3f})")
            print()
            print(f"  selection: oracle                  {wmean('mean_score_oriented_of_2'):.3f}"
                  f"   <- assumes M4 always picks the best candidate")
            print(f"             naive (largest volume)  {wmean('mean_score_naive_selection_of_2'):.3f}"
                  f"   <- what duplicates actually cost")
            print(f"             candidates per target   {wmean('mean_candidates_per_target'):.2f}")
            print(f"  if we published the wireframe      {0.0:.3f}   <- structural zero, not a low score")

    # Never let a write failure discard results that are already on screen: the run dir is
    # created by the container under a different uid, so this is a plausible failure and
    # the numbers above are the actual product of the run.
    out_path = args.out or os.path.join(run_dir, "summary.json")
    try:
        with open(out_path, "w") as fh:
            json.dump({"run": run, "scenes": results}, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {out_path}")
    except OSError as err:
        print(f"\ncould not write {out_path}: {err}", file=sys.stderr)
        print("results above are still valid; pass --out to write them elsewhere",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
