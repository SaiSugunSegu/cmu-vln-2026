#!/usr/bin/env python3
"""Draw each category-2 object's oriented 3D box on the camera frame that saw it best.

``object_visibility.py``'s report says an object is visible and gives its axis-aligned
pixel bbox, which looks the same however the box is rotated -- a wrong yaw, or a
target's corners quietly read in the wrong frame, would not show up in that crop. This
script instead projects the box's own 8 corners with the *same* deployed camera model
and draws all 12 edges, which is the check that catches that class of bug: if the
wireframe does not sit on the object in the image, the ground truth is wrong, not the
camera model.

By default (``--source qa``) the corners drawn are read straight out of an object's own
``bbox_corners`` in the category-2 QA file itself -- the exact numbers a scorer would
read -- not recomputed from ``<scene>_objects.json``. That is a strictly stronger check
than "the metadata this file was built from is right": a stale QA file, a hand edit, or
a `gen-cat2` that ran before a metadata fix would all pass a metadata-only check while
still shipping a wrong answer. Each drawn object also gets a QA-vs-metadata numeric
diff printed, so a mismatch is caught even if it happens to still look fine on screen.
Both a question's *answer* and its *anchors* carry geometry in the QA file, so this
mode covers every object any question names; pass ``--source objects`` for a full-scene
sweep over objects no question names at all, which has to come from the metadata instead.

Runs entirely on the host: no docker, no ROS2, no rclpy -- see scripts/utils/mcap_io.py.
Needs ``pip install mcap mcap-ros2-support`` and the scene's bag on disk.

    python3 scripts/eval/project_bboxes.py --scene arabic_room
    python3 scripts/eval/project_bboxes.py --scene arabic_room --object-id 73 2
    python3 scripts/eval/project_bboxes.py --scene arabic_room --source objects --all --no-points
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import object_visibility as ov  # noqa: E402 (column_span-style helpers, pure numpy)
import utils.geometry as geom  # noqa: E402
import utils.mcap_io as mcap_io  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 1920, 640
# Bottom-face 0-3, top-face 4-7, with corner i+4 directly above corner i -- the IRef-VLA
# bbox_corners convention (checked against the QA files) and the same list
# ros_markers.create_wireframe_marker_from_corners uses for /obj_boxes.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

BOX_COLOR = (0, 235, 255)     # yellow-ish, matches object_visibility.py's annotate()
POINT_COLOR = (255, 80, 0)    # blue dots: lidar returns counted inside the box
CORNER_COLOR = (0, 0, 255)


def load_visibility(scene: str) -> dict:
    path = REPO / "data" / "benchmark" / scene / "visibility" / f"{scene}_visibility.json"
    if not path.exists():
        raise SystemExit(f"no visibility report at {path} -- run `just visibility {scene}` first")
    return json.loads(path.read_text())["objects"]


def qa_object_ids(scene: str, qa_path: Path | None) -> list[str]:
    """Every object a category-2 question names, target or anchor, in question order."""
    path = qa_path or (REPO / "data" / "benchmark" / scene / "category_2" /
                       f"{scene}_category2_qa.json")
    if not path.exists():
        raise SystemExit(f"no category-2 QA file at {path}")
    qa = json.loads(path.read_text())
    ids, seen = [], set()
    for q in qa.get("questions", []):
        candidates = [(q.get("answer") or {}).get("object_id")]
        candidates += [a.get("object_id") for a in q.get("anchors") or []]
        for oid in candidates:
            oid = str(oid) if oid is not None else None
            if oid and oid not in seen:
                seen.add(oid)
                ids.append(oid)
    return ids


def unwrap_columns(cols: np.ndarray, width: int) -> tuple[np.ndarray, int]:
    """Place every column of an 8-corner box into one contiguous [start, start+width)
    arc -- the 12-edge analogue of object_visibility.py's column_span, which only
    needed the extent of a single quad, not every corner positioned for line-drawing.
    """
    sorted_cols = np.sort(np.mod(cols, width).astype(int))
    gaps = np.diff(np.concatenate([sorted_cols, sorted_cols[:1] + width]))
    cut = int(np.argmax(gaps))
    start = int(sorted_cols[(cut + 1) % len(sorted_cols)])
    unwrapped = start + np.mod(cols - start, width)
    return unwrapped, start


def points_in_box(cloud: np.ndarray, obj, tol: float = 0.10) -> np.ndarray:
    """Cloud points (world frame) inside the object's dilated footprint. A light,
    one-shot version of object_visibility.py's score_object, for visual overlay only
    -- not the accept/reject gate, which is what the visibility report already is.
    """
    near = (
        (cloud[:, 0] >= obj.lo[0] - tol) & (cloud[:, 0] <= obj.hi[0] + tol)
        & (cloud[:, 1] >= obj.lo[1] - tol) & (cloud[:, 1] <= obj.hi[1] + tol)
        & (cloud[:, 2] >= obj.lo[2] - tol) & (cloud[:, 2] <= obj.hi[2] + tol)
    )
    if not near.any():
        return cloud[:0]
    quad = ov.dilate_quad(obj.corners[0:4, :2], tol)
    inside = ov.inside_footprint(cloud[near][:, :2], quad)
    return cloud[near][inside]


def render(image: np.ndarray, obj, rot: np.ndarray, trans: np.ndarray,
          cloud_pts: np.ndarray | None, caption: str):
    """Draws the box's wireframe (+ optional in-box lidar returns) on the frame.

    Returns (rolled_panorama, crop, abs_px_bbox): the panorama re-centred so the box
    never crosses the seam, a padded close-up, and the box's own (col0, row0, col1,
    row1) in the same "unwrapped" convention object_visibility.py's px_bbox_unwrapped
    uses, for a numeric cross-check against the shipped report.
    """
    corners_px = mcap_io.project(obj.corners, rot, trans)
    cols, rows = corners_px[:, 0], corners_px[:, 1]
    unwrapped, start = unwrap_columns(cols, WIDTH)

    abs_bbox = (
        int(round(float(unwrapped.min()))), int(np.clip(rows.min(), 0, HEIGHT - 1)),
        int(round(float(unwrapped.max()))), int(np.clip(rows.max(), 0, HEIGHT - 1)),
    )

    canvas = np.hstack([image, image])
    clip = lambda v, lo, hi: int(np.clip(round(v), lo, hi))  # noqa: E731
    for a, b in EDGES:
        pa = (clip(unwrapped[a], -4 * WIDTH, 4 * WIDTH), clip(rows[a], -4 * HEIGHT, 4 * HEIGHT))
        pb = (clip(unwrapped[b], -4 * WIDTH, 4 * WIDTH), clip(rows[b], -4 * HEIGHT, 4 * HEIGHT))
        cv2.line(canvas, pa, pb, BOX_COLOR, 2, cv2.LINE_AA)
    for x, y in zip(unwrapped, rows):
        if 0 <= y < HEIGHT:
            cv2.circle(canvas, (clip(x, 0, 2 * WIDTH - 1), clip(y, 0, HEIGHT - 1)), 3, CORNER_COLOR, -1)

    if cloud_pts is not None and len(cloud_pts):
        pts_px = mcap_io.project(cloud_pts, rot, trans)
        pts_col = start + np.mod(pts_px[:, 0] - start, WIDTH)
        for x, y in zip(pts_col, pts_px[:, 1]):
            if 0 <= y < HEIGHT:
                cv2.circle(canvas, (clip(x, 0, 2 * WIDTH - 1), int(y)), 1, POINT_COLOR, -1)

    rolled = canvas[:, start:start + WIDTH].copy()
    local_cols = unwrapped - start
    x0, x1 = float(local_cols.min()), float(local_cols.max())
    y0, y1 = float(max(0, rows.min())), float(min(HEIGHT, rows.max()))
    pad_x = max(60, int(0.35 * (x1 - x0)))
    pad_y = max(60, int(0.6 * (y1 - y0)))
    cx0, cx1 = max(0, int(x0 - pad_x)), min(WIDTH, int(x1 + pad_x))
    cy0, cy1 = max(0, int(y0 - pad_y)), min(HEIGHT, int(y1 + pad_y))
    crop = rolled[cy0:cy1, cx0:cx1].copy()
    for pos, fg in ((3, (0, 0, 0)), (1, BOX_COLOR)):
        cv2.putText(crop, caption, (6, max(18, crop.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, fg, pos, cv2.LINE_AA)
    return rolled, crop, abs_bbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="arabic_room")
    ap.add_argument("--bags-dir", default=str(REPO / "data" / "bags"))
    ap.add_argument("--qa", type=Path, default=None,
                    help="category-2 QA file (default: data/benchmark/<scene>/category_2/*.json)")
    ap.add_argument("--source", choices=("qa", "objects"), default="qa",
                    help="where box corners come from: the QA file's own bbox_corners "
                         "(default, verifies what was actually shipped, for answer and "
                         "anchors alike) or a fresh read of <scene>_objects.json (needed for "
                         "--all, for objects no question names)")
    ap.add_argument("--object-id", nargs="+", default=None, help="check only these object ids")
    ap.add_argument("--all", action="store_true",
                    help="every object in the scene (implies --source objects), not just the QA's")
    ap.add_argument("--show-points", dest="show_points", action="store_true", default=True)
    ap.add_argument("--no-points", dest="show_points", action="store_false",
                    help="skip the lidar-returns overlay")
    ap.add_argument("--out-dir", default=str(REPO / "data" / "crops" / "bbox_check"))
    ap.add_argument("--tol", type=float, default=0.10, help="footprint dilation, metres")
    args = ap.parse_args()

    metadata_objects = geom.load_objects(args.scene, Path(args.bags_dir))
    source = "objects" if args.all else args.source
    if source == "qa":
        objects = geom.load_qa_answers(args.scene, args.qa)
    else:
        objects = metadata_objects
    vis = load_visibility(args.scene)

    if args.object_id:
        oids = [str(o) for o in args.object_id]
    elif source == "objects" and args.all:
        oids = sorted(objects, key=lambda o: int(o) if o.isdigit() else 0)
    elif source == "qa":
        oids = sorted(objects, key=lambda o: int(o) if o.isdigit() else 0)
    else:
        oids = qa_object_ids(args.scene, args.qa)

    missing = [o for o in oids if o not in objects]
    if missing:
        reason = ("no bbox_corners for these in the QA file (no question names them, as "
                  "target or anchor -- try --source objects)" if source == "qa" else
                  "not in scene metadata")
        print(f"WARNING: {reason}, skipping: {missing}")
    oids = [o for o in oids if o in objects]
    if not oids:
        raise SystemExit("no objects to check")

    bag_path = Path(args.bags_dir) / args.scene / f"{args.scene}_0.mcap"
    if not bag_path.exists():
        raise SystemExit(f"no bag at {bag_path}")

    # Project onto the exact frame the visibility gate already picked as this object's
    # best view, so this check is *of* that pipeline's own evidence, not a fresh guess.
    stamps: dict[str, float] = {}
    skipped_no_view = []
    for oid in oids:
        best = (vis.get(oid) or {}).get("best_view")
        if best and best.get("stamp") is not None:
            stamps[oid] = float(best["stamp"])
        else:
            skipped_no_view.append(oid)
    if not stamps:
        raise SystemExit("none of the requested objects have a recorded best_view in the visibility report")

    print(f"reading {bag_path} ...")
    track = mcap_io.read_track(bag_path)
    images = mcap_io.read_image_near(bag_path, stamps.values())

    out_dir = Path(args.out_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for oid in oids:
        obj = objects[oid]
        best_view = (vis.get(oid) or {}).get("best_view") or {}
        stamp = stamps.get(oid)
        if stamp is None:
            rows_out.append((oid, obj.display, "no recorded best_view in visibility report", "SKIP"))
            continue
        image = images.get(stamp)
        if image is None:
            rows_out.append((oid, obj.display, f"no /camera/image within tol of stamp {stamp}", "SKIP"))
            continue
        rot, trans = track.pose_at(stamp)
        if rot is None:
            rows_out.append((oid, obj.display, f"odometry does not cover stamp {stamp}", "SKIP"))
            continue

        cloud_pts = None
        if args.show_points:
            cloud = track.cloud_near(stamp)
            if cloud is not None:
                cloud_pts = points_in_box(cloud, obj, args.tol)

        caption = f"{obj.display}#{oid}  {best_view.get('distance_m', '?')}m"
        rolled, crop, abs_bbox = render(image, obj, rot, trans, cloud_pts, caption)

        stem = geom.norm_class(obj.display).replace(" ", "_") or "object"
        cv2.imwrite(str(out_dir / f"{oid}_{stem}_full.png"), rolled)
        cv2.imwrite(str(out_dir / f"{oid}_{stem}_crop.png"), crop)

        reported = best_view.get("px_bbox_unwrapped") or best_view.get("px_bbox")
        status = "?"
        if reported and len(reported) == 4:
            status = "OK" if all(abs(a - b) <= 8 for a, b in zip(abs_bbox, reported)) else "CHECK"

        detail = f"recomputed {abs_bbox}  report {reported}"
        meta_obj = metadata_objects.get(oid)
        if source == "qa" and meta_obj is not None:
            qa_delta = float(np.abs(obj.corners - meta_obj.corners).max())
            detail += f"  |  qa-vs-metadata Δ={qa_delta:.4f}m"
            if qa_delta > 5e-4 and status != "SKIP":
                status = "MISMATCH" if status != "CHECK" else status
        rows_out.append((oid, obj.display, detail, status))

    print(f"\n{'id':>4}  {'label':<20} {'detail':<75} status")
    for oid, label, detail, status in rows_out:
        print(f"{oid:>4}  {label:<20} {detail:<75} {status}")
    n_written = sum(1 for *_r, status in rows_out if status != "SKIP")
    print(f"\nwrote {n_written} object(s) to {out_dir}")
    n_check = sum(1 for *_r, status in rows_out if status == "CHECK")
    if n_check:
        print(f"{n_check} object(s) flagged CHECK -- open their _crop.png and compare against "
              f"data/crops/visibility/{args.scene}/ before trusting the answer box")
    n_mismatch = sum(1 for *_r, status in rows_out if status == "MISMATCH")
    if n_mismatch:
        print(f"{n_mismatch} object(s) flagged MISMATCH -- the QA file's bbox_corners for that "
              f"object_id disagree with {args.scene}_objects.json by more than 0.5mm; the QA "
              f"file is likely stale and gen-cat2 should be rerun")


if __name__ == "__main__":
    main()
