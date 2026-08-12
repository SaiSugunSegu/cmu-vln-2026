#!/usr/bin/env python3
"""Which IRef-VLA objects did the robot actually see, and from where?

IRef-VLA annotates a scene from the Unity model, not from the robot. A lamp 2.7 m up is
in the metadata whether or not a 0.85 m-high camera with a 120 degree vertical band ever
had it in frame, and a chair in the next room is in the metadata whether or not a wall was
between it and the sensor. Asking "point at the vase left of the sink" about something the
robot never imaged is unanswerable however good the model is, so the category-2 generator
gates its candidates on this report (docs/cat2_benchmark.md, "Visibility gate").

Visibility is measured, not assumed. For every lidar frame in the scene bag:

1. the sweeps in the mapper's own fusion window are gathered and projected with the
   *deployed* camera model -- ``sam_mapper.cloud_image_fusion``, the same 1920x640 equirect
   mapping the mapper uses, so a "visible" verdict here means visible to the pipeline as
   configured and not to an idealised pinhole camera;
2. the returns falling inside an object's box are counted, but only those whose pixel row
   lands inside the image. That single test covers both failure modes at once: geometry
   the camera cannot reach (too high, too close, cropped band) drops out because its rows
   fall outside, and geometry the camera cannot see through drops out because the lidar
   never returned from it -- an occluder stops both beams.

An object is called visible when some frame gives it enough returns, enough pixel area and
little enough foreground in front of it. The best such frame is written out as an annotated
crop, which is what makes the verdict reviewable rather than a number to be trusted, and
every rejection carries the reason it failed (``never_scanned``, ``outside_camera_band``,
``too_small``, ``occluded``) so the gate can be argued with.

    # in the ai_module container (needs ros2 + the bag)
    python3 /home/docker/scripts/eval/object_visibility.py --scene arabic_room
    just visibility all            # host wrapper, copies the report next to the benchmark
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from bisect import bisect_left

import numpy as np

# The generator's object model, so "the box" means the same thing in both places.
for _cand in ("/data/bags", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bags")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, os.path.abspath(_cand))
import cat2_geometry as geom  # noqa: E402

CLOUD_TOPIC = "/registered_scan"
ODOM_TOPIC = "/state_estimation"
IMAGE_TOPIC = "/camera/image"

# ---------------------------------------------------------------- thresholds
#
# Defaults are calibrated in docs/cat2_benchmark.md against the annotated crops: every
# object the gate accepts is recognisable in its own best view, and the objects it rejects
# are above the band, behind a wall, or a handful of stray returns.

# Lidar returns inside the box that also landed inside the image. Below this the "object"
# is a few points clipped off its edge, not a surface the camera resolved.
MIN_POINTS = 12
# Apparent size, in px^2 on the 1920x640 panorama. 900 px^2 is ~30x30 px, i.e. about 5.6
# degrees square -- small, but a VLM can still ground it.
MIN_PX_AREA = 900
# Fraction of the returns in the object's footprint that sit well in front of it.
MAX_OCCLUSION = 0.60
# How much closer a return must be to count as foreground rather than the object itself.
OCCLUDER_MARGIN_M = 0.30
# Objects past this range are not scored: the lidar thins out and nothing that far is a
# reasonable thing to ask about.
MAX_RANGE_M = 15.0
# Lidar window per frame, in seconds. Mirrors the mapper's runtime cloud_window_before /
# cloud_window_after: a single Mid-360 sweep is 10.6k points over the full sphere, and
# judging a 5-degree-wide object on one sweep measures the sensor's sparsity, not what the
# robot saw.
WINDOW_BEFORE = 0.5
WINDOW_AFTER = 0.1
# Slack on the box when deciding whether a return came off this object. IRef-VLA boxes are
# tight, the lidar has its own noise, and a flush picture or a door leaf in its frame is a
# few centimetres of geometry -- with no slack those read as "never scanned".
BOX_TOL = 0.10


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def read_geometry(scene_dir: str):
    """Lidar sweeps (world frame) and the odometry needed to place them, in stamp order."""
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from sam_mapper.tools.dump_frames import open_reader
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    reader = open_reader(scene_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CLOUD_TOPIC, ODOM_TOPIC]))
    clouds, cloud_stamps, odom_stack, odom_stamps = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == CLOUD_TOPIC:
            msg = deserialize_message(data, PointCloud2)
            clouds.append(point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z")))
            cloud_stamps.append(_stamp_s(msg.header))
        else:
            msg = deserialize_message(data, Odometry)
            pose, orient = msg.pose.pose.position, msg.pose.pose.orientation
            odom_stack.append(([pose.x, pose.y, pose.z], [orient.x, orient.y, orient.z, orient.w]))
            odom_stamps.append(_stamp_s(msg.header))
    return clouds, cloud_stamps, odom_stack, odom_stamps


def read_images(scene_dir: str, wanted: set[float], tol: float = 0.2):
    """Decode only the camera frames some object picked as its best view.

    Decoding the whole bag is 375 frames x 1920x640x3; the report needs a handful.
    """
    import rosbag2_py
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from sam_mapper.tools.dump_frames import open_reader
    from sensor_msgs.msg import Image

    if not wanted:
        return {}
    targets = sorted(wanted)
    bridge = CvBridge()
    reader = open_reader(scene_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[IMAGE_TOPIC]))
    best: dict[float, tuple[float, object]] = {}
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        msg = deserialize_message(data, Image)
        stamp = _stamp_s(msg.header)
        idx = min(bisect_left(targets, stamp), len(targets) - 1)
        for target in targets[max(0, idx - 1):idx + 2]:
            delta = abs(target - stamp)
            if delta <= tol and (target not in best or delta < best[target][0]):
                best[target] = (delta, bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
    return {stamp: image for stamp, (_delta, image) in best.items()}


def pose_at(odom_stack, odom_stamps, stamp):
    """Pose interpolated to `stamp`, via the pipeline's own sync arithmetic."""
    from sam_mapper import frame_sync

    stack = [{"position": p, "orientation": q, "linear_velocity": [0.0] * 3,
              "angular_velocity": [0.0] * 3} for p, q in odom_stack]
    odom, status = frame_sync.interpolate_odom(stack, odom_stamps, stamp)
    if status != frame_sync.OK:
        return None, None
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(odom["orientation"]).as_matrix(), np.asarray(odom["position"], float)


def project(points_world: np.ndarray, rot_b2w: np.ndarray, trans_b2w: np.ndarray) -> np.ndarray:
    """World points -> (col, row, horizontal range) on the robot's panorama.

    Unclipped on purpose: a row outside [0, height) is the answer to "could the camera see
    it", so it must not be pinned to the edge (see CloudImageFusion.BOUNDS_MODES).
    """
    from sam_mapper.cloud_image_fusion import scan2pixels_mecanum_sim

    body = (points_world - trans_b2w) @ rot_b2w
    return scan2pixels_mecanum_sim(body, clip=False)


def inside_footprint(points_xy: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Mask of points inside a convex quad (the box's rotated top face, corners 0-3)."""
    if points_xy.size == 0:
        return np.zeros(0, dtype=bool)
    inside = np.ones(len(points_xy), dtype=bool)
    sign = 0.0
    for i in range(4):
        edge = quad[(i + 1) % 4] - quad[i]
        rel = points_xy - quad[i]
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        if sign == 0.0:
            sign = 1.0 if float(np.sum(cross)) >= 0 else -1.0
        inside &= (cross * sign) >= 0
    return inside


def dilate_quad(quad: np.ndarray, tol: float) -> np.ndarray:
    """Grow a footprint outwards by `tol`, so a flush-mounted picture's returns -- which
    land within centimetres of its box, not inside it -- still count as hits."""
    centre = quad.mean(axis=0)
    offsets = quad - centre
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    return centre + offsets * (1.0 + tol / np.maximum(norms, 1e-6))


def column_span(cols: np.ndarray, width: int) -> tuple[int, int]:
    """Narrowest circular column interval covering `cols`, as (start, length).

    The panorama wraps, so an object straddling the seam has columns at both 0 and 1919 and
    a plain min/max would claim it spans the whole image.
    """
    order = np.sort(cols.astype(int))
    gaps = np.diff(np.concatenate([order, order[:1] + width]))
    cut = int(np.argmax(gaps))
    start = int(order[(cut + 1) % len(order)])
    length = int((order[cut] - start) % width)
    return start, length


def score_object(obj, cloud_world, pixels, in_band, width, height, x_sorted, rot, trans,
                 tol=BOX_TOL):
    """One object in one frame: how much of it the camera resolved, and what is in front.

    Returns (points_in_box, view_or_None) -- the first count separates "the sensor never
    got a return off this object" from "it did, but the camera band cut it off".
    """
    order, xs = x_sorted
    lo_i = int(np.searchsorted(xs, obj.lo[0] - tol, side="left"))
    hi_i = int(np.searchsorted(xs, obj.hi[0] + tol, side="right"))
    if hi_i - lo_i < MIN_POINTS:
        return 0, None
    idx = order[lo_i:hi_i]
    pts = cloud_world[idx]
    near = ((pts[:, 1] >= obj.lo[1] - tol) & (pts[:, 1] <= obj.hi[1] + tol)
            & (pts[:, 2] >= obj.lo[2] - tol) & (pts[:, 2] <= obj.hi[2] + tol))
    if near.sum() < MIN_POINTS:
        return int(near.sum()), None
    near[near] = inside_footprint(pts[near][:, :2], dilate_quad(obj.corners[0:4, :2], tol))
    in_box = int(near.sum())
    idx = idx[near]
    hit = idx[in_band[idx]]
    if hit.size < MIN_POINTS:
        return in_box, None

    # Apparent size comes from the box's own corners, not from the returns. Returns are
    # sparse and, with any box slack at all, partly belong to whatever the object is flush
    # against -- measuring extent from them makes a book on a table as big as the table.
    corners = project(obj.corners, rot, trans)
    col_start, col_len = column_span(corners[:, 0], width)
    row0 = float(np.clip(corners[:, 1].min(), 0, height - 1))
    row1 = float(np.clip(corners[:, 1].max(), 0, height - 1))
    clipped = bool(corners[:, 1].min() < 0 or corners[:, 1].max() >= height)
    px_area = max(col_len, 1) * max(row1 - row0, 1.0)

    # Foreground share: of everything the sensor returned inside this object's own patch of
    # the image, how much sits at least a margin closer than the object's near face.
    depth = float(np.percentile(pixels[hit, 2], 10))
    patch = (((pixels[:, 0].astype(int) - col_start) % width <= col_len)
             & (pixels[:, 1] >= row0) & (pixels[:, 1] <= row1) & in_band)
    occlusion = (float(np.mean(pixels[patch, 2] < depth - OCCLUDER_MARGIN_M))
                 if int(patch.sum()) else 0.0)

    return in_box, {
        "n_points": int(hit.size),
        "px_bbox": [col_start, int(row0), (col_start + col_len) % width, int(row1)],
        "px_bbox_unwrapped": [col_start, int(row0), col_start + col_len, int(row1)],
        "px_area": int(px_area),
        "px_area_frac": round(px_area / (width * height), 5),
        "distance_m": round(depth, 2),
        "occlusion": round(occlusion, 3),
        "clipped": clipped,
    }


def accepts(view: dict) -> bool:
    return (view["n_points"] >= MIN_POINTS and view["px_area"] >= MIN_PX_AREA
            and view["occlusion"] <= MAX_OCCLUSION)


def reason_for(view, points_in_box: int, min_range, max_range: float) -> str:
    """Why an object failed the gate, in the terms a fix would be phrased in."""
    if min_range is None or min_range > max_range:
        return "out_of_range"
    if points_in_box < MIN_POINTS:
        # No returns off its surface at all: behind a wall, or in a corner the robot's path
        # never faced.
        return "never_scanned"
    if view is None:
        # Scanned, but every return projected outside the 120-degree band -- the ceiling
        # fixtures and floor patches a 0.85 m camera passes under.
        return "outside_camera_band"
    if view["px_area"] < MIN_PX_AREA:
        return "too_small"
    if view["occlusion"] > MAX_OCCLUSION:
        return "occluded"
    return "too_few_points"


def annotate(image, view, label, width) -> np.ndarray:
    """Crop the panorama around the object and outline what the report measured."""
    import cv2

    col0, row0, col1, row1 = view["px_bbox_unwrapped"]
    tiled = np.hstack([image, image])  # so a box across the seam stays contiguous
    pad_x = max(60, int(0.35 * (col1 - col0)))
    pad_y = max(60, int(0.60 * (row1 - row0)))
    x0, x1 = max(0, col0 - pad_x), min(tiled.shape[1], col1 + pad_x)
    y0, y1 = max(0, row0 - pad_y), min(tiled.shape[0], row1 + pad_y)
    crop = tiled[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (col0 - x0, row0 - y0), (col1 - x0, row1 - y0), (0, 235, 255), 2)
    caption = f"{label}  {view['distance_m']}m  {view['n_points']}pts"
    cv2.putText(crop, caption, (6, max(18, crop.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(crop, caption, (6, max(18, crop.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 235, 255), 1, cv2.LINE_AA)
    return crop


def measure_scene(scene: str, args) -> dict:
    scene_dir = os.path.join(args.bags_dir, scene)
    if not os.path.isdir(scene_dir):
        raise SystemExit(f"no bag directory {scene_dir}")
    objects = geom.load_objects(scene)
    clouds, cloud_stamps, odom_stack, odom_stamps = read_geometry(scene_dir)
    if not clouds:
        raise SystemExit(f"{scene}: bag has no {CLOUD_TOPIC}")

    from sam_mapper import frame_sync

    width, height = 1920, 640
    best: dict[str, dict] = {}
    seen_frames: dict[str, int] = {}
    closest: dict[str, float] = {}
    scanned: dict[str, int] = {}
    started = time.perf_counter()

    for index in range(0, len(clouds), args.stride):
        stamp = cloud_stamps[index]
        cloud = frame_sync.gather_cloud(clouds, cloud_stamps, stamp,
                                        args.window_before, args.window_after)
        rot, trans = pose_at(odom_stack, odom_stamps, stamp)
        if rot is None or cloud is None or cloud.shape[0] == 0:
            continue
        pixels = project(cloud, rot, trans)
        in_band = (pixels[:, 1] >= 0) & (pixels[:, 1] < height) & (pixels[:, 2] > 0.05)
        if not in_band.any():
            continue
        order = np.argsort(cloud[:, 0])
        x_sorted = (order, cloud[order, 0])

        for oid, obj in objects.items():
            reach = float(np.linalg.norm(obj.center - trans))
            if reach > args.max_range:
                continue
            closest[oid] = min(closest.get(oid, reach), reach)
            in_box, view = score_object(obj, cloud, pixels, in_band, width, height, x_sorted,
                                        rot, trans, args.box_tol)
            scanned[oid] = max(scanned.get(oid, 0), in_box)
            if view is None:
                continue
            view["stamp"] = round(stamp, 6)
            view["frame"] = index
            view["elevation_deg"] = round(float(np.degrees(np.arctan2(
                obj.center[2] - (trans[2] + 0.1), max(1e-3, np.hypot(*(obj.center[:2] - trans[:2])))))), 1)
            if accepts(view):
                seen_frames[oid] = seen_frames.get(oid, 0) + 1
            # Rank on resolved pixel area, tie-broken by return count: the frame a human
            # would pick to check the label is the one where the object is biggest.
            key = (view["px_area"], view["n_points"])
            if oid not in best or key > (best[oid]["px_area"], best[oid]["n_points"]):
                best[oid] = view

    views_dir = os.path.join(args.views_dir, scene)
    images = {}
    if not args.no_views:
        os.makedirs(views_dir, exist_ok=True)
        images = read_images(scene_dir, {v["stamp"] for v in best.values()
                                        if args.views_all or accepts(v)})

    report = {"scene": scene, "camera": {"width": width, "height": height,
                                         "model": "equirect 360x120, sam_mapper mecanum_sim"},
              "thresholds": {"min_points": MIN_POINTS, "min_px_area": MIN_PX_AREA,
                             "max_occlusion": MAX_OCCLUSION, "max_range_m": args.max_range,
                             "window_s": [args.window_before, args.window_after]},
              "frames": len(range(0, len(clouds), args.stride)), "objects": {}}

    import cv2
    for oid, obj in sorted(objects.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        view = best.get(oid)
        visible = bool(view and accepts(view))
        entry = {
            "label": obj.display,
            "region": obj.region,
            "visible": visible,
            "reason": None if visible else reason_for(view, scanned.get(oid, 0),
                                                     closest.get(oid), args.max_range),
            "frames_visible": seen_frames.get(oid, 0),
            "min_range_m": round(closest[oid], 2) if oid in closest else None,
            "best_view": view,
        }
        if view and view["stamp"] in images and (visible or args.views_all):
            crop = annotate(images[view["stamp"]], view, f"{obj.display} #{oid}", width)
            stem = geom.norm_class(obj.display).replace(" ", "_") or "object"
            name = f"{oid}_{stem}.png" if visible else f"rejected_{entry['reason']}_{oid}_{stem}.png"
            cv2.imwrite(os.path.join(views_dir, name), crop)
            # Only accepted objects get their crop recorded. The report is a tracked artifact
            # and --views-all is a debug flag: if the flag changed the JSON, two runs of the
            # documented recipe would produce different committed files. Reject crops are
            # still written, and named for the reason they were rejected.
            if visible:
                # Repo-relative, because /data is only a path inside the container and the
                # report is read on the host by the generator.
                entry["view_image"] = (
                    "data/" + os.path.relpath(os.path.join(views_dir, name), "/data")
                    if views_dir.startswith("/data") else os.path.join(views_dir, name))
        report["objects"][oid] = entry

    visible = sum(1 for e in report["objects"].values() if e["visible"])
    report["summary"] = {
        "objects": len(objects),
        "visible": visible,
        "hidden": len(objects) - visible,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="arabic_room",
                        help="scene name, or 'all' for the 13 scenes with referential statements")
    parser.add_argument("--bags-dir", default="/data/bags")
    parser.add_argument("--out", default="/data/runs/visibility",
                        help="where the per-scene JSON reports go")
    parser.add_argument("--views-dir", default="/data/crops/visibility",
                        help="where the annotated best-view crops go")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth lidar frame")
    parser.add_argument("--max-range", type=float, default=MAX_RANGE_M)
    parser.add_argument("--window-before", type=float, default=WINDOW_BEFORE)
    parser.add_argument("--window-after", type=float, default=WINDOW_AFTER)
    parser.add_argument("--box-tol", type=float, default=BOX_TOL,
                        help="metres a return may sit outside a box and still count as a hit")
    parser.add_argument("--no-views", action="store_true", help="skip image decode entirely")
    parser.add_argument("--views-all", action="store_true",
                        help="also crop rejected objects, for calibrating the thresholds")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = sorted(
            os.path.basename(os.path.dirname(os.path.dirname(path)))
            for path in glob.glob(os.path.join(args.bags_dir, "*", "iref_vla_metadata",
                                               "*_referential_statements.json")))
    else:
        scenes = [args.scene]

    os.makedirs(args.out, exist_ok=True)
    for scene in scenes:
        report = measure_scene(scene, args)
        path = os.path.join(args.out, f"{scene}_visibility.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        summary = report["summary"]
        print(f"{scene:16s} {summary['visible']:3d}/{summary['objects']:3d} visible "
              f"({summary['seconds']}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
