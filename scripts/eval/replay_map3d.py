#!/usr/bin/env python3
"""Rebuild a scene's 3D map offline from a source bag + its companion /sam3/* bag.

No GPU, no ROS graph. Assembles frames through the SAME frame_sync code map_node uses,
drives ObjMapper.update_map, dumps the map as JSON. ~30 s/scene and byte-identical across
runs, which is what makes an A/B at n=13 scenes meaningful.

Differs from the live node deliberately: every frame is processed (no latest-frame-wins),
whole-bag arrays instead of ring buffers, no publishing."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from sam_mapper import frame_sync
from sam_mapper.cloud_image_fusion import CloudImageFusion
from sam_mapper.detections import PromptTable
from sam_mapper.mapping_config import MappingConfig
from sam_mapper.node_base import load_config, resolve_config_path
from sam_mapper.object_mapper import ObjMapper
from sam_mapper.tools.dump_frames import open_reader

CLOUD_TOPIC = "/registered_scan"
ODOM_TOPIC = "/state_estimation"
INSTANCE_MAP_TOPIC = "/sam3/instance_map"
DETECTIONS_TOPIC = "/sam3/detections"


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _odom_of(msg: Odometry) -> dict:
    p, o = msg.pose.pose.position, msg.pose.pose.orientation
    lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
    return {
        'position': [p.x, p.y, p.z],
        'orientation': [o.x, o.y, o.z, o.w],        # scipy xyzw
        'linear_velocity': [lin.x, lin.y, lin.z],
        'angular_velocity': [ang.x, ang.y, ang.z],
    }


def read_source_bag(scene_dir: str):
    """Whole-bag clouds and odom, in stamp order."""
    reader = open_reader(scene_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CLOUD_TOPIC, ODOM_TOPIC]))
    clouds, cloud_stamps, odom_stack, odom_stamps = [], [], [], []
    while reader.has_next():
        topic, data, _log_time = reader.read_next()
        if topic == CLOUD_TOPIC:
            msg = deserialize_message(data, PointCloud2)
            clouds.append(point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z")))
            cloud_stamps.append(_stamp_s(msg.header))
        else:
            msg = deserialize_message(data, Odometry)
            odom_stack.append(_odom_of(msg))
            odom_stamps.append(_stamp_s(msg.header))
    return clouds, cloud_stamps, odom_stack, odom_stamps


def read_companion_bag(path: str, bridge: CvBridge):
    """Yield (stamp_s, id_map, entries) pairs, matched on the embedded stamp.

    The two topics are written back-to-back per frame by record_companion, so pairing is
    a simple zip on the stamp key — but it is checked rather than assumed, because a torn
    pair would silently attribute one frame's masks to another frame's labels.
    """
    reader = open_reader(path)
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[INSTANCE_MAP_TOPIC, DETECTIONS_TOPIC]))
    maps, dets = {}, {}
    order = []
    while reader.has_next():
        topic, data, _log_time = reader.read_next()
        if topic == INSTANCE_MAP_TOPIC:
            msg = deserialize_message(data, Image)
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            maps[key] = bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            order.append(key)
        else:
            payload = json.loads(deserialize_message(data, String).data)
            key = (payload["stamp"]["sec"], payload["stamp"]["nanosec"])
            dets[key] = payload["entries"]

    for key in order:
        if key in dets:
            yield key[0] + key[1] * 1e-9, maps[key], dets[key]


def replay_scene(scene: str, args) -> dict:
    scene_dir = os.path.join(args.bags_dir, scene)
    # rosbag2 writes a bag DIRECTORY; open_reader resolves the .mcap inside it. The
    # ".mcap"-suffixed form is accepted too, since early recordings used that name.
    stem = f"{scene}_sam3" + (f"_{args.variant}" if args.variant else "")
    companion = os.path.join(scene_dir, stem)
    if not os.path.exists(companion):
        legacy = f"{companion}.mcap"
        if os.path.exists(legacy):
            companion = legacy
        else:
            variant_flag = f" --variant {args.variant}" if args.variant else ""
            raise SystemExit(
                f"no companion bag {companion}\n"
                f"record it first:  just map3d-record {scene} 5 \"{variant_flag.strip()}\"")

    manifest_path = os.path.join(scene_dir, f"{stem}.manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    objects = manifest.get("objects")
    if objects is None:
        objects = json.load(open(args.prompts))["scenes"][scene]["objects"]

    # Prefer what the recorder stamped into the manifest — the companion bag's detections
    # were produced under that config, so replaying them under a different platform (i.e.
    # different camera extrinsics) would silently project every point wrong.
    config = load_config(resolve_config_path(args.config), required_keys=())
    runtime = manifest.get("runtime") or config.get("runtime", {})
    platform = manifest.get("platform") or config.get("platform") or args.platform
    before = float(runtime.get("cloud_window_before", 0.5))
    after = float(runtime.get("cloud_window_after", 0.1))
    bias = float(config.get("detection_linear_state_time_bias", 0.0))

    table = PromptTable(objects)
    # Same yaml the node reads, so a bench result is a statement about the deployed
    # configuration and not about whatever the defaults happened to be.
    mapping_config = MappingConfig.from_dict(config.get("mapping", {}))
    mapper = ObjMapper(
        cloud_image_fusion=CloudImageFusion(platform=platform,
                                            bounds_mode=mapping_config.bounds_mode),
        label_template=table.label_template(),
        captioner=None,
        log_info=(lambda m: None) if args.quiet else (lambda m: print(f"  {m}", flush=True)),
        config=mapping_config,
    )

    bridge = CvBridge()
    clouds, cloud_stamps, odom_stack, odom_stamps = read_source_bag(scene_dir)

    started = time.perf_counter()
    frames = skipped_odom = skipped_cloud = 0
    per_frame_ms = []
    for stamp, id_map, entries in read_companion_bag(companion, bridge):
        if args.limit and frames >= args.limit:
            break
        odom, status = frame_sync.interpolate_odom(odom_stack, odom_stamps, stamp + bias)
        if odom is None:
            skipped_odom += 1
            continue
        cloud = frame_sync.gather_cloud(clouds, cloud_stamps, stamp, before, after)
        if cloud is None:
            skipped_cloud += 1
            continue

        detections = frame_sync.reconstruct_detections(id_map, entries)
        t0 = time.perf_counter()
        mapper.update_map(detections, stamp, odom, cloud)
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        frames += 1

    elapsed = time.perf_counter() - started
    # describe BEFORE serialize, so the funnel's unpublished counter reflects this call
    # rather than being double-counted by both.
    tracked = mapper.describe_objects()
    objects_dict = mapper.serialize_map_to_dict()
    # The map carries a `_schema` entry alongside the track ids, which is not an object and
    # must not be counted as one.
    n_objects = sum(1 for key in objects_dict if not str(key).startswith("_"))

    result = {
        "scene": scene,
        "frames_processed": frames,
        "skipped_no_odom": skipped_odom,
        "skipped_no_cloud": skipped_cloud,
        "wall_seconds": round(elapsed, 2),
        "update_map_ms": {
            "p50": round(float(np.percentile(per_frame_ms, 50)), 1) if per_frame_ms else None,
            "p99": round(float(np.percentile(per_frame_ms, 99)), 1) if per_frame_ms else None,
        },
        "n_objects": n_objects,
        "n_tracked": len(tracked),
        # Per-label, per-stage counts of every place a detection can be discarded. Whole
        # classes vanish between detection and map; this is what makes that visible.
        "funnel": {label: dict(stages) for label, stages in mapper.funnel.items()},
        # Every TRACKED object, published or not — the difference between the two is the
        # step the funnel could not see.
        "tracked_objects": tracked,
        "objects": objects_dict,
        # Subsampled robot path. Carried here so score_map3d can tell "the mapper missed
        # it" from "the robot never went near it" — an M2 failure from an M1 one — without
        # itself needing to open a bag (which would make the scorer container-only).
        "trajectory": [[round(float(v), 3) for v in odom_stack[i]["position"]]
                       for i in range(0, len(odom_stack), max(1, len(odom_stack) // 500))],
        "source": {
            "companion_bag": os.path.basename(companion),
            "prompts": table.prompts,
            "platform": platform,
            "recorded_at": manifest.get("recorded_at"),
            "config_path": manifest.get("config_path"),
        },
        "replay_run_id": _run_id(),
    }
    return result


def _run_id() -> str:
    """Identifier for this code revision, used to key the output directory.

    The container mounts scripts/ without a .git, so `git rev-parse` cannot work here —
    the justfile computes the sha on the host and passes it in. Falling back to a plain
    timestamp keeps direct invocations working without silently colliding with a
    previous run's directory.
    """
    env = os.environ.get("MAP3D_RUN_ID")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
             "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return time.strftime("dev-%Y%m%d-%H%M%S")


def _map_digest(result: dict) -> str:
    """Hash of the object map alone — excludes timings, which legitimately vary."""
    payload = json.dumps(result["objects"], sort_keys=True, default=_jsonable)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _print_funnel(funnel: dict) -> None:
    """Where each label's detections were lost, stage by stage.

    Read left to right: every detection either survives to `created`/`merged`, or is
    accounted for by exactly one drop column. `erode%` is the fraction of mask pixels that
    survive B2 and `erased` counts masks it deleted outright — the asymmetry between a
    143x54 sofa (~80% survives) and a 4x17 book (~38%) is the kind of thing a per-frame log
    never shows.
    """
    if not funnel:
        return

    # Frame-level, not per label: B6 now runs BEFORE the projection, so points it drops are
    # never attributed to a mask. Printed here so the range filter stays visible.
    frame = funnel.get("_frame", {})
    total, beyond = frame.get("cloud_points_in", 0), frame.get("cloud_points_beyond_range", 0)
    if total:
        print(f"    cloud: {total / 1000:.0f}k points in, {beyond / max(total, 1):.0%} "
              f"beyond the {'range filter' if beyond else 'range filter (disabled)'}")

    hdr = (f"    {'label':<12}{'dets':>6}{'erode%':>8}{'ero_it':>7}{'erased':>8}"
           f"{'pts/det':>9}{'gapcut':>8}{'0pts':>6}{'sparse':>8}"
           f"{'created':>9}{'merged':>8}{'wmerge':>8}{'covis':>7}{'pruned':>8}"
           f"{'vox/det':>9}{'unpub':>8}")
    print(hdr)
    for label in sorted(funnel, key=lambda k: -funnel[k].get("detections_in", 0)):
        if label == "_frame":
            continue
        s = funnel[label]
        dets = s.get("detections_in", 0)
        before = s.get("mask_px_before_erosion", 0)
        after = s.get("mask_px_after_erosion", 0)
        kept = f"{after / before:.0%}" if before else "-"
        considered = dets - s.get("dropped_low_confidence", 0) or 1
        surviving = s.get("created_new", 0) + s.get("merged_into_existing", 0)
        print(f"    {label:<12}{dets:>6}{kept:>8}"
              f"{s.get('erosion_iterations', 0) / considered:>7.1f}"
              # Masks erosion deleted outright. A sliver can score "1 iteration" and still
              # vanish, so this is NOT implied by the erode% average.
              f"{s.get('erased_by_erosion', 0):>8}"
              # Lidar points this mask claimed, per detection that got past B1.
              f"{s.get('points_in_mask', 0) / considered:>9.1f}"
              # Points B7 removed as sitting behind the object. 0 when range_gap is off.
              f"{s.get('points_dropped_range_gap', 0):>8}"
              f"{s.get('dropped_zero_points', 0):>6}{s.get('dropped_sparse_points', 0):>8}"
              f"{s.get('created_new', 0):>9}{s.get('merged_into_existing', 0):>8}"
              f"{s.get('world_merged', 0):>8}"
              # merges D8 refused because the two ids were seen in one frame (B7's sibling:
              # a criterion distance cannot express)
              f"{s.get('merge_blocked_covisible', 0):>7}"
              f"{s.get('pruned_too_few_voxels', 0):>8}"
              # voxels contributed PER SURVIVING DETECTION (the counter accumulates over
              # every detection that reaches the merge/create branch, not only new objects)
              f"{s.get('voxels_at_creation', 0) / max(surviving, 1):>9.1f}"
              f"{s.get('unpublished_no_centroid', 0):>8}")


def _print_tracked(tracked: list) -> None:
    """Every tracked object and whether it survives to the output.

    `voxels_total` -> `voxels_regularized` -> `voxels_valid` is the whole tail of the
    pipeline: DBSCAN + DIMENSION_PRIORS acceptance, then the diversity trim. An object
    reaching voxels_valid == 0 is silently dropped by serialize_map_to_dict, which is the
    step no per-detection counter can see.
    """
    if not tracked:
        return
    hdr = (f"    {'label':<12}{'ids':<18}{'vox':>7}{'regul':>7}{'valid':>7}"
           f"{'clust':>7}{'noise':>7}{'pub':>5}   why rejected")
    print(hdr)
    for t in sorted(tracked, key=lambda t: (t["label"], -t["voxels_total"])):
        r = t.get("reject", {})
        why = " ".join(f"{k}={v}" for k, v in r.items() if v) or "-"
        print(f"    {t['label']:<12}{str(t['ids'])[:17]:<18}{t['voxels_total']:>7}"
              f"{t['voxels_regularized']:>7}{t['voxels_valid']:>7}"
              f"{t['n_clusters']:>7}{t['n_noise']:>7}"
              f"{'yes' if t['published'] else 'NO':>5}   {why}")


def _run_one(scene_and_args):
    scene, args = scene_and_args
    return replay_scene(scene, args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="arabic_room", help="scene name, or 'all'")
    ap.add_argument("--bags-dir", default="/data/bags")
    ap.add_argument("--prompts", default="/data/benchmark/bench_prompts.json",
                    help="fallback prompt source when a companion manifest is missing")
    ap.add_argument("--config", default="sam3_mecanum_sim.yaml",
                    help="node config yaml (package config/ name, or an absolute path); "
                         "the companion manifest wins where it disagrees")
    ap.add_argument("--platform", default="mecanum_sim",
                    help="last-resort fallback if neither manifest nor config names one")
    ap.add_argument("--variant", default=None,
                    help="companion-bag variant to replay, e.g. 'focus4' -> "
                         "<scene>_sam3_focus4/. Must match the recording's --variant")
    ap.add_argument("--out-dir", default="/data/runs/map3d")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--jobs", type=int, default=1, help="scenes to replay in parallel")
    ap.add_argument("--quiet", action="store_true", help="suppress ObjMapper logging")
    ap.add_argument("--determinism-check", action="store_true",
                    help="replay each scene twice and fail unless the maps are identical")
    args = ap.parse_args(argv)

    if args.scene == "all":
        scenes = sorted({
            os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(args.bags_dir, "*", "*_sam3"))
            + glob.glob(os.path.join(args.bags_dir, "*", "*_sam3.mcap"))})
        if not scenes:
            raise SystemExit(f"no companion bags under {args.bags_dir}/*/*_sam3")
    else:
        scenes = [args.scene]

    # Variant results go to their own run directory: a 4-class recording and a 19-class one
    # are different experiments and must not overwrite each other's scores.
    out_dir = os.path.join(args.out_dir, _run_id() + (f"-{args.variant}" if args.variant else ""))
    os.makedirs(out_dir, exist_ok=True)

    if args.jobs > 1 and len(scenes) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_run_one, [(s, args) for s in scenes]))
    else:
        results = [replay_scene(s, args) for s in scenes]

    failures = 0
    for result in results:
        scene = result["scene"]
        digest = _map_digest(result)
        result["map_digest"] = digest
        path = os.path.join(out_dir, f"{scene}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2, default=_jsonable)
            fh.write("\n")
        line = (f"{scene:<18} {result['frames_processed']:>4} frames  "
                f"{result['n_objects']:>3} objects  "
                f"{result['wall_seconds']:>6.1f}s  "
                f"update_map p50={result['update_map_ms']['p50']}ms  {digest}")
        if result["skipped_no_odom"] or result["skipped_no_cloud"]:
            line += (f"  [skipped {result['skipped_no_odom']} odom, "
                     f"{result['skipped_no_cloud']} cloud]")
        print(line)
        _print_funnel(result.get("funnel", {}))
        _print_tracked(result.get("tracked_objects", []))

        if args.determinism_check:
            again = _map_digest(replay_scene(scene, args))
            if again != digest:
                print(f"  NONDETERMINISTIC: {digest} != {again}", file=sys.stderr)
                failures += 1

    print(f"\nwrote {len(results)} scene report(s) to {out_dir}")
    if failures:
        print(f"{failures} scene(s) failed the determinism check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
