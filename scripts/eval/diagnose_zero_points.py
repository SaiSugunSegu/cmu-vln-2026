#!/usr/bin/env python3
"""Why do detections with healthy masks receive ZERO lidar points?

Splits the two causes, which need opposite fixes: no lidar coverage where the mask sits
(sensor geometry, unfixable here) vs coverage beside the mask but not in it (alignment).

    just map3d-zeropoints <scene> [variant]"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from sam_mapper import frame_sync
from sam_mapper.cloud_image_fusion import CloudImageFusion
from sam_mapper.node_base import load_config, resolve_config_path
from sam_mapper.tools.dump_frames import open_reader

CLOUD_TOPIC, ODOM_TOPIC = "/registered_scan", "/state_estimation"
MAP_TOPIC, DET_TOPIC = "/sam3/instance_map", "/sam3/detections"

#: How far (in pixels) around a mask to look for lidar coverage before concluding the
#: region is genuinely unlit rather than narrowly missed.
NEIGHBOURHOOD_PX = 40


def read_source(scene_dir):
    reader = open_reader(scene_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CLOUD_TOPIC, ODOM_TOPIC]))
    clouds, cstamps, odom, ostamps = [], [], [], []
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == CLOUD_TOPIC:
            m = deserialize_message(data, PointCloud2)
            clouds.append(point_cloud2.read_points_numpy(m, field_names=("x", "y", "z")))
            cstamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        else:
            m = deserialize_message(data, Odometry)
            p, o = m.pose.pose.position, m.pose.pose.orientation
            lin, ang = m.twist.twist.linear, m.twist.twist.angular
            odom.append({"position": [p.x, p.y, p.z], "orientation": [o.x, o.y, o.z, o.w],
                         "linear_velocity": [lin.x, lin.y, lin.z],
                         "angular_velocity": [ang.x, ang.y, ang.z]})
            ostamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
    return clouds, cstamps, odom, ostamps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="livingroom_1")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--bags-dir", default="/data/bags")
    ap.add_argument("--config", default="sam3_mecanum_sim.yaml")
    args = ap.parse_args(argv)

    scene_dir = os.path.join(args.bags_dir, args.scene)
    stem = f"{args.scene}_sam3" + (f"_{args.variant}" if args.variant else "")
    companion = os.path.join(scene_dir, stem)
    manifest_path = os.path.join(scene_dir, f"{stem}.manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    config = load_config(resolve_config_path(args.config), required_keys=())
    runtime = manifest.get("runtime") or config.get("runtime", {})
    fusion = CloudImageFusion(platform=manifest.get("platform")
                              or config.get("platform", "mecanum_sim"))
    before = float(runtime.get("cloud_window_before", 0.5))
    after = float(runtime.get("cloud_window_after", 0.1))

    clouds, cstamps, odom, ostamps = read_source(scene_dir)
    bridge = CvBridge()

    reader = open_reader(companion)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[MAP_TOPIC, DET_TOPIC]))
    maps, dets, order = {}, {}, []
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == MAP_TOPIC:
            m = deserialize_message(data, Image)
            key = (m.header.stamp.sec, m.header.stamp.nanosec)
            maps[key] = bridge.imgmsg_to_cv2(m, desired_encoding="mono16")
            order.append(key)
        else:
            p = json.loads(deserialize_message(data, String).data)
            dets[(p["stamp"]["sec"], p["stamp"]["nanosec"])] = p["entries"]

    stats = defaultdict(lambda: defaultdict(int))
    rows_with_lidar = np.zeros(0, dtype=bool)
    per_label_rows = defaultdict(list)
    frames = []          # per-frame: (idx, n_cloud, n_inbounds, n_dets, n_zero, clipped)
    clip_total = clip_edge = 0

    for frame_idx, key in enumerate(order):
        entries = dets.get(key)
        if entries is None:
            continue
        stamp = key[0] + key[1] * 1e-9
        pose, status = frame_sync.interpolate_odom(odom, ostamps, stamp)
        if pose is None:
            continue
        cloud = frame_sync.gather_cloud(clouds, cstamps, stamp, before, after)
        if cloud is None:
            continue

        R_b2w = Rotation.from_quat(pose["orientation"]).as_matrix()
        t_b2w = np.array(pose["position"])
        cloud_body = cloud @ R_b2w + (-R_b2w.T @ t_b2w)   # world -> body
        uv = fusion.scan2pixels(cloud_body)
        id_map = maps[key]
        H, W = id_map.shape

        inb = ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        uvi = uv[inb].astype(int)
        coverage = np.zeros((H, W), dtype=bool)
        coverage[uvi[:, 1], uvi[:, 0]] = True
        if rows_with_lidar.size == 0:
            rows_with_lidar = np.zeros(H, dtype=bool)
        rows_with_lidar |= coverage.any(axis=1)

        # Defect 2 in numbers: out-of-FOV points are CLIPPED to row 0 / row H-1 rather
        # than rejected, so any mask touching an edge row silently absorbs them. This
        # also inflates the apparent lidar coverage reported above.
        on_edge = int(np.count_nonzero((uvi[:, 1] == 0) | (uvi[:, 1] == H - 1)))
        clip_total += uvi.shape[0]
        clip_edge += on_edge

        n_zero_here = 0
        recon = frame_sync.reconstruct_detections(id_map, entries)
        for label, mask in zip(recon["labels"], recon["masks"]):
            label = str(label)
            ys, xs = np.nonzero(mask)
            if not ys.size:
                stats[label]["mask_empty"] += 1
                continue
            n_in = int(np.count_nonzero(coverage & mask))
            stats[label]["detections"] += 1
            per_label_rows[label].append((int(ys.mean()), int(ys.min()), int(ys.max())))
            if n_in:
                stats[label]["has_points"] += 1
                continue
            stats[label]["zero_points"] += 1
            n_zero_here += 1
            # Widen the search: is there lidar ANYWHERE near this mask?
            y0, y1 = max(ys.min() - NEIGHBOURHOOD_PX, 0), min(ys.max() + NEIGHBOURHOOD_PX, H)
            x0, x1 = max(xs.min() - NEIGHBOURHOOD_PX, 0), min(xs.max() + NEIGHBOURHOOD_PX, W)
            near = int(np.count_nonzero(coverage[y0:y1, x0:x1]))
            stats[label]["zero_but_lidar_nearby" if near else "zero_and_no_lidar_nearby"] += 1
            # Was the mask inside the band of rows the lidar ever illuminates?
            lit = np.flatnonzero(coverage.any(axis=1))
            if lit.size and (ys.max() < lit.min() or ys.min() > lit.max()):
                stats[label]["zero_outside_lidar_rows"] += 1

        frames.append({
            "idx": frame_idx,
            "cloud": int(cloud.shape[0]),
            "inbounds": int(uvi.shape[0]),
            "dets": len(recon["labels"]),
            "zero": n_zero_here,
            "edge": on_edge,
            # A contiguous block of failing frames points at motion or pose, not geometry.
            "speed": float(np.linalg.norm(pose["linear_velocity"])),
            "yaw_rate": float(abs(pose["angular_velocity"][2])),
            "pos": np.asarray(pose["position"], dtype=float),
        })

    lit_rows = np.flatnonzero(rows_with_lidar)
    print(f"=== {args.scene} / {args.variant or 'full'} ===")
    if lit_rows.size:
        H = rows_with_lidar.size
        scale = W / (2 * np.pi)
        top = np.degrees((H / 2 - lit_rows.min()) / scale)
        bot = np.degrees((H / 2 - lit_rows.max()) / scale)
        print(f"lidar illuminates image rows {lit_rows.min()}..{lit_rows.max()} of {H} "
              f"(camera row 0..{H-1} spans +60..-60 deg)")
        print(f"  -> lidar vertical coverage ~ {bot:+.1f}..{top:+.1f} deg  "
              f"({100*lit_rows.size/H:.0f}% of the camera's rows)\n")

    hdr = (f"{'label':<10}{'dets':>6}{'has_pts':>9}{'zero':>7}"
           f"{'zero+near':>11}{'zero+none':>11}{'outside_rows':>14}{'mask_row':>10}")
    print(hdr)
    for label, s in sorted(stats.items(), key=lambda kv: -kv[1]["detections"]):
        rows = per_label_rows[label]
        med_row = int(np.median([r[0] for r in rows])) if rows else 0
        print(f"{label:<10}{s['detections']:>6}{s['has_points']:>9}{s['zero_points']:>7}"
              f"{s['zero_but_lidar_nearby']:>11}{s['zero_and_no_lidar_nearby']:>11}"
              f"{s['zero_outside_lidar_rows']:>14}{med_row:>10}")

    if clip_total:
        print(f"\n-- CLIPPING (defect 2) --")
        print(f"  {clip_edge}/{clip_total} projected points ({clip_edge/clip_total:.1%}) land on "
              f"row 0 or row {rows_with_lidar.size - 1}.")
        print(f"  These are out-of-FOV points CLAMPED onto an edge row instead of rejected,")
        print(f"  so any mask touching an edge absorbs them — and they inflate the lidar")
        print(f"  coverage band reported above.")

    # Is the zero-point failure spread evenly, or concentrated in particular frames?
    # A mean of ~380 points/detection cannot produce zeros by chance, so if the failures
    # cluster by frame the cause is per-frame (pose, timing, cloud gap), not per-object.
    if frames:
        total_dets = sum(f["dets"] for f in frames)
        total_zero = sum(f["zero"] for f in frames)
        bad = [f for f in frames if f["dets"] and f["zero"] == f["dets"]]
        part = [f for f in frames if 0 < f["zero"] < f["dets"]]
        good = [f for f in frames if f["zero"] == 0]
        print(f"\n-- PER-FRAME --")
        print(f"  frames {len(frames)} | detections {total_dets} | zero-point {total_zero}")
        print(f"  every detection zero : {len(bad)} | some : {len(part)} | none : {len(good)}")

        def summarise(name, group):
            if not group:
                return
            print(f"  {name:<22}"
                  f"cloud {int(np.median([f['cloud'] for f in group])):>7}"
                  f" | speed {np.median([f['speed'] for f in group]):>5.2f} m/s"
                  f" | yaw_rate {np.median([f['yaw_rate'] for f in group]):>5.2f} rad/s"
                  f" | inbounds {int(np.median([f['inbounds'] for f in group])):>7}")
        print()
        summarise("TOTAL-FAILURE frames", bad)
        summarise("partial frames", part)
        summarise("clean frames", good)

        if bad:
            print(f"\n  total-failure indices: {[f['idx'] for f in bad][:30]}"
                  f"{' ...' if len(bad) > 30 else ''}")
            # Where was the robot? A block of failures at one spot means the objects were
            # occluded or out of range from there; a block while moving means motion.
            p = np.array([f["pos"] for f in bad])
            q = np.array([f["pos"] for f in good]) if good else np.zeros((0, 3))
            print(f"  robot position during failures: "
                  f"x[{p[:,0].min():.2f},{p[:,0].max():.2f}] "
                  f"y[{p[:,1].min():.2f},{p[:,1].max():.2f}]"
                  f"  (travelled {np.linalg.norm(p[-1]-p[0]):.2f} m across the block)")
            if q.size:
                print(f"  robot position during clean   : "
                      f"x[{q[:,0].min():.2f},{q[:,0].max():.2f}] "
                      f"y[{q[:,1].min():.2f},{q[:,1].max():.2f}]")

    print("\n-- READING --")
    print("  zero+none  : no lidar anywhere near the mask.")
    print("  zero+near  : lidar landed beside the mask but not in it -> ALIGNMENT")
    print("               (extrinsics, odom interpolation, or cloud/image timing).")
    print("  If failures cluster into whole frames, the cause is per-frame (pose/timing/")
    print("  empty cloud). If they are spread across frames, it is per-object (geometry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
