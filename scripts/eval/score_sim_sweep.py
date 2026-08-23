#!/usr/bin/env python3
"""Decompose a live cat-2 sweep into the three losses that produce its score.

    just sim-sweep-score [<report>]

`challenge_report_sim_cat2.json` already carries the score. What it cannot say is WHY a
question scored what it did, and the three causes want completely different work:

    recall     the GT object never reached the map        -> perception / exploration
    selection  it was in the map and we published another -> the reasoner
    geometry   we published it and the box was wrong      -> map_node

Reading them apart is the whole point. On the sweep this was written for, the GT object was
in the map within 0.5 m in 68% of questions and the reasoner picked it 74% of the time, so
neither recall nor selection was the binding constraint -- box geometry was, because cat 2 is
graded `2 * IoU3D` and nothing else. A run whose score moves for the wrong reason looks
identical in the report and completely different here.

The oracle line is the number to watch across an A/B: it is the score a perfect reasoner
would have got from the map the run actually built, i.e. the ceiling the answer path is
allowed to reach.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same host/container split as score_map3d.py, and for the same reason: scripts/ is
# bind-mounted at /home/docker/scripts in the container, so a repo-relative path resolves to
# the wrong place there.
DATA_ROOT = os.environ.get("MAP3D_DATA_ROOT") or (
    os.path.join(REPO, "data") if os.path.isdir(os.path.join(REPO, "data")) else "/data")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import iou3d  # noqa: E402  -- one IoU in the tree, not two

#: eval_orchestrator.HIT_IOU. Duplicated rather than imported because that module needs rclpy.
HIT_IOU = 0.25

#: How close a map entry's centre must be to GT's to count as "the same object". Well above
#: the measured median centroid error (0.09 m) and well below the spacing of two same-class
#: instances, so it separates "wrong box" from "wrong object" rather than blurring them.
SAME_OBJECT_M = 0.5


def canon(label: str) -> str:
    """sam_mapper.detections.default_label, without importing a ROS-adjacent package."""
    return str(label or "").strip().replace(" ", "").lower()


def load_gt(scene: str, qid: str) -> dict | None:
    path = os.path.join(DATA_ROOT, "benchmark", scene, "category_2",
                        f"{scene}_category2_qa.json")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    questions = payload if isinstance(payload, list) else payload.get("questions", [])
    return next((q for q in questions if q.get("id") == qid), None)


def load_map(best_view_dir: str | None) -> dict:
    """The obj_map.json this question's run wrote, keyed by map id.

    best_view_dir is a container path; rewrite it onto DATA_ROOT so the report scores the
    same on the host that produced it.
    """
    if not best_view_dir:
        return {}
    path = best_view_dir
    if path.startswith("/data/"):
        path = os.path.join(DATA_ROOT, path[len("/data/"):])
    try:
        with open(os.path.join(path, "obj_map.json"), encoding="utf-8") as handle:
            return json.load(handle) or {}
    except (OSError, ValueError):
        return {}


def percentiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    n = len(ordered)
    return (f"median {statistics.median(ordered):+.3f}  "
            f"p25 {ordered[n // 4]:+.3f}  p75 {ordered[3 * n // 4]:+.3f}  "
            f"max {ordered[-1]:+.3f}")


def analyse(rows: list[dict]) -> dict:
    out = {
        "n": 0, "absent": 0, "wrong_instance": 0, "present": 0,
        "achieved_iou": [], "achieved_hits": 0,
        "oracle_iou": [], "oracle_hits": 0,
        "picked_oracle": 0, "picked_other": 0,
        "horiz_err": [], "vert_err": [], "centre_xy": [], "centre_z": [],
        "misses": [],
    }
    for row in rows:
        if row.get("category") != 2:
            continue
        entry = load_gt(row["scene"], row["id"])
        if entry is None:
            continue
        out["n"] += 1
        answer = entry["answer"]
        gt_c, gt_s = answer["center"], answer["size"]
        label = canon(answer.get("label"))

        marker = row.get("marker") or {}
        achieved = 0.0 if marker.get("placeholder", True) else float(row.get("iou") or 0.0)
        out["achieved_iou"].append(achieved)
        out["achieved_hits"] += achieved >= HIT_IOU

        obj_map = load_map(row.get("best_view_dir"))
        same = [(k, o) for k, o in obj_map.items() if canon(o.get("label")) == label]
        if not same:
            out["absent"] += 1
            out["oracle_iou"].append(0.0)
            out["misses"].append((row["scene"], row["id"], label, "label absent from map"))
            continue

        # Nearest by centre decides "is the GT object here at all"; best by IoU decides what a
        # perfect reasoner could have scored. They are usually the same entry and must not be
        # conflated: a duplicate fragment can be nearer while a merged box overlaps better.
        nearest = min(same, key=lambda ko: math.dist(ko[1]["bbox3d"]["center"], gt_c))
        if math.dist(nearest[1]["bbox3d"]["center"], gt_c) > SAME_OBJECT_M:
            out["wrong_instance"] += 1
            out["misses"].append((row["scene"], row["id"], label,
                                  f"{len(same)} in map, nearest "
                                  f"{math.dist(nearest[1]['bbox3d']['center'], gt_c):.2f} m away"))
        else:
            out["present"] += 1

        best_id, best = max(
            same, key=lambda ko: iou3d(ko[1]["bbox3d"]["center"], ko[1]["bbox3d"]["extent"],
                                       gt_c, gt_s))
        best_iou = iou3d(best["bbox3d"]["center"], best["bbox3d"]["extent"], gt_c, gt_s)
        out["oracle_iou"].append(best_iou)
        out["oracle_hits"] += best_iou >= HIT_IOU

        if not marker.get("placeholder", True):
            picked = str(marker.get("id"))
            if picked == str(best_id):
                out["picked_oracle"] += 1
            else:
                out["picked_other"] += 1

        # Orientation-free: compare the two horizontal extents sorted, z on its own. A box may
        # be right and yawed; an axis-by-axis diff would call that an extent error.
        ext, ctr = best["bbox3d"]["extent"], best["bbox3d"]["center"]
        map_h, gt_h = sorted(ext[:2]), sorted(gt_s[:2])
        out["horiz_err"] += [map_h[0] - gt_h[0], map_h[1] - gt_h[1]]
        out["vert_err"].append(ext[2] - gt_s[2])
        out["centre_xy"].append(math.hypot(ctr[0] - gt_c[0], ctr[1] - gt_c[1]))
        out["centre_z"].append(abs(ctr[2] - gt_c[2]))
    return out


def report(a: dict, verbose: bool) -> None:
    n = a["n"]
    if not n:
        print("no category-2 rows with matching benchmark ground truth", file=sys.stderr)
        return
    mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731

    print(f"category-2 sweep decomposition — {n} question(s)\n")

    print("RECALL   is the GT object in the map at all?")
    print(f"  present (nearest same-label box within {SAME_OBJECT_M} m)  "
          f"{a['present']:>3}/{n}  {100 * a['present'] / n:.0f}%")
    print(f"  label present, wrong instance                    "
          f"{a['wrong_instance']:>3}/{n}  {100 * a['wrong_instance'] / n:.0f}%")
    print(f"  label absent from the map                        "
          f"{a['absent']:>3}/{n}  {100 * a['absent'] / n:.0f}%\n")

    print("GEOMETRY best same-label box vs GT (the ceiling a perfect reasoner reaches)")
    print(f"  oracle  hits@{HIT_IOU}  {a['oracle_hits']:>3}/{n}  {100 * a['oracle_hits'] / n:.0f}%"
          f"    mean score {2 * mean(a['oracle_iou']):.2f}/2")
    print(f"  achieved hits@{HIT_IOU} {a['achieved_hits']:>3}/{n}  "
          f"{100 * a['achieved_hits'] / n:.0f}%    mean score {2 * mean(a['achieved_iou']):.2f}/2")
    print(f"  loss to selection                    "
          f"{2 * (mean(a['oracle_iou']) - mean(a['achieved_iou'])):.2f}/2\n")

    print("SELECTION did the reasoner publish the map's best entry?")
    picked = a["picked_oracle"] + a["picked_other"]
    if picked:
        print(f"  yes {a['picked_oracle']:>3}/{picked}  {100 * a['picked_oracle'] / picked:.0f}%"
              f"    no {a['picked_other']}\n")
    else:
        print("  no non-placeholder answers\n")

    print("EXTENT error, orientation-free (map - GT, metres)")
    print(f"  horizontal  {percentiles(a['horiz_err'])}")
    print(f"  vertical    {percentiles(a['vert_err'])}")
    print(f"  centre xy   {percentiles(a['centre_xy'])}")
    print(f"  centre z    {percentiles(a['centre_z'])}")

    if verbose and a["misses"]:
        print("\nquestions whose GT object never reached the map:")
        for scene, qid, label, why in a["misses"]:
            print(f"  {scene:<15}{qid:<5}{label:<18}{why}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report",
                    default=os.path.join(DATA_ROOT, "runs", "challenge_report_sim_cat2.json"))
    ap.add_argument("--verbose", action="store_true",
                    help="list the questions whose GT object never reached the map")
    args = ap.parse_args()

    try:
        with open(args.report, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 1

    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        print(f"{args.report} does not hold a results list", file=sys.stderr)
        return 1

    report(analyse(rows), args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
