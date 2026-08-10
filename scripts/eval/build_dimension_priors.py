#!/usr/bin/env python3
"""Derive per-class size CAPS (D3) from VLA-3D ground truth. One-off generator.

Caps are upper bounds for rejecting bled clusters, NOT expected sizes. The statistic is
p90 x margin, not max, which a handful of mislabelled annotations can set on their own.
Output ships inside the package: there is no ground truth at challenge time.

    python3 scripts/eval/build_dimension_priors.py [--write]"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Both GT roots (13 shared + home_building_1/_2). data/bags/office_building_*/ exist but
#: are empty; when that GT lands, --write picks it up with no code change.
GT_GLOBS = (
    os.path.join(REPO, "data", "bags", "*", "iref_vla_metadata", "*_object_data.json"),
    os.path.join(os.path.dirname(REPO), "IRef-VLA", "unity", "Unity", "*", "*_object_data.json"),
)

#: Inside the package, not under data/, because it is loaded at RUNTIME and must ship with
#: the installed node — data/ is a gitignored host bind-mount that will not exist on the
#: challenge machine. setup.py installs it via package_data.
OUT_PATH = os.path.join(REPO, "ai_module", "src", "sam_mapper", "sam_mapper",
                        "dimension_priors.json")

#: p90 rather than max: five 2.93 m bench-style seats labelled `chair` would otherwise set
#: the chair cap to 3.66 m, wide enough to re-admit the merge blobs D3 exists to reject.
#: Margin and slack were chosen by bench measurement — see docs/map_node_pipeline.md.
P90_MIN_SAMPLES = 5


def margin_for(n: int) -> float:
    if n >= 20:
        return 1.3
    if n >= P90_MIN_SAMPLES:
        return 1.45
    return 1.6


#: Never emit a cap below this on any axis. Guards against a class whose only instances are
#: tiny or degenerate (a 4 cm picture) producing a cap that measurement noise alone exceeds.
MIN_CAP = 0.20

#: No axis may be capped below this fraction of the largest instance seen. See build().
MAX_FLOOR_FRACTION = 0.5

#: Added to every axis, in metres. THE CAPS ARE DERIVED FROM TRUE OBJECT DIMENSIONS BUT
#: APPLIED TO MEASURED CLUSTER EXTENTS, and the latter are systematically larger:
#:   * get_box_3d adds +voxel_size to every extent (a voxel is a cell, not a point)
#:   * voxel quantisation snaps points to cell centres, +-voxel_size/2 per side
#:   * DBSCAN eps = 3 voxels admits strays attached to the object
#: Roughly two voxels at voxel_size 0.05. Revisit if the voxel size changes.
#:
#: It has to be ADDITIVE. The inflation is constant in metres, so a multiplicative margin
#: helps least exactly where it is needed most — x1.3 adds 0.11 m to a 0.38 m vase but only
#: 0.06 m to a 0.20 m bowl. Measured: raising the margin 1.2 -> 1.3 moved japanese_room,
#: livingroom_3, office_2 and studio by EXACTLY ZERO, and those four scenes are the ones
#: whose question targets are small classes (lantern 0.27, bowl 0.38, beerbottle on the
#: 0.20 floor). Same shape of error as the erosion bug: a constant-magnitude effect that
#: was modelled as proportional.
MEASUREMENT_SLACK = 0.10

#: Applies to every class with no entry. Deliberately generous: an unknown class is exactly
#: the case where a wrong cap silently deletes objects, and D3's job is catching gross
#: bleed, not fine-grained sizing.
DEFAULT_CAP = (6.0, 6.0, 3.5)


def scan_ground_truth() -> tuple[dict, list]:
    """

{label: [size triples]}, and the scene names contributing. Needs the GT roots."""
    sizes = collections.defaultdict(list)
    scenes = {}
    for pattern in GT_GLOBS:
        for path in sorted(glob.glob(pattern)):
            scene = os.path.basename(path).replace("_object_data.json", "")
            if scene in scenes:            # same scene from both roots — count once
                continue
            scenes[scene] = path
            for region in json.load(open(path)).values():
                for obj in region.values():
                    # Same normalisation detections.default_label applies to prompts, so
                    # generated keys match the labels get_dominant_label() will look up.
                    label = (obj.get("raw_label") or "").strip().replace(" ", "").lower()
                    if label:
                        sizes[label].append(np.asarray(obj["size"], dtype=float))
    return sizes, sorted(scenes)


def build(sizes: dict) -> dict:
    priors = {}
    for label, rows in sizes.items():
        arr = np.array(rows)
        # Horizontals sorted descending, matching _fits_prior's comparison, so the cap is
        # orientation-independent exactly as the test is.
        horiz = np.sort(arr[:, :2], axis=1)[:, ::-1]
        m = margin_for(len(arr))
        # Per axis independently, matching how _fits_prior tests them.
        axes = (horiz[:, 0], horiz[:, 1], arr[:, 2])
        if len(arr) < P90_MIN_SAMPLES:
            priors[label] = tuple(round(max(float(v.max()) * m, MIN_CAP) + MEASUREMENT_SLACK, 2)
                                  for v in axes)
            continue
        # A per-axis p90 collapses on classes whose shape is bimodal: most `window`s are a
        # few cm deep and a handful are 2.69 m, so p90 of that axis lands at 0.20 and the
        # deep ones are rejected. `ceilinglight` collapsed to (0.21, 0.21, 0.20) against a
        # real 1.29 x 1.26. The floor says no axis may fall below HALF the largest instance
        # seen — a cap under that is an artefact of a skewed distribution rather than a
        # statement about the class. It leaves `chair` at 1.58 (p90 still wins there) while
        # restoring window h2 to 1.35 and lifting overall acceptance 96.4% -> 96.8%.
        priors[label] = tuple(
            round(max(float(np.percentile(v, 90)) * m, float(v.max()) * MAX_FLOOR_FRACTION,
                      MIN_CAP) + MEASUREMENT_SLACK, 2)
            for v in axes)
    return priors


def acceptance_rate(priors: dict, sizes: dict) -> float:
    """

Fraction of the source objects that still fit their own class cap.

    Uses the same sorted-horizontals comparison _fits_prior applies, on ALL THREE axes —
    scoring one axis reported 97.6% for a rule whose true rate was 95.9%.
    """
    kept = total = 0
    for label, rows in sizes.items():
        prior = sorted(priors[label][:2], reverse=True), priors[label][2]
        for size in np.asarray(rows):
            horiz = sorted(size[:2], reverse=True)
            total += 1
            kept += (horiz[0] <= prior[0][0] and horiz[1] <= prior[0][1]
                     and size[2] <= prior[1])
    return kept / max(total, 1)


def render(priors: dict, scenes: list, sizes: dict) -> dict:
    # Computed, never hand-written: an earlier version stated the margins and hit rate as
    # literals and they went stale the moment the margin changed, leaving a shipped artefact
    # describing itself wrongly.
    margins = f"{margin_for(20)} at n>=20, {margin_for(P90_MIN_SAMPLES)} at n>={P90_MIN_SAMPLES}"
    return {
        "_meta": {
            "generated": str(datetime.date.today()),
            "generator": "scripts/eval/build_dimension_priors.py",
            "scenes": scenes,
            "n_objects": sum(len(v) for v in sizes.values()),
            "n_classes": len(priors),
            "rule": (f"p90 x margin ({margins}; max x {margin_for(1)} below n={P90_MIN_SAMPLES}), "
                     f"floored so no axis falls under {MAX_FLOOR_FRACTION} of the largest "
                     f"instance seen; min {MIN_CAP} m; plus {MEASUREMENT_SLACK} m additive "
                     f"slack for voxel-quantisation inflation of measured clusters"),
            "accepts": (f"{acceptance_rate(priors, sizes):.1%} of the source objects still "
                        f"fit their class cap"),
            "note": ("UPPER BOUNDS for rejecting bled clusters, not expected sizes. "
                     "Generated — do not hand-edit; rerun the generator."),
        },
        "default": list(DEFAULT_CAP),
        # sorted by sample count so the most-supported caps read first
        "priors": {label: list(priors[label])
                   for label in sorted(priors, key=lambda k: (-len(sizes[k]), k))},
        "n_samples": {label: len(sizes[label]) for label in priors},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"write {OUT_PATH}")
    ap.add_argument("--show", type=int, default=25, help="rows to print")
    args = ap.parse_args()

    sizes, scenes = scan_ground_truth()
    if not sizes:
        raise SystemExit("no ground truth found — checked:\n  " + "\n  ".join(GT_GLOBS))
    priors = build(sizes)

    print(f"{len(scenes)} scenes, {sum(len(v) for v in sizes.values())} objects, "
          f"{len(priors)} classes\n")
    print(f"{'class':<20}{'n':>5}{'max h1':>8}{'max h2':>8}{'max z':>7}   cap")
    print("-" * 70)
    for label in sorted(priors, key=lambda k: -len(sizes[k]))[:args.show]:
        arr = np.array(sizes[label])
        h = np.sort(arr[:, :2], axis=1)[:, ::-1]
        print(f"{label:<20}{len(arr):>5}{h[:,0].max():>8.2f}{h[:,1].max():>8.2f}"
              f"{arr[:,2].max():>7.2f}   {priors[label]}")

    if args.write:
        with open(OUT_PATH, "w") as fh:
            json.dump(render(priors, scenes, sizes), fh, indent=1, sort_keys=False)
            fh.write("\n")
        print(f"\nwrote {OUT_PATH}")
    else:
        print("\n(dry run — pass --write to regenerate the module)")


if __name__ == "__main__":
    main()
