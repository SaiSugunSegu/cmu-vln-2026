#!/usr/bin/env python3
"""Verify benchmark_2 category-1 answers against the iref_vla object metadata.

Every question in a benchmark_2 QA file may carry a machine-readable ``spec``
describing, as a selector over scene objects, what its integer answer counts.
This script re-evaluates each spec against the scene's object bounding boxes and
compares the result with the stated ``answer``, so the whole benchmark can be
audited without trusting the hand-written numbers.

Selector grammar (a JSON object)::

    {
      "label":   ["sofa"],           # raw_label match, case-insensitive
      "color":   ["maroon", "red"],  # primary color_labels[0] match
      "regions": ["0"],              # iref region ids (a scene may be split by room)
      "ids":     ["40", "74"],       # restrict to explicit object ids
      "exclude_ids": ["12"],
      "bottom_below": 1.0,           # object's underside is below this height (m)
      "bottom_above": 0.4,           # object's underside is above this height (m)
      "filters": [                  # every filter must hold (logical AND)
        {"rel": "below", "target": {"label": ["window"]}}
      ]
    }

Relations (``rel``), each satisfied when ANY object in ``target`` matches:

    below       self footprint under the target, target sits above self
    above       inverse of below
    on          self rests on the target's top surface
    supports    something in target rests on self's top surface
    inside_xy   self's centre falls within the target's footprint
    contains    a target's centre falls within self's footprint
    near        self's centre within ``max_dist`` metres of the target's centre
    not         negation of the nested ``filters``

Usage::

    python3 scripts/eval/verify_benchmark2.py                # all scenes
    python3 scripts/eval/verify_benchmark2.py arabic_room
    python3 scripts/eval/verify_benchmark2.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

BENCH_ROOT = Path("data/benchmark_2")
BAGS_ROOT = Path("bags")

# Wall-mounted objects sit a few centimetres behind the furniture in front of
# them, so footprints are dilated before testing "below" / "above" / "inside".
FOOTPRINT_TOL = 0.15
# Tolerance on the gap between a supported object's underside and a support's top.
REST_TOL = 0.08

SCENE_NAME_RE = re.compile(r"^[a-z0-9_]+$")


class Obj:
    __slots__ = ("id", "region", "label", "colors", "lo", "hi", "center")

    def __init__(self, oid: str, region: str, raw: dict[str, Any]):
        bbox = np.asarray(raw["bbox"], dtype=float)
        self.id = oid
        self.region = region
        self.label = str(raw.get("raw_label", "")).lower()
        self.colors = [str(c).lower() for c in (raw.get("color_labels") or [])]
        self.lo = bbox.min(axis=0)
        self.hi = bbox.max(axis=0)
        self.center = bbox.mean(axis=0)

    @property
    def primary_color(self) -> str:
        return self.colors[0] if self.colors else ""

    def __repr__(self) -> str:
        return f"{self.label}#{self.id}"


def load_scene(scene: str) -> list[Obj]:
    if not SCENE_NAME_RE.match(scene):
        raise ValueError(f"invalid scene name: {scene!r}")
    path = BAGS_ROOT / scene / "iref_vla_metadata" / f"{scene}_object_data.json"
    resolved = path.resolve()
    if not str(resolved).startswith(str(BAGS_ROOT.resolve())):
        raise ValueError(f"scene path escapes {BAGS_ROOT}: {scene!r}")
    raw = json.loads(resolved.read_text())
    return [
        Obj(oid, region, obj)
        for region, entries in raw.items()
        for oid, obj in entries.items()
    ]


def footprints_overlap(a: Obj, b: Obj, tol: float = FOOTPRINT_TOL) -> bool:
    return (
        a.lo[0] - tol <= b.hi[0]
        and b.lo[0] - tol <= a.hi[0]
        and a.lo[1] - tol <= b.hi[1]
        and b.lo[1] - tol <= a.hi[1]
    )


def center_inside(inner: Obj, outer: Obj, tol: float = FOOTPRINT_TOL) -> bool:
    return (
        outer.lo[0] - tol <= inner.center[0] <= outer.hi[0] + tol
        and outer.lo[1] - tol <= inner.center[1] <= outer.hi[1] + tol
    )


def rests_on(item: Obj, support: Obj) -> bool:
    return center_inside(item, support, tol=0.05) and abs(item.lo[2] - support.hi[2]) <= REST_TOL


def relation_holds(obj: Obj, spec: dict[str, Any], scene: list[Obj]) -> bool:
    rel = spec["rel"]
    if rel == "not":
        return not all(relation_holds(obj, f, scene) for f in spec["filters"])

    targets = [t for t in select(scene, spec["target"]) if t.id != obj.id]
    if rel == "below":
        return any(footprints_overlap(obj, t) and t.lo[2] >= obj.hi[2] for t in targets)
    if rel == "above":
        return any(footprints_overlap(obj, t) and obj.lo[2] >= t.hi[2] for t in targets)
    if rel == "on":
        return any(rests_on(obj, t) for t in targets)
    if rel == "supports":
        return any(rests_on(t, obj) for t in targets)
    if rel == "inside_xy":
        return any(center_inside(obj, t) for t in targets)
    if rel == "contains":
        return any(center_inside(t, obj) for t in targets)
    if rel == "near":
        max_dist = float(spec["max_dist"])
        return any(float(np.linalg.norm(obj.center[:2] - t.center[:2])) <= max_dist for t in targets)
    raise ValueError(f"unknown relation: {rel!r}")


def select(scene: list[Obj], spec: dict[str, Any]) -> list[Obj]:
    out = scene
    if ids := spec.get("ids"):
        wanted = set(ids)
        out = [o for o in out if o.id in wanted]
    if excluded := spec.get("exclude_ids"):
        drop = set(excluded)
        out = [o for o in out if o.id not in drop]
    if labels := spec.get("label"):
        wanted = {label.lower() for label in labels}
        out = [o for o in out if o.label in wanted]
    if colors := spec.get("color"):
        wanted = {c.lower() for c in colors}
        out = [o for o in out if o.primary_color in wanted]
    if regions := spec.get("regions"):
        wanted = {str(r) for r in regions}
        out = [o for o in out if o.region in wanted]
    if (limit := spec.get("bottom_below")) is not None:
        out = [o for o in out if o.lo[2] <= float(limit)]
    if (limit := spec.get("bottom_above")) is not None:
        out = [o for o in out if o.lo[2] >= float(limit)]
    for f in spec.get("filters", []):
        out = [o for o in out if relation_holds(o, f, scene)]
    return out


def qa_files(scenes: list[str] | None) -> list[Path]:
    found = sorted(BENCH_ROOT.glob("*/category_1/*_category1_qa.json"))
    if scenes:
        wanted = set(scenes)
        found = [p for p in found if p.parts[-3] in wanted]
    return found


def verify_scene(path: Path, verbose: bool) -> tuple[int, int, int]:
    qa = json.loads(path.read_text())
    scene_name = qa["scene"]
    questions = qa.get("questions") or []
    try:
        scene = load_scene(scene_name)
    except FileNotFoundError:
        print(f"[{scene_name}] SKIP no iref_vla object metadata ({len(questions)} questions unverified)")
        return 0, 0, len(questions)

    n_ok = n_bad = n_skip = 0
    for q in questions:
        qid = q["id"]
        stated = int(q["answer"])
        spec = q.get("spec")
        if not spec:
            n_skip += 1
            print(f"[{scene_name}] {qid} SKIP no spec (stated={stated})")
            continue
        matched = select(scene, spec)
        got = len(matched)
        if got == stated:
            n_ok += 1
            if verbose:
                print(f"[{scene_name}] {qid} OK  {stated}  {sorted(matched, key=lambda o: int(o.id))}")
        else:
            n_bad += 1
            print(
                f"[{scene_name}] {qid} MISMATCH stated={stated} computed={got} "
                f"matched={sorted(matched, key=lambda o: int(o.id))}\n"
                f"    {q['question']}"
            )
    status = "OK" if n_bad == 0 else "FAIL"
    print(f"[{scene_name}] {status}: {n_ok} verified, {n_bad} mismatched, {n_skip} unverified")
    return n_ok, n_bad, n_skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help="Scene names to check (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print matched objects for passing questions")
    args = ap.parse_args()

    files = qa_files(args.scenes or None)
    if not files:
        print(f"no QA files under {BENCH_ROOT}")
        return 1

    totals = [0, 0, 0]
    for path in files:
        ok, bad, skip = verify_scene(path, args.verbose)
        totals[0] += ok
        totals[1] += bad
        totals[2] += skip
        print()

    print(f"TOTAL: {totals[0]} verified, {totals[1]} mismatched, {totals[2]} unverified")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    raise SystemExit(main())
