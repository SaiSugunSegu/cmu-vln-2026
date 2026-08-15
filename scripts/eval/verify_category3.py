#!/usr/bin/env python3
"""Audit the category-3 benchmark against the recordings it was derived from.

The ground truth in ``data/benchmark/<scene>/category_3/`` is a reading of the organizers'
demonstration trajectories. A reading can be wrong, and a benchmark that disagrees with
the runs it came from is worse than none: it would send the planner away from behaviour
that is known to be correct.

So the headline check is the one that cannot be fudged -- **replay each demo through
``scripts/eval/score.py::score_instruction`` and require 6.0/6**. That is the same scorer
the eval harness runs on our own executed trajectories, so a passing entry means "the
reference run, scored exactly as we will score ourselves, earns full marks". If a radius
is too tight, an order is wrong, or an avoid zone is drawn over the demo's own path, this
is what says so.

Seven checks per question:

1. every ``object_ids`` entry exists, and its ``objects`` payload still matches the box in
   the IRef-VLA metadata (nothing has drifted since the file was written);
2. the reference ``.ply`` scores 6.0/6 under ``score_instruction``;
3. ``traj_index`` increases across pass_near and lands at the end for the goal;
4. the demo enters none of the avoid polygons;
5. ``reference_trajectory`` still describes the ``.ply`` on disk;
6. every ``anchor_ids`` entry is carried in ``objects`` with ``role: reference``, every
   ``object_ids`` entry with ``role: constraint``, and no box is carried that no
   constraint references;
7. ``target_objects`` is exactly the nouns of the ``objects`` block, in its order -- the
   SAM prompt list cannot drift from the boxes it is supposed to cover.

Questions carrying a ``review_note`` are known deviations, recorded by hand with their
evidence. They are reported and excluded from the pass count rather than silently
tolerated or silently failed.

Usage::

    python3 scripts/eval/verify_category3.py
    python3 scripts/eval/verify_category3.py loft chinese_room -v
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from utils.geometry import answer_payload, load_objects, object_nouns  # noqa: E402

BENCH = REPO / "data" / "benchmark"
MAX_SCORE = 6.0
# The payloads are rounded to 4 decimals on the way out, so compare at that resolution.
TOL = 5e-4


def load_scorer():
    """Import scripts/eval/score.py by path -- it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("cat3_score", Path(__file__).parent / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ply(path: Path) -> np.ndarray:
    _, _, body = path.read_text().partition("end_header")
    return np.asarray([float(v) for v in body.split()], dtype=float).reshape(-1, 3)


def as_results(question: dict[str, Any], pts: np.ndarray) -> dict[str, Any]:
    """The demo dressed as a qa_recorder result, which is what score.py consumes.

    qa_recorder writes ``trajectory`` as (stamp, x, y) triples sampled from
    /state_estimation; score_instruction reads p[1] and p[2]. Feeding the reference run
    through that same shape is the point -- the GT is scored by the identical code path
    that will score a real run.
    """
    return {"trajectory": [[0.0, float(x), float(y)] for x, y, _ in pts]}


def as_gt(question: dict[str, Any]) -> dict[str, Any]:
    gt = question["gt"]
    return {
        "pass_near": [
            {"center": c["center"], "radius": c["radius"]} for c in gt.get("pass_near", [])
        ],
        "avoid": [{"polygon": z["polygon"]} for z in gt.get("avoid", [])],
        "goal": (
            {"center": gt["goal"]["center"], "radius": gt["goal"]["radius"]}
            if gt.get("goal")
            else None
        ),
    }


def check_question(
    scene: str, question: dict[str, Any], objects: dict[str, Any], scorer, verbose: bool
) -> list[str]:
    fails: list[str] = []
    qid = question["id"]
    gt = question["gt"]

    # 1. the boxes still say what the file says they say
    for oid, payload in question["objects"].items():
        if oid not in objects:
            fails.append(f"{qid}: object {oid} is not in the scene metadata")
            continue
        fresh = answer_payload(objects[oid])
        for field in ("center", "size", "aabb", "raw_label"):
            got, want = payload.get(field), fresh.get(field)
            if isinstance(want, str):
                if got != want:
                    fails.append(f"{qid}: object {oid} {field} is {got!r}, metadata says {want!r}")
            elif not close(got, want):
                fails.append(f"{qid}: object {oid} {field} drifted from the metadata")

    constraints = list(gt.get("pass_near", [])) + ([gt["goal"]] if gt.get("goal") else [])
    all_cited = constraints + list(gt.get("avoid", []))
    for c in constraints:
        unknown = [i for i in c["object_ids"] if i not in question["objects"]]
        if unknown:
            fails.append(f"{qid}: constraint {c['phrase']!r} cites unlisted objects {unknown}")

    # 6. constraint objects and reference objects agree with the boxes carried beside them
    fails += check_roles(qid, question, all_cited)

    # 7. the SAM prompt list is exactly the nouns of the boxes, in their order
    want = object_nouns([p["label"] for p in question["objects"].values()])
    if question["target_objects"] != want:
        fails.append(
            f"{qid}: target_objects {question['target_objects']} does not match the "
            f"objects block {want}"
        )

    # 5. the recording is the one the file describes
    ply = REPO / question["reference_trajectory"]["ply"]
    if not ply.exists():
        fails.append(f"{qid}: reference trajectory {ply} is missing")
        return fails
    pts = read_ply(ply)
    ref = question["reference_trajectory"]
    if len(pts) != ref["n_vertices"]:
        fails.append(f"{qid}: ply has {len(pts)} vertices, file says {ref['n_vertices']}")
    if not close(list(pts[-1][:2]), ref["end"], tol=1e-3):
        fails.append(f"{qid}: ply ends at {np.round(pts[-1][:2], 3).tolist()}, file says {ref['end']}")

    # 3. the order the file states is the order the run walks
    indices = [c["traj_index"] for c in gt.get("pass_near", [])]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        fails.append(f"{qid}: pass_near traj_index is not increasing: {indices}")
    if gt.get("goal") and gt["goal"]["traj_index"] != len(pts) - 1:
        fails.append(f"{qid}: goal traj_index {gt['goal']['traj_index']} is not the last vertex")

    # 4. no avoid zone is drawn over the demo's own path
    for zone in gt.get("avoid", []):
        hits = sum(1 for p in pts if scorer.point_in_poly((p[0], p[1]), zone["polygon"]))
        if hits:
            fails.append(f"{qid}: the demo puts {hits} vertices inside the avoid zone {zone['label']!r}")

    # 2. the headline check
    score, note = scorer.score_instruction(as_results(question, pts), as_gt(question))
    if abs(score - MAX_SCORE) > 1e-6:
        fails.append(f"{qid}: the reference run scores {score:.2f}/{MAX_SCORE} ({note})")
    elif verbose:
        print(f"    {qid} scores {score:.2f}/{MAX_SCORE}")
    return fails


def check_roles(qid: str, question: dict[str, Any], cited: list[dict[str, Any]]) -> list[str]:
    """The objects block and the constraints must name the same objects, with the same roles.

    A reference object is only worth carrying if some constraint says what it is doing
    there, and a constraint's landmark is only usable if its box is in the file. Checking
    both directions is what stops the two drifting apart when a clause is re-grounded.
    """
    fails: list[str] = []
    constraint_ids = {i for c in cited for i in c["object_ids"]}
    anchor_ids = {i for c in cited for i in c.get("anchor_ids", [])}

    for c in cited:
        unknown = [i for i in c.get("anchor_ids", []) if i not in question["objects"]]
        if unknown:
            fails.append(f"{qid}: constraint {c['phrase']!r} cites unlisted anchors {unknown}")

    for oid, payload in question["objects"].items():
        role = payload.get("role")
        want = "constraint" if oid in constraint_ids else "reference"
        if role != want:
            fails.append(f"{qid}: object {oid} has role {role!r}, but it is cited as a {want}")
        if oid not in constraint_ids and oid not in anchor_ids:
            fails.append(f"{qid}: object {oid} is carried but no constraint references it")
    return fails


def close(got: Any, want: Any, tol: float = TOL) -> bool:
    if isinstance(want, dict):
        return all(close(got.get(k), v, tol) for k, v in want.items()) if isinstance(got, dict) else False
    try:
        return bool(np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), atol=tol))
    except (TypeError, ValueError):
        return got == want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenes", nargs="*", help="limit to these scenes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    scorer = load_scorer()
    files = sorted(BENCH.glob("*/category_3/*_category3_qa.json"))
    if args.scenes:
        files = [f for f in files if f.parts[-3] in set(args.scenes)]
    if not files:
        print(f"no category-3 QA files under {BENCH}", file=sys.stderr)
        return 1

    n_ok = n_review = 0
    failures: list[str] = []
    for path in files:
        doc = json.loads(path.read_text())
        scene = doc["scene"]
        objects = load_objects(scene)
        print(f"[{scene}]")
        for question in doc["questions"]:
            fails = check_question(scene, question, objects, scorer, args.verbose)
            if question.get("review_note"):
                n_review += 1
                print(f"    {question['id']} REVIEW  {question['review_note'][:96]}...")
                for f in fails:
                    print(f"      known: {f}")
                continue
            if fails:
                failures.extend(f"{scene} {f}" for f in fails)
                for f in fails:
                    print(f"    FAIL  {f}")
            else:
                n_ok += 1

    total = sum(len(json.loads(p.read_text())["questions"]) for p in files)
    deviations = f"; {n_review} known deviation(s) recorded" if n_review else ""
    if failures:
        print(f"\n{n_ok}/{total} verified, {len(failures)} failure(s){deviations}")
        return 1
    print(
        f"\n{n_ok}/{total} verified, all scoring {MAX_SCORE}/{MAX_SCORE} "
        f"against their reference run{deviations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
