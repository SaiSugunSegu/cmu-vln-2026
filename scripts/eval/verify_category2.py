#!/usr/bin/env python3
"""Audit the category-2 (object reference) benchmark against the IRef-VLA metadata.

``bags/generate_category2_qa.py`` writes the QA files; this re-reads the scene metadata
from scratch and checks that what they claim still holds, so a stale QA file, a hand edit,
or a metadata refresh cannot quietly ship a wrong ground-truth box.

Per question::

    answer.object_id exists in <scene>_objects.json
    every geometry field (centre, size, yaw, aabb, corners, volume) matches the metadata
    the answer is not a structural class nobody can point at
    anchors exist, and none of them is the answer
    the stated relation still holds and still picks out this object uniquely
    official questions still resolve to this object from their text alone
    ids are unique, and the scene has the expected number of questions

Usage::

    python3 scripts/eval/verify_category2.py                 # every scene
    python3 scripts/eval/verify_category2.py arabic_room -v
    python3 scripts/eval/verify_category2.py --expect 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "bags"))

import cat2_text_solver as solver  # noqa: E402
from cat2_geometry import (  # noqa: E402
    STRUCTURAL,
    Obj,
    answer_payload,
    load_objects,
    load_statements,
    region_objects,
    unaskable,
    verify,
)

BENCH_ROOT = REPO / "data" / "benchmark"
OVERRIDES = REPO / "bags" / "category2_overrides.json"

COMPARATIVE = {"closest", "farthest", "near"}


def qa_files(scenes: list[str] | None) -> list[Path]:
    found = sorted(BENCH_ROOT.glob("*/category_2/*_category2_qa.json"))
    if scenes:
        wanted = set(scenes)
        found = [p for p in found if p.parts[-3] in wanted]
    return found


def pinned_targets(scene: str) -> dict[str, str]:
    """Official questions whose answer was set by hand during the review pass."""
    if not OVERRIDES.exists():
        return {}
    rules = json.loads(OVERRIDES.read_text()).get(scene) or {}
    return {solver.normalize(k): str(v) for k, v in (rules.get("pin") or {}).items()}


def reworded(scene: str) -> dict[str, str]:
    """New text -> original text, so a reworded question can still be re-solved."""
    if not OVERRIDES.exists():
        return {}
    rules = json.loads(OVERRIDES.read_text()).get(scene) or {}
    return {solver.normalize(v): k for k, v in (rules.get("reword") or {}).items()}


def geometry_problems(answer: dict[str, Any], obj: Obj) -> list[str]:
    """Recompute the stored box from the metadata and diff it field by field."""
    expected = answer_payload(obj)
    out = []
    for field, want in expected.items():
        got = answer.get(field)
        if field == "aabb":
            if got != want:
                out.append(f"aabb {got} != metadata {want}")
            continue
        if got != want:
            out.append(f"{field} {got} != metadata {want}")
    return out


def check_question(
    q: dict[str, Any],
    objects: dict[str, Obj],
    statements: dict[str, Any],
    pins: dict[str, str],
    rewords: dict[str, str],
) -> list[str]:
    problems: list[str] = []
    answer = q.get("answer")
    if not answer:
        return ["no answer"]

    target = objects.get(str(answer.get("object_id")))
    if target is None:
        return [f"answer.object_id {answer.get('object_id')!r} is not in the scene metadata"]
    if target.structural:
        problems.append(f"answer {target} is a structural class ({sorted(STRUCTURAL)[:3]}...)")
    problems += geometry_problems(answer, target)

    if not q.get("target_objects"):
        problems.append("target_objects is empty, so the question cannot be prompted")
    if not q.get("verified"):
        problems.append("question is flagged unverified")

    anchor_ids = [str(a.get("object_id")) for a in q.get("anchors") or []]
    anchors = [objects[i] for i in anchor_ids if i in objects]
    if len(anchors) != len(anchor_ids):
        problems.append(f"anchors {anchor_ids} include ids missing from the metadata")
    if target.id in anchor_ids:
        problems.append("the answer is also listed as one of its own anchors")
    for oid in q.get("distractor_ids") or []:
        if str(oid) not in objects:
            problems.append(f"distractor id {oid!r} is not in the metadata")

    relation = q.get("relation") or ""
    scene_objs = region_objects(objects, target.region)
    if q.get("source") == "official":
        # Re-solve from the text, which is the only ground truth the organizers gave us.
        text = rewords.get(solver.normalize(q["question"]), q["question"])
        pin = pins.get(solver.normalize(text))
        if pin:
            if pin != target.id:
                problems.append(f"override pins {pin} but the answer is {target.id}")
        else:
            result = solver.solve_in_regions(text, objects)
            got = result["target"]
            if got is None:
                problems.append(f"no longer resolves from its text ({result['reason']}) and is not pinned")
            elif got.id != target.id:
                problems.append(f"text now resolves to {got} instead of {target}")
    elif relation:
        if not anchors:
            problems.append(f"relation {relation!r} with no usable anchor")
        else:
            # True but unaskable is still a bad question, and the rules that decide which
            # is which live next to the geometry so both sides apply the same ones.
            if reason := unaskable(target, anchors, relation, scene_objs):
                problems.append(f"nobody would ask this: {reason}")
            ok, margin, why, n_rivals = verify(target, anchors, relation, scene_objs)
            if not ok:
                problems.append(f"relation {relation} no longer holds uniquely: {why}")
            if q.get("competitors") != n_rivals:
                problems.append(f"competitors {q.get('competitors')} != recomputed {n_rivals}")
            stated = q.get("margin")
            if relation in COMPARATIVE:
                if stated is None:
                    problems.append(f"{relation} question has no margin")
                elif abs(float(stated) - margin) > 0.011:
                    problems.append(f"margin {stated} != recomputed {margin:.3f}")
            elif stated is not None:
                problems.append(f"{relation} question should not carry a distance margin")
    else:
        problems.append("no relation recorded")

    return problems


def verify_scene(path: Path, expect: int, verbose: bool) -> tuple[int, int]:
    qa = json.loads(path.read_text())
    scene = qa["scene"]
    questions = qa.get("questions") or []
    objects = load_objects(scene)
    statements = load_statements(scene)
    pins, rewords = pinned_targets(scene), reworded(scene)

    n_ok = n_bad = 0
    ids = [q.get("id") for q in questions]
    scene_problems = []
    if len(set(ids)) != len(ids):
        scene_problems.append(f"duplicate question ids: {ids}")
    if expect and len(questions) != expect:
        scene_problems.append(f"{len(questions)} questions, expected {expect}")
    if qa.get("category") != 2 or qa.get("category_name") != "object_reference":
        scene_problems.append("envelope is not category 2 / object_reference")

    for q in questions:
        problems = check_question(q, objects, statements, pins, rewords)
        if problems:
            n_bad += 1
            print(f"[{scene}] {q.get('id')} FAIL  {q.get('question')}")
            for problem in problems:
                print(f"    - {problem}")
        else:
            n_ok += 1
            if verbose:
                answer = q["answer"]
                print(
                    f"[{scene}] {q['id']} OK  {answer['label']}#{answer['object_id']}  "
                    f"{q.get('relation')}  {q.get('question')}"
                )

    for problem in scene_problems:
        print(f"[{scene}] SCENE FAIL - {problem}")
    status = "OK" if not (n_bad or scene_problems) else "FAIL"
    print(f"[{scene}] {status}: {n_ok} verified, {n_bad} failed")
    return n_ok, n_bad + len(scene_problems)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenes", nargs="*", help="Scene names to check (default: all)")
    ap.add_argument("--expect", type=int, default=10, help="Questions per scene (0 to skip)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print passing questions too")
    args = ap.parse_args()

    files = qa_files(args.scenes or None)
    if not files:
        print(f"no category-2 QA files under {BENCH_ROOT}")
        return 1

    total_ok = total_bad = 0
    for path in files:
        ok, bad = verify_scene(path, args.expect, args.verbose)
        total_ok += ok
        total_bad += bad
        print()

    print(f"TOTAL: {total_ok} verified, {total_bad} failed across {len(files)} scene(s)")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
