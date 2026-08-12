#!/usr/bin/env python3
"""Generate Category-2 (object reference) QA JSONs from IRef-VLA metadata.

Category 2 asks the robot to *point at one object* — the answer is a 3D box, not a count
— so the ground truth is a join that already exists on disk:

    <scene>_referential_statements.json   utterance -> target_index, relation, anchors
    <scene>_objects.json                  object_id -> 8-corner bbox, centre, size

The join alone is not enough. IRef-VLA emits every statement its grammar allows, including
ones no human could resolve ("the pillow closest to the plant", with three plants in the
room), so each candidate is re-derived from the geometry in ``cat2_geometry`` and kept only
when its answer is unique by a margin under both distance metrics and both class
granularities. The two official questions per scene come from the PDF text verbatim and
are resolved by ``cat2_text_solver`` rather than by the join, because the organizers'
phrasing usually has no generated counterpart.

Default output:
  <repo>/data/benchmark/<scene>/category_2/<scene>_category2_qa.json

Usage::

    python3 bags/generate_category2_qa.py                        # all scenes with statements
    python3 bags/generate_category2_qa.py --scenes arabic_room -v
    python3 bags/generate_category2_qa.py --n 10 --out-root data/benchmark
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cat2_text_solver as solver
from cat2_geometry import (
    BAGS,
    MIN_MARGIN,
    REPO,
    SCENE_NAME_RE,
    Obj,
    answer_payload,
    between_holds,
    center_dist,
    load_objects,
    load_statements,
    metadata_path,
    norm_class,
    object_nouns,
    relation_holds,
    unaskable,
    verify,
)

QUESTIONS_JSON = REPO / "questions" / "questions.json"
PDF_ASSETS = REPO / "data" / "pdf_assets"
DEFAULT_BENCHMARK = REPO / "data" / "benchmark"
DEFAULT_OVERRIDES = BAGS / "category2_overrides.json"
DEFAULT_CANDIDATES = REPO / "data" / "runs" / "cat2_candidates"

TEMPLATES = {
    "closest": "Find the {target} that {be} closest to {anchor}.",
    "farthest": "Find the {target} that {be} farthest from {anchor}.",
    "near": "Find the {target} that {be} near {anchor}.",
    "below": "Find the {target} that {be} below {anchor}.",
    "above": "Find the {target} that {be} above {anchor}.",
    "on": "Find the {target} on top of {anchor}.",
    "in": "Find the {target} inside {anchor}.",
    "supports": "Find the {target} with {anchor} on it.",
    "between": "Find the {target} that {be} between {anchor} and {anchor2}.",
}

COMPARATIVE = {"closest", "farthest", "near"}

# Relations derived from the boxes rather than mined from an utterance. IRef's grammar
# emits 16330 "near" statements and 155 "on" ones, which is a property of the generator
# that wrote them, not of the scenes: every room is full of objects resting on, over,
# under and between other objects. Synthesising these directly is what keeps the benchmark
# from being nine tenths distance comparisons.
SYNTHESIZED = ("on", "supports", "in", "above", "below", "between")

# How many of each relation a scene should get before any relation gets a second helping.
# Ordered scarcest-first: whatever is hard to come by picks its target while the pool is
# still untouched, and the distance comparisons — of which there are always hundreds — fill
# what is left. Comparatives still get the largest quota, since the official questions are
# overwhelmingly "closest to".
QUOTAS = (
    ("between", 1),
    ("in", 1),
    ("supports", 1),
    ("above", 1),
    ("below", 1),
    ("on", 2),
    ("near", 1),
    ("closest", 2),
    ("farthest", 2),
)

# IRef labels some single objects with a plural noun (a stack of books is "books", a
# window bank is "windows"), which needs "that are", not "that is".
SINGULAR_S_WORDS = {"glass", "grass", "dress", "compass", "mattress", "bass", "canvas"}

QUESTION_KEYS = [
    "id", "question", "source", "difficulty", "relation", "target_objects", "answer",
    "anchors", "distractor_ids", "competitors", "margin", "evidence", "statement",
    "region", "images", "verified", "solver_trace", "reworded", "review_note",
]


# ---------------------------------------------------------------- phrasing


def anchor_phrase(anchor: Obj, region_objs: list[Obj]) -> str | None:
    """A noun phrase that picks out exactly this anchor, or None if none does.

    An ambiguous anchor makes the whole question ambiguous however clean the target
    geometry is, so those candidates are dropped rather than patched. Colour is not used
    to rescue one: the palette comes from clustering mesh vertex colours, so "the purple
    potted plant" can disagree with what a person sees.
    """
    twins = [
        o
        for o in region_objs
        if o.id != anchor.id and (o.fine == anchor.fine or o.coarse == anchor.coarse)
    ]
    return f"the {anchor.display}" if not twins else None


def selection_score(candidate: dict) -> float:
    """Ranking for selection: how unmistakable the answer is.

    Comparative questions are ranked by their distance margin. Predicates have no margin,
    so they get a fixed score that lands them mid-pack — the diversity pass gives each
    relation a slot anyway, and beyond that a wide margin is worth more than another
    "on the cabinet".
    """
    return float(candidate.get("margin", 1.0))


def plural_label(label: str) -> bool:
    head = label.split()[-1] if label else ""
    return head.endswith("s") and head not in SINGULAR_S_WORDS


def difficulty_of(relation: str, n_rivals: int) -> str:
    if n_rivals >= 4 or relation == "between":
        return "hard"
    if n_rivals >= 2 or relation in ("above", "below", "in", "supports"):
        return "medium"
    return "easy"


# ---------------------------------------------------------------- candidate building


def build_candidate(
    target: Obj,
    anchors: list[Obj],
    phrases: list[str],
    relation: str,
    margin: float,
    why: str,
    n_rivals: int,
    derivation: str,
    statement: str = "",
    distractor_ids: list[str] | None = None,
) -> dict:
    candidate = {
        "question": TEMPLATES[relation].format(
            target=target.display,
            be="are" if plural_label(target.display) else "is",
            anchor=phrases[0],
            anchor2=phrases[1] if len(phrases) > 1 else "",
        ),
        "source": "generated",
        "derivation": derivation,
        "difficulty": difficulty_of(relation, n_rivals),
        "relation": relation,
        "target_objects": object_nouns([target.display] + [a.display for a in anchors]),
        "answer": answer_payload(target),
        "anchors": [
            {"object_id": a.id, "label": a.display, "phrase": p}
            for a, p in zip(anchors, phrases)
        ],
        "distractor_ids": sorted(set(distractor_ids or []), key=str),
        "competitors": n_rivals,
        "evidence": f"{relation}: {why}",
        "statement": statement,
        "region": target.region,
        "verified": True,
    }
    # Only the comparative relations have a distance margin; for the others the useful
    # number is how many same-class rivals the relation ruled out.
    if relation in COMPARATIVE:
        candidate["margin"] = round(float(margin), 3)
    return candidate


def synthesize_candidates(
    objects: dict[str, Obj], rejected: Counter
) -> dict[tuple[str, str, str], dict]:
    """Predicate candidates derived from the boxes, with no utterance behind them.

    The comparative relations are not synthesised: IRef already emits thousands of them,
    and every pair of same-class objects generates one, so the pool would drown in
    "farthest from" questions that only differ in which corner of the room they name.
    """
    out: dict[tuple[str, str, str], dict] = {}
    for region in sorted({o.region for o in objects.values()}):
        region_objs = [o for o in objects.values() if o.region == region]
        named = [(a, p) for a in region_objs if (p := anchor_phrase(a, region_objs))]
        for target in region_objs:
            if target.structural:
                continue
            for relation in SYNTHESIZED:
                if relation == "between":
                    continue
                for anchor, phrase in named:
                    if anchor.id == target.id:
                        continue
                    if reason := unaskable(target, [anchor], relation, region_objs):
                        rejected[reason] += 1
                        continue
                    # Cheap predicate first: verify() walks the whole region, and all but a
                    # handful of (target, anchor) pairs fail on the geometry alone.
                    if not relation_holds(target, [anchor], relation):
                        continue
                    ok, margin, why, n_rivals = verify(target, [anchor], relation, region_objs)
                    if not ok:
                        rejected[f"{relation}: answer not unique"] += 1
                        continue
                    if n_rivals < 1:
                        rejected[f"{relation}: relation not needed to disambiguate"] += 1
                        continue
                    out[(target.id, relation, anchor.id)] = build_candidate(
                        target, [anchor], [phrase], relation, margin, why, n_rivals, "geometry"
                    )
            out.update(synthesize_between(target, named, region_objs, rejected))
    return out


def synthesize_between(
    target: Obj, named: list[tuple[Obj, str]], region_objs: list[Obj], rejected: Counter
) -> dict[tuple[str, str, str], dict]:
    """The best few "between A and B" claims about one target.

    Anchor pairs are quadratic and a room yields hundreds of true betweenness claims, most
    of them technically correct and useless. Only the tightest are kept: the target closest
    to the line, relative to how far apart the anchors are.
    """
    if target.structural:
        return {}
    found: list[tuple[float, tuple[str, str, str], dict]] = []
    for i, (a1, p1) in enumerate(named):
        for a2, p2 in named[i + 1:]:
            if target.id in (a1.id, a2.id):
                continue
            if reason := unaskable(target, [a1, a2], "between", region_objs):
                rejected[reason] += 1
                continue
            holds, lateral = between_holds(target, a1, a2)
            if not holds:
                continue
            ok, margin, why, n_rivals = verify(target, [a1, a2], "between", region_objs)
            if not ok:
                rejected["between: answer not unique"] += 1
                continue
            if n_rivals < 1:
                rejected["between: relation not needed to disambiguate"] += 1
                continue
            span = center_dist(a1, a2)
            found.append((
                lateral / max(span, 1e-6),
                (target.id, "between", "|".join([a1.id, a2.id])),
                build_candidate(
                    target, [a1, a2], [p1, p2], "between", margin, why, n_rivals, "geometry"
                ),
            ))
    found.sort(key=lambda item: item[0])
    return {key: candidate for _score, key, candidate in found[:2]}


def mine_candidates(
    objects: dict[str, Obj], statements: dict[str, Any]
) -> tuple[list[dict], dict[str, int]]:
    """Every claim that survives geometric verification, from utterances and from boxes."""
    by_region: dict[str, list[Obj]] = defaultdict(list)
    for obj in objects.values():
        by_region[obj.region].append(obj)

    rejected: Counter = Counter()
    # Synthesised first, so a statement-backed duplicate of the same claim overwrites it
    # and keeps its utterance and distractor list.
    best: dict[tuple[str, str, str], dict] = synthesize_candidates(objects, rejected)

    for region, utterances in (statements.get("regions") or {}).items():
        region_objs = by_region.get(str(region), [])
        for utterance, entries in utterances.items():
            for entry in entries:
                relation = str(entry.get("relation") or "")
                if relation not in TEMPLATES:
                    rejected[f"relation not used: {relation}"] += 1
                    continue
                target = objects.get(str(entry.get("target_index")))
                if target is None:
                    rejected["target missing from objects.json"] += 1
                    continue
                if target.structural:
                    rejected["structural target"] += 1
                    continue
                anchor_ids = [str(a.get("index")) for a in (entry.get("anchors") or {}).values()]
                anchors = [objects[i] for i in anchor_ids if i in objects]
                if not anchors or len(anchors) != len(anchor_ids):
                    rejected["anchor missing from objects.json"] += 1
                    continue

                phrases = [anchor_phrase(a, region_objs) for a in anchors]
                if any(p is None for p in phrases):
                    rejected["ambiguous anchor"] += 1
                    continue

                if reason := unaskable(target, anchors, relation, region_objs):
                    rejected[reason] += 1
                    continue

                ok, margin, why, n_rivals = verify(target, anchors, relation, region_objs)
                if not ok:
                    rejected[f"{relation}: answer not unique"] += 1
                    continue

                # With no same-class competitor the relation is decoration: "the shower
                # tap on the shower" is answered by "the shower tap".
                if n_rivals < 1:
                    rejected[f"{relation}: relation not needed to disambiguate"] += 1
                    continue

                key = (target.id, relation, "|".join(a.id for a in anchors))
                candidate = build_candidate(
                    target,
                    anchors,
                    phrases,
                    relation,
                    margin,
                    why,
                    n_rivals,
                    "statement",
                    statement=utterance,
                    distractor_ids=[str(i) for i in (entry.get("distractor_ids") or [])],
                )
                # The same claim arrives under several phrasings ("below" / "under" /
                # "beneath"); keep one, preferring the shortest utterance. A synthesised
                # entry has no statement at all, so it always loses to a real one.
                previous = best.get(key)
                if previous is None or len(utterance) < len(previous["statement"] or "z" * 999):
                    best[key] = candidate

    candidates = sorted(
        best.values(),
        key=lambda c: (-selection_score(c), c["answer"]["object_id"], c["question"]),
    )
    return candidates, dict(rejected.most_common())


# ---------------------------------------------------------------- official questions


def official_questions() -> dict[str, list[str]]:
    if not QUESTIONS_JSON.exists():
        return {}
    data = json.loads(QUESTIONS_JSON.read_text())
    return {
        item["scene"]: (item.get("questions") or {}).get("object_reference") or []
        for item in data
        if item.get("scene")
    }


def official_images(scene: str) -> dict[str, list[str]]:
    """Question text -> extracted questions.pdf screenshots, when the assets exist."""
    index = PDF_ASSETS / scene / "pdf_text.json"
    if not index.exists():
        return {}
    data = json.loads(index.read_text())
    return {
        solver.normalize(q.get("question", "")): [
            f"data/pdf_assets/{scene}/{name}" for name in q.get("images") or []
        ]
        for q in data.get("questions", [])
        if q.get("category") == 2
    }


def matching_statement(
    target_id: str, relation: str, text: str, statements: dict[str, Any]
) -> tuple[str, list[str]]:
    """The IRef utterance for this target, for provenance and its distractor list."""
    wanted = solver.normalize(text)
    best: tuple[float, str, list[str]] = (0.0, "", [])
    for utterances in (statements.get("regions") or {}).values():
        for utterance, entries in utterances.items():
            for entry in entries:
                if str(entry.get("target_index")) != target_id:
                    continue
                score = difflib.SequenceMatcher(None, utterance, wanted).ratio()
                if entry.get("relation") == relation:
                    score += 0.3
                if score > best[0]:
                    best = (
                        score,
                        utterance,
                        sorted({str(i) for i in (entry.get("distractor_ids") or [])}, key=str),
                    )
    return best[1], best[2]


def build_official(
    text: str,
    objects: dict[str, Obj],
    statements: dict[str, Any],
    images: dict[str, list[str]],
    pinned_id: str | None,
) -> dict:
    result = solver.solve_in_regions(text, objects)
    target = objects.get(pinned_id) if pinned_id else result["target"]
    spec = solver.parse(text)
    # The solver only reports a relation when it resolved the question; a pinned answer
    # still has one, and it is right there in the text.
    relation = result.get("relation") or (spec["hops"][0]["relation"] if spec["hops"] else "")
    anchors = [
        {"object_id": a.id, "label": a.display} for a in (result.get("anchors") or [])
    ]

    question: dict[str, Any] = {
        "question": solver.normalize(text).capitalize() + ".",
        "source": "official",
        # Nesting is what makes the official questions harder than anything generated
        # here: a second hop is a second object to pin down before the target is reachable.
        "difficulty": "hard" if len(spec["hops"]) > 1 else "medium",
        "relation": relation,
        "anchors": anchors,
        "solver_trace": result["trace"] + [f"result: {result['reason']}"],
    }

    mentioned = solver.mentioned_labels(text, list(objects.values()))
    if target is None:
        question["answer"] = None
        question["target_objects"] = object_nouns(mentioned or [spec["head"]])
        question["distractor_ids"] = []
        question["statement"] = ""
        question["region"] = ""
        question["verified"] = False
        question["evidence"] = f"unresolved: {result['reason']}"
    else:
        statement, distractors = matching_statement(target.id, relation, text, statements)
        question["answer"] = answer_payload(target)
        question["target_objects"] = object_nouns(
            [target.display] + [a["label"] for a in anchors] + mentioned
        )
        question["distractor_ids"] = distractors
        question["statement"] = statement
        question["region"] = target.region
        question["verified"] = True
        question["evidence"] = (
            f"target pinned by override to {target}"
            if pinned_id
            else f"solved from geometry: {result['reason']}"
        )
    if imgs := images.get(solver.normalize(text)):
        question["images"] = imgs
    return question


# ---------------------------------------------------------------- selection


def select(candidates: list[dict], n: int, taken_relations: Counter) -> list[dict]:
    """Fill the remaining slots by relation quota, scarcest relation first.

    Ranking candidates purely by how unmistakable they are produces ten distance
    comparisons, because a "farthest from" question across a room wins every margin contest
    a support relation can enter. So each relation gets a quota and spends it in QUOTAS
    order, and only then does a free-for-all pass top the scene up to n. The official
    questions' relations count against the quotas, so a scene whose two official questions
    are both "closest" does not then pick two more.

    Passes relax the class and anchor caps for scenes too thin to fill ten slots. Two rules
    never relax — one question per target object, and one per (target class, anchor) pair,
    since "closest to X" and "farthest from X" over the same class each give away the
    other's answer.
    """
    picked: list[int] = []
    used_targets: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    used_anchors: Counter = Counter()
    by_relation = Counter(taken_relations)
    by_class: Counter = Counter()

    def take(index: int, per_class: int, per_anchor: int) -> bool:
        candidate = candidates[index]
        target_id = candidate["answer"]["object_id"]
        cls = norm_class(candidate["answer"]["label"])
        anchor_ids = "|".join(a["object_id"] for a in candidate["anchors"])
        if target_id in used_targets or (cls, anchor_ids) in used_pairs:
            return False
        if by_class[cls] >= per_class:
            return False
        if any(used_anchors[a["object_id"]] >= per_anchor for a in candidate["anchors"]):
            return False
        picked.append(index)
        used_targets.add(target_id)
        used_pairs.add((cls, anchor_ids))
        by_relation[candidate["relation"]] += 1
        by_class[cls] += 1
        for a in candidate["anchors"]:
            used_anchors[a["object_id"]] += 1
        return True

    # Quotas first at their nominal size, then raised a step at a time, so a scene that
    # cannot fill ten slots from its scarce relations widens the common ones evenly instead
    # of handing every leftover slot to whichever comparative had the biggest margin.
    for extra, per_class, per_anchor in ((0, 2, 2), (1, 2, 2), (2, 3, 3), (n, n, n)):
        for relation, quota in QUOTAS:
            for index, candidate in enumerate(candidates):
                if len(picked) >= n:
                    return [candidates[i] for i in picked]
                if by_relation[relation] >= quota + extra:
                    break
                if candidate["relation"] == relation and index not in picked:
                    take(index, per_class, per_anchor)
    return [candidates[i] for i in picked]


# ---------------------------------------------------------------- overrides


def scene_overrides(overrides: dict[str, Any], scene: str) -> dict[str, Any]:
    rules = overrides.get(scene) or {}
    return {
        "drop": {solver.normalize(t) for t in rules.get("drop") or []},
        "reword": {solver.normalize(k): v for k, v in (rules.get("reword") or {}).items()},
        "pin": {solver.normalize(k): str(v) for k, v in (rules.get("pin") or {}).items()},
        "note": {solver.normalize(k): v for k, v in (rules.get("note") or {}).items()},
    }


def apply_text_rules(questions: list[dict], rules: dict[str, Any]) -> list[dict]:
    """Reword and annotate after selection; drops are applied before it, so the
    generator can backfill and still land on n questions."""
    for q in questions:
        key = solver.normalize(q["question"])
        if new_text := rules["reword"].get(key):
            q["question"] = new_text
            q["reworded"] = True
        if note := rules["note"].get(key):
            q["review_note"] = note
    return questions


# ---------------------------------------------------------------- generation


def generate_scene(
    scene: str,
    n_per_scene: int,
    out_root: Path,
    overrides: dict[str, Any],
    candidates_dir: Path | None,
    verbose: bool,
) -> dict[str, Any]:
    objects = load_objects(scene)
    statements = load_statements(scene)
    rules = scene_overrides(overrides, scene)
    candidates, rejected = mine_candidates(objects, statements)
    images = official_images(scene)

    questions: list[dict] = []
    for text in official_questions().get(scene, []):
        if solver.normalize(text) in rules["drop"]:
            continue
        questions.append(
            build_official(text, objects, statements, images, rules["pin"].get(solver.normalize(text)))
        )

    taken = {q["answer"]["object_id"] for q in questions if q.get("answer")}
    pool = [
        c
        for c in candidates
        if c["answer"]["object_id"] not in taken
        and solver.normalize(c["question"]) not in rules["drop"]
    ]
    official_relations = Counter(q["relation"] for q in questions if q.get("relation"))
    questions += select(pool, max(0, n_per_scene - len(questions)), official_relations)
    questions = apply_text_rules(questions, rules)

    for i, q in enumerate(questions, start=1):
        q["id"] = f"Q{i:02d}"
    questions = [
        {k: q[k] for k in QUESTION_KEYS if k in q}
        | {k: v for k, v in q.items() if k not in QUESTION_KEYS}
        for q in questions
    ]

    out = {
        "scene": scene,
        "category": 2,
        "category_name": "object_reference",
        "description": (
            "Category 2 (object reference) evaluation questions. Each question names exactly "
            "one object; the answer is that object's 3D bounding box and centre, taken from "
            "the IRef-VLA scene metadata. target_objects lists SAM3-relevant object nouns "
            "(no color adjectives) needed to ground each question."
        ),
        "notes": [
            "answer.center + answer.size + answer.yaw are the oriented box; answer.aabb + "
            "answer.size_aabb are the axis-aligned equivalent for scorers that ignore "
            "orientation. Answer with a CUBE marker, never the RViz wireframe.",
            "source=official questions are verbatim from questions/<scene>/questions.pdf; "
            "their target object is solved from the geometry by bags/cat2_text_solver.py and "
            "solver_trace records how. images points at the screenshot the PDF shows next to "
            "the question, in which the expected answer is outlined.",
            "source=generated questions are mined from the scene's referential statements and "
            f"re-derived from the boxes: the answer wins by at least {MIN_MARGIN} m under both "
            "centre-to-centre and box-gap distance, against both raw-label and NYU-label "
            "competitors, and every anchor is uniquely named in its region.",
            "difficulty: easy = one same-class competitor and a uniquely named anchor; medium "
            "= 2-3 competitors, a vertical relation, or a single-hop official question; hard = "
            "4+ competitors, a between relation, or a multi-hop official question.",
            f"source_statements: {metadata_path(scene, 'referential_statements.json').relative_to(REPO)}",
            f"source_objects: {metadata_path(scene, 'objects.json').relative_to(REPO)}",
        ],
        "difficulty_counts": dict(Counter(q["difficulty"] for q in questions)),
        "relation_counts": dict(Counter(q["relation"] for q in questions if q.get("relation"))),
        "target_object_coverage": sorted({t for q in questions for t in q["target_objects"]}),
        "questions": questions,
    }

    out_dir = out_root / scene / "category_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scene}_category2_qa.json").write_text(json.dumps(out, indent=2) + "\n")

    if candidates_dir is not None:
        candidates_dir.mkdir(parents=True, exist_ok=True)
        (candidates_dir / f"{scene}.json").write_text(
            json.dumps(
                {
                    "scene": scene,
                    "verified_candidates": len(candidates),
                    "rejected": rejected,
                    "candidates": [
                        {k: v for k, v in c.items() if k != "answer"}
                        | {
                            "answer_object_id": c["answer"]["object_id"],
                            "answer_label": c["answer"]["label"],
                            "answer_center": c["answer"]["center"],
                        }
                        for c in candidates
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    unresolved = [q["id"] for q in questions if not q.get("verified")]
    print(
        f"{scene:16s} questions={len(questions):2d} candidates={len(candidates):4d} "
        f"relations={out['relation_counts']} difficulty={out['difficulty_counts']}"
        + (f"  UNRESOLVED: {unresolved}" if unresolved else "")
    )
    if verbose:
        for q in questions:
            answer = q.get("answer")
            where = f"{answer['label']}#{answer['object_id']}" if answer else "unresolved"
            print(f"    {q['id']} [{q['source']:9s}] {q['question']}")
            print(f"        -> {where}: {q['evidence']}")
    return out


def discover_scenes() -> list[str]:
    return sorted(
        p.name
        for p in BAGS.iterdir()
        if p.is_dir()
        and SCENE_NAME_RE.match(p.name)
        and (p / "iref_vla_metadata" / f"{p.name}_referential_statements.json").exists()
        and (p / "iref_vla_metadata" / f"{p.name}_objects.json").exists()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", nargs="*", default=None, help="Scenes (default: all with statements)")
    ap.add_argument("--n", type=int, default=10, help="Questions per scene")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark output root")
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES, help="Review overrides JSON")
    ap.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Where to dump every verified candidate for auditing ('' to skip)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Print each selected question")
    args = ap.parse_args()

    scenes = args.scenes or discover_scenes()
    if not scenes:
        print("no scenes with referential statements found under bags/")
        return 1

    overrides: dict[str, Any] = {}
    if args.overrides and args.overrides.exists():
        overrides = {
            k: v for k, v in json.loads(args.overrides.read_text()).items() if not k.startswith("_")
        }

    candidates_dir = args.candidates_dir if str(args.candidates_dir) else None
    total = unresolved = short = 0
    for scene in scenes:
        out = generate_scene(scene, args.n, args.out_root, overrides, candidates_dir, args.verbose)
        total += len(out["questions"])
        unresolved += sum(1 for q in out["questions"] if not q.get("verified"))
        short += 1 if len(out["questions"]) < args.n else 0

    print(f"\n{total} questions across {len(scenes)} scene(s) -> {args.out_root}")
    if short:
        print(f"{short} scene(s) produced fewer than {args.n} questions")
    if unresolved:
        print(f"{unresolved} question(s) unresolved — pin their target in {args.overrides}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
