#!/usr/bin/env python3
"""Generate Category-1 (numerical) QA JSONs from IRef-VLA / bag metadata.

Default output:
  <repo>/data/benchmark/<scene>/category_1/<scene>_category1_qa.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BAGS = REPO / "bags"
QUESTIONS_JSON = REPO / "questions" / "questions.json"
DEFAULT_BENCHMARK = REPO / "data" / "benchmark"
DEFAULT_IREF_UNITY = Path("/home/ubuntu/myspace/SORT3D/data/IRef-VLA/Unity")

STRUCTURAL = {
    "wall",
    "floor",
    "ceiling",
    "door frame",
    "window frame",
    "unknown",
}
# Prefer not answering with these alone, but they can appear as anchors.
WEAK_ANSWER_CLASSES = STRUCTURAL | {"air vent", "light switch", "column"}
BAD_ON_ANCHORS = STRUCTURAL  # "on the floor/ceiling/wall" is noisy

COLORS = {
    "black",
    "gray",
    "grey",
    "white",
    "brown",
    "maroon",
    "olive",
    "pink",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "aqua",
    "beige",
    "tan",
    "purple",
    "cyan",
    "magenta",
    "gold",
    "silver",
}

ARTICLE = {"a", "an", "the"}

IRREGULAR_PLURALS = {
    "glass": "glasses",
    "eye glasses": "eye glasses",
    "glasses": "glasses",
    "books": "books",
    "windows": "windows",
    "papers": "papers",
    "pillow": "pillows",
    "pillows": "pillows",
    "flower": "flowers",
    "flowers": "flowers",
    "mouse": "mice",
    "computer mouse": "computer mice",
}


def pluralize(name: str) -> str:
    name = name.strip().lower()
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith("s"):
        return name  # already plural-looking
    if name.endswith(("x", "z", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def a_an(name: str) -> str:
    name = name.strip().lower()
    # If already plural, use bare noun ("with flowers on it")
    if name in IRREGULAR_PLURALS.values() or (
        name.endswith("s") and name not in {"glass", "bus", "iris"}
    ):
        return name
    return f"an {name}" if name[:1].lower() in "aeiou" else f"a {name}"


def object_targets(names: list[str]) -> list[str]:
    """SAM3 targets: object nouns only (no colors/articles)."""
    out = []
    seen = set()
    for n in names:
        t = (n or "").strip().lower()
        if not t or t in COLORS or t in ARTICLE:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def primary_color(obj: dict) -> str | None:
    colors = obj.get("color_labels") or []
    for c in colors:
        if c and c != "N/A":
            return c.lower()
    return None


def nice_label(obj: dict) -> str:
    """SAM-friendly display label: prefer specific raw_label."""
    raw = (obj.get("raw_label") or "").strip().lower()
    nyu = (obj.get("nyu_label") or "").strip().lower()
    if raw and raw not in ("unknown",):
        return raw
    return nyu or raw or "object"


def count_label(obj: dict) -> str:
    """Stable class label used for counting."""
    nyu = (obj.get("nyu_label") or "").strip().lower()
    raw = (obj.get("raw_label") or "").strip().lower()
    return nyu or raw


def resolve_scene_graph(scene: str, meta_roots: list[Path]) -> Path | None:
    """Find <scene>_scene_graph.json under known metadata layouts."""
    candidates = [
        # IRef-VLA Unity: <root>/<scene>/<scene>_scene_graph.json
        # bags layout: <root>/<scene>/iref_vla_metadata/<scene>_scene_graph.json
        # flat: <root>/<scene>_scene_graph.json
    ]
    for root in meta_roots:
        candidates.extend(
            [
                root / scene / f"{scene}_scene_graph.json",
                root / scene / "iref_vla_metadata" / f"{scene}_scene_graph.json",
                root / f"{scene}_scene_graph.json",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_scene(scene: str, meta_roots: list[Path] | None = None) -> dict[str, Any] | None:
    roots = meta_roots or [BAGS, DEFAULT_IREF_UNITY]
    sg_path = resolve_scene_graph(scene, roots)
    if not sg_path:
        return None
    sg = json.loads(sg_path.read_text())
    objects: dict[str, dict] = {}
    relationships: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    region_of: dict[str, str] = {}

    for rid, reg in (sg.get("regions") or {}).items():
        objs = reg.get("objects") or []
        if isinstance(objs, dict):
            objs = list(objs.values())
        for o in objs:
            if not isinstance(o, dict):
                continue
            oid = str(o.get("object_id"))
            objects[oid] = o
            region_of[oid] = str(rid)
        rels = reg.get("relationships") or {}
        for rtype, mapping in rels.items():
            if not isinstance(mapping, dict):
                continue
            for a, vs in mapping.items():
                if not isinstance(vs, list):
                    continue
                # between is list of pairs; keep as-is for later
                relationships[rtype][str(a)].extend(vs)

    return {
        "scene": scene,
        "objects": objects,
        "relationships": relationships,
        "region_of": region_of,
        "source_scene_graph": str(sg_path),
    }


def related(scene_data: dict, rtype: str, anchor: str, pred=None) -> list[str]:
    vs = scene_data["relationships"].get(rtype, {}).get(str(anchor), [])
    out = []
    for v in vs:
        # skip ternary between pairs when scanning binary relations
        if isinstance(v, list):
            continue
        vid = str(v)
        if vid not in scene_data["objects"]:
            continue
        if pred and not pred(vid):
            continue
        out.append(vid)
    return out


def unique_anchors(scene_data: dict, cls: str) -> list[str]:
    objs = scene_data["objects"]
    ids = [i for i, o in objs.items() if count_label(o) == cls or nice_label(o) == cls]
    return ids


def class_counts(scene_data: dict) -> Counter:
    return Counter(count_label(o) for o in scene_data["objects"].values())


def color_counts(scene_data: dict, cls: str) -> Counter:
    c = Counter()
    for o in scene_data["objects"].values():
        if count_label(o) != cls:
            continue
        col = primary_color(o)
        if col:
            c[col] += 1
    return c


def make_q(
    qid: str,
    question: str,
    answer: int,
    difficulty: str,
    qtype: str,
    target_objects: list[str],
    evidence: str,
    object_ids: list[str] | None = None,
) -> dict:
    return {
        "id": qid,
        "question": question,
        "answer": int(answer),
        "difficulty": difficulty,
        "type": qtype,
        "target_objects": object_targets(target_objects),
        "evidence": evidence,
        "object_ids": object_ids or [],
    }


def candidates_for_scene(scene_data: dict, official: list[str] | None) -> list[dict]:
    objs = scene_data["objects"]
    R = scene_data["relationships"]
    cands: list[dict] = []
    seen_questions: set[str] = set()

    def add(question, answer, difficulty, qtype, targets, evidence, ids=None, force=False):
        qn = " ".join(question.lower().split())
        if not force and qn in seen_questions:
            return
        if answer < 0 or answer > 40:
            return
        # Prefer non-trivial; allow 0 only rarely for hard negatives? skip 0.
        if answer == 0:
            return
        seen_questions.add(qn)
        clean_targets = object_targets(targets)
        cands.append(
            {
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "type": qtype,
                "target_objects": clean_targets,
                "evidence": evidence,
                "object_ids": [str(x) for x in (ids or [])],
                "coverage": set(clean_targets),
            }
        )

    # Official questions first (easy/medium depending on nesting)
    if official:
        for oq in official:
            # We cannot always verify official answer from graph; try to mine similarly later.
            # Still include as seed text if we can compute an answer with heuristics below.
            pass

    # --- EASY: attribute counts (group by nice_label for SAM consistency) ---
    by_nice: dict[str, list[str]] = defaultdict(list)
    for oid, o in objs.items():
        nl = nice_label(o)
        if count_label(o) in WEAK_ANSWER_CLASSES or nl in WEAK_ANSWER_CLASSES:
            continue
        by_nice[nl].append(oid)

    for sample, ids_all in by_nice.items():
        n = len(ids_all)
        if n < 2:
            continue
        # color splits using same nice_label set
        cc = Counter(primary_color(objs[i]) for i in ids_all if primary_color(objs[i]))
        for col, cn in cc.most_common():
            if cn < 2:
                continue
            if cn == n and n > 6:
                continue
            ids = [i for i in ids_all if primary_color(objs[i]) == col]
            add(
                f"How many {col} {pluralize(sample)} are in the scene?",
                cn,
                "easy",
                "attribute",
                [sample],
                f"{cn} {sample} with primary color {col}: {ids}",
                ids,
            )
        if 2 <= n <= 10:
            add(
                f"How many {pluralize(sample)} are in the scene?",
                n,
                "easy",
                "attribute",
                [sample],
                f"Count of {sample}: {ids_all}",
                ids_all,
            )

    # --- EASY/MEDIUM: on / near / above binary ---
    for rtype, difficulty_base in (("on", "easy"), ("near", "easy"), ("above", "medium"), ("below", "medium")):
        mapping = R.get(rtype, {})
        for anchor, _ in mapping.items():
            if anchor not in objs:
                continue
            a = objs[anchor]
            a_cls = count_label(a)
            a_name = nice_label(a)
            if a_cls in STRUCTURAL and rtype in ("near", "on"):
                # structural anchors are noisy
                continue
            if a_cls in BAD_ON_ANCHORS and rtype == "on":
                continue
            # group related by count_label
            rel_ids = related(scene_data, rtype, anchor)
            by_cls: dict[str, list[str]] = defaultdict(list)
            for vid in rel_ids:
                vcls = count_label(objs[vid])
                if vcls in STRUCTURAL:
                    continue
                by_cls[vcls].append(vid)
            for vcls, vids in by_cls.items():
                if not vids:
                    continue
                v_name = nice_label(objs[vids[0]])
                # Unique or rare anchors make clearer questions
                n_anchor_cls = sum(1 for o in objs.values() if count_label(o) == a_cls)
                uniq = n_anchor_cls == 1
                if rtype == "on":
                    q = f"How many {pluralize(v_name)} are on the {a_name}?"
                elif rtype == "near":
                    q = f"How many {pluralize(v_name)} are near the {a_name}?"
                elif rtype == "above":
                    q = f"How many {pluralize(v_name)} are above the {a_name}?"
                else:
                    q = f"How many {pluralize(v_name)} are below the {a_name}?"
                diff = difficulty_base if uniq else "medium"
                # color filter variant
                add(
                    q,
                    len(vids),
                    diff,
                    "relational",
                    [v_name, a_name],
                    f"{rtype}[{anchor}:{a_name}] -> {vids}",
                    vids + [anchor],
                )
                # colored answer objects
                col_groups = defaultdict(list)
                for vid in vids:
                    col = primary_color(objs[vid])
                    if col:
                        col_groups[col].append(vid)
                for col, cvids in col_groups.items():
                    if len(cvids) == len(vids) and len(vids) > 2:
                        continue
                    if rtype == "on":
                        cq = f"How many {col} {pluralize(v_name)} are on the {a_name}?"
                    elif rtype == "near":
                        cq = f"How many {col} {pluralize(v_name)} are near the {a_name}?"
                    elif rtype == "above":
                        cq = f"How many {col} {pluralize(v_name)} are above the {a_name}?"
                    else:
                        cq = f"How many {col} {pluralize(v_name)} are below the {a_name}?"
                    add(
                        cq,
                        len(cvids),
                        "medium",
                        "relational+attribute",
                        [v_name, a_name],
                        f"{rtype}[{anchor}] colored {col}: {cvids}",
                        cvids + [anchor],
                    )
                # colored unique anchor
                a_col = primary_color(a)
                if a_col and uniq:
                    if rtype == "on":
                        aq = f"How many {pluralize(v_name)} are on the {a_col} {a_name}?"
                    elif rtype == "near":
                        aq = f"How many {pluralize(v_name)} are near the {a_col} {a_name}?"
                    else:
                        aq = f"How many {pluralize(v_name)} are {rtype} the {a_col} {a_name}?"
                    add(
                        aq,
                        len(vids),
                        "medium",
                        "relational+attribute",
                        [v_name, a_name],
                        f"{rtype}[{anchor} {a_col} {a_name}] -> {vids}",
                        vids + [anchor],
                    )

    # --- HARD: nested "X near/on the Y with a Z on it" ---
    on_map = R.get("on", {})
    near_map = R.get("near", {})
    for table_id, on_vs in on_map.items():
        if table_id not in objs:
            continue
        table = objs[table_id]
        t_name = nice_label(table)
        t_cls = count_label(table)
        if t_cls in STRUCTURAL:
            continue
        on_ids = [str(v) for v in on_vs if not isinstance(v, list) and str(v) in objs]
        on_ids = [v for v in on_ids if count_label(objs[v]) not in STRUCTURAL]
        if not on_ids:
            continue
        # Pick distinctive toppers
        for top_id in on_ids:
            top = objs[top_id]
            top_name = nice_label(top)
            top_cls = count_label(top)
            # How many X are near the Y with a Z on it?
            near_ids = related(scene_data, "near", table_id)
            by_cls = defaultdict(list)
            for vid in near_ids:
                vcls = count_label(objs[vid])
                if vcls in STRUCTURAL or vcls == t_cls or vcls == top_cls:
                    continue
                by_cls[vcls].append(vid)
            for vcls, vids in by_cls.items():
                if len(vids) < 1:
                    continue
                v_name = nice_label(objs[vids[0]])
                q = (
                    f"How many {pluralize(v_name)} are near the {t_name} "
                    f"with {a_an(top_name)} on it?"
                )
                add(
                    q,
                    len(vids),
                    "hard",
                    "relational",
                    [v_name, t_name, top_name],
                    f"near[{table_id}:{t_name} with {top_id}:{top_name} on] -> {vids}",
                    vids + [table_id, top_id],
                )
                # color on counted set
                col_groups = defaultdict(list)
                for vid in vids:
                    col = primary_color(objs[vid])
                    if col:
                        col_groups[col].append(vid)
                for col, cvids in col_groups.items():
                    if len(cvids) == len(vids) and len(vids) > 3:
                        continue
                    cq = (
                        f"How many {col} {pluralize(v_name)} are near the {t_name} "
                        f"with {a_an(top_name)} on it?"
                    )
                    add(
                        cq,
                        len(cvids),
                        "hard",
                        "relational+attribute",
                        [v_name, t_name, top_name],
                        f"colored {col} near nested: {cvids}",
                        cvids + [table_id, top_id],
                    )

            # How many Z are on the Y near the X? (anchor via near)
            for other_id in related(scene_data, "near", table_id):
                other = objs[other_id]
                o_cls = count_label(other)
                if o_cls in STRUCTURAL:
                    continue
                # only if other is somewhat unique
                n_o = sum(1 for o in objs.values() if count_label(o) == o_cls)
                if n_o > 3:
                    continue
                o_name = nice_label(other)
                q2 = (
                    f"How many {pluralize(top_name)} are on the {t_name} "
                    f"near the {o_name}?"
                )
                # count toppers of that class on this table
                tops_same = [v for v in on_ids if count_label(objs[v]) == top_cls]
                add(
                    q2,
                    len(tops_same),
                    "hard",
                    "relational",
                    [top_name, t_name, o_name],
                    f"on[{table_id}] near {other_id}:{o_name} tops={tops_same}",
                    tops_same + [table_id, other_id],
                )

    # --- HARD: above nested with color ---
    for anchor, vs in R.get("above", {}).items():
        if anchor not in objs:
            continue
        a = objs[anchor]
        if count_label(a) in STRUCTURAL:
            continue
        a_name = nice_label(a)
        rel_ids = related(scene_data, "above", anchor)
        by_cls = defaultdict(list)
        for vid in rel_ids:
            vcls = count_label(objs[vid])
            if vcls in STRUCTURAL:
                continue
            by_cls[vcls].append(vid)
        for vcls, vids in by_cls.items():
            v_name = nice_label(objs[vids[0]])
            # require something on the anchor for nesting
            on_ids = related(scene_data, "on", anchor)
            on_ids = [v for v in on_ids if count_label(objs[v]) not in STRUCTURAL]
            if not on_ids:
                continue
            top = objs[on_ids[0]]
            top_name = nice_label(top)
            q = (
                f"How many {pluralize(v_name)} are above the {a_name} "
                f"with {a_an(top_name)} on it?"
            )
            add(
                q,
                len(vids),
                "hard",
                "relational",
                [v_name, a_name, top_name],
                f"above[{anchor}] with on={on_ids[0]} -> {vids}",
                vids + [anchor, on_ids[0]],
            )

    # Try to attach official questions if we can match a candidate or compute
    if official:
        for oq in official:
            # If an identical/similar candidate exists, boost it; else try parse lightly
            oq_norm = " ".join(oq.lower().split())
            matched = False
            for c in cands:
                if c["question"].lower() == oq_norm:
                    c["difficulty"] = c.get("difficulty", "medium")
                    c["official"] = True
                    matched = True
                    break
            if matched:
                continue
            # Heuristic: leave official as medium if we find a cand with same answer pattern later;
            # attempt simple "How many X are on the Y" recomputation via tokens
            # Fall through: add official only when we find exact graph support among cands by fuzzy containment
            for c in cands:
                # same noun skeleton
                if oq_norm.replace("count the number of", "how many") == c["question"].lower():
                    c["official"] = True
                    matched = True
                    break

    return cands


def template_key(question: str) -> str:
    """Normalize question to a coarse template for diversity."""
    q = question.lower()
    q = re.sub(
        r"\b(black|gray|grey|white|brown|maroon|olive|pink|red|blue|green|yellow|orange|aqua|beige|tan|purple)\b",
        "COLOR",
        q,
    )
    # collapse object phrases coarsely: keep relation skeleton
    q = re.sub(r"how many |are |in the scene\??|the |a |an ", "", q)
    parts = q.split(" with ")
    head = parts[0]
    # keep relation word + counted noun family
    tokens = head.split()
    rel = next((t for t in tokens if t in {"on", "near", "above", "below"}), "attr")
    counted = tokens[0] if tokens else "x"
    return f"{rel}:{counted}:{len(parts)>1}"


def select_diverse(cands: list[dict], n: int = 30) -> list[dict]:
    """Select n questions with difficulty mix and wide target coverage."""

    def score(c: dict) -> tuple:
        ans = c["answer"]
        ans_pen = abs(ans - 3)
        if ans > 12:
            ans_pen += 8
        cov = len(object_targets(list(c["coverage"])))
        official_bonus = -100 if c.get("official") else 0
        return (official_bonus, ans_pen - cov, c["question"])

    for c in cands:
        c["coverage"] = set(object_targets(list(c["coverage"])))
        c["target_objects"] = object_targets(c["target_objects"])
        c["template"] = template_key(c["question"])

    by_diff = {"easy": [], "medium": [], "hard": []}
    for c in cands:
        by_diff.setdefault(c["difficulty"], []).append(c)
    for d in by_diff:
        by_diff[d].sort(key=score)

    quotas = {"easy": 10, "medium": 12, "hard": 8}
    selected: list[dict] = []
    used_q = set()
    covered: Counter = Counter()
    template_counts: Counter = Counter()

    def pick_from(pool: list[dict], k: int):
        picked = 0
        for prefer_novel in (True, False):
            for c in pool:
                if picked >= k:
                    return
                qn = c["question"].lower()
                if qn in used_q:
                    continue
                if template_counts[c["template"]] >= 2 and prefer_novel:
                    continue
                novel = sum(1 for t in c["coverage"] if covered[t] == 0)
                if prefer_novel and novel == 0 and covered:
                    continue
                if c["target_objects"]:
                    main = c["target_objects"][0]
                    if covered[main] >= 3 and prefer_novel:
                        continue
                selected.append(c)
                used_q.add(qn)
                template_counts[c["template"]] += 1
                for t in c["coverage"]:
                    covered[t] += 1
                picked += 1

    for diff, k in quotas.items():
        pick_from(by_diff.get(diff, []), k)

    if len(selected) < n:
        rest = [c for c in sorted(cands, key=score) if c["question"].lower() not in used_q]
        pick_from(rest, n - len(selected))

    if len(selected) < n:
        for c in sorted(cands, key=score):
            if len(selected) >= n:
                break
            qn = c["question"].lower()
            if qn in used_q:
                continue
            selected.append(c)
            used_q.add(qn)

    order = {"easy": 0, "medium": 1, "hard": 2}
    selected = selected[:n]
    selected.sort(key=lambda c: (order.get(c["difficulty"], 9), c["question"]))
    return selected


def load_official() -> dict[str, list[str]]:
    if not QUESTIONS_JSON.exists():
        return {}
    data = json.loads(QUESTIONS_JSON.read_text())
    out = {}
    for item in data:
        scene = item.get("scene")
        nums = (item.get("questions") or {}).get("numerical") or []
        if scene:
            out[scene] = nums
    return out


def generate_scenes(
    scenes: list[str],
    *,
    n_per_scene: int = 30,
    meta_roots: list[Path] | None = None,
    out_root: Path = DEFAULT_BENCHMARK,
) -> None:
    official = load_official()
    roots = meta_roots or [DEFAULT_IREF_UNITY, BAGS]
    summary = []
    for scene in scenes:
        data = load_scene(scene, roots)
        if not data:
            summary.append((scene, "skipped (no scene graph)", 0))
            continue
        cands = candidates_for_scene(data, official.get(scene))
        chosen = select_diverse(cands, n_per_scene)
        questions = []
        for i, c in enumerate(chosen, 1):
            questions.append(
                make_q(
                    qid=f"Q{i:02d}",
                    question=c["question"],
                    answer=c["answer"],
                    difficulty=c["difficulty"],
                    qtype=c["type"],
                    target_objects=c["target_objects"],
                    evidence=c["evidence"],
                    object_ids=c.get("object_ids"),
                )
            )
        hist = Counter(q["difficulty"] for q in questions)
        targets_all = sorted({t for q in questions for t in q["target_objects"]})
        out = {
            "scene": scene,
            "category": 1,
            "category_name": "numerical",
            "description": (
                "Category 1 (numerical) evaluation questions with integer answers. "
                "Grounded in scene-graph relations and object color attributes. "
                "target_objects lists SAM3-relevant object nouns (no color adjectives) "
                "needed to ground each question."
            ),
            "notes": [
                "difficulty: easy = attribute or single relation; medium = relation+attribute or non-unique anchors; hard = nested relations (e.g. near X with Y on it).",
                "raw_label is preferred for SAM prompts when more specific than nyu_label.",
                "Structural classes (wall/floor/ceiling) are excluded from counted answers and noisy on/near anchors.",
                "object_ids lists supporting scene-graph ids for auditing; SAM3 should use target_objects.",
                f"source_scene_graph: {data.get('source_scene_graph')}",
            ],
            "difficulty_counts": dict(hist),
            "target_object_coverage": targets_all,
            "questions": questions,
        }
        out_dir = out_root / scene / "category_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene}_category1_qa.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        summary.append(
            (scene, str(out_path), len(questions), dict(hist), len(targets_all), len(cands))
        )

    print("Generation summary:")
    for row in summary:
        print(row)


def discover_scenes(meta_roots: list[Path]) -> list[str]:
    scenes: set[str] = set()
    for root in meta_roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if resolve_scene_graph(name, [root]):
                scenes.add(name)
    return sorted(scenes)


def generate_all(n_per_scene: int = 30) -> None:
    roots = [DEFAULT_IREF_UNITY, BAGS]
    generate_scenes(discover_scenes(roots), n_per_scene=n_per_scene, meta_roots=roots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names to generate (default: all discoverable, or remaining missing if --missing-only)",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only generate scenes not already present under the benchmark output root",
    )
    parser.add_argument(
        "--iref-root",
        type=Path,
        default=DEFAULT_IREF_UNITY,
        help="IRef-VLA Unity root containing per-scene folders",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Benchmark output root (default: data/benchmark)",
    )
    parser.add_argument("--n", type=int, default=30, help="Questions per scene")
    args = parser.parse_args()

    meta_roots = [args.iref_root, BAGS]
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = discover_scenes(meta_roots)
        if args.missing_only:
            existing = {
                p.name
                for p in args.out_root.iterdir()
                if p.is_dir() and (p / "category_1").exists()
            }
            scenes = [s for s in scenes if s not in existing]

    generate_scenes(
        scenes,
        n_per_scene=args.n,
        meta_roots=meta_roots,
        out_root=args.out_root,
    )
