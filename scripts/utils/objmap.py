#!/usr/bin/env python3
"""Read the robot's own 3D map as the boxes the benchmark solver already understands.

`obj_map.json` is what map_node writes beside a question's best-view crops: one entry per
tracked instance, keyed by its track id, holding a label and an axis-aligned `bbox3d`. The
category-2 answer *is* one of those entries -- the marker we publish is its box -- so
choosing an answer is choosing a key.

Everything that knows what "closest to", "between" or "on" means already lives in
`utils.geometry`, written against IRef-VLA's `Obj`. Rather than a second implementation of
those predicates over a second box format, this adapts a map entry into the same `Obj`: the
benchmark verifier and the answer path then agree on the geometry by construction, and a
predicate that is too loose shows up in both at once instead of only in the one nobody ran.

Two differences from an IRef object survive the adaptation and matter downstream:

  * No colour palette. The mapper stores no appearance, so a colour constraint cannot be
    resolved here at all and has to come from the crops. Only 2 of the 122 category-2
    questions need one.
  * Mangled labels. map_node names a class with `detections.default_label(prompt)`, which
    strips the spaces out of the prompt ("potted plant" -> "pottedplant"), and `norm_class`
    cannot fold a spelling it has never seen. The prompts that armed SAM are recorded in the
    run's manifest, so the mangling is inverted from those rather than guessed at.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

import utils.text_solver as solver
from utils.geometry import (
    BAD_LANDMARKS,
    MIN_MARGIN,
    Obj,
    between_holds,
    center_dist,
    color_matches,
    footprints_overlap,
    gap_dist,
    is_above,
    is_inside,
    rests_on,
    set_dist,
    vertical_gap,
)

__all__ = [
    "aabb_iou",
    "candidate_table",
    "is_decisive",
    "load_obj_map",
    "same_class",
    "shortlist",
]

# Identity quaternion, which is what map_node writes unless publish_oriented_box is on.
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

# One region, because a live map has no region annotation: IRef's regions come from the
# dataset's floorplan, and nothing in the pipeline reconstructs them. Every helper in
# utils.geometry that takes `region_objs` therefore gets the whole map.
MAP_REGION = "0"


def quat_to_yaw(quat: Sequence[float]) -> float:
    """Heading about +z from an xyzw quaternion. Mirrors sam_mapper.challenge_marker."""
    x, y, z, w = (float(v) for v in quat)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def box_corners(center: Sequence[float], extent: Sequence[float], yaw: float) -> np.ndarray:
    """The 8 corners of a gravity-aligned box, in the order `Obj` expects.

    Two conventions have to be honoured at once, both of them load-bearing:

      * corners 0-3 are the top face, which is what `scripts/eval/score_map3d.py` reads as
        a rotated footprint straight out of IRef without recovering a heading; and
      * the 0->1 edge runs along the box's local +x, because `Obj.yaw` recovers the heading
        from exactly that edge, and a CUBE marker built from a wrong yaw is a wrong answer.
    """
    cx, cy, cz = (float(v) for v in center)
    dx, dy, dz = (abs(float(v)) / 2.0 for v in extent)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # Local XY, starting at (-x, -y) so that corner 1 is one +x step away from corner 0.
    local = ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy))
    out = []
    for z in (cz + dz, cz - dz):
        for lx, ly in local:
            out.append((cx + lx * cos_y - ly * sin_y, cy + lx * sin_y + ly * cos_y, z))
    return np.asarray(out, dtype=float)


def prompt_labels(prompts: Iterable[str]) -> dict[str, str]:
    """Mangled map label -> the prompt it came from ("pottedplant" -> "potted plant")."""
    out: dict[str, str] = {}
    for prompt in prompts or ():
        phrase = " ".join(str(prompt or "").lower().split())
        if phrase:
            out.setdefault(phrase.replace(" ", ""), phrase)
    return out


def map_entry_to_obj(track_id: str, entry: dict[str, Any], labels: dict[str, str]) -> Obj | None:
    """One `obj_map.json` entry as an `Obj`, or None when it has no usable box.

    `bbox3d.center` rather than the sibling `center` key: those are different points -- the
    latter is an observation-weighted voxel centroid -- and the box we claim has to be
    described by its own midpoint. `challenge_marker.payload_from_map_object` makes the same
    choice for the same reason.
    """
    bbox = entry.get("bbox3d") or {}
    center, extent = bbox.get("center"), bbox.get("extent")
    if not center or not extent or len(center) != 3 or len(extent) != 3:
        return None
    if not all(float(v) > 0.0 for v in extent):
        return None

    raw = str(entry.get("label") or "").strip().lower()
    yaw = quat_to_yaw(bbox.get("rotation") or IDENTITY_QUAT)
    return Obj(
        str(track_id),
        MAP_REGION,
        {
            "bbox": box_corners(center, extent, yaw),
            "center": [float(v) for v in center],
            "size": [abs(float(v)) for v in extent],
            # Both label slots get the un-mangled prompt: the mapper knows one name per
            # class, so there is no raw/NYU distinction to preserve, and leaving nyu_label
            # empty would make `Obj.coarse` fall back to the raw label anyway.
            "raw_label": labels.get(raw, raw),
            "nyu_label": labels.get(raw, raw),
            "color_labels": [],
        },
    )


def load_obj_map(path: str | Path, prompts: Iterable[str] = ()) -> dict[str, Obj]:
    """Load a map written beside a question's crops, keyed by track id as a string.

    Entries the mapper could not place are dropped rather than kept as degenerate boxes: a
    zero-extent candidate can never win an IoU and would only pad the table we show a model.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    labels = prompt_labels(prompts)
    out: dict[str, Obj] = {}
    for track_id, entry in (raw or {}).items():
        if isinstance(entry, dict) and (obj := map_entry_to_obj(track_id, entry, labels)):
            out[str(track_id)] = obj
    return out


# ---------------------------------------------------------------- scoring


def aabb_iou(a: Obj, b: Obj) -> float:
    """IoU of the two boxes' world-axis-aligned envelopes.

    This is the number the challenge scorer computes: `scripts/eval/score.py` reads a
    marker's `position` and `scale` and ignores its orientation, so both boxes are compared
    as centre +- size/2 whatever their true heading. Oriented IoU is the more flattering
    metric and is reported separately by the bench, never instead of this one.
    """
    lo = np.maximum(a.center - a.size / 2.0, b.center - b.size / 2.0)
    hi = np.minimum(a.center + a.size / 2.0, b.center + b.size / 2.0)
    span = hi - lo
    if np.any(span <= 0.0):
        return 0.0
    inter = float(np.prod(span))
    union = float(np.prod(a.size)) + float(np.prod(b.size)) - inter
    return float(inter / union) if union > 0.0 else 0.0


def same_class(pred: Obj, gt: Obj) -> bool:
    """Could this prediction be describing that GT object at all.

    Compared after `norm_class`, so the synonym folding is the one the benchmark already
    uses; the prediction's label has to have been un-mangled first or nothing matches.
    """
    return pred.fine in (gt.fine, gt.coarse) or pred.coarse in (gt.fine, gt.coarse)


# ---------------------------------------------------------------- candidates


def relation_score(cand: Obj, groups: Sequence[Sequence[Obj]], relation: str) -> float:
    """How *nearly* a candidate satisfies a relation. Lower is better.

    Strictly an ordering, never a verdict: `relation_holds` decides whether a relation is
    true, and this only decides who to try first when it is true of nobody — which happens
    often, because the predicates are tuned to be sure rather than to be generous (a lamp
    120 degrees off the axis between two anchors is not "between" them, but if no lamp
    qualifies then the nearest thing to between is the best answer available, and it beats
    whichever candidate happened to be listed first).

    Returns infinity when the relation has no ordering to offer, which leaves the caller's
    existing order untouched rather than imposing a meaningless one.
    """
    anchors = [a for group in groups for a in group]
    if not anchors:
        return math.inf

    if relation == "between" and len(groups) >= 2:
        best = math.inf
        for a1 in groups[0]:
            for a2 in groups[1]:
                if a1.id != a2.id:
                    best = min(best, between_holds(cand, a1, a2)[1])
        return best
    if relation in ("closest", "near"):
        return set_dist(cand, anchors, gap_dist if relation == "near" else center_dist)
    if relation == "farthest":
        return -set_dist(cand, anchors, center_dist)
    if relation in ("on", "above", "below", "in", "supports"):
        # Vertical relations are decided by a gap plus an overlap test, so order by the gap
        # and push candidates that do not overlap at all behind every candidate that does.
        flip = relation in ("below", "supports")
        gaps = []
        for anchor in anchors:
            lower, upper = (cand, anchor) if not flip else (anchor, cand)
            penalty = 0.0 if footprints_overlap(lower, upper) else 100.0
            gaps.append(abs(vertical_gap(lower, upper)) + penalty)
        return min(gaps) if gaps else math.inf
    return math.inf


def shortlist(text: str, objects: Sequence[Obj]) -> dict[str, Any]:
    """Rank a map's objects by how well they answer a category-2 question.

    `text_solver.solve` is deliberately unwilling to guess: a tie inside `MIN_MARGIN`, an
    unresolvable anchor or two objects that both satisfy the relation all return no target,
    which is right for verifying a benchmark and wrong for answering one. Publishing nothing
    and publishing the wrong box both score zero, so this keeps the ordering the solver
    computed and commits to its head regardless, recording whether the solver would have
    stood behind that choice.

    Room-scale structure is dropped from the *candidates* only. Anchors are resolved against
    every object, because half the corpus is anchored on exactly the things pruning removes
    -- "between a door frame and a window", "the chair between the fireplace and the stairs"
    -- and pruning those away leaves the relation unjudgeable and the question lost.

    Returns `{candidates, anchors, anchor_groups, all_anchors, anchors_unmatched, head,
    relation, committed, reason, trace}` -- `candidates` best-first and never empty unless
    nothing matches the head noun at all. `anchors` and `anchor_groups` are the FIRST hop's,
    which is what the ranking and `relation_holds` are about; `all_anchors` is every hop's,
    deduplicated, which is what "not the answer" is about.
    """
    objs = list(objects)
    targets = [o for o in objs if not prunable(o)]
    spec = solver.parse(text)
    trace: list[str] = [f"head={spec['head']!r} color={spec['color']!r} hops={spec['hops']}"]

    hops = spec["hops"]
    relation = hops[0]["relation"] if hops else ""
    groups = [solver.match_class(p, objs) for p in hops[0]["phrases"]] if hops else []
    anchors = [o for group in groups for o in group]

    # Every noun the question names other than the head, across ALL hops rather than only the
    # first. `groups` stops at hop 0 because that is the hop the ranking filters on, but "the
    # bowl on the table closest to the folding screen" names the screen in its second hop and
    # the screen is no more answerable than the table is -- one of the measured losses is
    # exactly that answer. Dict rather than list: a phrase repeated across hops is one anchor.
    matched = {p: solver.match_class(p, objs) for hop in hops for p in hop["phrases"]}
    seen: set[str] = set()
    all_anchors: list[Obj] = []
    for anchor in (a for group in matched.values() for a in group):
        if anchor.id not in seen:
            seen.add(anchor.id)
            all_anchors.append(anchor)
    # Which named anchors the map does not hold. Recorded rather than inferred from an empty
    # group later, because only here are the phrase and its matches still side by side -- and
    # a reasoner that is not told an anchor is missing ranks on a relation that collapsed.
    unmatched = [phrase for phrase, group in matched.items() if not group]
    if unmatched:
        trace.append(f"anchors named but not in the map: {unmatched}")

    # Resolved before the head noun is looked for, not after, because the no-candidate
    # return below is the one the caller answers by falling back -- and a fallback that is
    # not told which objects are anchors answers with one of them.
    heads = solver.match_class(spec["head"], targets)
    if spec["color"]:
        # The mapper stores no colours, so this only ever narrows a GT-sourced run. Kept so
        # both sources go through one code path rather than two that drift.
        if colored := [o for o in heads if color_matches(spec["color"], o.colors)]:
            heads = colored
            trace.append(f"colour {spec['color']}: {len(heads)} candidate(s)")
    if not heads:
        return {
            "candidates": [], "anchors": anchors, "anchor_groups": groups,
            "all_anchors": all_anchors, "anchors_unmatched": unmatched,
            "head": spec["head"], "relation": relation,
            "committed": None, "reason": f"nothing in the map matches {spec['head']!r}",
            "trace": trace,
        }
    trace.append(f"head candidates: {heads[:8]}")

    ranked = heads
    if hops and anchors:
        # For the comparatives this returns every head in ranked order; for the predicates
        # only those that hold.
        kept, note = solver.matches(heads, groups, relation)
        trace.append(f"target hop ({relation}): {note}")
        ranked = kept or heads
        if not kept:
            # Nobody satisfies it, so order by how close each comes instead of leaving the
            # arbitrary order the objects were loaded in.
            ranked = sorted(heads, key=lambda o: relation_score(o, groups, relation))
            trace.append(f"no head satisfies {relation}: ranked by proximity to it instead")
        elif len(kept) > 1 and relation not in ("closest", "farthest", "near"):
            # `matches` returns predicate survivors unordered, and the caller needs a first
            # choice. The comparatives are already ranked and must not be resorted.
            ranked = sorted(kept, key=lambda o: relation_score(o, groups, relation))
            trace.append(f"{len(kept)} candidates satisfy {relation}: ranked by margin")
    elif hops:
        trace.append(f"target hop ({relation}): no anchor matches {hops[0]['phrases']}")

    # The full solver runs second, over the same objects, purely to find out whether it
    # would commit. When it does, its pick leads -- it collapses multi-hop anchor chains
    # ("closest to the book on the stool") that the single hop above ignores.
    verdict = solver.solve(text, objs)
    committed = verdict.get("target")
    if committed is not None and prunable(committed):
        trace.append(f"solver committed to {committed}, which is structural — ignored")
        committed = None
    if committed is not None:
        ranked = [committed] + [o for o in ranked if o.id != committed.id]
        trace.append(f"solver committed: {committed} ({verdict.get('reason')})")
    else:
        trace.append(f"solver declined to commit: {verdict.get('reason')}")

    return {
        "candidates": ranked,
        "anchors": anchors,
        "anchor_groups": groups,
        "all_anchors": all_anchors,
        "anchors_unmatched": unmatched,
        "head": spec["head"],
        "relation": relation,
        "committed": committed,
        "reason": str(verdict.get("reason") or ""),
        "trace": trace + [f"solver trace: {t}" for t in verdict.get("trace", [])],
    }


def facts_for(cand: Obj, groups: Sequence[Sequence[Obj]]) -> dict[str, Any]:
    """Every number about one candidate a model would otherwise have to derive.

    Arithmetic over coordinates is the documented weak spot of the whole family of methods
    that hand a language model a list of boxes (Transcrib3D gives its LLM a Python
    interpreter for exactly this). Distances, vertical gaps and the support predicates are
    cheap and exact here, so the model is never asked to compute one -- it is asked to
    choose among them.
    """
    anchors = [a for group in groups for a in group]
    per_anchor = []
    for anchor in anchors:
        per_anchor.append({
            "anchor": f"{anchor.display}#{anchor.id}",
            "centre_m": round(center_dist(cand, anchor), 2),
            "gap_m": round(gap_dist(cand, anchor), 2),
            "vertical_gap_m": round(vertical_gap(cand, anchor), 2),
            "overlaps_footprint": bool(footprints_overlap(cand, anchor)),
            "on": bool(rests_on(cand, anchor)),
            "supports": bool(rests_on(anchor, cand)),
            "inside": bool(is_inside(cand, anchor)),
            "above": bool(is_above(cand, anchor)),
            "below": bool(is_above(anchor, cand)),
        })

    facts: dict[str, Any] = {
        "id": cand.id,
        "label": cand.display,
        "centre": [round(float(v), 2) for v in cand.center],
        "size": [round(float(v), 2) for v in cand.size],
        "anchors": per_anchor,
    }

    # `between` has to pair across the two phrase groups, never inside one: a lamp between
    # two windows is not an answer to "between a door frame and a window".
    if len(groups) >= 2:
        best: tuple[bool, float, str] | None = None
        for a1 in groups[0]:
            for a2 in groups[1]:
                if a1.id == a2.id:
                    continue
                holds, lateral = between_holds(cand, a1, a2)
                pair = f"{a1.display}#{a1.id}+{a2.display}#{a2.id}"
                if best is None or (holds, -lateral) > (best[0], -best[1]):
                    best = (holds, lateral, pair)
        if best is not None:
            facts["between"] = {
                "holds": best[0],
                "lateral_offset_m": round(best[1], 2) if math.isfinite(best[1]) else None,
                "pair": best[2],
            }
    return facts


def candidate_table(
    candidates: Sequence[Obj],
    groups: Sequence[Sequence[Obj]],
    relation: str = "",
    limit: int = 12,
    head: str = "",
    anchors: Optional[Sequence[Obj]] = None,
    unmatched_anchors: Sequence[str] = (),
) -> str:
    """The object list as text: what is answerable, what is only a reference, and the numbers.

    Capped, because a table long enough to hold a whole scene buries the handful of objects
    the question is actually about -- the pruning every recent zero-shot grounding method
    reports as its largest single gain. The cap applies to an already-ranked list, so what
    gets dropped is what the geometry likes least.

    Three things are stated that the flat version left the reader to infer, each one a
    measured loss:

      * the head noun, because an answer has to be an instance of it;
      * the anchors, in their own section, because their ids appear in every candidate's
        fact rows and with no role marker three answers came back naming one of them;
      * the anchors the question named that the map does not hold, because the alternative
        is a reasoner silently ranking on a relation that collapsed.

    Units are stated once here rather than suffixed to each of the four numbers on every
    row, which on a full table is the difference between reading a sentence and reading a
    unit conversion.

    `anchors` defaults to the first hop's, flattened out of `groups`. Pass every hop's when
    the caller has them: an anchor from a deeper hop ("closest to the folding screen") has no
    fact row of its own, so this section is the only place its id is explained at all.
    """
    kept = list(candidates)[: max(1, limit)]
    lines: list[str] = []
    if head:
        lines.append(f"The answer is a {head!r}.")
    if relation:
        lines.append(f"Relation asked for: {relation}")
    lines.append("All lengths are in metres. centre and size are (x, y, z).")

    lines.append("")
    lines.append("CANDIDATES — the answer is exactly one of these:" if kept else
                 f"CANDIDATES: none — the map holds no {head or 'matching object'}.")
    for cand in kept:
        facts = facts_for(cand, groups)
        lines.append(f"[{facts['id']}] {facts['label']}  centre={tuple(facts['centre'])}  "
                     f"size={tuple(facts['size'])}")
        for row in facts["anchors"]:
            flags = [name for name in ("on", "supports", "inside", "above", "below")
                     if row[name]]
            lines.append(
                f"    to {row['anchor']}: centre {row['centre_m']}, box gap "
                f"{row['gap_m']}, vertical gap {row['vertical_gap_m']}"
                + (f", holds: {', '.join(flags)}" if flags else "")
            )
        if between := facts.get("between"):
            lines.append(
                f"    between {between['pair']}: "
                f"{'yes' if between['holds'] else 'no'}"
                + (f", lateral offset {between['lateral_offset_m']}"
                   if between["lateral_offset_m"] is not None else "")
            )
    dropped = max(0, len(candidates) - max(1, limit))
    if dropped:
        lines.append(f"({dropped} further candidate(s) ranked lower, omitted)")

    # Deduplicated: "between a window and a window frame" can match one object from both
    # phrases, and listing it twice under two roles is worse than not listing it.
    seen: set[str] = set()
    anchor_lines: list[str] = []
    for anchor in (a for group in groups for a in group) if anchors is None else anchors:
        if anchor.id in seen:
            continue
        seen.add(anchor.id)
        anchor_lines.append(
            f"[{anchor.id}] {anchor.display}  "
            f"centre={tuple(round(float(v), 2) for v in anchor.center)}  "
            f"size={tuple(round(float(v), 2) for v in anchor.size)}")
    if anchor_lines:
        lines.append("")
        lines.append("ANCHORS — reference objects the question names to say which candidate "
                     "it means. Never the answer:")
        lines.extend(anchor_lines)

    if unmatched_anchors:
        lines.append("")
        lines.append("NOT DETECTED — named by the question, absent from the map: "
                     + ", ".join(repr(str(p)) for p in unmatched_anchors))
    return "\n".join(lines)


def prunable(obj: Obj) -> bool:
    """Objects that cannot be an answer: room-scale structure and known bad landmarks.

    The same exclusions the benchmark generator applies when deciding a question is fair to
    ask, so nothing is pruned here that could have been the answer there.
    """
    return obj.structural or obj.fine in BAD_LANDMARKS


def is_decisive(candidates: Sequence[Obj], groups: Sequence[Sequence[Obj]], relation: str) -> bool:
    """Did the geometry settle this on its own, or is a second opinion worth paying for?

    Measured as the distance between the top candidate and the runner-up against
    `MIN_MARGIN`, which is the benchmark's own threshold for "a person could tell these
    apart". A sole candidate is decisive; a tie inside the margin is not.
    """
    anchors = [a for group in groups for a in group]
    if len(candidates) < 2 or not anchors or relation not in ("closest", "farthest", "near"):
        return len(candidates) == 1
    metric = gap_dist if relation == "near" else center_dist
    scores = sorted(
        (set_dist(c, anchors, metric) for c in candidates[:2]),
        reverse=relation == "farthest",
    )
    return abs(scores[0] - scores[-1]) >= MIN_MARGIN
