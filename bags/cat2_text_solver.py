#!/usr/bin/env python3
"""Resolve an object-reference question to the object it names, from the boxes.

The organizers publish the official category-2 questions as text only
(``questions/<scene>/questions.pdf``), so their answer object has to be recovered.
Matching the text against IRef-VLA's generated utterances is not enough on its own:
"the wall lamp that is between a door frame and a window" has no generated counterpart
in ``arabic_room``, and the nearest string is a *farthest-from* statement about a
different lamp. So this parses the question instead and solves it against the geometry.

Every official question in ``questions/questions.json`` has the same shape — a head noun
followed by a chain of spatial hops::

    Find the pillow closest to the book on the stool.
      head: pillow            hops: [(closest, book), (on, stool)]

    Find the picture above the suitcase furthest from the floor.
      head: picture           hops: [(above, suitcase), (farthest, floor)]

Each hop constrains the noun *before* it, so the chain is solved right to left: find the
suitcases furthest from the floor, then the picture above one of those. What comes back
is the surviving object plus a trace of how it was picked, or the ambiguity that stopped
it — never a guess.
"""

from __future__ import annotations

import re
from typing import Any

from cat2_geometry import (
    ARTICLES,
    COLORS,
    METRICS,
    MIN_MARGIN,
    Obj,
    between_holds,
    center_dist,
    color_matches,
    gap_dist,
    norm_class,
    relation_holds,
    set_dist,
)

# The first of these in a question binds its *target*; later ones qualify anchors.
RELATION_KEYWORDS = {
    "closest": "closest",
    "nearest": "closest",
    "closer": "closest",
    "farthest": "farthest",
    "furthest": "farthest",
    "between": "between",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "on": "on",
    "near": "near",
    "with": "supports",
    "has": "supports",
    "holding": "supports",
}

# Dropped wholesale before parsing: they carry no object and no relation.
FILLER = ARTICLES | {
    "find", "locate", "identify", "point", "to", "at", "from", "of", "it", "is", "are",
    "that", "which", "in", "and", "then", "please", "this",
}


def normalize(text: str) -> str:
    out = " ".join(str(text or "").lower().split())
    out = out.rstrip(".").strip()
    out = out.replace(" that has ", " with ").replace(" which has ", " with ")
    return out


def parse(text: str) -> dict[str, Any]:
    """Split a question into (colour, head noun, [(relation, [noun phrases])])."""
    words = normalize(text).replace(",", " ").split()
    color = ""
    head: list[str] = []
    hops: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for word in words:
        relation = RELATION_KEYWORDS.get(word)
        # "on" only opens a hop once we have a noun to attach it to, so "on the table"
        # after "the bowl" is a hop while a leading "on" would be noise.
        if relation and (head or hops):
            current = {"relation": relation, "phrases": [[]]}
            hops.append(current)
            continue
        if word in FILLER:
            # "and" separates the two anchors of a between.
            if word == "and" and current is not None and current["relation"] == "between":
                current["phrases"].append([])
            continue
        if word in COLORS and not head and current is None:
            color = word
            continue
        if current is None:
            head.append(word)
        else:
            current["phrases"][-1].append(word)

    return {
        "color": color,
        "head": " ".join(head),
        # A hop with no noun left in it is a back-reference, not a new constraint: the
        # "on it" of "the cabinet with a TV on it" points at the cabinet we already have.
        "hops": [
            {"relation": h["relation"], "phrases": phrases}
            for h in hops
            if (phrases := [" ".join(p) for p in h["phrases"] if p])
        ],
    }


def match_class(phrase: str, objects: list[Obj], strict: bool = False) -> list[Obj]:
    """Objects whose label the phrase names, tolerating plurals and NYU vs raw labels."""
    phrase = " ".join(w for w in phrase.split() if w not in FILLER)
    if not phrase:
        return []
    key = norm_class(phrase)
    # Raw label before NYU label: "the cabinet" means the cabinet, not also the TV
    # cabinet, even though IRef files both under the NYU class "cabinet".
    if fine := [o for o in objects if o.fine == key]:
        return fine
    if coarse := [o for o in objects if o.coarse == key]:
        return coarse

    def stems(text: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in text.split()}

    wanted = stems(phrase)
    contains = [
        o
        for o in objects
        if wanted <= stems(o.raw_label) or wanted <= stems(o.nyu_label)
    ]
    if contains or strict:
        return contains
    # Last resort: any shared noun ("cup of coffee" -> "coffee cup").
    return [o for o in objects if wanted & (stems(o.raw_label) | stems(o.nyu_label))]


def mentioned_labels(text: str, objects: list[Obj]) -> list[str]:
    """Labels of every scene object the question names, for SAM prompts.

    Includes nouns from the deeper hops ("... on the stool"), which the resolved answer
    and its immediate anchor leave out but a grounding model still has to detect.
    """
    spec = parse(text)
    phrases = [spec["head"]] + [p for hop in spec["hops"] for p in hop["phrases"]]
    labels: list[str] = []
    for phrase in phrases:
        for obj in match_class(phrase, objects, strict=True):
            if obj.display not in labels:
                labels.append(obj.display)
    return labels


def vertical_gap(obj: Obj, datum: Obj) -> float:
    """Height of an object over (or under) a slab. See metrics_for()."""
    return max(obj.lo[2] - datum.hi[2], datum.lo[2] - obj.hi[2], 0.0)


def metrics_for(anchors: list[Obj]) -> dict[str, Any]:
    """Distance metrics appropriate to the anchors.

    "Furthest from the floor" means highest. Centre-to-centre distance to a floor slab
    is dominated by where the object sits in the room, so the usual pair of metrics
    ranks a low picture on a far wall above a high picture overhead.
    """
    if anchors and all(a.fine in ("floor", "ceiling") for a in anchors):
        return {"height": vertical_gap}
    return METRICS


def matches(
    candidates: list[Obj], groups: list[list[Obj]], relation: str
) -> tuple[list[Obj], str]:
    """Candidates satisfying `relation` against the anchor groups, best first.

    One group per noun phrase, which only matters for "between": pairing must cross the
    two groups, since pairing inside a flattened list accepts a lamp between two windows
    as an answer to "between a door frame and a window". No uniqueness is enforced here;
    that is pick_unique's job, so a later modifier still gets a chance to discriminate.
    """
    anchors = [o for group in groups for o in group]
    if not candidates or not anchors:
        return [], "no candidates or no anchors"

    if relation == "between":
        left = groups[0]
        right = groups[1] if len(groups) > 1 else groups[0]
        kept = [
            c
            for c in candidates
            if any(
                between_holds(c, a1, a2)[0] for a1 in left for a2 in right if a1.id != a2.id
            )
        ]
        return kept, (
            f"between: {len(kept)} in the corridor -> {kept[:4]}"
            if kept
            else "nothing sits between the two anchor sets"
        )

    if relation in ("closest", "farthest"):
        metric = next(iter(metrics_for(anchors).values()))
        ranked = sorted(
            candidates,
            key=lambda c: set_dist(c, anchors, metric),
            reverse=relation == "farthest",
        )
        return ranked, f"{relation}: ranked {ranked[:3]}"

    kept = [c for c in candidates if relation_holds(c, anchors, relation)]
    if relation == "near":
        kept = sorted(kept, key=lambda c: set_dist(c, anchors, gap_dist))
    return kept, (
        f"{relation}: {len(kept)} match(es) -> {kept[:3]}"
        if kept
        else f"nothing is {relation} the anchor(s)"
    )


def pick_unique(
    candidates: list[Obj], groups: list[list[Obj]], relation: str
) -> tuple[Obj | None, str]:
    """Commit to one object, or explain why the question does not pin one down."""
    if not candidates:
        return None, "no candidate satisfies the question"
    if len(candidates) == 1:
        return candidates[0], f"{relation}: {candidates[0]} is the only match"

    anchors = [o for group in groups for o in group]
    if relation in ("closest", "farthest", "near"):
        metrics = {"box-gap": gap_dist} if relation == "near" else metrics_for(anchors)
        winners, notes = [], []
        for name, metric in metrics.items():
            ranked = sorted(
                ((set_dist(c, anchors, metric), c) for c in candidates),
                key=lambda p: p[0],
                reverse=relation == "farthest",
            )
            (best_d, best), (next_d, runner_up) = ranked[0], ranked[1]
            if abs(next_d - best_d) < MIN_MARGIN:
                return None, (
                    f"{relation} by {name} is a tie: {best} {best_d:.2f} m vs "
                    f"{runner_up} {next_d:.2f} m"
                )
            winners.append(best)
            notes.append(f"{name} {best_d:.2f} m (next {next_d:.2f} m)")
        if len({w.id for w in winners}) > 1:
            return None, f"{relation}: metrics disagree ({winners})"
        return winners[0], f"{relation}: " + "; ".join(notes)

    return None, f"{len(candidates)} objects are {relation} the anchor(s): {candidates[:4]}"


def solve_in_regions(text: str, objects: dict[str, Obj]) -> dict[str, Any]:
    """Solve per region, then scene-wide, so a two-room scene cannot mix rooms.

    A question that resolves in two regions at once is ambiguous by construction, and is
    reported as such rather than answered with whichever room came first.
    """
    regions = sorted({o.region for o in objects.values()})
    hits: list[tuple[str, dict[str, Any]]] = []
    last: dict[str, Any] = {"target": None, "trace": [], "reason": "scene has no objects"}
    for region in regions:
        result = solve(text, [o for o in objects.values() if o.region == region])
        last = result
        if result["target"] is not None:
            hits.append((region, result))
    if len(hits) == 1:
        return hits[0][1]
    if len(hits) > 1:
        return {
            "target": None,
            "trace": hits[0][1]["trace"],
            "reason": "resolves in more than one region: "
            + ", ".join(f"{r}->{h['target']}" for r, h in hits),
        }
    if len(regions) > 1:
        whole = solve(text, list(objects.values()))
        if whole["target"] is not None:
            whole["trace"].append("resolved scene-wide (no single region resolved it)")
        return whole
    return last


def solve(text: str, objects: list[Obj]) -> dict[str, Any]:
    """Resolve a question to one object.

    Returns ``{"target": Obj|None, "parse": ..., "trace": [...], "reason": str}``.
    """
    spec = parse(text)
    trace: list[str] = [f"head={spec['head']!r} color={spec['color']!r} hops={spec['hops']}"]

    heads = match_class(spec["head"], objects)
    if spec["color"]:
        colored = [o for o in heads if color_matches(spec["color"], o.colors)]
        if colored:
            heads = colored
            trace.append(f"colour {spec['color']}: {len(heads)} candidate(s)")
    if not heads:
        return {"target": None, "parse": spec, "trace": trace, "reason": f"no object matches head noun {spec['head']!r}"}
    trace.append(f"head candidates: {heads[:6]}{' ...' if len(heads) > 6 else ''}")

    hops = spec["hops"]
    if not hops:
        if len(heads) == 1:
            return {"target": heads[0], "parse": spec, "trace": trace, "reason": "unique by class"}
        return {
            "target": None,
            "parse": spec,
            "trace": trace,
            "reason": f"{len(heads)} objects share the head noun and no relation narrows it",
        }

    # One group of anchor candidates per noun phrase in each hop.
    groups: list[list[list[Obj]]] = [
        [match_class(phrase, objects) for phrase in hop["phrases"]] for hop in hops
    ]
    # Each hop constrains the noun before it, i.e. the last phrase of the previous hop,
    # so the chain collapses right to left.
    deferred: list[int] = []
    for i in range(len(hops) - 1, 0, -1):
        host = groups[i - 1][-1]
        if len(host) <= 1:
            # A modifier that cannot discriminate its host is not describing the host:
            # "the picture above the suitcase furthest from the floor" has one suitcase,
            # so "furthest from the floor" is telling us which *picture*.
            deferred.append(i)
            trace.append(
                f"anchor hop {i} ({hops[i]['relation']}): host already unique, applying it "
                "to the target instead"
            )
            continue
        refined, note = matches(host, groups[i], hops[i]["relation"])
        picked, unique_note = pick_unique(refined, groups[i], hops[i]["relation"])
        trace.append(f"anchor hop {i} ({hops[i]['relation']}): {note}; {unique_note}")
        if picked is None:
            return {
                "target": None,
                "parse": spec,
                "trace": trace,
                "reason": f"could not resolve the anchor of hop {i}: {unique_note}",
            }
        groups[i - 1][-1] = [picked]

    if hops[0]["relation"] == "between" and len(groups[0]) < 2:
        return {
            "target": None,
            "parse": spec,
            "trace": trace,
            "reason": "between needs two resolvable anchor phrases",
        }

    candidates, note = matches(heads, groups[0], hops[0]["relation"])
    trace.append(f"target hop ({hops[0]['relation']}): {note}")
    last_hop = 0
    for i in reversed(deferred):
        narrowed, note = matches(candidates, groups[i], hops[i]["relation"])
        trace.append(f"deferred hop {i} ({hops[i]['relation']}): {note}")
        if not narrowed:
            break
        candidates, last_hop = narrowed, i

    target, reason = pick_unique(candidates, groups[last_hop], hops[last_hop]["relation"])
    trace.append(f"pick: {reason}")
    if target is None:
        return {"target": None, "parse": spec, "trace": trace, "reason": reason}
    return {
        "target": target,
        "parse": spec,
        "trace": trace,
        "reason": reason,
        "anchors": [o for group in groups[0] for o in group],
        "relation": hops[0]["relation"],
    }
