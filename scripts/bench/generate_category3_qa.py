#!/usr/bin/env python3
"""Build the category-3 (instruction-following) benchmark from the organizers' demos.

The challenge scores instruction following out of 6 -- two thirds of a scene's points --
on "whether it follows the path constraints in the command and in the correct order"
(README, "Question Types and Initial Scoring"). Nothing else is published: the per-scene
``questions.pdf`` states only "Response: Trajectory (series of waypoints)", with no radius,
no tolerance, and no mention of the ``.ply`` files at all. So the ground truth has to be
recovered, and the only evidence of what "near" and "between" mean in this challenge is
``questions/<scene>/trajectory_q4.ply`` / ``trajectory_q5.ply`` -- one dense recording of a
correct run per instruction question, 30 in total.

Two facts make that recovery exact rather than a guess:

* **The trajectory frame and the IRef-VLA annotation frame are the same frame.** Every
  demo starts at (0, 0) -- the robot spawn -- and runs at a constant z = 0.75, the
  ``vehicleHeight`` of the simulator. No offset, no axis swap. So a demo vertex and an
  annotated box can be compared directly, and the object a leg of the path is about is the
  one the path comes closest to.
* **Position along the path *is* the constraint order.** Sorting the grounded objects by
  the index of their closest path vertex reproduces the order the sentence states, which
  is the thing being scored.

The pipeline is therefore: split the instruction at its verbs into ordered clauses, hand
each clause's noun phrase to ``utils.text_solver`` (the same solver the category-2
benchmark is generated and audited with, so "closest" means one thing across both), then
*cross-examine every grounding against the demo*. A clause whose object the demo never
approaches, or approaches out of turn, is not written out as ground truth -- it is written
to ``category3_candidates/<scene>.json`` with its alternatives for a human to settle in
``category3_overrides.json``. Nothing here invents a constraint the recording does not
support.

Usage::

    python3 scripts/bench/generate_category3_qa.py --dry-run
    python3 scripts/bench/generate_category3_qa.py --dry-run --scene loft -v
    python3 scripts/bench/generate_category3_qa.py --write
    python3 scripts/bench/generate_category3_qa.py --plot loft
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.text_solver as solver
from utils.geometry import (
    BETWEEN_LATERAL,
    BETWEEN_LATERAL_MAX,
    BETWEEN_SPAN,
    COLORS,
    VACUOUS_ANCHORS,
    Obj,
    answer_payload,
    between_holds,
    gap_dist,
    load_objects,
    object_nouns,
    relation_holds,
)

REPO = Path(__file__).resolve().parents[2]
QUESTIONS = REPO / "questions"
BENCH = REPO / "data" / "benchmark"
OVERRIDES = Path(__file__).resolve().parent / "category3_overrides.json"
CANDIDATES = Path(__file__).resolve().parent / "category3_candidates"

# score.py's constraint radius, and the floor for every constraint here. Most demos come
# well inside it, but a wall-mounted target cannot be approached: the picture in
# hotel_room_2 is viewed from 2.6 m because there is a bed in the way. Rather than declare
# the organizers' own run a failure, such a constraint keeps the distance the demo proves
# is required, recorded as radius_source so the widening is never silent.
RADIUS = 1.5
RADIUS_SLACK = 0.25
# A clause may name a set rather than one thing ("go to the vases on the cabinet"). Treat
# the set as one waypoint only when its members are close enough to stand at once.
GROUP_SPREAD = 1.20
# Resampling step for the polyline copied into the QA file. The recordings are ~1 cm
# apart, which is 30x denser than any scorer needs and would dominate the file.
POLYLINE_STEP = 0.25
# A clause whose object the demo never comes this close to was not grounded to the thing
# the sentence meant, whatever the text says.
MAX_ANCHOR_DIST = 3.00
# When the text alone cannot choose, the demo may -- but only decisively. Same margin the
# category-2 benchmark requires of an answer over its runner-up.
MIN_MARGIN = 0.30

# Adjectives that qualify a noun without naming a class. IRef labels carry no size, so
# "the small table" and "the big picture" have to lose the adjective to match anything --
# the relation after it is what actually discriminates.
SIZE_WORDS = {"small", "big", "large", "little", "tall", "short", "long", "tiny", "huge"}

# Instruction verbs, longest first so "take the path between" wins over "between".
# The verb is the only thing that says whether a named place is to be sought or shunned:
# "take the path between the two columns" and "avoiding the path between the chair and the
# folding screen" are the same geometry with opposite signs.
MARKERS: list[tuple[str, str]] = [
    (r"avoid(?:ing)?(?: the path)? between\b", "avoid_between"),
    (r"avoid(?:ing)?(?: the path)?(?: near| by)?\b", "avoid_near"),
    (r"take the path between\b", "pass_between"),
    (r"(?:go|pass|passing|move)\s+between\b", "pass_between"),
    (r"take the path (?:near|by|past)\b", "pass_near"),
    (r"pass(?:ing)?\s+(?:by|near|past)\b", "pass_near"),
    (r"stop(?:ping)?\s+(?:at|by|near|in front of)\b", "goal"),
    (r"(?:and\s+)?finally,?\s+(?:go\s+)?to\b", "goal"),
    (r"(?:go|head|drive|navigate)\s+(?:to|near|towards|toward|by)\b", "pass_near"),
    (r"then,?\s+to\b", "pass_near"),
    (r"(?:go|head|drive|navigate)\b", "pass_near"),
]
MARKER_RE = re.compile("|".join(f"(?:{pat})" for pat, _ in MARKERS))
_MARKER_KIND = [(re.compile(pat), kind) for pat, kind in MARKERS]

# " to " opens a second constraint inside one clause -- "take the path between the TV and
# the bed *to* the picture closest to the TV" is a pass and a goal. But "closest to the
# TV" is one constraint, so only an infinitive that does not belong to a comparative
# splits. These are the words that own the "to" after them.
COMPARATIVES = {"closest", "nearest", "farthest", "furthest", "close", "next", "adjacent"}
TO_SPLIT_RE = re.compile(r"\s+to\s+")

# "the two columns", "the two tables" -- one phrase naming a pair.
COUNT_WORDS = {"two": 2, "three": 3, "both": 2}
COUNT_RE = re.compile(r"\b(two|three|both)\b")


# ---------------------------------------------------------------- trajectory


class Trajectory:
    """A demo run: the (x, y) track the organizers recorded for one question."""

    def __init__(self, path: Path):
        text = path.read_text()
        head, _, body = text.partition("end_header")
        declared = int(re.search(r"element vertex (\d+)", head).group(1))
        values = [float(v) for v in body.split()]
        pts = np.asarray(values, dtype=float).reshape(-1, 3)
        if len(pts) != declared:
            raise ValueError(f"{path}: header says {declared} vertices, found {len(pts)}")
        self.path = path
        self.z = float(pts[0, 2])
        self.xy = pts[:, :2]
        steps = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.length = float(steps.sum())

    def __len__(self) -> int:
        return len(self.xy)

    def approach(self, center: Sequence[float], after: int = 0) -> tuple[int, float]:
        """(index of the closest vertex, distance to it), searching from ``after`` on.

        Constraints are scored in order, so each one is looked for in the part of the run
        that is still ahead -- exactly how ``score_instruction`` advances its cursor. A
        global argmin would credit an incidental early pass: in arabic_room the demo
        clips the midpoint of the two columns on its way out of the room long before it
        takes the path between them.
        """
        start = min(max(after, 0), len(self.xy) - 1)
        d = np.linalg.norm(self.xy[start:] - np.asarray(center[:2], dtype=float), axis=1)
        i = int(d.argmin())
        return start + i, float(d[i])

    def final_dist(self, center: Sequence[float]) -> float:
        """Distance from where the run stops -- the only thing a goal is scored on."""
        return float(np.linalg.norm(self.xy[-1] - np.asarray(center[:2], dtype=float)))

    def polyline(self, step: float = POLYLINE_STEP) -> list[list[float]]:
        """The track resampled to ~step metres, endpoints always kept."""
        out = [self.xy[0]]
        for p in self.xy[1:]:
            if np.linalg.norm(p - out[-1]) >= step:
                out.append(p)
        if not np.allclose(out[-1], self.xy[-1]):
            out.append(self.xy[-1])
        return [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in out]


def trajectory_path(scene: str, qid: str) -> Path:
    return QUESTIONS / scene / f"trajectory_{qid.lower().replace('q0', 'q')}.ply"


# ---------------------------------------------------------------- clauses


def strip_size_words(phrase: str) -> str:
    kept = [w for w in phrase.split() if w not in SIZE_WORDS]
    return " ".join(kept) if kept else phrase


def split_clauses(instruction: str) -> list[dict[str, Any]]:
    """Cut an instruction into ordered (kind, text) clauses at its verbs."""
    text = solver.normalize(instruction)
    marks = [(m.start(), m.end(), m.group(0)) for m in MARKER_RE.finditer(text)]
    if not marks:
        return [{"kind": "goal", "verb": "", "text": text}]

    clauses: list[dict[str, Any]] = []
    for n, (start, end, verb) in enumerate(marks):
        stop = marks[n + 1][0] if n + 1 < len(marks) else len(text)
        body = trim_clause(text[end:stop])
        kind = next(k for rx, k in _MARKER_KIND if rx.fullmatch(verb.strip()))
        if not body:
            continue
        clauses.extend(_split_on_infinitive(kind, verb.strip(), body))
    return clauses


# A clause runs up to the next verb, so it keeps the conjunction that introduced that
# verb: "take the path between the two columns, and" before "stop at".
_TRAILING = re.compile(r"[\s,.;]*\b(?:and|then|finally|next|also)\b[\s,.;]*$")
_LEADING = re.compile(r"^[\s,.;]*(?:\b(?:and|then|first|finally|next|also)\b[\s,.;]*)*")


def trim_clause(body: str) -> str:
    body = _LEADING.sub("", body)
    while True:
        trimmed = _TRAILING.sub("", body)
        if trimmed == body:
            return trimmed.strip(" ,.;")
        body = trimmed


def _split_on_infinitive(kind: str, verb: str, body: str) -> list[dict[str, Any]]:
    """Break "<pass> X to Y" into the pass and the leg that follows it."""
    for m in TO_SPLIT_RE.finditer(body):
        before = body[: m.start()].split()
        if before and before[-1] in COMPARATIVES:
            continue  # "closest to the TV" -- one constraint, not two
        head, tail = body[: m.start()].strip(), body[m.end():].strip()
        if head and tail:
            return [
                {"kind": kind, "verb": verb, "text": head},
                {"kind": "pass_near", "verb": "to", "text": tail},
            ]
    return [{"kind": kind, "verb": verb, "text": body}]


def mark_goal(clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name the clause the run has to end on, before anything is grounded.

    An explicit "stop at" says which one; otherwise it is the last positive clause, since
    a sentence that finishes on a place finishes *at* it. Knowing this up front matters:
    a goal is scored on the final pose and every other constraint on the path, so they
    are looked for in the recording in different ways.
    """
    positives = [c for c in clauses if not c["kind"].startswith("avoid")]
    if not positives:
        return clauses
    explicit = [c for c in positives if c["kind"] == "goal"]
    goal = explicit[-1] if explicit else positives[-1]
    for c in positives:
        c["kind"] = "goal" if c is goal else ("pass_between" if c["kind"].endswith("between") else "pass_near")
    return clauses


def constraint_radius(dist: float) -> float:
    """The smallest radius, at or above score.py's default, that admits the demo."""
    return max(RADIUS, math.ceil((dist + RADIUS_SLACK) * 20) / 20)


def _count_word(text: str) -> str:
    m = COUNT_RE.search(text)
    return m.group(1) if m else ""


def anchor_phrases(kind: str, text: str) -> list[str]:
    """The noun phrases a clause names: two for a between, one otherwise."""
    if not kind.endswith("between"):
        return [text]
    if _count_word(text):
        # "between the two columns" names a class, not a pair of different things.
        return [COUNT_RE.sub("", text).strip()]
    parts = re.split(r"\s+and\s+", text, maxsplit=1)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------- grounding


def ground(
    phrase: str,
    objects: dict[str, Obj],
    traj: Trajectory,
    cursor: int,
    want: int = 1,
    is_goal: bool = False,
    is_anchor: bool = False,
) -> dict[str, Any]:
    """Resolve one noun phrase to the object(s) it names, with the demo as referee.

    The text solver goes first, because it encodes what the words mean. But an
    instruction is not a referring expression in isolation -- it was performed, and the
    recording says which of two cabinets was meant far more reliably than the sentence
    does. So a solver answer the demo never approaches is rejected, and a phrase the
    solver leaves ambiguous is settled by the run when the run is decisive about it.

    ``want`` > 1 asks for a set the sentence counted ("the two columns"). ``is_anchor``
    marks one side of a between, which the run is *not* expected to approach -- passing
    between two things means staying away from both -- so only its midpoint is held to
    the demo, back in the caller.
    """
    cleaned = strip_size_words(phrase)
    head = solver.parse(cleaned)["head"] or cleaned
    limit = math.inf if is_anchor else MAX_ANCHOR_DIST

    def reach(obj: Obj) -> float:
        return traj.final_dist(obj.center) if is_goal else traj.approach(obj.center, cursor)[1]

    if want > 1:
        members = solver.match_class(cleaned, list(objects.values()))
        if len(members) == want:
            return {"objects": members, "reason": f"{want} objects share the class", "trace": []}
        return {
            "objects": [],
            "reason": f"phrase counts {want} but {len(members)} objects match {cleaned!r}",
            "trace": [],
        }

    result = solver.solve_in_regions(cleaned, objects)
    trace = list(result.get("trace", []))
    if result["target"] is not None and reach(result["target"]) <= limit:
        return {"objects": [result["target"]], "reason": result["reason"], "trace": trace}
    if result["target"] is not None:
        trace.append(
            f"rejected {result['target']}: the demo stays {reach(result['target']):.2f} m from it"
        )

    members = [o for o in solver.match_class(head, list(objects.values())) if navigable(o)]
    plausible = sorted(
        ((reach(o), o) for o in members if reach(o) <= limit), key=lambda r: r[0]
    )
    if not plausible:
        return {"objects": [], "reason": result["reason"], "trace": trace}

    # The candidates the demo likes may be one place rather than several: four vases on a
    # cabinet, or a window the annotation split into two panes. Take everything huddled
    # around the best candidate, and accept it only if the rivals left outside are clearly
    # worse -- otherwise the set is not a waypoint, it is an unresolved ambiguity.
    best_dist, best_obj = plausible[0]
    cluster = [o for _, o in plausible if float(np.linalg.norm(o.center[:2] - best_obj.center[:2])) <= GROUP_SPREAD]
    rivals = [d for d, o in plausible if o not in cluster]
    if len(cluster) > 1 and (not rivals or min(rivals) - max(
        d for d, o in plausible if o in cluster
    ) >= MIN_MARGIN):
        return {
            "objects": cluster,
            "reason": f"{len(cluster)} co-located {head or 'object'}s the demo visits together",
            "trace": trace,
        }
    if len(plausible) == 1 or plausible[1][0] - plausible[0][0] >= MIN_MARGIN:
        best = plausible[0]
        return {
            "objects": [best[1]],
            "reason": (
                f"text was indecisive ({result['reason']}); the demo picks {best[1]} "
                f"at {best[0]:.2f} m"
            ),
            "trace": trace,
        }
    return {
        "objects": [],
        "reason": (
            f"{result['reason']}; the demo does not choose either: "
            + ", ".join(f"{o} {d:.2f} m" for d, o in plausible[:3])
        ),
        "trace": trace,
    }


# ---------------------------------------------------------------- reference objects


# Relations whose truth the boxes can check, so the anchor that makes a clause true can be
# singled out. "closest"/"farthest" are deliberately absent: they compare against the whole
# set the noun names, so there is nothing to filter -- every candidate is the comparison.
CHECKABLE = {"on", "supports", "above", "below", "in", "near"}


def anchor_objects(
    phrase: str, targets: Sequence[Obj], objects: dict[str, Obj]
) -> tuple[list[Obj], list[str]]:
    """The objects a clause names only to say *which* thing it means.

    "The stool under the picture" is two objects, not one: a system has to find the
    picture before it can tell that stool from the other one. Those reference objects are
    in the sentence and annotated in the metadata, so the benchmark carries their boxes
    too -- as evidence, never as a constraint. Nothing here can move a waypoint.

    Resolved from the *already grounded* target rather than from the solver's internals,
    so a clause pinned by an override gets its anchors the same way a solved one does.
    Returns the anchors in the order the sentence names them, with the reason each was
    kept.
    """
    spec = solver.parse(strip_size_words(phrase))
    if not spec["hops"] or not targets:
        return [], []

    # Stay in the target's room unless the room holds nothing by that name: in
    # hotel_room_2 the door frame "the picture closest to the door" refers to is annotated
    # in a different region from the picture, and dropping it would lose a real reference.
    pool = [o for o in objects.values() if o.region in {t.region for t in targets}]
    everywhere = list(objects.values())

    picked: list[Obj] = []
    reasons: list[str] = []
    seen = {t.id for t in targets}
    hosts = list(targets)
    for hop in spec["hops"]:
        relation = hop["relation"]
        groups = []
        for text in hop["phrases"]:
            cleaned = drop_colors(text)
            found = solver.match_class(cleaned, pool, strict=True) or solver.match_class(
                cleaned, everywhere, strict=True
            )
            groups.append((cleaned, [o for o in found if o.id not in seen]))

        if relation == "between" and len(groups) == 2:
            kept, why = bracketing_pair(groups[0][1], groups[1][1], hosts)
            notes = [why]
        else:
            kept, notes = [], []
            for cleaned, found in groups:
                side, why = narrow_anchors(cleaned, found, hosts, relation)
                kept.extend(side)
                notes.append(why)

        for obj in kept:
            if obj.id not in seen:
                seen.add(obj.id)
                picked.append(obj)
        reasons.extend(n for n in notes if n)
        # A later hop qualifies the anchor before it, not the target -- "the folder on the
        # cabinet closest to the whiteboard" measures the cabinet, not the folder. Only a
        # hop that resolved to exactly one object can host the next one.
        hosts = kept if len(kept) == 1 else list(targets)
    return picked, reasons


def drop_colors(phrase: str) -> str:
    """Colour words qualify an object without naming a class, and IRef labels carry none.

    "the black chair" matches nothing at all with the colour left in, which is how
    livingroom_1's "the lamp closest to the black chair" loses its reference object.
    """
    kept = [w for w in phrase.split() if w not in COLORS]
    return " ".join(kept) if kept else phrase


def plural_phrase(phrase: str, candidates: Sequence[Obj]) -> bool:
    """Does the phrase name a set? "the columns" does; "the flowers" does not.

    A label that is itself plural ("flowers", "windows") is one object with a plural name,
    so the test asks whether any candidate is actually called that.
    """
    word = phrase.split()[-1] if phrase.split() else ""
    if len(word) <= 3 or not word.endswith("s"):
        return False
    return not any(o.fine.endswith(word) or o.coarse.endswith(word) for o in candidates)


def narrow_anchors(
    phrase: str, candidates: Sequence[Obj], hosts: Sequence[Obj], relation: str
) -> tuple[list[Obj], str]:
    """Which of the objects the noun could name the sentence actually points at."""
    if not candidates:
        return [], f"{relation} {phrase!r}: nothing in the scene carries that label"

    kept, note = list(candidates), ""
    if relation in CHECKABLE:
        holds = [a for a in candidates if any(relation_holds(h, [a], relation) for h in hosts)]
        if holds:
            kept = holds
            listed = ", ".join(str(o) for o in holds)
            note = f"{relation} {phrase!r}: the boxes put {listed} there"
        else:
            note = f"{relation} {phrase!r}: no box supports the relation"

    if plural_phrase(phrase, candidates):
        return kept, note or f"{relation} {phrase!r}: plural, so all {len(kept)} are meant"
    if len(kept) == 1:
        return kept, note or f"{relation} {phrase!r}: {kept[0]} is the only one"

    # Still undecided, and the sentence says one thing. Take the nearest, which is the
    # object the solver ranked on too: set_dist is a min over the anchor set, so even a
    # "farthest from" comparison is decided by the closest member of it.
    best = min(kept, key=lambda a: min(gap_dist(h, a) for h in hosts))
    gap = min(gap_dist(h, best) for h in hosts)
    prefix = f"{note}; " if note else f"{relation} {phrase!r}: "
    return [best], f"{prefix}nearest of {len(kept)} is {best} at {gap:.2f} m"


def bracketing_pair(
    left: Sequence[Obj], right: Sequence[Obj], hosts: Sequence[Obj]
) -> tuple[list[Obj], str]:
    """The two objects "between the X and the Y" is really about.

    Several pairs can satisfy betweenness at once -- livingroom_1's vase sits between three
    different tv/door combinations -- so the pair that brackets the target most squarely
    wins, by the same lateral offset ``between_holds`` reports.
    """
    best: tuple[Obj, Obj] | None = None
    offset = math.inf
    for a1 in left:
        for a2 in right:
            if a1.id == a2.id:
                continue
            for host in hosts:
                holds, lateral = between_holds(host, a1, a2)
                if holds and lateral < offset:
                    best, offset = (a1, a2), lateral
    if best is not None:
        return list(best), (
            f"between: {best[0]} and {best[1]} bracket the target most squarely "
            f"(lateral offset {offset:.2f} m)"
        )
    # The instruction says "between", so both sides are referenced whether or not the
    # heuristic agrees; fall back to the nearest of each.
    kept = [
        min(side, key=lambda a: min(gap_dist(h, a) for h in hosts))
        for side in (left, right)
        if side
    ]
    return kept, "between: no pair brackets the target, keeping the nearest of each side"


def navigable(obj: Obj) -> bool:
    """Can this object be a place to go?

    Deliberately *not* ``Obj.structural``, which is category 2's "makes a poor answer"
    list. A column is a bad thing to point a bounding box at and an excellent thing to
    walk between, and the organizers write instructions that way: "take the path between
    the two columns", "pass by the stairs", "to the door near the exit sign". Only the
    room-sized datums are excluded, because "near the floor" names no place at all.
    """
    return obj.fine not in VACUOUS_ANCHORS and obj.coarse not in VACUOUS_ANCHORS


def waypoint(kind: str, groups: list[list[Obj]]) -> list[float]:
    """Where the robot has to be for the constraint to hold.

    For a between, that is the middle of the gap the two objects leave -- not the midpoint
    of their centres, which for a wall picture and a dining table lands inside the table.
    """
    if kind.endswith("between") and len(groups) == 2:
        return list(corridor(groups[0], groups[1])[1])
    flat = [o for g in groups for o in g]
    return list(np.mean([o.center[:2] for o in flat], axis=0))


# ---------------------------------------------------------------- avoid zones


def footprint(obj: Obj) -> np.ndarray:
    """The box's oriented ground outline: corners 0-3 are its top face, in order."""
    return np.asarray(obj.corners[0:4, :2], dtype=float)


def convex_hull(points: np.ndarray) -> list[list[float]]:
    """Monotone chain, counter-clockwise. Small inputs only -- no scipy in the container."""
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in points})
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return [list(p) for p in lower[:-1] + upper[:-1]]


def corridor(side_a: Sequence[Obj], side_b: Sequence[Obj]) -> tuple[list[list[float]], np.ndarray]:
    """The region "between" two objects, as (polygon, its centre).

    This is ``utils.geometry.between_holds`` written for a point instead of an object,
    with the same constants -- the middle stretch of the segment joining the two centres,
    widened by a slack proportional to their separation and capped at a metre. Reusing
    that definition is the whole point: VLA-3D's ``between`` heuristic is what generated
    these questions, so an avoid zone here means what the question meant.

    The alternative, a convex hull of the two footprints, spans the objects themselves
    and everything outboard of them; in livingroom_2 it swallows half the room and calls
    a run that passed nowhere near the TV a violation.
    """
    p1 = np.mean([o.center[:2] for o in side_a], axis=0)
    p2 = np.mean([o.center[:2] for o in side_b], axis=0)
    seg = p2 - p1
    span = float(np.linalg.norm(seg))
    if span < 1e-6:
        pts = np.vstack([footprint(o) for o in list(side_a) + list(side_b)])
        return [[round(x, 3), round(y, 3)] for x, y in convex_hull(pts)], p1

    u = seg / span
    n = np.array([-u[1], u[0]])
    half = min(BETWEEN_LATERAL * span, BETWEEN_LATERAL_MAX)
    lo, hi = BETWEEN_SPAN
    poly = [
        p1 + lo * seg + half * n,
        p1 + hi * seg + half * n,
        p1 + hi * seg - half * n,
        p1 + lo * seg - half * n,
    ]
    return [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in poly], p1 + 0.5 * seg


def avoid_polygon(kind: str, groups: list[list[Obj]]) -> list[list[float]]:
    """The ground region the command forbids."""
    if kind.endswith("between") and len(groups) == 2:
        return corridor(groups[0], groups[1])[0]
    # "avoid the path near X" is about X's surroundings; X's own footprint is space the
    # robot could not occupy anyway, so it is grown until it forbids something.
    flat = [o for g in groups for o in g]
    hull = np.array(convex_hull(np.vstack([footprint(o) for o in flat])))
    centroid = hull.mean(axis=0)
    grown = centroid + (hull - centroid) * _near_growth(hull, centroid)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in grown]


def _near_growth(hull: np.ndarray, centroid: np.ndarray) -> float:
    """Scale that puts the hull's edge ~1 m out, without a polygon-offset dependency."""
    reach = float(np.linalg.norm(hull - centroid, axis=1).mean())
    return 1.0 + (1.0 / reach) if reach > 1e-6 else 1.0


def point_in_poly(pt: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    """Ray cast. Kept identical to scripts/eval/score.py so both agree on a violation."""
    x, y, inside = pt[0], pt[1], False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


# ---------------------------------------------------------------- per question


def clause_anchors(
    phrases: Sequence[str],
    groups: list[list[Obj]],
    counted: bool,
    objects: dict[str, Obj],
    pin: dict[str, Any] | None,
) -> tuple[list[Obj], list[str]]:
    """Reference objects for one clause, one noun phrase at a time.

    A between names two things and grounds each separately, so each phrase is resolved
    against its own side of the gap -- except when the pair came from a single counted
    phrase ("the two columns"), where both sides are what the phrase means.
    """
    if pin is not None and pin.get("anchor_ids"):
        ids = [str(i) for i in pin["anchor_ids"]]
        missing = [i for i in ids if i not in objects]
        if missing:
            raise KeyError(f"override pins unknown reference object(s) {missing}")
        return [objects[i] for i in ids], [pin.get("anchor_why", "pinned by override")]

    flat = [o for g in groups for o in g]
    seen = {o.id for o in flat}
    picked: list[Obj] = []
    reasons: list[str] = []
    for n, phrase in enumerate(phrases):
        side = flat if counted or len(groups) != len(phrases) else groups[n]
        found, why = anchor_objects(phrase, side, objects)
        for obj in found:
            if obj.id not in seen:
                seen.add(obj.id)
                picked.append(obj)
        reasons.extend(why)
    return picked, reasons


def build_question(
    scene: str,
    qid: str,
    instruction: str,
    objects: dict[str, Obj],
    traj: Trajectory,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (question entry, unresolved clauses needing a human)."""
    clauses = mark_goal(split_clauses(instruction))
    key = f"{scene}/{qid}"
    pinned = dict(overrides.get(key, {}))
    # A question-level "_review" says a human read this one and found the recording and
    # the sentence genuinely at odds. It is carried into the file so the disagreement is
    # visible to anyone scoring against it, rather than smoothed away.
    review = pinned.pop("_review", None)
    problems: list[dict[str, Any]] = []

    resolved: list[dict[str, Any]] = []
    cursor = 0
    for n, clause in enumerate(clauses):
        phrases = anchor_phrases(clause["kind"], clause["text"])
        # "between the two columns" is one phrase naming a pair; "between the TV and the
        # bed" is two phrases naming one thing each.
        counted = clause["kind"].endswith("between") and len(phrases) == 1
        want = COUNT_WORDS.get(_count_word(clause["text"]), 2) if counted else 1

        pin = pinned.get(str(n))
        groups: list[list[Obj]] = []
        reasons: list[str] = []
        trace: list[str] = []
        if pin is not None:
            for ids in pin["object_ids"]:
                ids = ids if isinstance(ids, list) else [ids]
                missing = [i for i in ids if i not in objects]
                if missing:
                    raise KeyError(f"{key}[{n}] override names unknown object(s) {missing}")
                groups.append([objects[i] for i in ids])
            reasons.append(pin.get("why", "pinned by override"))
            if pin.get("kind"):
                clause["kind"] = pin["kind"]
        else:
            for phrase in phrases:
                got = ground(
                    phrase,
                    objects,
                    traj,
                    cursor,
                    want,
                    is_goal=clause["kind"] == "goal",
                    is_anchor=clause["kind"].endswith("between") and not counted,
                )
                reasons.append(got["reason"])
                trace.extend(got["trace"])
                if not got["objects"]:
                    groups = []
                    break
                # A counted phrase yields the pair itself, so each member is its own side
                # of the corridor; an ordinary phrase yields one side.
                groups.extend([[o] for o in got["objects"]] if counted else [got["objects"]])

        if not groups:
            problems.append(
                {
                    "clause": n,
                    "kind": clause["kind"],
                    "verb": clause["verb"],
                    "text": clause["text"],
                    "reason": "; ".join(reasons) or "no grounding",
                    "candidates": rank_candidates(clause["text"], objects, traj),
                }
            )
            continue

        anchors, anchor_why = clause_anchors(phrases, groups, counted, objects, pin)

        center = waypoint(clause["kind"], groups)
        is_avoid = clause["kind"].startswith("avoid")
        index, dist = traj.approach(center, 0 if is_avoid else cursor)
        # A goal is met or missed on the final pose, so that is the distance it is held to.
        if clause["kind"] == "goal":
            dist = traj.final_dist(center)
            index = len(traj) - 1
        flat = [o for g in groups for o in g]
        entry = {
            "clause": n,
            "kind": clause["kind"],
            "phrase": clause["text"],
            "label": ", ".join(dict.fromkeys(o.display for o in flat)),
            "object_ids": [o.id for o in flat],
            "center": [round(float(center[0]), 4), round(float(center[1]), 4)],
            "radius": RADIUS,
            "traj_index": index,
            "traj_frac": round(index / max(len(traj) - 1, 1), 3),
            "min_dist_m": round(dist, 3),
            "why": "; ".join(reasons),
            "anchor_ids": [o.id for o in anchors],
            "anchor_why": "; ".join(anchor_why),
            "trace": trace,
        }
        # Widen only where the demo actually satisfies the constraint at that distance.
        # Stretching the radius to cover a constraint the run never meets would turn a
        # known failure into a permissive rule and hide it.
        if not is_avoid and dist <= MAX_ANCHOR_DIST:
            entry["radius"] = constraint_radius(dist)
            if entry["radius"] > RADIUS:
                entry["radius_source"] = (
                    f"widened from {RADIUS} m: the demo's own closest approach is "
                    f"{dist:.2f} m, so a tighter radius would fail the reference run"
                )
        if not is_avoid and dist > MAX_ANCHOR_DIST:
            problems.append(
                {
                    "clause": n,
                    "kind": clause["kind"],
                    "text": clause["text"],
                    "reason": f"grounded to {entry['label']} but the demo stays {dist:.2f} m away",
                    "candidates": rank_candidates(clause["text"], objects, traj),
                }
            )
            # A pinned clause is kept anyway: a human looked at this one and said what it
            # means, so the finding is a recorded disagreement with the demo, not grounds
            # for dropping a constraint the sentence plainly states.
            if pin is None:
                continue
        if is_avoid:
            # Built here, while it is still known which objects form which side of the
            # gap: object_ids alone is a flat list, and "between" needs the two sides.
            entry["polygon"] = avoid_polygon(clause["kind"], groups)
        else:
            cursor = index if clause["kind"] != "goal" else cursor
        resolved.append(entry)

    entry, order_problems = assemble(scene, qid, instruction, resolved, objects, traj)
    problems += order_problems
    # A clause that never grounded is as disqualifying as one that grounded wrongly, so
    # the flag is set once, after every check has had its say.
    entry["verified"] = not problems
    if review:
        entry["review_note"] = review
    return entry, problems


def rank_candidates(
    phrase: str, objects: dict[str, Obj], traj: Trajectory, top: int = 6, by_end: bool = False
) -> list[dict[str, Any]]:
    """What the demo suggests this clause meant, best first.

    Anything the phrase could name, scored by how close the recording gets to it -- or,
    for a goal, by how close it stops to it. A clause the solver could not settle is
    usually settled at a glance by looking at where the robot actually went.
    """
    named = {o.id for o in solver.match_class(strip_size_words(phrase), list(objects.values()))}
    rows = []
    for oid, obj in objects.items():
        if not navigable(obj):
            continue
        index, dist = traj.approach(obj.center)
        rows.append(
            {
                "object_id": oid,
                "label": obj.display,
                "region": obj.region,
                "matches_phrase": oid in named,
                "min_dist_m": round(dist, 3),
                "end_dist_m": round(traj.final_dist(obj.center), 3),
                "traj_index": index,
                "traj_frac": round(index / max(len(traj) - 1, 1), 3),
            }
        )
    key = "end_dist_m" if by_end else "min_dist_m"
    rows.sort(key=lambda r: (not r["matches_phrase"], r[key]))
    return rows[:top]


def assemble(
    scene: str,
    qid: str,
    instruction: str,
    resolved: list[dict[str, Any]],
    objects: dict[str, Obj],
    traj: Trajectory,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn resolved clauses into the QA entry, checking the demo agrees on the order."""
    problems: list[dict[str, Any]] = []
    avoids = [c for c in resolved if c["kind"].startswith("avoid")]
    goal = next((c for c in resolved if c["kind"] == "goal"), None)
    passes = [c for c in resolved if c["kind"].startswith("pass")]

    if goal is None:
        problems.append(
            {
                "clause": -1,
                "kind": "goal",
                "text": instruction,
                "reason": "no clause grounded to a goal, so there is nothing to end on",
                "candidates": rank_candidates(instruction, objects, traj, by_end=True),
            }
        )

    indices = [c["traj_index"] for c in passes]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        problems.append(
            {
                "clause": -1,
                "kind": "order",
                "text": instruction,
                "reason": "the demo visits the grounded constraints out of the stated order: "
                + " -> ".join(f"{c['label']}@{c['traj_index']}" for c in passes),
                "candidates": [],
            }
        )

    for zone in avoids:
        if any(point_in_poly(p, zone["polygon"]) for p in traj.xy):
            problems.append(
                {
                    "clause": zone["clause"],
                    "kind": zone["kind"],
                    "text": zone["phrase"],
                    "reason": f"the demo enters the avoid zone built for {zone['label']}",
                    "candidates": [],
                }
            )

    used = object_roles(resolved)
    n_constraints = len(passes) + (1 if goal else 0)
    entry = {
        "id": qid,
        "question": instruction,
        "source": "official",
        "type": "instruction_following",
        "difficulty": difficulty(n_constraints, len(avoids)),
        "target_objects": object_nouns([objects[i].display for i in used]),
        "gt": {
            "pass_near": [scorable(c) for c in passes],
            "avoid": [scorable_zone(c) for c in avoids],
            "goal": scorable(goal) if goal else None,
        },
        "objects": {
            i: {**answer_payload(objects[i]), "role": role} for i, role in used.items()
        },
        "reference_trajectory": {
            "ply": str(traj.path.relative_to(REPO)),
            "n_vertices": len(traj),
            "length_m": round(traj.length, 3),
            "z": traj.z,
            "start": [round(float(traj.xy[0][0]), 3), round(float(traj.xy[0][1]), 3)],
            "end": [round(float(traj.xy[-1][0]), 3), round(float(traj.xy[-1][1]), 3)],
            "polyline": traj.polyline(),
        },
        "evidence": evidence(passes, goal, avoids, traj),
        "solver_trace": [line for c in resolved for line in c.get("trace", [])],
        "verified": not problems,
    }
    return entry, problems


def object_roles(resolved: list[dict[str, Any]]) -> dict[str, str]:
    """Every object the question references, in the order the sentence names them.

    ``constraint`` is a place the robot is scored on reaching; ``reference`` is a thing it
    only has to find in order to know which place that is. An object that is both -- one
    clause's landmark and another's destination -- is a constraint.
    """
    roles: dict[str, str] = {}
    for c in resolved:
        for oid in c["object_ids"]:
            roles[oid] = "constraint"
        for oid in c.get("anchor_ids", []):
            roles.setdefault(oid, "reference")
    return roles


def scorable(c: dict[str, Any]) -> dict[str, Any]:
    out = {
        "kind": c["kind"],
        "phrase": c["phrase"],
        "label": c["label"],
        "object_ids": c["object_ids"],
        "anchor_ids": c["anchor_ids"],
        "center": c["center"],
        "radius": c["radius"],
        "traj_index": c["traj_index"],
        "traj_frac": c["traj_frac"],
        "min_dist_m": c["min_dist_m"],
        "why": c["why"],
    }
    if c["anchor_why"]:
        out["anchor_why"] = c["anchor_why"]
    if "radius_source" in c:
        out["radius_source"] = c["radius_source"]
    return out


def scorable_zone(c: dict[str, Any]) -> dict[str, Any]:
    out = {
        "kind": c["kind"],
        "phrase": c["phrase"],
        "label": c["label"],
        "object_ids": c["object_ids"],
        "anchor_ids": c["anchor_ids"],
        "polygon": c["polygon"],
        "why": c["why"],
    }
    if c["anchor_why"]:
        out["anchor_why"] = c["anchor_why"]
    return out


def difficulty(n_constraints: int, n_avoid: int) -> str:
    if n_avoid:
        return "hard"
    return "easy" if n_constraints <= 2 else "medium"


def evidence(passes, goal, avoids, traj: Trajectory) -> str:
    legs = [
        f"{c['label']} at {c['min_dist_m']:.2f} m ({c['traj_frac']:.0%} along)" for c in passes
    ]
    text = "Demo path passes " + ", then ".join(legs) if legs else "Demo path runs"
    if goal:
        text += f", and ends {goal['min_dist_m']:.2f} m from {goal['label']}"
    if avoids:
        text += "; it never enters the zone for " + ", ".join(c["label"] for c in avoids)
    return text + "."


# ---------------------------------------------------------------- scene / driver


def build_scene(scene: str, instructions: list[str], overrides: dict[str, Any]) -> dict[str, Any]:
    objects = load_objects(scene)
    questions, problems = [], {}
    for n, instruction in enumerate(instructions):
        qid = f"Q0{n + 4}"  # the official numbering: Q1 numerical, Q2/Q3 object reference
        traj = Trajectory(trajectory_path(scene, qid))
        entry, issues = build_question(scene, qid, instruction, objects, traj, overrides)
        questions.append(entry)
        if issues:
            problems[qid] = issues

    doc = {
        "scene": scene,
        "category": 3,
        "category_name": "instruction_following",
        "description": (
            "Category 3 (instruction following) evaluation questions, verbatim from "
            "questions/<scene>/questions.pdf. Ground truth is the ordered constraint set "
            "the organizers' own demonstration satisfies: each constraint names IRef-VLA "
            "objects and the map-frame point the robot must come within radius metres of, "
            "in order, ending at goal. Derived and audited by "
            "scripts/bench/generate_category3_qa.py and scripts/eval/verify_category3.py."
        ),
        "notes": [
            "gt.pass_near is ORDERED and excludes the goal; gt.goal is scored on the "
            "final pose, gt.avoid on the whole path. The block is shaped for "
            "scripts/eval/score.py::score_instruction, which reads center/radius/polygon "
            "and ignores the diagnostic keys beside them.",
            "traj_index / traj_frac / min_dist_m record where the organizers' demo "
            "satisfies each constraint. They are the evidence for the grounding, not "
            "something a system under test has to reproduce.",
            "The trajectory .ply and the IRef-VLA boxes share one map frame: demos start "
            "at the robot spawn (0, 0) and run at z = 0.75, the simulator's vehicleHeight. "
            "Compare (x, y) directly; z = 0.75 is the sensor plane, not the floor.",
            "reference_trajectory.polyline is the demo resampled to ~0.25 m. It is a "
            "recording of a correct executed path, not the sparse waypoint sequence a "
            "system publishes on /way_point_with_heading.",
            "objects holds every referenced IRef-VLA box so the file is self-contained, "
            "in the order the instruction names them, each tagged role: constraint (a "
            "place the robot is scored on reaching) or reference (a thing it must find "
            "only to know which place that is -- the picture in 'the stool under the "
            "picture'). target_objects lists the SAM3 nouns for both: a system that never "
            "detects the reference objects cannot ground the constraint ones.",
            "Each constraint's anchor_ids names its own reference objects and anchor_why "
            "says how each was chosen. They are documentation of the grounding, not "
            "geometry: no anchor moves a centre, a radius or a polygon.",
        ],
        "difficulty_counts": tally(q["difficulty"] for q in questions),
        "target_object_coverage": sorted({t for q in questions for t in q["target_objects"]}),
        "questions": questions,
    }
    return {"doc": doc, "problems": problems}


def tally(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return {k: out[k] for k in ("easy", "medium", "hard") if k in out}


def report(scene: str, built: dict[str, Any], verbose: bool) -> None:
    for q in built["doc"]["questions"]:
        gt = q["gt"]
        mark = "ok " if q["verified"] else "REVIEW"
        legs = [f"{c['label']}@{c['traj_frac']:.0%}/{c['min_dist_m']:.2f}m" for c in gt["pass_near"]]
        goal = gt["goal"]
        legs.append(
            f"GOAL {goal['label']}@{goal['traj_frac']:.0%}/{goal['min_dist_m']:.2f}m"
            if goal
            else "GOAL ???"
        )
        for z in gt["avoid"]:
            legs.append(f"AVOID {z['label']}")
        print(f"  {q['id']} {mark:<7}{'  ->  '.join(legs)}")
        if verbose:
            print(f"        {q['question']}")
            print(f"        nouns: {', '.join(q['target_objects'])}")
            for c in list(gt["pass_near"]) + ([goal] if goal else []) + gt["avoid"]:
                refs = ", ".join(
                    f"{q['objects'][i]['label']}#{i}" for i in c.get("anchor_ids", [])
                )
                print(
                    f"        {c['kind']:<14}{c['label']}#{'/'.join(c['object_ids'])}"
                    + (f"  <- {refs}" if refs else "")
                )
    for qid, issues in built["problems"].items():
        for p in issues:
            print(f"  {qid} !! [{p['kind']}] {p['reason']}")
            for c in p["candidates"][:4]:
                star = "*" if c["matches_phrase"] else " "
                print(
                    f"         {star} {c['object_id']:>4} {c['label']:<24}"
                    f" d={c['min_dist_m']:.2f} t={c['traj_frac']:.0%}"
                )


def plot_scene(scene: str, built: dict[str, Any], objects: dict[str, Obj], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    qs = built["doc"]["questions"]
    fig, axes = plt.subplots(1, len(qs), figsize=(9 * len(qs), 8))
    axes = np.atleast_1d(axes)
    for ax, q in zip(axes, qs):
        traj = Trajectory(REPO / q["reference_trajectory"]["ply"])
        for obj in objects.values():
            fp = footprint(obj)
            ax.add_patch(Polygon(fp, closed=True, fc="0.92", ec="0.75", lw=0.5, zorder=1))
        ax.plot(traj.xy[:, 0], traj.xy[:, 1], "-", color="#1f77b4", lw=2, zorder=4)
        ax.plot(*traj.xy[0], "o", color="green", ms=10, zorder=6)
        ax.plot(*traj.xy[-1], "*", color="red", ms=18, zorder=6)
        cons = list(q["gt"]["pass_near"]) + ([q["gt"]["goal"]] if q["gt"]["goal"] else [])
        for n, c in enumerate(cons, 1):
            ax.add_patch(
                Circle(c["center"], c["radius"], fc="none", ec="#d62728", ls="--", lw=1.2, zorder=5)
            )
            ax.annotate(
                f"{n}. {c['label']}",
                c["center"],
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                zorder=7,
            )
            for oid in c["object_ids"]:
                ax.add_patch(
                    Polygon(footprint(objects[oid]), closed=True, fc="#ffbb78", ec="#ff7f0e", zorder=3)
                )
        for z in q["gt"]["avoid"]:
            ax.add_patch(
                Polygon(z["polygon"], closed=True, fc="#d62728", alpha=0.18, ec="#d62728", zorder=2)
            )
        ax.set_aspect("equal")
        ax.set_title(f"{scene} {q['id']}\n{q['question']}", fontsize=9, wrap=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", action="append", help="limit to this scene (repeatable)")
    ap.add_argument("--write", action="store_true", help="write the QA files")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--plot", metavar="SCENE", help="write a top-down PNG for one scene")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    catalog = json.loads((QUESTIONS / "questions.json").read_text())
    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
    wanted = set(args.scene or [])
    if args.plot:
        wanted = {args.plot}

    total, flagged = 0, 0
    for scene_entry in catalog:
        scene = scene_entry["scene"]
        if wanted and scene not in wanted:
            continue
        instructions = scene_entry["questions"]["instruction_following"]
        built = build_scene(scene, instructions, overrides)
        total += len(built["doc"]["questions"])
        flagged += sum(1 for q in built["doc"]["questions"] if not q["verified"])
        print(f"[{scene}]")
        report(scene, built, args.verbose)

        CANDIDATES.mkdir(parents=True, exist_ok=True)
        candidates_file = CANDIDATES / f"{scene}.json"
        if built["problems"]:
            candidates_file.write_text(json.dumps(built["problems"], indent=2) + "\n")
        elif candidates_file.exists():
            candidates_file.unlink()

        if args.plot:
            out = CANDIDATES / f"{scene}.png"
            plot_scene(scene, built, load_objects(scene), out)
            print(f"  plot -> {out}")
        if args.write:
            out = BENCH / scene / "category_3" / f"{scene}_category3_qa.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(built["doc"], indent=2) + "\n")
            print(f"  wrote {out.relative_to(REPO)}")

    print(f"\n{total - flagged}/{total} questions fully grounded, {flagged} need review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
