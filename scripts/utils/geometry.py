#!/usr/bin/env python3
"""IRef-VLA object boxes and the spatial predicates that read them.

Shared by ``utils.text_solver``, the category-2/3 generators under ``scripts/bench/``, and
``scripts/eval/verify_category{2,3}.py``, so a constraint is generated, resolved and
audited against one definition of "closest", "on" and "between" rather than several.

Two decisions here shape everything downstream:

* **Every comparison is run under two metrics** — centre-to-centre and box-gap. The
  challenge's overlap function is unpublished, and the two metrics disagree about which
  object is closest whenever a large object is involved. A grounding is only trusted when
  both agree, so the answer does not depend on which one the grader picked.
* **Ambiguity is checked at two class granularities** — the raw IRef label and the
  coarser NYU label. Uniqueness over raw labels alone lets "the table closest to the
  sofa" through in a room that also holds a "tea table".
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Scene recordings and the IRef-VLA metadata beside them. Every reader takes this as a
# default it can override, because the ai_module container mounts the recordings at
# /data/bags and never sees the repo root.
BAGS = REPO / "data" / "bags"
# home_building_1 and home_building_2 have no recording, so no metadata was bundled beside
# one -- see the note at the foot of data/bags/scenes.yaml. Their annotations only exist in
# the dataset checkout next to the repo. Where both roots hold a scene the files are
# byte-identical, so this is a fallback, not a second source of truth. Same two-root union
# scripts/eval/build_dimension_priors.py already walks.
DATASET = REPO.parent / "IRef-VLA" / "unity" / "Unity"

SCENE_NAME_RE = re.compile(r"^[a-z0-9_]+$")

# The answer must beat its runner-up by this much (metres) under every metric.
MIN_MARGIN = 0.30
# "near" is absolute, not comparative: the target must be this close ...
NEAR_MAX = 1.50
# ... and every same-class competitor at least this far, or "near" is ambiguous.
NEAR_CLEAR = 2.50
# Wall-mounted objects hang a few centimetres off the wall behind the furniture in
# front of them, so footprints are dilated before above/below tests.
FOOTPRINT_TOL = 0.15
REST_TOL = 0.10
# An object resting on another sits this far up the support's own height; anything lower is
# standing beside it, inside its box. Supports flatter than FLAT_SUPPORT are exempt: on a
# 2 cm tray, the tray and everything on it share a base height to within the box precision.
REST_FRACTION = 0.30
FLAT_SUPPORT = 0.10
# A landmark this much smaller than the thing it locates is not a landmark. It is what
# separates "the picture above the suitcase" from "the window above the book" — both are
# a wall object over a floor object, with the same footprints and the same clearance, and
# only the sizes say which one a person would say out loud.
LANDMARK_VOLUME_RATIO = 0.3
# "in" is containment, not support: the inner object's base has to sit this far below the
# outer's rim, and the outer needs enough depth for "inside" to mean anything (a tray is
# 2 cm deep — things are *on* it).
INSIDE_DEPTH = 0.10
INSIDE_MIN_HEIGHT = 0.15
# "between" needs the target inside the middle stretch of the anchor-anchor segment,
# not merely somewhere on the infinite line through them.
BETWEEN_SPAN = (0.15, 0.85)
# Lateral slack scales with how far apart the anchors are, but never past a metre: with
# two door frames and four windows in a room, a purely proportional corridor calls four
# different wall lamps "between a door frame and a window".
BETWEEN_LATERAL = 0.35  # as a fraction of the anchor separation
BETWEEN_LATERAL_MAX = 1.00  # metres
# Betweenness also has to be *visible*. Two anchors 6 m apart with the target dead on the
# line between them is geometrically perfect and reads as nonsense ("the table between the
# shoes and the jar"), so the anchors must be near each other and near the target.
BETWEEN_MAX_SPAN = 4.50
BETWEEN_MAX_ANCHOR_GAP = 3.00

# Classes that make poor answers, though they stay usable as anchors when unique.
# Either there is no meaningful box to point at (wall, floor), or the scene holds a grid
# of visually identical instances that no image-space grounding can tell apart — twenty
# recessed downlights are unique only in the metadata.
STRUCTURAL = {
    "wall",
    "walls",
    "exterior walls",
    "interior walls",
    "bathroom walls",
    "glass wall",
    "partition wall",
    "floor",
    "ceiling",
    "tatami",
    "door",
    "door frame",
    "window frame",
    "air vent",
    "light switch",
    "handle",
    "focus light",
    "spot light",
    # Architecture, not furniture: a column is a featureless repeat and its box is flush
    # with the wall, and "the vase inside the stairs" is what a stairwell box invites.
    "column",
    "stairs",
    # Cables have long thin boxes that swallow whatever they run past, so they support and
    # sit above things they merely lie beside.
    "hookah wire",
    "cable",
    "wire",
    "cord",
    "unknown",
}

# Room-sized datums. "Farthest from the tatami" is not a question anyone can answer,
# because the tatami is the whole floor -- and a wall is the same thing stood upright: its
# box is the room's perimeter, so "the cup between the lamp and the wall" holds for anything
# near any wall. Door and window frames are deliberately NOT here: they are room-sized
# datums' opposite, a specific spot on a wall, and are how the organizers write questions.
VACUOUS_ANCHORS = {
    "floor", "tatami", "ceiling",
    "wall", "walls", "exterior walls", "interior walls", "bathroom walls", "glass wall",
    "partition wall",
}
# Additionally vacuous for the vertical relations: everything in the room is above the
# carpet, though "near the carpet" is still a real constraint.
SURFACE_ANCHORS = VACUOUS_ANCHORS | {"carpet"}
# Never a landmark, whatever the relation.
#   Cables: the box is a long thin diagonal that runs past half the objects on the table, so
#   it is neither where it looks nor worth naming.
#   Cameras: "the chair closest to the camera" reads as the robot's own camera, and a 13 cm
#   photo camera on a shelf is not what anyone would take it to mean.
BAD_LANDMARKS = {
    "hookah wire",
    "cable",
    "wire",
    "cord",
    "power cord",
    "camera",
    "security camera",
}

# Class folding, same spirit (and mostly the same entries) as scripts/eval/score_map3d.py:
# every wrong entry here merges two genuinely different classes and lets an ambiguous
# question through, so keep it short.
SYNONYM_GROUPS = [
    {"sofa", "couch"},
    {"picture", "painting", "photo", "poster", "framed record", "framed records"},
    {"carpet", "rug"},
    {"potted plant", "plant"},
    {"nightstand", "night stand", "bedside table"},
    {"tv", "television"},
    {"book", "books"},
    {"curtain", "curtains"},
    {"window", "windows"},
    {"lamp", "light"},
    {"trash can", "trash bin", "garbage can"},
]
CLASS_ALIAS: dict[str, str] = {}
for _group in SYNONYM_GROUPS:
    _canon = sorted(_group)[0]
    for _name in _group:
        CLASS_ALIAS[_name] = _canon

COLORS = {
    "black", "gray", "grey", "white", "brown", "maroon", "olive", "pink", "red", "blue",
    "green", "yellow", "orange", "aqua", "beige", "tan", "purple", "cyan", "magenta",
    "gold", "silver", "navy", "teal", "lime", "fuchsia",
}

ARTICLES = {"a", "an", "the"}

# IRef's palette is CSS colour names fitted to mesh vertices, so a question's everyday
# colour word and the stored label can differ on the same object: every pillow in
# japanese_room is "maroon", and the official question calls one of them red.
COLOR_ALIAS = {
    "red": {"red", "maroon"},
    "maroon": {"maroon", "red"},
    "gray": {"gray", "grey", "silver"},
    "grey": {"grey", "gray", "silver"},
    "beige": {"beige", "tan"},
    "tan": {"tan", "beige"},
    "green": {"green", "olive", "lime"},
    "purple": {"purple", "fuchsia", "magenta"},
    "cyan": {"cyan", "aqua", "teal"},
    "aqua": {"aqua", "cyan", "teal"},
}


def color_matches(wanted: str, colors: list[str]) -> bool:
    accepted = COLOR_ALIAS.get(wanted, {wanted})
    return any(c in accepted for c in colors)


def norm_class(name: str) -> str:
    """Canonical class key: lowercased, whitespace-collapsed, synonyms folded."""
    key = " ".join(str(name or "").lower().split())
    return CLASS_ALIAS.get(key, key)


class Obj:
    """One IRef-VLA object: an oriented box with labels and a colour palette."""

    __slots__ = (
        "id", "region", "raw_label", "nyu_label", "colors", "corners",
        "lo", "hi", "center", "size", "volume",
    )

    def __init__(self, oid: str, region: str, raw: dict[str, Any]):
        corners = np.asarray(raw["bbox"], dtype=float)
        self.id = str(oid)
        self.region = str(region)
        self.raw_label = " ".join(str(raw.get("raw_label") or "").lower().split())
        self.nyu_label = " ".join(str(raw.get("nyu_label") or "").lower().split())
        self.colors = [str(c).lower() for c in (raw.get("color_labels") or []) if c and c != "N/A"]
        self.corners = corners
        self.lo = corners.min(axis=0)
        self.hi = corners.max(axis=0)
        center = raw.get("center")
        self.center = np.asarray(center, dtype=float) if center else corners.mean(axis=0)
        size = raw.get("size")
        self.size = np.asarray(size, dtype=float) if size else (self.hi - self.lo)
        self.volume = float(raw.get("volume") or float(np.prod(self.size)))

    @property
    def display(self) -> str:
        if self.raw_label and self.raw_label != "unknown":
            return self.raw_label
        return self.nyu_label or "object"

    @property
    def fine(self) -> str:
        return norm_class(self.raw_label or self.nyu_label)

    @property
    def coarse(self) -> str:
        return norm_class(self.nyu_label or self.raw_label)

    @property
    def structural(self) -> bool:
        return self.fine in STRUCTURAL or self.coarse in STRUCTURAL

    @property
    def yaw(self) -> float:
        """Heading of the box's first horizontal edge, for a CUBE marker's pose."""
        edge = self.corners[1][:2] - self.corners[0][:2]
        return float(math.atan2(edge[1], edge[0]))

    def __repr__(self) -> str:
        return f"{self.display}#{self.id}"


# ---------------------------------------------------------------- loading


def _resolve_under(root: Path, *parts: str, scene: str) -> Path:
    """Join under root, refusing traversal out of it."""
    resolved = root.resolve()
    path = resolved.joinpath(*parts).resolve()
    if not str(path).startswith(str(resolved)):
        raise ValueError(f"scene path escapes {resolved}: {scene!r}")
    return path


def metadata_path(scene: str, suffix: str, bags: Path | None = None) -> Path:
    """Resolve <scene>_<suffix> from the bags metadata, or the dataset checkout.

    Prefers <bags>/<scene>/iref_vla_metadata/. Falls back to the IRef-VLA checkout beside
    the repo for the scenes that were never recorded; the returned path is the bags one
    when neither exists, so the caller's error names the location it is meant to have.
    """
    if not SCENE_NAME_RE.match(scene):
        raise ValueError(f"invalid scene name: {scene!r}")
    bagged = _resolve_under(
        Path(bags or BAGS), scene, "iref_vla_metadata", f"{scene}_{suffix}", scene=scene
    )
    if bagged.exists() or bags is not None:
        return bagged
    dataset = _resolve_under(DATASET, scene, f"{scene}_{suffix}", scene=scene)
    return dataset if dataset.exists() else bagged


def load_objects(scene: str, bags: Path | None = None) -> dict[str, Obj]:
    """Every annotated object in the scene, keyed by id.

    ``object_data.json`` is the newer of the two per-object builds IRef-VLA ships and the
    one the rest of the repo scores against (``scripts/eval/score_map3d.py``,
    ``build_dimension_priors.py``). It carries the same boxes as ``objects.json`` -- same
    centres, sizes and corner sets -- but a different corner *order*, which is what
    ``Obj.yaw`` reads, and a re-fitted colour palette. Preferring it keeps a box's
    orientation the same here as it is in the map scorer.
    """
    path = metadata_path(scene, "object_data.json", bags)
    if not path.exists():
        path = metadata_path(scene, "objects.json", bags)
    raw = json.loads(path.read_text())
    return {
        str(oid): Obj(oid, region, obj)
        for region, entries in raw.items()
        for oid, obj in entries.items()
    }


def load_statements(scene: str, bags: Path | None = None) -> dict[str, Any]:
    return json.loads(metadata_path(scene, "referential_statements.json", bags).read_text())


def region_objects(objects: dict[str, Obj], region: str) -> list[Obj]:
    return [o for o in objects.values() if o.region == str(region)]


# ---------------------------------------------------------------- distances


def center_dist(a: Obj, b: Obj) -> float:
    return float(np.linalg.norm(a.center - b.center))


def gap_dist(a: Obj, b: Obj) -> float:
    """Axis-aligned separation between two boxes; 0 when they overlap."""
    gaps = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(gaps))


METRICS = {"centre": center_dist, "box-gap": gap_dist}


def set_dist(obj: Obj, anchors: list[Obj], metric) -> float:
    return min(metric(obj, a) for a in anchors)


# ---------------------------------------------------------------- predicates


def footprints_overlap(a: Obj, b: Obj, tol: float = FOOTPRINT_TOL) -> bool:
    return bool(
        a.lo[0] - tol <= b.hi[0]
        and b.lo[0] - tol <= a.hi[0]
        and a.lo[1] - tol <= b.hi[1]
        and b.lo[1] - tol <= a.hi[1]
    )


def footprint_area(obj: Obj) -> float:
    return float((obj.hi[0] - obj.lo[0]) * (obj.hi[1] - obj.lo[1]))


def vertical_gap(a: Obj, b: Obj) -> float:
    """Clear vertical space between two boxes; 0 when their z-ranges overlap.

    Symmetric, so it doubles as the distance metric for "furthest from the floor" —
    see ``utils.text_solver.metrics_for``.
    """
    return float(max(a.lo[2] - b.hi[2], b.lo[2] - a.hi[2], 0.0))


def center_inside(inner: Obj, outer: Obj, tol: float = FOOTPRINT_TOL) -> bool:
    return bool(
        outer.lo[0] - tol <= inner.center[0] <= outer.hi[0] + tol
        and outer.lo[1] - tol <= inner.center[1] <= outer.hi[1] + tol
    )


def rests_on(item: Obj, support: Obj) -> bool:
    """Is the item held up by the support — "on" as a person would say it.

    Not "its underside touches the support's top plane": a pillow *on* a sofa has its
    underside at seat height, well below the top of the backrest, and a strict plane test
    rejects every pillow in every scene. The item's base has to sit inside the support's
    footprint, no lower than the support's own base and no higher than its top.

    Support is asymmetric and the boxes alone do not say which way round it is: a glass and
    the tray under it have overlapping z-ranges, so the height test passes both ways. Two
    further conditions pin the direction down.

    The support has to have the bigger footprint — that alone rules out "the tray on the
    glass". Volume is deliberately not compared: a spray of flowers has a bigger box than
    the ledge holding it up, and the ledge is still what it stands on.

    And the item has to sit in the support's upper reaches rather than merely inside its
    box, which is what separates a glass on a tray from a glass standing next to the hookah
    whose wire box it happens to fall inside. The threshold is a fraction of the support's
    height rather than its top, because a bed's box is 1.74 m tall thanks to the headboard
    and a sofa's includes the backrest: the pillow on either one sits nowhere near the top.
    Flat supports are exempt, since on a 2 cm tray everything shares one base height.
    """
    if not center_inside(item, support, tol=0.05):
        return False
    if not (support.lo[2] - REST_TOL <= item.lo[2] <= support.hi[2] + REST_TOL):
        return False
    if footprint_area(support) < footprint_area(item):
        return False
    if support.size[2] <= FLAT_SUPPORT:
        return True
    return bool(item.lo[2] - support.lo[2] >= REST_FRACTION * support.size[2])


def is_above(upper: Obj, lower: Obj) -> bool:
    """One object clear of another, over its footprint.

    The tolerance is not slack: a picture hangs a couple of centimetres off the wall and the
    furniture below it stops short of the skirting, so the true footprints of a picture and
    the suitcase under it do not intersect at all. Requiring a real intersection here loses
    every wall-mounted object, including the one the organizers ask about in hotel_room_1.
    """
    return footprints_overlap(upper, lower) and upper.lo[2] >= lower.hi[2]


def is_inside(inner: Obj, outer: Obj) -> bool:
    """Containment: the inner object is down inside the outer one, not sitting on it.

    The flower *in* the vase has its stems well above the rim, so this cannot ask for full
    enclosure; what separates "in" from "on" is that the inner object's base is below the
    outer's rim rather than level with it.
    """
    if outer.size[2] < INSIDE_MIN_HEIGHT or inner.volume > 0.5 * outer.volume:
        return False
    for axis in (0, 1):
        if not (outer.lo[axis] - 0.05 <= inner.lo[axis] and inner.hi[axis] <= outer.hi[axis] + 0.05):
            return False
    return bool(
        inner.lo[2] <= outer.hi[2] - INSIDE_DEPTH and inner.lo[2] >= outer.lo[2] - REST_TOL
    )


def between_holds(target: Obj, a1: Obj, a2: Obj) -> tuple[bool, float]:
    """True when the target sits in the corridor between two anchors.

    Returns (holds, lateral_offset). The projection parameter must land in the middle
    stretch of the segment, otherwise "between" degenerates into "somewhere on the line
    through both anchors", which is how many raw statements read.
    """
    p1, p2 = a1.center[:2], a2.center[:2]
    seg = p2 - p1
    span = float(np.linalg.norm(seg))
    if span < 1e-6 or span > BETWEEN_MAX_SPAN:
        return False, math.inf
    if max(gap_dist(target, a1), gap_dist(target, a2)) > BETWEEN_MAX_ANCHOR_GAP:
        return False, math.inf
    t = float(np.dot(target.center[:2] - p1, seg) / (span * span))
    lateral = float(np.linalg.norm(target.center[:2] - (p1 + t * seg)))
    allowed = min(BETWEEN_LATERAL * span, BETWEEN_LATERAL_MAX)
    holds = BETWEEN_SPAN[0] <= t <= BETWEEN_SPAN[1] and lateral <= allowed
    return holds, lateral


def unaskable(target: Obj, anchors: list[Obj], relation: str, region_objs: list[Obj]) -> str:
    """Why nobody would ask this question, or "" when it is fair game.

    Separate from the geometry: every rule here is about a claim that is *true* and still
    useless. Shared by the generator (which drops these candidates), the synthesiser and the
    verifier (which fails a benchmark question that violates them), so a claim mined from an
    utterance is held to the same standard as one derived from the boxes — being written
    down in IRef earns an utterance no credit.
    """
    # "Find the carpet below the eye glasses" is a support relation read backwards, and
    # "closest to the book" when the book sits on the target is proximity in costume: the
    # distance is zero either way and the question turns into a riddle.
    if relation not in ("on", "supports", "in") and any(
        rests_on(a, target) or rests_on(target, a) for a in anchors
    ):
        return f"{relation}: anchor rests on target"
    for anchor in anchors:
        if anchor.fine in BAD_LANDMARKS:
            return f"{relation}: {anchor} is never a landmark"
    vacuous = SURFACE_ANCHORS if relation in ("on", "above", "below", "between") else VACUOUS_ANCHORS
    if any(a.fine in vacuous for a in anchors):
        return f"{relation}: room-sized anchor"
    if relation in ("in", "above", "below", "between") and target.fine in SURFACE_ANCHORS:
        # Everything in the room is above the carpet and most of it is between two things
        # the carpet reaches under, so neither picks out one carpet from another.
        return f"{relation}: room-sized target"
    # A wall's box is the room's perimeter, so it "supports" and sits "above" everything
    # inside it. Structural classes stay usable for the relations that only need a
    # position -- "between a door frame and a window" is how the organizers write them.
    if relation in ("on", "in", "supports", "above", "below") and any(a.structural for a in anchors):
        return f"{relation}: structural anchor"
    if relation in ("above", "below") and any(
        a.volume < LANDMARK_VOLUME_RATIO * target.volume for a in anchors
    ):
        return f"{relation}: landmark far smaller than the target"
    if relation in ("above", "below"):
        for anchor in anchors:
            if middle := stacked_through(target, anchor, region_objs, relation == "above"):
                return f"{relation}: same stack, with {middle} between them"
    return ""


def stacked_through(target: Obj, anchor: Obj, region_objs: list[Obj], upward: bool) -> Obj | None:
    """The object in the middle when target and anchor are two levels of one stack.

    The flowers are in a vase that stands on the display ledge, so "the flowers above the
    display ledge" is true — and nobody says it, because the vase is right there to name.
    """
    def held_by(item: Obj, base: Obj) -> bool:
        return rests_on(item, base) or is_inside(item, base)

    for middle in region_objs:
        if middle.id in (target.id, anchor.id):
            continue
        pair = (held_by(target, middle), held_by(middle, anchor))
        if not upward:
            pair = (held_by(anchor, middle), held_by(middle, target))
        if all(pair):
            return middle
    return None


def relation_holds(target: Obj, anchors: list[Obj], relation: str) -> bool:
    """One object against its anchor(s), for the non-comparative relations."""
    if not anchors:
        return False
    if relation == "on":
        return any(rests_on(target, a) for a in anchors)
    if relation == "supports":
        return any(rests_on(a, target) for a in anchors)
    if relation == "in":
        return any(is_inside(target, a) for a in anchors)
    if relation == "above":
        return any(is_above(target, a) for a in anchors)
    if relation == "below":
        return any(is_above(a, target) for a in anchors)
    if relation == "near":
        return any(gap_dist(target, a) <= NEAR_MAX for a in anchors)
    if relation == "between":
        return any(
            between_holds(target, a1, a2)[0]
            for i, a1 in enumerate(anchors)
            for a2 in anchors[i + 1:]
        )
    raise ValueError(f"unsupported relation: {relation!r}")


# ---------------------------------------------------------------- uniqueness


def competitors(target: Obj, anchors: list[Obj], region_objs: list[Obj], coarse: bool) -> list[Obj]:
    """Objects a reader could confuse with the target when resolving the question."""
    exclude = {target.id} | {a.id for a in anchors}
    attr = "coarse" if coarse else "fine"
    key = getattr(target, attr)
    return [o for o in region_objs if o.id not in exclude and getattr(o, attr) == key]


def rank_holds(target: Obj, anchor: Obj, rivals: list[Obj], mode: str) -> tuple[bool, float, str]:
    """The target is the unique argmin (closest) / argmax (farthest), by MIN_MARGIN."""
    if not rivals:
        return False, 0.0, "no competitor of the same class"
    worst_margin = math.inf
    notes = []
    for name, metric in METRICS.items():
        d_target = metric(target, anchor)
        ranked = sorted(((metric(r, anchor), r) for r in rivals), key=lambda p: p[0])
        d_rival, rival = ranked[0] if mode == "closest" else ranked[-1]
        margin = (d_rival - d_target) if mode == "closest" else (d_target - d_rival)
        if margin < MIN_MARGIN:
            return (
                False,
                margin,
                f"{mode} by {name}: {target} {d_target:.2f} m vs {rival} {d_rival:.2f} m "
                f"(margin {margin:.2f} m < {MIN_MARGIN})",
            )
        worst_margin = min(worst_margin, margin)
        notes.append(f"{name} {d_target:.2f} m vs runner-up {rival} {d_rival:.2f} m")
    return True, worst_margin, "; ".join(notes)


def near_holds(target: Obj, anchor: Obj, rivals: list[Obj]) -> tuple[bool, float, str]:
    d_target = gap_dist(target, anchor)
    if d_target > NEAR_MAX:
        return False, 0.0, f"target {d_target:.2f} m from anchor, over NEAR_MAX {NEAR_MAX}"
    if not rivals:
        return False, 0.0, "no competitor of the same class"
    d_rival, rival = min(((gap_dist(r, anchor), r) for r in rivals), key=lambda p: p[0])
    if d_rival < NEAR_CLEAR:
        return (
            False,
            d_rival - d_target,
            f"{rival} is also {d_rival:.2f} m from the anchor (needs >= {NEAR_CLEAR} m)",
        )
    return (
        True,
        d_rival - d_target,
        f"box-gap {d_target:.2f} m; next {target.fine} is {d_rival:.2f} m away",
    )


def predicate_unique(
    target: Obj, anchors: list[Obj], rivals: list[Obj], relation: str
) -> tuple[bool, float, str]:
    """Uniqueness for on / above / below / between."""
    if relation == "between" and len(anchors) < 2:
        return False, 0.0, "between needs two anchors"
    if not relation_holds(target, anchors, relation):
        return False, 0.0, f"geometry does not support {relation} for {target}"
    others = [o for o in rivals if relation_holds(o, anchors, relation)]
    if others:
        return False, 0.0, f"{len(others)} other {target.fine}(s) also {relation}: {others[:3]}"

    if relation == "between":
        _holds, lateral = max(
            (between_holds(target, a1, a2) for i, a1 in enumerate(anchors) for a2 in anchors[i + 1:]),
            key=lambda p: p[0],
        )
        detail = f"between {anchors[0]} and {anchors[1]} (lateral offset {lateral:.2f} m)"
    elif relation == "on":
        detail = f"rests on {anchors[0]} (underside {target.lo[2]:.2f} m, top {anchors[0].hi[2]:.2f} m)"
    elif relation == "supports":
        held = [a for a in anchors if rests_on(a, target)]
        detail = f"holds {held[0]} (top {target.hi[2]:.2f} m, its underside {held[0].lo[2]:.2f} m)"
    elif relation == "in":
        detail = (
            f"inside {anchors[0]}: base {target.lo[2]:.2f} m sits "
            f"{anchors[0].hi[2] - target.lo[2]:.2f} m below its rim"
        )
    elif relation == "above":
        detail = (
            f"underside {target.lo[2]:.2f} m clears {anchors[0]} top {anchors[0].hi[2]:.2f} m "
            f"by {vertical_gap(target, anchors[0]):.2f} m"
        )
    elif relation == "below":
        detail = (
            f"top {target.hi[2]:.2f} m sits {vertical_gap(target, anchors[0]):.2f} m under "
            f"{anchors[0]} underside {anchors[0].lo[2]:.2f} m"
        )
    else:
        detail = relation
    return True, float(len(rivals)), detail


def verify(
    target: Obj, anchors: list[Obj], relation: str, region_objs: list[Obj]
) -> tuple[bool, float, str, int]:
    """Re-derive a statement's claim from the boxes. Returns (ok, margin, why, n_rivals).

    The claim has to survive both class granularities; see the module docstring.
    """
    fine_rivals = competitors(target, anchors, region_objs, coarse=False)
    coarse_rivals = competitors(target, anchors, region_objs, coarse=True)
    merged = {o.id: o for o in fine_rivals + coarse_rivals}
    n_rivals = len(fine_rivals)

    checks: list[tuple[bool, float, str]] = []
    for rivals in (fine_rivals, list(merged.values())):
        if relation in ("closest", "farthest"):
            checks.append(rank_holds(target, anchors[0], rivals, relation))
        elif relation == "near":
            checks.append(near_holds(target, anchors[0], rivals))
        else:
            checks.append(predicate_unique(target, anchors, rivals, relation))

    for ok, margin, why in checks:
        if not ok:
            return False, margin, why, n_rivals
    return True, min(c[1] for c in checks), checks[0][2], n_rivals


# ---------------------------------------------------------------- answer payload


def round_vec(values) -> list[float]:
    return [round(float(v), 4) for v in values]


def answer_payload(obj: Obj) -> dict[str, Any]:
    """The GT box in every shape a scorer might want it in."""
    return {
        "object_id": obj.id,
        "region": obj.region,
        "label": obj.display,
        "raw_label": obj.raw_label,
        "nyu_label": obj.nyu_label,
        "colors": obj.colors,
        "center": round_vec(obj.center),
        "size": round_vec(obj.size),
        "yaw": round(obj.yaw, 4),
        "aabb": {"min": round_vec(obj.lo), "max": round_vec(obj.hi)},
        "size_aabb": round_vec(obj.hi - obj.lo),
        "bbox_corners": [round_vec(c) for c in obj.corners],
        "volume": round(obj.volume, 5),
    }


def object_nouns(names: list[str]) -> list[str]:
    """SAM-3 prompt nouns: object names only, no colours, articles or duplicates."""
    out: list[str] = []
    for name in names:
        noun = " ".join(str(name or "").lower().split())
        noun = " ".join(w for w in noun.split() if w not in COLORS and w not in ARTICLES)
        if noun and noun not in out:
            out.append(noun)
    return out
