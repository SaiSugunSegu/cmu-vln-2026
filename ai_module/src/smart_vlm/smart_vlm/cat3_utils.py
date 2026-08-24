"""Pure helpers for the instruction-following (category-3) reasoner.

Everything the reasoner decides lives here so it can be decided without a robot: what the
model is shown, what its reply is worth, and what to do when the drive goes wrong. No ROS, no
cv2, no GPU.

Category 3 does its own spatial reasoning in the MODEL, not in a solver. There is deliberately
no candidate ranking, no relation predicate and no shared geometry module here:

  * the maps are small -- a scene runs to a couple of dozen rows, so the whole thing fits in
    one prompt and a shortlist would only hide objects;
  * the pruning a shortlist applies removes room-scale structure, and columns, stairs, windows
    and door frames are destinations in a third of the corpus;
  * the failures a solver cannot fix are the ones a picture can. In one measured run the object
    the command called a "stool" was labelled `table` by the mapper: no label match grounds
    that, and a model looking at the tagged crop does.

What the model cannot be trusted with is arithmetic on its own output, so `parse_route` checks
every coordinate against the map rows the model cited and replaces it when the two disagree.
That is the whole safety net: reasoning stays with the model, bookkeeping stays here.
"""
from __future__ import annotations

import json
import math
import re

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "ARRIVAL_RADIUS_M",
    "GOAL_RADIUS_M",
    "MAX_SNAP_M",
    "MAX_TRIES",
    "MIN_EXTENT_M",
    "OBSTACLE_CLEARANCE_M",
    "SETTLE_RADIUS_M",
    "SETTLE_S",
    "SNAP_M",
    "TRY_DURATION_S",
    "Waypoint",
    "fallback_route",
    "heuristic_targets",
    "map_table",
    "parse_route",
    "plan_payload",
    "route_summary",
    "snap_to_traversable",
    "synthetic_trajectory",
    "usable_objects",
]

#: The scoring circle. `scripts/eval/score.py::score_instruction` credits a constraint when the
#: trajectory comes within `radius` of its centre, defaulting to this. Ground truth widens it
#: for eight wall-mounted constraints; we cannot know which, so we always assume the tight one.
GOAL_RADIUS_M = 1.5

#: A map row whose box is below this on EVERY axis is a track that never accumulated geometry
#: -- map_node emits such a stub as a few-centimetre cube, and a live map carries one sitting at
#: exactly the origin, which is the robot's own spawn point. Offering those to the model invites
#: a waypoint at (0, 0). The threshold is deliberately just above that stub (5 cm) rather than
#: comfortably above it: a wrongly kept row costs a line of prompt, a wrongly dropped one costs
#: a constraint the robot can no longer reach at all. Flat detections that are large in x or y
#: -- floor smears -- are KEPT for the same reason; the prompt tells the model to ignore noise.
MIN_EXTENT_M = 0.06

#: How far a model-written coordinate may sit from the objects it says it is standing at before
#: we stop believing the number and use those objects' own centre instead. Generous on purpose:
#: a midpoint of two boxes is legitimately offset from either, and the check only has to catch
#: a fabricated or transposed coordinate.
SNAP_M = 1.0

#: Close enough to count as "there" -- the ONE distance the drive loop tests. It is the
#: scoring radius, deliberately: a leg that declares success outside the circle it is graded on
#: has not arrived in any sense that counts. An earlier 2.0 m was justified by "a 1.5 m
#: tolerance was met 0 times in 17 goal legs", but that 0 came from `closest_m` being sampled
#: before the settle; measured on where the robot actually stopped, 8 of those 17 were already
#: inside 1.5 m. With `snap_to_traversable` aiming at ground the robot can occupy, the rest are
#: reachable too.
SETTLE_RADIUS_M = GOAL_RADIUS_M

#: Once inside that radius, stop publishing and let the converter finish its own approach.
SETTLE_S = 5.0

#: One try: how long the waypoint is published before giving up on this attempt. Republishing
#: IS the retry -- every message resets the converter's arrival latch and makes it re-pick a
#: traversable point.
TRY_DURATION_S = 15.0

#: Tries per waypoint before the route moves on. Three, because more buys nothing: a measured
#: wedged goal spent 149 s and 13 recovery attempts while its distance moved 6 cm.
MAX_TRIES = 3

#: Close enough to the point we PUBLISHED to call the approach finished. The converter latches
#: "waypoint reached" and cuts speed at `waypointXYRadius = 0.3` (waypoint_converter.launch),
#: so 0.3 m is the floor and this is that plus margin. It is the second of the drive loop's two
#: arrival tests, and it exists because the first one cannot always be won: the distance to the
#: MODEL's point is bounded below by `snap_m + 0.3`, and a leg whose snap moved 1.4 m can never
#: reach a 1.5 m circle no matter how well it drives. One measured leg burned three tries and
#: 45.7 s that way, having arrived 0.28 m from where it was aimed on the first.
ARRIVAL_RADIUS_M = 0.5

#: Extra erosion applied to the traversable area before the snap, in metres. OFF by default:
#: the aim is the point closest to what the model asked for, and every metre of clearance is a
#: metre given away. The base autonomy applies its own 0.75 m (`obstacleDisThre`,
#: waypointConverter.cpp:217-224) when it decides where to actually drive; reproducing that here
#: only moved our published point 0.94-1.41 m off the waypoint on a measured run. Kept as a knob
#: so it can be dialled back up from config without a code change.
OBSTACLE_CLEARANCE_M = 0.0

#: How far the snap may move a waypoint. Beyond this the model's point is not "slightly inside
#: the furniture", it is somewhere we have no traversable reading for, and moving it that far
#: would aim at a different place than the one that was reasoned about.
MAX_SNAP_M = 3.0

ROLES = ("pass", "goal")

# Function words only -- object nouns (window, table, fridge) stay as SAM prompts. Relation
# words are stripped because this list arms a DETECTOR, which has no use for "closest"; the
# route call gets the sentence itself and does its own reading of them.
_STOP = frozenset({
    "take", "the", "path", "to", "and", "go", "near", "avoid", "from", "then",
    "stop", "at", "a", "an", "of", "that", "which", "with", "without", "by",
    "past", "around", "walk", "drive", "follow", "through", "into", "onto",
    "closest", "farthest", "between", "under", "above", "below", "on", "in",
    "for", "your", "you", "please", "toward", "towards",
})


@dataclass(frozen=True)
class Waypoint:
    """One place on the route, ready to publish as a Pose2D.

    Every waypoint here was chosen by the model. Nothing is interpolated, offset or inserted:
    the first sim run wedged for 30 s on a hop we invented between two of these, while every
    waypoint the model actually named was reached without incident.

    `reach_m` is the ONE distance the drive loop tests: come within it and the leg is done.
    It is not the distance the constraint is scored on -- see SETTLE_RADIUS_M for why those
    cannot be the same number.
    """

    x: float
    y: float
    role: str                                   # pass | goal
    object_ids: list[str] = field(default_factory=list)
    why: str = ""
    reach_m: float = SETTLE_RADIUS_M

    @property
    def scored(self) -> bool:
        return self.role in ROLES

    def as_dict(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "role": self.role,
                "object_ids": list(self.object_ids), "why": self.why,
                "reach_m": round(self.reach_m, 2)}


# ---------------------------------------------------------------- the map


def _center(entry: Any) -> Optional[tuple[float, float, float]]:
    box = ((entry or {}).get("bbox3d") or {})
    center = box.get("center") or []
    if len(center) < 3:
        return None
    try:
        values = tuple(float(v) for v in center[:3])
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(v) for v in values) else None


def _extent(entry: Any) -> Optional[tuple[float, float, float]]:
    box = ((entry or {}).get("bbox3d") or {})
    extent = box.get("extent") or []
    if len(extent) < 3:
        return None
    try:
        values = tuple(abs(float(v)) for v in extent[:3])
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(v) for v in values) else None


def usable_objects(raw_map: Any) -> dict[str, dict]:
    """Map rows that are real objects, keyed by id as a string.

    Two things are dropped: a row with no usable box at all, and a row whose box is degenerate
    on every axis. The second matters -- a live map carries entries like a 5 cm `column` at
    exactly the origin, which is a track that never accumulated geometry. Offering those to the
    model invites a waypoint at (0, 0), and the robot drives to its own spawn point.
    """
    out: dict[str, dict] = {}
    for key, entry in (raw_map or {}).items():
        if not isinstance(entry, dict):
            continue
        center, extent = _center(entry), _extent(entry)
        if center is None or extent is None:
            continue
        if max(extent) < MIN_EXTENT_M:
            continue
        out[str(key)] = entry
    return out


def object_xy(entry: Any) -> Optional[tuple[float, float]]:
    center = _center(entry)
    return None if center is None else (center[0], center[1])


def map_table(objects: dict[str, dict], robot_xy: Optional[Sequence[float]] = None) -> str:
    """The whole map as the model sees it: one line per object, nothing ranked or hidden.

    Deliberately free of derived facts. Working out which table is farthest from the columns is
    the job we are paying the model for; handing it the answer would make the images pointless
    and would put a solver back in the loop under another name.
    """
    lines = []
    if robot_xy is not None and len(robot_xy) >= 2:
        lines.append(f"Robot at ({float(robot_xy[0]):.2f}, {float(robot_xy[1]):.2f}).")
        lines.append("")
    lines.append("id | label | centre (x, y, z) m | size (w, d, h) m")
    for key in sorted(objects, key=lambda k: (len(k), k)):
        entry = objects[key]
        c, e = _center(entry), _extent(entry)
        if c is None or e is None:
            continue
        label = str(entry.get("label") or "object")
        lines.append(f"{key} | {label} | ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                     f"| ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f})")
    return "\n".join(lines)


# ---------------------------------------------------------------- the reply


def _cited_center(ids: Sequence[str], objects: dict[str, dict]
                  ) -> tuple[Optional[tuple[float, float]], list[str]]:
    """Mean centre of the map rows a waypoint cites, and the subset that actually exists."""
    known, points = [], []
    for oid in ids:
        entry = objects.get(str(oid))
        xy = object_xy(entry) if entry is not None else None
        if xy is None:
            continue
        known.append(str(oid))
        points.append(xy)
    if not points:
        return None, known
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points)), known


def parse_route(reply: Any, objects: dict[str, dict], log=None,
                reach_m: float = SETTLE_RADIUS_M) -> tuple[list[Waypoint], list[str]]:
    """A model reply becomes a drivable route, plus a trace of every correction applied.

    The model owns which object each phrase means; this owns only that the numbers are real and
    the shape is drivable. In order: drop malformed entries, forget cited ids the map does not
    have, replace a coordinate that contradicts its own citation, then guarantee exactly one
    goal and put it last -- the goal is the only constraint scored on where the robot *ends*,
    so a route that finishes anywhere else has thrown that constraint away.
    """
    trace: list[str] = []
    note = trace.append
    route: list[Waypoint] = []

    for index, raw in enumerate(_iter_waypoints(reply)):
        role = str(raw.get("role") or "").strip().lower()
        if role not in ROLES:
            note(f"waypoint {index}: unknown role {role!r} — treated as pass")
            role = "pass"
        ids = [str(v) for v in (raw.get("object_ids") or []) if str(v).strip()]
        cited, known = _cited_center(ids, objects)
        if len(known) != len(ids):
            note(f"waypoint {index}: dropped cited ids not in the map: "
                 f"{sorted(set(ids) - set(known))}")

        xy = _finite_xy(raw)
        if xy is None and cited is None:
            note(f"waypoint {index}: no usable coordinate and no known object — dropped")
            continue
        offset = 0.0
        if cited is None:
            note(f"waypoint {index}: no cited object — the coordinate stands unchecked")
        elif xy is None:
            xy = cited
            note(f"waypoint {index}: no coordinate — using the cited objects' centre")
        else:
            offset = math.dist(xy, cited)
            if offset > SNAP_M:
                note(f"waypoint {index}: ({xy[0]:.2f}, {xy[1]:.2f}) is {offset:.2f} m from "
                     f"objects {known} — snapped to their centre")
                xy, offset = cited, 0.0

        route.append(Waypoint(
            x=float(xy[0]), y=float(xy[1]), role=role, object_ids=known,
            why=str(raw.get("why") or "").strip(), reach_m=reach_m))

    route = _one_goal_last(route, note)
    if log is not None:
        for line in trace:
            log(f"cat3: {line}")
    return route, trace


def _iter_waypoints(reply: Any) -> Iterable[dict]:
    """Waypoints out of a pydantic model, a dict, or a bare list — whatever the backend gave."""
    if reply is None:
        return []
    items = getattr(reply, "waypoints", None)
    if items is None and isinstance(reply, dict):
        items = reply.get("waypoints")
    if items is None and isinstance(reply, (list, tuple)):
        items = reply
    out: list[dict] = []
    for item in items or []:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({k: getattr(item, k, None)
                        for k in ("role", "x", "y", "object_ids", "why")})
    return out


def _finite_xy(raw: dict) -> Optional[tuple[float, float]]:
    try:
        xy = (float(raw.get("x")), float(raw.get("y")))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(v) for v in xy) else None


def _one_goal_last(route: list[Waypoint], note) -> list[Waypoint]:
    """Exactly one goal, at the end. A route with no goal scores nothing for its final pose."""
    if not route:
        return route
    goals = [i for i, wp in enumerate(route) if wp.role == "goal"]
    if not goals:
        note("no goal in the reply — promoting the last waypoint")
        last = route[-1]
        route[-1] = Waypoint(last.x, last.y, "goal", last.object_ids,
                             last.why or "promoted: the reply named no stopping place",
                             last.reach_m)
        return route
    keep = goals[-1]
    if len(goals) > 1:
        note(f"{len(goals)} goals in the reply — keeping the last and passing through the rest")
        for i in goals[:-1]:
            wp = route[i]
            route[i] = Waypoint(wp.x, wp.y, "pass", wp.object_ids, wp.why, wp.reach_m)
    if keep != len(route) - 1:
        note("the goal was not last — moved to the end so the route finishes on it")
        goal = route.pop(keep)
        route.append(goal)
    return route


# ---------------------------------------------------------------- the drive


def snap_to_traversable(waypoint: Sequence[float],
                        area: Any,
                        clearance_m: float = OBSTACLE_CLEARANCE_M,
                        max_snap_m: float = MAX_SNAP_M) -> tuple[float, float]:
    """Move a waypoint onto the nearest floor the robot has actually seen.

    The model reasons about object CENTRES, so a correct waypoint routinely lands inside the
    furniture it names -- livingroom_1's "between the sofa and the round tables" is a midpoint
    that, measured against ground truth's own boxes, sits inside the sofa AND inside the table.
    Something has to move it onto ground, and this is where.

    One rule: the closest traversable point, from `TraversableArea` -- the floor grid built up
    across the run from `/terrain_map_ext`. Nothing is traded against that distance. An earlier
    version minimised

        ||p - waypoint|| + 0.10 * ||p - vehicle||     over ground eroded by 0.75 m

    where both terms pushed the aim point away from the waypoint on purpose: the erosion to stay
    inside the base autonomy's own candidate set so it would pass our point through untouched,
    the vehicle term to break ties toward the robot's side. Measured over livingroom_1 Q04+Q05
    they moved every waypoint 0.94-1.41 m, and since the robot tracks what we publish to within
    0.3 m, that displacement WAS the error. Both are gone; `clearance_m` survives only as a knob
    defaulting to zero.

    Returns `waypoint` unchanged when there is no answer -- no floor observed yet, the waypoint
    outside the grid, or the nearest free cell farther than `max_snap_m`. That last one is the
    guard that keeps the snap honest: past a few metres the nearest reading is not "the edge of
    the sofa the model meant", it is somewhere else entirely, and aiming there would be aiming
    at a different place than the one that was reasoned about.
    """
    home = (float(waypoint[0]), float(waypoint[1]))
    if area is None:
        return home
    nearest = area.nearest_free(home, clearance_m=clearance_m)
    if nearest is None:
        return home
    if math.dist(home, nearest) > max_snap_m:
        return home
    return nearest


def synthetic_trajectory(start_xy: Optional[Sequence[float]], route: Sequence[Any],
                         step_m: float = 0.15) -> list[list[float]]:
    """The path a PERFECT drive of `route` would trace, in score.py's (t, x, y) shape.

    Scoring this says what the plan itself was worth, independent of whether the robot managed
    to follow it -- which is the one thing a bare score cannot tell you. A run that plans well
    and drives badly and a run that plans badly are both "goal missed" otherwise.

    Straight legs, densely sampled, because `score_instruction` walks the trajectory with a
    monotone cursor: a leg represented by its endpoints alone would skip over a constraint the
    robot would really have driven through. The timestamps are synthetic and only have to
    increase -- nothing in the scorer reads them as durations.

    `route` items may be `Waypoint`s or the plain dicts `instruction_plan.json` stores.
    """
    def xy(item: Any) -> Optional[tuple[float, float]]:
        get = item.get if isinstance(item, dict) else lambda k, _d=None: getattr(item, k, _d)
        try:
            point = (float(get("x")), float(get("y")))
        except (TypeError, ValueError):
            return None
        return point if all(math.isfinite(v) for v in point) else None

    points: list[tuple[float, float]] = []
    if start_xy is not None and len(start_xy) >= 2:
        points.append((float(start_xy[0]), float(start_xy[1])))
    for item in route or []:
        target = xy(item)
        if target is None:
            continue
        if not points:
            points.append(target)
            continue
        previous = points[-1]
        span = math.dist(previous, target)
        for k in range(1, max(1, int(span / max(step_m, 1e-3))) + 1):
            t = min(1.0, k * max(step_m, 1e-3) / span) if span > 0 else 1.0
            points.append((previous[0] + t * (target[0] - previous[0]),
                           previous[1] + t * (target[1] - previous[1])))
        # Land exactly on the waypoint: the goal is scored on the FINAL point, so an
        # interpolation that stops a few centimetres short would misreport the plan.
        if points[-1] != target:
            points.append(target)
    return [[round(i * 0.5, 2), round(x, 3), round(y, 3)]
            for i, (x, y) in enumerate(points)]


def route_summary(route: Sequence[Waypoint]) -> str:
    parts = [f"{wp.role}({wp.x:.2f}, {wp.y:.2f})" + (f" {wp.why}" if wp.why else "")
             for wp in route if wp.scored]
    return " -> ".join(parts) if parts else "empty route"


def fallback_route(objects: dict[str, dict], prompts: Sequence[str]) -> list[Waypoint]:
    """Last resort when the model call fails outright: the named objects, in prompt order.

    This is not a second planner and must never be mistaken for one -- prompt order interleaves
    the objects a command names to IDENTIFY a place with the places themselves, so it is wrong
    about the route roughly as often as it is right. It exists because a robot that drives
    somewhere plausible can still satisfy a constraint, and one that never moves scores zero.
    Callers record `plan_source: "fallback"` so no report reads this as a plan.
    """
    used: set[str] = set()
    route: list[Waypoint] = []
    for prompt in prompts or []:
        phrase = str(prompt).strip().lower().replace(" ", "")
        if not phrase:
            continue
        best, best_volume = None, 0.0
        for oid, entry in objects.items():
            if oid in used:
                continue
            label = str(entry.get("label") or "").strip().lower().replace(" ", "")
            if not label or not (phrase == label or phrase in label or label in phrase):
                continue
            extent = _extent(entry) or (0.0, 0.0, 0.0)
            volume = extent[0] * extent[1] * extent[2]
            if volume >= best_volume:
                best, best_volume = oid, volume
        if best is None:
            continue
        xy = object_xy(objects[best])
        if xy is None:
            continue
        used.add(best)
        route.append(Waypoint(xy[0], xy[1], "pass", [best],
                              f"fallback: prompt {prompt!r}", SETTLE_RADIUS_M))
    return _one_goal_last(route, lambda _msg: None)


def heuristic_targets(question: str) -> list[str]:
    """Noun-ish tokens from an instruction, used only when the VLM extract is empty.

    Relation words are in `_STOP` on purpose: this list arms a detector, and SAM has no prompt
    for "closest". The route call reads the sentence itself, so nothing is lost by dropping them
    here.
    """
    # Imported here, not at module scope: `clean_targets` lives beside the counting reasoner
    # and pulls in the whole VLM backend stack, which would make this module -- documented as
    # pure, and unit-tested without a robot -- importable only where pydantic is installed.
    from smart_vlm.numerical_utils import clean_targets

    text = re.sub(r"[^a-z\s]", " ", (question or "").lower())
    tokens = [raw for raw in text.split() if raw not in _STOP and len(raw) > 2]
    return clean_targets(tokens)


def plan_payload(question: str, objects: dict[str, dict], route: Sequence[Waypoint],
                 *, table: str, images: Sequence[str], reply: Any, trace: Sequence[str],
                 source: str, robot_xy: Optional[Sequence[float]],
                 view_source: str = "") -> dict:
    """Everything needed to explain this route afterwards, written before the robot moves.

    The harness tears the pipeline down as soon as the question ends, so anything not on disk
    by then is gone. Recording the table and the images alongside the reply is what separates
    "the model reasoned badly" from "the model was shown a map without the object in it".
    """
    return {
        "question": question,
        "plan_source": source,
        "robot_xy": [round(float(v), 3) for v in (robot_xy or [])[:2]],
        "n_map_objects": len(objects),
        "map_table": table,
        # Which vqa.yaml view_source produced these paths, and the paths themselves. sam_node
        # can still overwrite the files afterwards, so this records what was sent rather than
        # preserving it -- enough to notice a mismatch, not enough to reconstruct one.
        "view_source": view_source,
        "images": [str(p) for p in images],
        "model_reply": _jsonable(reply),
        "corrections": list(trace),
        "route": [wp.as_dict() for wp in route],
        "summary": route_summary(route),
    }


def _jsonable(reply: Any) -> Any:
    if reply is None:
        return None
    dump = getattr(reply, "model_dump", None)
    if callable(dump):
        return dump()
    try:
        json.dumps(reply)
    except (TypeError, ValueError):
        return repr(reply)
    return reply
