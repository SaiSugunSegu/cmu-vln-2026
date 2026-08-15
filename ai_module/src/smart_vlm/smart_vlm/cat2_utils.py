"""Pure helpers for the object-reference (category-2) reasoner: no ROS, no GPU.

The live node and the offline bench must make the *same* choice from the same inputs, for
the same reason `numerical_utils` exists: a bench that measured a slightly different prompt,
a different candidate ordering or a different tie-break would be worse than no bench at all.
So the whole selection policy lives here, behind `select_object`, and both callers pass in
their own way of reaching a model.

The geometry comes from `scripts/utils`, which is where the benchmark's own predicates live.
That tree is bind-mounted into the container read-only (`/home/docker/scripts`) rather than
copied into the image, so the import can fail in a way `captioner` never can: when it does,
`SOLVER_AVAILABLE` goes false and selection degrades to class match plus largest volume
instead of raising. A degraded answer scores something; an exception scores zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Sequence

# ---------------------------------------------------------------- prompts

EXTRACT_SYSTEM = (
    "You list the objects a detector should look for in order to answer a question that "
    "points at one specific object. Include EVERY referenced object: the thing being asked "
    "for and every landmark it is described relative to, such as a window or a table. Bare "
    "nouns only, no colours and no other adjectives."
)

# The candidate list is data, not instruction — a question is user text and must never be
# able to talk the model out of the format. Hence "copy an id from the list" rather than
# "answer the question": the only free choice offered is which line.
ANSWER_SYSTEM = (
    "You choose which object a question points at. You are given photographs of one room "
    "taken by a robot, with numbered tags drawn on the objects it detected, and a list of "
    "candidate objects carrying those same numbers. The list already states every distance "
    "and spatial relation, measured in 3D: trust those numbers over your impression of the "
    "photographs, and use the photographs to judge what the numbers cannot say — what an "
    "object is, what it looks like, and whether a tag is on the thing you expect. Answer "
    "with the id of exactly one candidate, copied from the list."
)

# How many candidates the model is shown. A table long enough to hold a whole scene buries
# the handful of objects the question is about, which is the largest single gain every recent
# zero-shot grounding method reports.
TABLE_LIMIT = 12


class Selection(NamedTuple):
    """One answer, with enough of its provenance to explain a wrong one afterwards."""

    object_id: Optional[str]
    source: str                 # solver | vlm | naive | none
    reason: str
    candidates: list[str]       # ranked ids the choice was made from
    trace: list[str]
    vlm_calls: int


# ---------------------------------------------------------------- solver bridge


def scripts_root() -> Optional[Path]:
    """The repo's `scripts/` directory, wherever this happens to be running.

    Checked in order of how explicit they are: an override, the container's fixed mount
    point, then the tree this file sits in. `scripts/utils/__init__.py` documents that
    consumers put `scripts/` on the path and `import utils.<module>`.
    """
    candidates = [
        os.environ.get("VLN_SCRIPTS_DIR"),
        "/home/docker/scripts",
        str(Path(__file__).resolve().parents[4] / "scripts"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "utils" / "geometry.py").is_file():
            return Path(candidate)
    return None


_ROOT = scripts_root()
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import utils.objmap as objmap                                        # noqa: E402
    from utils.geometry import Obj, relation_holds                       # noqa: E402
    SOLVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in an image without scripts/
    objmap = None                                                       # type: ignore[assignment]
    Obj = Any                                                           # type: ignore[misc,assignment]
    relation_holds = None                                               # type: ignore[assignment]
    SOLVER_AVAILABLE = False


def solver_status() -> str:
    if SOLVER_AVAILABLE:
        return f"spatial solver from {_ROOT}"
    return ("spatial solver UNAVAILABLE (scripts/utils not importable) — selection will fall "
            "back to class match + largest volume")


# ---------------------------------------------------------------- views


def marked_views(
    run_dir: Path,
    manifest: dict,
    ids: Sequence[str],
    max_views: int,
    labels: Optional[dict[str, str]] = None,
) -> list[Path]:
    """Best views with the candidates' track ids drawn on, best-ranked first.

    Written next to the crops as `marked/`, so a run directory keeps the exact images an
    answer was given — a selection that cannot be looked at afterwards cannot be debugged.

    Marks come from `manifest.selected[].instances[]`, whose `bbox` is already in the crop's
    own pixel coordinates and whose `track_id` is the key of the 3D map. That correspondence
    is the one thing that lets a model reason about pixels and answer with a map id; it is
    recorded for free by the best-view collector, so nothing has to be re-detected here.

    Only the ids passed in are marked. Tagging every instance puts thirty tabs on a crop and
    buries the handful of objects the question is about.
    """
    import cv2  # local: the bench imports this module for prompts alone on hosts without cv2

    from sam_mapper.annotate import mark_frame

    wanted = {str(i) for i in ids}
    out_dir = run_dir / "marked"
    marked: list[Path] = []
    for entry in (manifest.get("selected") or [])[: max(1, max_views)]:
        name = entry.get("file")
        if not name:
            continue
        # A silhouette copy outlines each mask in the same colour the tab will use, so a
        # marked silhouette reads as one object per colour. Plain crop when there is none.
        source = next((run_dir / sub / name for sub in ("silhouette", "")
                       if (run_dir / sub / name).is_file()), None)
        if source is None:
            continue
        image = cv2.imread(str(source))
        if image is None:
            continue

        marks = []
        for inst in entry.get("instances") or []:
            track = str(inst.get("track_id"))
            bbox = inst.get("bbox")
            if track not in wanted or not bbox or len(bbox) != 4:
                continue
            label = (labels or {}).get(track) or str(inst.get("label") or "")
            marks.append((bbox, int(float(track)) if track.lstrip("-").isdigit() else 0,
                          f"[{track}] {label}".strip()))
        if not marks:
            # No candidate is visible in this view. Showing it anyway invites the model to
            # pick from pixels that carry no id at all.
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / name
        cv2.imwrite(str(target), mark_frame(image, marks))
        marked.append(target)
    return marked


# ---------------------------------------------------------------- selection


def heuristic_targets(question: str) -> list[str]:
    """The nouns to arm SAM with, parsed rather than guessed at.

    Category-2 questions have one shape — a head noun followed by spatial hops — and
    `text_solver.parse` already recovers it, so this is a real fallback rather than the crude
    regex the counting reasoner has to make do with: it recovers the landmark nouns from the
    deeper hops too ("... on the stool"), which are exactly what a detector must be armed
    with for the relation to be judgeable at all.
    """
    if not SOLVER_AVAILABLE:
        return []
    import utils.text_solver as solver
    from utils.geometry import object_nouns

    spec = solver.parse(question)
    phrases = [spec["head"]] + [p for hop in spec["hops"] for p in hop["phrases"]]
    return object_nouns([p for p in phrases if p])


def naive_from_raw(raw_map: dict, question: str = "") -> tuple[Optional[str], str]:
    """The best guess obtainable from an unparsed `obj_map.json` and the question text.

    Deliberately free of every import this module might not have: this is the answer of last
    resort, used when the solver tree is missing or when the clock has run out, and it must
    not be able to fail for the same reason the thing it stands in for failed.

    With a question it prefers instances whose class the question names *earliest* — English
    puts the asked-for object before its landmarks ("the pillow closest to the book"), so the
    first class named is almost always the one being asked for, and the largest instance of
    the right class beats the largest instance in the room. Labels are compared with their
    spaces removed because that is how map_node spells them ("potted plant" -> "pottedplant").
    """
    def volume_of(entry) -> float:
        extent = ((entry or {}).get("bbox3d") or {}).get("extent") or []
        if len(extent) != 3:
            return 0.0
        return abs(float(extent[0]) * float(extent[1]) * float(extent[2]))

    squashed = "".join(ch for ch in str(question or "").lower() if ch.isalnum())
    usable = {str(k): v for k, v in (raw_map or {}).items() if volume_of(v) > 0.0}
    if not usable:
        return None, "the map holds no usable box"

    def rank(item) -> tuple[int, float]:
        label = str((item[1] or {}).get("label") or "").strip().lower().replace(" ", "")
        at = squashed.find(label) if (squashed and label) else -1
        # Unnamed classes sort last; among named ones, earliest mention then biggest box.
        return (at if at >= 0 else len(squashed) + 1, -volume_of(item[1]))

    if squashed:
        track_id, entry = min(usable.items(), key=rank)
        label = str((entry or {}).get("label") or "?")
        if rank((track_id, entry))[0] <= len(squashed):
            return track_id, f"largest {label} the question names"
        return track_id, "largest box in the map (no class the question names)"

    track_id = max(usable, key=lambda k: volume_of(usable[k]))
    return track_id, "largest box in the map"


def naive_pick(question: str, objects: dict) -> tuple[Optional[str], str]:
    """The uninformed answer: the biggest object whose class the question names.

    The floor every other mode is measured against. Largest volume rather than first-seen
    because that is what `score_map3d`'s naive column already measures, so the two numbers
    describe the same policy.
    """
    if not objects:
        return None, "the map is empty"

    import utils.text_solver as solver

    head = solver.parse(question)["head"]
    pool = solver.match_class(head, list(objects.values())) or list(objects.values())
    best = max(pool, key=lambda o: float(o.volume))
    return best.id, f"largest {head or 'object'} in the map"


def select_object(
    question: str,
    objects: dict,
    *,
    mode: str = "hybrid",
    ask: Optional[Callable[..., Any]] = None,
    views_for: Optional[Callable[[list[str], dict[str, str]], list[Path]]] = None,
    log: Callable[[str], None] = print,
) -> Selection:
    """Pick the object a category-2 question points at, out of a 3D map.

    Modes, cheapest first:

      naive   class match, largest volume. No model call.
      solver  the benchmark's own predicates over the map. No model call.
      vlm     the model choosing from a candidate table and the marked views.
      hybrid  solver when the geometry is decisive, model when it is not.

    `hybrid` is the one worth running: the corpus is dominated by comparatives and `between`,
    which are arithmetic the solver does exactly, and the cases it cannot settle — a margin
    under the benchmark's own `MIN_MARGIN`, an anchor it could not resolve, a predicate no
    candidate satisfies — are precisely the ones where appearance decides. Spending a call
    only there keeps the cost proportional to the difficulty rather than the corpus size.

    `ask(system, user, images, schema)` is however the caller reaches a model. Any failure it
    raises is caught: a model that errors leaves the solver's answer standing rather than
    losing the question.

    `views_for(ids, labels)` renders the images, and is called only once the candidates are
    known so that the tags drawn on the crops are exactly the ids listed in the table.
    """
    if not objects:
        return Selection(None, "none", "the map holds no objects", [], [], 0)
    if not SOLVER_AVAILABLE:
        raise RuntimeError(
            "select_object needs scripts/utils on the path; callers must check "
            "SOLVER_AVAILABLE and fall back to naive_from_raw")
    if mode == "naive":
        oid, why = naive_pick(question, objects)
        return Selection(oid, "naive", why, [oid] if oid else [], [solver_status()], 0)

    # `shortlist` prunes room-scale structure out of the *candidates* itself and still needs
    # to see it, because those objects are half the corpus's anchors.
    picked = objmap.shortlist(question, list(objects.values()))
    ranked: list = picked["candidates"]
    trace: list[str] = list(picked["trace"])
    groups = picked["anchor_groups"]
    relation = picked["relation"]

    if not ranked:
        oid, why = naive_pick(question, objects)
        return Selection(oid, "naive", f"{picked['reason']}; {why}", [oid] if oid else [],
                         trace, 0)

    ids = [o.id for o in ranked]
    solver_choice = ranked[0]
    decisive = picked["committed"] is not None and objmap.is_decisive(ranked, groups, relation)

    if mode == "solver" or ask is None:
        return Selection(solver_choice.id, "solver", picked["reason"] or "top of the ranking",
                         ids, trace, 0)
    if mode == "hybrid" and decisive:
        trace.append("geometry is decisive — no model call")
        return Selection(solver_choice.id, "solver", picked["reason"], ids, trace, 0)

    # The model sees the ranked shortlist for `hybrid` and every same-class candidate for
    # `vlm`: the point of the `vlm` mode is to measure a model that was not handed the
    # solver's ordering, so it must not inherit the relation filter either.
    shown = ranked if mode == "hybrid" else sorted(ranked, key=lambda o: o.id)
    table = objmap.candidate_table(shown, groups, relation, limit=TABLE_LIMIT)
    allowed = {o.id: o for o in shown[:TABLE_LIMIT]}

    images: list[Path] = []
    if views_for is not None:
        # Anchors are marked as well as candidates: "between a door frame and a window" is
        # unjudgeable from a crop where only the lamps carry tags.
        labels = {o.id: o.display for o in shown[:TABLE_LIMIT]}
        labels.update({a.id: a.display for group in groups for a in group})
        images = views_for(list(labels), labels)
        trace.append(f"marked views: {[p.name for p in images]}")

    calls = 0
    try:
        from captioner.vlm_backends.schemas import ObjectChoice

        user = (f"Question: {question}\n\nCandidate objects:\n{table}\n\n"
                "Reply with the id of the one object the question points at.")
        result = ask(ANSWER_SYSTEM, user, images, ObjectChoice)
        calls += 1
        chosen = str(int(result.object_id))
        reason = str(getattr(result, "reason", "") or "")
        if chosen not in allowed:
            trace.append(f"model answered {chosen!r}, which is not a candidate — keeping the "
                         "geometry's pick")
            return Selection(solver_choice.id, "solver",
                             f"model chose an unlisted id: {reason}", ids, trace, calls)
        trace.append(f"model chose {chosen}: {reason}")

        # The solver verifies the model's pick, not the other way round: a relation the
        # geometry can check is not a matter of opinion, and a model that picks an object
        # failing the stated relation while another satisfies it has misread the tags.
        if (relation and groups and relation_holds is not None
                and relation not in ("closest", "farthest")):
            anchors = [a for group in groups for a in group]
            try:
                holds = relation_holds(allowed[chosen], anchors, relation)
                if not holds and (ok := [o for o in shown
                                         if relation_holds(o, anchors, relation)]):
                    trace.append(f"{chosen} fails {relation}; falling back to {ok[0].id}")
                    return Selection(ok[0].id, "solver",
                                     f"model pick failed {relation}", ids, trace, calls)
            except ValueError:
                pass    # a relation with no predicate (comparatives) — nothing to verify

        return Selection(chosen, "vlm", reason or "model choice", ids, trace, calls)
    except Exception as exc:  # noqa: BLE001 — a model failure must not lose the question
        log(f"cat2: model selection failed ({type(exc).__name__}: {exc}) — using the geometry")
        trace.append(f"model call failed: {type(exc).__name__}: {exc}")
        return Selection(solver_choice.id, "solver", picked["reason"], ids, trace, calls)
