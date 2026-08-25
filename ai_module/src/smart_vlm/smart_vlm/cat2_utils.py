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
from typing import Any, Callable, Collection, NamedTuple, Optional, Sequence

from captioner.image_input import image_is_complete
from captioner.vlm_backends import constants as vlm_constants
from captioner.vlm_backends.constants import SILHOUETTE_POLL_S, SILHOUETTE_WAIT_S, is_silhouette, view_dir
from smart_vlm.numerical_utils import EXTRACT_SYSTEM

# ---------------------------------------------------------------- prompts

# The candidate list is data, not instruction — a question is user text and must never be
# able to talk the model out of the format. Hence "copy an id from the list" rather than
# "answer the question": the only free choice offered is which line.
#
# Every section below is anchored to a bucket of measured losses rather than to prompting
# folklore: three answers naming an anchor instead of an instance of the head noun; `between`
# 2/8, `near` 0/4 and vertical relations 3/9; four questions lost to the model displacing a
# correct geometric pick against four lost to it failing to correct a wrong one, which is
# what the "which evidence settles what" split exists to name; a median box 1.78x the true
# volume; and one whole reasoning chain lost to an anchor the map never held.
#
# WHERE THE DETECTOR GOES WRONG names behaviours of the code that draws the images and fills
# the table, one clause each: `annotate._CaptionLayout` slides a tag off its object and
# leads a line back to it, `annotate._color_for` collides for ids 44 apart,
# `best_view._resolve` tags one of two outline pieces and falls back to a track id when the
# mapper never boxed the object, `detections.default_label` spells a class out of the prompt
# that armed SAM, `object_mapper` leaves unmerged fragments as separate entries and merges
# touching objects into one box, and `marked_views` drops a view no candidate appears in.
# None of it is guessable from the images alone: a model told none of it reads a mislabelled
# candidate as the wrong class, a track id as a choosable answer, and an undetected landmark
# as one that is not in the room.
#
# The clause rejecting a box thin on two axes comes from "the potted plant farthest from the
# hookah" on the 40-question sim cache: the hookah went undetected, so the table carried two
# bare centres and nothing that could order them, and the model answered with a 5cm x 5cm x
# 83cm "pottedplant" — a stalk. It is stated as a rule rather than left to judgement because
# it follows from the scorer: an answer is graded on box overlap, so a sliver scores zero even
# when it sits on the right object. Costed over the same cache before it was added: 10 of the
# 40 questions hold an object under 12cm on both of its smallest axes, such an object was the
# best reachable answer in none of them, the model chose one only here, and the narrowest
# answer in the corpus (a 0.19 x 0.09 x 0.36 wall lamp) clears the threshold with room to
# spare. The one-axis carve-out is load-bearing rather than hedging — a picture or a carpet IS
# flat, and those are the boxes the section above already says are inflated worst.
ANSWER_SYSTEM = """\
You choose which object a question points at, out of a robot's own map of one room.

WHAT YOU ARE GIVEN
Photographs the robot took. Every object it detected is outlined and tagged with its name
and its number, and the candidates and the anchors are drawn with a thicker box. A text list
of the same objects under the same numbers, split into CANDIDATES and ANCHORS, in which
every distance and spatial relation has already been measured in 3D.

WHAT TO ANSWER
The id of exactly one CANDIDATE, copied from the list. The list states the one kind of
thing the question is looking for, and your answer must be an instance of that kind.
ANCHORS are the objects the question names in order to say WHICH candidate it means: they
are listed so that you can refer to them, and are never the answer.

WHICH EVIDENCE SETTLES WHAT
The measurements settle where things are. They are computed from the 3D map, so do not
re-estimate a distance or a gap from a photograph and do not overturn one because a
photograph looks otherwise.
The photographs settle what a thing is: whether a candidate really is the kind of object
being asked for, and whether its tag sits on the thing you expect. A candidate that is
visibly the wrong kind of object is wrong however good its numbers are — so if your reason
says a candidate is not the thing the question asks for, do not then answer with it.

READING THE PHOTOGRAPHS
A tag sits just above the object it names, unless that space was already taken: then it was
slid aside, and a thin line in its own colour leads back to the object it belongs to. Follow
that line rather than the nearest outline. One object can be outlined in two separate
pieces, of which only one carries a tag. An outline's colour is computed from its number, so
two far-apart numbers can share one colour: read colour as a hint, never as proof that two
outlines are the same object.

READING THE MEASUREMENTS
between — read the candidate's `between` line. `yes` means it holds; among several that
  hold, the smaller `lateral offset` sits more squarely between the two anchors. `no` means
  it is off to one side, however close it is to both of them.
near, closest, farthest — read `box gap` (surface to surface) for near and `centre` for
  closest and farthest, on the row for the anchor the question names, and ignore the rows
  for the others.
on, above, below, in — read the `holds:` flags, which already combine the vertical gap with
  an overlap in plan view. `holds: on` is support. A small `vertical gap` with no `on` flag
  means the two do not overlap from above, so neither is on the other.

WHERE THE DETECTOR GOES WRONG
The outlines come from a detector armed with a few words guessed from the question, and from
a tracker following objects across frames. Both fail in specific ways, and the question
stays answerable through every one of them.
1. Wrong names. An object is named with the closest word the detector was armed with rather
   than with what it is, and a whole class can be misnamed the same way. Spelling is
   unreliable on top of that: spaces are stripped out ("potted plant" becomes
   "pottedplant"), and singular and plural are not consistent. Believe the photograph over
   the name, on a tag and in the list alike, and never rule a candidate out over its
   spelling.
2. A number on a photograph that the list does not hold. Not every outlined object is a
   candidate or an anchor, and a tag can carry the tracker's own number instead of a list
   number. Say which object you mean by what and where it is, then answer with the CANDIDATE
   line that fits it. Never answer with a number the list does not hold.
3. One object under two numbers. The tracker does not always merge its own tracks, so two
   lines whose centres are a few tens of centimetres apart and whose sizes are alike are
   probably one object counted twice, and one object can be tagged differently in two
   photographs. Answer with whichever of them covers the object better, and do not treat
   them as two separate things.
4. Two objects under one number. Objects standing against each other can be caught in a
   single box, whose size then spans them all: a `size` much larger than the thing you are
   looking at is the sign of it. Such a box is still the right answer if the object asked for
   is the one inside it that the question describes.
5. An object with no number at all. One the detector missed is unmarked in the photographs
   and absent from the list however plainly you can see it, and one it saw but could not
   place in 3D is outlined with no number after its name. Either way it cannot be answered
   with, and this happens most often to the landmark a question is anchored on. When the list
   says an object the question names was not detected, or you can see one that no line
   describes, say so in your reason, place it in the photographs relative to the objects that
   ARE tagged, and decide on the constraints that remain. Do not put a different object in
   its place, and do not answer with an object of the wrong kind because the right kind is
   absent.
6. A candidate in none of the photographs. Each photograph is one viewpoint of a room the
   robot drove through, and a view holding no candidate at all is not shown to you, so a
   candidate can be tagged in none of the ones you have. That is not evidence against it:
   judge that one from its measurements.

Give your reason first, then the id of one candidate.
"""

# How many candidates the model is shown. A table long enough to hold a whole scene buries
# the handful of objects the question is about, which is the largest single gain every recent
# zero-shot grounding method reports.
TABLE_LIMIT = 12


class Selection(NamedTuple):
    """One answer, with enough of its provenance to explain a wrong one afterwards."""

    object_id: Optional[str]
    source: str  # solver | vlm | naive | none
    reason: str
    candidates: list[str]  # ranked ids the choice was made from
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
    import utils.objmap as objmap  # noqa: E402
    from utils.geometry import Obj, relation_holds  # noqa: E402

    SOLVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in an image without scripts/
    objmap = None  # type: ignore[assignment]
    Obj = Any  # type: ignore[misc,assignment]
    relation_holds = None  # type: ignore[assignment]
    SOLVER_AVAILABLE = False


def solver_status() -> str:
    if SOLVER_AVAILABLE:
        return f"spatial solver from {_ROOT}"
    return (
        "spatial solver UNAVAILABLE (scripts/utils not importable) — selection will fall "
        "back to class match + largest volume"
    )


# ---------------------------------------------------------------- views


def _view_source(run_dir: Path, name: str, deadline: float) -> tuple[Optional[Path], bool]:
    """(path to draw on, whether it is a finalized silhouette).

    Gated by `view_source_object_reference` (see captioner.vlm_backends.constants,
    backed by config/vqa.yaml) -- category 2's own switch, independent of what category 1
    or 3 are set to. `crop` never looks at silhouette/, even once one exists;
    `silhouette` (the default) waits out `deadline` for sam_node's finalize pass — the
    bare crop has neither the outlines nor the ids — before falling back to the plain
    crop, so a run with save_silhouette_copy disabled still answers. The `full*` variants
    name the uncropped panorama under `full/`, falling back the same way when
    save_full_views is off.

    A COMPLETE image, not merely a created one: `cv2.imwrite` publishes the path before
    the bytes, so waiting on `is_file()` hands back the prefix of a 1.6 MB encode. See
    `numerical_utils._view_source`, which this mirrors.
    """
    import time

    source = vlm_constants.VIEW_SOURCE_OBJECT_REFERENCE
    plain = run_dir / name
    subdir = view_dir(source)
    if not subdir:
        return (plain, False) if image_is_complete(plain) else (None, False)
    wanted = run_dir / subdir / name
    if not is_silhouette(source):
        # `full` is written with the crop, not by the finalize pass: nothing to wait for,
        # but absent entirely unless save_full_views is on.
        return (
            (wanted, False)
            if image_is_complete(wanted)
            else ((plain, False) if image_is_complete(plain) else (None, False))
        )
    while True:
        if image_is_complete(wanted):
            return wanted, True
        if time.monotonic() >= deadline:
            return (plain, False) if image_is_complete(plain) else (None, False)
        time.sleep(SILHOUETTE_POLL_S)


def marked_views(
    run_dir: Path,
    manifest: dict,
    ids: Sequence[str],
    max_views: int,
    labels: Optional[dict[str, str]] = None,
    *,
    wait_s: Optional[float] = None,
) -> list[Path]:
    """Best views with the candidate objects marked, best-ranked first.

    Written next to the crops as `marked/`, so a run directory keeps the exact images an
    answer was given — a selection that cannot be looked at afterwards cannot be debugged.

    Marks come from `manifest.selected[].instances[]`, whose `bbox` is already in the crop's
    own pixel coordinates — recorded for free by the best-view collector. Each `track_id` is
    resolved through `obj_map.json` rather than compared to a candidate id directly: map_node
    keys an entry by `id[0]` and folds later track ids into `id` on a world merge, so an
    object whose crop shows a secondary id would never match and would lose its evidence.

    Only the ids passed in are marked. Tagging every instance puts thirty tabs on a crop and
    buries the handful of objects the question is about.

    A finalized silhouette already prints `label [map id]`, so marking it adds a box and no
    text — two tabs for one object is worse than one. The `[id] label` tab is drawn only on
    the bare-crop fallback, which carries no ids at all.

    `wait_s` bounds how long a silhouette-valued caller waits for sam_node to finish the
    finalize pass, shared across every requested rank; omit it for the live reasoner's
    default (SILHOUETTE_WAIT_S), or pass 0 for an offline replay against a cache nothing
    is still writing to.
    """
    import json
    import time

    import cv2  # local: the bench imports this module for prompts alone on hosts without cv2

    from sam_mapper.annotate import mark_frame
    from sam_mapper.challenge_marker import track_to_map_id

    try:
        with open(run_dir / "obj_map.json", "r", encoding="utf-8") as handle:
            track_to_map = track_to_map_id(json.load(handle) or {})
    except (OSError, json.JSONDecodeError):
        # Nothing to resolve against: read a track id as a map key, which is right for every
        # unmerged object and is what this did before.
        track_to_map = {}

    wanted = {str(i) for i in ids}
    budget = SILHOUETTE_WAIT_S if wait_s is None else max(0.0, wait_s)
    deadline = time.monotonic() + budget
    out_dir = run_dir / "marked"
    marked: list[Path] = []
    for entry in (manifest.get("selected") or [])[: max(1, max_views)]:
        name = entry.get("file")
        if not name:
            continue
        source, finalized = _view_source(run_dir, name, deadline)
        if source is None:
            continue
        image = cv2.imread(str(source))
        if image is None:
            continue

        marks = []
        for inst in entry.get("instances") or []:
            bbox = inst.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                track = int(float(inst.get("track_id")))
            except (TypeError, ValueError):
                continue
            map_id = track_to_map.get(track, track)
            if str(map_id) not in wanted:
                continue
            label = (labels or {}).get(str(map_id)) or str(inst.get("label") or "")
            # The MAP id, not the track id: `_color_for` is a pure function of what it is
            # given, and the silhouette outlines this object by its map id too, so anything
            # else would box it in a colour that is not its own.
            marks.append((bbox, map_id, "" if finalized else f"[{map_id}] {label}".strip()))
        if not marks:
            # No candidate is visible in this view. Showing it anyway invites the model to
            # pick from pixels that carry no id at all.
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / name
        cv2.imwrite(str(target), mark_frame(image, marks, thickness=3 if finalized else 1))
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


def answerable(candidates: Sequence[Any], anchors: Collection[str]) -> list[Any]:
    """`candidates` without the anchors — unless that empties it, in which case unchanged.

    A category-2 question describes its target *relative to* the objects it names as anchors
    ("the bowl on the table closest to the folding screen"), so an anchor is the one thing in
    the map the question itself rules out. That makes it a constraint rather than a
    preference, which is why it is enforced here and not merely asked for in the prompt.

    Class matching is deliberately loose at its last resort (`text_solver.match_class` falls
    back to any shared noun), so an anchor can turn up among the head-noun candidates. It
    cannot be the answer, but neither can nothing: dropping every candidate would trade a
    wrong box for no box, and both score zero.
    """
    return [o for o in candidates if str(o.id) not in anchors] or list(candidates)


def naive_pick(question: str, objects: dict, *, exclude: Collection[str] = ()) -> tuple[Optional[str], str]:
    """The uninformed answer: the biggest object whose class the question names.

    The floor every other mode is measured against. Largest volume rather than first-seen
    because that is what `score_map3d`'s naive column already measures, so the two numbers
    describe the same policy.

    `exclude` drops ids that cannot be the answer whatever else the map holds, which in
    practice means the anchors. It only bites on the path that arrives here with no class
    match at all, and that path is where three of the measured losses came from: with the
    head noun absent from the map, ranking every object by volume answered the anchor itself
    — a folding screen to a bowl question, a tv to a map question, a display ledge to a
    flowers question — while claiming in its reason to have found the largest bowl.
    """
    if not objects:
        return None, "the map is empty"

    import utils.text_solver as solver

    head = solver.parse(question)["head"]
    excluded = {str(i) for i in exclude}
    pool = [o for o in objects.values() if str(o.id) not in excluded]
    if not pool:
        return None, "every object in the map is an anchor of the question"
    if matched := solver.match_class(head, pool):
        best = max(matched, key=lambda o: float(o.volume))
        return best.id, f"largest {head or 'object'} in the map"
    # Said plainly, because the old wording ("largest bowl in the map") named a class the
    # pool did not contain and made a fallback read like a match.
    best = max(pool, key=lambda o: float(o.volume))
    return best.id, (
        f"no {head or 'match'} in the map; largest box that is not an anchor"
        if excluded
        else f"no {head or 'match'} in the map; largest box"
    )


def select_object(
    question: str,
    objects: dict,
    *,
    mode: str = "vlm",
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

    `vlm` is the default: every question gets a model call, which is what the submission
    is scored on. `hybrid` is the cheaper alternative worth knowing about: the corpus is
    dominated by comparatives and `between`, which are arithmetic the solver does exactly,
    and the cases it cannot settle — a margin under the benchmark's own `MIN_MARGIN`, an
    anchor it could not resolve, a predicate no candidate satisfies — are precisely the ones
    where appearance decides. Spending a call only there keeps the cost proportional to the
    difficulty rather than the corpus size.

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
            "SOLVER_AVAILABLE and fall back to naive_from_raw"
        )
    if mode == "naive":
        oid, why = naive_pick(question, objects)
        return Selection(oid, "naive", why, [oid] if oid else [], [solver_status()], 0)

    # `shortlist` prunes room-scale structure out of the *candidates* itself and still needs
    # to see it, because those objects are half the corpus's anchors.
    picked = objmap.shortlist(question, list(objects.values()))
    trace: list[str] = list(picked["trace"])
    groups = picked["anchor_groups"]
    relation = picked["relation"]
    # Every hop's anchors, not just `groups`' first hop: the constraint below is about what
    # the question rules out, and a question names landmarks at every depth. Kept under its
    # own name because `anchors` further down means the first hop's, which is what the
    # relation predicates take.
    all_anchors = picked["all_anchors"]
    anchor_ids = {str(a.id) for a in all_anchors}

    # The one thing a question states about its own answer: whatever it is, it is not one of
    # the objects named to locate it.
    ranked: list = answerable(picked["candidates"], anchor_ids)
    if len(ranked) < len(picked["candidates"]):
        trace.append(f"dropped {len(picked['candidates']) - len(ranked)} candidate(s) the " "question names as anchors")

    if not ranked:
        oid, why = naive_pick(question, objects, exclude=anchor_ids)
        return Selection(oid, "naive", f"{picked['reason']}; {why}", [oid] if oid else [], trace, 0)

    ids = [o.id for o in ranked]
    solver_choice = ranked[0]
    decisive = picked["committed"] is not None and objmap.is_decisive(ranked, groups, relation)

    if mode == "solver" or ask is None:
        return Selection(solver_choice.id, "solver", picked["reason"] or "top of the ranking", ids, trace, 0)
    if mode == "hybrid" and decisive:
        trace.append("geometry is decisive — no model call")
        return Selection(solver_choice.id, "solver", picked["reason"], ids, trace, 0)

    # The model sees the ranked shortlist for `hybrid` and every same-class candidate for
    # `vlm`: the point of the `vlm` mode is to measure a model that was not handed the
    # solver's ordering, so it must not inherit the relation filter either.
    shown = ranked if mode == "hybrid" else sorted(ranked, key=lambda o: o.id)
    table = objmap.candidate_table(
        shown,
        groups,
        relation,
        limit=TABLE_LIMIT,
        head=picked["head"],
        anchors=all_anchors,
        unmatched_anchors=picked["anchors_unmatched"],
    )
    allowed = {o.id: o for o in shown[:TABLE_LIMIT]}

    images: list[Path] = []
    if views_for is not None:
        # Anchors are marked as well as candidates: "between a door frame and a window" is
        # unjudgeable from a crop where only the lamps carry tags. Every hop's, matching the
        # table — an id listed there and tagged nowhere cannot be found in a photograph.
        labels = {o.id: o.display for o in shown[:TABLE_LIMIT]}
        labels.update({a.id: a.display for a in all_anchors})
        images = views_for(list(labels), labels)
        trace.append(f"marked views: {[p.name for p in images]}")

    calls = 0
    try:
        from captioner.vlm_backends.schemas import ObjectChoice

        user = (
            f"Question: {question}\n\nObjects:\n{table}\n\n"
            "Reply with the id of the one candidate the question points at."
        )
        result = ask(ANSWER_SYSTEM, user, images, ObjectChoice)
        calls += 1
        chosen = str(int(result.object_id))
        reason = str(getattr(result, "reason", "") or "")
        if chosen not in allowed:
            # Naming the anchor case separately because it is a different failure: not a
            # hallucinated id but the question's own landmark, whose id the fact rows and the
            # ANCHORS section both show. Prompted against and rejected here regardless.
            what = "an anchor, not a candidate" if chosen in anchor_ids else "not in the candidate list"
            trace.append(f"model answered {chosen!r}, {what} — keeping the geometry's pick")
            return Selection(
                solver_choice.id, "solver", f"model chose an id that is {what}: {reason}", ids, trace, calls
            )
        trace.append(f"model chose {chosen}: {reason}")

        # The solver verifies the model's pick, not the other way round: a relation the
        # geometry can check is not a matter of opinion, and a model that picks an object
        # failing the stated relation while another satisfies it has misread the tags.
        if relation and groups and relation_holds is not None and relation not in ("closest", "farthest"):
            anchors = [a for group in groups for a in group]
            try:
                holds = relation_holds(allowed[chosen], anchors, relation)
                if not holds and (ok := [o for o in shown if relation_holds(o, anchors, relation)]):
                    trace.append(f"{chosen} fails {relation}; falling back to {ok[0].id}")
                    return Selection(ok[0].id, "solver", f"model pick failed {relation}", ids, trace, calls)
            except ValueError:
                pass  # a relation with no predicate (comparatives) — nothing to verify

        return Selection(chosen, "vlm", reason or "model choice", ids, trace, calls)
    except Exception as exc:  # noqa: BLE001 — a model failure must not lose the question
        log(f"cat2: model selection failed ({type(exc).__name__}: {exc}) — using the geometry")
        trace.append(f"model call failed: {type(exc).__name__}: {exc}")
        return Selection(solver_choice.id, "solver", picked["reason"], ids, trace, calls)
