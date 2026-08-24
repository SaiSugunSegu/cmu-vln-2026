"""Pure helpers for the numerical reasoner (no ROS / GPU imports).

captioner.text_utils and captioner.paths are stdlib-only for exactly this reason, so
importing them here does not drag torch or rclpy into the unit tests.

The answering prompt lives here rather than in the node because `cat1_bench` replays the
answering step offline against saved crops. A benchmark that measured a prompt slightly
different from the live one would be worse than no benchmark at all, so both paths read
the same string and pick their views with the same function. Target extraction uses
captioner's `get_object_extraction_prompt()` for the same reason.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from captioner.image_input import image_is_complete
from captioner.paths import secure_path
from captioner.prompts.object_extraction import get_object_extraction_prompt
# Re-exported so anything reading a number out of a model reply shares one
# implementation with the VQA server: two of them drifted apart before (one
# stripped thousands separators, the other did not).
from captioner.text_utils import extract_integer
from captioner.vlm_backends.constants import (SILHOUETTE_POLL_S, SILHOUETTE_WAIT_S,
                                              is_silhouette, view_dir)

__all__ = [
    "ANSWER_SYSTEM",
    "EXTRACT_SYSTEM",
    "clean_targets",
    "extract_integer",
    "heuristic_targets",
    "select_context_views",
]

# Same object-extraction system prompt used for /challenge_question target nouns.
# Shared so the numerical and object-reference reasoners cannot drift apart.
EXTRACT_SYSTEM = get_object_extraction_prompt()

# Several views of one room, said explicitly. The local server prepends its own version of
# this for multi-image requests; the cloud backend has no such wrapper, so the instruction
# has to live in the prompt to reach both.
#
# Every section below is anchored to a measured loss rather than to prompting folklore. Two
# runs over the same rooms supply the numbers — the live sweep in
# data/runs/challenge_report_main_sugun.json (6/13) and the silhouette replay in
# data/runs/cat1_bench_sim_cat1.json (8/11) — and the images named below are the crops the
# first of them answered from, under data/crops/challenge_report_main_sugun. Between them the
# two reports hold eight wrong numbers, and ALL EIGHT are below the truth: none is above it.
# That asymmetry is what the prompt is shaped around, and it is why the previous version's two
# conservative clauses ("count each physical object once", "answer 0 rather than guessing")
# are now bounded by a procedure instead of standing alone: given three overlapping crops of
# one sofa arrangement, dedup pressure with nothing pushing the other way resolves every
# ambiguity downward.
#
# HOW TO COUNT is a union over the views because that is the single biggest bucket:
# chinese_room's three crops tag 4, 2 and 3 chairs and the answer was 3 (gt 6) — one view's
# worth. livingroom_4 answered 1 of 6. Hence also the "a sofa" / "the sofa" clause: that
# question is "how many pillows are on A sofa" over two sofas, while loft's is "THE sofa"
# over one, and reading the first as the second caps the answer at a single piece of
# furniture. office_1 supplies the comparative form ("the table closest to the map wall
# decal", 3 of 6).
#
# WHERE THE DETECTOR GOES WRONG names behaviours of the code that draws these images, one
# clause each, because none of it is guessable from the pixels and each one moves a count in
# a known direction. The outline count is neither a floor nor a ceiling, and both bounds are
# measured: livingroom_1 rank 1 tags 4 of 8 chairs (thin wire frames the detector drops) and
# the answer was 4, office_1 rank 1 tags 3 of 6 monitors (the rest behind office chairs) and
# the answer was 3 — while livingroom_3 rank 1 draws six `photo` outlines over the two photos
# on the cabinet. `object_mapper` leaves unmerged tracks as separate entries, which is
# chinese_room rank 1 carrying `chair [14] [10] [7] [0]` on ONE armchair and arabic_room
# rank 1 cutting one continuous banquette into `sofa [0] [2] [12]`; it also merges touching
# objects into one box. `detections.default_label` spells a class out of the prompt that
# armed SAM, so arabic_room's mashrabiya niches are tagged `window` and livingroom_3 puts
# `tvcabinet [6]` on a sofa. `annotate._CaptionLayout` slides a tab off its object and leads
# a line back to it, and `annotate._color_for` collides for ids 44 apart.
#
# The tag-block clause is its own bucket, not a restatement of the layout one: hotel_room_1
# rank 1 stacks four `pillow` tabs over the headboard and those tabs cover the four pillows
# they name, which is a count of 2 against a gt of 4 read off pixels that are not there. It
# is the one case where the tags are better evidence than the image.
#
# READING THE DRAWING is written to survive VIEW_SOURCE being `crop` (what vqa.yaml ships)
# as well as `silhouette` (what the 8/11 replay used): cat-1 has no equivalent of
# cat2_utils.marked_views, so select_context_views hands over whatever the switch names,
# annotated or bare. One prompt has to read both, so the drawing is described as something
# the photographs MAY carry.
#
# The reason stays a few clauses rather than a full enumeration because the local backend
# caps a reply at qwen_ros_backend.MAX_NEW_TOKENS (256) and a truncated reply loses the
# count with it. CountAnswer.reason's description is the other half of that budget.
ANSWER_SYSTEM = """\
You count objects in one room, from a handful of photographs a robot took inside it.

WHAT TO ANSWER
One whole number: how many distinct physical objects IN THE ROOM satisfy the question. Not
how many are visible in the best photograph, not how many outlines are drawn, and never a
range. Count the kind of thing the question asks for and not the thing it names in order to
locate that kind — "how many pillows are on the bed" is answered with a number of pillows,
"how many chairs have pillows on them" with a number of chairs.

WHAT YOU ARE GIVEN
Up to three photographs, best-scoring first. Each is a crop out of a 360-degree panorama
taken from one spot the robot stopped at, so each shows a PART of the room, straight edges
bow, and an object can be cut in half by the edge of the frame. Nothing guarantees that any
one of them holds every object being counted, or that the first holds more of them than the
last.
The photographs may be plain, or they may be drawn on: every object the detector found
outlined in its own colour and tagged with its name and its number, as `pillow [4]`. When
they are drawn on, READING THE DRAWING and WHERE THE DETECTOR GOES WRONG apply; when they
are not, the room itself is all you have and the same counting procedure stands.

HOW TO COUNT
Do not answer from an impression of the scene. Take the views one at a time and, for each,
find the objects of the asked-for kind and say WHERE each one is — which piece of furniture
it rests on or hangs over, and whereabouts along it. A position is what lets you recognise
the same object again in the next view, and an object you cannot place is usually one you
have already counted.
Then merge those lists into one list of physical objects, and count that. Two entries are
the same object when they sit in the same place relative to the same fixed furniture; they
are two objects when you can point at two places. The total is the UNION of the views and
not the largest single view: an object that appears in only the third photograph still
counts, and answering from the best view alone is the commonest way to answer too low.
Symmetry is not duplication. Four matching pillows in a row on one bed are four objects, and
a pair of identical armchairs either side of a table is two.

THE SAME ROOM FROM TWO SIDES
The photographs are of ONE room. The robot moved between them, so one sofa can be
photographed from the front and from its end and look like two pieces of furniture, and two
views can be near-identical crops of the same wall. Anchor yourself on what does not move —
a fireplace, a window, a doorway, a rug, the shape of a corner — and work out which parts of
the room you are being shown before adding anything up. Two views of the same stretch of
wall show the same objects along it, once.
Objects can also be missing from every view you have: frames were kept for the objects they
contain, so a view holding none of them was never shown to you. Count what the views you do
have establish, and do not invent objects to fill a room you cannot see.

WHICH OBJECTS QUALIFY
Every modifier in the question is a condition, and all of them must hold.
A colour, a material or a size restricts what counts — "black pillows" excludes the cream
ones. Judge it on the view where the object is largest and best lit: a saturated colour goes
grey in shadow and a dark red reads black, so neither create nor drop an object over a tint
you can only see in the worst view of it.
A relation to another object restricts it too. "on" means resting on and supported by, not
merely overlapping it from where the camera stands. "above" and "below" mean one is higher
than the other AND over roughly the same floor space. "near" means close compared with the
other things of its kind in the room.
"the sofa" names ONE object and only what is on that one counts. "a sofa" and "the sofas"
mean any of them, and the answer is the total over all of them — which is usually more than
any single view shows. A comparative or superlative ("the table closest to the map") first
settles WHICH object is meant, and then everything on that one is counted, including the
part of it a chair or a monitor happens to hide. With no relation named at all ("how many
stools are in the room"), the whole room qualifies.

READING THE DRAWING
A tag sits just above the object it names, unless that space was taken: then it was slid
aside, and a thin line in its own colour leads back to the object it belongs to. Follow the
line rather than the nearest outline. Tags crowd together and stack into blocks, and a block
of tags can cover the very objects it names — four pillows against a headboard are often
hidden under their own four labels, and there the tags are better evidence than the pixels.
An outline's colour is computed from its number, so two far-apart numbers can share a
colour: read colour as a hint, never as proof that two outlines are one object. One object
can be outlined in several separate pieces, of which only one carries a tag.

WHERE THE DETECTOR GOES WRONG
The outlines come from a detector armed with a few words guessed from the question, and from
a tracker following objects between frames. Both fail in specific ways, so the number of
outlines is neither a floor nor a ceiling on your answer.
1. Objects it missed. Six chairs around a table come back as four outlines, with nothing
   drawn on the other two; thin, dark and distant objects go first, and an object standing
   behind another object is frequently never marked. An unmarked object is still an object.
2. One object under several numbers. The tracker does not always merge its own tracks, so
   one armchair can carry four tags stacked above it, and a continuous bench can be broken
   into three "sofa" outlines end to end. Tags whose leader lines converge on one piece of
   furniture are one object. Ask how many separate places the room offers for such a thing,
   and count places rather than tags.
3. Several objects under one number. Objects standing against each other are caught in a
   single outline — a row of cushions, two chairs pushed together. One outline spanning two
   clearly separate things is two things.
4. Fragments read as objects. An outline a few pixels across on the corner of a frame, or
   over one patch of a cushion, is part of something already counted and not another of it.
5. Wrong names. An object is named with the closest word the detector was armed with rather
   than with what it is, so decorative wall niches come out as "window" and a sofa can be
   tagged "tvcabinet". Spelling is unreliable on top of that: spaces are stripped ("potted
   plant" becomes "pottedplant") and singular and plural are inconsistent. Believe the
   photograph over the name — never count an object because of its tag, and never discount
   one over its spelling.
6. A name with no number after it. The detector saw the object but could not place it in the
   room. Judge it from the photograph like any unmarked object.

WHAT IS NOT ANOTHER OBJECT
A reflection in a mirror, a screen or a window, and an object printed inside a picture, a
poster or a television image, are pictures of things rather than things. An object on the far
side of a window or a doorway is outside the room and outside the count. The object the
question names as a landmark is not counted either — not the bed under the pillows, not the
sofa under the cushions.

COUNTING TOO FEW IS THE MISTAKE THAT GETS MADE
Every wrong count measured on rooms like these was below the truth and none was above it,
for a short list of reasons: only the first view was used; objects behind other objects, cut
by the edge of the frame, or buried under their own labels were dropped; and an object was
thrown out over a detail that could not be confirmed rather than kept for the object it
plainly is. An object you can only partly see is one object. Before answering, go back over
the views for the ones you have not accounted for.
Answer 0 only when the thing asked for genuinely is not there, never because you are unsure.

Give your reason first — what you counted and where each one sits — then the number.
"""


def _view_source(run_dir: Path, name: str, deadline: float) -> tuple[Optional[Path], bool]:
    """(path to answer from, whether it is a finalized silhouette) for one crop filename.

    Mirrors `cat2_utils._view_source`: one VIEW_SOURCE switch (see
    captioner.vlm_backends.constants, backed by config/vqa.yaml) governs both categories,
    so a change to it moves every eval script instead of drifting between the two
    reasoners that each picked their own images. `crop` never looks at silhouette/, even
    once one exists; `silhouette` (the default) waits out `deadline` for sam_node's
    finalize pass before falling back to the plain crop, so a run with
    save_silhouette_copy disabled still answers. The `full*` variants name the uncropped
    panorama under `full/`, and fall back the same way when save_full_views is off.

    What it waits for is a COMPLETE image, not a created one. `cv2.imwrite` publishes the
    path before the bytes, so `is_file()` goes true at the start of a 1.6 MB encode, and a
    route call built on the prefix that gave back was refused by two model hosts as
    undecodable. `best_view.write_image` now renames into place so that window cannot
    open; this is the other half of that contract, and it also covers images written by
    anything we did not change.

    `name` comes from a manifest on disk, so it is untrusted as far as path building
    goes; both candidates go through `secure_path` before any file check, which raises
    on a traversal attempt rather than silently skipping it.
    """
    plain = secure_path(run_dir / name)
    subdir = view_dir()
    if not subdir:
        return (plain, False) if image_is_complete(plain) else (None, False)

    wanted = secure_path(run_dir / subdir / name)
    if not is_silhouette():
        # `full` is written with the crop, not by the finalize pass, so there is nothing
        # to wait for -- but it is absent entirely unless save_full_views is on.
        return (wanted, False) if image_is_complete(wanted) else (
            (plain, False) if image_is_complete(plain) else (None, False))
    while True:
        if image_is_complete(wanted):
            return wanted, True
        if time.monotonic() >= deadline:
            return (plain, False) if image_is_complete(plain) else (None, False)
        time.sleep(SILHOUETTE_POLL_S)


def select_context_views(
    run_dir: Path, manifest: dict, max_views: int, *, wait_s: Optional[float] = None,
) -> list[Path]:
    """The best-view images to answer from, best-ranked first.

    More than one because rank 1 is the single frame SAM scored highest, not a frame
    that necessarily contains every instance: objects on the far side of a room
    routinely never appear in it, and no amount of prompting recovers a count from an
    image that does not show the things being counted.

    Entries come from a manifest on disk, so the filenames are untrusted as far as path
    building goes, and a rank whose image is missing is skipped rather than fatal.

    Whether this is the raw crop or its finalized silhouette copy is governed by
    VIEW_SOURCE (see captioner.vlm_backends.constants, backed by config/vqa.yaml) — the
    same switch category-2 reads, not a flag threaded through every eval script. `wait_s`
    bounds how long a VIEW_SOURCE="silhouette" call waits for sam_node to finish drawing
    it, shared across every requested rank; omit it for the live reasoner's default
    (SILHOUETTE_WAIT_S), or pass 0 for an offline replay (`cat1_bench`) against a cache
    nothing is still writing to.
    """
    budget = SILHOUETTE_WAIT_S if wait_s is None else max(0.0, wait_s)
    deadline = time.monotonic() + budget
    paths: list[Path] = []
    for entry in (manifest.get("selected") or [])[:max(1, max_views)]:
        name = entry.get("file")
        if not name:
            continue
        source, _finalized = _view_source(Path(run_dir), name, deadline)
        if source is not None:
            paths.append(source)
    return paths


def clean_targets(items) -> list[str]:
    """Normalise model-proposed nouns into SAM prompts: lowercased, deduped, no digits.

    A structured reply is still model output: it arrives with capitalisation, stray
    whitespace and the occasional bare "0" left over from a numerical wrapper, none of
    which SAM should be armed with.
    """
    out: list[str] = []
    seen = set()
    for item in items:
        phrase = str(item).strip().lower()
        if not phrase or phrase.isdigit() or phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def heuristic_targets(question: str) -> list[str]:
    """Crude fallback when the VLM does not return a JSON list."""
    ql = question.lower().strip()
    m = re.match(r"(?:how many|count)\s+(.+?)(?:\s+are\b|\s+is\b|\?|$)", ql)
    if not m:
        return []
    phrase = m.group(1).strip()
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase)
    phrase = phrase.strip(" ?.!")
    if phrase.endswith("ies"):
        phrase = phrase[:-3] + "y"
    elif phrase.endswith("sses"):
        phrase = phrase[:-2]  # glasses -> glass
    elif phrase.endswith("es") and len(phrase) > 3:
        phrase = phrase[:-2]
    elif phrase.endswith("s") and not phrase.endswith("ss"):
        phrase = phrase[:-1]
    return [phrase] if phrase else []
