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

from captioner.paths import secure_path
from captioner.prompts.object_extraction import get_object_extraction_prompt
# Re-exported so anything reading a number out of a model reply shares one
# implementation with the VQA server: two of them drifted apart before (one
# stripped thousands separators, the other did not).
from captioner.text_utils import extract_integer
from captioner.vlm_backends.constants import SILHOUETTE_POLL_S, SILHOUETTE_WAIT_S, VIEW_SOURCE

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

# Several views of one room, said explicitly. The local server prepends its own
# version of this for multi-image requests; the cloud backend has no such wrapper,
# so the instruction has to live in the prompt to reach both.
ANSWER_SYSTEM = (
    "You answer counting questions about a room from photographs of it. The images "
    "are different views of the SAME room, taken by a robot as it moved around, so "
    "an object can appear in more than one of them — count each physical object "
    "once, however many views it shows up in. Count only what you can see, and "
    "answer 0 rather than guessing when nothing matches."
)


def _view_source(run_dir: Path, name: str, deadline: float) -> tuple[Optional[Path], bool]:
    """(path to answer from, whether it is a finalized silhouette) for one crop filename.

    Mirrors `cat2_utils._view_source`: one VIEW_SOURCE switch (see
    captioner.vlm_backends.constants, backed by config/vqa.yaml) governs both categories,
    so a change to it moves every eval script instead of drifting between the two
    reasoners that each picked their own images. `crop` never looks at silhouette/, even
    once one exists; `silhouette` (the default) waits out `deadline` for sam_node's
    finalize pass before falling back to the plain crop, so a run with
    save_silhouette_copy disabled still answers.

    `name` comes from a manifest on disk, so it is untrusted as far as path building
    goes; both candidates go through `secure_path` before any file check, which raises
    on a traversal attempt rather than silently skipping it.
    """
    plain = secure_path(run_dir / name)
    if VIEW_SOURCE != "silhouette":
        return (plain, False) if plain.is_file() else (None, False)

    silhouette = secure_path(run_dir / "silhouette" / name)
    while True:
        if silhouette.is_file():
            return silhouette, True
        if time.monotonic() >= deadline:
            return (plain, False) if plain.is_file() else (None, False)
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
