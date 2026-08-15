"""Pure helpers for the numerical reasoner (no ROS / GPU imports).

captioner.text_utils and captioner.paths are stdlib-only for exactly this reason, so
importing them here does not drag torch or rclpy into the unit tests.

The two prompts live here rather than in the node because `cat1_bench` replays the
answering step offline against saved crops. A benchmark that measured a prompt slightly
different from the live one would be worse than no benchmark at all, so both paths read
the same strings and pick their views with the same function.
"""
from __future__ import annotations

import re
from pathlib import Path

from captioner.paths import secure_path
# Re-exported so anything reading a number out of a model reply shares one
# implementation with the VQA server: two of them drifted apart before (one
# stripped thousands separators, the other did not).
from captioner.text_utils import extract_integer

__all__ = [
    "ANSWER_SYSTEM",
    "EXTRACT_SYSTEM",
    "clean_targets",
    "extract_integer",
    "heuristic_targets",
    "select_context_views",
]

EXTRACT_SYSTEM = (
    "You list the objects a detector should look for in order to answer a counting "
    "question. Include EVERY referenced object: the things being counted and any "
    "landmark they are described relative to, such as a jar or a table. Bare nouns "
    "only, no colours and no other adjectives."
)

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


def select_context_views(run_dir: Path, manifest: dict, max_views: int) -> list[Path]:
    """The best-view images to answer from, best-ranked first.

    More than one because rank 1 is the single frame SAM scored highest, not a frame
    that necessarily contains every instance: objects on the far side of a room
    routinely never appear in it, and no amount of prompting recovers a count from an
    image that does not show the things being counted.

    Entries come from a manifest on disk, so the filenames are untrusted as far as path
    building goes, and a rank whose image is missing is skipped rather than fatal.
    """
    paths: list[Path] = []
    for entry in (manifest.get("selected") or [])[:max(1, max_views)]:
        name = entry.get("file")
        if not name:
            continue
        candidate = secure_path(Path(run_dir) / name)
        if candidate.is_file():
            paths.append(candidate)
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
