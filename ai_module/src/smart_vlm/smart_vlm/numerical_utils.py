"""Pure helpers for numerical reasoner (no ROS / GPU imports).

captioner.text_utils is stdlib-only for exactly this reason, so importing it
here does not drag torch or rclpy into the unit tests.
"""
from __future__ import annotations

import re

# Re-exported so anything reading a number out of a model reply shares one
# implementation with the VQA server: two of them drifted apart before (one
# stripped thousands separators, the other did not).
from captioner.text_utils import extract_integer

__all__ = ["clean_targets", "extract_integer", "heuristic_targets"]


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
