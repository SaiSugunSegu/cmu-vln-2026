"""Pure helpers for category-1 reasoner (no ROS / GPU imports).

captioner.text_utils is stdlib-only for exactly this reason, so importing it
here does not drag torch or rclpy into the unit tests.
"""
from __future__ import annotations

import ast
import json
import re

# Re-exported: the reasoner parses integers both from its own fallback path and
# from the VQA server's reply, and two implementations drifted apart before
# (one stripped thousands separators, the other did not).
from captioner.text_utils import extract_integer

__all__ = ["extract_integer", "heuristic_targets", "parse_target_list"]

_JSON_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_target_list(text: str) -> list[str]:
    """Parse a JSON (or Python-literal) list of object noun phrases from model text."""
    if not text:
        return []
    candidates = []
    stripped = text.strip()
    candidates.append(stripped)
    match = _JSON_LIST_RE.search(stripped)
    if match:
        candidates.append(match.group(0))

    for raw in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(raw)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(value, list):
                out = []
                seen = set()
                for item in value:
                    phrase = str(item).strip().lower()
                    # Reject bare integers ("0") from the numerical VQA wrapper.
                    if not phrase or phrase.isdigit() or phrase in seen:
                        continue
                    seen.add(phrase)
                    out.append(phrase)
                if out:
                    return out
    return []


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
