"""Pure helpers for the instruction-following (category-3) reasoner.

The live node and the unit tests must pick the same (x, y) sequence from the same
obj_map + prompt list. No ROS or GPU imports.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from smart_vlm.numerical_utils import clean_targets

__all__ = [
    "center_xy",
    "heuristic_targets",
    "label_matches",
    "waypoints_from_map",
]

# Function words only — object nouns (window, table, fridge) stay as SAM prompts.
_STOP = frozenset({
    "take", "the", "path", "to", "and", "go", "near", "avoid", "from", "then",
    "stop", "at", "a", "an", "of", "that", "which", "with", "without", "by",
    "past", "around", "walk", "drive", "follow", "through", "into", "onto",
    "closest", "farthest", "between", "under", "above", "below", "on", "in",
    "for", "your", "you", "please", "toward", "towards",
})


def label_matches(prompt: str, label: str) -> bool:
    """True when a map label and a SAM prompt name the same class."""
    a = str(prompt or "").strip().lower().replace(" ", "")
    b = str(label or "").strip().lower().replace(" ", "")
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _volume(entry: Any) -> float:
    extent = ((entry or {}).get("bbox3d") or {}).get("extent") or []
    if len(extent) != 3:
        return 0.0
    return abs(float(extent[0]) * float(extent[1]) * float(extent[2]))


def center_xy(entry: Any) -> Optional[tuple[float, float]]:
    """Map-frame (x, y) of a box centre; None if the entry has no bbox3d.center."""
    center = ((entry or {}).get("bbox3d") or {}).get("center") or []
    if len(center) < 2:
        return None
    return float(center[0]), float(center[1])


def heuristic_targets(question: str) -> list[str]:
    """Noun-ish tokens from an instruction, used only when the VLM extract is empty."""
    text = re.sub(r"[^a-z\s]", " ", (question or "").lower())
    tokens: list[str] = []
    for raw in text.split():
        if raw in _STOP or len(raw) <= 2:
            continue
        tokens.append(raw)
    return clean_targets(tokens)


def waypoints_from_map(
    raw_map: dict,
    prompts: list[str],
) -> list[dict]:
    """One Pose2D-ready waypoint per prompt, in prompt order.

    Each prompt claims the largest unused map object whose label matches. The same
    object is never claimed twice; a prompt that matches nothing is skipped.
    """
    usable = {
        str(k): v for k, v in (raw_map or {}).items()
        if _volume(v) > 0.0 and center_xy(v) is not None
    }
    used: set[str] = set()
    out: list[dict] = []
    for prompt in prompts or []:
        phrase = str(prompt).strip().lower()
        if not phrase:
            continue
        candidates = [
            (oid, entry) for oid, entry in usable.items()
            if oid not in used and label_matches(phrase, (entry or {}).get("label") or "")
        ]
        if not candidates:
            continue
        oid, entry = max(candidates, key=lambda item: _volume(item[1]))
        xy = center_xy(entry)
        if xy is None:
            continue
        used.add(oid)
        out.append({
            "x": xy[0],
            "y": xy[1],
            "object_id": oid,
            "label": str((entry or {}).get("label") or phrase),
            "prompt": phrase,
        })
    return out
