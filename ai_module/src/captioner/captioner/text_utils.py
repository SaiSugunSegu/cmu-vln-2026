"""Text helpers shared by the captioner backends and the smart_vlm reasoners.

Stdlib only, on purpose: smart_vlm.category1_utils re-exports from here and must
stay importable without torch, CUDA, or rclpy so its unit tests run on any host.
"""
from __future__ import annotations

import re
from typing import Optional

_INTEGER_RE = re.compile(r"-?\d+")


def extract_integer(text: Optional[str]) -> Optional[int]:
    """First integer in a VLM reply, or None.

    Handles "4", "There are 4 pillows", and thousands separators ("1,024").
    Separators are stripped before matching so the same string cannot parse as
    1024 here and 1 somewhere else.
    """
    if not text:
        return None
    match = _INTEGER_RE.search(text.replace(",", ""))
    return int(match.group(0)) if match else None
