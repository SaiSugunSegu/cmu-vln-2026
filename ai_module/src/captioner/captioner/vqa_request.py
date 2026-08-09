"""Pure parsing for /qwen_vqa/request JSON payloads (no ROS / torch)."""
from __future__ import annotations

import json
from typing import Optional

# A malformed or hostile request must not be able to pin the GPU for minutes.
MAX_NEW_TOKENS_LIMIT = 512


def parse_vqa_request(
        raw: str,
        ) -> tuple[
            Optional[str], str, list[Optional[str]], bool, Optional[int], Optional[str]]:
    """Parse a VQA request JSON string.

    Returns
    ``(id, question, image_paths, freeform, max_new_tokens, instruction)``.

    ``image_paths`` is a list: multiple paths from ``images``, a single path from
    ``image``, or ``[None]`` for text-only (blank RGB placeholder). Prefer
    ``images`` when present and non-empty.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    req_id = payload.get("id")
    if "question" not in payload:
        raise ValueError("need keys: id, question, and image or images")

    question = payload["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    images_field = payload.get("images")
    image_paths: list[Optional[str]]
    if images_field is not None:
        if not isinstance(images_field, list) or not images_field:
            raise ValueError("images must be a non-empty list of path strings")
        if not all(isinstance(p, str) and p.strip() for p in images_field):
            raise ValueError("images entries must be non-empty strings")
        image_paths = list(images_field)
    elif "image" in payload:
        image_path = payload["image"]
        if image_path is not None and not isinstance(image_path, str):
            raise ValueError("image must be a string path or null")
        image_paths = [image_path]
    else:
        raise ValueError("need keys: id, question, and image or images")

    mode = str(payload.get("mode") or "numerical").lower()
    freeform = mode in ("freeform", "text", "extract")

    instruction = payload.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str):
            raise ValueError("instruction must be a string when provided")
        instruction = instruction.strip() or None

    max_new_tokens = payload.get("max_new_tokens")
    if max_new_tokens is not None:
        max_new_tokens = int(max_new_tokens)
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise ValueError(
                f"max_new_tokens must be in 1..{MAX_NEW_TOKENS_LIMIT}")

    return req_id, question, image_paths, freeform, max_new_tokens, instruction
