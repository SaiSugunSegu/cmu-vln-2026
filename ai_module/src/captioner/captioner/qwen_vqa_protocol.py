"""The /qwen_vqa wire contract: topic names, and how a request is written and read.

Both directions live here so they cannot drift, and the module stays stdlib-only so
the parsing rules can be unit-tested without rclpy, torch or a GPU — the server that
uses them imports all three.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

STATUS_TOPIC = "/qwen_vqa/status"
REQUEST_TOPIC = "/qwen_vqa/request"
RESPONSE_TOPIC = "/qwen_vqa/response"

# A malformed or hostile request must not be able to pin the GPU for minutes.
MAX_NEW_TOKENS_LIMIT = 512

# Each view costs a full max_pixels worth of visual tokens, so the cap is about VRAM
# and latency, not correctness. Four covers the best-view manifests this serves (top_n
# is 3 by default, 10 at its widest) while keeping one request from evicting SAM 3.
MAX_IMAGES_PER_REQUEST = 4

# mode: "numerical" (default) wraps the question in an integer-only template server
# side; anything in this set sends the text through untouched.
_FREEFORM_MODES = ("freeform", "text", "extract")


def vqa_image_fields(images: Sequence) -> dict:
    """The image half of a request payload, in the spelling the server expects.

    One view keeps using the older `image` field, which is what every pre-existing
    client and log line already speaks; more than one switches to `images`.
    """
    paths = [str(p) for p in images]
    if len(paths) > 1:
        return {"images": paths}
    return {"image": paths[0] if paths else None}


def parse_vqa_request(raw: str) -> tuple[Optional[str], str, list, bool, Optional[int]]:
    """(id, question, image_paths, freeform, max_new_tokens) or raise ValueError.

    `image_paths` is normalised to a list whichever field the client used, so callers
    can stop caring. It may be empty: a null image is how text-only prompts (target
    extraction, JSON planning) reach the server.

    Paths are not resolved or opened here — that is the server's job, through
    `secure_image_path`, and keeping it there is what lets this stay import-light.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    req_id = payload.get("id")
    has_image = "image" in payload
    has_images = "images" in payload
    if not (has_image or has_images) or "question" not in payload:
        raise ValueError("need keys: id, question, and image (may be null) or images")
    if has_image and has_images:
        raise ValueError("send image or images, not both")

    question = payload["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if has_images:
        image_paths = payload["images"]
        if not isinstance(image_paths, list) or not all(
                isinstance(p, str) for p in image_paths):
            raise ValueError("images must be a list of string paths")
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"images holds {len(image_paths)}, limit is {MAX_IMAGES_PER_REQUEST}")
    else:
        image_path = payload["image"]
        if image_path is not None and not isinstance(image_path, str):
            raise ValueError("image must be a string path or null")
        image_paths = [image_path] if image_path else []

    freeform = str(payload.get("mode") or "numerical").lower() in _FREEFORM_MODES

    max_new_tokens = payload.get("max_new_tokens")
    if max_new_tokens is not None:
        max_new_tokens = int(max_new_tokens)
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise ValueError(f"max_new_tokens must be in 1..{MAX_NEW_TOKENS_LIMIT}")

    return req_id, question, image_paths, freeform, max_new_tokens
