"""Backend interface and the lenient JSON parsing both backends fall back on.

Deliberately free of ROS, torch and openai imports so the parsing rules — the part
most likely to break on a new model — stay unit-testable.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# ```json ... ``` or bare ``` ... ``` — small models wrap JSON in a fence even when
# told not to, and stripping it is cheaper than a retry.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class VLMError(RuntimeError):
    """A backend could not produce a valid answer."""


def _balanced_object(text: str) -> str | None:
    """Extract the first brace-balanced {...} run, ignoring braces inside strings.

    A regex cannot do this: `reason` fields routinely contain braces and quotes,
    and a non-greedy `\\{.*?\\}` truncates at the first inner brace while a greedy
    one swallows trailing prose.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_object(text: str, schema: Type[T]) -> T:
    """Coerce model text into `schema`, or raise VLMError.

    Tries the raw text, then the contents of a code fence, then the first
    brace-balanced object anywhere in the reply — which is what catches the
    common "Sure, here is the JSON: {...}" preamble.
    """
    if not text or not text.strip():
        raise VLMError("empty reply")

    stripped = text.strip()
    candidates = [stripped]
    fence = _FENCE_RE.search(stripped)
    if fence:
        candidates.append(fence.group(1))
    balanced = _balanced_object(stripped)
    if balanced:
        candidates.append(balanced)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = TypeError(f"expected a JSON object, got {type(payload).__name__}")
            continue
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            last_error = exc
    raise VLMError(f"could not parse {schema.__name__} from reply: {last_error}")


class VLMBackend(ABC):
    """One call: a system instruction, a user turn, some images, a shape to return."""

    name: str = "vlm"

    @abstractmethod
    def ask(
        self,
        system: str,
        user: str,
        images: Sequence[Path],
        schema: Type[T],
        *,
        lite: bool = False,
    ) -> T:
        """Return a validated `schema` instance, or raise VLMError.

        `lite` requests the cheaper/faster model where the backend has one; it is a
        hint, not a guarantee.
        """
