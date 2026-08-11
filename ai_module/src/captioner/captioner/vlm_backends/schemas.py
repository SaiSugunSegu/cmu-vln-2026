"""Structured outputs the VLM is asked to produce.

Pydantic rather than free text because both backends need the same contract from
opposite directions: the cloud backend hands these straight to
`chat.completions.parse(response_format=...)` for constrained decoding, while the
local backend renders `json_hint()` into the prompt and validates whatever comes
back. One definition, so the two paths cannot drift.

No ROS or torch imports — these are unit-testable on their own.
"""
from __future__ import annotations

import json
from typing import Type, get_origin

from pydantic import BaseModel, Field


class TargetList(BaseModel):
    """The nouns a detector should be armed with to answer a question.

    The whole list goes to SAM as prompts, landmarks included: a question about
    pillows on a sofa needs the sofa detected too, or there is nothing to judge
    "on" against.
    """

    targets: list[str] = Field(
        description='Object nouns to detect, e.g. ["glass", "arabic jar"]')


class CountAnswer(BaseModel):
    """The count itself, read off several views of one room at once.

    `reason` is not decoration: asking for it before the number is what makes the
    model enumerate what it sees rather than guess a plausible total.
    """

    reason: str = Field(description="One short sentence explaining the count")
    count: int = Field(description="Number of matching objects, 0 or more")


def _example_value(name: str, field) -> object:
    """The description, wrapped so the example has the field's own shape.

    A list field shown as `"targets": "some words"` gets answered with a string,
    which then fails validation — the example is the strongest signal in the hint,
    so its brackets have to be real.
    """
    text = field.description or name
    origin = get_origin(field.annotation)
    if origin in (list, set, tuple):
        return [text]
    return text


def json_hint(schema: Type[BaseModel]) -> str:
    """A compact 'reply with exactly this shape' instruction for text-only backends.

    Field descriptions are inlined as the example values: a 4B model follows a
    filled-in example far more reliably than it follows a JSON Schema document,
    and the schema document alone would spend a few hundred tokens saying less.
    """
    example = {
        name: _example_value(name, field)
        for name, field in schema.model_fields.items()
    }
    return (
        "Reply with ONLY a JSON object, no prose and no code fence, "
        f"with exactly these keys: {json.dumps(example)}"
    )
