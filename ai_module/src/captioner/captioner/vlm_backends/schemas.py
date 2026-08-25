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
from typing import Type, get_args, get_origin

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
    model enumerate what it sees rather than guess a plausible total. It asks for the
    objects and their positions because that is the procedure `ANSWER_SYSTEM` sets out
    — the position of each one is what merges two views into a union instead of a
    maximum, and "one short sentence" here used to contradict the instruction to walk
    the views. Still a few clauses and not paragraphs: the local backend truncates a
    reply at `qwen_ros_backend.MAX_NEW_TOKENS`, which would take the count with it.
    """

    reason: str = Field(
        description="The objects counted and where each one sits, in a few clauses")
    count: int = Field(description="Number of matching objects, 0 or more")


class ObjectChoice(BaseModel):
    """Which candidate object a reference question points at.

    An id rather than a description, because the answer to a category-2 question is a box:
    the id indexes the 3D map the boxes come from, so a chosen id converts to a marker
    without a second round of matching words to objects. Asking for the reason first is what
    makes the model cite the evidence in the candidate list rather than pick the first
    plausible line.
    """

    reason: str = Field(description="One short sentence naming the evidence for this choice")
    object_id: int = Field(
        description="Id of the chosen object, copied exactly from the candidate list")


class RelationCheck(BaseModel):
    """Does one spatial relation hold between two named objects, judged from the pixels.

    The tie-break for candidates the geometry cannot separate: a mapper box is a voxel hull
    and two of them can both plausibly satisfy "on" when only one object is really resting
    on the other.
    """

    reason: str = Field(description="One short sentence describing what the image shows")
    holds: bool = Field(description="True if the stated relation is visibly true")


class RouteWaypoint(BaseModel):
    """One place on an instruction-following route, as the robot will drive it.

    `x`/`y` are what actually reach `/way_point_with_heading`, but `object_ids` is what
    makes them checkable: a waypoint that names the map rows it stands at can be verified
    against those rows' centres and corrected, where a bare coordinate can only be trusted.
    `why` carries the words of the command it satisfies, which is how a wrong route is read
    as a misparse rather than a misgrounding after the fact.
    """

    role: str = Field(description="One of: pass | goal")
    x: float = Field(description="Map-frame x of the place to drive to, in metres")
    y: float = Field(description="Map-frame y of the place to drive to, in metres")
    object_ids: list[str] = Field(
        description="Ids of the mapped objects this waypoint stands at, copied from the "
                    "object table; empty only for a detour point in open floor")
    why: str = Field(
        description="The words from the command that this waypoint satisfies")


class RoutePlan(BaseModel):
    """The whole route for one instruction-following command, in driving order.

    Asking for `reason` before the waypoints is what makes the model segment the command
    aloud rather than emit the first plausible coordinates — the same trick `CountAnswer`
    and `ObjectChoice` use.
    """

    reason: str = Field(
        description="One short sentence: the places the command names, in order")
    waypoints: list[RouteWaypoint] = Field(
        description="The route in order, one entry per place, with the goal last")


def _example_object(model: Type[BaseModel]) -> dict:
    """One nested model rendered as a filled-in example object."""
    return {name: _example_value(name, field)
            for name, field in model.model_fields.items()}


def _example_value(name: str, field) -> object:
    """The description, wrapped so the example has the field's own shape.

    A list field shown as `"targets": "some words"` gets answered with a string,
    which then fails validation — the example is the strongest signal in the hint,
    so its brackets have to be real. The same argument applies one level down: a
    `list[SubModel]` rendered as `["ordered steps"]` teaches a list of STRINGS,
    and the reply then fails validation for the shape the hint itself asked for.
    So a nested model is expanded rather than described.
    """
    text = field.description or name
    annotation = field.annotation
    if get_origin(annotation) in (list, set, tuple):
        args = get_args(annotation)
        inner = args[0] if args else None
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [_example_object(inner)]
        return [text]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _example_object(annotation)
    return text


def json_hint(schema: Type[BaseModel]) -> str:
    """A compact 'reply with exactly this shape' instruction for text-only backends.

    Field descriptions are inlined as the example values: a 4B model follows a
    filled-in example far more reliably than it follows a JSON Schema document,
    and the schema document alone would spend a few hundred tokens saying less.
    """
    return (
        "Reply with ONLY a JSON object, no prose and no code fence, "
        f"with exactly these keys: {json.dumps(_example_object(schema))}"
    )
