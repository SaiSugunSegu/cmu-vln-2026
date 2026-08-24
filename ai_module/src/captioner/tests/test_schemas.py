"""`json_hint` renders the shape a text-only backend has to answer in.

The hint is the strongest signal in a local prompt, so an example of the wrong shape is worse
than no example: the model follows it, and the reply then fails the validation the hint itself
was describing. These pin that the example matches the schema at every level.
"""
import json

import pytest

pytest.importorskip("pydantic")

from pydantic import BaseModel, Field  # noqa: E402

from captioner.vlm_backends.schemas import (  # noqa: E402
    CountAnswer,
    RoutePlan,
    TargetList,
    json_hint,
)


def example_of(schema) -> dict:
    """The JSON object out of a hint, so a test asserts on shape rather than on prose."""
    hint = json_hint(schema)
    return json.loads(hint[hint.index("{"):])


def test_a_list_of_models_is_rendered_as_a_list_of_objects():
    """The regression the route call needs: a nested list used to render as a list of strings."""
    waypoints = example_of(RoutePlan)["waypoints"]
    assert isinstance(waypoints, list) and waypoints
    assert isinstance(waypoints[0], dict), "a list[SubModel] example must contain an OBJECT"
    assert set(waypoints[0]) == {"role", "x", "y", "object_ids", "why"}


def test_a_list_of_strings_inside_a_nested_model_stays_a_list():
    assert isinstance(example_of(RoutePlan)["waypoints"][0]["object_ids"], list)


def test_a_flat_list_field_is_unchanged():
    targets = example_of(TargetList)["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], str)


def test_a_flat_scalar_schema_is_unchanged():
    assert set(example_of(CountAnswer)) == {"reason", "count"}


def test_the_hint_still_forbids_prose_and_fences():
    hint = json_hint(RoutePlan)
    assert "no prose" in hint and "no code fence" in hint


def test_the_rendered_example_names_every_field_of_the_schema():
    """A field missing from the example is a field the model will not be asked for."""
    example = example_of(RoutePlan)
    assert set(example) == set(RoutePlan.model_fields)


def test_a_bare_nested_model_is_expanded_too():
    class Inner(BaseModel):
        note: str = Field(description="a note")

    class Outer(BaseModel):
        inner: Inner = Field(description="the inner one")

    assert example_of(Outer)["inner"] == {"note": "a note"}
