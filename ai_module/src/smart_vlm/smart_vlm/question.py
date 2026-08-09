"""Challenge-question parsing (no ROS / GPU imports).

Lives apart from the supervisor node so both smart_vlm and numerical_reasoner can
classify a question without importing rclpy, and so the heuristics stay unit-testable.
"""
from __future__ import annotations

import json
from enum import Enum

__all__ = ["QuestionType", "classify", "question_text"]


class QuestionType(str, Enum):
    """The three scored categories (README 'Question Types and Initial Scoring')."""

    NUMERICAL = "numerical"                        # Int32 on /numerical_response
    OBJECT_REFERENCE = "object_reference"          # Marker on /selected_object_marker
    INSTRUCTION_FOLLOWING = "instruction_following"  # Pose2D on /way_point_with_heading


def question_text(raw: str) -> str:
    """The question string, unwrapped from a JSON envelope if there is one.

    The challenge evaluation node publishes a bare String, but our own harnesses
    sometimes wrap it as {"question": ..., "id": ...}; accept both.
    """
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()
    if isinstance(payload, dict):
        return str(payload.get("question", raw)).strip()
    return raw.strip()


def classify(raw: str) -> QuestionType:
    """Route a question to its answer head.

    Prefix heuristics, not a model: the three categories are lexically distinct in
    every example the organisers publish, and a misroute costs a whole question.
    Instruction-following is the default because it is the only category whose
    statements do not open with a fixed marker.
    """
    text = question_text(raw).lower().strip()
    if text.startswith(("how many", "count")):
        return QuestionType.NUMERICAL
    if text.startswith(("find", "the ")):
        return QuestionType.OBJECT_REFERENCE
    return QuestionType.INSTRUCTION_FOLLOWING
