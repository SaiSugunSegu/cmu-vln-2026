"""Pure-python tests for question routing (no ROS / GPU)."""
from __future__ import annotations

from smart_vlm.question import QuestionType, classify, question_text


def test_classify_numerical():
    assert classify("How many blue chairs are between the table and the wall?") \
        is QuestionType.NUMERICAL
    assert classify("Count the black trash cans near the window") is QuestionType.NUMERICAL


def test_classify_object_reference():
    assert classify("Find the potted plant on the kitchen island closest to the fridge.") \
        is QuestionType.OBJECT_REFERENCE
    assert classify("The orange chair between the table and sink") \
        is QuestionType.OBJECT_REFERENCE


def test_classify_instruction_following():
    assert classify("Take the path near the window to the fridge.") \
        is QuestionType.INSTRUCTION_FOLLOWING
    assert classify("Avoid the path between the two tables and go near the blue trash can") \
        is QuestionType.INSTRUCTION_FOLLOWING


def test_classify_is_case_and_whitespace_insensitive():
    assert classify("   HOW MANY glasses are there?  ") is QuestionType.NUMERICAL


def test_classify_unwraps_json_envelope():
    assert classify('{"id": "Q01", "question": "How many glasses?"}') \
        is QuestionType.NUMERICAL


def test_question_text_plain_string():
    assert question_text("  How many glasses?  ") == "How many glasses?"


def test_question_text_json_envelope():
    assert question_text('{"question": "Find the red chair", "id": "Q02"}') \
        == "Find the red chair"


def test_question_text_survives_malformed_json():
    # A bare question containing braces must not be swallowed by the JSON path.
    assert question_text("How many {things} are there?") == "How many {things} are there?"


def test_question_text_empty():
    assert question_text("") == ""


def test_question_type_serialises_as_its_value():
    # str-Enum, so it drops straight into a JSON status payload.
    assert QuestionType.NUMERICAL == "numerical"
