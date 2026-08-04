"""Unit tests for the shared integer parser (no ROS / GPU / torch)."""
from __future__ import annotations

from captioner.text_utils import extract_integer


def test_bare_integer():
    assert extract_integer("4") == 4


def test_integer_in_a_sentence():
    assert extract_integer("There are 4 pillows on the bed.") == 4


def test_thousands_separator():
    # The reasoner and the VQA server both parse replies; when one stripped
    # separators and the other did not, this string gave 1024 or 1.
    assert extract_integer("1,024") == 1024


def test_negative():
    assert extract_integer("-3 items") == -3


def test_no_digits():
    assert extract_integer("no number here") is None


def test_empty_and_none():
    assert extract_integer("") is None
    assert extract_integer(None) is None
