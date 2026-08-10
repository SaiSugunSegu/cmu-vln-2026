"""The /qwen_vqa request contract, read and written.

Both spellings of the image field have to keep working at once: the multi-view
reasoner sends `images`, while qwen_vqa_client, qwen_numerical and every per-instance
verification call still send `image`. A change that breaks either one shows up as a
rejected request at runtime, in a node that then has no answer to publish.
"""
import json

import pytest

from captioner.qwen_vqa_protocol import (
    MAX_IMAGES_PER_REQUEST,
    parse_vqa_request,
    vqa_image_fields,
)


def _raw(**payload) -> str:
    return json.dumps({"id": "abc", "question": "How many chairs?", **payload})


def test_single_image_field_still_accepted():
    _, _, images, _, _ = parse_vqa_request(_raw(image="/tmp/a.png"))
    assert images == ["/tmp/a.png"]


def test_null_image_means_text_only():
    _, _, images, _, _ = parse_vqa_request(_raw(image=None))
    assert images == []


def test_images_list_accepted():
    _, _, images, _, _ = parse_vqa_request(_raw(images=["/tmp/a.png", "/tmp/b.png"]))
    assert images == ["/tmp/a.png", "/tmp/b.png"]


def test_empty_images_list_is_text_only():
    _, _, images, _, _ = parse_vqa_request(_raw(images=[]))
    assert images == []


def test_over_long_images_list_rejected():
    too_many = [f"/tmp/{i}.png" for i in range(MAX_IMAGES_PER_REQUEST + 1)]
    with pytest.raises(ValueError, match="limit is"):
        parse_vqa_request(_raw(images=too_many))


def test_both_image_fields_rejected():
    with pytest.raises(ValueError, match="not both"):
        parse_vqa_request(_raw(image="/tmp/a.png", images=["/tmp/b.png"]))


def test_neither_image_field_rejected():
    with pytest.raises(ValueError, match="need keys"):
        parse_vqa_request(json.dumps({"id": "abc", "question": "How many?"}))


def test_non_string_image_entries_rejected():
    with pytest.raises(ValueError, match="list of string paths"):
        parse_vqa_request(_raw(images=["/tmp/a.png", 7]))


def test_images_must_be_a_list():
    with pytest.raises(ValueError, match="list of string paths"):
        parse_vqa_request(_raw(images="/tmp/a.png"))


def test_empty_question_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        parse_vqa_request(json.dumps({"id": "abc", "question": "  ", "image": None}))


def test_mode_defaults_to_numerical():
    _, _, _, freeform, _ = parse_vqa_request(_raw(image=None))
    assert freeform is False


def test_freeform_mode_recognised():
    _, _, _, freeform, _ = parse_vqa_request(_raw(image=None, mode="freeform"))
    assert freeform is True


def test_max_new_tokens_bounds_enforced():
    with pytest.raises(ValueError, match="max_new_tokens"):
        parse_vqa_request(_raw(image=None, max_new_tokens=100000))


# -- the writing side ------------------------------------------------------

@pytest.mark.parametrize("images, expected", [
    ([], {"image": None}),
    (["/tmp/a.png"], {"image": "/tmp/a.png"}),
    (["/tmp/a.png", "/tmp/b.png"], {"images": ["/tmp/a.png", "/tmp/b.png"]}),
])
def test_image_fields_spelling(images, expected):
    assert vqa_image_fields(images) == expected


def test_image_fields_stringifies_paths():
    from pathlib import Path
    assert vqa_image_fields([Path("/tmp/a.png")]) == {"image": "/tmp/a.png"}


@pytest.mark.parametrize("images", [
    [],
    ["/tmp/a.png"],
    ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"],
])
def test_round_trip(images):
    """Whatever the writer emits, the parser reads back as the same list."""
    raw = json.dumps({"id": "abc", "question": "How many?", **vqa_image_fields(images)})
    _, _, parsed, _, _ = parse_vqa_request(raw)
    assert parsed == images
