"""The lenient JSON parsing the local backend depends on.

A 4B model asked for JSON returns it wrapped, prefixed, or with prose trailing it,
and every one of those is a lost call if parsing is strict. These are the shapes
actually observed from Qwen3-VL.
"""
import json

import pytest

from captioner.vlm_backends.base import VLMError, parse_json_object
from captioner.vlm_backends.schemas import CountAnswer, TargetList, json_hint


def test_parses_bare_object():
    result = parse_json_object('{"reason": "two on the sofa", "count": 2}', CountAnswer)
    assert result.count == 2
    assert result.reason == "two on the sofa"


def test_parses_code_fenced_object():
    text = '```json\n{"reason": "one by the window", "count": 1}\n```'
    assert parse_json_object(text, CountAnswer).count == 1


def test_parses_object_after_preamble():
    text = 'Sure! Here is the JSON:\n{"reason": "nothing matches", "count": 0}'
    assert parse_json_object(text, CountAnswer).count == 0


def test_parses_object_with_trailing_prose():
    text = '{"reason": "three chairs", "count": 3}\nLet me know if you need more.'
    assert parse_json_object(text, CountAnswer).count == 3


def test_reason_may_contain_braces_and_quotes():
    """The balanced-brace scan exists for this: a regex truncates or over-reads here."""
    text = 'Here: {"reason": "the label {pillow} is \\"black\\"", "count": 4}'
    result = parse_json_object(text, CountAnswer)
    assert result.count == 4
    assert "{pillow}" in result.reason


def test_rejects_empty_reply():
    with pytest.raises(VLMError):
        parse_json_object("   ", CountAnswer)


def test_rejects_reply_with_no_json():
    with pytest.raises(VLMError):
        parse_json_object("I think there are four pillows.", CountAnswer)


def test_rejects_json_missing_required_field():
    with pytest.raises(VLMError):
        parse_json_object('{"reason": "no number given"}', CountAnswer)


def test_json_hint_names_every_field():
    hint = json_hint(CountAnswer)
    for name in CountAnswer.model_fields:
        assert name in hint


def test_json_hint_shows_list_fields_as_lists():
    """The example is the strongest signal in the hint, so its shape has to be right.

    Shown a string example, a small model replies with a string, which then fails
    validation and costs a retry every single time.
    """
    example = json.loads(json_hint(TargetList).split("keys: ", 1)[1])
    assert isinstance(example["targets"], list)


def test_local_backend_sends_every_view():
    """Multi-view counting only works if the backend actually forwards all the views."""
    from captioner.vlm_backends.qwen_ros_backend import QwenRosBackend

    sent = {}

    def fake_ask_vqa(question, images, max_new_tokens, mode):
        sent["images"] = images
        sent["question"] = question
        return '{"reason": "three chairs", "count": 3}'

    backend = QwenRosBackend(ask_vqa=fake_ask_vqa)
    result = backend.ask("system", "How many chairs?",
                         ["/tmp/a.png", "/tmp/b.png"], CountAnswer)

    assert result.count == 3
    assert sent["images"] == ["/tmp/a.png", "/tmp/b.png"]
    # The question goes last, after the schema hint — see the ordering note in the
    # backend; putting it earlier made a 4B model answer from the prompt's examples.
    assert sent["question"].rstrip().endswith("How many chairs?")


def test_local_backend_retries_once_on_unparseable_reply():
    from captioner.vlm_backends.qwen_ros_backend import QwenRosBackend

    replies = iter(["I think three.", '{"reason": "three", "count": 3}'])

    def fake_ask_vqa(question, images, max_new_tokens, mode):
        return next(replies)

    backend = QwenRosBackend(ask_vqa=fake_ask_vqa)
    assert backend.ask("system", "How many?", [], CountAnswer).count == 3


def test_parses_target_list():
    result = parse_json_object('{"targets": ["glass", "arabic jar"]}', TargetList)
    assert result.targets == ["glass", "arabic jar"]


def test_rejects_target_list_given_a_bare_string():
    with pytest.raises(VLMError):
        parse_json_object('{"targets": "glass"}', TargetList)


@pytest.fixture
def constants_with(monkeypatch):
    """Re-import constants under a given environment, then put it back.

    Everything there resolves at import time, so the only honest way to test the
    provider fallbacks is to reload the module — and to reload it once more on the way
    out, or every later test in the session sees the last one's environment.
    """
    import importlib

    from captioner.vlm_backends import constants

    def load(**env):
        for name in ("VLM_PROVIDER", "VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL",
                     "VLM_MODEL_LITE", "GEMINI_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(constants)

    yield load
    monkeypatch.undo()
    importlib.reload(constants)


def test_provider_preset_supplies_endpoint_and_key(constants_with):
    consts = constants_with(VLM_PROVIDER="dashscope", DASHSCOPE_API_KEY="k")
    assert "dashscope" in consts.VLM_BASE_URL
    assert consts.VLM_API_KEY == "k"
    assert consts.MODEL_NAME


def test_explicit_settings_beat_the_preset(constants_with):
    consts = constants_with(
        VLM_PROVIDER="gemini",
        GEMINI_API_KEY="preset-key",
        VLM_BASE_URL="https://example.test/v1",
        VLM_API_KEY="explicit-key",
        VLM_MODEL="some-model",
    )
    assert consts.VLM_BASE_URL == "https://example.test/v1"
    assert consts.VLM_API_KEY == "explicit-key"
    assert consts.MODEL_NAME == "some-model"


def test_unlisted_provider_needs_everything_spelled_out(constants_with):
    """An unknown name must not silently inherit Gemini's endpoint."""
    consts = constants_with(VLM_PROVIDER="some-new-vendor")
    assert consts.VLM_BASE_URL == ""
    assert consts.MODEL_NAME == ""


def test_lite_model_falls_back_to_the_main_one(constants_with):
    consts = constants_with(VLM_PROVIDER="openrouter", VLM_MODEL="vendor/model")
    assert consts.MODEL_NAME_LITE == "vendor/model"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.test",
    "https://",
    "http://example.test/v1",   # plaintext to a remote host
])
def test_rejects_unusable_base_urls(url, constants_with):
    consts = constants_with()
    with pytest.raises(ValueError):
        consts.checked_base_url(url)


@pytest.mark.parametrize("url", [
    "https://example.test/v1",
    "http://localhost:8000/v1",  # a locally hosted OpenAI-compatible server
])
def test_accepts_usable_base_urls(url, constants_with):
    consts = constants_with()
    assert consts.checked_base_url(url) == url
