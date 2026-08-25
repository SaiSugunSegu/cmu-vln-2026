"""The lenient JSON parsing the local backend depends on.

A 4B model asked for JSON returns it wrapped, prefixed, or with prose trailing it,
and every one of those is a lost call if parsing is strict. These are the shapes
actually observed from Qwen3-VL.
"""
import json
import os
from types import SimpleNamespace

import pytest
import yaml

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


class _BadRequest(Exception):
    """Stands in for openai.BadRequestError, which the host does not have installed."""


def _completion(content, parsed=None):
    message = SimpleNamespace(content=content, parsed=parsed)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIClient:
    def __init__(self, on_parse, create_reply=""):
        self._on_parse = on_parse
        self._create_reply = create_reply
        self.create_calls = []
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse)))
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _parse(self, **kwargs):
        if isinstance(self._on_parse, Exception):
            raise self._on_parse
        return self._on_parse

    def _create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _completion(self._create_reply)


def _cloud_backend(client):
    """An OpenAIBackend around a fake client, without importing the openai SDK.

    __init__ builds a real client and reads the environment, neither of which this is
    about, and the SDK is not installed on the host where these tests run.
    """
    from captioner.vlm_backends.openai_backend import OpenAIBackend

    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.name = "cloud:test"
    backend._client = client
    backend._log = lambda _msg: None
    backend._bad_request = _BadRequest
    return backend


def test_cloud_backend_returns_constrained_output_without_a_second_call():
    client = _FakeOpenAIClient(_completion('{"reason": "two", "count": 2}',
                                           parsed=CountAnswer(reason="two", count=2)))
    result = _cloud_backend(client).ask("system", "How many?", [], CountAnswer)

    assert result.count == 2
    assert client.create_calls == []


@pytest.mark.parametrize("failure", [
    # Endpoint ignored response_format and answered in prose: the SDK validates the
    # reply text itself, so the failure surfaces as a ValidationError from `parse`.
    CountAnswer.model_validate_json,
    # Endpoint rejected the json_schema response_format outright.
    None,
])
def test_cloud_backend_falls_back_when_the_schema_is_not_honoured(failure):
    from pydantic import ValidationError

    if failure is None:
        error: Exception = _BadRequest("response_format is not supported")
    else:
        try:
            failure("Here are three chairs.")
            raise AssertionError("expected a ValidationError")
        except ValidationError as exc:
            error = exc

    client = _FakeOpenAIClient(error, create_reply='```json\n{"reason": "3", "count": 3}\n```')
    result = _cloud_backend(client).ask("Count them.", "How many chairs?", [], CountAnswer)

    assert result.count == 3
    assert len(client.create_calls) == 1
    # The shape has to reach the model some other way once the parameter is gone, and
    # the question must still come last — same ordering rule as the local backend.
    system = client.create_calls[0]["messages"][0]["content"]
    assert "count" in system and "reason" in system


def test_cloud_backend_does_not_retry_an_auth_or_transport_failure():
    """A second call there would bill twice and fail the same way."""
    from captioner.vlm_backends.base import VLMError as CloudError

    client = _FakeOpenAIClient(RuntimeError("401 invalid api key"))
    with pytest.raises(CloudError):
        _cloud_backend(client).ask("system", "How many?", [], CountAnswer)

    assert client.create_calls == []


def test_parses_target_list():
    result = parse_json_object('{"targets": ["glass", "arabic jar"]}', TargetList)
    assert result.targets == ["glass", "arabic jar"]


def test_parses_bare_target_list():
    """The extract prompt asks for a bracket list, not a JSON object."""
    result = parse_json_object('["pillow with black stripes", "couch"]', TargetList)
    assert result.targets == ["pillow with black stripes", "couch"]


def test_parses_bare_target_list_with_preamble():
    result = parse_json_object(
        'Sure, here you go:\n["red box", "chair", "desk"]\n', TargetList)
    assert result.targets == ["red box", "chair", "desk"]


def test_rejects_target_list_given_a_bare_string():
    with pytest.raises(VLMError):
        parse_json_object('{"targets": "glass"}', TargetList)


@pytest.fixture
def constants_with(tmp_path, monkeypatch):
    """Re-import constants under a given vqa.yaml + environment, then put it back.

    Everything there resolves at import time, so the only honest way to test the
    provider fallbacks is to reload the module — and to reload it once more on the way
    out, or every later test in the session sees the last one's config/environment.
    Non-secret settings (`provider`, `base_url`, `model`, ...) go in the YAML `config`
    dict, exactly like the real vqa.yaml; only API keys are still environment vars,
    since that is the one thing constants.py still reads from the environment.
    """
    import importlib

    from captioner.vlm_backends import constants

    def load(config=None, **env):
        for name in ("VLM_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
                     "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        config_path = tmp_path / "vqa.yaml"
        config_path.write_text(yaml.safe_dump(config or {}))
        monkeypatch.setenv("CAPTIONER_VQA_CONFIG", str(config_path))
        return importlib.reload(constants)

    yield load
    monkeypatch.undo()
    importlib.reload(constants)


def test_provider_preset_supplies_endpoint_and_key(constants_with):
    consts = constants_with({"provider": "dashscope"}, DASHSCOPE_API_KEY="k")
    assert "dashscope" in consts.VLM_BASE_URL
    assert consts.VLM_API_KEY == "k"
    assert consts.MODEL_NAME


def test_anthropic_preset_points_at_the_compatibility_layer(constants_with):
    """Claude is reachable with a key alone — /v1/ is the OpenAI-compatible path."""
    consts = constants_with({"provider": "anthropic"}, ANTHROPIC_API_KEY="k")
    assert consts.VLM_BASE_URL == "https://api.anthropic.com/v1/"
    assert consts.VLM_API_KEY == "k"
    assert consts.MODEL_NAME.startswith("claude-")


def test_explicit_settings_beat_the_preset(constants_with):
    consts = constants_with(
        {"provider": "gemini", "base_url": "https://example.test/v1", "model": "some-model"},
        GEMINI_API_KEY="preset-key",
        VLM_API_KEY="explicit-key",
    )
    assert consts.VLM_BASE_URL == "https://example.test/v1"
    assert consts.VLM_API_KEY == "explicit-key"
    assert consts.MODEL_NAME == "some-model"


def test_unlisted_provider_needs_everything_spelled_out(constants_with):
    """An unknown name must not silently inherit Gemini's endpoint."""
    consts = constants_with({"provider": "some-new-vendor"})
    assert consts.VLM_BASE_URL == ""
    assert consts.MODEL_NAME == ""


def test_lite_model_falls_back_to_the_main_one(constants_with):
    consts = constants_with({"provider": "openrouter", "model": "vendor/model"})
    assert consts.MODEL_NAME_LITE == "vendor/model"


def test_invalid_backends_default_to_cloud_independently(constants_with):
    consts = constants_with({})
    assert consts.VLM_BACKEND == "cloud"
    assert consts.TARGET_EXTRACT_BACKEND == "cloud"
    assert consts.NEED_LOCAL_VQA is False
    consts = constants_with({"vlm_backend": "bogus", "target_extract_backend": "bogus"})
    assert consts.VLM_BACKEND == "cloud"
    assert consts.TARGET_EXTRACT_BACKEND == "cloud"
    consts = constants_with({"vlm_backend": "cloud", "target_extract_backend": "local"})
    assert consts.VLM_BACKEND == "cloud"
    assert consts.TARGET_EXTRACT_BACKEND == "local"
    assert consts.NEED_LOCAL_VQA is True
    assert consts.local_vqa_launch_flag() == "true"


def test_invalid_view_source_falls_back_to_silhouette(constants_with):
    consts = constants_with({
        "view_source_numerical": "bogus",
        "view_source_object_reference": "bogus",
        "view_source_instruction_following": "bogus",
    })
    assert consts.VIEW_SOURCE_NUMERICAL == "silhouette"
    assert consts.VIEW_SOURCE_OBJECT_REFERENCE == "silhouette"
    assert consts.VIEW_SOURCE_INSTRUCTION_FOLLOWING == "silhouette"


def test_missing_view_source_falls_back_to_silhouette(constants_with):
    consts = constants_with({})
    assert consts.VIEW_SOURCE_NUMERICAL == "silhouette"
    assert consts.VIEW_SOURCE_OBJECT_REFERENCE == "silhouette"
    assert consts.VIEW_SOURCE_INSTRUCTION_FOLLOWING == "silhouette"


def test_valid_target_extract_backend_and_view_source_pass_through(constants_with):
    consts = constants_with({"target_extract_backend": "local", "view_source_numerical": "crop"})
    assert consts.TARGET_EXTRACT_BACKEND == "local"
    assert consts.VIEW_SOURCE_NUMERICAL == "crop"


def test_view_source_keys_are_independent_per_category(constants_with):
    """Each category's key is read and validated on its own -- no shared fallback."""
    consts = constants_with({
        "view_source_numerical": "crop",
        "view_source_object_reference": "full",
        "view_source_instruction_following": "full_silhouette",
    })
    assert consts.VIEW_SOURCE_NUMERICAL == "crop"
    assert consts.VIEW_SOURCE_OBJECT_REFERENCE == "full"
    assert consts.VIEW_SOURCE_INSTRUCTION_FOLLOWING == "full_silhouette"


def test_resolve_config_path_defaults_to_the_bundled_file(monkeypatch):
    """No CAPTIONER_VQA_CONFIG override -> the config/vqa.yaml shipped with the package."""
    monkeypatch.delenv("CAPTIONER_VQA_CONFIG", raising=False)
    from captioner.vlm_backends.config import resolve_config_path

    path = resolve_config_path()
    assert os.path.join("config", "vqa.yaml") in path
    assert os.path.isfile(path)


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
