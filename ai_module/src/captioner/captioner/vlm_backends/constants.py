"""Environment-driven VLM configuration.

VLM_BACKEND = cloud | local — WHERE inference runs.

`cloud` is the scored path: a submission may use an externally hosted model, and any
OpenAI-compatible endpoint will do. `local` runs the resident Qwen server instead,
which costs nothing per call, so it is the development loop and the default here —
a stray run should never bill credits.

Pick a hosted provider with VLM_PROVIDER (see PROVIDERS below); that only supplies
defaults, and each one is individually overridable:

  VLM_PROVIDER    gemini | dashscope | openrouter | openai | <anything, with the below>
  VLM_BASE_URL    OpenAI-compatible base URL
  VLM_API_KEY     key, if you would rather not use the provider's own variable
  VLM_MODEL       model id
  VLM_MODEL_LITE  cheaper model for the `lite=True` calls; falls back to VLM_MODEL

So a provider nobody has listed here needs no code change — set VLM_BASE_URL,
VLM_API_KEY and VLM_MODEL and it works. Nothing is validated at import time: a
local-only run must not fail because a cloud variable is malformed.
"""
import os
from urllib.parse import urlparse

VLM_BACKEND = os.environ.get("VLM_BACKEND", "").strip().lower()
if VLM_BACKEND not in ("local", "cloud"):
    VLM_BACKEND = "local"

# provider -> (base URL, its own API-key variable, model, lite model)
#
# Models are pinned to explicit versions rather than `-latest` aliases: a benchmark
# number is only comparable to an earlier one if the model behind it did not change
# underneath. The cost is that a retirement breaks the default rather than sliding past
# unnoticed — Gemini's 2.5 pair that used to sit here now 404s for new keys while still
# appearing in models.list(), so a model being listed does not mean it can be called.
#
# The two aggregators are left without a default model on purpose. Their ids are
# vendor-namespaced and change constantly, so guessing one would fail at call time with
# a confusing 404 instead of the clear "set VLM_MODEL" error the backend raises.
PROVIDERS = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ),
    "dashscope": (
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        "qwen3.6-plus",
        "qwen3.6-flash",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "",
        "",
    ),
    "openai": (
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "",
        "",
    ),
}

VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "gemini").strip().lower()
# An unrecognised name is not an error: it is how you reach a provider that is not
# listed, by supplying the three variables yourself. The backend reports what is
# missing if you only supply some of them.
_base_url, VLM_API_KEY_ENV, _model, _model_lite = PROVIDERS.get(
    VLM_PROVIDER, ("", "VLM_API_KEY", "", ""))

VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "").strip() or _base_url
VLM_API_KEY = os.environ.get("VLM_API_KEY", "") or os.environ.get(VLM_API_KEY_ENV, "")
MODEL_NAME = os.environ.get("VLM_MODEL", "").strip() or _model
MODEL_NAME_LITE = (
    os.environ.get("VLM_MODEL_LITE", "").strip() or _model_lite or MODEL_NAME)


def checked_base_url(url: str) -> str:
    """Reject anything that is not plain HTTP(S) to a named host.

    VLM_BASE_URL is operator configuration rather than request input, but it still ends
    up as the destination of every outbound call, so a typo or a copy-pasted `file://`
    should fail here with an explanation instead of somewhere inside the SDK. Plaintext
    HTTP is allowed only for loopback, where there is no network to eavesdrop on and
    where a locally hosted OpenAI-compatible server would live.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            f"VLM_BASE_URL must be an http(s) URL with a host, got {url!r}")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"VLM_BASE_URL must use https for a remote host, got {url!r}")
    return url
