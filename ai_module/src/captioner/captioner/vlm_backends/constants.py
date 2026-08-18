"""VLM configuration, read from config/vqa.yaml plus API keys from the environment.

VLM_BACKEND = cloud | local — WHERE inference runs.

`cloud` is the scored path: a submission may use an externally hosted model, and any
OpenAI-compatible endpoint will do. `local` runs the resident Qwen server instead,
which costs nothing per call, so it is the development loop and the default here —
a stray run should never bill credits.

Pick a hosted provider with `provider` in vqa.yaml (see PROVIDERS below); that only
supplies defaults, and each one is individually overridable there too:

  provider     gemini | anthropic | dashscope | openrouter | openai | <anything, below>
  base_url     OpenAI-compatible base URL
  model        model id
  model_lite   cheaper model for the `lite=True` calls; falls back to `model`

So a provider nobody has listed here needs no code change — set base_url and model in
vqa.yaml and it works. Nothing is validated at import time: a local-only run must not
fail because a cloud setting is malformed.

API keys are the one thing NOT read from vqa.yaml: that file is committed to git and
shipped in the submission image, so GEMINI_API_KEY / DASHSCOPE_API_KEY / etc. (or the
generic VLM_API_KEY override) stay in the environment (.env / docker compose env_file).

Everything else here — backend, extract_backend, provider, base_url, model, model_lite,
view_source, the silhouette wait/poll timings — lives in
ai_module/src/captioner/config/vqa.yaml. See captioner/vlm_backends/config.py for how
that file is found and loaded.
"""
import os
from urllib.parse import urlparse

from captioner.vlm_backends.config import load_vqa_config

_CONFIG = load_vqa_config()

VLM_BACKEND = str(_CONFIG.get("backend", "local")).strip().lower()
if VLM_BACKEND not in ("local", "cloud"):
    VLM_BACKEND = "local"

# auto | local | cloud — backend for the text-only target-extraction call the reasoners
# fire before SAM is armed. Left unresolved here on purpose: "auto" means "whatever this
# run's main backend is", and only the reasoner (which may have that overridden by its
# own `backend` ROS parameter) knows what that is. See numerical_reasoner.py /
# object_reference_reasoner.py for the resolution.
EXTRACT_BACKEND = str(_CONFIG.get("extract_backend", "auto")).strip().lower()
if EXTRACT_BACKEND not in ("auto", "local", "cloud"):
    EXTRACT_BACKEND = "auto"

# Which image of a best-view crop the model sees: `silhouette` is the mask-outline +
# label copy sam_node writes next to each crop (see sam_mapper's `save_silhouette_copy`);
# `crop` is the plain, unannotated photo. Read by both category-1
# (`numerical_utils.select_context_views`) and category-2 (`cat2_utils.marked_views`),
# live reasoners and their offline benches alike.
VIEW_SOURCE = str(_CONFIG.get("view_source", "silhouette")).strip().lower()
if VIEW_SOURCE not in ("crop", "silhouette"):
    VIEW_SOURCE = "silhouette"

# How long a view_source: silhouette run waits for sam_node to finish writing the
# finalized copy before falling back to the plain crop, and how often it polls while
# waiting. Both reasoners fire on the same /pipeline/explore_done that starts the
# finalize pass, so arriving a few hundred ms early is normal, not a fault.
try:
    SILHOUETTE_WAIT_S = float(_CONFIG.get("silhouette_wait_s", 5.0))
except (TypeError, ValueError):
    SILHOUETTE_WAIT_S = 5.0
try:
    SILHOUETTE_POLL_S = float(_CONFIG.get("silhouette_poll_s", 0.25))
except (TypeError, ValueError):
    SILHOUETTE_POLL_S = 0.25

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
# a confusing 404 instead of the clear "set model in vqa.yaml" error the backend raises.
#
# Anthropic is reached through its OpenAI compatibility layer, which ignores
# `response_format` outright — the backend notices and falls back to asking for the shape
# in the prompt, so it costs a retry on the first call of a run, not correctness.
PROVIDERS = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ),
    # Claude ids from the 4.6 generation on are dateless but still pinned snapshots,
    # so these name one fixed model each, the same as the dated ids elsewhere here.
    "anthropic": (
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-5",
        "claude-haiku-4-5",
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

VLM_PROVIDER = str(_CONFIG.get("provider", "gemini")).strip().lower()
# An unrecognised name is not an error: it is how you reach a provider that is not
# listed, by supplying base_url/model/model_lite yourself in vqa.yaml. The backend
# reports what is missing if you only supply some of them.
_base_url, VLM_API_KEY_ENV, _model, _model_lite = PROVIDERS.get(
    VLM_PROVIDER, ("", "VLM_API_KEY", "", ""))

VLM_BASE_URL = str(_CONFIG.get("base_url", "")).strip() or _base_url
# The only setting still read from the environment: a secret must never live in a file
# checked into git, whatever else moved into vqa.yaml.
VLM_API_KEY = os.environ.get("VLM_API_KEY", "") or os.environ.get(VLM_API_KEY_ENV, "")
MODEL_NAME = str(_CONFIG.get("model", "")).strip() or _model
MODEL_NAME_LITE = (
    str(_CONFIG.get("model_lite", "")).strip() or _model_lite or MODEL_NAME)


def checked_base_url(url: str) -> str:
    """Reject anything that is not plain HTTP(S) to a named host.

    base_url is operator configuration rather than request input, but it still ends
    up as the destination of every outbound call, so a typo or a copy-pasted `file://`
    should fail here with an explanation instead of somewhere inside the SDK. Plaintext
    HTTP is allowed only for loopback, where there is no network to eavesdrop on and
    where a locally hosted OpenAI-compatible server would live.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            f"base_url must be an http(s) URL with a host, got {url!r}")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"base_url must use https for a remote host, got {url!r}")
    return url
