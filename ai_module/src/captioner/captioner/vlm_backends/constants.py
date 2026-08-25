"""VLM configuration, read from config/vqa.yaml plus API keys from the environment.

vqa.yaml is the only switch for WHERE inference runs. Launch files, reasoners, the
supervisor, and the eval harness all import these names — there is no ROS override.

  vlm_backend              cloud | local   answering / counting
  target_extract_backend   cloud | local   noun extraction before SAM is armed
  provider                 gemini | anthropic | dashscope | openrouter | openai | <custom>
  base_url                 OpenAI-compatible URL (blank = the provider preset)
  model / model_lite
  view_source_numerical              crop | silhouette | full | full_silhouette
  view_source_object_reference       crop | silhouette | full | full_silhouette
  view_source_instruction_following  crop | silhouette | full | full_silhouette
  silhouette_wait_s / silhouette_poll_s

`cloud` is a hosted OpenAI-compatible endpoint. `local` is the in-image Qwen server.
Both backend keys are independent and must be set; a missing or unknown value
falls back to `cloud`.

API keys stay in the environment (.env / compose env_file / baked image ENV), not
in vqa.yaml. See captioner/vlm_backends/config.py for how that file is loaded.
"""
import os
from urllib.parse import urlparse

from captioner.vlm_backends.config import load_vqa_config

_CONFIG = load_vqa_config()


def _backend(value, fallback: str) -> str:
    name = str(value or "").strip().lower()
    return name if name in ("local", "cloud") else fallback


VLM_BACKEND = _backend(_CONFIG.get("vlm_backend"), "cloud")
TARGET_EXTRACT_BACKEND = _backend(_CONFIG.get("target_extract_backend"), "cloud")
# Qwen must be running if either call is local.
NEED_LOCAL_VQA = VLM_BACKEND == "local" or TARGET_EXTRACT_BACKEND == "local"


def local_vqa_launch_flag() -> str:
    """'true' / 'false' for smart_vlm.launch's `if=` — XML needs a string."""
    return "true" if NEED_LOCAL_VQA else "false"

# Which image of a best-view crop the model sees: `silhouette` is the mask-outline +
# label copy sam_node writes next to each crop (see sam_mapper's `save_silhouette_copy`);
# `crop` is the plain, unannotated photo. Each of the three scored categories has its own
# key -- `view_source_numerical` (`numerical_utils.select_context_views`),
# `view_source_object_reference` (`cat2_utils.marked_views`) and
# `view_source_instruction_following` (`instruction_reasoner`) -- read by both the live
# reasoners and their offline benches, and validated independently of one another below.
#
# The `full*` values send the WHOLE 360 panorama instead of the ROI crop cut out of it, which
# is what category 3 wants: a route is planned over a room, and a crop shows a corner of one.
# They require sam_mapper's `save_full_views: true`; with it off nothing is written under
# `full/`, and a caller must fall back to the cropped equivalent rather than send nothing.
#: view_source -> the run-directory subpath its images live in.
VIEW_DIRS = {
    "crop": "",
    "silhouette": "silhouette",
    "full": "full",
    "full_silhouette": "full/silhouette",
}


def _view_source(value) -> str:
    name = str(value or "").strip().lower()
    return name if name in VIEW_DIRS else "silhouette"


VIEW_SOURCE_NUMERICAL = _view_source(_CONFIG.get("view_source_numerical"))
VIEW_SOURCE_OBJECT_REFERENCE = _view_source(_CONFIG.get("view_source_object_reference"))
VIEW_SOURCE_INSTRUCTION_FOLLOWING = _view_source(_CONFIG.get("view_source_instruction_following"))


def view_dir(source: str) -> str:
    """The subdirectory of a run dir that `source` names. '' is the run dir itself."""
    return VIEW_DIRS.get(source, VIEW_DIRS["silhouette"])


def is_silhouette(source: str) -> bool:
    """True when the images are written by sam_node's finalize pass, so a caller must wait."""
    return "silhouette" in source

# How long a silhouette run waits for sam_node to finish writing the
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
