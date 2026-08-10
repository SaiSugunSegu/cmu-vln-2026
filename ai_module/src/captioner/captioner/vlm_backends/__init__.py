"""VLM backends: where inference actually runs.

`make_backend` is the only entry point a node uses, so swapping local for cloud
is an environment variable rather than a code path.

Lives in captioner because that is the package that already owns the local Qwen
server these talk to, and every reasoning package already depends on it — so a
second consumer costs no new edge in the colcon graph.
"""
from __future__ import annotations

from typing import Callable, Optional

from captioner.vlm_backends.base import VLMBackend, VLMError, parse_json_object
from captioner.vlm_backends.constants import VLM_BACKEND

__all__ = ["VLMBackend", "VLMError", "parse_json_object", "make_backend"]


def make_backend(
    backend: Optional[str] = None,
    *,
    ask_vqa: Optional[Callable[..., str]] = None,
    log=None,
) -> VLMBackend:
    """Build the configured backend.

    `ask_vqa` is the node's `/qwen_vqa/request` round-trip, required by the local
    backend and ignored by the cloud one. Imports are deferred so a container
    without the `openai` package can still run the local path.
    """
    name = (backend or VLM_BACKEND).strip().lower()
    if name == "cloud":
        from captioner.vlm_backends.openai_backend import OpenAIBackend
        return OpenAIBackend(log=log)
    if name == "local":
        if ask_vqa is None:
            raise ValueError("the local backend needs an ask_vqa callable")
        from captioner.vlm_backends.qwen_ros_backend import QwenRosBackend
        return QwenRosBackend(ask_vqa=ask_vqa, log=log)
    raise ValueError(f"unknown VLM backend {name!r} (expected 'local' or 'cloud')")
