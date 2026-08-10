"""Any hosted model behind an OpenAI-compatible endpoint. Selected with VLM_BACKEND=cloud.

The scored path: a submission may use an externally hosted model, and one counts far
better than the 4B local model does. Which model is a deployment question, not a code
one, so this talks to whatever VLM_PROVIDER / VLM_BASE_URL point at — Gemini, DashScope
and OpenRouter are all just presets in constants.py.

`beta.chat.completions.parse` with a Pydantic `response_format` gets constrained
decoding where the endpoint supports it, so the schema is enforced server-side rather
than begged for in the prompt the way the local backend has to.

Where it does not — Anthropic's compatibility layer ignores `response_format`, and the
aggregators honour it per underlying model — the SDK still tries to validate whatever
text came back and raises, so `ask` catches that and retries once the way the local
backend works: schema in the prompt, lenient parse of the reply.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Sequence, Type

from pydantic import ValidationError

from captioner.vlm_backends.base import T, VLMBackend, VLMError, parse_json_object
from captioner.vlm_backends.constants import (
    MODEL_NAME,
    MODEL_NAME_LITE,
    VLM_API_KEY,
    VLM_API_KEY_ENV,
    VLM_BASE_URL,
    VLM_PROVIDER,
    checked_base_url,
)
from captioner.vlm_backends.schemas import json_hint


def _data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


class OpenAIBackend(VLMBackend):
    def __init__(self, log=None):
        # deferred: local-only containers need not ship it
        from openai import BadRequestError, OpenAI

        # Say which of the three is missing, since an unlisted provider needs all of
        # them and a listed one usually needs only the key.
        if not VLM_BASE_URL:
            raise VLMError(
                f"VLM_PROVIDER={VLM_PROVIDER!r} has no preset, so set VLM_BASE_URL "
                "to its OpenAI-compatible endpoint in the repo-root .env")
        if not MODEL_NAME:
            raise VLMError(
                f"VLM_PROVIDER={VLM_PROVIDER!r} has no default model, so set VLM_MODEL "
                "in the repo-root .env")
        if not VLM_API_KEY:
            raise VLMError(
                f"VLM_BACKEND=cloud but neither {VLM_API_KEY_ENV} nor VLM_API_KEY is in "
                "the environment (put one in the repo-root .env)")
        # The manifest should record which hosted model produced a count, not just that
        # some cloud did, or two sweeps become indistinguishable after the fact.
        self.name = f"cloud:{VLM_PROVIDER}"
        self._log = log or (lambda _msg: None)
        self._client = OpenAI(api_key=VLM_API_KEY, base_url=checked_base_url(VLM_BASE_URL))
        # An endpoint that refuses the json_schema response_format outright answers 400,
        # which is a fallback signal rather than a failure. Held as an attribute because
        # the import is deferred with the client's.
        self._bad_request = BadRequestError
        self._log(
            f"cloud backend ready: {VLM_PROVIDER} {MODEL_NAME} / {MODEL_NAME_LITE}")

    def ask(
        self,
        system: str,
        user: str,
        images: Sequence[Path],
        schema: Type[T],
        *,
        lite: bool = False,
    ) -> T:
        content: list[dict] = [{"type": "text", "text": user}]
        for path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _data_url(Path(path))},
            })

        model = MODEL_NAME_LITE if lite else MODEL_NAME
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        try:
            completion = self._client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=schema)
            parsed = getattr(completion.choices[0].message, "parsed", None)
            if parsed is not None:
                return parsed
            # `parse` validates the reply text itself whenever there is any, so reaching
            # here means the endpoint sent none — a refusal, or a filtered response.
            reason = "the reply had no content"
        except (ValidationError, json.JSONDecodeError, self._bad_request) as exc:
            # The endpoint either rejected the schema or ignored it and answered in
            # prose. Both are recoverable, and only these are: an auth failure or a
            # rate limit must surface now rather than after a second billed call.
            reason = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — the SDK raises a wide family
            raise VLMError(f"{type(exc).__name__}: {exc}") from exc

        self._log(f"{VLM_PROVIDER} gave no usable structured output ({reason}); "
                  "retrying with the schema in the prompt")
        return self._ask_unconstrained(model, system, content, schema)

    def _ask_unconstrained(
        self,
        model: str,
        system: str,
        content: list[dict],
        schema: Type[T],
    ) -> T:
        """Ask again with the shape described in the prompt, and parse leniently.

        Exactly what the local backend does, for the same reason: an endpoint without
        constrained decoding still answers correctly, it just wraps the JSON in a fence
        or a sentence. The hint goes on the system message so the question stays last.
        """
        messages = [
            {"role": "system", "content": f"{system.strip()}\n\n{json_hint(schema)}"},
            {"role": "user", "content": content},
        ]
        try:
            completion = self._client.chat.completions.create(
                model=model, messages=messages)
        except Exception as exc:  # noqa: BLE001 — the SDK raises a wide family
            raise VLMError(f"{type(exc).__name__}: {exc}") from exc
        return parse_json_object(completion.choices[0].message.content or "", schema)
