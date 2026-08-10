"""Local Qwen-VL over the resident `qwen_vqa_server` (`/qwen_vqa/request`).

The development backend: it answers worse than the hosted model but costs nothing per
call, so the inner loop runs here and only scored sweeps spend credits. It reuses the
server the rest of the AI module already keeps warm rather than loading a second
copy of the weights, which would not fit alongside SAM 3 on one GPU.

One constraint of that server's protocol shapes this file: text out, no constrained
decoding. So the JSON shape goes into the prompt and the reply is parsed leniently,
with one stricter retry before giving up.

Several images per call are supported (the server's `images` field). It caps the
list, and exceeding the cap comes back as a rejected request rather than an OOM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Type

from captioner.vlm_backends.base import T, VLMBackend, VLMError, parse_json_object
from captioner.vlm_backends.schemas import json_hint

# The server rejects anything above 512. Verdicts are a sentence plus a boolean, so
# this is generous; it exists to stop a rambling reply from eating the answer budget.
MAX_NEW_TOKENS = 256

RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. Output the JSON object only: "
    "no explanation before it, no explanation after it, no code fence."
)


class QwenRosBackend(VLMBackend):
    name = "local"

    def __init__(self, ask_vqa: Callable[..., str], log=None):
        self._ask_vqa = ask_vqa
        self._log = log or (lambda _msg: None)

    def ask(
        self,
        system: str,
        user: str,
        images: Sequence[Path],
        schema: Type[T],
        *,
        lite: bool = False,  # noqa: ARG002 — one local model, so nothing to downshift to
    ) -> T:
        # Instruction, then output shape, then the actual request LAST. Ordering is not
        # cosmetic here: with the schema hint after the question, a 4B model treats the
        # hint's field descriptions as the most recent context and drifts back toward the
        # prompt's worked examples. Moving the question to the end fixed five of seven
        # decompositions that had been answering with an example's nouns.
        prompt = f"{system.strip()}\n\n{json_hint(schema)}\n\n{user.strip()}"
        reply = self._call(prompt, images)
        try:
            return parse_json_object(reply, schema)
        except VLMError as first_error:
            self._log(f"retrying after unparseable reply: {first_error}")

        reply = self._call(prompt + RETRY_SUFFIX, images)
        return parse_json_object(reply, schema)

    def _call(self, prompt: str, images: Sequence[Path]) -> str:
        try:
            # mode=freeform: the server's "numerical" mode wraps the prompt in an
            # integer-only template, which would strip the JSON we are asking for.
            return self._ask_vqa(
                question=prompt,
                images=list(images),
                max_new_tokens=MAX_NEW_TOKENS,
                mode="freeform",
            )
        except Exception as exc:  # noqa: BLE001 — timeouts, server errors, transport
            raise VLMError(f"{type(exc).__name__}: {exc}") from exc
