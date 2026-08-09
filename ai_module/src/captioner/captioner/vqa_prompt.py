"""VQA chat-content helpers shared by the HF backend (stdlib + typing only)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

VQA_PROMPT_SINGULAR = (
    "Answer the question about this image with a single integer only. "
    "Do not include units, words, or explanation."
    "{instruction_block}"
    "\nQuestion: {question}"
)
VQA_PROMPT_MULTI = (
    "Answer the question about these images with a single integer only. "
    "Use all images as context. "
    "Do not include units, words, or explanation."
    "{instruction_block}"
    "\nQuestion: {question}"
)
FREEFORM_VQA_PROMPT = "{question}"


def _instruction_block(instruction: Optional[str]) -> str:
    text = (instruction or "").strip()
    if not text:
        return ""
    return f"\nAdditional instructions: {text}"


def numerical_vqa_text(
        question: str,
        n_images: int,
        *,
        freeform: bool = False,
        instruction: Optional[str] = None,
        ) -> str:
    """Format the text block for a VQA turn."""
    if freeform:
        extra = (instruction or "").strip()
        if extra:
            return f"{extra}\n\n{question}"
        return question
    if n_images <= 1:
        template = VQA_PROMPT_SINGULAR
    else:
        template = VQA_PROMPT_MULTI
    return template.format(
        question=question,
        instruction_block=_instruction_block(instruction),
    )


def multi_image_user_content(
        images: Sequence[Any],
        question: str,
        *,
        freeform: bool = False,
        instruction: Optional[str] = None,
        ) -> list[dict]:
    """Build Qwen-VL user ``content``: N image blocks then one text block."""
    if not images:
        raise ValueError("multi_image_user_content requires at least one image")
    text = numerical_vqa_text(
        question, len(images), freeform=freeform, instruction=instruction)
    content: list[dict] = [
        {"type": "image", "image": image}
        for image in images
    ]
    content.append({"type": "text", "text": text})
    return content
