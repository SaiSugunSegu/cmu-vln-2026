"""Standalone Qwen-VL visual question answering CLI.

Example:
  python -m captioner.models.vqa /path/to/image.png \\
    --question "How many pillows are on the bed?" \\
    --quantization int4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import imageio.v3 as iio

from captioner.models.captioning import load_qwen_backend


def _secure_image_path(user_path: str) -> Path:
    """Resolve a user-supplied image path and reject traversal / missing files."""
    decoded = unquote(user_path)
    path = Path(decoded).expanduser().resolve()
    if ".." in Path(decoded).parts:
        raise ValueError(f"Path traversal rejected: {user_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Ask a Qwen-VL model a question about an image (int4 by default).")
    parser.add_argument("image", help="Path to an RGB image (png/jpg/…).")
    parser.add_argument(
        "--question", "-q", required=True,
        help='Natural-language question, e.g. "How many pillows are on the bed?"')
    parser.add_argument(
        "--captioning_model", default="qwen3vl",
        choices=["qwen3vl", "qwen2_5vl"])
    parser.add_argument(
        "--quantization", default="int4", choices=["int4", "int8", "none"],
        help='Weight quantization. "none" loads bf16.')
    parser.add_argument("--model_id", default=None,
                        help="Override the backend default checkpoint.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Decode budget. Keep small for numeric answers.")
    parser.add_argument(
        "--max_pixels", type=int, default=1280 * 28 * 28,
        help="Vision token budget (higher helps counting on full scenes).")
    parser.add_argument(
        "--raw", action="store_true",
        help="Print the raw model text instead of the parsed integer.")
    parser.add_argument(
        "--output_json", default=None,
        help="Optional path to write {question, answer, raw, …} as JSON.")
    args = parser.parse_args(argv)

    image_path = _secure_image_path(args.image)
    image = iio.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    model = load_qwen_backend(
        args.captioning_model,
        quantization=args.quantization,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )
    print(
        f"Loaded {model.model_id} (quantization={model.quantization}) "
        f"in {model.load_seconds:.1f}s")

    raw_answers = model.answer_questions([image], [args.question])
    raw = raw_answers[0]
    number = model.extract_integer(raw)

    if args.raw or number is None:
        print(raw)
    else:
        print(number)

    if args.output_json is not None:
        out_path = Path(unquote(args.output_json)).expanduser().resolve()
        if ".." in Path(args.output_json).parts:
            raise ValueError(f"Path traversal rejected: {args.output_json}")
        payload = {
            "image": str(image_path),
            "question": args.question,
            "raw_answer": raw,
            "integer_answer": number,
            "model_id": model.model_id,
            "quantization": model.quantization,
            "backend": args.captioning_model,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")

    if number is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
