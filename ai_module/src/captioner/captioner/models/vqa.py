"""Standalone Qwen-VL visual question answering CLI.

Loads the model on every invocation (~60 s). For repeated questions prefer the
persistent server: `just vqa-up` then `just vqa-ask "…" /data/img.png`.

Example:
  python -m captioner.models.vqa /data/crops/3_bed/crop.png \\
    --question "How many pillows are on the bed?" \\
    --quantization int4
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

import imageio.v3 as iio

from captioner.models.captioning import load_qwen_backend
from captioner.paths import secure_image_path, secure_output_path


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

    # Validate every path before the ~60 s model load, so a typo or an
    # out-of-mount path fails now rather than after inference has already run.
    image_path = secure_image_path(args.image)
    out_path = secure_output_path(args.output_json) if args.output_json else None

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

    raw = model.answer_questions([image], [args.question])[0]
    number = model.extract_integer(raw)

    if args.raw or number is None:
        print(raw)
    else:
        print(number)

    if out_path is not None:
        payload = {
            "image": str(image_path),
            "question": args.question,
            "raw_answer": raw,
            "integer_answer": number,
            "model_id": model.model_id,
            "quantization": model.quantization,
            "backend": args.captioning_model,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {out_path}")

    # Exit 2 means "ran fine, but produced no integer", so a caller scripting a
    # numeric question can tell that apart from a crash (exit 1).
    if number is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
