"""Standalone Qwen-VL visual question answering CLI.

Loads the model on every invocation (~60 s). For repeated questions prefer the
persistent server: `just vqa-up` then `just vqa-ask "…" /data/img.png`.

Examples:
  python -m captioner.models.vqa /data/crops/3_bed/crop.png \\
    --question "How many pillows are on the bed?" \\
    --quantization int4

  # Multi-image from an eval-crops folder (clean crops; question from folder name):
  python -m captioner.models.vqa --crop-dir \\
    /data/eval_bench/eval-crops/num_reasoner_871be12b_How_many_carpets_are_on_the_floor_carpet

  # Same folder, but use annotated overlays:
  python -m captioner.models.vqa --crop-dir .../num_reasoner_… --annotated

  # Ad-hoc multi-image:
  python -m captioner.models.vqa img1.png img2.png -q "How many carpets?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import imageio.v3 as iio

from captioner.crop_dir import list_crop_images, question_from_crop_dir
from captioner.models.captioning import load_qwen_backend
from captioner.paths import secure_image_path, secure_output_path, secure_path


def _roots_for_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Allowed roots for offline host paths (crop dir + parents of image files)."""
    roots: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        roots.append(resolved if resolved.is_dir() else resolved.parent)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return tuple(unique)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Ask a Qwen-VL model a question about one or more images "
                    "(int4 by default).")
    parser.add_argument(
        "images", nargs="*",
        help="Path(s) to RGB image(s). Omit when using --crop-dir.")
    parser.add_argument(
        "--crop-dir", default=None,
        help="Eval-crops run directory: load best_rank*.png (or annotated/ "
             "with --annotated) and derive the question from the folder name.")
    parser.add_argument(
        "--annotated", action="store_true",
        help="With --crop-dir, use annotated/*.png overlays instead of clean crops.")
    parser.add_argument(
        "--instruction", "-I", default=None,
        help="Extra textual guidance added to the prompt "
             "(e.g. counting rules across views).")
    parser.add_argument(
        "--freeform", action="store_true",
        help="Do not wrap with the integer-only numerical prompt.")
    parser.add_argument(
        "--question", "-q", default=None,
        help='Natural-language question (required unless --crop-dir).')
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

    if args.crop_dir and args.images:
        raise SystemExit("Pass either --crop-dir or image path(s), not both.")
    if not args.crop_dir and not args.images:
        raise SystemExit("Provide image path(s) or --crop-dir.")

    image_paths: list[Path]
    question: str
    roots: tuple[Path, ...]

    if args.crop_dir:
        crop_root = Path(args.crop_dir).expanduser().resolve()
        if not crop_root.is_dir():
            raise SystemExit(f"crop-dir is not a directory: {crop_root}")
        roots = (crop_root,)
        crop_dir = secure_path(str(crop_root), roots=roots)
        image_paths = list_crop_images(crop_dir, annotated=args.annotated)
        question = args.question or question_from_crop_dir(crop_dir)
    else:
        if not args.question or not str(args.question).strip():
            raise SystemExit("--question is required when not using --crop-dir.")
        question = args.question
        raw_paths = [Path(p).expanduser().resolve() for p in args.images]
        roots = _roots_for_paths(raw_paths)
        image_paths = [secure_image_path(str(p), roots=roots) for p in raw_paths]

    out_path = None
    if args.output_json:
        out_candidate = Path(args.output_json).expanduser().resolve()
        out_roots = roots + (out_candidate.parent,)
        out_path = secure_output_path(str(out_candidate), roots=out_roots)

    # Validate every path before the ~60 s model load, so a typo or an
    # out-of-mount path fails now rather than after inference has already run.
    validated = [secure_image_path(str(p), roots=roots) for p in image_paths]
    arrays = []
    for path in validated:
        image = iio.imread(str(path))
        if image is None:
            raise SystemExit(f"Failed to read image: {path}")
        arrays.append(image)

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
    print(f"Question: {question}")
    print(f"Images ({len(validated)}): " + ", ".join(p.name for p in validated))

    if len(arrays) == 1:
        raw = model.answer_questions(
            arrays, [question],
            freeform=args.freeform,
            instruction=args.instruction,
        )[0]
    else:
        raw = model.answer_multi_image(
            arrays, question,
            freeform=args.freeform,
            instruction=args.instruction,
        )
    number = model.extract_integer(raw)

    if args.raw or args.freeform or number is None:
        print(raw)
    else:
        print(number)

    if out_path is not None:
        payload = {
            "images": [str(p) for p in validated],
            "question": question,
            "instruction": args.instruction,
            "freeform": bool(args.freeform),
            "raw_answer": raw,
            "integer_answer": number,
            "model_id": model.model_id,
            "quantization": model.quantization,
            "backend": args.captioning_model,
        }
        if args.crop_dir:
            payload["crop_dir"] = str(Path(args.crop_dir).expanduser().resolve())
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {out_path}")

    # Exit 2 means "ran fine, but produced no integer", so a caller scripting a
    # numeric question can tell that apart from a crash (exit 1).
    if number is None and not args.freeform:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
