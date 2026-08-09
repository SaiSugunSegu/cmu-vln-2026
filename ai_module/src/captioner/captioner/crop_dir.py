"""Helpers for offline eval-crops folders (question from name, crop PNGs).

Stdlib only so unit tests run without torch / ROS.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_RUN_PREFIX_RE = re.compile(r"^num_reasoner_[0-9a-fA-F]{8}_(.+)$")
_RANK1_RE = re.compile(r"^best_rank1_(.+)\.png$", re.IGNORECASE)


def infer_target_tag(crop_dir: Path) -> Optional[str]:
    """Target tag from root/annotated best_rank1_*.png, else manifest targets."""
    crop_dir = Path(crop_dir)
    for directory in (crop_dir, crop_dir / "annotated"):
        if not directory.is_dir():
            continue
        matches = sorted(directory.glob("best_rank1_*.png"))
        if matches:
            match = _RANK1_RE.match(matches[0].name)
            if match:
                return match.group(1)

    manifest_path = crop_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        targets = manifest.get("targets") or []
        if isinstance(targets, list) and targets:
            return "+".join(str(t) for t in targets)
    return None


def question_from_folder_name(
        folder_name: str,
        target_tag: Optional[str] = None,
        ) -> str:
    """Recover the natural-language question from an eval-crops folder name.

    Pattern: ``num_reasoner_{8hex}_{sanitized_question}_{target_tag}``
    """
    name = Path(folder_name).name
    match = _RUN_PREFIX_RE.match(name)
    rest = match.group(1) if match else name

    if target_tag:
        suffix = f"_{target_tag}"
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
        elif rest == target_tag:
            rest = ""

    question = rest.replace("_", " ").strip()
    if not question:
        raise ValueError(f"Could not extract question from folder name: {folder_name}")
    return question


def question_from_crop_dir(crop_dir: Path | str) -> str:
    """Extract the question from a crop run directory's folder name."""
    crop_dir = Path(crop_dir)
    return question_from_folder_name(crop_dir.name, infer_target_tag(crop_dir))


def list_crop_images(
        crop_dir: Path | str,
        *,
        annotated: bool = False,
        ) -> list[Path]:
    """Sorted crop PNGs from a run directory.

    Default: root ``best_rank*.png`` (clean ROI crops).
    ``annotated=True``: ``annotated/*.png`` overlays (masks/boxes/labels).
    """
    crop_dir = Path(crop_dir)
    if annotated:
        annotated_dir = crop_dir / "annotated"
        if annotated_dir.is_dir():
            images = sorted(annotated_dir.glob("*.png"))
            if images:
                return images
        raise FileNotFoundError(
            f"No annotated crop images found under {crop_dir}/annotated")

    images = sorted(crop_dir.glob("best_rank*.png"))
    if images:
        return images
    raise FileNotFoundError(
        f"No crop images found under {crop_dir} (expected best_rank*.png)")
