"""Loads config/vqa.yaml — the non-secret VLM settings constants.py is built from.

Mirrors sam_mapper's node_base.resolve_config_path/load_config, except resolvable
without ROS: constants.py is imported by plain pytest (no rclpy) as well as by nodes, so
the path can only come from an env var or the installed package share dir, never a ROS
parameter.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def resolve_config_path() -> str:
    """Where vqa.yaml lives: an explicit override, else the package's bundled copy.

    CAPTIONER_VQA_CONFIG is a path override for tests and for pointing a container at a
    config outside the image, not a substitute for editing the bundled file day to day.
    """
    override = os.environ.get("CAPTIONER_VQA_CONFIG", "").strip()
    if override:
        return override
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("captioner")
    except Exception:
        # Not running under a colcon install (e.g. plain pytest) — fall back to the
        # source tree: .../captioner/vlm_backends/config.py -> parents[2] is the
        # captioner package root, which is where config/ sits.
        share = str(Path(__file__).resolve().parents[2])
    return os.path.join(share, "config", "vqa.yaml")


def load_vqa_config() -> dict[str, Any]:
    """The parsed vqa.yaml, or {} if it is missing — every reader here has its own
    documented default, so an absent file degrades to those rather than failing import.
    """
    path = resolve_config_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
