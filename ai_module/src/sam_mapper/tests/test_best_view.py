"""Unit tests for BestViewCollector manifest labels / run_id naming."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam_mapper.best_view import BestViewCollector, BestViewConfig  # noqa: E402
from sam_mapper.detections import PromptTable  # noqa: E402


def _collector(tmp_path, run_id=None):
    table = PromptTable([
        {"prompt": "cabinet", "instance": True},
        {"prompt": "tv", "instance": True},
    ])
    cfg = BestViewConfig.from_dict({
        "top_n": 1,
        "output_dir": str(tmp_path),
        "save_annotated_copy": False,
        "min_instance_score": 0.0,
        "crop_to_roi": True,
        "roi_padding_frac": 0.0,
        "roi_min_size_px": 1,
        "roi_cluster_gap_px": 1000,
    }, table)
    return BestViewCollector(cfg, log=lambda *_: None, run_id=run_id)


def test_manifest_includes_instance_labels(tmp_path):
    collector = _collector(tmp_path, run_id="arabic_room_Q01")
    assert "arabic_room_Q01" in collector.run_dir

    h, w = 200, 800
    image = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[40:120, 300:420] = True
    detections = {
        "labels": np.array(["cabinet"], dtype=object),
        "ids": np.array([7], dtype=int),
        "masks": np.asarray([mask]),
        "confidences": np.array([0.9], dtype=float),
        "bboxes": np.array([[300.0, 40.0, 420.0, 120.0]], dtype=float),
    }
    collector.consider(image, detections, stamp=1.0)

    manifest_path = os.path.join(collector.run_dir, "manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["targets"] == ["cabinet", "tv"]
    instances = manifest["selected"][0]["instances"]
    assert len(instances) == 1
    assert instances[0]["track_id"] == 7
    assert instances[0]["label"] == "cabinet"
    assert "bbox" in instances[0]
    assert "score" in instances[0]
