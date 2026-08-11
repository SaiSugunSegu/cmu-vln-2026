"""Unit tests for BestViewCollector manifest labels / run_id naming / overlay copies."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from sam_mapper.annotate import _CaptionLayout, silhouette_frame
from sam_mapper.best_view import BestViewCollector, BestViewConfig
from sam_mapper.detections import PromptTable


def _collector(tmp_path, run_id=None, save_silhouette_copy=False):
    table = PromptTable([
        {"prompt": "cabinet", "instance": True},
        {"prompt": "tv", "instance": True},
    ])
    cfg = BestViewConfig.from_dict({
        "top_n": 1,
        "output_dir": str(tmp_path),
        "save_annotated_copy": False,
        "save_silhouette_copy": save_silhouette_copy,
        "min_instance_score": 0.0,
        "crop_to_roi": True,
        "roi_padding_frac": 0.0,
        "roi_min_size_px": 1,
        "roi_cluster_gap_px": 1000,
    }, table)
    return BestViewCollector(cfg, log=lambda *_: None, run_id=run_id)


def _consider_one_cabinet(collector):
    """Feed one frame holding a single cabinet detection, away from the seam margins."""
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


def _consider_bigger_cabinet(collector):
    """The same cabinet, more visible — a new best for that track id, so a new flush."""
    h, w = 200, 800
    image = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[30:170, 250:520] = True
    detections = {
        "labels": np.array(["cabinet"], dtype=object),
        "ids": np.array([7], dtype=int),
        "masks": np.asarray([mask]),
        "confidences": np.array([0.95], dtype=float),
        "bboxes": np.array([[250.0, 30.0, 520.0, 170.0]], dtype=float),
    }
    collector.consider(image, detections, stamp=2.0)


def test_flush_keeps_keys_written_by_the_reasoner(tmp_path):
    """Frames keep arriving after the reasoner records its answer in this same file.

    An overwriting flush threw the question, the prompts and the answer away every time,
    which is how a finished run ended up with a manifest that only described geometry.
    """
    collector = _collector(tmp_path, run_id="arabic_room/Q01")
    _consider_one_cabinet(collector)
    manifest_path = os.path.join(collector.run_dir, "manifest.json")

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["question"] = "How many cabinets are in the room?"
    manifest["sam_prompts"] = ["cabinet"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    _consider_bigger_cabinet(collector)

    with open(manifest_path, encoding="utf-8") as handle:
        after = json.load(handle)
    assert after["question"] == "How many cabinets are in the room?"
    assert after["sam_prompts"] == ["cabinet"]
    # And the collector's own keys are still the fresh ones, not the stale copy.
    assert after["selected"][0]["stamp"] == 2.0


def test_reusing_a_run_id_starts_from_an_empty_directory(tmp_path):
    """A stable run id means a rebuild lands on the previous attempt's files."""
    first = _collector(tmp_path, run_id="arabic_room/Q01")
    _consider_one_cabinet(first)
    assert os.listdir(first.run_dir)

    second = _collector(tmp_path, run_id="arabic_room/Q01")
    assert second.run_dir == first.run_dir
    leftover = [name for name in os.listdir(second.run_dir)
                if name.startswith("best_rank") or name == "manifest.json"]
    assert leftover == []


def test_manifest_includes_instance_labels(tmp_path):
    collector = _collector(tmp_path, run_id="arabic_room_Q01")
    assert "arabic_room_Q01" in collector.run_dir

    _consider_one_cabinet(collector)

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


def test_silhouette_copy_written_only_when_enabled(tmp_path):
    collector = _collector(tmp_path / "off")
    _consider_one_cabinet(collector)
    assert not os.path.exists(os.path.join(collector.run_dir, "silhouette"))

    collector = _collector(tmp_path / "on", save_silhouette_copy=True)
    _consider_one_cabinet(collector)

    name = "best_rank1_cabinet+tv.png"
    crop = cv2.imread(os.path.join(collector.run_dir, name))
    silhouette = cv2.imread(os.path.join(collector.run_dir, "silhouette", name))
    assert silhouette is not None
    assert silhouette.shape == crop.shape
    assert not np.array_equal(silhouette, crop)      # something was actually drawn
    # save_annotated_copy stays off: the two copies are independent switches.
    assert not os.path.exists(os.path.join(collector.run_dir, "annotated"))


def test_silhouette_frame_outlines_without_filling():
    # A disc, so the mask outline and the bounding box are distinguishable: a box would
    # colour the bbox corners, which the disc's contour never reaches.
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    disc = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(disc, (50, 50), 20, 1, -1)
    detections = {
        "labels": np.array(["tv"], dtype=object),
        "ids": np.array([3], dtype=int),
        "masks": np.asarray([disc.astype(bool)]),
        "confidences": np.array([0.9], dtype=float),
        "bboxes": np.array([[30.0, 30.0, 70.0, 70.0]], dtype=float),
    }

    out = silhouette_frame(image, detections)

    assert np.array_equal(out[45:55, 45:55], image[45:55, 45:55])   # interior not filled
    assert out[50, 30:32].any()                                    # left edge outlined
    assert not out[68, 30:32].any()                                # no bbox border drawn


def _tab_bbox(image, color):
    """(y0, y1) rows covered by a caption tab of this colour, or None."""
    rows = np.flatnonzero(np.all(image == np.array(color, dtype=np.uint8), axis=2).any(axis=1))
    return None if rows.size == 0 else (int(rows[0]), int(rows[-1]))


def test_captions_never_overlap_each_other():
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    red, green, blue = (0, 0, 255), (0, 255, 0), (255, 0, 0)

    layout = _CaptionLayout()
    for color in (red, green, blue):
        layout.add(40, 60, "cabinet", color)     # three objects, one anchor point
    layout.draw(canvas)

    spans = sorted(filter(None, (_tab_bbox(canvas, c) for c in (red, green, blue))))
    assert len(spans) == 3                        # none was painted over by a later tab
    for (_, lower_end), (upper_start, _) in zip(spans, spans[1:]):
        assert lower_end < upper_start            # and they occupy separate rows
