"""Unit tests for BestViewCollector manifest labels / run_id naming / overlay copies."""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from sam_mapper.annotate import (
    MAX_SLIDE_PX,
    _CaptionLayout,
    _mask_anchor,
    silhouette_frame,
)
from sam_mapper.best_view import BestViewCollector, BestViewConfig
from sam_mapper.challenge_marker import track_to_map_id
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


def _capture(drawn, ids=None):
    """A stand-in overlay renderer recording the captions, and optionally the ids, it got.

    finalize() is about which text and which ids land on the image; reading either back out
    of the pixels would test OpenCV instead. The ids are what `_color_for` keys the outline
    colour on, so capturing them is how "one object, one colour" gets asserted.
    """
    def render(crop, detections):
        drawn.append([str(label) for label in detections["labels"]])
        if ids is not None:
            ids.append([int(i) for i in detections["ids"]])
        return crop
    return render


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


def _consider_two_cabinets(collector, tids):
    """One frame, two cabinets in one cluster. Both clear of the seam margins (200 px)."""
    h, w = 200, 800
    image = np.zeros((h, w, 3), dtype=np.uint8)
    boxes = [(300, 420), (440, 560)]
    masks = []
    for x0, x1 in boxes:
        mask = np.zeros((h, w), dtype=bool)
        mask[40:120, x0:x1] = True
        masks.append(mask)
    collector.consider(image, {
        "labels": np.array(["cabinet"] * 2, dtype=object),
        "ids": np.array(tids, dtype=int),
        "masks": np.asarray(masks),
        "confidences": np.array([0.9, 0.9], dtype=float),
        "bboxes": np.array([[x0, 40.0, x1, 120.0] for x0, x1 in boxes], dtype=float),
    }, stamp=1.0)


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
    collector.finalize(None)
    assert not os.path.exists(os.path.join(collector.run_dir, "silhouette"))

    collector = _collector(tmp_path / "on", save_silhouette_copy=True)
    _consider_one_cabinet(collector)

    name = "best_rank1_cabinet+tv.png"
    silhouette_path = os.path.join(collector.run_dir, "silhouette", name)
    # Not per flush: the labels carry map_node's ids, which are still moving mid-run.
    assert not os.path.exists(silhouette_path)

    collector.finalize(None)

    crop = cv2.imread(os.path.join(collector.run_dir, name))
    silhouette = cv2.imread(silhouette_path)
    assert silhouette is not None
    assert silhouette.shape == crop.shape
    assert not np.array_equal(silhouette, crop)      # something was actually drawn
    # save_annotated_copy stays off: the two copies are independent switches.
    assert not os.path.exists(os.path.join(collector.run_dir, "annotated"))


def test_track_to_map_id_finds_ids_a_world_merge_absorbed():
    # The 12% case: map_node keys an entry by id[0], so track 12 is in the map but is not
    # a key. Looking it up as one is how a merged object loses its crop evidence.
    objects = {
        "3": {"label": "chair", "id": [3, 12]},
        "5": {"label": "sofa", "id": [5]},
        "8": {"label": "tv"},                     # no id list at all — key stands for itself
    }

    assert track_to_map_id(objects) == {3: 3, 12: 3, 5: 5, 8: 8}


def test_finalize_labels_each_instance_with_its_obj_map_key(tmp_path):
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_one_cabinet(collector)                    # one instance, track id 7

    drawn = []
    collector._overlays = {"silhouette": _capture(drawn)}

    # Track 7 was absorbed by the object keyed 4 — the caption must say 4, not 7.
    collector.finalize({7: 4})
    assert drawn[-1] == ["cabinet [4]"]


def test_finalize_leaves_an_unmapped_instance_without_a_bracket(tmp_path):
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_one_cabinet(collector)

    drawn = []
    collector._overlays = {"silhouette": _capture(drawn)}

    # The map exists but track 7 never reached a box. An id here would resolve to nothing.
    collector.finalize({})
    assert drawn[-1] == ["cabinet"]


def test_finalize_falls_back_to_track_ids_without_a_map(tmp_path):
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_one_cabinet(collector)

    drawn = []
    collector._overlays = {"silhouette": _capture(drawn)}

    collector.finalize(None)                            # map_node not in the launch at all
    assert drawn[-1] == ["cabinet [7]"]


def test_finalize_colours_by_map_id_so_one_object_is_one_colour(tmp_path):
    # The observed failure: map object 1 held track ids [1, 3], so it outlined purple in two
    # of a run's crops and green in the third while all three captions read `sofa [1]`.
    # `_color_for` keys on ids, so the ids reaching the renderer must be the map's.
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_two_cabinets(collector, tids=[1, 3])

    drawn, ids = [], []
    collector._overlays = {"silhouette": _capture(drawn, ids)}

    collector.finalize({1: 1, 3: 1})
    assert ids[-1] == [1, 1]


def test_finalize_captions_a_duplicated_map_id_once(tmp_path):
    # Both instances still get an outline — they are separate mask regions — but two tabs
    # reading `cabinet [1]` would be noise, so only the first is captioned.
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_two_cabinets(collector, tids=[1, 3])

    drawn = []
    collector._overlays = {"silhouette": _capture(drawn)}

    collector.finalize({1: 1, 3: 1})
    assert drawn[-1] == ["cabinet [1]", ""]


def test_finalize_keeps_the_track_id_of_an_unmapped_instance(tmp_path):
    # No map entry, so no id to inherit. The track id cannot collide with a map key: every
    # key is its own entry's id[0], hence a track id that DOES resolve.
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_two_cabinets(collector, tids=[1, 3])

    drawn, ids = [], []
    collector._overlays = {"silhouette": _capture(drawn, ids)}

    collector.finalize({1: 4})                          # track 3 reached no 3D box
    assert ids[-1] == [4, 3]
    assert drawn[-1] == ["cabinet [4]", "cabinet"]


def test_finalize_re_renders_only_when_the_lookup_changed(tmp_path):
    collector = _collector(tmp_path, save_silhouette_copy=True)
    _consider_one_cabinet(collector)

    drawn = []
    collector._overlays = {"silhouette": _capture(drawn)}

    # Bag loop, explore_done and shutdown all call it; only better data should cost a render.
    assert collector.finalize({7: 4}) is True
    assert collector.finalize({7: 4}) is False
    assert collector.finalize({7: 9}) is True
    assert [d[0] for d in drawn] == ["cabinet [4]", "cabinet [9]"]

    # A new selection invalidates what is on disk, so the same lookup must render again.
    _consider_bigger_cabinet(collector)
    assert collector.finalize({7: 9}) is True


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
    """(y0, y1) rows covered by a caption TAB of this colour, or None.

    Under 12 px of the colour in a row does not count: a leader line is 1 px thick and its
    anchor dot 7 across, and either would make a tab look as tall as the gap it spans. The
    narrowest real tab ("tv") is 19 px.
    """
    hits = np.all(image == np.array(color, dtype=np.uint8), axis=2)
    rows = np.flatnonzero(hits.sum(axis=1) >= 12)
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


def test_placement_never_slides_further_than_the_leash():
    # Solid tabs from y=100 to y=300, then clear space. The old search walked straight down
    # to y=300 -- 116 px from the anchor, reading as a label for whatever it landed on.
    tab_w, tab_h = 60, 16
    blocked = [(0, y, 400, y + tab_h) for y in range(100, 300, tab_h)]

    x0, top = _CaptionLayout._place(200, 200, tab_w, tab_h, 400, 400, blocked)

    x_pref, y_pref = 200 - tab_w // 2, 200 - tab_h
    assert math.hypot(x0 - x_pref, top - y_pref) <= MAX_SLIDE_PX


def test_caption_anchors_on_the_largest_mask_blob_not_the_bbox_corner():
    # The real failure: a stray sliver of the same mask far from the object stretches the
    # bbox, and anchoring on its corner puts the caption over empty pixels.
    mask = np.zeros((120, 400), dtype=bool)
    mask[40:100, 300:360] = True         # the object
    mask[60:62, 10:12] = True            # bleed, 290 px away
    bbox = (10.0, 40.0, 360.0, 100.0)

    x, y = _mask_anchor(mask, bbox)

    assert 300 <= x <= 360               # over the blob, not at the bbox's left edge
    assert y == 40


def test_leader_is_drawn_only_for_a_tab_that_left_its_anchor():
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    color = np.array((0, 255, 0), dtype=np.uint8)

    # A tab resting on its anchor owes nothing — the association is already obvious.
    _CaptionLayout._leader(canvas, (60, 84, 130, 100), (100, 100), tuple(int(c) for c in color))
    assert not canvas.any()

    # One pushed clear of it gets a dot at the anchor and a line back to the tab.
    _CaptionLayout._leader(canvas, (60, 20, 130, 36), (100, 100), tuple(int(c) for c in color))
    assert np.all(canvas[100, 100] == color)     # the dot
    assert np.all(canvas[70, 100] == color)      # the line, midway back to the tab
