"""Unit tests for marked_views: which instances get marked, and on which base image."""
from __future__ import annotations

import json
import threading

import cv2
import numpy as np
import pytest

from smart_vlm.cat2_utils import marked_views

NAME = "best_rank1_chair.png"


def _run_dir(tmp_path, obj_map=None, silhouette=False):
    """A crop directory as sam_node leaves it, with or without the finalize pass having run."""
    crop = np.full((80, 120, 3), 30, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / NAME), crop)
    if silhouette:
        (tmp_path / "silhouette").mkdir()
        # Visibly different from the crop, so a test can tell which one was drawn on.
        cv2.imwrite(str(tmp_path / "silhouette" / NAME), np.full_like(crop, 200))
    if obj_map is not None:
        (tmp_path / "obj_map.json").write_text(json.dumps(obj_map), encoding="utf-8")
    return tmp_path


def _manifest(*track_ids):
    return {"selected": [{
        "file": NAME,
        "instances": [{"track_id": t, "label": "chair", "bbox": [10.0, 10.0, 50.0, 50.0]}
                      for t in track_ids],
    }]}


def test_marks_an_instance_whose_track_id_a_world_merge_absorbed(tmp_path):
    # The 12% case. Candidate "3" is an obj_map key; the crop shows the object under 12,
    # which map_node folded into 3. Matching against the key directly left it unmarked.
    run_dir = _run_dir(tmp_path, obj_map={"3": {"label": "chair", "id": [3, 12]}})

    assert marked_views(run_dir, _manifest(12), ["3"], 1) == [run_dir / "marked" / NAME]


def test_skips_a_view_holding_no_candidate(tmp_path):
    run_dir = _run_dir(tmp_path, obj_map={"3": {"label": "chair", "id": [3]}})

    assert marked_views(run_dir, _manifest(99), ["3"], 1) == []


def test_marks_on_the_finalized_silhouette_without_repeating_its_labels(tmp_path):
    # The silhouette already prints `chair [3]`; a tab would be a second name for it.
    run_dir = _run_dir(tmp_path, obj_map={"3": {"label": "chair", "id": [3]}}, silhouette=True)

    marked_views(run_dir, _manifest(3), ["3"], 1)

    out = cv2.imread(str(run_dir / "marked" / NAME))
    assert out[0, 0].tolist() == [200, 200, 200]     # the silhouette (200), not the crop (30)
    interior = out[20:40, 20:40]                     # inside the box, clear of its outline
    assert np.all(interior == 200), "something was drawn inside the box — a tab, most likely"


def test_falls_back_to_the_bare_crop_and_labels_it_itself(tmp_path, monkeypatch):
    # finalize never ran, so the pixels carry no ids and the mark has to supply one.
    monkeypatch.setattr("smart_vlm.cat2_utils.SILHOUETTE_WAIT_S", 0.0)   # no point waiting
    run_dir = _run_dir(tmp_path, obj_map={"3": {"label": "chair", "id": [3, 12]}})

    marked_views(run_dir, _manifest(12), ["3"], 1)

    out = cv2.imread(str(run_dir / "marked" / NAME))
    assert out[0, 0].tolist() == [30, 30, 30]        # the raw crop
    assert not np.all(out == 30)                     # a tab and a box were drawn


def test_waits_for_a_silhouette_that_arrives_late(tmp_path, monkeypatch):
    """finalize() and this reasoner both fire on explore_done, so we can arrive first.

    Giving up on the first miss would cost the outlines and the ids over a few ms of
    scheduling.
    """
    run_dir = _run_dir(tmp_path, obj_map={"3": {"label": "chair", "id": [3]}})
    monkeypatch.setattr("smart_vlm.cat2_utils.SILHOUETTE_POLL_S", 0.01)

    # 50 ms against a 5 s deadline: two orders of magnitude of slack before this could flake.
    finalize = threading.Timer(0.05, lambda: _run_dir(tmp_path, silhouette=True))
    finalize.start()
    try:
        marked_views(run_dir, _manifest(3), ["3"], 1)
    finally:
        finalize.cancel()

    out = cv2.imread(str(run_dir / "marked" / NAME))
    assert out[0, 0].tolist() == [200, 200, 200]     # it waited, and got the silhouette


@pytest.mark.parametrize("obj_map", [None, {}])
def test_without_a_map_a_crop_track_id_is_read_as_a_map_key(tmp_path, obj_map, monkeypatch):
    # Every unmerged object has key == track id, so this holds with nothing to resolve
    # against -- which is what the offline bench relies on.
    monkeypatch.setattr("smart_vlm.cat2_utils.SILHOUETTE_WAIT_S", 0.0)
    run_dir = _run_dir(tmp_path, obj_map=obj_map)

    assert marked_views(run_dir, _manifest(7), ["7"], 1) == [run_dir / "marked" / NAME]
