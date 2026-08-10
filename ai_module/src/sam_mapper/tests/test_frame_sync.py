"""Unit tests for the frame-assembly math shared by map_node and the offline harness.

frame_sync was extracted in T0.1 precisely so the benchmark exercises the same code the
node does. That makes it the highest-leverage place to have tests: a regression here is
silently wrong in BOTH the live system and the numbers we use to judge changes.

Pure numpy/scipy — no GPU, no ROS. Run with:
    python -m pytest ai_module/src/sam_mapper/tests/test_frame_sync.py
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sam_mapper import frame_sync


def odom(pos, quat=(0.0, 0.0, 0.0, 1.0), lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    return {"position": list(pos), "orientation": list(quat),
            "linear_velocity": list(lin), "angular_velocity": list(ang)}


# -- find_neighbouring_stamps ------------------------------------------------

def test_neighbouring_stamps_brackets_interior_query():
    stamps = [1.0, 2.0, 3.0]
    assert frame_sync.find_neighbouring_stamps(stamps, 2.5) == (2.0, 3.0)


def test_neighbouring_stamps_clamps_at_both_ends():
    stamps = [1.0, 2.0, 3.0]
    assert frame_sync.find_neighbouring_stamps(stamps, 0.5) == (1.0, 1.0)
    assert frame_sync.find_neighbouring_stamps(stamps, 9.9) == (3.0, 3.0)


# -- interpolate_odom --------------------------------------------------------

def test_interpolate_odom_reports_why_it_failed():
    """The three failure modes must stay distinguishable: map_node logs a rate-limited
    warning for TOO_OLD only, and treats NOT_CAUGHT_UP as 'retry next frame'."""
    assert frame_sync.interpolate_odom([], [], 1.0) == (None, frame_sync.NO_ODOM)

    stack, stamps = [odom([0, 0, 0]), odom([1, 0, 0])], [10.0, 11.0]
    assert frame_sync.interpolate_odom(stack, stamps, 9.0)[1] == frame_sync.TOO_OLD
    assert frame_sync.interpolate_odom(stack, stamps, 12.0)[1] == frame_sync.NOT_CAUGHT_UP


def test_interpolate_odom_is_linear_in_position():
    stack = [odom([0, 0, 0]), odom([10, 20, 30])]
    result, status = frame_sync.interpolate_odom(stack, [0.0, 1.0], 0.25)
    assert status == frame_sync.OK
    np.testing.assert_allclose(result["position"], [2.5, 5.0, 7.5])


def test_interpolate_odom_slerps_orientation_not_lerps_it():
    """A 90 deg yaw gap sampled at the midpoint must give 45 deg. Componentwise linear
    interpolation of the quaternion would give a different (un-normalised) rotation."""
    stack = [odom([0, 0, 0], Rotation.from_euler("z", 0).as_quat()),
             odom([0, 0, 0], Rotation.from_euler("z", 90, degrees=True).as_quat())]
    result, _ = frame_sync.interpolate_odom(stack, [0.0, 1.0], 0.5)
    yaw = Rotation.from_quat(result["orientation"]).as_euler("zyx", degrees=True)[0]
    assert yaw == pytest.approx(45.0, abs=1e-6)


def test_interpolate_odom_hits_endpoints_exactly():
    stack = [odom([0, 0, 0]), odom([2, 4, 6])]
    lo, _ = frame_sync.interpolate_odom(stack, [5.0, 6.0], 5.0)
    hi, _ = frame_sync.interpolate_odom(stack, [5.0, 6.0], 6.0)
    np.testing.assert_allclose(lo["position"], [0, 0, 0])
    np.testing.assert_allclose(hi["position"], [2, 4, 6])


def test_interpolate_odom_does_not_mutate_caller_buffers():
    """map_node trims its own ring buffers under a lock; the pure function must not, or
    the offline harness (which passes whole-bag arrays) would lose samples as it went."""
    stack = [odom([0, 0, 0]), odom([1, 1, 1]), odom([2, 2, 2])]
    stamps = [0.0, 1.0, 2.0]
    frame_sync.interpolate_odom(stack, stamps, 1.5)
    assert len(stack) == 3 and stamps == [0.0, 1.0, 2.0]


# -- gather_cloud ------------------------------------------------------------

def test_gather_cloud_takes_only_the_window_and_is_inclusive():
    clouds = [np.full((2, 3), float(i)) for i in range(5)]
    stamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    out = frame_sync.gather_cloud(clouds, stamps, stamp=2.0, before=1.0, after=1.0)
    assert out.shape == (6, 3)
    assert sorted(np.unique(out).tolist()) == [1.0, 2.0, 3.0]


def test_gather_cloud_returns_none_when_window_is_empty():
    clouds = [np.zeros((2, 3))]
    assert frame_sync.gather_cloud(clouds, [100.0], stamp=0.0, before=0.5, after=0.1) is None
    assert frame_sync.gather_cloud([], [], stamp=0.0, before=0.5, after=0.1) is None


def test_gather_cloud_window_is_asymmetric():
    """before=0.5/after=0.1 is the shipped config: lidar accumulated BEFORE the image is
    usable, lidar after it is mostly not yet observed from that viewpoint."""
    clouds = [np.full((1, 3), 1.0), np.full((1, 3), 2.0)]
    out = frame_sync.gather_cloud(clouds, [9.6, 10.3], stamp=10.0, before=0.5, after=0.1)
    assert out is not None and out.shape == (1, 3) and out[0, 0] == 1.0


# -- reconstruct_detections --------------------------------------------------

def test_reconstruct_detections_recovers_per_object_masks():
    from sam_mapper.detections import encode_instance_id

    id_map = np.zeros((6, 8), dtype=np.uint16)
    id_map[1:3, 2:5] = encode_instance_id(0)      # 6 px
    id_map[4, 7] = encode_instance_id(1)          # 1 px
    entries = [{"id": 0, "label": "chair", "confidence": 0.9, "bbox": [2, 1, 5, 3]},
               {"id": 1, "label": "table", "confidence": 0.4, "bbox": [7, 4, 8, 5]}]

    out = frame_sync.reconstruct_detections(id_map, entries)
    assert out["masks"].shape == (2, 6, 8)
    assert out["masks"][0].sum() == 6 and out["masks"][1].sum() == 1
    assert out["labels"].tolist() == ["chair", "table"]
    assert out["ids"].tolist() == [0, 1]


def test_reconstruct_detections_handles_background_negative_ids():
    """Background classes encode as 65536 + id. If that round-trip broke, wall/floor
    points would silently land on instance objects instead."""
    from sam_mapper.detections import encode_instance_id

    id_map = np.zeros((4, 4), dtype=np.uint16)
    id_map[0, :] = encode_instance_id(-1)
    entries = [{"id": -1, "label": "wall", "confidence": 0.8, "bbox": [0, 0, 4, 1]}]
    out = frame_sync.reconstruct_detections(id_map, entries)
    assert out["masks"][0].sum() == 4


def test_reconstruct_detections_empty_keeps_the_five_key_contract():
    """ObjMapper.update_map indexes all five keys unconditionally, so an empty frame must
    still produce correctly-shaped arrays rather than a bare {}."""
    id_map = np.zeros((5, 7), dtype=np.uint16)
    out = frame_sync.reconstruct_detections(id_map, [])
    assert set(out) == {"bboxes", "confidences", "labels", "ids", "masks"}
    assert out["masks"].shape == (0, 5, 7)
    assert out["bboxes"].shape == (0, 4)
    assert out["ids"].shape == (0,)


def test_reconstruct_detections_ignores_ids_absent_from_the_map():
    """A detection whose pixels were overwritten by a later overlapping mask still has an
    entry, and must yield an empty mask rather than raising."""
    id_map = np.zeros((4, 4), dtype=np.uint16)
    entries = [{"id": 7, "label": "vase", "confidence": 0.5, "bbox": [0, 0, 1, 1]}]
    out = frame_sync.reconstruct_detections(id_map, entries)
    assert out["masks"].shape == (1, 4, 4) and out["masks"][0].sum() == 0
