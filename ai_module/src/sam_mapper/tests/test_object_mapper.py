"""Unit tests for ObjMapper: the funnel from detection to published 3D object.

Pure numpy + open3d, no ROS. xfail(strict=True) marks behaviour the design calls for but
the code lacks — they fail the suite if fixed without removing the marker."""
import collections

import numpy as np
import pytest

pytest.importorskip("open3d", reason="run in the container: just test sam_mapper")

from sam_mapper.cloud_image_fusion import CloudImageFusion  # noqa: E402
from sam_mapper.mapping_config import MappingConfig  # noqa: E402
from sam_mapper.object_mapper import ObjMapper  # noqa: E402

W, H = 1920, 640
CAM_Z = 0.1
LABELS = {
    "chair": {"is_instance": True, "prompts": ["chair"]},
    "table": {"is_instance": True, "prompts": ["table"]},
}


def mapper():
    return ObjMapper(cloud_image_fusion=CloudImageFusion(platform="mecanum_sim"),
                     label_template=LABELS, captioner=None, log_info=lambda *_: None)


def identity_odom():
    return {"position": np.zeros(3), "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "linear_velocity": np.zeros(3), "angular_velocity": np.zeros(3)}


def blob(center, half=0.15, n=9):
    """A dense little cube of points — enough voxels to survive downsampling."""
    g = np.linspace(-half, half, n)
    return np.array([[center[0] + x, center[1] + y, center[2] + z]
                     for x in g for y in g for z in g])


def mask_covering(fusion, points, pad=25):
    """A mask that exactly covers where `points` project to, plus a small margin.

    Built from the projection rather than hand-placed pixels so the test stays correct if
    the extrinsics change — and because update_map erodes masks by ~5 px, so a tight mask
    would vanish and the test would silently assert nothing.
    """
    uv = fusion.scan2pixels(points)[:, :2].astype(int)
    m = np.zeros((H, W), dtype=bool)
    u0, u1 = uv[:, 0].min() - pad, uv[:, 0].max() + pad
    v0, v1 = uv[:, 1].min() - pad, uv[:, 1].max() + pad
    m[max(v0, 0):min(v1, H), max(u0, 0):min(u1, W)] = True
    return m


def detections(ids, labels, masks):
    return {
        "bboxes": np.zeros((len(ids), 4), dtype=float),
        "confidences": np.full(len(ids), 0.9),
        "labels": np.asarray(labels, dtype=object),
        "ids": np.asarray(ids, dtype=int),
        "masks": np.asarray(masks, dtype=bool),
    }


def feed(mp, fusion, frames, blobs, ids, labels, start=100.0, dt=0.1):
    """Run `frames` identical frames of the given blobs, as map_node would."""
    cloud = np.vstack(blobs)
    masks = [mask_covering(fusion, b) for b in blobs]
    for i in range(frames):
        mp.update_map(detections(ids, labels, masks), start + i * dt,
                      identity_odom(), cloud)


# -- defect 13: the crash T0.1 found -----------------------------------------

def test_update_map_survives_without_ever_publishing():
    """The prune test used to read a lazily-computed attribute that only the PUBLISH path
    populated, so any caller that did not serialise every frame hit AttributeError on
    frame 6. That is how the offline harness died on its first run."""
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    feed(mp, fusion, 12, [blob([3.0, 0.0, CAM_Z])], ids=[0], labels=["chair"])
    assert len(mp.single_obj_list) >= 1


def test_serialize_emits_a_box_per_tracked_object():
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    feed(mp, fusion, 8, [blob([3.0, 0.0, CAM_Z])], ids=[0], labels=["chair"])
    out = mp.serialize_map_to_dict(101.0)
    assert out, "expected at least one serialised object"
    entry = next(iter(out.values()))
    assert entry["label"] == "chair"
    assert set(entry["bbox3d"]) == {"center", "extent", "rotation"}


# -- object identity ---------------------------------------------------------

def test_same_track_id_across_frames_stays_one_object():
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    feed(mp, fusion, 10, [blob([3.0, 0.0, CAM_Z])], ids=[7], labels=["chair"])
    instances = [o for o in mp.single_obj_list if o.obj_id[0] >= 0]
    assert len(instances) == 1
    assert 7 in instances[0].obj_id


def test_two_well_separated_objects_stay_separate():
    """Sanity floor for the merge rule: 4 m apart, same class, must never fuse."""
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    blobs = [blob([3.0, -2.0, CAM_Z]), blob([3.0, 2.0, CAM_Z])]
    feed(mp, fusion, 10, blobs, ids=[1, 2], labels=["chair", "chair"])
    assert len({tuple(o.obj_id) for o in mp.single_obj_list if o.obj_id[0] >= 0}) == 2


# -- defect 10: merging is nearest-centroid, not co-visibility ---------------

# Two 0.2 m objects, 0.4 m apart, 2 m ahead — two pillows on a sofa, say. Chosen so that
# BOTH conditions the test needs actually hold, which an earlier version of this test got
# wrong in both directions:
#   * their masks are genuinely disjoint (17 px gap, 45 px wide, so they survive the 5 px
#     erosion in update_map). At 3 m / 0.6 m apart with the default pad they OVERLAPPED,
#     so the "two distinct masks" premise silently did not hold.
#   * 0.4 m separation trips the merge rule's unconditional `dist < 0.5` clause. At 0.6 m
#     nothing merged, because dist_thresh for objects this small is only ~0.13 — so the
#     test passed for a reason that had nothing to do with the defect.
PAIR_RANGE, PAIR_SEP, PAIR_HALF, PAIR_PAD = 2.0, 0.4, 0.10, 6


@pytest.mark.xfail(strict=True, reason="defect 10: the merge rule's unconditional "
                                       "`dist < 0.5` fuses any two same-label objects "
                                       "within 0.5 m, even when they are co-visible "
                                       "under different masks (Tier 2.5 view-consensus)")
def test_objects_seen_simultaneously_under_different_masks_must_not_merge():
    """Appearing in the SAME frame under two different masks is conclusive evidence of two
    objects; no amount of centroid proximity should override it. The merge test today is
    `dist < norm((ext_a/2 + ext_b/2)/2) * 0.5 or dist < 0.5` — that second clause knows
    nothing about co-visibility, and 0.5 m is larger than the spacing of most of the small
    objects the questions actually ask about (pillows, vases, books, cups).
    """
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    blobs = [blob([PAIR_RANGE, -PAIR_SEP / 2, CAM_Z], half=PAIR_HALF),
             blob([PAIR_RANGE, PAIR_SEP / 2, CAM_Z], half=PAIR_HALF)]
    masks = [mask_covering(fusion, b, pad=PAIR_PAD) for b in blobs]

    # Guard the premise: if a future extrinsics change makes these overlap, fail loudly
    # here rather than quietly testing nothing.
    assert not (masks[0] & masks[1]).any(), "test geometry broken: masks overlap"

    cloud = np.vstack(blobs)
    for i in range(12):
        mp.update_map(detections([1, 2], ["chair", "chair"], masks),
                      100.0 + i * 0.1, identity_odom(), cloud)

    instances = [o for o in mp.single_obj_list if o.obj_id[0] >= 0]
    merged = [o for o in instances if len(o.obj_id) > 1]
    assert not merged, f"co-visible distinct objects were merged: {[o.obj_id for o in instances]}"


def test_same_label_objects_beyond_the_merge_radius_stay_separate():
    """The complement, and the reason the earlier version of the test above passed for the
    wrong reason: at 0.6 m these small objects are outside BOTH merge clauses, so nothing
    fuses. Pins that boundary so a change to the merge rule has to be deliberate."""
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    blobs = [blob([3.0, -0.3, CAM_Z], half=0.15), blob([3.0, 0.3, CAM_Z], half=0.15)]
    feed(mp, fusion, 12, blobs, ids=[1, 2], labels=["chair", "chair"])

    instances = [o for o in mp.single_obj_list if o.obj_id[0] >= 0]
    assert not [o for o in instances if len(o.obj_id) > 1]


# -- defect 8: stale objects freeze instead of being pruned ------------------

def test_long_unseen_objects_are_retained():
    """The scene is STATIC and the robot moves, so an object last seen 60 frames ago has
    not gone anywhere — the robot simply looked elsewhere. It must stay in the map.

    Age-based eviction (AB3DMOT's `max_age`) was implemented and measured, and it deleted
    correct map entries: 20 tracked objects -> 12, bestIoU 0.031 -> 0.022. `max_age` is a
    multi-object-TRACKING idea and this is not a tracking problem — the same category
    error the design notes call out for Kalman-filter 3D MOT.
    """
    mp, fusion = mapper(), CloudImageFusion(platform="mecanum_sim")
    chair, table = blob([3.0, -2.0, CAM_Z]), blob([3.0, 2.0, CAM_Z])

    feed(mp, fusion, 3, [chair, table], ids=[1, 2], labels=["chair", "table"])
    for i in range(60):                       # chair out of view; table still seen
        mp.update_map(detections([2], ["table"], [mask_covering(fusion, table)]),
                      200.0 + i * 0.1, identity_odom(), table)

    labels = [o.get_dominant_label() for o in mp.single_obj_list if o.obj_id[0] >= 0]
    assert "chair" in labels, "a static object the robot looked away from must be retained"
    assert "table" in labels


# -- B7: range-gap cut -------------------------------------------------------
#
# B6 has no z-buffer, so a mask claims every point along its line of sight. These cover the
# cut that removes the ones behind the object, and — as important — the cases where it must
# do nothing, because a filter that trims real geometry is worse than no filter.



def gap_mapper(**range_gap):
    cfg = MappingConfig.from_dict({"range_gap": {"enabled": True, **range_gap}})
    return ObjMapper(cloud_image_fusion=CloudImageFusion(platform="mecanum_sim"),
                     label_template=LABELS, captioner=None, log_info=lambda *_: None,
                     config=cfg)


def ray_cloud(ranges):
    """Points strung out along +x at the given ranges from the origin."""
    return np.array([[r, 0.0, 0.0] for r in ranges])


def test_range_gap_splits_object_from_the_wall_behind_it():
    """The motivating case: a sofa at ~2 m, a wall at ~5 m, one mask claiming both. The
    global range filter cannot help — both are inside 6 m."""
    mp = gap_mapper()
    cloud = ray_cloud([2.00, 2.05, 2.10, 2.15, 2.20, 5.00, 5.05, 5.10])
    stats = collections.defaultdict(int)

    kept = mp._range_gap_cut(cloud, np.zeros(3), None, stats)

    assert len(kept) == 5, "only the near cluster survives"
    assert kept[:, 0].max() < 3.0
    assert stats["points_dropped_range_gap"] == 3
    assert stats["detections_trimmed_range_gap"] == 1


def test_range_gap_keeps_a_long_object_whose_own_depth_is_continuous():
    """A 3 m sofa viewed end-on spans 3 m of range legitimately. A fixed depth budget would
    amputate it; a GAP test must not, because its returns are continuous."""
    mp = gap_mapper()
    cloud = ray_cloud(np.arange(2.0, 5.0, 0.1))
    stats = collections.defaultdict(int)

    kept = mp._range_gap_cut(cloud, np.zeros(3), None, stats)

    assert len(kept) == len(cloud)
    assert stats["points_dropped_range_gap"] == 0


def test_range_gap_ignores_gaps_below_the_threshold():
    """Within-object structure (an armrest, a cushion seam) must not be read as background."""
    mp = gap_mapper(min_gap=0.35)
    cloud = ray_cloud([2.0, 2.1, 2.2, 2.45, 2.55, 2.65])       # widest gap 0.25 < 0.35
    kept = mp._range_gap_cut(cloud, np.zeros(3), None, collections.defaultdict(int))
    assert len(kept) == len(cloud)


def test_range_gap_skips_the_histogram_when_there_are_too_few_points():
    """A pillow averages ~6 lidar points. Below min_points there is no distribution to read
    a gap from, and guessing is worse than passing through."""
    mp = gap_mapper(min_points=6)
    cloud = ray_cloud([2.0, 2.1, 5.0])                          # a real gap, but only 3 points
    kept = mp._range_gap_cut(cloud, np.zeros(3), None, collections.defaultdict(int))
    assert len(kept) == 3, "too few points to trust a gap"


def test_range_gap_depth_ceiling_works_at_low_point_counts():
    """...which is what the geometric ceiling is for: an object is not much deeper than its
    own apparent width, and that needs no histogram. A narrow mask (20 px ~ 0.065 rad) at
    2 m implies ~0.13 m of width, so a 5 m point cannot belong to it."""
    mp = gap_mapper(min_points=6, max_depth_scale=2.0)
    mask = np.zeros((640, 1920), dtype=bool)
    mask[300:320, 950:970] = True
    cloud = ray_cloud([2.0, 2.1, 5.0])

    kept = mp._range_gap_cut(cloud, np.zeros(3), mask, collections.defaultdict(int))

    assert len(kept) == 2, "the ceiling must catch what the histogram cannot"


def test_range_gap_disabled_is_a_pass_through():
    """Default config must reproduce the measured baseline exactly."""
    mp = mapper()
    cloud = ray_cloud([2.0, 2.1, 2.2, 5.0, 5.1, 5.2])
    kept = mp._range_gap_cut(cloud, np.zeros(3), None, collections.defaultdict(int))
    assert len(kept) == len(cloud)


def test_range_gap_measures_from_the_robot_not_the_origin():
    """Ranges are ||point - robot||. Reading them from the world origin would make the cut
    depend on where the map happens to be anchored."""
    mp = gap_mapper()
    robot = np.array([10.0, 0.0, 0.0])
    cloud = np.array([[10.0 + r, 0.0, 0.0]
                      for r in [2.00, 2.05, 2.10, 2.15, 2.20, 5.00, 5.05, 5.10]])

    kept = mp._range_gap_cut(cloud, robot, None, collections.defaultdict(int))

    assert len(kept) == 5


# -- D8: co-visibility gate --------------------------------------------------
#
# Distance cannot tell "two pillows 0.44 m apart" from "one pillow split across two ids".
# Co-visibility can: SAM 3 saw both in the SAME frame under different track ids, so they are
# different objects however close they are.

def covis_mapper(block=True):
    cfg = MappingConfig.from_dict({"world_merge": {"block_covisible": block}})
    return ObjMapper(cloud_image_fusion=CloudImageFusion(platform="mecanum_sim"),
                     label_template=LABELS, captioner=None, log_info=lambda *_: None,
                     config=cfg)


def test_adjacency_graph_tolerates_an_unseen_id():
    """is_adjacent used to index the dict directly, so an id that reached the merge check
    without ever being added as a vertex raised KeyError mid-frame."""
    mp = covis_mapper()
    assert mp.adjacency_graph.is_adjacent(999, 1000) is False
    assert mp.adjacency_graph.is_set_adjacent([999], [1000]) is False


def test_covisible_ids_are_never_merge_targets():
    """The arabic_room pillow case: 0.44 m apart, well inside the unconditional 0.5 m rule,
    but both visible in one frame — so they are two pillows, not one."""
    mp = covis_mapper(block=True)
    mp.adjacency_graph.add_edge(1, 2)          # seen together in some frame

    assert mp.adjacency_graph.is_set_adjacent([1], [2]), "precondition: ids are co-visible"
    assert not mp.adjacency_graph.is_set_adjacent([1], [3]), "id 3 never co-seen with 1"


def test_covisibility_survives_merging():
    """After ids 1 and 2 merge into one object, that object must still count as co-visible
    with anything either of them was seen beside — otherwise a blob keeps absorbing its
    neighbours one id at a time."""
    mp = covis_mapper()
    mp.adjacency_graph.add_edge(1, 5)
    merged_ids = [1, 2]                        # object that has already absorbed id 2
    assert mp.adjacency_graph.is_set_adjacent(merged_ids, [5])


def test_block_covisible_can_be_disabled():
    """Kept switchable because the counter-case is real: SAM 3 splitting one object across
    two ids in a single frame would have its merge blocked."""
    assert covis_mapper(block=False).config.world_merge.block_covisible is False
    assert covis_mapper(block=True).config.world_merge.block_covisible is True


def test_covisible_overlap_threshold_separates_the_two_populations():
    """The guard has to tell "two close objects" from "one object split across two ids",
    using PREDICTED boxes — which this pipeline is known to produce oversized.

    An earlier 0.10 threshold looked fine on true-size boxes and failed at 1.2x inflation.
    This pins the band: distinct objects stay below even at 2x, fragments stay above.
    """
    from sam_mapper.box_geometry import overlap_fraction
    thresh = MappingConfig().world_merge.covisible_overlap

    pillow = np.array([0.21, 0.42, 0.36])           # arabic_room, 0.44 m apart
    for inflation in (1.0, 1.5, 2.0):
        e = pillow * inflation
        ovl = overlap_fraction([0, 0, 0], e, [0, 0.44, 0], e)
        assert ovl < thresh, f"distinct pillows merge at {inflation}x inflation (ovl {ovl:.3f})"

    sofa = [2.19, 0.78, 0.68]
    assert overlap_fraction([0, 0, 0], sofa, [0.3, 0, 0], [0.4, 0.4, 0.4]) >= thresh
    assert overlap_fraction([0, 0, 0], sofa, [0.6, 0, 0], [1.2, 0.78, 0.68]) >= thresh
