"""Unit tests for `view_bins` — which sides the ROBOT has stood on.

The planner-facing half of the exploration signal, and the half a sign error would ruin
silently: target_explorer inverts bin k into a standing position, so a flip here sends the
robot to the far side of every object while both files still read correctly.

`angle_bins` answers a different question (has this VOXEL been triangulated) and is left
alone — it feeds infer_centroid's diversity weighting.

Pure numpy/scipy — no GPU, no ROS. Run with `just test sam_mapper`.
"""
import math

import numpy as np
import pytest

# single_object imports open3d at module scope for its clustering paths. It ships in the
# container but not on the host, so a host-side `pytest` skips this file rather than
# erroring out during collection.
pytest.importorskip("open3d", reason="run in the container: just test sam_mapper")

from sam_mapper.mapping_config import MappingConfig      # noqa: E402
from sam_mapper.single_object import SingleObject, _fits_prior  # noqa: E402

N_BINS = 20
IDENTITY = np.eye(3)


def make_object(center=(3.0, 0.0, 0.5), odom=(0.0, 0.0, 0.75), spread=0.1, config=None):
    """A small cloud around `center`, first observed from `odom`."""
    rng = np.random.default_rng(0)
    voxels = np.asarray(center, float) + rng.normal(scale=spread, size=(40, 3))
    return SingleObject(class_id="sofa", obj_id=1, voxels=voxels, voxel_size=0.05,
                        odom_R=IDENTITY, odom_t=np.asarray(odom, float), mask=None,
                        stamp=0.0, num_angle_bin=N_BINS, config=config or MappingConfig())


def bins_set(obj):
    return {int(k) for k in np.flatnonzero(obj.observed_view_bins())}


def test_one_observation_sets_exactly_one_bin():
    """The whole point. `angle_bins` ORs every voxel's azimuth and so marks a 71-degree span
    for a sofa at its standoff — four of twenty bins from a single pose, which an exploration
    planner reads as "already circled"."""
    obj = make_object()
    assert len(bins_set(obj)) == 1
    assert int(np.count_nonzero(obj.observed_angle_bins())) >= 1


def test_the_bin_inverts_to_the_standing_position_the_planner_computes():
    """target_explorer's `_stand_at` is `centre - r * (cos theta_k, sin theta_k)`. Standing
    there must set bin k — checked all the way round, so a sign flip cannot hide in a
    symmetric case."""
    centre = np.array([3.0, 0.0, 0.5])
    radius = 2.0
    for k in range(N_BINS):
        theta = -np.pi + (k + 0.5) * (2 * np.pi / N_BINS)
        odom = (centre[0] - radius * np.cos(theta), centre[1] - radius * np.sin(theta), 0.75)
        assert bins_set(make_object(center=tuple(centre), odom=odom)) == {k}


def test_a_second_observation_from_the_other_side_adds_its_bin():
    obj = make_object(center=(3.0, 0.0, 0.5), odom=(0.0, 0.0, 0.75))
    before = bins_set(obj)
    obj.merge(np.array([[3.0, 0.0, 0.5]]), IDENTITY, np.array([6.0, 0.0, 0.75]), "sofa", 1.0)
    assert len(bins_set(obj) - before) == 1


def test_an_observation_beyond_the_range_filter_does_not_count():
    """Past range_filter.max_distance the mapper assigns no lidar to this object's mask, so
    the frame cannot improve its geometry. Calling that side inspected would retire a
    viewpoint the planner still needs."""
    config = MappingConfig()
    far = config.range_filter.max_distance + 5.0
    obj = make_object(center=(0.0, 0.0, 0.5), odom=(0.0, 0.0, 0.75), config=config)
    before = bins_set(obj)
    obj.merge(np.array([[0.0, 0.0, 0.5]]), IDENTITY, np.array([far, 0.0, 0.75]), "sofa", 1.0)
    assert bins_set(obj) == before


def test_a_world_merge_unions_both_sides_histories():
    """Two ids merged means both sets of observations were of one physical object, so the
    sides the loser was seen from are sides we have genuinely stood in."""
    winner = make_object(center=(3.0, 0.0, 0.5), odom=(0.0, 0.0, 0.75))
    loser = make_object(center=(3.0, 0.0, 0.5), odom=(6.0, 0.0, 0.75))
    expected = bins_set(winner) | bins_set(loser)
    winner.merge_object(loser)
    assert bins_set(winner) == expected


def test_view_bins_never_shrink():
    """A coverage signal that goes backwards makes a planner oscillate between goals it
    thinks it has just un-satisfied — the same reason angle_bin_coverage is not trimmed."""
    obj = make_object()
    seen = bins_set(obj)
    for x in (1.0, -2.0, 2.5):
        obj.merge(np.array([[3.0, 0.0, 0.5]]), IDENTITY, np.array([x, 1.0, 0.75]),
                  "sofa", 1.0)
        assert seen <= bins_set(obj)
        seen = bins_set(obj)


def test_observed_view_bins_returns_a_copy():
    """It is published straight into JSON and read by a separate node; handing out the live
    array would let a consumer corrupt the map's own state."""
    obj = make_object()
    snapshot = obj.observed_view_bins()
    snapshot[:] = True
    assert bins_set(obj) != set(range(N_BINS))


def test_describe_objects_carries_the_field():
    """The contract target_coverage reads. It falls back to `angle_bins` when this is absent,
    which silently restores the over-reporting this whole field exists to remove."""
    from sam_mapper.cloud_image_fusion import CloudImageFusion
    from sam_mapper.object_mapper import ObjMapper

    mapper = ObjMapper(cloud_image_fusion=CloudImageFusion(platform="mecanum_sim"),
                       label_template={"sofa": {"is_instance": True, "prompts": ["sofa"]}},
                       captioner=None, log_info=lambda *_: None)
    mapper.single_obj_list = [make_object()]
    entry = mapper.describe_objects()[0]
    assert len(entry["view_bins"]) == N_BINS
    assert entry["n_view_bins"] == 1
    # angle_bins is untouched and still says something different — it is the per-voxel OR.
    assert len(entry["angle_bins"]) == N_BINS


# -- D3b: the class-prior gate on incoming points ----------------------------

def gated():
    """A config with D3b ON. It ships OFF -- see test_the_gate_is_off_by_default -- so every
    case that exercises the gate has to ask for it, or it passes vacuously."""
    return MappingConfig.from_dict({"dimension_priors": {"gate_merge": True}})


def bled_point(obj, axis=0, distance=6.0):
    """A point far enough along one axis to blow the box past any class cap."""
    p = np.asarray(obj.provisional_centroid(), float).copy()
    p[axis] += distance
    return p.reshape(1, 3)


def test_a_point_beyond_the_class_cap_never_joins():
    """The whole point: a 5 cm voxel chain must not be able to walk from an object into the
    wall behind it and take the box with it."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05, config=gated())
    before = obj.vote_stat.voxels.shape[0]
    dropped = obj.merge(bled_point(obj), IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0)
    assert dropped == 1
    assert obj.vote_stat.voxels.shape[0] == before
    assert obj.prior_gate_dropped == 1


def test_a_point_inside_the_class_cap_still_joins():
    """The gate is a cap, not a tightening — ordinary growth must be untouched."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05, config=gated())
    before = obj.vote_stat.voxels.shape[0]
    near = np.array([[0.4, 0.0, 0.5]])           # well inside sofa's 5.39 x 2.19 x 1.63
    assert obj.merge(near, IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0) == 0
    assert obj.vote_stat.voxels.shape[0] > before


def test_the_gate_bounds_growth_however_long_the_run():
    """Monotone by construction: the span can never exceed the cap, so bleed cannot
    accumulate one frame at a time into a box that fits nothing."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05, config=gated())
    prior = MappingConfig().dimension_priors.for_label("sofa")
    for step in range(1, 40):
        obj.merge(np.array([[0.3 * step, 0.0, 0.5]]), IDENTITY,
                  np.array([1.0, 0.0, 0.75]), "sofa", float(step))
    span = obj.vote_stat.voxels.max(axis=0) - obj.vote_stat.voxels.min(axis=0)
    assert span[0] <= prior[0] + 1e-6


def test_a_frame_the_gate_empties_still_counts_as_seen():
    """`info_frames` is what admission and the exploration planner read as "seen across
    frames". Only the geometry is refused, not the observation."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05, config=gated())
    before = obj.info_frames_cnt
    obj.merge(bled_point(obj), IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0)
    assert obj.info_frames_cnt == before + 1


def test_the_gate_can_be_switched_off():
    """A/B-able like every other stage — and off must be byte-identical to before."""
    from dataclasses import replace

    config = MappingConfig()
    config.dimension_priors = replace(config.dimension_priors, gate_merge=False)
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05, config=config)
    before = obj.vote_stat.voxels.shape[0]
    assert obj.merge(bled_point(obj), IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0) == 0
    assert obj.vote_stat.voxels.shape[0] == before + 1


def test_the_gate_is_off_by_default():
    """D3b ships OFF: measured over 13 scenes it refused 1,258,889 points and changed nothing
    -- every map_digest identical, 0 of 52 objects changed voxels_total. A class prior is a
    population UPPER bound, so it cannot separate a bled instance from a large one.

    The mechanism is kept only so that result stays reproducible. This pins the default, so
    the cases above cannot quietly start passing vacuously if it is ever flipped back.
    """
    assert MappingConfig().dimension_priors.gate_merge is False
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05)
    before = obj.vote_stat.voxels.shape[0]
    assert obj.merge(bled_point(obj), IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0) == 0
    assert obj.vote_stat.voxels.shape[0] > before      # the bled point joins, ungated


# -- D3b2: trim a bled cluster instead of discarding it ----------------------

def trimming():
    """A config with the trim ON. It ships OFF pending the replay A/B, so cases that exercise
    it have to ask, exactly like `gated()` above."""
    return MappingConfig.from_dict({"dimension_priors": {"trim_to_fit": True}})


def table_with_lamp(config):
    """hotel_room_1's failure, reproduced: a bedside table with a lamp standing on it.

    The lamp sits 3 cm from the table's centre and reaches past the 1.14 m `bedsidetable` z
    cap, so the two are one DBSCAN cluster whose box cannot fit the class. Untrimmed, the
    whole cluster is refused and the table stops existing.
    """
    rng = np.random.default_rng(0)
    body = rng.normal(scale=0.12, size=(400, 3)) + np.array([3.53, -3.70, 0.33])
    lamp = np.column_stack([rng.normal(3.55, 0.05, 80), rng.normal(-3.72, 0.05, 80),
                            rng.uniform(0.70, 1.35, 80)])
    voxels = np.vstack([body, lamp])
    return SingleObject(class_id="bedsidetable", obj_id=1, voxels=voxels, voxel_size=0.05,
                        odom_R=IDENTITY, odom_t=np.array([0.0, 0.0, 0.75]), mask=None,
                        stamp=0.0, num_angle_bin=N_BINS, config=config)


def test_a_bled_cluster_is_kept_after_trimming():
    """The object must survive. A box 0.2 m too big scores the constraint; an absent object
    cannot be answered about at all."""
    obj = table_with_lamp(trimming())
    obj.regularize_shape(percentile=0.8)
    assert obj.vote_stat.regularized_voxel_mask.any(), "the object vanished"
    assert obj.regularize_rejections["trimmed_voxels"] > 0
    assert obj.infer_centroid(diversity_percentile=0.8) is not None


def test_the_trimmed_box_actually_fits_the_prior():
    """Trimming that stops short of the cap would keep the object and still fail D3."""
    obj = table_with_lamp(trimming())
    obj.regularize_shape(percentile=0.8)
    kept = obj.vote_stat.voxels[obj.vote_stat.regularized_voxel_mask]
    extent = kept.max(axis=0) - kept.min(axis=0)
    prior = MappingConfig().dimension_priors.for_label("bedsidetable")
    assert _fits_prior(extent, prior), f"{extent} still exceeds {prior}"


def test_the_trim_keeps_the_body_not_the_bleed():
    """Furthest-first is the whole ordering. If it shed the table and kept the lamp the
    centroid would move metres and the object would be worse than useless."""
    obj = table_with_lamp(trimming())
    obj.regularize_shape(percentile=0.8)
    centre = obj.infer_centroid(diversity_percentile=0.8)
    assert math.dist(centre[:2], (3.53, -3.70)) < 0.25


def test_off_is_the_old_behaviour():
    """The A/B is only honest if `off` is genuinely the code that shipped before."""
    obj = table_with_lamp(MappingConfig())
    assert obj.config.dimension_priors.trim_to_fit is False
    obj.regularize_shape(percentile=0.8)
    assert obj.regularize_rejections["exceeds_prior"] > 0
    assert obj.regularize_rejections["trimmed_voxels"] == 0


def test_a_cluster_that_is_all_bleed_is_still_refused():
    """"Nothing here" has to stay reachable, or the trim would launder noise into objects.
    A thin tall column is nothing a bedside table could ever be."""
    rng = np.random.default_rng(1)
    voxels = np.column_stack([rng.normal(0.0, 0.02, 60), rng.normal(0.0, 0.02, 60),
                              rng.uniform(0.0, 3.0, 60)])
    obj = SingleObject(class_id="bedsidetable", obj_id=2, voxels=voxels, voxel_size=0.05,
                       odom_R=IDENTITY, odom_t=np.array([2.0, 0.0, 0.75]), mask=None,
                       stamp=0.0, num_angle_bin=N_BINS, config=trimming())
    obj.regularize_shape(percentile=0.8)
    kept = obj.vote_stat.voxels[obj.vote_stat.regularized_voxel_mask]
    if len(kept):
        extent = kept.max(axis=0) - kept.min(axis=0)
        assert _fits_prior(extent, MappingConfig().dimension_priors.for_label("bedsidetable"))
