"""Unit tests for `view_bins` — which sides the ROBOT has stood on.

The planner-facing half of the exploration signal, and the half a sign error would ruin
silently: target_explorer inverts bin k into a standing position, so a flip here sends the
robot to the far side of every object while both files still read correctly.

`angle_bins` answers a different question (has this VOXEL been triangulated) and is left
alone — it feeds infer_centroid's diversity weighting.

Pure numpy/scipy — no GPU, no ROS. Run with `just test sam_mapper`.
"""
import numpy as np
import pytest

# single_object imports open3d at module scope for its clustering paths. It ships in the
# container but not on the host, so a host-side `pytest` skips this file rather than
# erroring out during collection.
pytest.importorskip("open3d", reason="run in the container: just test sam_mapper")

from sam_mapper.mapping_config import MappingConfig      # noqa: E402
from sam_mapper.single_object import SingleObject        # noqa: E402

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

def bled_point(obj, axis=0, distance=6.0):
    """A point far enough along one axis to blow the box past any class cap."""
    p = np.asarray(obj.provisional_centroid(), float).copy()
    p[axis] += distance
    return p.reshape(1, 3)


def test_a_point_beyond_the_class_cap_never_joins():
    """The whole point: a 5 cm voxel chain must not be able to walk from an object into the
    wall behind it and take the box with it."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05)
    before = obj.vote_stat.voxels.shape[0]
    dropped = obj.merge(bled_point(obj), IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0)
    assert dropped == 1
    assert obj.vote_stat.voxels.shape[0] == before
    assert obj.prior_gate_dropped == 1


def test_a_point_inside_the_class_cap_still_joins():
    """The gate is a cap, not a tightening — ordinary growth must be untouched."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05)
    before = obj.vote_stat.voxels.shape[0]
    near = np.array([[0.4, 0.0, 0.5]])           # well inside sofa's 5.39 x 2.19 x 1.63
    assert obj.merge(near, IDENTITY, np.array([1.0, 0.0, 0.75]), "sofa", 1.0) == 0
    assert obj.vote_stat.voxels.shape[0] > before


def test_the_gate_bounds_growth_however_long_the_run():
    """Monotone by construction: the span can never exceed the cap, so bleed cannot
    accumulate one frame at a time into a box that fits nothing."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05)
    prior = MappingConfig().dimension_priors.for_label("sofa")
    for step in range(1, 40):
        obj.merge(np.array([[0.3 * step, 0.0, 0.5]]), IDENTITY,
                  np.array([1.0, 0.0, 0.75]), "sofa", float(step))
    span = obj.vote_stat.voxels.max(axis=0) - obj.vote_stat.voxels.min(axis=0)
    assert span[0] <= prior[0] + 1e-6


def test_a_frame_the_gate_empties_still_counts_as_seen():
    """`info_frames` is what admission and the exploration planner read as "seen across
    frames". Only the geometry is refused, not the observation."""
    obj = make_object(center=(0.0, 0.0, 0.5), spread=0.05)
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
