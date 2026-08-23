"""Unit tests for the equirectangular lidar->camera projection.

Geometry, verified against the implementation: cam_R maps body (x,y,z) -> (-y,-z,x);
scale = W/2pi = 305.577 px/rad on both axes; VFOV = H/scale = 120 deg, so the valid band
is +-60 deg elevation; horiDis is the HORIZONTAL range, not the Euclidean one.

xfail(strict=True) marks behaviour the design calls for but the code lacks."""
import numpy as np
import pytest

from sam_mapper.cloud_image_fusion import (
    CloudImageFusion, scan2pixels_mecanum, scan2pixels_mecanum_sim)
from sam_mapper.mapping_config import OcclusionConfig

W, H = 1920, 640
SCALE = W / (2 * np.pi)
CAM_Z = 0.1                      # mecanum_sim camera height above the lidar origin


def project(points):
    return scan2pixels_mecanum_sim(np.asarray(points, dtype=float))


def at_elevation(deg, horiz=5.0):
    """A point `horiz` metres ahead at `deg` above the camera (not the lidar) origin."""
    return [horiz, 0.0, CAM_Z + horiz * np.tan(np.radians(deg))]


# -- basic projection --------------------------------------------------------

def test_cardinal_directions_land_on_expected_columns():
    out = project([[5, 0, CAM_Z], [0, 5, CAM_Z], [0, -5, CAM_Z]])
    assert int(out[0, 0]) == W // 2 + 1        # forward
    assert int(out[1, 0]) == W // 4 + 1        # left  (+y) is a quarter turn back
    assert int(out[2, 0]) == 3 * W // 4 + 1    # right (-y)


def test_horizon_maps_to_the_middle_row():
    out = project([[5, 0, CAM_Z], [12, 0, CAM_Z]])
    assert int(out[0, 1]) == H // 2 + 1
    assert int(out[1, 1]) == H // 2 + 1        # independent of range


def test_third_column_is_horizontal_range_not_euclidean():
    """horiDis = sqrt(x^2+y^2) in body frame. Callers that want a true depth for occlusion
    ordering must not reuse it: at high elevation it under-reports badly."""
    out = project([[3.0, 4.0, CAM_Z + 10.0]])
    assert out[0, 2] == pytest.approx(5.0)     # 3-4-5, the +10 in z is ignored


def test_vertical_field_of_view_is_120_degrees():
    assert H / SCALE == pytest.approx(np.deg2rad(120.0))


def test_platforms_differ_only_by_translation():
    point = np.array([[5.0, 0.0, 0.0]])
    assert not np.allclose(scan2pixels_mecanum_sim(point), scan2pixels_mecanum(point))


def test_unknown_platform_is_rejected():
    with pytest.raises(ValueError, match="Invalid platform"):
        CloudImageFusion(platform="wheelchair")


# -- generate_seg_cloud ------------------------------------------------------

def make_masks(*specs, shape=(H, W)):
    masks = []
    for rows, cols in specs:
        m = np.zeros(shape, dtype=bool)
        m[rows[0]:rows[1], cols[0]:cols[1]] = True
        masks.append(m)
    return masks


def test_generate_seg_cloud_splits_points_by_mask_and_returns_world_frame():
    fusion = CloudImageFusion(platform="mecanum_sim")
    forward, left = [5.0, 0.0, CAM_Z], [0.0, 5.0, CAM_Z]
    cloud = np.array([forward, left])
    masks = make_masks(((300, 340), (940, 980)),     # covers forward  (961, 321)
                       ((300, 340), (460, 500)))     # covers left     (481, 321)

    t = np.array([1.0, 2.0, 3.0])
    out = fusion.generate_seg_cloud(cloud, masks, R_b2w=np.eye(3), t_b2w=t)

    assert len(out) == 2
    np.testing.assert_allclose(out[0], [np.array(forward) + t])
    np.testing.assert_allclose(out[1], [np.array(left) + t])


def test_generate_seg_cloud_returns_none_without_masks():
    fusion = CloudImageFusion(platform="mecanum_sim")
    cloud = np.array([[5.0, 0.0, CAM_Z]])
    assert fusion.generate_seg_cloud(cloud, None, np.eye(3), np.zeros(3)) is None
    assert fusion.generate_seg_cloud(cloud, [], np.eye(3), np.zeros(3)) is None


# -- defect 2: out-of-FOV points are clipped instead of rejected -------------

def test_points_beyond_the_vertical_fov_currently_pile_onto_the_edge_rows():
    """Documents the CURRENT behaviour, so the fix has something to change.

    np.clip pins everything above +60 deg to row 0. In a real room that is not a rare
    corner case: with the sensor 0.75 m up and a 2.78 m ceiling, every ceiling return
    within 1.17 m horizontal radius lands on row 0.
    """
    rows = [int(project([at_elevation(d)])[0, 1]) for d in (61, 70, 85)]
    assert rows == [0, 0, 0]


def test_out_of_fov_points_must_not_be_claimed_by_an_edge_mask():
    """A point 85 deg above the horizon is outside the 120 deg FOV — the camera never saw
    it, so no mask may claim it. Under the old `clip` default it pinned to row 0 and was
    swallowed by any mask touching the top edge, which is how ceiling returns ended up
    inside tall objects."""
    fusion = CloudImageFusion(platform="mecanum_sim")
    cloud = np.array([at_elevation(85), [5.0, 0.0, CAM_Z]])
    masks = make_masks(((0, 340), (940, 980)))       # spans the top edge to the horizon

    out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
    assert len(out[0]) == 1, "only the in-FOV point should be claimed"


# -- defect 3: azimuth wrap --------------------------------------------------

def test_azimuth_wraps_cleanly_across_the_rear_seam():
    """Two points either side of 'directly behind' must stay neighbours modulo width.

    They do, which is why this passes. The azimuth clip is a near-non-issue: u only ever
    reaches 1921, and the clip target (1919) is wrap-adjacent to the true value (0), so
    the worst error is ~2 px = 0.007 rad. The seam problem that actually costs us is
    upstream — SAM 3 sees a cut image and emits two tracks for one object — and map_node
    cannot fix it. See plan Tier 3.6.
    """
    left_of_seam = int(project([[-5.0, 1e-6, CAM_Z]])[0, 0])
    right_of_seam = int(project([[-5.0, -1e-6, CAM_Z]])[0, 0])
    wrap_gap = min(abs(left_of_seam - right_of_seam),
                   W - abs(left_of_seam - right_of_seam))
    assert wrap_gap <= 3


def test_azimuth_stays_in_bounds_all_the_way_round():
    """Sweep a full turn: every column must be a valid index. This is the property that
    actually matters; the exact value in the last ~2 px band is not pinned deliberately,
    because clip and mod differ there by 1-2 px and the behaviour is float-sensitive.
    Writing a brittle test for a 0.007 rad discrepancy would cost more than the fix."""
    angles = np.linspace(-np.pi, np.pi, 2000)
    pts = np.stack([5 * np.cos(angles), 5 * np.sin(angles),
                    np.full_like(angles, CAM_Z)], axis=1)
    cols = project(pts)[:, 0].astype(int)
    assert cols.min() >= 0 and cols.max() <= W - 1


# -- defect 1: no occlusion test ---------------------------------------------

def test_occluded_points_on_the_same_ray_must_not_be_claimed():
    """Two points on one ray at 2 m and 5 m. A mask over that direction covers the near
    surface; the far point is behind it and must be rejected.

    This is the wall-seen-through-the-gap-under-a-chair case, and it is the single
    largest source of inflated box extents.
    """
    fusion = CloudImageFusion(platform="mecanum_sim")
    near, far = [2.0, 0.0, CAM_Z], [5.0, 0.0, CAM_Z]
    cloud = np.array([near, far])
    masks = make_masks(((300, 340), (940, 980)))     # both project to (961, 321)

    out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
    assert len(out[0]) == 1
    np.testing.assert_allclose(out[0][0], near)


def test_disabling_the_z_buffer_restores_the_old_bleed():
    """The behaviour the z-buffer replaced, reachable by config so the change is revertible
    on a rig where it turns out to cost more than it saves."""
    fusion = CloudImageFusion(platform="mecanum_sim", occlusion=OcclusionConfig(enabled=False))
    cloud = np.array([[2.0, 0.0, CAM_Z], [5.0, 0.0, CAM_Z]])
    masks = make_masks(((300, 340), (940, 980)))

    out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
    assert len(out[0]) == 2


def test_an_oblique_surface_is_truncated_at_the_depth_tolerance():
    """The cost of the z-buffer, pinned rather than glossed over.

    A surface running away from the camera — a table seen end-on — puts all of its returns
    in one cell at steadily growing range, and the buffer cannot tell that from a mask that
    caught the wall behind it. Everything past `depth_tolerance` goes, so the far end of a
    genuinely deep object is lost along with the bleed.

    That is the deliberate trade: the tolerance is set well above the noise floor to keep
    the loss to objects deeper than half a metre WITHIN A SINGLE 8 px cell, which at
    3 m is a span of ~8 cm across. Sharpen `depth_tolerance` and this test is the one that
    should fail first.
    """
    fusion = CloudImageFusion(platform="mecanum_sim",
                              occlusion=OcclusionConfig(pixel_bin=8, depth_tolerance=0.5))
    # A run of points along +x, each 0.1 m behind the last, sharing one cell.
    cloud = np.array([[2.0 + 0.1 * i, 0.0, CAM_Z] for i in range(12)])
    masks = make_masks(((300, 340), (940, 980)))

    out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
    assert len(out[0]) == 6, "keeps exactly the points within 0.5 m of the nearest"


def test_the_z_buffer_is_per_mask_not_per_frame():
    """Two objects at different depths in different directions must not shadow each other:
    the near one is not between the camera and the far one."""
    fusion = CloudImageFusion(platform="mecanum_sim")
    cloud = np.array([[2.0, 0.0, CAM_Z], [5.0, 5.0, CAM_Z]])
    masks = make_masks(((300, 340), (940, 980)),      # forward, the near point
                       ((300, 340), (700, 740)))      # 45 deg left, the far one

    out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
    assert [len(c) for c in out] == [1, 1]


def test_range_filter_commutes_with_projection():
    """B3 before B4 must select exactly the points B3 after B6 would have kept.

    Legal only because rotation preserves norm, so ||cloud_body|| == ||cloud_world - t_b2w||.
    The reorder exists for speed — generate_seg_cloud runs one pass over the cloud PER MASK —
    and a speed change is only worth having if it is free. This is what makes "free" checkable."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(20260810)
    fusion = CloudImageFusion(platform="mecanum_sim")
    threshold = 6.0

    for trial in range(25):
        n = int(rng.integers(200, 2000))
        cloud_world = rng.uniform(-12, 12, size=(n, 3))
        R_b2w = Rotation.random(random_state=trial).as_matrix()
        t_b2w = rng.uniform(-5, 5, size=3)
        R_w2b = R_b2w.T
        cloud_body = cloud_world @ R_w2b.T + (-R_w2b @ t_b2w)

        masks = [rng.random((H, W)) < 0.3 for _ in range(int(rng.integers(1, 5)))]

        after = [c[np.linalg.norm(c[:, :3] - t_b2w, axis=1) < threshold]
                 for c in fusion.generate_seg_cloud(cloud_body, masks, R_b2w, t_b2w)]

        in_range = np.linalg.norm(cloud_body[:, :3], axis=1) < threshold
        before = fusion.generate_seg_cloud(cloud_body[in_range], masks, R_b2w, t_b2w)

        assert len(after) == len(before)
        for old, new in zip(after, before):
            assert old.shape == new.shape
            np.testing.assert_allclose(old, new, atol=1e-9)


def test_bounds_mode_reject_drops_out_of_fov_points():
    """B5 `reject` is the fix for the defect the xfail above documents.

    Same scene as test_out_of_fov_points_must_not_be_claimed_by_an_edge_mask: one point 85
    deg up (outside the +-60 deg band) and one on the horizon, under a mask spanning the top
    edge. Under `clip` the out-of-FOV point is pinned to row 0 and swallowed; under `reject`
    the projection leaves it out of range and generate_seg_cloud's in_bounds guard — which is
    unreachable while clipping — finally does its job.
    """
    cloud = np.array([at_elevation(85), [5.0, 0.0, CAM_Z]])
    masks = make_masks(((0, 340), (940, 980)))

    clipped = CloudImageFusion(platform="mecanum_sim", bounds_mode="clip")
    rejected = CloudImageFusion(platform="mecanum_sim", bounds_mode="reject")

    assert len(clipped.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))[0]) == 2
    assert len(rejected.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))[0]) == 1


def test_bounds_mode_reject_keeps_everything_inside_the_fov():
    """The fix must not cost in-FOV points — a reject mode that trims the valid band would
    look like an improvement in bleed and a regression everywhere else.

    Occlusion is off here because this cloud is a random shell, not a scene: two points
    that happen to share a cell at different ranges are legitimately culled by the
    z-buffer, and counting that as a bounds-mode loss would measure the wrong thing.
    """
    rng = np.random.default_rng(7)
    elevations = rng.uniform(-55, 55, size=400)
    azimuths = rng.uniform(-179, 179, size=400)
    r = rng.uniform(0.5, 8.0, size=400)
    cloud = np.stack([
        r * np.cos(np.radians(elevations)) * np.cos(np.radians(azimuths)),
        r * np.cos(np.radians(elevations)) * np.sin(np.radians(azimuths)),
        r * np.sin(np.radians(elevations)) + CAM_Z,
    ], axis=1)
    masks = [np.ones((H, W), dtype=bool)]

    for mode in ("clip", "reject"):
        fusion = CloudImageFusion(platform="mecanum_sim", bounds_mode=mode,
                                  occlusion=OcclusionConfig(enabled=False))
        out = fusion.generate_seg_cloud(cloud, masks, np.eye(3), np.zeros(3))
        assert len(out[0]) == len(cloud), f"{mode} dropped an in-FOV point"


def test_invalid_bounds_mode_is_rejected():
    with pytest.raises(ValueError, match="bounds_mode"):
        CloudImageFusion(platform="mecanum_sim", bounds_mode="clamp")
