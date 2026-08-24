"""Unit tests for 3D box fitting and the dimension priors.

Covers defects 6 (hard min/max extents) and 7 (DIMENSION_PRIORS z-cap, and extents
compared in arbitrary hull-edge order). The xfail(strict=True) cases encode the behaviour
the design calls for; they turn green when Tier 1.5 / Tier 2.2 land.

Pure numpy/scipy — no GPU, no ROS. Run with:
    python -m pytest ai_module/src/sam_mapper/tests/test_single_object.py
"""
import numpy as np
import pytest

# single_object imports open3d at module scope for its clustering paths. It ships in the
# container but not on the host, so a host-side `pytest` skips this file rather than
# erroring out during collection. Run these for real with `just test sam_mapper`.
pytest.importorskip("open3d", reason="run in the container: just test sam_mapper")

from sam_mapper.single_object import (  # noqa: E402
    DIMENSION_PRIORS, _fits_prior, get_bbox_3d_oriented, get_box_3d,
    minimum_bounding_rectangle)


def box_surface(center, extent, yaw=0.0, n=12):
    """Points on all six faces of a yaw-rotated box — a fully-observed object."""
    c, e = np.asarray(center, float), np.asarray(extent, float)
    g = np.linspace(-0.5, 0.5, n)
    pts = []
    for a in g:
        for b in g:
            for face in ([0.5, a, b], [-0.5, a, b], [a, 0.5, b],
                         [a, -0.5, b], [a, b, 0.5], [a, b, -0.5]):
                pts.append(np.array(face) * e)
    pts = np.array(pts)
    ca, sa = np.cos(yaw), np.sin(yaw)
    rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    return pts @ rot.T + c


def two_visible_faces(center, extent, n=14):
    """Only the +x and +y faces — what a ground robot actually sees of a cabinet."""
    c, e = np.asarray(center, float), np.asarray(extent, float)
    g = np.linspace(-0.5, 0.5, n)
    pts = [np.array([0.5, a, b]) * e for a in g for b in g]
    pts += [np.array([a, 0.5, b]) * e for a in g for b in g]
    return np.array(pts) + c


# -- get_box_3d --------------------------------------------------------------

def test_axis_aligned_box_recovers_centre_and_extent():
    pts = box_surface([1.0, -2.0, 0.5], [0.8, 0.4, 1.2])
    center, extent, quat = get_box_3d(pts)
    np.testing.assert_allclose(center, [1.0, -2.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(extent, [0.8, 0.4, 1.2], atol=1e-9)
    assert list(quat) == [0.0, 0.0, 0.0, 1.0]


def test_single_stray_point_sets_the_whole_extent():
    """Defect 6, KNOWN and deliberately not fixed. Percentile trimming has now lost TWICE.

    Attempt 1 — weighted percentile, rejected because boxes were UNDER-sized at the time
    (10/10 vs matched GT, median volume ratio 0.106), so trimming moved them further from
    the truth. That note ended "revisit once boxes err large".

    Attempt 2 — boxes now DO err large (median horizontal extent ratio 1.22x, p75 1.66x,
    absolute error median +0.062 m / p75 +0.207 m, while centroids and z extent are already
    good). So the revisit condition was met and an unweighted per-axis quantile was
    implemented and benched over 13 scenes at trim=0.02:

        best_iou_per_gt  0.317 -> 0.321   (+0.004, against a +0.02 bar)
        TP @ IoU 0.25      178 -> 175     while n_pred went 431 -> 433
        recall_askable   0.547 -> 0.533
        cat2 oracle      0.683 -> 0.675

    It published MORE objects and matched FEWER: the trim shrank boxes below the matching
    threshold. It cannot tell bleed from a real edge, because a quantile is a statistic over
    the object's own points — so it taxes the ~half of boxes already within 6 cm of GT to win
    a little on the rest, and the two cancel.

    The lesson, and the reason the code is gone rather than defaulted off: a filter gated on
    an object's OWN statistics bills every object. One gated on independent evidence bills
    only where the evidence is — which is why claimed_volume (D2b) works where this does not.
    A third variant needs a different theory of where the bleed enters, not another quantile.
    """
    pts = box_surface([0, 0, 0], [1.0, 1.0, 1.0])
    _, clean, _ = get_box_3d(pts)
    _, polluted, _ = get_box_3d(np.vstack([pts, [[5.0, 0.0, 0.0]]]))
    assert clean[0] == pytest.approx(1.0)
    assert polluted[0] == pytest.approx(5.5)


def test_oriented_box_recovers_a_rotated_footprint():
    pts = box_surface([0.0, 0.0, 0.0], [2.0, 0.6, 1.0], yaw=0.4)
    center, extent, _ = get_bbox_3d_oriented(pts)
    np.testing.assert_allclose(center[:2], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(sorted(extent[:2]), [0.6, 2.0], atol=1e-6)
    assert extent[2] == pytest.approx(1.0, abs=1e-6)


def test_oriented_box_extent_order_is_not_semantic():
    """Defect 7b. get_bbox_3d_oriented returns [edge1, edge2, z] in whatever order the
    hull produced, NOT (length, width). regularize_shape compares those components
    against anisotropic priors like table (5,3,2) positionally, so acceptance depends on
    hull ordering. Any prior comparison must sort first."""
    a = get_bbox_3d_oriented(box_surface([0, 0, 0], [2.0, 0.6, 1.0], yaw=0.0))[1]
    b = get_bbox_3d_oriented(box_surface([0, 0, 0], [2.0, 0.6, 1.0], yaw=np.pi / 2))[1]

    # Same physical box, rotated a quarter turn. Sorted, the extents agree exactly...
    np.testing.assert_allclose(sorted(a[:2]), sorted(b[:2]), atol=1e-6)
    # ...but the component ORDER flips, and that raw order is what the prior check reads.
    assert (a[0] > a[1]) and (b[0] < b[1]), f"expected the order to flip: {a[:2]} vs {b[:2]}"


def test_oriented_box_falls_back_to_axis_aligned_when_the_hull_degenerates():
    """A degenerate XY projection is NOT an error — it is the normal shape of a thin flat
    object. A window is 4 cm thick, one voxel at 0.05 m, so its footprint is a line and
    ConvexHull cannot run. Measured on arabic_room: 2 of 4 windows were discarded outright
    for this (`hull_failed=1`, `regul=0`), and a third lost 2 of its 3 clusters.

    A set with no convex hull has no meaningful orientation, so the axis-aligned box at
    yaw 0 is the correct degenerate answer — and crucially it is an ANSWER, not a None
    that silently deletes the object.
    """
    collinear = np.array([[float(i) * 0.1, 0.0, 0.0] for i in range(10)])
    center, extent, _ = get_bbox_3d_oriented(collinear)
    assert extent is not None, "a degenerate footprint must still yield a box"
    assert extent[0] == pytest.approx(0.9)
    assert center[0] == pytest.approx(0.45)

    # The underlying hull routine still reports failure; the fallback lives above it.
    assert minimum_bounding_rectangle(collinear[:, :2]) == (None, None)
    # Genuinely empty input has no answer and must still say so.
    assert get_bbox_3d_oriented(np.zeros((0, 3))) == (None, None, None)


def test_voxel_cell_size_floors_the_extent_without_inflating_it():
    """A voxel is a CELL, not a point, so min/max over voxel CENTRES collapses a
    one-voxel-thick object to exactly zero — which is how arabic_room's windows published
    as `[0.0, 0.52, 0.78]`: a zero-volume box scoring IoU 0 despite a 3 cm accurate
    centroid. `voxel_size` fixes that as a FLOOR.

    It used to be added to every axis instead, which fixed the degenerate case by
    inflating every other box. At 0.05 that is invisible on a sofa and decisive on the
    small objects category 2 asks about: a 0.20 x 0.15 x 0.03 book went out 4.4x its true
    volume, capping its IoU at 0.23 against a 0.25 threshold.
    """
    single_cell = np.zeros((1, 3))
    _, extent, _ = get_box_3d(single_cell, voxel_size=0.05)
    np.testing.assert_allclose(extent, [0.05, 0.05, 0.05])

    pts = box_surface([0, 0, 0], [1.0, 1.0, 1.0])
    _, plain, _ = get_box_3d(pts)
    _, floored, _ = get_box_3d(pts, voxel_size=0.05)
    np.testing.assert_allclose(floored, plain)

    # Only the axis below the floor moves, and only up to it.
    thin = box_surface([0, 0, 0], [0.40, 0.30, 0.0])
    _, extent, _ = get_box_3d(thin, voxel_size=0.05)
    np.testing.assert_allclose(extent, [0.40, 0.30, 0.05])


def test_the_oriented_box_floors_its_extent_the_same_way():
    # Both fitters feed the same obj_map.json, so a difference between them would make a
    # box's size depend on whether publish_oriented_box happened to be on.
    pts = box_surface([0, 0, 0], [1.2, 0.8, 1.0], yaw=0.3)
    _, plain, _ = get_bbox_3d_oriented(pts)
    _, floored, _ = get_bbox_3d_oriented(pts, voxel_size=0.05)

    np.testing.assert_allclose(floored, plain, atol=1e-9)


def test_min_area_rect_picks_the_wrong_yaw_on_a_partially_observed_box():
    """Why the design prefers the L-shape VARIANCE criterion over minimum AREA.

    Seeing only the +x and +y faces of a 1.2 x 0.8 cabinet, the minimum-area rectangle
    settles on a rotated fit of 1.44 x 0.67. Note what it gets right and wrong: the area
    is preserved to 0.01% (0.9599 vs 0.9600) — it is doing exactly what it optimises —
    while the individual dimensions are off by +20% and -17% because the yaw is wrong.
    An area-minimising objective cannot distinguish these, which is the whole argument
    for scoring yaw candidates by point-to-edge variance instead.
    """
    true_extent = np.array([1.2, 0.8])
    pts = two_visible_faces([0, 0, 0], [*true_extent, 1.0])
    _, extent, _ = get_bbox_3d_oriented(pts)
    fitted = np.sort(extent[:2])[::-1]

    rel = np.abs(fitted - true_extent) / true_extent
    assert rel.max() > 0.15, f"expected a materially wrong fit, got {fitted}"
    area_err = abs(fitted.prod() - true_extent.prod()) / true_extent.prod()
    assert area_err < 0.05, f"area should be nearly preserved, got {fitted.prod():.4f}"


# -- dimension priors --------------------------------------------------------

def test_priors_are_caps_above_the_objects_they_bound():
    """Caps must sit above the objects they bound. Two hand-set ones did not:
    sofa (3,3,2) rejected a 3.06 m sofa; default z 2.0 truncated 2.78 m columns.

    Fixtures are each class's MEDIAN real instance. The rule is a PER-AXIS p90, which does not
    promise any particular whole object fits — see test_priors_deliberately_exclude_the_extremes."""
    # Median real instance per class, from the VLA-3D ground truth the table is derived
    # from. The three marked (*) are cases the OLD hand-set caps rejected.
    real_sizes = {
        "sofa": (2.19, 0.78, 0.68),
        "column": (0.49, 0.51, 2.46),        # (*) old default z was exactly 3.0
        "pottedplant": (0.96, 0.34, 1.49),   # (*) old cap was (1.0, 1.0, 2.0)
        "window": (1.00, 0.06, 0.82),        # (*) old default was 5 m in XY
        "table": (1.38, 0.73, 0.49),
        "couch": (2.11, 0.99, 1.09),         # was silently on `default` while sofa was not
        "pillow": (0.85, 0.51, 0.13),
        "stool": (0.39, 0.39, 0.87),
    }
    for label, size in real_sizes.items():
        prior = DIMENSION_PRIORS.get(label, DIMENSION_PRIORS["default"])
        assert _fits_prior(size, prior), f"{label} prior {prior} rejects a real {size}"


def test_priors_cover_the_asked_about_classes():
    """INVERTED from the version that documented the gap.

    There used to be 5 hand-set caps, so every class the questions actually ask about —
    pillow, vase, picture, lantern, bowl, cup — fell through to one global default. The
    table is now derived from VLA-3D ground truth (263 classes, 15 scenes), so each of
    these has a cap fitted to its own class rather than to furniture in general."""
    asked = {"pillow", "vase", "picture", "lantern", "bowl", "cup", "book",
             "column", "door", "curtain", "window", "stool", "tv"}
    missing = asked - set(DIMENSION_PRIORS)
    assert not missing, f"no derived cap for {sorted(missing)} — regenerate the table"


def test_tall_objects_are_not_capped_below_their_real_height():
    """Columns, doors and curtains routinely exceed 2 m. The cap is still a CONSTANT,
    which remains the wrong model — it should be the per-scene ceiling height measured
    from the cloud — but it must at least clear the objects that exist."""
    tall = {"column", "door", "curtain", "bookcase"}
    assert all(DIMENSION_PRIORS.get(c, DIMENSION_PRIORS["default"])[2] >= 2.8 for c in tall)


def test_priors_are_derived_not_hand_set():
    """The six inherited constants were never connected to data and three of them rejected
    real objects. Guard the property that replaced them: broad class coverage, and a
    generous `default` for the classes an unseen challenge scene will inevitably introduce.
    """
    assert len(DIMENSION_PRIORS) > 200, "table looks hand-set, not generated"
    assert DIMENSION_PRIORS["default"][2] >= 3.0, "default z must clear a normal ceiling"
    # fireextinguisher occurred in NO scene; it was dead config carried over from SysNav.
    assert "fireextinguisher" not in DIMENSION_PRIORS


def test_prior_overrides_merge_rather_than_replace():
    """A yaml block naming one class must override that class and leave the other 260
    derived caps in place. Replacing wholesale would mean any override silently reverted
    every other class to `default`, which is the sort of thing that shows up as an
    unexplained regression three experiments later."""
    from sam_mapper.mapping_config import MappingConfig

    base = MappingConfig().dimension_priors
    tweaked = MappingConfig.from_dict(
        {"dimension_priors": {"priors": {"chair": [1.5, 1.5, 2.0]}}}).dimension_priors

    assert tweaked.for_label("chair") == (1.5, 1.5, 2.0)
    assert tweaked.for_label("sofa") == base.for_label("sofa")
    assert len(tweaked.priors) == len(base.priors)


def test_priors_deliberately_exclude_the_extremes():
    """The other half of the p90 bargain, asserted so nobody "fixes" it later.

    Five 2.93 m bench-style seats labelled `chair` would set the chair cap to 3.66 m on max —
    wide enough to admit the merge blobs D3 exists to reject. p90 excludes them, at a
    deliberate ~3% loss of real objects. If this fails, the generator went back to max."""
    chair = DIMENSION_PRIORS["chair"]
    assert chair[0] < 2.0, f"chair cap {chair} is wide enough to admit a merged blob"
    assert not _fits_prior((2.93, 0.67, 0.55), chair), "the bench outlier must not fit"
