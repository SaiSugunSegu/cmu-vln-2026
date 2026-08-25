"""The floor grid the cat-3 snap aims at.

Pure numpy/scipy -- no ROS, no GPU. Run with `just test smart_vlm`, or on the host with
`python3 -m pytest ai_module/src/smart_vlm/tests/test_traversable_area.py -q`.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_vlm.traversable_area import TraversableArea


def floor(area, x=(-3.0, 3.0), y=(-3.0, 3.0), step=0.02):
    """Mark a rectangle of open floor."""
    gx, gy = np.meshgrid(np.arange(x[0], x[1], step), np.arange(y[0], y[1], step))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    area.add(xy, np.zeros(len(xy), dtype=bool))
    return area


def slab(area, x, y, step=0.02):
    """Mark a rectangle of obstacle, as a piece of furniture would read."""
    gx, gy = np.meshgrid(np.arange(x[0], x[1], step), np.arange(y[0], y[1], step))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    area.add(xy, np.ones(len(xy), dtype=bool))
    return area


# -- construction ----------------------------------------------------------

def test_a_cell_nobody_has_looked_at_is_not_traversable():
    """UNKNOWN is a third state on purpose. Treating unobserved ground as free is how a
    waypoint gets aimed into furniture the terrain map simply has no reading for."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    assert area.nearest_free((0.0, 0.0)) is None


def test_a_bad_cell_size_is_refused():
    with pytest.raises(ValueError):
        TraversableArea(cell_m=0.0)
    with pytest.raises(ValueError):
        TraversableArea(half_span_m=-1.0)


def test_the_flags_must_match_the_points():
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    with pytest.raises(ValueError):
        area.add(np.zeros((4, 2)), np.zeros(3, dtype=bool))


def test_an_empty_reading_is_ignored():
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    area.add(np.empty((0, 2)), np.empty(0, dtype=bool))
    assert area.counts()["free"] == 0


# -- writing ---------------------------------------------------------------

def test_one_contrary_reading_does_not_erase_established_floor():
    """The failure this class exists to prevent.

    Terrain decays after 4 s and cuts a noisy height at a hard 0.05 m, so a single grazing
    return calling known floor an obstacle is routine -- measured at 464 FREE -> OBSTACLE flips
    between two consecutive question snapshots. `nearest_free` reads the grid at one instant,
    and on hotel_room_1 Q04 that instant cost a 2.83 m snap when there was floor 0.18 m away.
    """
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    cell = np.array([[1.0, 1.0]])
    for _ in range(20):
        area.add(cell, np.array([False]))
    area.add(cell, np.array([True]))
    assert area.counts()["free"] == 1, "one bad reading erased twenty good ones"
    assert area.nearest_free((1.0, 1.0)) == pytest.approx((1.05, 1.05), abs=0.1)


def test_ground_that_is_really_blocked_still_corrects_itself():
    """The other half: sticky must not mean frozen, or a first bad reading is permanent."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    cell = np.array([[1.0, 1.0]])
    for _ in range(20):
        area.add(cell, np.array([False]))
    for _ in range(TraversableArea.CLAMP * 2):
        area.add(cell, np.array([True]))
    assert area.counts()["obstacle"] == 1


def test_a_cell_votes_once_per_message_however_many_points_land_in_it():
    """Terrain density falls off with range, so per-point voting would let the same floor at
    3 m be outvoted by itself at 0.5 m -- a fact about the lidar, not about the floor."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    area.add(np.repeat(np.array([[1.0, 1.0]]), 500, axis=0), np.zeros(500, dtype=bool))
    area.add(np.array([[1.0, 1.0]]), np.array([True]))
    assert area.counts()["obstacle"] == 1, "500 points in one message outvoted a later message"


def test_free_and_blocked_in_the_same_message_reads_as_blocked():
    """Those free points are the floor AROUND the thing, sharing a 10 cm cell with it."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    area.add(np.array([[1.0, 1.0]] * 4), np.array([False, False, False, True]))
    assert area.counts()["obstacle"] == 1


def test_a_trusted_source_outweighs_a_doubtful_one():
    """/terrain_map is half the voxel size and does not decay inside 1.75 m; ext is a
    four-second rolling window. One message of the first is worth more than one of the second."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    cell = np.array([[1.0, 1.0]])
    area.add(cell, np.array([False]), weight=2)
    area.add(cell, np.array([True]), weight=1)
    assert area.counts()["free"] == 1


def test_the_nearest_free_cell_is_nearest_to_the_point_not_to_its_cell():
    """The transform works in cell indices, so on its own it answers for the query's cell
    centre -- measured 0.22 m where the true nearest was 0.18 m. Every centimetre given away
    here is a centimetre of the model's waypoint we did not have to give away."""
    area = TraversableArea(cell_m=0.5, half_span_m=5.0)
    area.add(np.array([[1.4, 0.1], [-1.4, 0.1]]), np.array([False, False]))
    query = (0.9, 0.1)
    got = area.nearest_free(query)
    free = np.argwhere(area.state == TraversableArea.FREE)
    world = area._corner + (free + 0.5) * area.cell_m
    best = np.linalg.norm(world - np.array(query), axis=1).min()
    assert math.dist(query, got) == pytest.approx(best, abs=1e-9)


def test_the_grid_grows_to_hold_a_reading_outside_it():
    """A scene bigger than the initial span must not silently lose its far half."""
    area = TraversableArea(cell_m=0.1, half_span_m=2.0)
    before = area.state.shape
    area.add(np.array([[9.0, 9.0]]), np.array([False]))
    assert area.state.shape > before
    assert area.nearest_free((9.0, 9.0)) == pytest.approx((9.05, 9.05), abs=0.1)


def test_growing_keeps_what_was_already_marked():
    area = TraversableArea(cell_m=0.1, half_span_m=2.0)
    area.add(np.array([[0.0, 0.0]]), np.array([False]))
    area.add(np.array([[9.0, 9.0]]), np.array([False]))
    assert area.counts()["free"] == 2
    assert area.nearest_free((0.0, 0.0)) == pytest.approx((0.05, 0.05), abs=0.1)


# -- the query -------------------------------------------------------------

def test_the_nearest_free_cell_is_the_one_returned():
    area = floor(TraversableArea(cell_m=0.05, half_span_m=5.0))
    got = area.nearest_free((1.0, 1.0))
    assert math.dist(got, (1.0, 1.0)) <= 0.05      # within one cell of the asked-for point


def test_a_waypoint_inside_furniture_lands_just_outside_it():
    """The case this exists for: the model aims at an object CENTRE, which is inside the object.
    livingroom_1's sofa is 1.88 m across, so its centre is ~0.94 m from open floor."""
    area = floor(TraversableArea(cell_m=0.05, half_span_m=6.0), x=(-4.0, 4.0), y=(-4.0, 4.0))
    slab(area, x=(-1.0, 1.0), y=(-0.5, 0.5))       # a 2.0 x 1.0 m piece of furniture

    got = area.nearest_free((0.0, 0.0))            # dead centre of it
    assert got is not None
    # Out of the slab, and out by about its half-depth rather than its half-width.
    assert abs(got[1]) > 0.5 or abs(got[0]) > 1.0
    assert math.dist(got, (0.0, 0.0)) == pytest.approx(0.5, abs=0.1)


def test_a_waypoint_outside_the_grid_has_no_answer():
    """None rather than a clamped guess: the caller keeps the model's own coordinate."""
    area = floor(TraversableArea(cell_m=0.1, half_span_m=3.0), x=(-1.0, 1.0), y=(-1.0, 1.0))
    assert area.nearest_free((50.0, 50.0)) is None


def test_clearance_pushes_the_answer_off_the_obstacle():
    """Off by default, but when dialled up it must actually erode -- that is what buys the base
    autonomy passing our point through instead of re-aiming it."""
    area = floor(TraversableArea(cell_m=0.05, half_span_m=6.0), x=(-4.0, 4.0), y=(-4.0, 4.0))
    slab(area, x=(-1.0, 1.0), y=(-0.5, 0.5))

    near = area.nearest_free((0.0, 0.0), clearance_m=0.0)
    far = area.nearest_free((0.0, 0.0), clearance_m=0.75)
    assert math.dist(far, (0.0, 0.0)) > math.dist(near, (0.0, 0.0))
    assert abs(far[1]) >= 0.5 + 0.75 - 0.1 or abs(far[0]) >= 1.0 + 0.75 - 0.1


def test_clearance_that_erodes_everything_has_no_answer():
    area = floor(TraversableArea(cell_m=0.05, half_span_m=3.0), x=(-1.0, 1.0), y=(-1.0, 1.0))
    slab(area, x=(-0.2, 0.2), y=(-0.2, 0.2))
    assert area.nearest_free((0.0, 0.0), clearance_m=50.0) is None


def test_the_answer_tracks_new_readings():
    """The cached transform must be invalidated by a write, or the snap would keep aiming at
    the floor as it looked when the robot first arrived."""
    area = TraversableArea(cell_m=0.05, half_span_m=5.0)
    area.add(np.array([[2.0, 0.0]]), np.array([False]))
    assert area.nearest_free((0.0, 0.0)) == pytest.approx((2.025, 0.025), abs=0.05)

    area.add(np.array([[0.5, 0.0]]), np.array([False]))
    assert area.nearest_free((0.0, 0.0)) == pytest.approx((0.525, 0.025), abs=0.05)


def test_a_repeated_query_is_answered_from_the_cache():
    """Same answer twice, and the second one must not need a rebuild -- driving asks about once
    per metre, and rebuilding the transform each time would put it back in the drive loop."""
    area = floor(TraversableArea(cell_m=0.05, half_span_m=5.0))
    first = area.nearest_free((1.0, 1.0))
    assert not area._dirty
    assert area.nearest_free((1.0, 1.0)) == first


# -- the range gate ---------------------------------------------------------

def test_a_reading_beyond_the_sensor_range_never_reaches_the_grid():
    """Bad floor is worse than no floor: `nearest_free` will happily snap a waypoint onto it.

    /terrain_map_ext is 20 m wide by construction, so a point past that is not distant floor,
    it is bad data. Measured with no gate: livingroom_4 accumulated 198,280 free cells --
    1,983 m2 -- against a room whose real traversable area is 24.7 m2, and loft 158,370, both
    from runs where the robot ended up outside the building.
    """
    area = TraversableArea(cell_m=0.1, half_span_m=20.0)
    xy = np.array([[1.0, 0.0], [2.0, 0.0], [30.0, 0.0]])
    area.add(xy, np.zeros(len(xy), dtype=bool), origin=(0.0, 0.0), max_range_m=21.0)
    assert area.counts()["free"] == 2
    assert area.nearest_free((30.0, 0.0)) != (30.0, 0.0)


def test_the_gate_is_off_without_a_pose():
    """Callers that have no odometry yet must still be able to accumulate."""
    area = TraversableArea(cell_m=0.1, half_span_m=20.0)
    xy = np.array([[1.0, 0.0], [30.0, 0.0]])
    area.add(xy, np.zeros(len(xy), dtype=bool))
    assert area.counts()["free"] == 2


def test_the_gate_follows_the_robot():
    """The range is from the SENSOR, not the origin — a robot 10 m out still sees its own
    surroundings, and gating on world distance would blank them."""
    area = TraversableArea(cell_m=0.1, half_span_m=20.0)
    xy = np.array([[10.5, 0.0]])
    area.add(xy, np.zeros(1, dtype=bool), origin=(10.0, 0.0), max_range_m=1.0)
    assert area.counts()["free"] == 1


# -- what a run leaves behind -----------------------------------------------

def test_bounds_describe_what_was_seen_not_the_grid():
    """A grid that grew to 40 m because of one stray reading is indistinguishable from a good
    one by shape alone. `bounds` is what lets a run be checked against its own room."""
    area = TraversableArea(cell_m=0.1, half_span_m=20.0)
    area.add(np.array([[1.0, 2.0], [1.5, 2.5]]), np.zeros(2, dtype=bool))
    lo_x, lo_y, hi_x, hi_y = area.counts()["bounds"]
    assert 0.9 <= lo_x <= 1.1 and 1.9 <= lo_y <= 2.1
    assert 1.5 <= hi_x <= 1.7 and 2.5 <= hi_y <= 2.7
    assert area.counts()["shape"] == [400, 400]      # the grid itself is far larger


def test_bounds_is_none_before_anything_is_seen():
    assert TraversableArea(cell_m=0.1, half_span_m=5.0).counts()["bounds"] is None


def test_the_snapshot_can_redraw_the_grid():
    """The overlay reads this. Only `free_cells` used to survive a question, and a count
    cannot show that the floor beside the target was never observed -- which is exactly the
    failure it kept hiding."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    area.add(np.array([[1.0, 1.0], [1.2, 1.0]]), np.array([False, True]))
    snap = area.snapshot()
    assert set(snap) == {"state", "score", "cell_m", "corner"}
    assert snap["state"].dtype == np.uint8
    # place a FREE cell back in world coordinates from the snapshot alone
    free = np.argwhere(snap["state"] == TraversableArea.FREE)
    assert len(free) == 1
    wx = snap["corner"][0] + (free[0][0] + 0.5) * snap["cell_m"]
    wy = snap["corner"][1] + (free[0][1] + 0.5) * snap["cell_m"]
    assert math.dist((wx, wy), (1.0, 1.0)) < 0.1
