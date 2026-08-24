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

def test_a_later_reading_overwrites_an_earlier_one():
    """Last write wins, because the robot revisits ground from closer up and terrain analysis
    estimates ground height from denser support the nearer it is."""
    area = TraversableArea(cell_m=0.1, half_span_m=5.0)
    area.add(np.array([[1.0, 1.0]]), np.array([True]))
    assert area.counts()["obstacle"] == 1
    area.add(np.array([[1.0, 1.0]]), np.array([False]))
    assert area.counts() == {**area.counts(), "obstacle": 0, "free": 1}


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
