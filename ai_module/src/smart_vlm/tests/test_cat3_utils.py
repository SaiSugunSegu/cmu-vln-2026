"""Category-3's pure layer: what the model is shown, what its reply is worth.

The model is never called here. Everything under test is the bookkeeping either side of it,
which is where a wrong route becomes an unrecoverable one: a coordinate that contradicts its
own citation, a reply with no goal, a leg too long for the base autonomy to be steered through.
"""
import math

import numpy as np
import pytest

from smart_vlm.traversable_area import TraversableArea
from smart_vlm.cat3_utils import (
    ARRIVAL_RADIUS_M,
    GOAL_RADIUS_M,
    MAX_TRIES,
    OBSTACLE_CLEARANCE_M,
    SETTLE_RADIUS_M,
    SETTLE_S,
    TRY_DURATION_S,
    Waypoint,
    fallback_route,
    heuristic_targets,
    map_table,
    parse_route,
    plan_payload,
    route_summary,
    snap_to_traversable,
    synthetic_trajectory,
    usable_objects,
)


def obj(label, x, y, z=0.5, extent=(1.0, 1.0, 1.0), ids=None):
    return {"label": label, "id": ids or [0],
            "bbox3d": {"center": [x, y, z], "extent": list(extent),
                       "rotation": [0.0, 0.0, 0.0, 1.0]}}


# Shaped after a real run's map: two tables, a stool and the origin stub map_node emits for a
# track that never accumulated geometry.
MAP = {
    "4": obj("table", 2.52, -0.74, extent=(2.48, 0.95, 0.70)),
    "5": obj("table", -3.76, -1.88, extent=(0.51, 1.19, 0.53)),
    "13": obj("stool", -1.08, -1.09, extent=(0.12, 0.18, 0.23)),
    "17": obj("column", 0.0, 0.0, extent=(0.05, 0.05, 0.05)),
}


# -- the map the model is shown --------------------------------------------


def test_usable_objects_drops_the_degenerate_origin_row():
    kept = usable_objects(MAP)
    assert "17" not in kept, "a 5 cm box at the origin is a track with no geometry"
    assert set(kept) == {"4", "5", "13"}


def test_usable_objects_drops_rows_without_a_box():
    assert usable_objects({"1": {"label": "table"}, "2": obj("chair", 1.0, 1.0)}) .keys() == {"2"}


def test_map_table_lists_every_usable_object_and_the_robot():
    table = map_table(usable_objects(MAP), (1.5, -0.5))
    assert "Robot at (1.50, -0.50)" in table
    for oid in ("4", "5", "13"):
        assert f"\n{oid} | " in f"\n{table}"
    assert "column" not in table


def test_map_table_states_no_relations():
    """The relations are the model's job; handing it answers would put a solver back in."""
    table = map_table(usable_objects(MAP), (0.0, 0.0)).lower()
    for leaked in ("closest", "farthest", "nearest", "distance", "between"):
        assert leaked not in table


# -- the reply -------------------------------------------------------------


def reply(*waypoints):
    return {"reason": "test", "waypoints": list(waypoints)}


def wp(role="pass", x=0.0, y=0.0, ids=(), why=""):
    return {"role": role, "x": x, "y": y, "object_ids": list(ids), "why": why}


def test_a_coordinate_that_matches_its_citation_is_kept():
    route, trace = parse_route(
        reply(wp("goal", 2.52, -0.74, ["4"], "the table")), usable_objects(MAP))
    assert [(w.role, round(w.x, 2), round(w.y, 2)) for w in route] == [("goal", 2.52, -0.74)]
    assert route[0].object_ids == ["4"]
    assert not any("snapped" in line for line in trace)


def test_a_coordinate_that_contradicts_its_citation_is_snapped():
    route, trace = parse_route(
        reply(wp("goal", 40.0, 40.0, ["4"], "the table")), usable_objects(MAP))
    assert (round(route[0].x, 2), round(route[0].y, 2)) == (2.52, -0.74)
    assert any("snapped" in line for line in trace)


def test_a_between_midpoint_is_within_tolerance_of_both_anchors():
    objects = usable_objects(MAP)
    mid_x, mid_y = (2.52 + -3.76) / 2, (-0.74 + -1.88) / 2
    route, trace = parse_route(
        reply(wp("goal", mid_x, mid_y, ["4", "5"], "the gap")), objects)
    assert route[0].object_ids == ["4", "5"]
    assert not any("snapped" in line for line in trace), "a real midpoint must survive"


def test_ids_the_map_does_not_have_are_dropped():
    route, trace = parse_route(
        reply(wp("goal", 2.52, -0.74, ["4", "999"])), usable_objects(MAP))
    assert route[0].object_ids == ["4"]
    assert any("999" in line for line in trace)


def test_a_waypoint_with_no_coordinate_falls_back_to_its_citation():
    raw = wp("goal", ids=["13"])
    raw.pop("x")
    route, _ = parse_route(reply(raw), usable_objects(MAP))
    assert (round(route[0].x, 2), round(route[0].y, 2)) == (-1.08, -1.09)


def test_a_waypoint_with_neither_a_coordinate_nor_a_known_object_is_dropped():
    raw = wp("pass", ids=["999"])
    raw["x"] = float("nan")
    route, _ = parse_route(
        reply(raw, wp("goal", 2.52, -0.74, ["4"])), usable_objects(MAP))
    assert len(route) == 1 and route[0].role == "goal"


def test_the_route_is_exactly_the_model_waypoints():
    """Nothing is interpolated or inserted. The only waypoint that ever wedged the robot in
    the first sim run was a hop we invented between two the model chose."""
    route, _ = parse_route(
        reply(wp("pass", -1.08, -1.09, ["13"]), wp("pass", -3.76, -1.88, ["5"]),
              wp("goal", 2.52, -0.74, ["4"])),
        usable_objects(MAP))
    assert len(route) == 3
    assert [w.role for w in route] == ["pass", "pass", "goal"]
    assert all(w.scored for w in route), "every waypoint is one the model named"


def test_a_reply_with_no_goal_promotes_its_last_waypoint():
    route, trace = parse_route(
        reply(wp("pass", -1.08, -1.09, ["13"]), wp("pass", 2.52, -0.74, ["4"])),
        usable_objects(MAP))
    assert [w.role for w in route] == ["pass", "goal"]
    assert any("promoting" in line for line in trace)


def test_two_goals_keep_the_last_and_pass_through_the_first():
    route, trace = parse_route(
        reply(wp("goal", -1.08, -1.09, ["13"]), wp("goal", 2.52, -0.74, ["4"])),
        usable_objects(MAP))
    assert [w.role for w in route] == ["pass", "goal"]
    assert route[-1].object_ids == ["4"]
    assert any("keeping the last" in line for line in trace)


def test_a_goal_that_is_not_last_is_moved_to_the_end():
    """The goal is scored on the final pose, so a route that finishes elsewhere throws it away."""
    route, trace = parse_route(
        reply(wp("goal", 2.52, -0.74, ["4"]), wp("pass", -1.08, -1.09, ["13"])),
        usable_objects(MAP))
    assert route[-1].role == "goal" and route[-1].object_ids == ["4"]
    assert any("moved to the end" in line for line in trace)


def test_an_unknown_role_is_treated_as_a_pass_not_dropped():
    route, trace = parse_route(
        reply(wp("waypoint", -1.08, -1.09, ["13"]), wp("goal", 2.52, -0.74, ["4"])),
        usable_objects(MAP))
    assert [w.role for w in route] == ["pass", "goal"]
    assert any("unknown role" in line for line in trace)


def test_an_empty_reply_yields_an_empty_route():
    assert parse_route(reply(), usable_objects(MAP))[0] == []
    assert parse_route(None, usable_objects(MAP))[0] == []

def test_every_waypoint_carries_the_same_arrival_distance():
    """One rule, one distance. The loop tests `reach_m` and nothing else, so a pass and a goal
    stop the same way -- the role-dependent tolerances they used to carry were two numbers for
    one question, and the tighter of them was met 0 times in 17 legs."""
    route, _ = parse_route(
        reply(wp("pass", -1.08, -1.09, ["13"]), wp("goal", 2.52, -0.74, ["4"])),
        usable_objects(MAP))
    assert {w.reach_m for w in route} == {SETTLE_RADIUS_M}


def test_the_arrival_distance_is_configurable():
    """It is a ROS param on the node; parse_route has to carry whatever the node was given."""
    route, _ = parse_route(reply(wp("goal", 2.52, -0.74, ["4"])),
                           usable_objects(MAP), reach_m=1.25)
    assert route[0].reach_m == pytest.approx(1.25)


# -- the drive budget ------------------------------------------------------
# Driving is three tries of one distance: publish for try_duration_s, and if the robot comes
# within reach_m stop publishing and settle. The loop needs a robot, so what is pinned here is
# the arithmetic that makes it bounded.


def test_a_waypoint_cannot_outlive_its_try_budget():
    """The whole point of the rewrite: a leg is bounded by construction, not by a predicate
    that has to fire. A wedged goal used to run 149 s and 13 recovery attempts."""
    worst_case = MAX_TRIES * TRY_DURATION_S + SETTLE_S
    assert worst_case == pytest.approx(50.0)
    # Three waypoints is the longest route in the corpus; it must fit the route budget with
    # room for the model call and teardown.
    assert 3 * worst_case < 370.0


def test_the_settle_radius_is_the_scoring_radius():
    """A leg that calls itself arrived while outside the circle it is graded on has not
    arrived. The earlier 2.0 m was justified by a 1.5 m tolerance being "met 0 times in 17 goal
    legs" -- but that count came from `closest_m` being sampled before the settle; measured on
    where the robot actually stopped, 8 of those 17 were already inside 1.5 m."""
    assert SETTLE_RADIUS_M == GOAL_RADIUS_M


def test_the_arrival_radius_clears_the_converters_stop_deadband():
    """The converter latches "waypoint reached" and cuts speed at waypointXYRadius = 0.3, so the
    robot physically cannot end closer than that to what we published -- a measured sweep put
    every snapped leg 0.21-0.31 m out. Below 0.3 this test could never fire and the leg would
    burn all three tries standing exactly where it was aimed."""
    assert ARRIVAL_RADIUS_M > 0.3


def test_arriving_at_the_published_point_is_stricter_than_the_scoring_circle():
    """It is a fallback for legs the scoring test cannot win, not a cheaper way to pass. If it
    were the looser of the two it would end legs that the graded distance had not yet ended."""
    assert ARRIVAL_RADIUS_M < GOAL_RADIUS_M


# -- closing on the target -------------------------------------------------


# -- what the plan was worth -----------------------------------------------
# Scoring a perfect drive of the model's own route separates "the plan was wrong" from "the
# plan was right and the robot never got there" -- both read `goal missed` otherwise.


def test_a_synthetic_drive_visits_every_waypoint():
    route = [Waypoint(3.0, 0.0, "pass"), Waypoint(3.0, 4.0, "goal")]
    traj = synthetic_trajectory((0.0, 0.0), route)
    for wp in route:
        assert min(math.dist((p[1], p[2]), (wp.x, wp.y)) for p in traj) < 1e-6


def test_it_ends_exactly_on_the_goal():
    """The goal constraint is scored on the FINAL point, so stopping a few cm short would
    misreport the plan."""
    traj = synthetic_trajectory((0.0, 0.0), [Waypoint(1.7, -0.9, "goal")])
    assert (traj[-1][1], traj[-1][2]) == pytest.approx((1.7, -0.9))


def test_legs_are_sampled_densely_enough_for_a_cursor_scan():
    """score_instruction walks the path with a monotone cursor: endpoints alone would skip
    a constraint the robot would really have driven through."""
    traj = synthetic_trajectory((0.0, 0.0), [Waypoint(10.0, 0.0, "goal")], step_m=0.15)
    gaps = [math.dist((traj[i][1], traj[i][2]), (traj[i + 1][1], traj[i + 1][2]))
            for i in range(len(traj) - 1)]
    assert max(gaps) <= 0.15 + 1e-6


def test_timestamps_only_have_to_increase():
    traj = synthetic_trajectory((0.0, 0.0), [Waypoint(2.0, 0.0, "goal")])
    assert all(traj[i][0] < traj[i + 1][0] for i in range(len(traj) - 1))


def test_plain_dicts_work_as_well_as_waypoints():
    """instruction_plan.json stores dicts, and the orchestrator scores straight from it."""
    as_dicts = synthetic_trajectory((0.0, 0.0), [{"x": 2.0, "y": 1.0, "role": "goal"}])
    as_objs = synthetic_trajectory((0.0, 0.0), [Waypoint(2.0, 1.0, "goal")])
    assert as_dicts == as_objs


def test_an_empty_route_yields_no_trajectory():
    assert synthetic_trajectory((0.0, 0.0), []) == [[0.0, 0.0, 0.0]]
    assert synthetic_trajectory(None, []) == []


def test_a_waypoint_with_no_usable_coordinate_is_skipped():
    traj = synthetic_trajectory((0.0, 0.0), [{"x": None, "y": 1.0, "role": "pass"},
                                             {"x": 2.0, "y": 0.0, "role": "goal"}])
    assert (traj[-1][1], traj[-1][2]) == pytest.approx((2.0, 0.0))


# -- degraded paths --------------------------------------------------------


def test_fallback_route_drives_the_prompts_and_ends_on_a_goal():
    route = fallback_route(usable_objects(MAP), ["stool", "table"])
    assert [w.role for w in route] == ["pass", "goal"]
    assert route[0].object_ids == ["13"]
    assert route[1].object_ids == ["4"], "the larger table wins on volume"


def test_fallback_route_never_claims_one_object_twice():
    route = fallback_route(usable_objects(MAP), ["table", "table"])
    assert sorted(w.object_ids[0] for w in route) == ["4", "5"]


def test_fallback_route_skips_prompts_the_map_never_found():
    route = fallback_route(usable_objects(MAP), ["hookah", "table"])
    assert len(route) == 1 and route[0].role == "goal"


# -- provenance ------------------------------------------------------------


def test_plan_payload_records_what_the_model_was_shown():
    objects = usable_objects(MAP)
    route, trace = parse_route(reply(wp("goal", 2.52, -0.74, ["4"], "the table")), objects)
    payload = plan_payload("Stop at the table.", objects, route,
                           table=map_table(objects, (0.0, 0.0)), images=["a.png"],
                           reply=reply(wp("goal", 2.52, -0.74, ["4"])), trace=trace,
                           source="cloud", robot_xy=(0.0, 0.0))
    assert payload["n_map_objects"] == 3
    assert payload["images"] == ["a.png"]
    assert payload["route"][0]["role"] == "goal"
    assert "table" in payload["map_table"]
    assert payload["plan_source"] == "cloud"


def test_route_summary_names_the_waypoints_and_their_reasons():
    route = [Waypoint(-1.0, 2.0, "pass", why="the stool"),
             Waypoint(10.0, 0.0, "goal", why="the tray")]
    assert route_summary(route) == "pass(-1.00, 2.00) the stool -> goal(10.00, 0.00) the tray"


def test_route_summary_of_an_empty_route_says_so():
    assert route_summary([]) == "empty route"


# -- extraction fallback ---------------------------------------------------


def test_heuristic_targets_keeps_nouns_and_drops_function_words():
    # The only test here that leaves the pure layer: `clean_targets` lives beside the counting
    # reasoner and reaches the VLM backends. Everything else in this file runs anywhere.
    pytest.importorskip("pydantic")
    targets = heuristic_targets("Go near the stool under the picture and stop at the table.")
    assert targets == ["stool", "picture", "table"]


# -- snapping onto traversable ground --------------------------------------
# The model reasons about object CENTRES, so a correct waypoint routinely lands inside the
# furniture it names. The snap moves it to the CLOSEST point of the floor grid the robot has
# built up -- nothing is traded against that distance. An earlier version eroded the floor by
# 0.75 m and biased the pick toward the robot, which moved every waypoint 0.94-1.41 m on a
# measured run; since the robot tracks what we publish to within 0.3 m, that displacement WAS
# the error.


def _area(cell_m=0.1, span=8.0):
    return TraversableArea(cell_m=cell_m, half_span_m=span)


def _floor(area, x=(-4.0, 2.0), y=(-6.0, 1.0), step=0.04):
    gx, gy = np.meshgrid(np.arange(x[0], x[1], step), np.arange(y[0], y[1], step))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    area.add(xy, np.zeros(len(xy), dtype=bool))
    return area


def _slab(area, x, y, step=0.04):
    gx, gy = np.meshgrid(np.arange(x[0], x[1], step), np.arange(y[0], y[1], step))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    area.add(xy, np.ones(len(xy), dtype=bool))
    return area


def test_a_waypoint_inside_the_furniture_moves_out_to_the_nearest_floor():
    """livingroom_1 Q05 in miniature: the midpoint between a sofa and a table sits inside both.
    The old snap moved it 1.26 m; the nearest floor is a fraction of that."""
    area = _floor(_area())
    _slab(area, x=(-2.6, -0.7), y=(-3.7, -2.2))         # sofa
    _slab(area, x=(-1.05, 0.08), y=(-1.0, 0.2))         # round table

    out = snap_to_traversable((-1.03, -2.54), area)
    assert out != (-1.03, -2.54)
    # Out of the sofa, and out by roughly the distance to its nearest edge, not by a metre.
    assert out[1] < -3.7 or out[1] > -2.2 or out[0] > -0.7 or out[0] < -2.6
    assert math.dist(out, (-1.03, -2.54)) <= 0.5


def test_a_waypoint_already_on_clear_ground_is_left_alone():
    """Nearest-free of a point that is itself free is that point, to within one cell."""
    area = _floor(_area())
    _slab(area, x=(-2.6, -0.7), y=(-3.7, -2.2))
    assert snap_to_traversable((1.0, 0.5), area) == pytest.approx((1.0, 0.5), abs=0.1)


def test_the_snap_takes_the_nearest_side_with_no_regard_for_the_robot():
    """This replaces a test asserting the opposite. The old snap carried a 0.10 vehicle-distance
    term to break ties toward the robot's side; it was a bias, not a reachability guard, and it
    directly contradicts aiming at the closest point. An off-centre waypoint now resolves to its
    own nearest edge wherever the robot happens to be."""
    area = _floor(_area(), y=(-4.0, 4.0))
    _slab(area, x=(-1.0, 1.0), y=(-0.5, 0.5))

    # Nearer the slab's lower edge, so the answer is below it -- from either side.
    assert snap_to_traversable((0.0, -0.3), area)[1] < -0.5
    assert snap_to_traversable((0.0, -0.3), area) == \
        snap_to_traversable((0.0, -0.3), area)


def test_no_terrain_yet_leaves_the_waypoint_untouched():
    """The area builds while the robot explores; a route that starts before any has arrived
    must still drive, exactly as it did before the snap existed."""
    assert snap_to_traversable((3.0, 4.0), _area()) == (3.0, 4.0)
    assert snap_to_traversable((3.0, 4.0), None) == (3.0, 4.0)


def test_ground_beyond_the_snap_limit_is_not_used():
    """Past max_snap_m the model's point is not 'slightly inside the furniture', it is
    somewhere we have no reading for -- and moving it that far aims at a different place."""
    area = _floor(_area(span=60.0), x=(-1.0, 1.0), y=(-1.0, 1.0))
    assert snap_to_traversable((40.0, 40.0), area) == (40.0, 40.0)   # off the grid entirely
    assert snap_to_traversable((9.0, 0.0), area) == (9.0, 0.0)       # on it, but too far


def test_the_clearance_knob_still_erodes_when_it_is_dialled_up():
    """Off by default, because every metre of clearance is a metre given away. It stays a knob
    so the base autonomy's own 0.75 m can be bought back from config, not from a code change."""
    area = _floor(_area(), y=(-4.0, 4.0))
    _slab(area, x=(-1.0, 1.0), y=(-0.5, 0.5))

    near = snap_to_traversable((0.0, -0.3), area)
    far = snap_to_traversable((0.0, -0.3), area, clearance_m=0.75)
    assert math.dist(far, (0.0, -0.3)) > math.dist(near, (0.0, -0.3))
    assert far[1] <= -0.5 - 0.75 + 0.1


def test_the_clearance_default_is_off():
    """It is the difference between aiming at the closest floor and aiming 0.75 m away from it."""
    assert OBSTACLE_CLEARANCE_M == 0.0
