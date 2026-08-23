"""Unit tests for the target-exploration decision layer.

Every case here is a defect that shipped, or the contract with a component that would
otherwise only be checked by a live sim run: the bin inversion against the mapper's own
binning, TARE's array-order-is-priority rule, and each of the ways a sector must be able to
stop being open.
"""
import math

import numpy as np
import pytest

from smart_vlm.target_coverage import (BLOCKED, COVERED, OPEN, CoverageModel,
                                       default_params, validate_params)

N_BINS = 20
ROBOT = (0.0, 0.0, 0.75)


def model(**overrides):
    params = default_params()
    params.update(overrides)
    m = CoverageModel(params)
    m.set_targets({"sofa", "book"})
    return m


def obj(label="sofa", center=(3.0, 0.0, 0.5), extent=(0.6, 0.6, 0.6), bins=None,
        published=True, voxels=900):
    return {
        "label": label, "center": list(center), "extent": list(extent),
        "view_bins": list(bins if bins is not None else [False] * N_BINS),
        "published": published, "life": 20, "info_frames": 9, "voxels_total": voxels,
    }


def only_goal(m):
    return next(iter(m.goals.values()))


def accept(m, *requests):
    """TARE snapped these requests onto real candidate viewpoints and put them in its tour."""
    m.note_feedback({"accepted": [[r[0], r[1]] for r in requests], "unreachable": []})


def refuse(m, point):
    m.note_feedback({"accepted": [], "unreachable": [point]})


# -- the inversion ----------------------------------------------------------

def test_stand_position_inverts_the_mappers_own_binning():
    """Standing where we ask must fill the bin we asked for.

    Checked against sam_mapper's real `_obs_angle_bins` rather than a restatement of it: the
    two halves of this contract live in different packages, and a sign flip in either sends
    the robot to the wrong side of every object while both files still look right.
    """
    # sam_mapper.single_object pulls in open3d, which the container has and the host may not.
    single_object = pytest.importorskip("sam_mapper.single_object")
    discretize_angles = single_object.discretize_angles
    normalize_angles_to_pi = single_object.normalize_angles_to_pi

    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    goal = only_goal(m)
    radius = m.radius(goal, ROBOT[2])

    for bin_index in range(N_BINS):
        x, y = m._stand_at(goal, bin_index, radius)
        offset = np.array(goal.center[:2]) - np.array([x, y])
        angle = normalize_angles_to_pi(np.arctan2(offset[1], offset[0]))
        assert int(discretize_angles(angle, N_BINS)) == bin_index


def test_requests_aim_at_the_sector_centre():
    """One request per sector, at its centre bin — so one verdict closes one sector.

    Asking for the cheapest bin instead cost a feedback round trip per bin (five per sector,
    and an escalation reset them all), which is why sectors used to stay open indefinitely.
    """
    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    bins = {r[4] for r in m.requests(ROBOT, 0.0)}
    assert bins == {2, 7, 12, 17}


# -- geometry ---------------------------------------------------------------

def test_bulky_object_keeps_its_clearance():
    m = model()
    m.ingest([obj(extent=(2.1, 1.0, 0.8))], ROBOT, 0.0)
    goal = only_goal(m)
    assert m.radius(goal, ROBOT[2]) == pytest.approx(1.05 + m.p["min_standoff_m"])


def test_standoff_grows_so_a_tall_object_fits_the_vertical_fov():
    """The panorama is 120 degrees vertical and `bounds_mode: clip` pins anything outside it
    to the image edge, where a neighbouring mask swallows its lidar. A thin, tall object has
    no footprint to buy clearance with, so the frame-fit floor has to.

    Measured from the FURTHEST of the object's top and bottom, not just its top: an object
    straddling the sensor height is cut off at whichever end is further away.
    """
    m = model()
    m.set_targets({"lamp"})          # prior 1.1 x 1.1 x 5.72 m, so the height is not capped
    m.ingest([obj(label="lamp", center=(3.0, 0.0, 0.5), extent=(0.1, 0.1, 4.0))], ROBOT, 0.0)
    goal = only_goal(m)
    dz = max(abs(0.5 + 2.0 - ROBOT[2]), abs(ROBOT[2] - (0.5 - 2.0)))
    assert m.radius(goal, ROBOT[2]) == pytest.approx(0.05 + dz / math.tan(math.radians(60.0)))
    assert m.radius(goal, ROBOT[2]) > 0.05 + m.p["min_standoff_m"]


def test_standoff_from_the_surface_never_exceeds_the_mappers_range_filter():
    """Past range_filter.max_distance no lidar point is assigned to the object's mask. The
    filter measures to the POINTS, so it is the surface standoff that must be capped — a wide
    object legitimately puts its own centroid far behind the face we are looking at."""
    m = model()
    m.ingest([obj(extent=(40.0, 40.0, 1.0))], ROBOT, 0.0)
    goal = only_goal(m)
    assert m.radius(goal, ROBOT[2]) - 20.0 <= m.p["max_range_m"]


def test_positions_are_stable_under_sub_decimetre_centroid_jitter():
    """TARE re-adopts its drive target when a request moves, and matches feedback back to the
    position it echoed. A centroid that twitches every frame broke both."""
    m = model()
    m.ingest([obj(center=(3.0, 0.0, 0.5))], ROBOT, 0.0)
    first = [(r[0], r[1]) for r in m.requests(ROBOT, 0.0)]
    m.ingest([obj(center=(3.012, -0.009, 0.5))], ROBOT, 1.0)
    assert [(r[0], r[1]) for r in m.requests(ROBOT, 1.0)] == first


# -- coverage over-reporting ------------------------------------------------

def test_view_bins_are_preferred_over_the_per_voxel_angle_bins():
    """`angle_bins` ORs every voxel's azimuth, so one close pose at a large object marks a
    whole quadrant seen. `view_bins` is one bin per observation and is what a planner reads."""
    wide = [False] * N_BINS
    for k in range(5, 10):          # a whole sector, as the per-voxel OR would report it
        wide[k] = True
    narrow = [False] * N_BINS
    narrow[17] = True

    m = model()
    m.ingest([{**obj(bins=narrow), "angle_bins": wide}], ROBOT, 0.0)
    states = m.sector_states(only_goal(m))
    assert states[1] == OPEN and states[3] == COVERED


def test_angle_bins_are_used_when_view_bins_are_absent():
    """Older mapper output still works, at the cost of over-reporting on large objects."""
    bins = [False] * N_BINS
    bins[17] = True
    entry = obj()
    entry.pop("view_bins")
    m = model()
    m.ingest([{**entry, "angle_bins": bins}], ROBOT, 0.0)
    assert m.sector_states(only_goal(m))[3] == COVERED


# -- every sector must reach a terminal state -------------------------------

def test_one_refusal_blocks_a_sector_once_the_retries_are_spent():
    m = model(radius_retries=0)
    m.ingest([obj()], ROBOT, 0.0)
    request = m.requests(ROBOT, 0.0)[0]
    refuse(m, [request[0], request[1]])
    assert m.sector_states(only_goal(m))[request[3]] == BLOCKED


def test_a_refusal_first_backs_off_and_asks_again():
    """"Refused" at 1.0 m is ambiguous: nothing may be able to stand there, or we may have
    asked the robot to stand inside the object. Escalating separates the two."""
    m = model(radius_retries=1)
    m.ingest([obj()], ROBOT, 0.0)
    first = m.requests(ROBOT, 0.0)[0]
    refuse(m, [first[0], first[1]])

    goal = only_goal(m)
    assert m.sector_states(goal)[first[3]] == OPEN
    again = next(r for r in m.requests(ROBOT, 1.0) if r[3] == first[3])
    # Further out AND a different angle: `refused` is never cleared, so the retry differs in
    # both, and asking the identical question twice is not a retry.
    assert math.hypot(again[0] - goal.center[0], again[1] - goal.center[1]) > \
        math.hypot(first[0] - goal.center[0], first[1] - goal.center[1])
    assert again[4] != first[4]

    refuse(m, [again[0], again[1]])
    assert m.sector_states(only_goal(m))[first[3]] == BLOCKED


def stand_and_wait(m, request, t0):
    """TARE accepts the request, the robot goes there, and the mapper reports back without the
    sector filling."""
    accept(m, request)
    m.note_position((request[0], request[1], ROBOT[2]))
    for tick in range(int(m.p["arrival_patience_updates"])):
        m.ingest([obj()], ROBOT, t0 + tick)


def test_standing_there_without_a_detection_first_backs_off():
    """The commonest cause is the lidar's blind cone below the sensor, which swallows
    floor-level objects at close range (measured: stool 82% of detections with zero points).
    The cure is standing further back, so a non-detection must escalate before it blocks."""
    m = model(radius_retries=1)
    m.ingest([obj()], ROBOT, 0.0)
    first = m.requests(ROBOT, 0.0)[0]
    stand_and_wait(m, first, 2.0)

    goal = only_goal(m)
    assert m.sector_states(goal)[first[3]] == OPEN
    again = next(r for r in m.requests(ROBOT, 10.0) if r[3] == first[3])
    assert math.hypot(again[0] - goal.center[0], again[1] - goal.center[1]) > \
        math.hypot(first[0] - goal.center[0], first[1] - goal.center[1])


def test_standing_there_without_a_detection_blocks_once_the_retries_are_spent():
    """The one closure TARE cannot supply. A side that is reachable but occluded — or that
    SAM does not fire on from that angle — would otherwise stay open forever and the robot
    would keep being sent back to a spot it has already visited."""
    m = model(radius_retries=0)
    m.ingest([obj()], ROBOT, 0.0)
    request = m.requests(ROBOT, 0.0)[0]
    stand_and_wait(m, request, 2.0)

    goal = only_goal(m)
    assert m.sector_states(goal)[request[3]] == BLOCKED
    # Recorded distinctly from an unreachable refusal: one says nothing can stand there, the
    # other says standing there achieved nothing, and only the report can tell them apart.
    assert m._goal_status(goal, 0.0)["blocked_reason"][request[3]] == "stood-there-no-detection"


def test_arrival_alone_does_not_block_a_sector():
    """Patience is counted in MAPPER updates, not wall time: only a frame in which the mapper
    saw the object can answer "did it register from here"."""
    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    request = m.requests(ROBOT, 0.0)[0]
    accept(m, request)
    m.note_position((request[0], request[1], ROBOT[2]))
    assert m.sector_states(only_goal(m))[request[3]] == OPEN


def test_a_verdict_that_crossed_a_republish_is_still_attributed():
    """TARE's feedback runs at ~1 Hz against our 2 Hz publish. Matching only the newest array
    silently discarded every verdict that crossed a republish, which is most of them."""
    m = model(radius_retries=0)
    m.ingest([obj()], ROBOT, 0.0)
    stale = m.requests(ROBOT, 0.0)[0]
    for tick in range(3):
        m.requests((0.1 * tick, 0.0, ROBOT[2]), 1.0 + tick)
    refuse(m, [stale[0], stale[1]])
    assert m.sector_states(only_goal(m))[stale[3]] == BLOCKED


# -- priority and commitment ------------------------------------------------

def test_an_unpublished_goal_outranks_a_nearer_published_one():
    """An unpublished object has no centroid, so it is ABSENT from obj_map.json — the answer
    path cannot name it at all. Its first sector is worth more than a published object's
    fourth however much nearer that one is."""
    near = obj(label="sofa", center=(1.5, 0.0, 0.5), published=True)
    far = obj(label="book", center=(6.0, 0.0, 0.5), published=False, voxels=200)
    m = model()
    m.ingest([near, far], ROBOT, 0.0)
    assert m.rank(m.pending(), ROBOT, prefer_committed=False)[0][0] == "book"


def test_the_committed_goal_leads_the_array():
    """TARE drives at the first ACCEPTED pose and holds it, so array order IS the priority
    signal — the committed goal must be at index 0."""
    m = model()
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5)),
              obj(label="book", center=(2.6, 0.0, 0.5))], ROBOT, 0.0)
    requests = m.requests(ROBOT, 0.0)
    assert requests[0][2] == m.committed


def test_commitment_survives_the_other_goal_becoming_nearer():
    """Re-electing on distance from a moving robot made the robot flip between two roughly
    equidistant objects and cover neither."""
    m = model()
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5)),
              obj(label="book", center=(-2.0, 0.0, 0.5))], ROBOT, 0.0)
    committed = m.committed
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5)),
              obj(label="book", center=(-2.0, 0.0, 0.5))], (-1.9, 0.0, 0.75), 5.0)
    assert m.committed == committed


def test_a_newly_unpublished_goal_preempts_the_commitment():
    """The one preemption the dwell must not block. It also keeps `committed` honest: the
    array is ordered by rank(), so anything outranking the committed goal would take index 0
    and TARE would drive at a goal we never committed to."""
    m = model()
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5), published=True)], ROBOT, 0.0)
    assert m.committed[0] == "sofa"

    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5), published=True),
              obj(label="book", center=(-5.0, 0.0, 0.5), published=False, voxels=500)],
             ROBOT, 1.0)
    assert m.committed[0] == "book"
    assert m.requests(ROBOT, 1.0)[0][2] == m.committed


def test_the_deadlock_guard_releases_a_goal_making_no_progress():
    m = model(goal_dwell_s=10.0)
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5)),
              obj(label="book", center=(-2.0, 0.0, 0.5))], ROBOT, 0.0)
    committed = m.committed
    m.ingest([obj(label="sofa", center=(2.0, 0.0, 0.5)),
              obj(label="book", center=(-2.0, 0.0, 0.5))], ROBOT, 100.0)
    assert m.committed != committed
    assert committed in m.pending()      # deprioritised, never dropped


def test_the_array_never_exceeds_tares_budget():
    """Poses past the eighth ACCEPTED one get no verdict and no priority cell, so sending
    more than kMaxTargetViewPointNum loses information."""
    m = model()
    m.set_targets({"chair"})
    m.ingest([obj(label="chair", center=(3.0 * i, 2.0 * i, 0.5), published=False, voxels=500)
              for i in range(1, 7)], ROBOT, 0.0)
    assert len(m.requests(ROBOT, 0.0)) <= m.p["max_viewpoints"]


def test_only_the_committed_goal_contributes_its_whole_orbit():
    """Each must-visit viewpoint marks its covered points covered before the coverage queues
    are scored, so a long array actively suppresses frontier exploration."""
    m = model()
    m.set_targets({"chair"})
    m.ingest([obj(label="chair", center=(2.0, 0.0, 0.5), published=False, voxels=500),
              obj(label="chair", center=(-6.0, 0.0, 0.5), published=False, voxels=500)],
             ROBOT, 0.0)
    requests = m.requests(ROBOT, 0.0)
    assert sum(1 for r in requests if r[2] != m.committed) == 1


def test_the_array_head_always_matches_the_commitment():
    """The invariant the whole ordering rests on: TARE drives at the first accepted pose, so
    if index 0 and `committed` can disagree, the log says one thing and the robot does another.
    Driven through goals appearing, being covered, and being refused."""
    m = model()
    m.set_targets({"sofa", "book", "lamp"})
    scene = [obj(label="sofa", center=(2.0, 0.0, 0.5), published=True)]
    for tick in range(1, 12):
        if tick == 3:
            scene.append(obj(label="book", center=(-4.0, 1.0, 0.5), published=False, voxels=500))
        if tick == 6:
            scene.append(obj(label="lamp", center=(5.0, -3.0, 0.5), published=True))
        m.ingest(scene, ROBOT, float(tick))
        requests = m.requests(ROBOT, float(tick))
        assert requests and requests[0][2] == m.committed
        refuse(m, [requests[0][0], requests[0][1]])


# -- preempt and the stop signal --------------------------------------------

def test_preempt_stays_off_while_a_label_has_no_instance():
    """An undiscovered label can be found only by ordinary frontier exploration, and preempt
    restricts TARE's global tour to subspaces already holding a target."""
    m = model()
    m.ingest([obj(label="sofa")], ROBOT, 0.0)
    assert m.preempt(0.0) is False


def test_preempt_turns_on_once_every_label_is_found():
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5)),
              obj(label="book", center=(-3.0, 0.0, 0.5))], ROBOT, 0.0)
    assert m.preempt(0.0) is True


def test_coverage_complete_needs_every_label_and_every_sector():
    covered = [True] * N_BINS
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=covered)], ROBOT, 0.0)
    assert m.coverage_complete() is False           # `book` never found

    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=covered),
              obj(label="book", center=(-3.0, 0.0, 0.5))], ROBOT, 1.0)
    assert m.coverage_complete() is False           # `book` found but not covered

    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=covered),
              obj(label="book", center=(-3.0, 0.0, 0.5), bins=covered)], ROBOT, 2.0)
    assert m.coverage_complete() is True


def test_coverage_is_never_complete_with_no_targets():
    m = CoverageModel(default_params())
    assert m.coverage_complete() is False


# -- clustering -------------------------------------------------------------

def test_fragments_of_one_object_become_one_goal_at_their_weighted_centre():
    """Unpublished objects never enter world merge — that path needs a non-None centroid on
    both sides — so the planner has to deduplicate them or it re-inspects one object twice."""
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), published=False, voxels=300),
              obj(label="sofa", center=(3.4, 0.0, 0.5), published=False, voxels=100)],
             ROBOT, 0.0)
    assert len(m.goals) == 1
    assert only_goal(m).center[0] == pytest.approx(3.1)


def test_the_clustering_radius_does_not_follow_the_mappers_merge_distance():
    """It used to read `world_merge.absolute_distance`, and then that was cut 0.5 -> 0.25.

    The two decisions are asymmetric: world merge publishes one box, so fusing two real
    objects loses a question, while this layer only chooses where to look, so failing to fuse
    two fragments costs a second inspection tour of an object already inspected. Reading one
    knob for both meant a mapper fix silently split every fragment pair 0.25-0.5 m apart.
    """
    from sam_mapper.mapping_config import MappingConfig

    assert default_params()["cluster_distance_m"] == 0.5
    assert MappingConfig().world_merge.absolute_distance < 0.5


def test_clustering_unions_the_bins_of_a_short_fragment():
    """zip_longest, not zip: a fragment carrying a shorter bin vector would truncate the union
    to [], leaving a goal that yields no viewpoint and can never be satisfied."""
    bins = [False] * N_BINS
    bins[17] = True
    short = obj(label="sofa", center=(3.3, 0.0, 0.5), published=False, voxels=100)
    short["view_bins"] = []
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=bins, published=False,
                  voxels=300), short], ROBOT, 0.0)
    assert len(only_goal(m).bins) == N_BINS
    assert m.sector_states(only_goal(m))[3] == COVERED


def test_a_published_object_is_admitted_however_small():
    """The mapper already judged it good enough for obj_map.json; a stricter bar here than the
    map's own would be incoherent. It measurably was — arabic_room's `book` reached
    obj_map.json yet never became a goal."""
    m = model()
    m.ingest([obj(label="book", published=True, voxels=1)], ROBOT, 0.0)
    assert len(m.goals) == 1


def test_an_unpublished_speck_is_not_admitted():
    m = model()
    m.ingest([obj(label="book", published=False, voxels=1)], ROBOT, 0.0)
    assert m.goals == {}


def test_new_targets_wipe_the_goals():
    """A re-arm wipes the map (map_node drops everything on a new run_id), so goals built from
    the old one are meaningless."""
    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    assert m.set_targets({"lamp"}) is True
    assert m.goals == {} and m.committed is None


def test_published_never_goes_backwards():
    """An object drops out of obj_map.json whenever regularize_shape has a bad frame;
    un-publishing the goal would re-promote something we already paid to cover."""
    m = model()
    m.ingest([obj(published=True)], ROBOT, 0.0)
    m.ingest([obj(published=False)], ROBOT, 1.0)
    assert only_goal(m).published is True


# -- reporting --------------------------------------------------------------

def test_summary_separates_never_found_from_never_published():
    """The two failures the score cannot tell apart: SAM never saw it, versus SAM saw it but
    the robot never circled it enough for a centroid."""
    m = model()
    m.ingest([obj(label="sofa", published=False, voxels=500)], ROBOT, 0.0)
    summary = m.summary()
    assert summary["labels_unseen"] == ["book"]
    assert summary["goals_unpublished"] == 1
    assert summary["coverage_complete"] is False


# -- goal lifecycle ---------------------------------------------------------

def test_an_orphaned_goal_goes_dormant_and_stops_blocking_the_stop_signal():
    """The defect that disabled the early stop. `describe_objects` reports every TRACKED
    object every frame, so a goal nothing matches is one the mapper let go of — a pruned SAM
    false positive, most often. Left alive it is `pending` for the rest of the run, and since
    `coverage_complete` needs an empty pending list, ONE phantom disables the stop for good."""
    covered = [True] * N_BINS
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=covered),
              obj(label="book", center=(-3.0, 0.0, 0.5), published=False, voxels=500)],
             ROBOT, 0.0)
    assert m.coverage_complete() is False

    for tick in range(int(m.p["goal_absence_limit"])):     # the mapper pruned the `book`
        m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), bins=covered)], ROBOT, 1.0 + tick)
    assert m.pending() == []
    assert m.summary()["goals_dormant"] == 1
    # `book` is gone from the map, so it is gone from `found` too — the stop condition still
    # refuses to fire, because a label with no instance means there is exploring left to do.
    assert m.coverage_complete() is False
    assert m.summary()["labels_unseen"] == ["book"]


def test_a_dormant_goal_revives_with_its_sector_history_intact():
    """An object dropping out for a few frames is ordinary. Deleting would throw away the
    sectors we already paid to cover and restart the orbit from scratch on the rebound."""
    bins = [False] * N_BINS
    bins[17] = True
    m = model()
    m.ingest([obj(bins=bins)], ROBOT, 0.0)
    for tick in range(int(m.p["goal_absence_limit"])):
        m.ingest([], ROBOT, 1.0 + tick)
    assert only_goal(m).dormant is True

    m.ingest([obj(bins=bins)], ROBOT, 20.0)
    assert only_goal(m).dormant is False
    assert m.sector_states(only_goal(m))[3] == COVERED


def test_a_walking_unpublished_centroid_does_not_accumulate_goals():
    """An unpublished object's `center` is `provisional_centroid()`, the raw voxel mean, which
    walks further in one frame than `_match` reaches — reproduced at four goals for one
    object. Each demanded four sectors and `coverage_complete` needed all of them."""
    m = model()
    for tick, x in enumerate([3.0, 3.6, 4.3, 5.1, 5.8, 6.5, 7.2, 7.9, 8.6]):
        m.ingest([obj(label="sofa", center=(x, 0.0, 0.5), published=False, voxels=500)],
                 ROBOT, float(tick))
    assert len(m.pending()) == 1
    assert len(m.live_goals()) == 1


def test_a_new_run_id_drops_the_map_state_but_keeps_the_labels():
    """map_node wipes its map on a new run_id, and two consecutive questions can share
    prompts — so the label set is not the signal that a re-arm happened."""
    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    m.requests(ROBOT, 0.0)
    m.reset_map_state()
    assert m.goals == {} and m.committed is None and not m.last_request
    assert m.targets == {"sofa", "book"}


# -- extent, clamped by the mapper's own class prior -------------------------

def test_a_bled_extent_is_held_to_the_class_prior():
    """An unpublished object's extent is `np.ptp` over every raw voxel, bleed included. A
    `book` inflated to 4.5 x 3.8 m asked for a 3.05 m standoff, where a book is ~30 px across
    a 1920 px panorama — the tier-0 objects the priority scheme exists to rescue were the ones
    being sent furthest away."""
    m = model()
    bled = obj(label="book", center=(3.0, 0.0, 0.5), published=False, voxels=500)
    bled["extent"] = [4.5, 3.8, 1.2]
    m.ingest([bled], ROBOT, 0.0)
    goal = only_goal(m)
    # book's prior is 0.73 x 0.42 x 0.44, so the footprint contributes 0.365 m, not 2.25 m.
    assert m.radius(goal, ROBOT[2]) == pytest.approx(0.365 + m.p["min_standoff_m"])


def test_an_unknown_label_falls_back_to_the_default_prior():
    m = model()
    m.set_targets({"widget"})
    entry = obj(label="widget", published=False, voxels=500)
    entry["extent"] = [40.0, 40.0, 40.0]
    m.ingest([entry], ROBOT, 0.0)
    assert m.radius(only_goal(m), ROBOT[2]) <= 0.5 * 6.0 + m.p["max_range_m"]


# -- TARE's accepted verdict -------------------------------------------------

def test_a_drive_by_is_not_an_arrival():
    """`accepted` is what separates "went there" from "clipped it on the way to something
    else". Without the guard, a request TARE never placed escalates and then blocks a sector
    nobody ever went to look at."""
    m = model()
    m.ingest([obj()], ROBOT, 0.0)
    request = m.requests(ROBOT, 0.0)[0]

    m.note_position((request[0], request[1], ROBOT[2]))     # near it, but TARE never placed it
    for tick in range(int(m.p["arrival_patience_updates"]) + 2):
        m.ingest([obj()], ROBOT, 2.0 + tick)
    assert m.sector_states(only_goal(m))[request[3]] == OPEN


def test_an_escalated_request_needs_a_fresh_acceptance():
    """The retry is a different place. Carrying the old verdict over would let the robot's
    current position count as an arrival at a request nobody has agreed to yet."""
    m = model(radius_retries=1)
    m.ingest([obj()], ROBOT, 0.0)
    first = m.requests(ROBOT, 0.0)[0]
    accept(m, first)
    refuse(m, [first[0], first[1]])

    sector = only_goal(m).sectors[first[3]]
    assert sector.accepted is False and sector.verdict is False


# -- preempt -----------------------------------------------------------------

def test_preempt_turns_on_for_an_in_horizon_goal_even_with_a_label_unfound():
    """Prompts carry anchors, and a scene whose stool SAM calls a chair leaves one label
    permanently unfound. Requiring every label would then disable the mechanism for the whole
    run. Restricting the global tour is inert when nothing is far — `priority_cell_indices_`
    is populated only from the `far` branch — so this is safe for discovery."""
    m = model()
    m.ingest([obj(label="sofa")], ROBOT, 0.0)
    assert m.preempt(0.0) is False                  # nothing judged yet: treat it as far

    accept(m, *m.requests(ROBOT, 0.0))
    assert m.preempt(0.0) is True


def test_preempt_stays_off_while_a_pending_goal_is_out_of_reach():
    """A request in neither feedback list is `far`. Preempting then WOULD restrict the global
    tour, which is the one thing that can still find the missing label."""
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5)),
              obj(label="lamp", center=(30.0, 0.0, 0.5))], ROBOT, 0.0)
    m.set_targets({"sofa", "lamp", "book"})
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5)),
              obj(label="lamp", center=(30.0, 0.0, 0.5))], ROBOT, 1.0)
    requests = m.requests(ROBOT, 1.0)
    accept(m, *[r for r in requests if abs(r[0]) < 10.0])   # only the near goal is judged
    assert m.preempt(1.0) is False


# -- clustering guard --------------------------------------------------------

def test_two_published_objects_are_never_fused():
    """World merge already refused to fuse them — `block_covisible` exists because distance
    alone fused 0.44 m-spaced pillows. Re-fusing here leaves one goal orbiting their midpoint."""
    m = model()
    m.set_targets({"pillow"})
    m.ingest([obj(label="pillow", center=(3.0, 0.0, 0.5), published=True),
              obj(label="pillow", center=(3.44, 0.0, 0.5), published=True)], ROBOT, 0.0)
    assert len(m.goals) == 2


def test_an_unpublished_fragment_still_joins_its_object():
    """Clustering exists for exactly this: world merge needs a non-None centroid on both
    sides, so an unpublished fragment can never reach it."""
    m = model()
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5), published=True),
              obj(label="sofa", center=(3.3, 0.0, 0.5), published=False, voxels=100)],
             ROBOT, 0.0)
    assert len(m.goals) == 1


# -- the deadlock guard ------------------------------------------------------

def test_the_deadlock_guard_fires_even_while_bins_keep_filling():
    """Bins fill whenever the robot passes near the object, including on the way to a
    different goal. Counting them as progress made the guard unfireable on a stuck goal."""
    bins = [False] * N_BINS
    bins[0] = True                              # sector 0 already covered
    m = model(goal_dwell_s=10.0)
    scene = lambda: [obj(label="sofa", center=(2.0, 0.0, 0.5), bins=list(bins)),
                     obj(label="book", center=(-2.0, 0.0, 0.5))]
    m.ingest(scene(), ROBOT, 0.0)
    committed = m.committed
    for tick, k in enumerate((1, 2, 3)):        # more bins inside that SAME covered sector
        bins[k] = True
        m.ingest(scene(), ROBOT, 50.0 + tick)
    assert m.committed != committed
    assert committed in m.pending()


def test_a_deferred_goal_competes_on_distance_again_after_its_cooldown():
    """A deferral tally means a twice-deferred goal loses to a once-deferred one across the
    room forever, and route order decays into round-robin as the run goes on."""
    m = model(goal_dwell_s=10.0)
    scene = [obj(label="sofa", center=(1.0, 0.0, 0.5)),
             obj(label="book", center=(-8.0, 0.0, 0.5))]
    m.ingest(scene, ROBOT, 0.0)
    near = m.committed
    m.ingest(scene, ROBOT, 100.0)                       # dwell expires, near goal deferred
    assert m.committed != near
    assert m.rank(m.pending(), ROBOT, prefer_committed=False, now=100.0)[0] != near
    # Long after the cooldown the deferral stops mattering and the nearer goal leads again.
    assert m.rank(m.pending(), ROBOT, prefer_committed=False, now=1000.0)[0] == near


# -- parameter invariants ----------------------------------------------------

def test_an_escalation_that_lands_inside_arrival_range_is_rejected():
    """Load-bearing, and silent when violated: the retry would be consumed as an arrival
    without the robot moving at all."""
    from smart_vlm.target_coverage import validate_params

    params = default_params()
    validate_params(params)                                  # the shipped set is legal
    params["radius_retry_step_m"] = params["arrival_radius_m"] - 0.1
    with pytest.raises(ValueError, match="arrival_radius_m"):
        validate_params(params)


def test_asking_for_more_viewpoints_than_tare_will_take_is_rejected():
    """TARE breaks out of its loop after kMaxTargetViewPointNum ACCEPTED poses, so the tail
    gets no verdict and no priority cell — not refused, unheard."""
    from smart_vlm.target_coverage import validate_params

    params = default_params()
    params["max_viewpoints"] = 32
    with pytest.raises(ValueError, match="kMaxTargetViewPointNum"):
        validate_params(params)


def test_the_viewpoint_cap_comes_from_tares_own_number():
    """The cap is TARE's `kMaxTargetViewPointNum`, injected by target_explorer from the
    scenario yaml. Mirroring it here meant three copies and no way to notice a drift."""
    from smart_vlm.target_coverage import TARE_DEFAULTS, validate_params

    params = default_params()
    assert params["max_viewpoints_cap"] == TARE_DEFAULTS["max_target_viewpoints"]

    params.update(max_viewpoints_cap=4, max_viewpoints=4)   # a yaml that lowered it
    validate_params(params)
    params["max_viewpoints"] = 5
    with pytest.raises(ValueError, match="kMaxTargetViewPointNum"):
        validate_params(params)


def test_arrival_radius_is_derived_from_the_snap_radius():
    """Not a chosen number: TARE snaps a request onto a candidate up to `snap_max_dist_m`
    away and the base planner stops within waypointXYRadius of that, so together they bound
    how far from the asked-for spot the robot can legitimately come to rest."""
    from smart_vlm.target_coverage import TARE_DEFAULTS, WAYPOINT_XY_RADIUS_M

    params = default_params()
    assert params["arrival_radius_m"] == pytest.approx(
        TARE_DEFAULTS["snap_max_dist_m"] + WAYPOINT_XY_RADIUS_M)
    # And the invariant the escalation depends on still holds at the shipped values.
    assert params["radius_retry_step_m"] > params["arrival_radius_m"]


def test_blocking_sectors_is_not_progress():
    """The live regression: a goal whose sectors were only being BLOCKED renewed its own hold
    on every refusal, so one `sofa` stayed committed for 84 s with nine goals pending. Giving
    up on a side is attrition; only coverage renews the dwell."""
    m = model(goal_dwell_s=10.0, radius_retries=0)
    scene = [obj(label="sofa", center=(2.0, 0.0, 0.5)),
             obj(label="book", center=(-2.0, 0.0, 0.5))]
    m.ingest(scene, ROBOT, 0.0)
    committed = m.committed

    # Refuse the committed goal's sectors one at a time, exactly as TARE does.
    now = 1.0
    for _ in range(3):
        request = next((r for r in m.requests(ROBOT, now) if r[2] == committed), None)
        if request is None:
            break
        refuse(m, [request[0], request[1]])
        now += 2.0
    assert BLOCKED in m.sector_states(m.goals[committed])

    m.ingest(scene, ROBOT, 100.0)          # well past the dwell, no sector ever covered
    assert m.committed != committed


def test_target_labels_use_the_mappers_spelling():
    """The live regression: prompts and map labels are two spellings of one class.

    map_node names every object with `sam_mapper.detections.default_label`, which strips
    INTERNAL spaces — `potted plant` becomes `pottedplant`. target_explorer used to lowercase
    the prompt and nothing else, so `_admit` compared `potted plant` against `pottedplant` and
    matched nothing: no goal was ever built, the label read "never found" for the whole run,
    and `preempt()` therefore stayed off, which let TARE declare exploration finished and park.
    Measured on the 13-scene sim sweep: 9 of 13 questions carried a multi-word target, and
    office_1 (both targets multi-word) requested zero viewpoints across its 150 s window.

    So this is not a style choice about label spelling — it is the join key between two
    processes, and it has exactly one definition.
    """
    from sam_mapper.detections import default_label

    m = model()
    m.set_targets({default_label(p) for p in ("potted plant", "File Cabinet")})
    m.ingest([obj(label="pottedplant", center=(2.0, 0.0, 0.5)),
              obj(label="filecabinet", center=(-2.0, 0.0, 0.5))], ROBOT, 0.0)
    assert m.found_labels() == m.targets
    assert m.pending()


# -- preempt hysteresis -------------------------------------------------------

def test_preempt_does_not_flip_off_on_a_single_volatile_frame():
    """The live regression: preempt flipped 4-8 times per question.

    Every flip switches TARE's global tour between priority-only and stock, so the tour is
    re-solved against a different cell set every second or two. The raw condition is volatile
    in both directions — one newly discovered far goal turns it off, one covered sector turns
    it back on — and over a 13-scene sweep that produced 194 transitions in 50 questions.

    Driven here exactly as the sim drives it: `lamp` is never found, so preempt rides on the
    "every pending goal in-horizon" disjunct, and one new goal TARE stays silent about is
    enough to break it.
    """
    m = model(coverage_hold_s=3.0)
    m.set_targets({"sofa", "book", "lamp"})
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5)),
              obj(label="book", center=(-3.0, 0.0, 0.5))], ROBOT, 0.0)
    accept(m, *m.requests(ROBOT, 0.0))
    assert m.preempt(0.0) is True

    # A third object appears and TARE says nothing about it: `far`, so the raw condition drops.
    scene = [obj(label="sofa", center=(3.0, 0.0, 0.5)),
             obj(label="book", center=(-3.0, 0.0, 0.5)),
             obj(label="sofa", center=(0.0, 9.0, 0.5))]
    m.ingest(scene, ROBOT, 1.0)
    requests = m.requests(ROBOT, 1.0)
    accept(m, *[r for r in requests if r[1] < 5.0])      # only the near goals are judged
    assert m._preempt_raw() is False
    assert m.preempt(1.0) is True                 # held: 0s of continuous false so far
    assert m.preempt(3.5) is True                 # 2.5s < coverage_hold_s
    assert m.preempt(4.5) is False                # 3.5s of continuous false: released


def test_preempt_turning_on_is_not_delayed():
    """Asymmetric on purpose. Coming ON is a real signal and every second it is withheld is a
    second the robot spends on the frontier tour instead of at a target; only the OFF edge is
    noise worth suppressing."""
    m = model(coverage_hold_s=3.0)
    m.ingest([obj(label="sofa")], ROBOT, 0.0)
    assert m.preempt(0.0) is False
    m.ingest([obj(label="sofa", center=(3.0, 0.0, 0.5)),
              obj(label="book", center=(-3.0, 0.0, 0.5))], ROBOT, 0.5)
    assert m.preempt(0.5) is True                 # same tick the condition became true


def test_a_recovered_condition_rearms_the_hold():
    """A condition that flickers false-true-false must get the full hold each time, or the
    latch degrades into a countdown that a single true tick cannot reset."""
    m = model(coverage_hold_s=3.0)
    m.set_targets({"sofa", "book", "lamp"})
    near = [obj(label="sofa", center=(3.0, 0.0, 0.5)),
            obj(label="book", center=(-3.0, 0.0, 0.5))]
    far = near + [obj(label="sofa", center=(0.0, 9.0, 0.5))]

    m.ingest(near, ROBOT, 0.0)
    accept(m, *m.requests(ROBOT, 0.0))
    assert m.preempt(0.0) is True

    m.ingest(far, ROBOT, 1.0)                                  # goes false at 1.0
    accept(m, *[r for r in m.requests(ROBOT, 1.0) if r[1] < 5.0])
    assert m.preempt(1.0) is True

    m.ingest(far, ROBOT, 2.0)                                  # recovers: TARE judges them all
    accept(m, *m.requests(ROBOT, 2.0))
    assert m.preempt(2.0) is True

    m.ingest(far, ROBOT, 3.0)                                  # ...and fails again at 3.0
    accept(m, *[r for r in m.requests(ROBOT, 3.0) if r[1] < 5.0])
    assert m.preempt(3.0) is True
    assert m.preempt(5.0) is True                 # 2s in, not 4s: the hold restarted at 3.0
    assert m.preempt(6.5) is False


def test_an_unreachable_sector_gets_two_tries_before_it_is_written_off():
    """TARE now answers `unreachable` for any square inside the lattice that is not a
    candidate, which is where two of a wall-mounted object's four sectors land every cycle.
    At radius_retries=1 that verdict blocked the sector after a single retry and 412 of 446
    goals closed with blocked sectors."""
    m = model()
    assert m.p["radius_retries"] == 2
    m.ingest([obj()], ROBOT, 0.0)
    goal = only_goal(m)

    first = m.requests(ROBOT, 0.0)[0]
    sector_index = first[3]
    refuse(m, [first[0], first[1]])
    assert m.sector_states(goal)[sector_index] == OPEN

    second = next(r for r in m.requests(ROBOT, 1.0) if r[3] == sector_index)
    refuse(m, [second[0], second[1]])
    assert m.sector_states(goal)[sector_index] == OPEN      # still trying, further out

    third = next(r for r in m.requests(ROBOT, 2.0) if r[3] == sector_index)
    refuse(m, [third[0], third[1]])
    assert m.sector_states(goal)[sector_index] == BLOCKED


# -- two-phase exploration -------------------------------------------------
# Phase one hands the room to TARE's frontier exploration and withholds target viewpoints;
# phase two is the rest of this file, unchanged. The gate itself lives on the ROS node, so it
# is exercised unbound against a stand-in -- the same trick test_cat3_eval.py uses.


def _phase_stub(on=True, limit=180.0, targets=("chair",)):
    import types
    return types.SimpleNamespace(
        _p={"tare_explore_priority": on, "tare_explore_max_time": limit},
        _tare_phase=on, _tare_phase_started=None, tare_finished=False,
        model=types.SimpleNamespace(targets=set(targets)),
        p=lambda n, _s=None: {"tare_explore_priority": on,
                              "tare_explore_max_time": limit}[n],
        get_logger=lambda: types.SimpleNamespace(info=lambda *_a, **_k: None),
    )


def _active(stub, now):
    explorer = pytest.importorskip("smart_vlm.target_explorer")
    return explorer.TargetExplorer._tare_phase_active(stub, now)


def test_the_global_phase_is_skipped_when_the_flag_is_off():
    """False is the old behaviour: targets compete with the frontier from the first second."""
    assert _active(_phase_stub(on=False), 0.0) is False


def test_the_global_phase_ends_on_its_cap():
    stub = _phase_stub()
    assert _active(stub, 0.0) is True
    assert _active(stub, 179.0) is True
    assert _active(stub, 180.0) is False


def test_the_global_phase_never_restarts_once_it_has_ended():
    """Latched. A late /exploration_finish or a clock wobble must not pull steering back off
    the targets half way through the window they were given."""
    stub = _phase_stub()
    _active(stub, 0.0)
    _active(stub, 180.0)
    assert _active(stub, 181.0) is False
    assert stub._tare_phase is False


def test_tare_finishing_ends_the_global_phase_early():
    stub = _phase_stub()
    _active(stub, 0.0)
    stub.tare_finished = True
    assert _active(stub, 42.0) is False


def test_the_clock_starts_when_the_targets_land_not_at_startup():
    """This node is constructed while SAM loads weights. Anchoring the cap at construction
    would spend the global phase before the scene was even released."""
    stub = _phase_stub(targets=())
    _active(stub, 0.0)
    _active(stub, 500.0)
    assert stub._tare_phase_started is None
    stub.model.targets = {"chair"}
    assert _active(stub, 500.0) is True
    assert stub._tare_phase_started == 500.0
    assert _active(stub, 679.0) is True          # cap is measured from arming
    assert _active(stub, 680.0) is False


def test_a_zero_cap_with_the_flag_on_is_rejected():
    """TARE never raised its finish signal once in a measured 15-scene sweep, so with no cap
    the target phase would never begin at all."""
    params = dict(default_params())
    params.update(max_viewpoints_cap=8, tare_explore_priority=True, tare_explore_max_time=0.0)
    with pytest.raises(ValueError, match="tare_explore_max_time"):
        validate_params(params)
