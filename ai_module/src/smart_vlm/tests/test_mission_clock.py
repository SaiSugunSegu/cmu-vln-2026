"""Pure-python tests for the mission budget arithmetic (no ROS / GPU)."""
from __future__ import annotations

import pytest

from smart_vlm.mission_clock import MissionBudget, MissionClock, Phase

BUDGET = MissionBudget(
    question_budget_s=600.0,
    explore_timeout_s=120.0,
    answer_reserve_s=90.0,
    fallback_reserve_s=30.0,
)


def clock(t0: float = 0.0, budget: MissionBudget = BUDGET) -> MissionClock:
    return MissionClock(budget, t0)


# -- fixed deadlines --------------------------------------------------------

def test_reserves_are_measured_back_from_the_hard_limit():
    c = clock()
    assert c.hard_deadline == 600.0
    assert c.answer_deadline == 510.0
    assert c.fallback_deadline == 570.0


def test_deadlines_track_t0():
    c = clock(t0=1000.0)
    assert c.hard_deadline == 1600.0
    assert c.answer_deadline == 1510.0


# -- explore deadline anchoring --------------------------------------------

def test_explore_deadline_anchors_at_mark_exploring_not_t0():
    c = clock()
    c.mark_exploring(70.0)          # 70 s of model loading first
    assert c.explore_deadline == 190.0   # 70 + 120, NOT 120


def test_explore_deadline_is_clamped_to_the_answer_deadline():
    c = clock()
    c.mark_exploring(500.0)         # data arrives absurdly late
    assert c.explore_deadline == 510.0   # clamped, not 620


def test_explore_deadline_before_any_data_falls_back_to_answer_deadline():
    c = clock()
    assert not c.exploring_started
    assert c.explore_deadline == c.answer_deadline


def test_mark_exploring_is_idempotent_first_call_wins():
    c = clock()
    c.mark_exploring(50.0)
    c.mark_exploring(80.0)
    assert c.explore_deadline == 170.0


# -- phases -----------------------------------------------------------------

def test_phase_is_warmup_until_data_arrives():
    c = clock()
    assert c.phase_at(0.0) is Phase.WARMUP
    assert c.phase_at(300.0) is Phase.WARMUP


def test_phase_walks_exploring_then_answering_then_fallback_then_expired():
    c = clock()
    c.mark_exploring(10.0)          # explore_deadline = 130
    assert c.phase_at(10.0) is Phase.EXPLORING
    assert c.phase_at(129.9) is Phase.EXPLORING
    assert c.phase_at(130.0) is Phase.ANSWERING
    assert c.phase_at(569.9) is Phase.ANSWERING
    assert c.phase_at(570.0) is Phase.FALLBACK
    assert c.phase_at(599.9) is Phase.FALLBACK
    assert c.phase_at(600.0) is Phase.EXPIRED


def test_a_clock_that_never_explores_still_reaches_fallback_and_expiry():
    c = clock()
    assert c.phase_at(570.0) is Phase.FALLBACK
    assert c.phase_at(600.0) is Phase.EXPIRED


def test_fallback_wins_over_answering_when_both_boundaries_are_passed():
    c = clock()
    c.mark_exploring(0.0)
    assert c.phase_at(580.0) is Phase.FALLBACK


# -- misc -------------------------------------------------------------------

def test_elapsed_is_relative_to_t0():
    assert clock(t0=100.0).elapsed(160.0) == 60.0


def test_snapshot_reports_countdowns():
    c = clock()
    c.mark_exploring(10.0)
    snap = c.snapshot(30.0)
    assert snap["phase"] == "exploring"
    assert snap["elapsed_s"] == 30.0
    assert snap["explore_in_s"] == 100.0     # 130 - 30
    assert snap["fallback_in_s"] == 540.0    # 570 - 30
    assert snap["budget_left_s"] == 570.0


def test_budget_rejects_inverted_reserves():
    with pytest.raises(ValueError):
        MissionBudget(answer_reserve_s=30.0, fallback_reserve_s=90.0)


def test_budget_rejects_reserve_larger_than_budget():
    with pytest.raises(ValueError):
        MissionBudget(question_budget_s=60.0, answer_reserve_s=90.0, fallback_reserve_s=30.0)
