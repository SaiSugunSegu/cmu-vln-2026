"""Pure-python tests for the mission budget arithmetic (no ROS / GPU)."""
from __future__ import annotations

import pytest

from smart_vlm.mission_clock import (CATEGORY_ANSWER_RESERVE_S, MissionBudget,
                                     MissionClock, Phase, budget_for)

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


# -- category-aware budgets -------------------------------------------------
# An instruction question is answered by DRIVING the route, not by publishing a message, so it
# needs the answering reserve a model call does not. The category is only known once the
# question arrives, which is after t0 and possibly after exploration has begun.


def test_budget_for_category_3_raises_the_answer_reserve():
    base = MissionBudget()
    raised = budget_for(3, base)
    assert raised.answer_reserve_s == CATEGORY_ANSWER_RESERVE_S[3]
    assert raised.answer_reserve_s > base.answer_reserve_s
    # Nothing else moves: the hard limit is the challenge's, not ours to spend.
    assert raised.question_budget_s == base.question_budget_s
    assert raised.fallback_reserve_s == base.fallback_reserve_s
    assert raised.explore_timeout_s == base.explore_timeout_s


def test_budget_for_other_categories_is_identity():
    base = MissionBudget()
    for category in (None, 1, 2):
        assert budget_for(category, base) is base, "callers apply this unconditionally"


def test_budget_for_is_identity_when_the_reserve_already_matches():
    base = MissionBudget(answer_reserve_s=CATEGORY_ANSWER_RESERVE_S[3])
    assert budget_for(3, base) is base


def test_a_category_3_budget_still_validates():
    raised = budget_for(3, MissionBudget())
    assert raised.fallback_reserve_s < raised.answer_reserve_s < raised.question_budget_s


def test_rebudget_preserves_both_anchors():
    clock = MissionClock(MissionBudget(), t0=100.0)
    clock.mark_exploring(160.0)
    swapped = clock.rebudget(budget_for(3, clock.budget))
    assert swapped.t0 == 100.0
    assert swapped.exploring_started
    assert swapped.exploring_elapsed(200.0) == clock.exploring_elapsed(200.0)


def test_rebudget_pulls_the_answer_deadline_in():
    clock = MissionClock(MissionBudget(), t0=100.0)
    clock.mark_exploring(160.0)
    swapped = clock.rebudget(budget_for(3, clock.budget))
    assert swapped.answer_deadline < clock.answer_deadline
    assert swapped.explore_deadline <= swapped.answer_deadline


def test_a_prompt_exploration_is_unaffected_by_the_raised_reserve():
    """The reserve is a backstop, not a shortening: the exploration CAP normally binds first.

    With the shipped numbers a run that starts exploring on time finishes on
    `explore_timeout_s`, well inside even the raised reserve — so raising it costs a healthy
    run nothing at all.
    """
    clock = MissionClock(MissionBudget(), t0=100.0)
    clock.mark_exploring(160.0)                      # 300 s cap ends at 460, reserve at 490
    swapped = clock.rebudget(budget_for(3, clock.budget))
    assert swapped.explore_deadline == clock.explore_deadline


def test_a_slow_start_is_what_the_raised_reserve_actually_cuts():
    """A run whose models loaded slowly is the case the reserve exists for.

    Here the exploration window would run past the point the robot must start driving, and the
    clamp to `answer_deadline` is what stops it — which is why the swap has to happen before
    exploration ends rather than when the answer is due.
    """
    clock = MissionClock(MissionBudget(), t0=100.0)
    clock.mark_exploring(400.0)                      # 300 s cap would end at 700, past the limit
    swapped = clock.rebudget(budget_for(3, clock.budget))
    assert swapped.explore_deadline < clock.explore_deadline
    assert swapped.explore_deadline == swapped.answer_deadline


def test_rebudget_before_any_data_leaves_the_clock_in_warmup():
    clock = MissionClock(MissionBudget(), t0=0.0).rebudget(budget_for(3, MissionBudget()))
    assert not clock.exploring_started
    assert clock.phase_at(1.0) is Phase.WARMUP
