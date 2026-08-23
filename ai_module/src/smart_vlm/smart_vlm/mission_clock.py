"""The 10-minute mission budget, as pure arithmetic (no ROS / GPU imports).

The challenge gives each question a single wall-clock budget that starts at system
startup and covers exploration *and* answering (README 'Timing'). smart_vlm is
relaunched per question, so one node lifetime == one MissionClock.

Splitting this out of the node keeps every deadline decision testable without a ROS
graph — the previous inline version had two competing branches, one of which was
unreachable with the default parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Phase", "MissionBudget", "MissionClock"]


class Phase(str, Enum):
    WARMUP = "warmup"        # models loading; the bag/sim has not been released yet
    EXPLORING = "exploring"  # sensor data flowing, perception collecting
    ANSWERING = "answering"  # exploration closed, reasoner working
    FALLBACK = "fallback"    # out of time — publish a guess rather than nothing
    EXPIRED = "expired"      # past the hard limit


@dataclass(frozen=True)
class MissionBudget:
    """Deadline offsets, all in seconds.

    The reserves are measured back from the hard limit, so shrinking the budget
    automatically pulls the answering and fallback deadlines in with it.
    """

    question_budget_s: float = 600.0    # challenge hard limit, includes model load
    #: Exploration window, measured from first data. A CAP, not a target: exploration also
    #: ends the moment target_explorer reports every target label found and covered, which is
    #: the normal exit on a small scene. 120 s was not enough to circle three objects, and
    #: since explore_deadline is clamped to answer_deadline anyway, raising it cannot eat the
    #: answer path -- it only stops the clock being the thing that decides coverage.
    explore_timeout_s: float = 300.0
    answer_reserve_s: float = 90.0      # T-90: stop exploring no matter what
    fallback_reserve_s: float = 30.0    # T-30: publish a best guess

    def __post_init__(self) -> None:
        if self.fallback_reserve_s >= self.answer_reserve_s:
            raise ValueError(
                "fallback_reserve_s must be smaller than answer_reserve_s "
                f"(got {self.fallback_reserve_s} >= {self.answer_reserve_s})"
            )
        if self.answer_reserve_s >= self.question_budget_s:
            raise ValueError(
                "answer_reserve_s must be smaller than question_budget_s "
                f"(got {self.answer_reserve_s} >= {self.question_budget_s})"
            )


class MissionClock:
    """One question's deadlines. Feed it a monotonic clock, never wall time.

    `t0` is node construction — which is system startup, since the node is relaunched
    per question. Model loading therefore spends the shared budget (honest, because the
    challenge charges us for it) without shortening the exploration window, which is
    anchored separately at `mark_exploring()`.
    """

    def __init__(self, budget: MissionBudget, t0: float) -> None:
        self.budget = budget
        self.t0 = t0
        self._t_exploring: float | None = None

    # -- anchors -----------------------------------------------------------

    def mark_exploring(self, now: float) -> None:
        """Record when sensor data first arrived. Idempotent — the first call wins."""
        if self._t_exploring is None:
            self._t_exploring = now

    @property
    def exploring_started(self) -> bool:
        return self._t_exploring is not None

    def elapsed(self, now: float) -> float:
        return now - self.t0

    def exploring_elapsed(self, now: float) -> float:
        """Seconds actually spent exploring, which is NOT elapsed(): the window opens on the
        first sensor data after arming, so model load time sits outside it. 0.0 before then."""
        if self._t_exploring is None:
            return 0.0
        return now - self._t_exploring

    # -- deadlines ---------------------------------------------------------

    @property
    def answer_deadline(self) -> float:
        """T-90. Exploration is over by here whatever else has happened."""
        return self.t0 + self.budget.question_budget_s - self.budget.answer_reserve_s

    @property
    def fallback_deadline(self) -> float:
        """T-30. Publish a guess rather than nothing — partial credit beats silence."""
        return self.t0 + self.budget.question_budget_s - self.budget.fallback_reserve_s

    @property
    def hard_deadline(self) -> float:
        return self.t0 + self.budget.question_budget_s

    @property
    def explore_deadline(self) -> float:
        """When to close exploration and hand over to the reasoner.

        Anchored at `mark_exploring()` rather than `t0` so a slow model load does not
        eat the exploration window, and clamped to `answer_deadline` so it can never
        push past T-90. Before any data has arrived the answer deadline is all we have.
        """
        if self._t_exploring is None:
            return self.answer_deadline
        return min(self._t_exploring + self.budget.explore_timeout_s, self.answer_deadline)

    # -- state -------------------------------------------------------------

    def phase_at(self, now: float) -> Phase:
        if now >= self.hard_deadline:
            return Phase.EXPIRED
        if now >= self.fallback_deadline:
            return Phase.FALLBACK
        if self._t_exploring is None:
            return Phase.WARMUP
        if now >= self.explore_deadline:
            return Phase.ANSWERING
        return Phase.EXPLORING

    def snapshot(self, now: float) -> dict:
        """Deadlines as seconds-from-now, for the status topic and logs."""
        return {
            "phase": self.phase_at(now).value,
            "elapsed_s": round(self.elapsed(now), 1),
            "explore_in_s": round(self.explore_deadline - now, 1),
            "fallback_in_s": round(self.fallback_deadline - now, 1),
            "budget_left_s": round(self.hard_deadline - now, 1),
        }
