"""Category-3 eval harness: when a question is over, and what it is scored against.

Two things here would be silent if wrong. The completion rule decides which pose ends up in
`trajectory[-1]`, and that pose IS the goal constraint — stop one moment early and every
question in the sweep misses its goal while looking like a planner failure. And the scorer
has to be the same one `just cat3-verify` gates the ground truth with, or the harness grades
against rules the GT was never checked against.

The node's methods touch only attributes, so they run unbound against stand-ins rather than
needing a ROS graph — same pattern as sam_mapper's test_session_lifetime.py.

    python -m pytest ai_module/src/smart_vlm/tests/test_cat3_eval.py
"""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[4]

try:                                    # eval_orchestrator imports rclpy; the host may not
    from smart_vlm.eval_orchestrator import (
        CAT3_IDLE_AFTER_WAYPOINT_S, EvalOrchestratorNode, grade_instruction)
except ImportError:                     # runs for real under `just test smart_vlm`
    EvalOrchestratorNode = None
needs_ros = pytest.mark.skipif(EvalOrchestratorNode is None, reason="needs rclpy")


def node(**overrides):
    """An EvalOrchestratorNode-shaped stand-in carrying only what _cat3_done reads."""
    stub = SimpleNamespace(
        category=3,
        trajectory=[],
        waypoints=[],
        instr_status=None,
        _instr_executing=False,
    )
    stub.__dict__.update(overrides)
    return stub


def straight_line(n=200, step=0.05, t0=0.0, dt=0.5):
    """A robot moving steadily — never idle."""
    return [(round(t0 + i * dt, 2), round(i * step, 3), 0.0) for i in range(n)]


def parked(t_end, n=40, dt=0.5):
    """A robot that has stopped: every sample within a few millimetres."""
    return [(round(t_end - (n - 1 - i) * dt, 2), 1.0 + i * 0.001, 0.0) for i in range(n)]


# -- the completion rule -----------------------------------------------------

@needs_ros
def test_execute_then_idle_ends_the_question():
    """The reasoner's route is finished: _reset_to_idle fires after the last _drive_to."""
    stub = node(_instr_executing=True, instr_status="idle")
    assert EvalOrchestratorNode._cat3_done(stub) is True


@needs_ros
def test_answered_does_not_end_the_question():
    """THE case this rule exists for.

    instruction_reasoner._execute publishes "answered" BEFORE its driving loop — it means "I
    have a plan", not "I have arrived". score_instruction takes the goal from trajectory[-1],
    so ending here would grade the pre-execution pose and miss the goal on every question,
    and it would look like the planner's fault rather than the harness's.
    """
    stub = node(_instr_executing=True, instr_status="answered",
                trajectory=straight_line(), waypoints=[(1.0, 5.0, 0.0, 0.0)])
    assert EvalOrchestratorNode._cat3_done(stub) is False


@needs_ros
def test_idle_before_any_execute_does_not_end_the_question():
    """The reasoner publishes "idle" once at startup, so the test is the transition out of
    `execute`, not the value on its own. Without this a question ends before it begins."""
    stub = node(_instr_executing=False, instr_status="idle")
    assert EvalOrchestratorNode._cat3_done(stub) is False


@needs_ros
def test_error_ends_the_question():
    """A planner that gave up still has to release the sweep."""
    stub = node(_instr_executing=True, instr_status="error")
    assert EvalOrchestratorNode._cat3_done(stub) is True


# -- the motion backstop -----------------------------------------------------

@needs_ros
def test_a_stopped_robot_ends_the_question_without_a_status():
    """Backstop for a reasoner that crashes mid-route and never returns to idle. Same rule as
    qa_recorder._done: it sent a waypoint, has sent none since, and has stopped moving."""
    traj = parked(t_end=100.0)
    stub = node(_instr_executing=True, instr_status="answered", trajectory=traj,
                waypoints=[(100.0 - CAT3_IDLE_AFTER_WAYPOINT_S - 5.0, 1.0, 0.0, 0.0)])
    assert EvalOrchestratorNode._cat3_done(stub) is True


@needs_ros
def test_a_moving_robot_does_not_end_the_question():
    """The backstop must not fire mid-route — that is the same early-stop bug by another
    path, and trajectory[-1] would again be a pose the robot was only passing through."""
    traj = straight_line(n=80)
    stub = node(_instr_executing=True, instr_status="answered", trajectory=traj,
                waypoints=[(0.0, 5.0, 0.0, 0.0)])
    assert EvalOrchestratorNode._cat3_done(stub) is False


@needs_ros
def test_no_waypoint_means_not_done():
    """A planner that published nothing has not finished a route; it has not started one.
    Letting the idle rule end it here would score a pure exploration path."""
    stub = node(_instr_executing=True, instr_status="answered",
                trajectory=parked(t_end=100.0), waypoints=[])
    assert EvalOrchestratorNode._cat3_done(stub) is False


# -- agreement with the ground-truth scorer ----------------------------------

def load_score_module():
    path = REPO / "scripts" / "eval" / "score.py"
    spec = importlib.util.spec_from_file_location("cat3_score_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arabic_q05():
    qa = json.load(open(REPO / "data" / "benchmark" / "arabic_room" / "category_3"
                        / "arabic_room_category3_qa.json"))
    return next(q for q in qa["questions"] if q["id"] == "Q05")


def test_the_reference_path_scores_full_marks():
    """The same assertion cat3-verify makes, reachable from the harness side.

    If this drifts from `just cat3-verify`, the sweep is grading against rules the ground
    truth was never checked against.
    """
    q = arabic_q05()
    traj = [[0.0, x, y] for x, y in q["reference_trajectory"]["polyline"]]
    score, note = load_score_module().score_instruction({"trajectory": traj}, q["gt"])
    assert score == pytest.approx(6.0), note


def test_a_path_that_stops_short_loses_exactly_the_goal():
    """Constraints are worth max/(n+1) each. Dropping only the final approach must cost the
    goal's share and nothing else — that is what makes a partial score readable."""
    q = arabic_q05()
    full = q["reference_trajectory"]["polyline"]
    goal_xy = q["gt"]["goal"]["center"][:2]
    radius = q["gt"]["goal"].get("radius", 1.5)
    short = [[0.0, x, y] for x, y in full if math.dist((x, y), goal_xy) > radius]
    assert short, "test premise: some of the path lies outside the goal radius"

    scorer = load_score_module()
    n_cons = len(q["gt"]["pass_near"]) + 1
    per = 6.0 / n_cons
    score, note = scorer.score_instruction({"trajectory": short}, q["gt"])
    assert "goal missed" in note
    assert score == pytest.approx(6.0 - per, abs=1e-6)


def test_an_empty_trajectory_scores_zero_and_says_so():
    """A planner that never moved must read as 'no trajectory', not as a low score — the two
    call for completely different fixes."""
    score, note = load_score_module().score_instruction({"trajectory": []}, arabic_q05()["gt"])
    assert score == 0.0 and note == "no trajectory"


# -- the graded row ----------------------------------------------------------

@needs_ros
def test_grade_instruction_reports_the_final_pose_and_the_time_limit():
    """`predicted` is where the robot actually finished, because that is what the goal
    constraint is scored on. `over_time` flags the 10-minute limit without touching the
    score: README.md says exceeding it 'incurs a penalty' but never says how much."""
    q = arabic_q05()
    traj = [(0.0, x, y) for x, y in q["reference_trajectory"]["polyline"]]
    stub = node(trajectory=traj, waypoints=[(1.0, 2.5, 0.0, 0.0)])

    row = grade_instruction(stub, q, elapsed=120.0)
    assert row["score"] == pytest.approx(6.0)
    assert row["correct"] is True
    assert row["predicted"] == [traj[-1][1], traj[-1][2]]
    assert row["over_time"] is False

    assert grade_instruction(stub, q, elapsed=601.0)["over_time"] is True
