"""SAM 3 must run ONE session for a whole question.

That is not a performance preference. `ObjMapper.update_map` associates a detection to an
existing 3D object by `obj_id in single_obj.obj_id`, and those ids are only stable within a
session -- so a mid-question reset silently re-numbers every object and the map loses the link
between "the sofa at frame 10" and "the sofa at frame 200".

The invariant held only by accident before these tests: both reasoners publish
/sam3/set_prompts and smart_vlm.launch gives them the same run_id, so a duplicate arm restarted
the session while map_node (seeing an unchanged run_id) kept its map.

Nothing here needs torch, weights or ROS -- the methods under test touch only attributes, so
they are exercised unbound against stand-ins, the same pattern as test_backend_config.py.

    python -m pytest ai_module/src/sam_mapper/tests/test_session_lifetime.py
"""
from types import SimpleNamespace

import pytest

from sam_mapper.sam3_backend import Sam3Backend

try:                                    # sam_node imports rclpy; the backend half does not
    from sam_mapper.sam_node import SamNode
except ImportError:                     # bare host -- these run under `just test sam_mapper`
    SamNode = None
needs_ros = pytest.mark.skipif(SamNode is None, reason="sam_node needs rclpy")


def _backend():
    """A Sam3Backend-shaped stand-in whose processor records session creations."""
    created = []

    def init_video_session(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(id=len(created))

    processor = SimpleNamespace(init_video_session=init_video_session,
                                add_text_prompt=lambda session, prompts: None)
    backend = SimpleNamespace(processor=processor, session=None, prompts=[], session_epoch=0,
                              cfg={}, device="cpu", dtype=None, log=lambda *_: None)
    # set_prompts() calls self.reset(); bind the real one so the pair is exercised together.
    backend.reset = lambda: Sam3Backend.reset(backend)
    return backend, created


def test_reset_numbers_each_session():
    backend, created = _backend()
    Sam3Backend.reset(backend)
    assert backend.session_epoch == 1 and len(created) == 1
    Sam3Backend.reset(backend)
    assert backend.session_epoch == 2 and len(created) == 2


def test_set_prompts_starts_exactly_one_session():
    """Arming is the FIRST and only session a question should ever see.

    sam_node used to call set_prompts([]) at boot even when unarmed, so a question logged
    "#1 (prompts=[])" then "#2" with the real prompts. Nothing consumed the boot session --
    unarmed the node drops every frame -- and it made the one-session-per-question invariant
    read as two. The pre-state assertion is what pins that: a backend that was never armed
    holds no session at all.
    """
    backend, created = _backend()
    assert backend.session_epoch == 0 and backend.session is None and created == []

    Sam3Backend.set_prompts(backend, ["sofa", "window"])
    assert backend.session_epoch == 1
    assert backend.prompts == ["sofa", "window"]
    assert len(created) == 1


# -- the node-level invariant ------------------------------------------------
#
# Marked needs_ros: sam_node imports rclpy, so these run under `just test sam_mapper` (in the
# container) and skip on a bare host. The methods themselves touch nothing but attributes.

def _node(**overrides):
    """A SamNode-shaped stand-in for the two methods that can restart a session."""
    backend, created = _backend()
    node = SimpleNamespace(
        backend=SimpleNamespace(reset=lambda: created.append("reset"), session_epoch=1),
        last_frame_stamp=None,
        id_offset=0,
        max_seen_id=-1,
        best_view_collector=None,
        TIME_JUMP_TOLERANCE=SamNode.TIME_JUMP_TOLERANCE,
        # Production default. Every caller that wants the bag-loop behaviour opts in, which
        # is the same way round as the yaml.
        reset_session_on_time_jump=False,
        log=lambda *_: None,
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: None,
                                           error=lambda *a, **k: None),
    )
    node.__dict__.update(overrides)
    return node, created


@needs_ros
def test_frames_moving_forward_never_reset_the_session():
    """THE invariant: a bag that plays forward once must produce exactly one session."""
    node, created = _node()
    for stamp in [100.0, 100.2, 100.5, 101.0, 103.0, 110.0, 160.0, 215.0]:
        SamNode._handle_time_jump(node, stamp)
    assert created == []
    assert node.id_offset == 0


@needs_ros
def test_a_stamp_that_regresses_within_tolerance_is_not_a_loop():
    """Small out-of-order jitter must not be mistaken for a new lap."""
    node, created = _node()
    SamNode._handle_time_jump(node, 100.0)
    SamNode._handle_time_jump(node, 99.5)          # 0.5s < TIME_JUMP_TOLERANCE
    assert created == []


@needs_ros
def test_a_real_bag_loop_still_resets_and_renumbers():
    """The `--loop` path must keep working -- this is why the reset is not simply removed."""
    node, created = _node(max_seen_id=41, reset_session_on_time_jump=True)
    SamNode._handle_time_jump(node, 200.0)
    SamNode._handle_time_jump(node, 100.0)         # 100s backwards: a new lap
    assert created == ["reset"]
    assert node.id_offset == 42                    # past max_seen_id, so ids cannot collide


@needs_ros
def test_the_sim_never_resets_however_far_the_stamp_regresses():
    """Sim-only testing: there is no lap, so no regression may restart the session.

    The live sim reorders /camera/image by a second or two. Over one 13-scene sweep that
    tripped 215 resets, every jump between 1.0s and 2.8s, and each one renumbered every track
    -- track ids per mapped object went 1.37 with no resets to 4.59 at 6-20, and one `chair`
    came back as eight ids for the world merge to reassemble.

    Deliberately tested with a 100s jump, not a 2s one: raising TIME_JUMP_TOLERANCE would
    pass a 2s case while leaving a threshold guarding a case that cannot occur. Off means off.
    """
    node, created = _node(max_seen_id=41)
    SamNode._handle_time_jump(node, 200.0)
    SamNode._handle_time_jump(node, 100.0)
    assert created == []
    assert node.id_offset == 0                     # ids continue, so tracks stay linked
    assert node.last_frame_stamp == 100.0          # ...but the clock still follows the stream
