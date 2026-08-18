"""Small rclpy helpers shared by the captioner and smart_vlm nodes."""
from __future__ import annotations

import contextlib
import signal
import time
from typing import Callable, Optional

DEFAULT_CONNECT_TIMEOUT_S = 2.0


@contextlib.contextmanager
def shutdown_guard():
    """Run node teardown without a second Ctrl-C turning it into a traceback.

    Every node here already catches KeyboardInterrupt around spin(), but the cleanup
    itself runs in a `finally:` that nothing protects. On a launch teardown a process
    gets SIGINT twice — once because the terminal delivers it to the whole foreground
    process group, and again because `ros2 launch` signals each child explicitly — so
    the second one lands mid-`destroy_node()` and escapes main(). The result is a wall
    of KeyboardInterrupt tracebacks out of rclpy's service cleanup on an ordinary,
    successful Ctrl-C, which is alarming and, across a per-question eval sweep, makes
    every normal teardown look like a crash.

    Ignoring SIGINT for the duration of cleanup addresses the cause rather than the
    symptom; the KeyboardInterrupt catch covers a signal that arrived a moment before
    the handler was installed. The previous handler is deliberately NOT restored — the
    process is exiting, and restoring it could let a queued signal fire after the
    guard, which is the very traceback this exists to prevent.

    Other exceptions propagate: a real bug in teardown should still be visible.
    """
    with contextlib.suppress(ValueError):  # not the main thread: nothing to install
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    except KeyboardInterrupt:
        pass


def wait_for_subscriber(
    publisher,
    timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    spin: Optional[Callable[[float], None]] = None,
    poll_s: float = 0.05,
) -> bool:
    """Block until `publisher` has at least one subscriber, or `timeout_s` passes.

    Publishing immediately after create_publisher() drops the message: discovery
    has not matched the remote subscription yet. Every request/response exchange
    here is one-shot, so a dropped first message means waiting out a client
    timeout rather than losing one sample of a stream.

    `spin` is the caller's pump (e.g. `lambda t: rclpy.spin_once(node, timeout_sec=t)`)
    for single-threaded nodes that must service their own callbacks while waiting.
    Nodes on a MultiThreadedExecutor leave it None and simply sleep.

    Returns True if a subscriber appeared. A False return is informational — the
    caller normally publishes anyway and lets its own response timeout decide.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if publisher.get_subscription_count() > 0:
            return True
        if spin is not None:
            spin(poll_s)
        else:
            time.sleep(poll_s)
    return publisher.get_subscription_count() > 0
