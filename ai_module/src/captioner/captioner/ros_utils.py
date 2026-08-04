"""Small rclpy helpers shared by the captioner and smart_vlm nodes."""
from __future__ import annotations

import time
from typing import Callable, Optional

DEFAULT_CONNECT_TIMEOUT_S = 2.0


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
