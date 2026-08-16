"""A minimal rclpy client for the resident `qwen_vqa_server`, for offline replay scripts.

`cat1_bench` / `cat2_bench` deliberately avoid the full smart_vlm ROS graph (no SAM, no
map, no reasoner) so a replay is minutes rather than hours. But the *local* Qwen backend
only exists behind the `/qwen_vqa/request` topic pair (see `qwen_vqa_protocol`), so using
it from a plain CLI still means holding one lightweight rclpy node alive long enough to
round-trip requests -- this is exactly that node, and nothing else.

Requires `just vqa-up` already running: this client only talks to the server, it never
starts it.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Callable, Optional, Sequence

from captioner.qwen_vqa_protocol import REQUEST_TOPIC, RESPONSE_TOPIC, STATUS_TOPIC, vqa_image_fields


class LocalVqaClient:
    """Context manager wrapping one rclpy node: `client.ask_vqa` is the local backend's transport.

        with LocalVqaClient(log=log) as client:
            backend = make_backend("local", ask_vqa=client.ask_vqa, log=log)
    """

    def __init__(self, timeout_s: float = 120.0, ready_timeout_s: float = 30.0,
                 log: Optional[Callable[[str], None]] = None):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._String = String
        self._timeout_s = timeout_s
        self._log = log or (lambda _msg: None)

        # A bench run started from a plain CLI has never called rclpy.init(); a caller
        # that already owns a ROS context (e.g. under a test harness) should keep it.
        self._own_rclpy = not rclpy.ok()
        if self._own_rclpy:
            rclpy.init()

        self._node = Node("vqa_bench_client")
        latch_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               history=HistoryPolicy.KEEP_LAST)

        self._qwen_ready = False
        self._vqa_lock = threading.Lock()
        self._wait_id: Optional[str] = None
        self._response: Optional[dict] = None
        self._event = threading.Event()

        self._node.create_subscription(String, STATUS_TOPIC, self._on_status, latch_qos)
        self._node.create_subscription(String, RESPONSE_TOPIC, self._on_response, 10)
        self._pub_req = self._node.create_publisher(String, REQUEST_TOPIC, 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        deadline = time.time() + ready_timeout_s
        while not self._qwen_ready and time.time() < deadline:
            time.sleep(0.1)
        if not self._qwen_ready:
            self._log(f"qwen_vqa_server not reporting ready after {ready_timeout_s:.0f}s "
                       "-- is `just vqa-up` running?")

    def _on_status(self, msg) -> None:
        self._qwen_ready = msg.data.strip() == "ready"

    def _on_response(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._vqa_lock:
            if self._wait_id is None or payload.get("id") != self._wait_id:
                return
            self._response = payload
            self._event.set()

    def ask_vqa(self, question: str, images: Sequence = (), max_new_tokens: int = 64,
                mode: str = "freeform") -> str:
        """The transport `QwenRosBackend` calls: one request, the answer text back."""
        if not self._qwen_ready:
            raise RuntimeError("qwen_vqa_server not ready -- start it with `just vqa-up`")

        req_id = str(uuid.uuid4())
        payload = {"id": req_id, "question": question, "max_new_tokens": max_new_tokens,
                   "mode": mode, **vqa_image_fields(images)}
        with self._vqa_lock:
            self._wait_id = req_id
            self._response = None
            self._event.clear()

        from captioner.ros_utils import wait_for_subscriber
        wait_for_subscriber(self._pub_req)
        self._pub_req.publish(self._String(data=json.dumps(payload)))
        if not self._event.wait(timeout=self._timeout_s):
            raise TimeoutError(f"no VQA response within {self._timeout_s:.0f}s")

        with self._vqa_lock:
            response = self._response or {}
            self._wait_id = None
        if response.get("error"):
            raise RuntimeError(f"VQA error: {response['error']}")
        return response.get("answer") or ""

    def close(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()
        if self._own_rclpy:
            import rclpy
            rclpy.shutdown()

    def __enter__(self) -> "LocalVqaClient":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False
