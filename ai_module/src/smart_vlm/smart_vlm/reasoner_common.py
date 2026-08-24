"""Shared extract / VQA / obj-map helpers and the reasoner lifecycle node."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from captioner.paths import secure_path
from captioner.qwen_vqa_protocol import vqa_image_fields
from captioner.ros_utils import shutdown_guard, wait_for_subscriber
from captioner.vlm_backends import VLMError, make_backend
from captioner.vlm_backends.constants import TARGET_EXTRACT_BACKEND, VLM_BACKEND
from captioner.vlm_backends.schemas import TargetList
from sam_mapper.best_view import sanitize_run_id
from smart_vlm.numerical_utils import EXTRACT_SYSTEM, clean_targets
from smart_vlm.question import QuestionType, classify

MAP_WAIT_S = 5.0
MAP_POLL_S = 0.25


def latch_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def make_run_id(fixed: str, question: str, prefix: str) -> str:
    if fixed:
        return sanitize_run_id(fixed)
    safe_q = sanitize_run_id(question[:48], fallback="q")
    return f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_q}"[:80]


def parse_gt_targets(raw: str) -> Optional[list[str]]:
    try:
        targets = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return list(targets) if targets else None


def publish_sam_prompts(pub, prompts: Sequence[str], run_id: Optional[str]) -> None:
    wait_for_subscriber(pub)
    pub.publish(String(data=json.dumps({"prompts": list(prompts), "run_id": run_id})))


def extract_nouns(
    backend, question: str, heuristic: Callable[[str], list[str]], log,
) -> tuple[list[str], str, Optional[list[str]]]:
    """Nouns to arm SAM with: (prompts, source, raw model list)."""
    reply: Optional[list[str]] = None
    try:
        result = backend.ask(EXTRACT_SYSTEM, question, [], TargetList, lite=True)
        reply = list(result.targets)
        if targets := clean_targets(result.targets):
            return targets, backend.name, reply
        log(f"extract returned nothing usable: {reply!r}")
    except VLMError as exc:
        log(f"extract failed: {exc}")
    targets = heuristic(question)
    log(f"falling back to parsed targets={targets}")
    return targets, "heuristic", reply


def map_objects_only(payload: dict | None) -> dict:
    """A raw obj_map payload with the non-object entries dropped.

    obj_map.json carries a `_schema` entry alongside the track ids (see
    `sam_mapper.object_mapper.SCHEMA_KEY`). Stripping it here, at the two points a raw map
    reaches a reasoner, keeps every consumer downstream able to treat the mapping as
    "track id -> object" — otherwise `len(raw_map)` over-counts by one and a map holding
    nothing but the marker reads as non-empty.
    """
    return {k: v for k, v in (payload or {}).items()
            if isinstance(v, dict) and not str(k).startswith("_")}


def read_obj_map(best_dir, log, wait_s: float = MAP_WAIT_S,
                 poll_s: float = MAP_POLL_S) -> dict:
    if not best_dir:
        return {}
    try:
        run_dir = best_dir if isinstance(best_dir, Path) else secure_path(best_dir)
    except (ValueError, PermissionError, RuntimeError) as exc:
        log(f"best_view_dir rejected: {exc}")
        return {}
    deadline = time.monotonic() + wait_s
    path = Path(run_dir) / "obj_map.json"
    while time.monotonic() < deadline and rclpy.ok():
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return map_objects_only(json.load(handle))
            except json.JSONDecodeError:
                pass
        time.sleep(poll_s)
    log(f"no readable obj_map.json in {run_dir} after {wait_s:.0f}s")
    return {}


class VqaBridge:
    """One-shot /qwen_vqa request-response used by every reasoner."""

    def __init__(self, pub_request, timeout_s: float):
        self.pub_request = pub_request
        self.timeout_s = timeout_s
        self.ready = False
        self._lock = threading.Lock()
        self._wait_id: Optional[str] = None
        self._response: Optional[dict] = None
        self._event = threading.Event()

    def on_status(self, msg: String) -> None:
        self.ready = (msg.data or "").strip() == "ready"

    def on_response(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            if self._wait_id is None or payload.get("id") != self._wait_id:
                return
            self._response = payload
            self._event.set()

    def ask(self, question: str, images: Sequence[str] = (),
            max_new_tokens: int = 64, mode: str = "freeform") -> str:
        if not self.ready:
            deadline = time.time() + min(30.0, self.timeout_s)
            while not self.ready and time.time() < deadline and rclpy.ok():
                time.sleep(0.1)
        if not self.ready:
            raise RuntimeError("qwen_vqa_server not ready")

        req_id = str(uuid.uuid4())
        payload = {
            "id": req_id,
            "question": question,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            **vqa_image_fields(images),
        }
        with self._lock:
            self._wait_id = req_id
            self._response = None
            self._event.clear()

        wait_for_subscriber(self.pub_request)
        self.pub_request.publish(String(data=json.dumps(payload)))
        if not self._event.wait(timeout=self.timeout_s):
            raise TimeoutError(f"No VQA response within {self.timeout_s:.0f}s")

        with self._lock:
            response = self._response or {}
            self._wait_id = None
        if response.get("error"):
            raise RuntimeError(f"VQA error: {response['error']}")
        return response.get("answer") or ""


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    tmp.replace(manifest_path)


def spin_reasoner(node) -> None:
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        with shutdown_guard():
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


class ReasonerNode(Node):
    """Extract nouns, arm SAM, wait for explore_done. Subclasses publish the answer."""

    QUESTION_TYPE: QuestionType
    STATUS_TOPIC: str
    ADHOC_PREFIX: str
    ANSWER_BACKEND = True
    PHASE_IDLE = "idle"
    PHASE_EXTRACT = "extract"
    PHASE_PROMPTS = "prompts"
    PHASE_EXPLORE = "explore"
    PHASE_ANSWER = "answer"

    def __init__(self, node_name: str, extra_params: Optional[dict[str, Any]] = None):
        super().__init__(node_name)
        self.declare_parameter("run_id", "")
        self.declare_parameter("vqa_timeout_s", 120.0)
        for name, default in (extra_params or {}).items():
            self.declare_parameter(name, default)

        self.fixed_run_id = str(self.get_parameter("run_id").value).strip()
        self.vqa_timeout_s = float(self.get_parameter("vqa_timeout_s").value)
        self.backend_name = VLM_BACKEND

        self._lock = threading.Lock()
        self._phase = self.PHASE_IDLE
        self._question: Optional[str] = None
        self._current_gt_targets: Optional[list[str]] = None
        self._active_gt_targets: Optional[list[str]] = None
        self._run_id: Optional[str] = None
        self._prompts: Optional[list[str]] = None
        self._target_source = "unknown"
        self._extract_reply: Optional[list[str]] = None
        self._crop_dir: Optional[str] = None

        self._qos = latch_qos()
        self._cb = MutuallyExclusiveCallbackGroup()
        self.create_subscription(String, "/gt_target_objects", self._on_gt_targets,
                                 self._qos, callback_group=self._cb)
        self.create_subscription(String, "/challenge_question", self._on_question, 10,
                                 callback_group=self._cb)
        self.create_subscription(String, "/pipeline/explore_done", self._on_explore_done,
                                 10, callback_group=self._cb)
        self.create_subscription(String, "/sam3/best_view_dir", self._on_crop_dir,
                                 self._qos, callback_group=self._cb)

        self.pub_prompts = self.create_publisher(String, "/sam3/set_prompts", 10)
        self.pub_vqa_req = self.create_publisher(String, "/qwen_vqa/request", 10)
        self.pub_status = self.create_publisher(String, self.STATUS_TOPIC, self._qos)

        self.vqa = VqaBridge(self.pub_vqa_req, self.vqa_timeout_s)
        self.create_subscription(String, "/qwen_vqa/response", self.vqa.on_response, 10,
                                 callback_group=self._cb)
        self.create_subscription(String, "/qwen_vqa/status", self.vqa.on_status,
                                 self._qos, callback_group=self._cb)

        extract_name = TARGET_EXTRACT_BACKEND
        if self.ANSWER_BACKEND:
            self.backend = make_backend(
                self.backend_name, ask_vqa=self.vqa.ask, log=self.get_logger().info)
            self.extract_backend = (
                self.backend if extract_name == self.backend_name
                else make_backend(
                    extract_name, ask_vqa=self.vqa.ask, log=self.get_logger().info)
            )
        else:
            self.backend = None
            self.extract_backend = make_backend(
                extract_name, ask_vqa=self.vqa.ask, log=self.get_logger().info)

        self._publish_status("idle")

    def _heuristic_targets(self, question: str) -> list[str]:
        raise NotImplementedError

    def _begin_answer(self, explore_msg: String, snap: dict) -> None:
        raise NotImplementedError

    def _publish_status(self, text: str) -> None:
        if not rclpy.ok():
            return
        self.pub_status.publish(String(data=text))

    def _on_gt_targets(self, msg: String) -> None:
        with self._lock:
            self._current_gt_targets = parse_gt_targets(msg.data)

    def _on_crop_dir(self, msg: String) -> None:
        with self._lock:
            self._crop_dir = msg.data

    def _on_question(self, msg: String) -> None:
        with self._lock:
            if self._phase != self.PHASE_IDLE:
                return
            q_text = msg.data.strip()
            if classify(q_text) is not self.QUESTION_TYPE:
                return
            self._phase = self.PHASE_EXTRACT
            self._question = q_text
            self._active_gt_targets = self._current_gt_targets
            self._run_id = make_run_id(self.fixed_run_id, q_text, self.ADHOC_PREFIX)
            self._prompts = None
            self._extract_reply = None
        self.get_logger().info(f"QUESTION: {self._question}")
        self._publish_status("extract")
        threading.Thread(target=self._run_extract_and_set_prompts, daemon=True).start()

    def _on_explore_done(self, msg: String) -> None:
        with self._lock:
            if self._phase != self.PHASE_EXPLORE:
                return
            self._phase = self.PHASE_ANSWER
            snap = {
                "question": self._question,
                "run_id": self._run_id,
                "crop_dir": self._crop_dir,
                "prompts": self._prompts,
                "source": self._target_source,
                "extract_reply": self._extract_reply,
            }
        self._begin_answer(msg, snap)

    def _run_extract_and_set_prompts(self) -> None:
        try:
            with self._lock:
                question, run_id = self._question, self._run_id
                gt_targets = self._active_gt_targets

            if gt_targets is not None:
                prompts, source, reply = list(gt_targets), "gt", None
            else:
                prompts, source, reply = extract_nouns(
                    self.extract_backend, question, self._heuristic_targets,
                    self.get_logger().warn)
            if not prompts:
                raise RuntimeError("Could not extract target objects from question")

            with self._lock:
                self._prompts = prompts
                self._target_source = source
                self._extract_reply = reply
                self._phase = self.PHASE_PROMPTS

            self._publish_status("prompts")
            self.get_logger().info(f"extracted targets ({source}): {prompts}")
            publish_sam_prompts(self.pub_prompts, prompts, run_id)

            with self._lock:
                self._phase = self.PHASE_EXPLORE
            self._publish_status("explore")
            self.get_logger().info("prompts applied; waiting for /pipeline/explore_done")
        except Exception as exc:  # noqa: BLE001 — keep the node alive
            self.get_logger().error(
                f"extract/set_prompts failed: {type(exc).__name__}: {exc}")
            self._reset_to_idle("error")

    def _reset_to_idle(self, status: str) -> None:
        with self._lock:
            self._phase = self.PHASE_IDLE
            self._question = None
            self._run_id = None
            self._prompts = None
            self._extract_reply = None
        self._publish_status(status)
