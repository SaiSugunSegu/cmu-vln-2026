"""SAM 3 2D detector node: /camera/image in, tracked 2D detections out.

Split out from the combined node (docs/M2_perception.md 3.6-split) after profiling showed
SAM3 inference genuinely runs ~1-3 s/frame in isolation but ballooned to 20-80 s/frame when
sharing a process with /state_estimation (50 Hz) and /registered_scan callbacks — GIL
contention (confirmed via nvidia-smi dmon: GPU busy only in brief scattered bursts), not GPU
or model slowness. Running SAM3 in its own process gives it its own GIL, immune to another
node's callback load.

Publishes:
    /annotated_image     sensor_msgs/Image   BGR debug overlay (mask/box/label), unchanged
    /sam3/instance_map   sensor_msgs/Image   mono16, pixel = encode_instance_id(obj_id)
    /sam3/detections     std_msgs/String     JSON {stamp, entries: [{id, label, confidence, bbox}]}
                                             (String has no header, so the stamp is embedded here,
                                              matching /sam3/instance_map's)
    /sam3/status         std_msgs/String     latched "loading" | "awaiting_prompts" |
                                             "ready" | "setting_prompts"
    /sam3/prompts_ack    std_msgs/String     JSON ack after /sam3/set_prompts
    /sam3/best_view_dir  std_msgs/String     latched path of current best-view run dir

Subscribes:
    /sam3/set_prompts    std_msgs/String     JSON {"prompts": [...], "run_id": "..."}
    /pipeline/explore_done std_msgs/String   obj_map.json has settled — label the best-view
                                             overlays with its ids. Optional; a bag loop and
                                             shutdown trigger the same pass.

With `wait_for_prompts:=true` the node ignores the config's `objects:` and boots UNARMED:
weights still load (the slow part, ~60 s), but no frame is processed and no best-view run
directory is created until the first /sam3/set_prompts arrives. The category-1 pipeline
needs this — its prompts come from the question, so anything detected beforehand would be
against the config's placeholder objects and pollute the run.

map_node (map_node.py) subscribes to the latter two and reconstructs each object's mask via
`id_map == encode_instance_id(id)` — the existing 5-key to_detections() contract, unchanged,
so ObjMapper.update_map() needs no changes.

  ros2 launch sam_mapper sam_node.launch
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_mapper.annotate import annotate_frame
from sam_mapper.best_view import BestViewCollector, BestViewConfig
from sam_mapper.challenge_marker import track_to_map_id
from sam_mapper.detections import PromptTable, build_id_map, to_detections
from sam_mapper.node_base import WorkerNodeMixin, run_node
from sam_mapper.sam3_backend import make_backend


class SamNode(WorkerNodeMixin, Node):

    HEARTBEAT_S = 5.0
    SLOW_STAGE_S = 5.0
    VERBOSE_FIRST = 3
    TIME_JUMP_TOLERANCE = 1.0        # seconds backwards before we call it a new lap

    def __init__(self, config: dict):
        super().__init__('sam_node')
        self.bridge = CvBridge()

        runtime = config.get('runtime', {})
        self.publish_annotated = runtime.get('publish_annotated', True)
        self.log_every_n = int(runtime.get('log_every_n_frames', 20))
        self.verbose_objects = bool(runtime.get('verbose_objects', False))
        # How much of the node's frame is NOT SAM 3. Costs a cuda sync per stage.
        self.profile = bool(runtime.get('profile', False))

        self.declare_parameter('wait_for_prompts', False)
        self.wait_for_prompts = bool(self.get_parameter('wait_for_prompts').value)

        # `armed` == "has prompts worth spending inference on". Unarmed, the node holds
        # loaded weights and an empty session, and drops every frame.
        if self.wait_for_prompts:
            self.prompt_table = None
            self.armed = False
            self.log("wait_for_prompts: ignoring config objects, waiting for "
                     "/sam3/set_prompts before processing any frame")
        else:
            self.prompt_table = PromptTable(config['objects'])
            self.armed = True
            self.log(f"prompts ({len(self.prompt_table.prompts)}): {self.prompt_table.prompts}")

        self.best_view_cfg = config.get('save_best_target_view_images', {})
        self.best_view_collector = None

        # -- ROS interface (status first so clients can wait through weight load) --
        def group():
            return MutuallyExclusiveCallbackGroup()

        latch_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.status_pub = self.create_publisher(String, '/sam3/status', latch_qos)
        self._publish_status('loading')

        # Unarmed: no collector yet, so no stray boot-time run dir under output_dir.
        if self.armed and self.best_view_cfg.get('enabled', False):
            self.best_view_collector = BestViewCollector(
                BestViewConfig.from_dict(self.best_view_cfg, self.prompt_table), log=self.log)

        self.log("loading SAM 3 (first run downloads weights, this can take a while) ...")
        self.backend = make_backend(config['sam3'], log=self.log, profile=self.profile)
        # Empty prompts are fine: both backends treat an empty prompt list as a valid,
        # promptless session rather than an error.
        self.backend.set_prompts(self.prompt_table.prompts if self.armed else [])
        self.log("SAM 3 ready" if self.armed else "SAM 3 weights loaded — awaiting prompts")

        # -- buffers ---------------------------------------------------------
        self.frame_lock = threading.Lock()
        self.latest_frame = None                # newest undecoded sensor_msgs/Image
        # Guards prompt_table / backend session / best_view_collector against /sam3/set_prompts.
        self.prompt_lock = threading.Lock()

        self.frames_in = 0
        self.frames_done = 0
        self.frames_dropped = 0

        # Bag-loop handling, scoped to this node's own state: the SAM3 session and id
        # namespace. map_node has its own copy of the same handling for its odom/cloud buffers.
        self.last_frame_stamp = None
        self.id_offset = 0
        self.max_seen_id = -1

        self.create_subscription(Image, '/camera/image', self.image_callback, 10,
                                 callback_group=group())
        self.create_subscription(String, '/sam3/set_prompts', self._on_set_prompts, 10,
                                 callback_group=group())
        # Optional: without smart_vlm nothing publishes it, and bag loop / shutdown finalize.
        self.create_subscription(String, '/pipeline/explore_done', self._on_explore_done, 10,
                                 callback_group=group())
        self.annotated_pub = self.create_publisher(Image, '/annotated_image', 2)
        self.instance_map_pub = self.create_publisher(Image, '/sam3/instance_map', 2)
        self.detections_pub = self.create_publisher(String, '/sam3/detections', 2)
        self.prompts_ack_pub = self.create_publisher(String, '/sam3/prompts_ack', 10)
        self.best_view_dir_pub = self.create_publisher(String, '/sam3/best_view_dir', latch_qos)

        if self.best_view_collector is not None:
            self.best_view_dir_pub.publish(String(data=self.best_view_collector.run_dir))

        self.create_timer(self.HEARTBEAT_S, self._heartbeat, callback_group=group())

        self._start_worker(self._worker_loop)
        self._publish_status('ready' if self.armed else 'awaiting_prompts')
        self.log('sam_node started')

    def log(self, msg):
        self.get_logger().info(str(msg))

    def _publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))

    # -- callbacks ------------------------------------------------------------

    def image_callback(self, msg: Image):
        """Store the raw message; decode later, in the worker. We drop most frames, so
        decoding on arrival would waste CPU on images that get thrown away."""
        if not self.armed:
            # Not even buffered: the first frame after arming must be a fresh one, not a
            # stale panorama captured before this question's prompts existed.
            return
        with self.frame_lock:
            if self.latest_frame is not None:
                self.frames_dropped += 1
            self.latest_frame = msg
            self.frames_in += 1

    def _take_frame(self):
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
            return frame

    def _nack_prompts(self, error: str) -> None:
        """Report a failed set_prompts and return the node to a serving state.

        Both callers below rely on this: a request that leaves /sam3/status at
        'setting_prompts' with no ack strands every client (category1_reasoner,
        run_cat1_bag_bench) until their own timeouts expire.
        """
        self.get_logger().error(f'/sam3/set_prompts rejected: {error}')
        self.prompts_ack_pub.publish(String(data=json.dumps({
            "ok": False, "error": error, "prompts": [], "run_dir": None,
        })))
        # 'ready' would be a lie for a node that was never armed: it drops every frame,
        # and clients gate on this status before releasing their sensor stream.
        self._publish_status('ready' if self.armed else 'awaiting_prompts')

    def _on_set_prompts(self, msg: String) -> None:
        """Replace SAM text prompts and start a fresh best-view run for the next bag pass.

        Prompts arriving here are plain strings, so every one becomes an instance
        object with its default label — any `label:` override from the YAML
        `objects:` config is not preserved across a set_prompts call.
        """
        try:
            payload = json.loads(msg.data)
            prompts = payload["prompts"]
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("'prompts' must be a non-empty list of strings")
            prompts = [str(p).strip() for p in prompts if str(p).strip()]
            if not prompts:
                raise ValueError("'prompts' has no non-empty strings")
            # Preserve order, drop duplicates (PromptTable rejects duplicate prompt strings).
            seen = set()
            unique = []
            for prompt in prompts:
                if prompt not in seen:
                    seen.add(prompt)
                    unique.append(prompt)
            prompts = unique
            run_id = payload.get("run_id")
            if run_id is not None:
                run_id = str(run_id)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
            self._nack_prompts(str(err))
            return

        objects = [{"prompt": p, "instance": True} for p in prompts]
        self._publish_status('setting_prompts')
        with self.prompt_lock:
            previous_table = self.prompt_table
            previous_collector = self.best_view_collector
            previous_armed = self.armed
            try:
                # PromptTable validates, backend.set_prompts() re-inits the SAM
                # session, and BestViewConfig.from_dict() parses config — any of
                # them can raise on a bad request or a transient CUDA error.
                self.prompt_table = PromptTable(objects)
                self.backend.set_prompts(self.prompt_table.prompts)
                self.id_offset = 0
                self.max_seen_id = -1
                self.last_frame_stamp = None
                with self.frame_lock:
                    self.latest_frame = None

                run_dir = None
                if self.best_view_cfg.get('enabled', False):
                    self.best_view_collector = BestViewCollector(
                        BestViewConfig.from_dict(self.best_view_cfg, self.prompt_table),
                        log=self.log,
                        run_id=run_id,
                    )
                    run_dir = self.best_view_collector.run_dir
                    self.best_view_dir_pub.publish(String(data=run_dir))
                else:
                    self.best_view_collector = None
                # Last, so a raise above can never leave us armed with a half-built table.
                self.armed = True
            except Exception as err:  # noqa: BLE001 — a bad request must not kill the node
                # Roll back so the node keeps detecting with the prompts it had — or
                # stays unarmed, if it never had any.
                self.prompt_table = previous_table
                self.best_view_collector = previous_collector
                self.armed = previous_armed
                try:
                    self.backend.set_prompts(
                        previous_table.prompts if previous_table is not None else [])
                except Exception as restore_err:  # noqa: BLE001
                    self.get_logger().error(
                        f'could not restore previous prompts: {restore_err}')
                self._nack_prompts(f'{type(err).__name__}: {err}')
                return

        ack = {
            "ok": True,
            "error": None,
            "prompts": list(self.prompt_table.prompts),
            "labels": [s.label for s in self.prompt_table.specs],
            "run_id": run_id,
            "run_dir": run_dir,
        }
        self.prompts_ack_pub.publish(String(data=json.dumps(ack)))
        self._publish_status('ready')
        self.log(f'set_prompts ok: {ack["prompts"]} -> {run_dir}')

    # -- best-view finalize -----------------------------------------------------

    def _on_explore_done(self, msg: String) -> None:
        """Exploration is closed, so map_node's obj_map.json has stopped moving.

        Off the executor thread: this waits on another process's file, and blocking a
        callback group would stall the frames queued behind it.
        """
        self.log(f"explore_done ({msg.data or 'ok'}) — finalizing best-view overlays")
        threading.Thread(target=self._finalize_best_views, daemon=True).start()

    def _finalize_best_views(self, wait_s: float | None = None) -> None:
        """Render the overlay copies with the 3D map's object ids drawn on.

        End of run, not per flush: map_node rewrites obj_map.json on every publish, so an
        overlay drawn mid-run would show ids a later world merge renames. Safe to call more
        than once — the collector re-renders only on a changed lookup.
        """
        with self.prompt_lock:
            collector = self.best_view_collector
        if collector is None:
            return

        path = os.path.join(collector.run_dir, 'obj_map.json')
        if wait_s is None:
            wait_s = collector.config.finalize_obj_map_wait_s
        objects = self._read_obj_map(path, wait_s)
        try:
            # None, not {}: "no 3D map at all" makes the track id the only id there is,
            # while an empty map means every instance genuinely failed to reach a box.
            rendered = collector.finalize(None if objects is None else track_to_map_id(objects))
        except Exception:  # noqa: BLE001 — a render fault must not take the node down
            self.get_logger().error(f'best-view finalize failed:\n{traceback.format_exc()}')
            return
        if not rendered:
            # Three callers land here; only the first with a given map costs a render. Logged
            # so a silent second pass reads as a no-op, not as one that failed.
            self.log('best-view overlays already current — nothing re-rendered')

    def _read_obj_map(self, path: str, wait_s: float) -> dict | None:
        """map_node's map, or None if it never landed.

        Polled because map_node is a separate process with no ordering guarantee. A partial
        read cannot happen (it writes a temp file and os.replace()s it), but the file can be
        late, or absent forever when map_node is not in the launch.
        """
        deadline = time.monotonic() + max(wait_s, 0.0)
        while True:
            try:
                with open(path) as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    self.log(f'no readable {path} — labelling crops with SAM track ids')
                    return None
                time.sleep(0.25)

    # -- bag-loop handling ------------------------------------------------------

    def _handle_time_jump(self, stamp: float) -> None:
        """Detect a bag loop on /camera/image and start a clean SAM3 session.

        Only this node's own state needs resetting here: the SAM3 session (ids only mean
        anything within one session) and the id namespace (new ids must never collide with
        ones map_node has already placed in the map).
        """
        if self.last_frame_stamp is None or stamp >= self.last_frame_stamp - self.TIME_JUMP_TOLERANCE:
            self.last_frame_stamp = stamp
            return

        jump = self.last_frame_stamp - stamp
        self.log(f'time jumped backwards {jump:.1f}s (bag loop?) — resetting SAM 3 session; '
                 f'new object ids offset by {self.max_seen_id + 1}')
        self.id_offset = self.max_seen_id + 1
        self.backend.reset()
        self.last_frame_stamp = stamp
        # The id offset above means the same physical object returns under a new id, which
        # the collector would otherwise count as another object to cover.
        if self.best_view_collector:
            self.best_view_collector.on_time_jump()
            # Frozen from here, so draw now rather than trust a shutdown the harness may
            # never deliver as a clean SIGINT. No wait: one full pass is already in the map.
            threading.Thread(target=self._finalize_best_views, args=(0.0,),
                             daemon=True).start()

    # -- worker ---------------------------------------------------------------

    def _worker_loop(self):
        # rclpy.ok() as well as self.running: SIGINT invalidates the context before
        # destroy_node() gets to clear the flag, and publishing into a dead context
        # raises RCLError mid-frame.
        while self.running and rclpy.ok():
            if not self.armed:
                # Set the stage once, not every 50 ms, so stage_since keeps meaning
                # "how long we have been here" for the heartbeat.
                if self.stage != 'awaiting prompts':
                    self._set_stage('awaiting prompts')
                time.sleep(0.05)
                continue
            self._set_stage('waiting')
            msg = self._take_frame()
            if msg is None:
                time.sleep(0.005)
                continue
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = self._stamp_of(msg)

            with self.prompt_lock:
                self._handle_time_jump(stamp)
                try:
                    self._process(image, stamp)
                except Exception as err:              # noqa: BLE001 — one bad frame must not kill the node
                    # A frame takes seconds (SAM3 inference), so SIGINT routinely lands
                    # mid-_process and the publish fails on a dead context. That is a
                    # normal shutdown, not a frame error — leave quietly rather than
                    # logging a traceback nobody should act on.
                    if not rclpy.ok():
                        break
                    # rclpy's logger takes none of logging's kwargs (only throttle /
                    # skip_first / once), so exc_info=True made the handler itself raise
                    # TypeError and kill this thread. Format the traceback into the text.
                    self.get_logger().error(
                        f'frame at {stamp:.3f} failed: {type(err).__name__}: {err}\n'
                        f'{traceback.format_exc()}')

    def _heartbeat(self):
        held = time.monotonic() - self.stage_since
        if not self.armed:
            self.log('awaiting prompts — publish /sam3/set_prompts to arm')
        elif self.stage == 'waiting' and not self.frames_done:
            hint = ('no /camera/image — is the bag playing / sim running?' if not self.frames_in
                    else 'inputs present, waiting on worker')
            self.log(f'waiting: {self.frames_in} images; {hint}')
        elif self.stage != 'waiting' and held >= self.SLOW_STAGE_S:
            self.log(f'stage: {self.stage} for {held:.0f}s (frame {self.frames_done + 1})')

    def _process(self, image, stamp):
        rgb = image[:, :, ::-1].copy()                # cv_bridge gives BGR; SAM 3 wants RGB
        # The backend's timer, so SAM stages and the node's own land in one table.
        timer = self.backend.timer

        with timer.frame():
            self._set_stage('sam3 inference')
            start = time.perf_counter()
            result = self.backend.process_frame(rgb)
            infer_ms = (time.perf_counter() - start) * 1000.0

            self._set_stage('detections')
            with timer.stage('node_to_detections'):
                detections = to_detections(result, self.prompt_table)

            # Shift instance ids into a fresh namespace after any session reset, and track the
            # high-water mark so the next reset can clear it. Background ids (< 0) are a fixed
            # per-class encoding and must not be touched.
            instance = detections['ids'] >= 0
            if self.id_offset:
                detections['ids'][instance] += self.id_offset
            if instance.any():
                self.max_seen_id = max(self.max_seen_id, int(detections['ids'][instance].max()))

            if self.best_view_collector:
                with timer.stage('node_best_view'):
                    self.best_view_collector.consider(image, detections, stamp)

            self._set_stage('publish')
            with timer.stage('node_publish'):
                self._publish_detections(detections, stamp, image.shape[:2])
                if self.publish_annotated:
                    self._publish_annotated(image, detections, stamp)

        self.frames_done += 1
        if self.frames_done <= self.VERBOSE_FIRST or self.frames_done % self.log_every_n == 0:
            self.log(f'frame {self.frames_done}: {len(detections["ids"])} detections | '
                     f'SAM3 {infer_ms:.0f} ms | dropped {self.frames_dropped}/{self.frames_in}')
        if self.profile and self.frames_done % self.log_every_n == 0:
            from sam_mapper.profiling import format_summary

            self.log(format_summary(timer.summary(), title=f'sam_node frame {self.frames_done}'))
        if self.verbose_objects:
            self._log_verbose(detections)

    def _log_verbose(self, detections: dict) -> None:
        """Every 2D detection, one line each — validation only."""
        for label, score, obj_id, bbox in zip(detections['labels'], detections['confidences'],
                                              detections['ids'], detections['bboxes']):
            x0, y0, x1, y1 = (round(float(v)) for v in bbox)
            self.log(f'  2D  {label:<14} id={obj_id:<4} score={score:.2f} bbox=({x0},{y0},{x1},{y1})')

    # -- publishing -----------------------------------------------------------

    def _publish_detections(self, detections: dict, stamp: float, hw: tuple) -> None:
        height, width = hw
        seconds = int(stamp)
        header_stamp = Time(seconds=seconds, nanoseconds=int((stamp - seconds) * 1e9)).to_msg()

        # One image regardless of object count, and directly viewable in RViz/Foxglove
        # (each object is a distinct pixel value) — see docs/M2_perception.md 3.6-split.
        # Later entries win on overlapping pixels; rare, not worth resolving further.
        id_map = build_id_map(detections['ids'], detections['masks'], height, width)
        map_msg = self.bridge.cv2_to_imgmsg(id_map, encoding='mono16')
        map_msg.header.stamp = header_stamp
        map_msg.header.frame_id = 'camera'
        self.instance_map_pub.publish(map_msg)

        entries = [
            {'id': int(obj_id), 'label': str(label), 'confidence': float(score),
             'bbox': [float(v) for v in bbox]}
            for obj_id, label, score, bbox in zip(detections['ids'], detections['labels'],
                                                  detections['confidences'], detections['bboxes'])
        ]
        # std_msgs/String has no header, so it can't carry a ROS stamp — embed one in the
        # payload instead, matching the instance map's, so map_node can pair the two.
        payload = {'stamp': {'sec': seconds, 'nanosec': int((stamp - seconds) * 1e9)}, 'entries': entries}
        self.detections_pub.publish(String(data=json.dumps(payload)))

    def _publish_annotated(self, image, detections, stamp: float):
        annotated = annotate_frame(image, detections)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        seconds = int(stamp)
        msg.header.stamp = Time(seconds=seconds, nanoseconds=int((stamp - seconds) * 1e9)).to_msg()
        msg.header.frame_id = 'camera'
        self.annotated_pub.publish(msg)

    def destroy_node(self):
        """Last chance to draw the overlays: `just run-sam` on a bag that never loops and
        never sees /pipeline/explore_done gets them here and nowhere else.

        Inline, no wait — run_node() allows the worker 2 s to join, and obj_map.json has
        either landed by now or is not coming.
        """
        try:
            self._finalize_best_views(wait_s=0.0)
        except Exception:  # noqa: BLE001 — an exception here reads as a crashed node to
            pass           # the eval harness; teardown must stay clean
        super().destroy_node()


def main(args=None):
    run_node(SamNode, 'sam_node_bootstrap', ('objects', 'sam3'),
             'ros2 launch sam_mapper sam_node.launch', args=args)


if __name__ == '__main__':
    main()
