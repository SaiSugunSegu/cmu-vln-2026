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
    # How long finalize() may wait for queued crop writes to land. The crops are written
    # asynchronously now, so this is real waiting, not a formality.
    FINALIZE_DRAIN_S = 30.0          # end of run: let every queued crop land
    SHUTDOWN_DRAIN_S = 5.0           # bag loop / teardown: eval_orchestrator SIGKILLs at 45 s
    VERBOSE_FIRST = 3
    # Seconds backwards before the stream counts as having regressed. Only ever a LOG
    # threshold now -- see runtime.reset_session_on_time_jump.
    TIME_JUMP_TOLERANCE = 1.0

    def __init__(self, config: dict):
        super().__init__('sam_node')
        self.bridge = CvBridge()

        runtime = config.get('runtime', {})
        self.publish_annotated = runtime.get('publish_annotated', True)
        self.log_every_n = int(runtime.get('log_every_n_frames', 20))
        self.verbose_objects = bool(runtime.get('verbose_objects', False))
        # How much of the node's frame is NOT SAM 3. Costs a cuda sync per stage.
        self.profile = bool(runtime.get('profile', False))
        # Restarting the SAM 3 session on a backwards timestamp is correct for
        # `ros2 bag play --loop` and wrong for everything else -- see _handle_time_jump.
        self.reset_session_on_time_jump = bool(
            runtime.get('reset_session_on_time_jump', False))

        self.declare_parameter('wait_for_prompts', False)
        self.wait_for_prompts = bool(self.get_parameter('wait_for_prompts').value)

        # `armed` == "has prompts worth spending inference on". Unarmed, the node holds
        # loaded weights and an empty session, and drops every frame.
        # What the node is currently armed with. A repeat of the same pair is a duplicate
        # request, not a re-arm -- see _on_set_prompts.
        self.armed_run_id = None
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
        # The in-flight /pipeline/explore_done finalize, if any. destroy_node joins it rather
        # than starting a second pass over the same crops.
        self._finalize_thread = None

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
        # Only when there is something to arm with. Unarmed, image_callback returns before it
        # buffers and the worker just sleeps, so no frame can reach process_frame before
        # /sam3/set_prompts — a promptless boot session would be dead state whose only effect
        # is to make "one session per question" print as two. Both backends lazily create one
        # on first use anyway (sam3_backend.process_frame, sam31_backend._run), and failing
        # here would kill the node, where failing in _on_set_prompts rolls back and nacks.
        if self.armed:
            self.backend.set_prompts(self.prompt_table.prompts)
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
            # Wall clock, not the header stamp: the header carries BAG time, which a replay
            # rate scales. Only this tells us how long a frame really waited. time.time()
            # rather than monotonic because map_node, a different process, differences
            # against it (see _publish_detections).
            self.latest_frame = (msg, time.time())
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

        # A duplicate request must not restart anything. Both reasoners publish
        # /sam3/set_prompts and smart_vlm.launch hands them the SAME run_id, so a second arm
        # would start a fresh SAM session (ids back to 0, max_seen_id back to -1) while
        # map_node's prompts_ack_callback, seeing an unchanged run_id, keeps its map -- the
        # new session's object 0 then merges into whatever held id 0 before, silently. It
        # would also rebuild BestViewCollector, whose _clear_stale_crops() deletes the crops
        # collected so far. Keyed on the PAIR: the bench loop re-arms with a new run_id per
        # question and must still get its fresh session and run directory.
        with self.prompt_lock:
            duplicate = (self.armed and self.prompt_table is not None
                         and list(self.prompt_table.prompts) == prompts
                         and self.armed_run_id == run_id)
            collector = self.best_view_collector
        if duplicate:
            self.log(f'set_prompts ignored: already armed with {prompts} (run {run_id})')
            self.prompts_ack_pub.publish(String(data=json.dumps({
                "ok": True, "error": None, "prompts": list(prompts),
                "labels": [spec.label for spec in self.prompt_table.specs],
                "run_id": run_id,
                "run_dir": collector.run_dir if collector is not None else None,
            })))
            self._publish_status('ready')
            return

        # C3: a genuine re-arm after frames have flowed is a MID-QUESTION session restart.
        # Legitimate only between questions (the persistent bench loop); inside one it breaks
        # id continuity and therefore the 3D map.
        if self.armed and self.frames_done:
            self.get_logger().warning(
                f'RE-ARMING mid-run after {self.frames_done} frames: {prompts} (run {run_id}) '
                '— this starts a new SAM 3 session, so object ids restart and the '
                '3D map loses continuity. Expected only between questions.')

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
                self.armed_run_id = run_id
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

        # Past the rollback path, so the previous collector is genuinely retired rather
        # than restored. Its writer thread would otherwise sit parked on its condition for
        # the life of the process, one per question on a bench loop that re-arms in place.
        if previous_collector is not None and previous_collector is not self.best_view_collector:
            try:
                previous_collector.stop()
            except Exception:  # noqa: BLE001 — teardown of the old run must not fail the new one
                self.get_logger().error(
                    f'could not stop previous best-view writer:\n{traceback.format_exc()}')

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
        # Kept so destroy_node can join it. The eval orchestrator SIGINTs the moment the
        # answer lands, which is milliseconds after this fires, and a daemon thread dies with
        # the process: hotel_room_1 lost its overlays to exactly that four-millisecond gap.
        self._finalize_thread = threading.Thread(target=self._finalize_best_views,
                                                 daemon=True)
        self._finalize_thread.start()

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
        # wait_s == 0.0 is the caller saying "do not wait around" — bag loop and teardown.
        # finalize() drains the pending crop writes first, so an unbounded wait here would
        # spend the harness's whole SIGINT budget before it ever escalates.
        drain_s = self.SHUTDOWN_DRAIN_S if wait_s == 0.0 else self.FINALIZE_DRAIN_S
        objects = self._read_obj_map(path, wait_s)
        try:
            # None, not {}: "no 3D map at all" makes the track id the only id there is,
            # while an empty map means every instance genuinely failed to reach a box.
            rendered = collector.finalize(
                None if objects is None else track_to_map_id(objects), drain_timeout=drain_s)
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
        """Start a clean SAM3 session when /camera/image loops back to the start.

        Only this node's own state needs resetting here: the SAM3 session (ids only mean
        anything within one session) and the id namespace (new ids must never collide with
        ones map_node has already placed in the map).

        A LOOP is the only thing this is for, and only `ros2 bag play --loop` produces one.
        The live sim does not loop; it reorders `/camera/image` by a second or two, which is
        indistinguishable from a loop to a threshold and nothing like one in consequence.
        Measured over a 13-scene sweep: 215 resets fired, every jump between 1.0 s and 2.8 s,
        not one of them a loop (a loop regresses by the bag's whole length). Each false reset
        renumbers every track, so D8 world merge has to reassemble each object from fragments
        it cannot always prove belong together -- track ids per mapped object went 1.37 with no
        resets, 2.14 at 1-5, 4.59 at 6-20, and one loft `chair` came back as eight ids.

        So the reset is OFF by default and the regression is merely reported. Raising the
        tolerance instead would have left a threshold guarding a case that cannot occur.
        """
        if self.last_frame_stamp is None or stamp >= self.last_frame_stamp - self.TIME_JUMP_TOLERANCE:
            self.last_frame_stamp = stamp
            return

        jump = self.last_frame_stamp - stamp
        if not self.reset_session_on_time_jump:
            # Throttled and still a WARN: the reordering is not acted on, but it is real, and
            # a frame that regresses past runtime.cloud_window_before cannot be paired with a
            # cloud at all. map_node's "older than oldest odom (skipped N)" is the same event
            # seen from the other side; the two together say whether the window needs widening.
            self.get_logger().warning(
                f'time jumped backwards {jump:.1f}s — keeping SAM 3 session '
                f'#{getattr(self.backend, "session_epoch", "?")} '
                f'(runtime.reset_session_on_time_jump is off)',
                throttle_duration_sec=10.0)
            self.last_frame_stamp = stamp
            return

        # WARN, not INFO: this ends the run's single-session guarantee, and every id map_node
        # has already placed in the map is about to be superseded.
        self.get_logger().warning(
            f'time jumped backwards {jump:.1f}s (bag loop) — resetting SAM 3 session '
            f'(was #{getattr(self.backend, "session_epoch", "?")}); '
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
            taken = self._take_frame()
            if taken is None:
                time.sleep(0.005)
                continue
            msg, arrived_at = taken
            # Every stage boundary from here is recorded into `timing`, which rides along
            # in the published detections so map_node can close the loop end to end.
            timing = {'t_arrival': arrived_at, 't_pickup': time.time()}
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = self._stamp_of(msg)

            with self.prompt_lock:
                self._handle_time_jump(stamp)
                try:
                    self._process(image, stamp, timing)
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

    def _process(self, image, stamp, timing: dict | None = None):
        rgb = image[:, :, ::-1].copy()                # cv_bridge gives BGR; SAM 3 wants RGB
        # The backend's timer, so SAM stages and the node's own land in one table.
        timer = self.backend.timer
        # Plain time.time(), never StageTimer: its stage boundaries each cost a
        # torch.cuda.synchronize(), which would perturb the very frame time being measured
        # and is why it stays behind runtime.profile. These are free, so they stay on.
        timing = {} if timing is None else timing

        with timer.frame():
            self._set_stage('sam3 inference')
            start = time.perf_counter()
            result = self.backend.process_frame(rgb)
            infer_ms = (time.perf_counter() - start) * 1000.0

            timing['t_sam_done'] = time.time()

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
            # Recorded whether or not a collector exists, so the term is always present and
            # always means the same thing. After the move to _FlushWriter this is selection
            # only -- if it is not near zero, something is back on the critical path.
            timing['t_bestview_done'] = time.time()

            self._set_stage('publish')
            with timer.stage('node_publish'):
                self._publish_detections(detections, stamp, image.shape[:2], timing)
                if self.publish_annotated:
                    self._publish_annotated(image, detections, stamp)

        self.frames_done += 1
        if self.frames_done <= self.VERBOSE_FIRST or self.frames_done % self.log_every_n == 0:
            self.log(f'frame {self.frames_done}: {len(detections["ids"])} detections | '
                     f'SAM3 {infer_ms:.0f} ms | dropped {self.frames_dropped}/{self.frames_in} | '
                     f'session {getattr(self.backend, "session_epoch", "?")}')
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

    def _publish_detections(self, detections: dict, stamp: float, hw: tuple,
                            timing: dict | None = None) -> None:
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
        if timing is not None:
            # The instance map goes out first and this String second, so t_published is the
            # last thing sam_node does with the frame -- map_node's `link` term measures from
            # here. Both processes share a clock (same container), which is what makes the
            # subtraction meaningful; an NTP step mid-run would show up as one absurd frame.
            timing['t_published'] = time.time()
            # Carried so map_node can print both nodes' losses on one line; it has no other
            # way to see how many frames never made it out of sam_node at all.
            timing['sam_in'] = self.frames_in
            timing['sam_dropped'] = self.frames_dropped
            payload['timing'] = timing
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
            # Let the explore_done pass finish rather than racing it. It is already doing this
            # work with better data (it waits for obj_map.json; this path does not), and two
            # passes over one crop set differ only in which one wins the `_rendered_with`
            # claim. Bounded by the same drain budget, so a wedged pass cannot eat the
            # harness's SIGINT window.
            in_flight = self._finalize_thread
            if in_flight is not None and in_flight.is_alive():
                in_flight.join(timeout=self.SHUTDOWN_DRAIN_S)
            self._finalize_best_views(wait_s=0.0)
        except Exception:  # noqa: BLE001 — an exception here reads as a crashed node to
            pass           # the eval harness; teardown must stay clean
        # After finalize, which drains it: stopping first would leave the overlays drawn
        # over a selection the writer had not landed yet.
        with self.prompt_lock:
            collector = self.best_view_collector
        if collector is not None:
            try:
                collector.stop(timeout=self.SHUTDOWN_DRAIN_S)
            except Exception:  # noqa: BLE001
                pass
        super().destroy_node()


def main(args=None):
    run_node(SamNode, 'sam_node_bootstrap', ('objects', 'sam3'),
             'ros2 launch sam_mapper sam_node.launch', args=args)


if __name__ == '__main__':
    main()
