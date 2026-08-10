"""Lidar-fusion / 3D instance-mapping node: SAM3 detections + lidar + odom in, 3D map out.

Subscribes to sam_node's output (docs/M2_perception.md 3.6-split) instead of running SAM3
itself — the split exists because SAM3 inference shares nothing well with high-frequency ROS
callbacks (measured: 20-80 s/frame when /state_estimation at 50 Hz and /registered_scan
shared this node's GIL, vs 1-3 s/frame with SAM3 alone in its own process).

    /sam3/instance_map  sensor_msgs/Image   mono16, pixel = encode_instance_id(obj_id)
    /sam3/detections    std_msgs/String     JSON {stamp, entries: [{id, label, confidence, bbox}]}
    /registered_scan    PointCloud2         world-frame lidar
    /state_estimation   nav_msgs/Odometry   ~50 Hz pose
    /sam3/prompts_ack   std_msgs/String     new run_id = fresh SAM session, drop the map

and publishes /obj_points, /obj_boxes, /obj_labels, /obj_map_json.
Everything is inspected through ROS 2 / Foxglove; rerun is not used.

  ros2 launch sam_mapper map_node.launch

Design notes (full write-up in docs/M2_perception.md 3.6):

  * LATEST-FRAME-WINS. The 3D stage is unlikely to sustain the camera's 10 Hz, so each
    callback keeps a single slot and drops anything the worker has not picked up.
    A queue would grow without bound and make the map lag reality.
  * The instance-map/detections pair is matched by stamp — /sam3/instance_map's real
    header.stamp against the stamp embedded in /sam3/detections' JSON (std_msgs/String has
    no header of its own). Both are published back-to-back by sam_node from the same call,
    so they always share a stamp.
  * The 3D stage (object_mapper.py) is sam_mapper's own, trimmed port of semantic_mapping's
    ObjMapper (docs/M2_perception.md 3.6-split) — no more semantic_mapping dependency at all.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from bisect import bisect_left

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from sam_mapper.cloud_image_fusion import CloudImageFusion
from sam_mapper.detections import PromptTable, encode_instance_id
from sam_mapper.node_base import WorkerNodeMixin, run_node
from sam_mapper.object_mapper import ObjMapper


def _jsonable(value):
    """serialize_map_to_dict returns numpy arrays; json cannot encode those."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _find_neighbouring_stamps(stamp_list, q):
    """The two stamps in stamp_list bracketing q (clamped at the ends)."""
    i = bisect_left(stamp_list, q)
    if i == 0:
        return stamp_list[0], stamp_list[0]
    if i == len(stamp_list):
        return stamp_list[-1], stamp_list[-1]
    return stamp_list[i - 1], stamp_list[i]


class MapNode(WorkerNodeMixin, Node):

    def __init__(self, config: dict):
        super().__init__('map_node')
        self.config = config
        self.bridge = CvBridge()

        runtime = config.get('runtime', {})
        self.log_every_n = int(runtime.get('log_every_n_frames', 20))
        self.cloud_window_before = float(runtime.get('cloud_window_before', 0.5))
        self.cloud_window_after = float(runtime.get('cloud_window_after', 0.1))
        self.verbose_objects = bool(runtime.get('verbose_objects', False))

        self.linear_time_bias = config.get('detection_linear_state_time_bias', 0.0)
        self.angular_time_bias = config.get('detection_angular_state_time_bias', 0.0)

        # -- perception config (labels only — SAM3 itself runs in sam_node) --------
        self.prompt_table = PromptTable(config['objects'])

        # -- 3D mapping stage --------------------------------------------------
        self.cloud_img_fusion = CloudImageFusion(platform=config['platform'])
        self.obj_mapper = ObjMapper(
            cloud_image_fusion=self.cloud_img_fusion,
            label_template=self.prompt_table.label_template(),
            captioner=None,  # TODO (docs backlog §6 item 3): wire up a real Captioner
            log_info=self.log,
        )

        # -- buffers ---------------------------------------------------------
        # One slot for the (instance_map, detections) pair (drop-to-latest); ring buffers
        # for cloud and odom, because a frame needs the samples that BRACKET its own stamp.
        self.frame_lock = threading.Lock()
        self.latest_id_map_msg = None           # newest undecoded /sam3/instance_map
        self.latest_detections = None           # newest parsed /sam3/detections JSON

        self.cloud_lock = threading.Lock()
        self.cloud_stack, self.cloud_stamps = [], []

        self.odom_lock = threading.Lock()
        self.odom_stack, self.odom_stamps = [], []

        self.frames_in = 0
        self.frames_dropped = 0
        self.frames_done = 0
        self.skipped_no_odom = 0
        self.skipped_no_cloud = 0
        # Bag-loop handling for THIS node's own state: the odom/cloud ring buffers.
        # (The SAM3 session / id namespace is sam_node's concern now, handled entirely
        # upstream — ids arriving here are already offset and never collide.)
        self.last_frame_stamp = None
        self._sync_warn_count = 0
        # Set by the /sam3/prompts_ack callback, consumed by the worker between frames:
        # rebuilding the mapper underneath an in-flight update_map would tear its object
        # list in half.
        self._run_id = None
        self._reset_pending = False

        # -- ROS interface ---------------------------------------------------
        def group():
            return MutuallyExclusiveCallbackGroup()

        self.create_subscription(Image, '/sam3/instance_map', self.instance_map_callback, 10,
                                 callback_group=group())
        self.create_subscription(String, '/sam3/detections', self.detections_callback, 10,
                                 callback_group=group())
        self.create_subscription(PointCloud2, '/registered_scan', self.cloud_callback, 10,
                                 callback_group=group())
        self.create_subscription(Odometry, '/state_estimation', self.odom_callback, 50,
                                 callback_group=group())
        self.create_subscription(String, '/sam3/prompts_ack', self.prompts_ack_callback, 10,
                                 callback_group=group())

        self.obj_cloud_pub = self.create_publisher(PointCloud2, '/obj_points', 10)
        self.obj_box_pub = self.create_publisher(MarkerArray, '/obj_boxes', 10)
        self.obj_text_pub = self.create_publisher(MarkerArray, '/obj_labels', 10)
        self.map_json_pub = self.create_publisher(String, '/obj_map_json', 2)

        # Own callback group so the executor can run it while other callbacks are busy.
        self.create_timer(self.HEARTBEAT_S, self._heartbeat, callback_group=group())

        # `stage`/`stage_since` (set by _start_worker) are reported by the heartbeat timer
        # above rather than the worker itself: a stage that blocks (update_map runs two
        # DBSCANs per object) never returns to the top of the loop, which is exactly when
        # you need to be told.
        self._start_worker(self._worker_loop)
        self.log('map_node started')

    def log(self, msg):
        self.get_logger().info(str(msg))

    # -- callbacks ------------------------------------------------------------

    @staticmethod
    def _stamp_key(msg) -> tuple:
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def instance_map_callback(self, msg: Image):
        with self.frame_lock:
            self.latest_id_map_msg = msg
            self.frames_in += 1

    def detections_callback(self, msg: String):
        # std_msgs/String has no header, so sam_node embeds a matching stamp in the JSON
        # itself (see its _publish_detections) — parse once here rather than per pairing check.
        payload = json.loads(msg.data)
        with self.frame_lock:
            self.latest_detections = payload

    def cloud_callback(self, msg: PointCloud2):
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        with self.cloud_lock:
            self.cloud_stack.append(points)
            self.cloud_stamps.append(self._stamp_of(msg))
            # Bound here, not only in _gather_cloud: the worker may not run for tens of
            # seconds, and an unbounded stack grows the window it must scan each time.
            self._trim(self.cloud_stack, self.cloud_stamps, self.MAX_CLOUDS)

    @staticmethod
    def _odom_of(msg: Odometry) -> dict:
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        return {
            'position': [p.x, p.y, p.z],
            'orientation': [o.x, o.y, o.z, o.w],        # scipy xyzw
            'linear_velocity': [lin.x, lin.y, lin.z],
            'angular_velocity': [ang.x, ang.y, ang.z],
        }

    def odom_callback(self, msg: Odometry):
        with self.odom_lock:
            self.odom_stack.append(self._odom_of(msg))
            self.odom_stamps.append(self._stamp_of(msg))
            self._trim(self.odom_stack, self.odom_stamps, 500)

    @staticmethod
    def _trim(values, stamps, limit):
        while len(stamps) > limit:
            values.pop(0)
            stamps.pop(0)

    # -- synchronization ------------------------------------------------------

    MAX_CLOUDS = 40                  # ~6 s of /registered_scan at 7 Hz
    HEARTBEAT_S = 5.0                # stage-report tick
    SLOW_STAGE_S = 5.0               # only report a stage once it has run this long
    VERBOSE_FIRST = 3                # log the first N frames individually
    TIME_JUMP_TOLERANCE = 1.0        # seconds backwards before we call it a new lap

    def _handle_time_jump(self, stamp: float) -> None:
        """Detect a bag loop and reset this node's own sync buffers.

        Only the odom/cloud ring buffers need resetting here — they still hold the
        previous lap's newer stamps otherwise, and every frame fails sync as "older than
        oldest odom", a permanent stall. The SAM3 session / id namespace is sam_node's
        concern, handled entirely upstream before ids ever reach this node.
        """
        if self.last_frame_stamp is None or stamp >= self.last_frame_stamp - self.TIME_JUMP_TOLERANCE:
            self.last_frame_stamp = stamp
            return

        jump = self.last_frame_stamp - stamp
        self.log(f'time jumped backwards {jump:.1f}s (bag loop?) — resetting sync buffers')
        with self.odom_lock:
            self.odom_stack.clear(); self.odom_stamps.clear()
        with self.cloud_lock:
            self.cloud_stack.clear(); self.cloud_stamps.clear()

        self.last_frame_stamp = stamp
        self._sync_warn_count = 0

    def _interpolate_odom(self, stamp: float):
        """Pose at exactly `stamp`, interpolated from the bracketing odom samples.

        Linear for position/velocity, SLERP for orientation. Ported from
        semantic_mapper's mapping_ros2_node.py:514-570.

        Returns None when the frame cannot be synced yet (too old, or odom has not
        caught up) — the caller should skip the frame.
        """
        with self.odom_lock:
            if not self.odom_stamps:
                return None

            target = stamp + self.linear_time_bias
            left, right = _find_neighbouring_stamps(self.odom_stamps, target)
            if left > target:
                # Normal for the first frames (odom buffer not yet spanning the image
                # stamp). Rate-limited: if it never stops, something is actually wrong.
                self._sync_warn_count += 1
                if self._sync_warn_count in (1, 10) or self._sync_warn_count % 100 == 0:
                    self.log(f'frame {target:.2f} older than oldest odom {left:.2f} '
                             f'(skipped {self._sync_warn_count}); normal at startup, '
                             f'but persistent means /state_estimation is lagging or absent')
                return None
            if right < target:
                return None                      # odom has not caught up yet; retry later
            self._sync_warn_count = 0

            left_odom = self.odom_stack[self.odom_stamps.index(left)]
            right_odom = self.odom_stack[self.odom_stamps.index(right)]
            ratio = (target - left) / (right - left) if right != left else 0.5
            ratio = float(np.clip(ratio, 0.0, 1.0))

            odom = {
                'position': np.array(right_odom['position']) * ratio
                            + np.array(left_odom['position']) * (1 - ratio),
                'linear_velocity': np.array(right_odom['linear_velocity']) * ratio
                                   + np.array(left_odom['linear_velocity']) * (1 - ratio),
                'angular_velocity': np.array(right_odom['angular_velocity']) * ratio
                                    + np.array(left_odom['angular_velocity']) * (1 - ratio),
            }
            rotations = Rotation.from_quat([left_odom['orientation'], right_odom['orientation']])
            odom['orientation'] = Slerp([0, 1], rotations)(ratio).as_quat()

            while self.odom_stamps and self.odom_stamps[0] < left:
                self.odom_stack.pop(0)
                self.odom_stamps.pop(0)
            return odom

    def _gather_cloud(self, stamp: float):
        """Concatenate the lidar scans that fall in a window around the image stamp."""
        with self.cloud_lock:
            while self.cloud_stamps and self.cloud_stamps[0] < stamp - 1.0:
                self.cloud_stack.pop(0)
                self.cloud_stamps.pop(0)
            if not self.cloud_stamps:
                return None
            window = [
                cloud for cloud, t in zip(self.cloud_stack, self.cloud_stamps)
                if stamp - self.cloud_window_before <= t <= stamp + self.cloud_window_after
            ]
        return np.concatenate(window, axis=0) if window else None

    # -- worker ---------------------------------------------------------------

    def _take_detection_frame(self):
        """Both messages only ever advance together (published back-to-back by sam_node
        with the same stamp — embedded in the JSON itself, since std_msgs/String has no
        header), so a stamp mismatch just means the pair hasn't landed yet — wait for the
        next callback rather than processing a torn pair."""
        with self.frame_lock:
            id_map_msg, detections = self.latest_id_map_msg, self.latest_detections
            if id_map_msg is None or detections is None:
                return None
            det_stamp = (detections['stamp']['sec'], detections['stamp']['nanosec'])
            if self._stamp_key(id_map_msg) != det_stamp:
                return None
            self.latest_id_map_msg = None
            self.latest_detections = None
            return id_map_msg, detections

    def _reconstruct_detections(self, id_map_msg: Image, detections: dict) -> dict:
        """/sam3/instance_map + parsed /sam3/detections -> the same 5-key dict
        to_detections() would have produced — ObjMapper.update_map needs no changes."""
        id_map = self.bridge.imgmsg_to_cv2(id_map_msg, desired_encoding='mono16')
        entries = detections['entries']

        if not entries:
            return {
                'bboxes': np.zeros((0, 4), dtype=float),
                'confidences': np.zeros(0, dtype=float),
                'labels': np.zeros(0, dtype=object),
                'ids': np.zeros(0, dtype=int),
                'masks': np.zeros((0, *id_map.shape), dtype=bool),
            }

        bboxes, confidences, labels, ids, masks = [], [], [], [], []
        for entry in entries:
            obj_id = int(entry['id'])
            bboxes.append(entry['bbox'])
            confidences.append(entry['confidence'])
            labels.append(entry['label'])
            ids.append(obj_id)
            masks.append(id_map == encode_instance_id(obj_id))

        return {
            'bboxes': np.asarray(bboxes, dtype=float),
            'confidences': np.asarray(confidences, dtype=float),
            'labels': np.asarray(labels, dtype=object),
            'ids': np.asarray(ids, dtype=int),
            'masks': np.asarray(masks, dtype=bool),
        }

    def prompts_ack_callback(self, msg: String) -> None:
        """A new run id means sam_node re-armed with a fresh SAM 3 session.

        The whole map has to go with it. Track ids restart at that point, so a stale
        object still holding id 3 would silently absorb the new session's object 3 --
        `update_map` associates on `obj_id in single_obj.obj_id` and cannot tell the two
        apart. Nodes that live one question (eval_orchestrator relaunches everything)
        never hit this; the persistent bench loop re-arms in place and always would.
        """
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not ack.get("ok", True):
            return
        run_id = ack.get("run_id")
        if run_id is None or run_id == self._run_id:
            return
        first = self._run_id is None
        self._run_id = run_id
        if not first:
            self._reset_pending = True
            self.log(f'sam_node re-armed (run {run_id}) — dropping the previous map')

    def _reset_map(self) -> None:
        self.obj_mapper = ObjMapper(
            cloud_image_fusion=self.cloud_img_fusion,
            label_template=self.prompt_table.label_template(),
            captioner=None,
            log_info=self.log,
        )
        with self.odom_lock:
            self.odom_stack.clear(); self.odom_stamps.clear()
        with self.cloud_lock:
            self.cloud_stack.clear(); self.cloud_stamps.clear()
        self.last_frame_stamp = None

    def _worker_loop(self):
        # rclpy.ok() as well as self.running: SIGINT invalidates the context before
        # destroy_node() clears the flag, and publishing into a dead context raises
        # RCLError mid-frame (same guard as sam_node).
        while self.running and rclpy.ok():
            self._set_stage('waiting')
            if self._reset_pending:
                self._reset_pending = False
                self._reset_map()
            pair = self._take_detection_frame()
            if pair is None:
                time.sleep(0.005)
                continue
            id_map_msg, detections_payload = pair
            stamp = self._stamp_of(id_map_msg)

            self._handle_time_jump(stamp)
            odom = self._interpolate_odom(stamp)
            if odom is None:
                self.skipped_no_odom += 1
                continue
            cloud = self._gather_cloud(stamp)
            if cloud is None:
                self.skipped_no_cloud += 1
                continue

            try:
                detections = self._reconstruct_detections(id_map_msg, detections_payload)
                self._process(detections, stamp, odom, cloud)
            except Exception as err:                  # noqa: BLE001 — one bad frame must not kill the node
                # SIGINT can land mid-frame and kill the context under us; that is a
                # normal shutdown, not a frame error (same guard as sam_node).
                if not rclpy.ok():
                    break
                # rclpy's logger rejects logging's kwargs, so exc_info=True would make
                # this handler raise TypeError and kill the worker (same bug as sam_node).
                self.get_logger().error(
                    f'frame at {stamp:.3f} failed: {type(err).__name__}: {err}\n'
                    f'{traceback.format_exc()}')

    def _heartbeat(self):
        """Report what the worker is doing. Runs on a ROS timer, not in the worker.

        Two failure modes look identical from outside — a pipeline waiting on a missing
        input, and one blocked inside a slow stage — so name both the stage and how long
        it has been there.
        """
        held = time.monotonic() - self.stage_since
        if self.stage == 'waiting' and not self.frames_done:
            with self.cloud_lock:
                clouds = len(self.cloud_stamps)
            with self.odom_lock:
                odoms = len(self.odom_stamps)
            if not self.frames_in:
                hint = 'no /sam3/instance_map — is sam_node running?'
            elif not clouds:
                hint = 'no /registered_scan'
            elif not odoms:
                hint = 'no /state_estimation'
            else:
                hint = 'inputs present, waiting on sync'
            self.log(f'waiting: {self.frames_in} detections, {odoms} odom, {clouds} clouds; '
                     f'skipped {self.skipped_no_odom} no-odom / {self.skipped_no_cloud} '
                     f'no-cloud — {hint}')
        elif self.stage != 'waiting' and held >= self.SLOW_STAGE_S:
            self.log(f'stage: {self.stage} for {held:.0f}s (frame {self.frames_done + 1})')

    def _process(self, detections, stamp, odom, cloud):
        if len(detections['ids']) == 0:
            self.log(f'frame {stamp:.2f}: 0 detections received — nothing to map')
            self._report(0.0, 0, 0)
            return

        self._set_stage(f'update_map ({len(detections["ids"])} dets, {len(cloud)} pts)')
        start = time.perf_counter()
        self.obj_mapper.update_map(detections, stamp, odom, cloud, image=None)
        map_ms = (time.perf_counter() - start) * 1000.0

        self._set_stage('publish')
        objects_3d = self._publish_map(stamp)
        if self.verbose_objects:
            self._log_verbose(detections, objects_3d)

        self._report(map_ms, len(detections['ids']), len(objects_3d))

    def _report(self, map_ms: float, detected: int, published: int):
        self.frames_done += 1
        # Log the first few frames individually — waiting for the periodic report leaves
        # the node looking dead for a while otherwise.
        if self.frames_done > self.VERBOSE_FIRST and self.frames_done % self.log_every_n:
            return
        tracked = len(self.obj_mapper.single_obj_list)
        # tracked vs published matters: an object only reaches the map once it has voxels
        # surviving regularization with non-zero observation weight, so infer_centroid can
        # return a centroid. tracked >> published means fusion is producing thin or noisy
        # clouds — check the extrinsics and the mask erosion, not SAM 3.
        self.log(
            f'frame {self.frames_done}: {detected} detections | map {map_ms:.0f} ms | '
            f'{tracked} tracked, {published} published'
        )

    def _log_verbose(self, detections: dict, objects_3d: dict) -> None:
        """Every 2D detection and every 3D map object, one line each — validation only."""
        for label, score, obj_id, bbox in zip(detections['labels'], detections['confidences'],
                                              detections['ids'], detections['bboxes']):
            x0, y0, x1, y1 = (round(float(v)) for v in bbox)
            self.log(f'  2D  {label:<14} id={obj_id:<4} score={score:.2f} bbox=({x0},{y0},{x1},{y1})')
        for obj in objects_3d.values():
            ids = ','.join(str(i) for i in obj['id'])
            cx, cy, cz = obj['center'][:3]
            ex, ey, ez = obj['bbox3d']['extent'][:3]
            self.log(f'  3D  {obj["label"]:<14} id={ids} center=({cx:.2f},{cy:.2f},{cz:.2f}) '
                     f'extent=({ex:.2f},{ey:.2f},{ez:.2f})')

    # -- publishing -----------------------------------------------------------

    def _publish_map(self, stamp: float) -> dict:
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        bbox_msgs, text_msgs, ros_pcd = self.obj_mapper.to_ros2_msgs(stamp)

        # DELETEALL first, so markers for objects that were merged away disappear
        # instead of lingering. Stamped a hair earlier so it is ordered first.
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = Time(seconds=seconds,
                                  nanoseconds=max(nanoseconds - 10000, 0)).to_msg()
        clear.action = Marker.DELETEALL

        if ros_pcd is not None:
            self.obj_cloud_pub.publish(ros_pcd)
        if bbox_msgs:
            self.obj_box_pub.publish(MarkerArray(markers=[clear] + list(bbox_msgs)))
        if text_msgs:
            self.obj_text_pub.publish(MarkerArray(markers=[clear] + list(text_msgs)))

        # Published unconditionally, even when empty — an empty map is itself the answer
        # when nothing else is coming out. `just sam-status` decodes it.
        objects = self.obj_mapper.serialize_map_to_dict(stamp)
        self.map_json_pub.publish(String(data=json.dumps(objects, default=_jsonable)))
        return objects

def main(args=None):
    run_node(MapNode, 'map_node_bootstrap', ('platform', 'objects'),
             'ros2 launch sam_mapper map_node.launch', args=args)


if __name__ == '__main__':
    main()
