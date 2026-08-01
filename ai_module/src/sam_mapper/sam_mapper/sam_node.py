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

map_node (map_node.py) subscribes to the latter two and reconstructs each object's mask via
`id_map == encode_instance_id(id)` — the existing 5-key to_detections() contract, unchanged,
so ObjMapper.update_map() needs no changes.

  ros2 launch sam_mapper sam_node.launch
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_mapper.annotate import annotate_frame
from sam_mapper.detections import PromptTable, encode_instance_id, to_detections
from sam_mapper.node_base import WorkerNodeMixin, run_node
from sam_mapper.sam3_backend import Sam3Backend


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

        self.prompt_table = PromptTable(config['objects'])
        self.log(f"prompts ({len(self.prompt_table.prompts)}): {self.prompt_table.prompts}")

        self.log("loading SAM 3 (first run downloads weights, this can take a while) ...")
        self.backend = Sam3Backend(config['sam3'], log=self.log)
        self.backend.set_prompts(self.prompt_table.prompts)
        self.log("SAM 3 ready")

        # -- buffers ---------------------------------------------------------
        self.frame_lock = threading.Lock()
        self.latest_frame = None                # newest undecoded sensor_msgs/Image

        self.frames_in = 0
        self.frames_done = 0
        self.frames_dropped = 0

        # Bag-loop handling, scoped to this node's own state: the SAM3 session and id
        # namespace. map_node has its own copy of the same handling for its odom/cloud buffers.
        self.last_frame_stamp = None
        self.id_offset = 0
        self.max_seen_id = -1

        # -- ROS interface ---------------------------------------------------
        def group():
            return MutuallyExclusiveCallbackGroup()

        self.create_subscription(Image, '/camera/image', self.image_callback, 10,
                                 callback_group=group())
        self.annotated_pub = self.create_publisher(Image, '/annotated_image', 2)
        self.instance_map_pub = self.create_publisher(Image, '/sam3/instance_map', 2)
        self.detections_pub = self.create_publisher(String, '/sam3/detections', 2)

        self.create_timer(self.HEARTBEAT_S, self._heartbeat, callback_group=group())

        self._start_worker(self._worker_loop)
        self.log('sam_node started')

    def log(self, msg):
        self.get_logger().info(str(msg))

    # -- callbacks ------------------------------------------------------------

    def image_callback(self, msg: Image):
        """Store the raw message; decode later, in the worker. We drop most frames, so
        decoding on arrival would waste CPU on images that get thrown away."""
        with self.frame_lock:
            if self.latest_frame is not None:
                self.frames_dropped += 1
            self.latest_frame = msg
            self.frames_in += 1

    def _take_frame(self):
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
            return frame

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

    # -- worker ---------------------------------------------------------------

    def _worker_loop(self):
        while self.running:
            self._set_stage('waiting')
            msg = self._take_frame()
            if msg is None:
                time.sleep(0.005)
                continue
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = self._stamp_of(msg)

            self._handle_time_jump(stamp)
            try:
                self._process(image, stamp)
            except Exception as err:                  # noqa: BLE001 — one bad frame must not kill the node
                self.get_logger().error(f'frame at {stamp:.3f} failed: {type(err).__name__}: {err}',
                                        exc_info=True)

    def _heartbeat(self):
        held = time.monotonic() - self.stage_since
        if self.stage == 'waiting' and not self.frames_done:
            hint = ('no /camera/image — is the bag playing / sim running?' if not self.frames_in
                    else 'inputs present, waiting on worker')
            self.log(f'waiting: {self.frames_in} images; {hint}')
        elif self.stage != 'waiting' and held >= self.SLOW_STAGE_S:
            self.log(f'stage: {self.stage} for {held:.0f}s (frame {self.frames_done + 1})')

    def _process(self, image, stamp):
        rgb = image[:, :, ::-1].copy()                # cv_bridge gives BGR; SAM 3 wants RGB

        self._set_stage('sam3 inference')
        start = time.perf_counter()
        result = self.backend.process_frame(rgb)
        infer_ms = (time.perf_counter() - start) * 1000.0

        self._set_stage('detections')
        detections = to_detections(result, self.prompt_table)

        # Shift instance ids into a fresh namespace after any session reset, and track the
        # high-water mark so the next reset can clear it. Background ids (< 0) are a fixed
        # per-class encoding and must not be touched.
        instance = detections['ids'] >= 0
        if self.id_offset:
            detections['ids'][instance] += self.id_offset
        if instance.any():
            self.max_seen_id = max(self.max_seen_id, int(detections['ids'][instance].max()))

        self._set_stage('publish')
        self._publish_detections(detections, stamp, image.shape[:2])
        if self.publish_annotated:
            self._publish_annotated(image, detections, stamp)

        self.frames_done += 1
        if self.frames_done <= self.VERBOSE_FIRST or self.frames_done % self.log_every_n == 0:
            self.log(f'frame {self.frames_done}: {len(detections["ids"])} detections | '
                     f'SAM3 {infer_ms:.0f} ms | dropped {self.frames_dropped}/{self.frames_in}')
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
        id_map = np.zeros((height, width), dtype=np.uint16)
        for obj_id, mask in zip(detections['ids'], detections['masks']):
            id_map[mask] = encode_instance_id(int(obj_id))
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

def main(args=None):
    run_node(SamNode, 'sam_node_bootstrap', ('objects', 'sam3'),
             'ros2 launch sam_mapper sam_node.launch', args=args)


if __name__ == '__main__':
    main()
