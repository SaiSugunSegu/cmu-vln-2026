"""Is sam_mapper working? Watch every output topic and summarise what the map contains.

Answers three questions in one shot:
  * is the node publishing at all, and how fast;
  * what objects does it think exist, by class;
  * where are they, so obviously-wrong 3D positions show up immediately.

    python -m sam_mapper.tools.status [--seconds 15]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

TOPICS = [
    ("/obj_points", PointCloud2),
    ("/obj_boxes", MarkerArray),
    ("/obj_labels", MarkerArray),
    ("/annotated_image", Image),
    ("/obj_map_json", String),
]


class StatusNode(Node):
    def __init__(self):
        super().__init__("sam_mapper_status")
        self.counts = defaultdict(int)
        self.latest_map = None
        for topic, msg_type in TOPICS:
            self.create_subscription(msg_type, topic,
                                     lambda msg, t=topic: self._on(t, msg), 10)

    def _on(self, topic, msg):
        self.counts[topic] += 1
        if topic == "/obj_map_json":
            self.latest_map = msg.data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = StatusNode()
    print(f"listening for {args.seconds:.0f}s ...\n")
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    elapsed = time.monotonic() - start

    print(f"{'topic':<20} {'msgs':>6} {'Hz':>7}")
    for topic, _ in TOPICS:
        count = node.counts[topic]
        print(f"{topic:<20} {count:>6} {count / elapsed:>7.2f}")

    if node.latest_map is None:
        print("\nNo /obj_map_json yet. Either the node is still loading SAM 3, no frame has "
              "synced (check its log for odom warnings), or nothing was detected.")
    else:
        objects = json.loads(node.latest_map)
        print(f"\n{len(objects)} objects in the map")
        by_label = Counter(o.get("label") for o in objects.values())
        for label, n in by_label.most_common():
            print(f"  {label:<14} {n}")
        print("\n  id     label          centroid (x, y, z)")
        for key, obj in list(objects.items())[:25]:
            centre = obj.get("center") or []
            coords = ", ".join(f"{c:7.2f}" for c in centre[:3]) if centre else "?"
            print(f"  {str(key):<6} {str(obj.get('label')):<14} {coords}")
        if len(objects) > 25:
            print(f"  ... and {len(objects) - 25} more")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
