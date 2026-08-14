#!/usr/bin/env python3
"""Copy a scene bag and add a ``/gt_boxes`` MarkerArray, viewable in Foxglove next to
the live camera image and point cloud.

``project_bboxes.py`` checks one object against one frame at a time; this is the
"drive around and look at everything at once" version -- open the output file in
Foxglove's 3D panel and the category-2 ground-truth boxes sit in the scene exactly
where the point cloud says the room is, coloured by whether the visibility gate
accepted them.

By default (``--source qa``) box corners are read straight out of the category-2 QA
file's own ``bbox_corners`` -- the exact numbers a scorer would read -- not recomputed
from ``<scene>_objects.json``. That verifies what was actually shipped, not just the
metadata it was built from, and each such object is also checked numerically against
the metadata: a mismatch beyond 0.5mm gets a "!! MISMATCH" prefix on its label in
Foxglove, in addition to being printed to the console. Both a question's *answer* and
its *anchors* carry geometry in the QA file, so ``--source qa`` restricts
``--objects all/visible`` to the set of objects some question names, target or anchor;
pass ``--source objects`` for the full scene, including objects no question names.

Every other topic (``/camera/image``, ``/registered_scan``, ``/state_estimation``,
``/tf``, ``/tf_static``) is copied through byte-for-byte from the source mcap -- no
ROS2 install needed, see ``scripts/utils/mcap_io.py`` and the ``mcap``/
``mcap-ros2-support`` packages this only additionally needs for the encode side.

    python3 scripts/eval/make_bbox_mcap.py --scene arabic_room
    python3 scripts/eval/make_bbox_mcap.py --scene arabic_room --object-id 73
    python3 scripts/eval/make_bbox_mcap.py --scene arabic_room --source objects --objects all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.geometry as geom  # noqa: E402

from mcap.reader import make_reader  # noqa: E402
from mcap.writer import Writer as McapWriter  # noqa: E402
from mcap_ros2._dynamic import serialize_dynamic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

COPY_TOPICS = {"/camera/image", "/registered_scan", "/state_estimation", "/tf", "/tf_static"}
SCAN_TOPIC = "/registered_scan"
MARKER_TOPIC = "/gt_boxes"
MARKER_FRAME = "map"  # header.frame_id of /registered_scan and /state_estimation

# rgba
COLOR_VISIBLE = (0.15, 0.90, 0.20, 0.55)
COLOR_HIDDEN = (0.95, 0.30, 0.05, 0.45)
LINE_WIDTH = 0.05
TEXT_HEIGHT = 0.12

# Bottom-face 0-3, top-face 4-7, corner i+4 directly above corner i -- see
# project_bboxes.py's EDGES for the same convention, checked against the QA files.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

# The canonical ROS2 visualization_msgs/msg/MarkerArray definition, hand-embedded so
# writing it needs nothing beyond `pip install mcap mcap-ros2-support` -- no rosidl,
# no colcon, no ai_module build. Field layout matches upstream exactly (this is what
# `ros2 interface show visualization_msgs/msg/Marker` prints) so Foxglove's built-in
# Marker/MarkerArray renderer recognises it without any special-casing on our part.
MARKER_ARRAY_MSGDEF = """
Marker[] markers

================================================================================
MSG: visualization_msgs/Marker
uint8 ARROW=0
uint8 CUBE=1
uint8 SPHERE=2
uint8 CYLINDER=3
uint8 LINE_STRIP=4
uint8 LINE_LIST=5
uint8 CUBE_LIST=6
uint8 SPHERE_LIST=7
uint8 POINTS=8
uint8 TEXT_VIEW_FACING=9
uint8 MESH_RESOURCE=10
uint8 TRIANGLE_LIST=11
uint8 ADD=0
uint8 MODIFY=0
uint8 DELETE=2
uint8 DELETEALL=3
std_msgs/Header header
string ns
int32 id
int32 type
int32 action
geometry_msgs/Pose pose
geometry_msgs/Vector3 scale
std_msgs/ColorRGBA color
builtin_interfaces/Duration lifetime
bool frame_locked
geometry_msgs/Point[] points
std_msgs/ColorRGBA[] colors
string text
string mesh_resource
bool mesh_use_embedded_materials

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: builtin_interfaces/Duration
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a
""".strip() + "\n"


def load_visibility(scene: str) -> dict:
    path = REPO / "data" / "benchmark" / scene / "visibility" / f"{scene}_visibility.json"
    return json.loads(path.read_text())["objects"] if path.exists() else {}


def qa_object_ids(scene: str, qa_path: Path | None) -> set[str]:
    path = qa_path or (REPO / "data" / "benchmark" / scene / "category_2" /
                       f"{scene}_category2_qa.json")
    if not path.exists():
        raise SystemExit(f"no category-2 QA file at {path}")
    qa = json.loads(path.read_text())
    ids: set[str] = set()
    for q in qa.get("questions", []):
        answer = q.get("answer") or {}
        if oid := answer.get("object_id"):
            ids.add(str(oid))
        for anchor in q.get("anchors") or []:
            if oid := anchor.get("object_id"):
                ids.add(str(oid))
    return ids


def select_objects(scene: str, objects: dict, mode: str, qa_path: Path | None,
                   vis: dict, object_id: list[str] | None = None) -> list[str]:
    if object_id:
        wanted = {str(o) for o in object_id}
    elif mode == "qa":
        wanted = qa_object_ids(scene, qa_path)
    elif mode == "visible":
        wanted = {oid for oid, e in vis.items() if e.get("visible")}
    elif mode == "all":
        wanted = set(objects)
    else:
        raise SystemExit(f"unknown --objects mode {mode!r}")
    dropped = sorted(oid for oid in wanted if oid not in objects)
    if dropped:
        print(f"WARNING: no box geometry available for {dropped}, skipping "
              f"(try --source objects if no question names these)")
    return sorted((oid for oid in wanted if oid in objects), key=lambda o: int(o) if o.isdigit() else 0)


def _rgba(c: tuple[float, float, float, float]) -> dict:
    return {"r": c[0], "g": c[1], "b": c[2], "a": c[3]}


def _point(p) -> dict:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def wireframe_marker(oid: str, obj, color, sec: int, nsec: int) -> dict:
    points = []
    for a, b in EDGES:
        points.append(_point(obj.corners[a]))
        points.append(_point(obj.corners[b]))
    return {
        "header": {"stamp": {"sec": sec, "nanosec": nsec}, "frame_id": MARKER_FRAME},
        "ns": "gt_box", "id": int(oid) if oid.isdigit() else hash(oid) & 0x7FFFFFFF,
        "type": 5,  # LINE_LIST
        "action": 0,  # ADD
        "scale": {"x": LINE_WIDTH, "y": 0.0, "z": 0.0},
        "color": _rgba(color),
        "points": points,
    }


def label_marker(oid: str, obj, color, sec: int, nsec: int, mismatch: bool = False) -> dict:
    top_center = obj.corners.max(axis=0)
    top_center[:2] = obj.center[:2]
    text = f"{obj.display}#{oid}"
    if mismatch:
        text = f"!! MISMATCH !! {text}"
    return {
        "header": {"stamp": {"sec": sec, "nanosec": nsec}, "frame_id": MARKER_FRAME},
        "ns": "gt_label", "id": int(oid) if oid.isdigit() else hash(oid) & 0x7FFFFFFF,
        "type": 9,  # TEXT_VIEW_FACING
        "action": 0,
        "pose": {"position": _point(top_center + [0, 0, 0.08])},
        "scale": {"x": 0.0, "y": 0.0, "z": TEXT_HEIGHT},
        "color": _rgba(color),
        "text": text,
    }


def delete_all_marker(sec: int, nsec: int) -> dict:
    return {
        "header": {"stamp": {"sec": sec, "nanosec": nsec}, "frame_id": MARKER_FRAME},
        "ns": "", "id": 0, "type": 0, "action": 3,  # DELETEALL
    }


def build_marker_array(oids: list[str], objects: dict, vis: dict, sec: int, nsec: int,
                       mismatches: set[str] = frozenset()) -> dict:
    markers = [delete_all_marker(sec, nsec)]
    for oid in oids:
        obj = objects[oid]
        visible = bool((vis.get(oid) or {}).get("visible"))
        color = COLOR_VISIBLE if visible else COLOR_HIDDEN
        markers.append(wireframe_marker(oid, obj, color, sec, nsec))
        markers.append(label_marker(oid, obj, color, sec, nsec, mismatch=oid in mismatches))
    return {"markers": markers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="arabic_room")
    ap.add_argument("--bag", default=None, help="default: data/bags/<scene>/<scene>_0.mcap")
    ap.add_argument("--bags-dir", default=str(REPO / "data" / "bags"))
    ap.add_argument("--objects", choices=["qa", "all", "visible"], default="qa",
                    help="qa = objects the category-2 QA file names (default); "
                         "all = every scene object; visible = only ones the gate accepted")
    ap.add_argument("--object-id", nargs="+", default=None,
                    help="draw only these object ids (overrides --objects)")
    ap.add_argument("--source", choices=("qa", "objects"), default="qa",
                    help="where box corners come from: the QA file's own bbox_corners "
                         "(default, verifies what was actually shipped, for answer and "
                         "anchors alike) or a fresh read of <scene>_objects.json (needed for "
                         "--objects all, for objects no question names)")
    ap.add_argument("--qa", type=Path, default=None)
    ap.add_argument("--out", default=None,
                    help="default: data/runs/foxglove/<scene>_gt_boxes.mcap")
    args = ap.parse_args()

    bag_path = Path(args.bag) if args.bag else Path(args.bags_dir) / args.scene / f"{args.scene}_0.mcap"
    if not bag_path.exists():
        raise SystemExit(f"no bag at {bag_path}")
    out_path = Path(args.out) if args.out else REPO / "data" / "runs" / "foxglove" / f"{args.scene}_gt_boxes.mcap"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_objects = geom.load_objects(args.scene, Path(args.bags_dir))
    objects = geom.load_qa_answers(args.scene, args.qa) if args.source == "qa" else metadata_objects
    vis = load_visibility(args.scene)
    if not vis:
        print(f"WARNING: no visibility report for {args.scene} -- all boxes drawn as 'hidden' colour")
    oids = select_objects(args.scene, objects, args.objects, args.qa, vis, args.object_id)
    if not oids:
        raise SystemExit("no objects selected")
    n_visible = sum(1 for o in oids if (vis.get(o) or {}).get("visible"))
    print(f"{len(oids)} object(s) selected ({n_visible} visible, {len(oids) - n_visible} not) "
          f"from {args.objects!r}, geometry from {args.source!r}")

    mismatches: set[str] = set()
    if args.source == "qa":
        for oid in oids:
            meta_obj = metadata_objects.get(oid)
            if meta_obj is None:
                continue
            delta = float(np.abs(objects[oid].corners - meta_obj.corners).max())
            if delta > 5e-4:
                mismatches.add(oid)
                print(f"MISMATCH: object {oid} ({objects[oid].display}) QA-file bbox_corners "
                      f"differ from {args.scene}_objects.json by up to {delta:.4f}m")
        if mismatches:
            print(f"{len(mismatches)} object(s) flagged MISMATCH -- their /gt_boxes label in "
                  f"Foxglove is prefixed '!! MISMATCH !!'; the QA file is likely stale and "
                  f"gen-cat2 should be rerun")

    encoders = serialize_dynamic("visualization_msgs/msg/MarkerArray", MARKER_ARRAY_MSGDEF)
    encode_markers = encoders["visualization_msgs/msg/MarkerArray"]

    print(f"reading {bag_path} ...")
    with open(bag_path, "rb") as src, open(out_path, "wb") as dst:
        reader = make_reader(src)
        writer = McapWriter(dst)
        writer.start(profile="ros2", library="cmu-vln-2026 make_bbox_mcap.py")

        schema_ids: dict[int, int] = {}
        channel_ids: dict[int, int] = {}
        scan_times: list[tuple[int, int]] = []  # (log_time_ns, publish_time_ns)
        n_copied = 0
        for schema, channel, message in reader.iter_messages(topics=sorted(COPY_TOPICS)):
            if schema.id not in schema_ids:
                schema_ids[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
            if channel.id not in channel_ids:
                channel_ids[channel.id] = writer.register_channel(
                    topic=channel.topic, message_encoding=channel.message_encoding,
                    schema_id=schema_ids[schema.id], metadata=channel.metadata)
            writer.add_message(
                channel_id=channel_ids[channel.id], log_time=message.log_time,
                publish_time=message.publish_time, sequence=message.sequence, data=message.data)
            n_copied += 1
            if channel.topic == SCAN_TOPIC:
                scan_times.append((message.log_time, message.publish_time))

        if not scan_times:
            raise SystemExit(f"{bag_path} has no {SCAN_TOPIC} messages to time /gt_boxes against")

        marker_schema_id = writer.register_schema(
            "visualization_msgs/msg/MarkerArray", "ros2msg", MARKER_ARRAY_MSGDEF.encode())
        marker_channel_id = writer.register_channel(
            topic=MARKER_TOPIC, message_encoding="cdr", schema_id=marker_schema_id)

        # Republished at every lidar frame's timestamp (not once) so scrubbing anywhere
        # in Foxglove's timeline shows the boxes immediately -- mcap playback has no
        # ROS "latched"/transient-local topic semantics to rely on for a one-shot publish.
        for seq, (log_time, publish_time) in enumerate(scan_times):
            sec, nsec = divmod(log_time, 1_000_000_000)
            data = encode_markers(build_marker_array(oids, objects, vis, sec, nsec, mismatches))
            writer.add_message(channel_id=marker_channel_id, log_time=log_time,
                              publish_time=publish_time, sequence=seq, data=data)

        writer.finish()

    size_mb = out_path.stat().st_size / 1e6
    print(f"copied {n_copied} message(s) across {len(COPY_TOPICS)} topic(s), "
          f"published /gt_boxes {len(scan_times)} time(s)")
    print(f"wrote {out_path} ({size_mb:.1f} MB) -- open it in Foxglove and add a 3D panel")


if __name__ == "__main__":
    main()
