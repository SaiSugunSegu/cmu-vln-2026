"""Marker/PointCloud2 builders for /obj_boxes, /obj_labels, /obj_points.

Owned copy of the 5 functions ObjMapper.to_ros2_msgs needs from
semantic_mapping/tools/ros2_bag_utils.py (docs/M2_perception.md 3.6-split) — that module imports
rosbag2_py at module scope for bag-writing helpers we don't use, so porting just these avoids it.
"""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


def get_3d_box(center, extent, yaw: float) -> list:
    """8 corners of a yaw-rotated box, given its center and full (not half) extent."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    lx, ly, lz = extent
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    corners = rot @ (np.array(signs) * [lx / 2, ly / 2, lz / 2]).T
    return (corners.T + np.asarray(center)).tolist()


def create_wireframe_marker_from_corners(corners, ns: str, box_id: str, color, seconds: int,
                                        nanoseconds: int, frame_id: str = "map") -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.id = int(box_id)
    marker.ns = ns
    marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
    marker.color.a = color[3] if len(color) == 4 else 0.8
    marker.scale.x = 0.05

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        marker.points.append(Point(x=corners[a][0], y=corners[a][1], z=corners[a][2]))
        marker.points.append(Point(x=corners[b][0], y=corners[b][1], z=corners[b][2]))
    return marker


def create_wireframe_marker(center, extent, yaw: float, ns: str, box_id: str, color,
                           seconds: int, nanoseconds: int, frame_id: str = "map") -> Marker:
    corners = get_3d_box(center, extent, yaw)
    return create_wireframe_marker_from_corners(corners, ns, box_id, color, seconds, nanoseconds, frame_id)


def create_text_marker(center, marker_id: int, text: str, color, text_height: float,
                      seconds: int, nanoseconds: int, frame_id: str = "map") -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    marker.ns = "text"
    marker.id = int(marker_id)
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center[0], center[1], center[2]
    marker.scale.z = text_height
    marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
    marker.color.a = color[3] if len(color) == 4 else 1.0
    marker.text = text
    return marker


def create_colored_point_cloud(points: np.ndarray, colors: np.ndarray, seconds: int,
                              nanoseconds: int, frame_id: str = "map") -> PointCloud2:
    header = Header()
    header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    header.frame_id = frame_id

    if colors.max() <= 1:
        colors = colors * 255
    rgb = colors.astype(np.uint32)
    rgb = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    cloud_data = np.concatenate((points, rgb.view(np.float32)[:, None]), axis=1).astype(np.float32)

    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 16
    cloud.row_step = cloud.point_step * len(points)
    cloud.is_dense = True
    cloud.data = cloud_data.tobytes()
    return cloud
