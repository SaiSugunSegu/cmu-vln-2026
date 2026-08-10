"""

The category-2 answer payload: one CUBE Marker per selected object.

Single definition of pose+scale so map_node, the offline scorer and any future selector
cannot drift. Pure numpy — importable on the host without ROS.

A CUBE, not the LINE_LIST wireframe: that leaves `pose` at identity and uses `scale.x` as
a line width, so a pose+scale reader sees a 5 cm x 0 x 0 box at the origin and scores 0."""

from __future__ import annotations

import numpy as np


def yaw_to_quaternion(yaw: float) -> list[float]:
    """

xyzw quaternion for a rotation about +z."""
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


def quaternion_to_yaw(quat) -> float:
    x, y, z, w = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def marker_payload(center, extent, yaw: float = 0.0, label: str = "") -> dict:
    """

Everything that goes into the answer Marker, as plain data.

    `extent` is the FULL box size along its own axes (not half-extents), matching
    Marker.scale semantics for a CUBE and matching VLA-3D's `size` field, so the two are
    directly comparable without a convention change on either side.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    extent = np.asarray(extent, dtype=float).reshape(3)
    return {
        "type": "CUBE",
        "position": center.tolist(),
        "orientation": yaw_to_quaternion(yaw),
        "scale": np.abs(extent).tolist(),
        "text": str(label),
    }


def payload_from_map_object(obj: dict) -> dict:
    """

Build the answer payload from one entry of /obj_map_json.

    Uses `bbox3d`, never the sibling `center` key: those two are different points (the
    latter is an observation-weighted voxel centroid, the former the box midpoint), and
    the answer must be the centre of the box we are claiming.
    """
    bbox = obj["bbox3d"]
    return marker_payload(
        center=bbox["center"],
        extent=bbox["extent"],
        yaw=quaternion_to_yaw(bbox.get("rotation", [0.0, 0.0, 0.0, 1.0])),
        label=obj.get("label", ""),
    )
