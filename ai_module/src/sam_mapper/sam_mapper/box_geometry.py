"""Exact IoU for gravity-aligned (yaw-only) 3D boxes. Pure numpy.

Yaw-only boxes factorise into a rotated-rectangle intersection in XY times the z overlap,
so this needs no pytorch3d (the stated blocker on upstream's dormant merge path), shapely
or numba — and stays importable on the host, so the node and the offline scorer share one
definition rather than two that drift."""
from __future__ import annotations

import numpy as np


def rect_from_center_extent(center, extent, yaw: float = 0.0) -> np.ndarray:
    """XY footprint corners (4,2), counter-clockwise, for a gravity-aligned box."""
    hx, hy = extent[0] / 2.0, extent[1] / 2.0
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    if yaw:
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c, -s], [s, c]]).T
    return corners + np.asarray(center[:2])


def ccw(poly) -> np.ndarray:
    """Force counter-clockwise winding — the clip below assumes it for the clipper."""
    p = np.asarray(poly, dtype=float)
    signed = float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    return p if signed >= 0 else p[::-1]


def clip(subject, clipper) -> list:
    """Sutherland-Hodgman. Exact for a convex clipper; both our polygons are rectangles."""
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        if not out:
            return []
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            # cross > 0 == left of the directed edge == inside, for a CCW clipper
            pin = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0
            qin = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0]) >= 0
            if pin:
                out.append(p)
            if pin != qin:
                d = q - p
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > 1e-12:
                    t = (edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])) / -denom
                    out.append(p + t * d)
    return out


def area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2.0)


def box_iou_poly(poly_a, az0, az1, vol_a, poly_b, bz0, bz1, vol_b) -> float:
    """Exact IoU from footprints + z spans + volumes. The shared primitive."""
    dz = min(az1, bz1) - max(az0, bz0)
    if dz <= 0:
        return 0.0
    inter_xy = area(clip(ccw(poly_a), ccw(poly_b)))
    if inter_xy <= 0:
        return 0.0
    inter = inter_xy * dz
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def overlap_fraction(centre_a, extent_a, centre_b, extent_b,
                     yaw_a: float = 0.0, yaw_b: float = 0.0) -> float:
    """Intersection over the SMALLER box's volume.

    IoU is the wrong question for "is this one object split in two". A tracker fragment is
    often much smaller than the object it broke off — a 0.2 m sliver inside a 2 m sofa has
    an IoU of ~0.01 but is entirely contained. Intersection-over-minimum reports 1.0 there,
    which is what "one of these is part of the other" actually means.
    """
    extent_a = np.asarray(extent_a, dtype=float)
    extent_b = np.asarray(extent_b, dtype=float)
    if np.any(extent_a <= 0) or np.any(extent_b <= 0):
        return 0.0

    az0, az1 = centre_a[2] - extent_a[2] / 2.0, centre_a[2] + extent_a[2] / 2.0
    bz0, bz1 = centre_b[2] - extent_b[2] / 2.0, centre_b[2] + extent_b[2] / 2.0
    dz = min(az1, bz1) - max(az0, bz0)
    if dz <= 0:
        return 0.0

    poly_a = ccw(rect_from_center_extent(centre_a, extent_a, yaw_a))
    poly_b = ccw(rect_from_center_extent(centre_b, extent_b, yaw_b))
    inter_xy = area(clip(poly_a, poly_b))
    if inter_xy <= 0:
        return 0.0

    smaller = min(float(np.prod(extent_a)), float(np.prod(extent_b)))
    return float(inter_xy * dz / smaller) if smaller > 0 else 0.0
