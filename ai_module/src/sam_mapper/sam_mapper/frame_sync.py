"""Frame assembly: pair a detection with the pose and lidar that belong to it.

Extracted so the offline bench runs this arithmetic rather than a reimplementation of it —
a bench that syncs frames differently from the node measures a different pipeline.
Pure numpy, no ROS."""
from __future__ import annotations

from bisect import bisect_left

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sam_mapper.detections import encode_instance_id

# _interpolate_odom outcomes. Only OK carries a pose; the rest tell the caller why a
# frame could not be synced, so map_node can keep its distinct log lines.
OK = "ok"
NO_ODOM = "no_odom"                 # odom buffer empty
TOO_OLD = "too_old"                 # frame predates the oldest odom sample
NOT_CAUGHT_UP = "not_caught_up"     # odom has not reached the frame stamp yet


def find_neighbouring_stamps(stamp_list, q):
    """The two stamps in stamp_list bracketing q (clamped at the ends)."""
    i = bisect_left(stamp_list, q)
    if i == 0:
        return stamp_list[0], stamp_list[0]
    if i == len(stamp_list):
        return stamp_list[-1], stamp_list[-1]
    return stamp_list[i - 1], stamp_list[i]


def interpolate_odom(odom_stack, odom_stamps, target):
    """Pose at exactly `target`, from the bracketing odom samples.

    Linear for position/velocity, SLERP for orientation. mapping_ros2_node.py:514-570.

    Returns (odom_dict, OK), or (None, reason) when the frame cannot be synced.
    Does not mutate odom_stack/odom_stamps — trimming is the caller's business.
    """
    if not odom_stamps:
        return None, NO_ODOM

    left, right = find_neighbouring_stamps(odom_stamps, target)
    if left > target:
        return None, TOO_OLD
    if right < target:
        return None, NOT_CAUGHT_UP

    left_odom = odom_stack[odom_stamps.index(left)]
    right_odom = odom_stack[odom_stamps.index(right)]
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
    return odom, OK


def gather_cloud(cloud_stack, cloud_stamps, stamp, before, after):
    """Concatenate the lidar scans falling in [stamp - before, stamp + after]."""
    window = [
        cloud for cloud, t in zip(cloud_stack, cloud_stamps)
        if stamp - before <= t <= stamp + after
    ]
    return np.concatenate(window, axis=0) if window else None


def reconstruct_detections(id_map, entries):
    """mono16 instance map + parsed detection entries -> the same 5-key dict
    to_detections() would have produced, so ObjMapper.update_map needs no changes."""
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
