"""Equirectangular lidar->camera projection, owned copy (see docs/M2_perception.md 3.6-split).

Ported from semantic_mapping/cloud_image_fusion.py, trimmed to the two platforms sam_mapper
actually uses (mecanum_sim, mecanum) — the wheelchair/diablo/scannet variants, the generic
scan2pixels, and the unused generate_seg_cloud_v2 (debug-only, never called) are dropped.
"""
from __future__ import annotations

import numpy as np


def scan2pixels_mecanum_sim(cloud: np.ndarray, clip: bool = True) -> np.ndarray:
    """Unity sim rig: camera 0.1m above the lidar, no other offset."""
    return _scan2pixels(cloud, cam_xyz=(0.0, 0.0, 0.1), clip=clip)


def scan2pixels_mecanum(cloud: np.ndarray, clip: bool = True) -> np.ndarray:
    """Real rig (Livox Mid-360 + Ricoh Theta Z1): camera offset measured on the physical mount."""
    return _scan2pixels(cloud, cam_xyz=(-0.12, -0.075, 0.265), clip=clip)


# 360x120 FOV, 30 degrees (160 px) cropped top and bottom. Both platforms share this camera
# and a fixed -90/0/-90 camera rotation relative to the lidar — only the translation differs.
_CAMERA = {"width": 1920, "height": 640}
_CAM_ROTATION = (-1.5707963, 0.0, -1.5707963)  # roll, pitch, yaw


def _scan2pixels(cloud: np.ndarray, cam_xyz: tuple, clip: bool = True) -> np.ndarray:
    roll, pitch, yaw = _CAM_ROTATION
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    cam_R = R_z @ R_y @ R_x

    xyz = (cloud[:, :3] - np.array(cam_xyz)) @ cam_R

    width, height = _CAMERA["width"], _CAMERA["height"]
    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (width / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + width / 2 + 1).astype(int)
    vertPixelID = (width / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + height / 2 + 1).astype(int)

    if clip:
        horiPixelID = np.clip(horiPixelID, 0, width - 1)
        vertPixelID = np.clip(vertPixelID, 0, height - 1)
    return np.array([horiPixelID, vertPixelID, horiDis]).T


class CloudImageFusion:
    _PLATFORMS = {
        "mecanum_sim": scan2pixels_mecanum_sim,
        "mecanum": scan2pixels_mecanum,
    }

    #: What to do with a point outside the image (B5).
    #:
    #: "clip" (the inherited behaviour) pins it to row 0 / row 639, where any mask touching
    #: that edge swallows it. That is how ceiling returns end up inside tall objects: with
    #: the sensor 0.75 m up and a 2.78 m ceiling, EVERY ceiling return within a 1.17 m
    #: horizontal radius lands on row 0. It also makes the `in_bounds` guard in
    #: generate_seg_cloud unreachable, since the ids have already been forced in range.
    #:
    #: "reject" drops it. The camera never saw the point, so no mask may claim it.
    BOUNDS_MODES = ("clip", "reject")

    def __init__(self, platform: str, bounds_mode: str = "clip"):
        if platform not in self._PLATFORMS:
            raise ValueError(f"Invalid platform: {platform}. Available: {list(self._PLATFORMS)}")
        if bounds_mode not in self.BOUNDS_MODES:
            raise ValueError(f"Invalid bounds_mode: {bounds_mode}. Available: {list(self.BOUNDS_MODES)}")
        self.platform = platform
        self.bounds_mode = bounds_mode
        self.scan2pixels = self._PLATFORMS[platform]

    def generate_seg_cloud(self, cloud: np.ndarray, masks, R_b2w, t_b2w):
        """Project cloud (body frame) into the image, then split by mask into per-object world clouds.
        Returns None if there are no masks (callers should already guard against this)."""
        point_pixel_idx = self.scan2pixels(cloud, clip=self.bounds_mode == "clip")
        if masks is None or len(masks) == 0:
            return None

        image_shape = masks[0].shape
        in_bounds = ((point_pixel_idx[:, 0] >= 0) & (point_pixel_idx[:, 0] < image_shape[1]) &
                    (point_pixel_idx[:, 1] >= 0) & (point_pixel_idx[:, 1] < image_shape[0]))
        point_pixel_idx = point_pixel_idx[in_bounds].astype(int)
        cloud = cloud[in_bounds]

        obj_clouds_world = []
        for mask in masks:
            cloud_mask = mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            obj_clouds_world.append(cloud[cloud_mask][:, :3] @ R_b2w.T + t_b2w)
        return obj_clouds_world
