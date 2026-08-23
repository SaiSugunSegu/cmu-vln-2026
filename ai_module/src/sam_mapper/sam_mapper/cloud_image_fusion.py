"""Equirectangular lidar->camera projection, owned copy (see docs/M2_perception.md 3.6-split).

sam_mapper actually uses (mecanum_sim, mecanum) — the wheelchair/diablo/scannet variants, the generic
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
    #: "reject" drops it, and is the default. The camera never saw the point, so no mask
    #: may claim it.
    BOUNDS_MODES = ("clip", "reject")

    def __init__(self, platform: str, bounds_mode: str = "reject", occlusion=None):
        if platform not in self._PLATFORMS:
            raise ValueError(f"Invalid platform: {platform}. Available: {list(self._PLATFORMS)}")
        if bounds_mode not in self.BOUNDS_MODES:
            raise ValueError(f"Invalid bounds_mode: {bounds_mode}. Available: {list(self.BOUNDS_MODES)}")
        self.platform = platform
        self.bounds_mode = bounds_mode
        self.scan2pixels = self._PLATFORMS[platform]
        # Imported lazily so this module keeps its "numpy only" property, which is what lets
        # the host-side eval scripts import it without the rest of sam_mapper.
        if occlusion is None:
            from sam_mapper.mapping_config import OcclusionConfig
            occlusion = OcclusionConfig()
        self.occlusion = occlusion

    def _visible(self, pixels: np.ndarray, depth: np.ndarray, width: int) -> np.ndarray:
        """Which of these points are not hidden behind a nearer return in the same cell.

        A z-buffer, coarsened to `pixel_bin` square cells because the lidar is far sparser
        than the 1920x640 panorama: at ~1.6% pixel occupancy an exact-pixel buffer almost
        never sees two returns in one cell and would be a no-op. The cell is the smallest
        unit at which "these two returns are on the same ray" is answerable at all.

        `depth` is the lidar-origin range, deliberately the same quantity B3 filters on.
        Sharing it makes the two stages commute exactly: the range filter only ever drops
        points that are FARTHER than the ones it keeps, so it can never remove the occluder
        that decides another point's fate. Using `scan2pixels`'s third column instead would
        not do — that is the HORIZONTAL range, which ignores elevation entirely and calls a
        point on the ceiling directly overhead "zero metres away".
        """
        bin_px = max(int(self.occlusion.pixel_bin), 1)
        cells = (pixels[:, 1] // bin_px) * (width // bin_px + 1) + (pixels[:, 0] // bin_px)
        nearest = np.full(int(cells.max()) + 1, np.inf) if cells.size else np.zeros(0)
        np.minimum.at(nearest, cells, depth)
        return depth <= nearest[cells] + float(self.occlusion.depth_tolerance)

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
        depth = np.linalg.norm(cloud[:, :3], axis=1) if self.occlusion.enabled else None

        obj_clouds_world = []
        for mask in masks:
            cloud_mask = mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            if depth is not None and cloud_mask.any():
                # Per mask, not once for the whole frame: the question is whether a point is
                # behind the surface THIS mask covers. A chair's mask holds the chair and the
                # wall visible through the gap beneath it, and it is that wall — metres behind
                # the near surface, inside the same silhouette — that inflates the box.
                cloud_mask[cloud_mask] = self._visible(
                    point_pixel_idx[cloud_mask], depth[cloud_mask], image_shape[1])
            obj_clouds_world.append(cloud[cloud_mask][:, :3] @ R_b2w.T + t_b2w)
        return obj_clouds_world
