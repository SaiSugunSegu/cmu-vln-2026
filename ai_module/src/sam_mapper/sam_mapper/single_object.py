"""Per-object 3D representation: voxel voting + DBSCAN shape regularization.

Owned copy of semantic_mapping/single_object.py (docs/M2_perception.md 3.6-split), trimmed to
what ObjMapper actually calls (including its dormant TODOs — see object_mapper.py). Dropped:
reproject_filter (dead, only reproject_obs_angle is called), retrieve_valid_voxels_with_weights
and retrieve_valid_voxels_clustered (rerun-visualizer-only), get_info_str (only the dropped
print_obj_info used it), cal_distance (unused anywhere), add_key_frame and the
key_frames/key_pose/is_active/merged_obj_ids attributes (write-only, never read).

Fixed while porting: compute_valid_indices used to call cal_clusters() a second time
unconditionally (in addition to the one regularize_shape already triggers), running DBSCAN
twice per object per frame whenever new data just merged in. The reset of req_clustering now
lives inside cal_clusters() itself, so it's gated correctly and runs once. Also dropped a dead
weighted-total computation in retrieve_valid_voxel_indices that indexed self.vote with indices
sorted from a possibly-smaller filtered array (mismatched even before considering it was unused).
"""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


def normalize_angles_to_pi(angles):
    """Wrap radians into [-pi, pi)."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def R_to_yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])


def discretize_angles(angles, num_bin=20):
    bin_width = 2 * np.pi / num_bin
    return np.floor((angles + np.pi) / bin_width).astype(int)


DIMENSION_PRIORS = {
    "default": (5.0, 5.0, 2.0),
    "table": (5.0, 3.0, 2.0),
    "chair": (1.5, 1.5, 2.0),
    "sofa": (3.0, 3.0, 2.0),
    "pottedplant": (1.0, 1.0, 1.0),
    "fireextinguisher": (0.5, 0.5, 0.5),
}


def percentile_index_search_binary(sorted_weights, percentile):
    """First index (into ascending-sorted weights) where cumulative weight passes `percentile`."""
    total_weight = np.sum(sorted_weights)
    percentile_weight = total_weight * percentile
    current_weight = 0
    i = 0
    while i < len(sorted_weights) and current_weight < percentile_weight:
        current_weight += sorted_weights[i]
        i += 1
    return i


def get_box_3d(points):
    """Axis-aligned box: center, extent, identity quaternion (xyzw)."""
    min_xyz = points[:, :3].min(axis=0)
    max_xyz = points[:, :3].max(axis=0)
    return (min_xyz + max_xyz) / 2, max_xyz - min_xyz, [0.0, 0.0, 0.0, 1.0]


def get_bbox_3d_oriented(points):
    """Minimum-area oriented box in the XY plane, full extent in Z. TODO (docs backlog §6 item
    5): switch serialize_map_to_dict/to_ros2_msgs to this instead of get_box_3d for better
    fidelity — needs the ConvexHull-failure case (below) handled first."""
    bbox2d, _ = minimum_bounding_rectangle(points[:, :2])
    if bbox2d is None:
        return None, None, None
    center2d = np.mean(bbox2d, axis=0)
    edge1, edge2 = bbox2d[1] - bbox2d[0], bbox2d[2] - bbox2d[1]
    edge1_length, edge2_length = np.linalg.norm(edge1), np.linalg.norm(edge2)
    longest_edge = edge1 if edge1_length > edge2_length else edge2
    q = Rotation.from_euler("z", math.atan2(longest_edge[1], longest_edge[0])).as_quat()
    extent = np.array([edge1_length, edge2_length, points[:, 2].max() - points[:, 2].min()])
    center = np.array([center2d[0], center2d[1], points[:, 2].max() - extent[2] / 2])
    return center, extent, q


def minimum_bounding_rectangle(points):
    """Rotating-calipers minimum-area rectangle around a 2D point set. Returns (None, None) on
    ConvexHull failure (e.g. degenerate/collinear points) — callers must handle that."""
    try:
        hull_points = points[ConvexHull(points).vertices]
        min_area, best = float("inf"), None
        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            edge_vec = edge / np.linalg.norm(edge)
            perp_vec = np.array([-edge_vec[1], edge_vec[0]])
            proj_edge, proj_perp = points @ edge_vec, points @ perp_vec
            lo_e, hi_e, lo_p, hi_p = proj_edge.min(), proj_edge.max(), proj_perp.min(), proj_perp.max()
            area = (hi_e - lo_e) * (hi_p - lo_p)
            if area < min_area:
                min_area, best = area, (lo_e, hi_e, lo_p, hi_p, edge_vec, perp_vec)
        lo_e, hi_e, lo_p, hi_p, edge_vec, perp_vec = best
        corners = np.array([lo_e * edge_vec + lo_p * perp_vec, hi_e * edge_vec + lo_p * perp_vec,
                            hi_e * edge_vec + hi_p * perp_vec, lo_e * edge_vec + hi_p * perp_vec])
        return corners, min_area
    except Exception:
        return None, None


class VoteStatistics:
    """Per-voxel observation count + which viewing-angle bins have seen it (confidence signal —
    see docs/M2_perception.md 2.6, "voxel voting")."""

    def __init__(self, voxels: np.ndarray, voxel_size: float, odom_R, odom_t, num_angle_bin=15):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.num_angle_bin = num_angle_bin
        self.tree = cKDTree(voxels)
        self.vote = np.ones(voxels.shape[0])
        self.observation_angles = np.zeros([voxels.shape[0], num_angle_bin])
        # Only this initial batch rotates into odom_R first — later calls (update,
        # reproject_obs_angle) don't. Preserved as-is from the original; unclear if intentional,
        # but changing it would silently alter the voxel-voting signal this session validated.
        obs_angles = self._obs_angle_bins(voxels, odom_R, odom_t, apply_rotation=True)
        self.observation_angles[np.arange(voxels.shape[0]), obs_angles] = 1
        self.regularized_voxel_mask = np.zeros(voxels.shape[0], dtype=bool)

    def _obs_angle_bins(self, voxels, odom_R, odom_t, apply_rotation=False):
        voxel_to_odom = voxels - odom_t
        if apply_rotation:
            voxel_to_odom = voxel_to_odom @ odom_R
        angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        return discretize_angles(normalize_angles_to_pi(angles), self.num_angle_bin)

    def update(self, voxels, odom_R, odom_t):
        obs_angles = self._obs_angle_bins(voxels, odom_R, odom_t)
        distances, indices = self.tree.query(voxels)
        merge_mask = distances < self.voxel_size
        self.vote[indices[merge_mask]] += 1
        self.observation_angles[indices[merge_mask], obs_angles[merge_mask]] = 1

        new_voxels = voxels[~merge_mask]
        new_obs = np.zeros([new_voxels.shape[0], self.num_angle_bin])
        new_obs[np.arange(new_voxels.shape[0]), obs_angles[~merge_mask]] = 1
        self.voxels = np.concatenate([self.voxels, new_voxels], axis=0)
        self.vote = np.concatenate([self.vote, np.ones(new_voxels.shape[0])])
        self.observation_angles = np.concatenate([self.observation_angles, new_obs], axis=0)
        self.regularized_voxel_mask = np.concatenate(
            [self.regularized_voxel_mask, np.zeros(new_voxels.shape[0], dtype=bool)])
        self.tree = cKDTree(self.voxels)

    def update_through_vote_stat(self, vote_stat: "VoteStatistics"):
        distances, indices = self.tree.query(vote_stat.voxels)
        merge_mask = distances < self.voxel_size
        self.vote[indices[merge_mask]] += vote_stat.vote[merge_mask]
        self.observation_angles[indices[merge_mask]] = np.logical_or(
            self.observation_angles[indices[merge_mask]], vote_stat.observation_angles[merge_mask])

        keep = ~merge_mask
        self.voxels = np.concatenate([self.voxels, vote_stat.voxels[keep]])
        self.vote = np.concatenate([self.vote, vote_stat.vote[keep]])
        self.observation_angles = np.concatenate([self.observation_angles, vote_stat.observation_angles[keep]])
        self.regularized_voxel_mask = np.concatenate(
            [self.regularized_voxel_mask, vote_stat.regularized_voxel_mask[keep]])
        self.tree = cKDTree(self.voxels)

    def update_through_mask(self, mask):
        """Keep only the voxels where `mask` is True — used by SingleObject.pop() (dormant
        IoU split/merge path, docs backlog §6 item 4)."""
        self.voxels = self.voxels[mask]
        self.vote = self.vote[mask]
        self.observation_angles = self.observation_angles[mask]
        self.tree = cKDTree(self.voxels)

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func):
        voxels_body = self.voxels @ R_w2b.T + t_w2b
        voxels_on_image = projection_func(voxels_body).astype(int)
        if mask.size == 4:  # bbox fallback
            xmin, ymin, xmax, ymax = mask
            voxels_mask = ((voxels_on_image[:, 0] >= xmin) & (voxels_on_image[:, 0] <= xmax) &
                          (voxels_on_image[:, 1] >= ymin) & (voxels_on_image[:, 1] <= ymax))
        else:
            voxels_mask = mask[voxels_on_image[:, 1], voxels_on_image[:, 0]].astype(bool)

        odom_t, odom_R = -R_w2b.T @ t_w2b, R_w2b.T
        obs_angles = self._obs_angle_bins(self.voxels[voxels_mask], odom_R, odom_t)
        self.observation_angles[voxels_mask, obs_angles] = 1

    def retrieve_valid_voxel_indices(self, diversity_percentile=0.3, regularized=True):
        obs_angles = self.observation_angles[self.regularized_voxel_mask] if regularized else self.observation_angles
        if len(obs_angles) == 0:
            return np.empty(0, dtype=int)

        angle_diversity = np.sum(obs_angles, axis=1)
        sorted_indices = np.argsort(angle_diversity)  # smaller to larger
        percentile_index = percentile_index_search_binary(angle_diversity[sorted_indices], 1 - diversity_percentile)
        return sorted_indices[percentile_index:]


class SingleObject:
    """One tracked 3D instance — a voxel-voted point cluster plus its class/id/lifecycle state."""

    def __init__(self, class_id, obj_id, voxels, voxel_size, odom_R, odom_t, mask, stamp, num_angle_bin=15):
        self.class_id = {class_id: 1}
        self.obj_id = [obj_id]
        self.vote_stat = VoteStatistics(voxels, voxel_size, odom_R, odom_t, num_angle_bin)

        self.life = 0
        self.inactive_frame = -1
        self.latest_stamp = stamp
        self.info_frames_cnt = 1

        self.valid_indices = None
        self.valid_indices_regularized = None
        self.clustering_labels = None

        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True

    def merge(self, voxels, odom_R, odom_t, label, stamp):
        self.vote_stat.update(voxels, odom_R, odom_t)
        self.info_frames_cnt += 1
        self.latest_stamp = stamp
        self.class_id[label] = self.class_id.get(label, 0) + 1
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True

    def merge_object(self, other: "SingleObject"):
        self.obj_id.extend(other.obj_id)
        self.vote_stat.update_through_vote_stat(other.vote_stat)
        self.life = max(self.life, other.life)
        self.info_frames_cnt += other.info_frames_cnt
        self.latest_stamp = max(self.latest_stamp, other.latest_stamp)
        for key, count in other.class_id.items():
            self.class_id[key] = self.class_id.get(key, 0) + count
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True

    def reproject_obs_angle(self, R_w2b, t_w2b, mask, projection_func):
        self.vote_stat.reproject_obs_angle(R_w2b, t_w2b, mask, projection_func)
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True

    def get_dominant_label(self):
        return max(self.class_id, key=self.class_id.get)

    def dbscan_cluster_params(self):
        min_points = 5 if (self.info_frames_cnt < 3 and self.inactive_frame < 5) else 20
        return self.vote_stat.voxel_size * 2.0, min_points

    def cal_clusters(self):
        if self.req_clustering:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.vote_stat.voxels)
            eps, min_points = self.dbscan_cluster_params()
            self.clustering_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
            self.req_clustering = False

    def regularize_shape(self, percentile=None):
        """Cluster voxels (DBSCAN), then keep clusters — largest observation-weight first — that
        fit the class's DIMENSION_PRIORS, up to `percentile` of total weight."""
        if not self.req_shape_regularization:
            return
        self.cal_clusters()
        unique_labels = np.unique(self.clustering_labels)
        dim_prior = DIMENSION_PRIORS.get(self.get_dominant_label(), DIMENSION_PRIORS["default"])

        cluster_masks, cluster_weights = [], []
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.clustering_labels == label
            cluster_masks.append(mask)
            cluster_weights.append(np.sum(self.vote_stat.observation_angles[mask]))

        valid_mask = np.zeros(self.vote_stat.voxels.shape[0], dtype=bool)
        total_weight = np.sum(cluster_weights)
        current_weight = 0
        for weight_index in reversed(np.argsort(cluster_weights)):
            if cluster_weights[weight_index] < 10:  # drop tiny clusters (odom noise, bleed-through)
                continue
            attempt = np.logical_or(valid_mask, cluster_masks[weight_index])
            _, extent, _ = get_bbox_3d_oriented(self.vote_stat.voxels[attempt])
            if extent is None or extent[0] > dim_prior[0] or extent[1] > dim_prior[1] or extent[2] > dim_prior[2]:
                continue
            valid_mask = attempt
            if percentile is not None:
                current_weight += cluster_weights[weight_index]
                if current_weight > percentile * total_weight:
                    break

        self.vote_stat.regularized_voxel_mask = valid_mask
        self.req_recompute_indices = True
        self.req_shape_regularization = False

    def pop(self, mask):
        """Split off the voxels NOT in `mask`, for the dormant IoU-overlap voxel-exchange path
        (docs backlog §6 item 4, object_mapper.py) — not currently called."""
        voxels_pop = self.vote_stat.voxels[~mask]
        obs_angles_pop = self.vote_stat.observation_angles[~mask]
        votes_pop = self.vote_stat.vote[~mask]
        self.vote_stat.update_through_mask(mask)
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True
        return voxels_pop, obs_angles_pop, votes_pop

    def add(self, voxels, obs_angles, votes):
        """Counterpart to pop() — also dormant, same feature."""
        self.vote_stat.voxels = np.concatenate([self.vote_stat.voxels, voxels])
        self.vote_stat.observation_angles = np.concatenate([self.vote_stat.observation_angles, obs_angles])
        self.vote_stat.vote = np.concatenate([self.vote_stat.vote, votes])
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True

    def compute_valid_indices(self, diversity_percentile):
        if not self.req_recompute_indices:
            return
        self.valid_indices = self.vote_stat.retrieve_valid_voxel_indices(diversity_percentile, regularized=False)
        self.regularize_shape(percentile=diversity_percentile)
        self.valid_indices_regularized = self.vote_stat.retrieve_valid_voxel_indices(diversity_percentile, regularized=True)
        self.req_recompute_indices = False

    def retrieve_valid_voxels(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        regularized = regularized and self.obj_id[0] >= 0  # background points aren't regularized
        if regularized:
            return self.vote_stat.voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        return self.vote_stat.voxels[self.valid_indices]

    def infer_centroid(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        voxels, obs_angles = self.vote_stat.voxels, self.vote_stat.observation_angles
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask]
            obs_angles = obs_angles[self.vote_stat.regularized_voxel_mask]
            valid_voxels, valid_obs = voxels[self.valid_indices_regularized], obs_angles[self.valid_indices_regularized]
        else:
            valid_voxels, valid_obs = voxels[self.valid_indices], obs_angles[self.valid_indices]
        weights = np.sum(valid_obs, axis=1)
        if np.sum(weights) == 0:
            return None
        return np.average(valid_voxels, axis=0, weights=weights)

    def infer_bbox(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        voxels = self.vote_stat.voxels
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            voxels = voxels[self.valid_indices]
        return get_box_3d(voxels) if len(voxels) else None

    def infer_bbox_oriented(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        voxels = self.vote_stat.voxels
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            voxels = voxels[self.valid_indices]
        return get_bbox_3d_oriented(voxels) if len(voxels) else None
