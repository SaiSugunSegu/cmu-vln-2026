"""One tracked 3D instance: voxel-voted points, class/id lifecycle, and box fitting.

Thresholds live in mapping_config (DbscanConfig, DimensionPriorsConfig, PruneConfig)."""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .mapping_config import MappingConfig


def normalize_angles_to_pi(angles):
    """

Wrap radians into [-pi, pi)."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def R_to_yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])


def discretize_angles(angles, num_bin=20):
    bin_width = 2 * np.pi / num_bin
    return np.floor((angles + np.pi) / bin_width).astype(int)


#: Read-only view of the DEFAULT per-class caps, generated from VLA-3D ground truth into
#: sam_mapper/dimension_priors.json. Code reads `self.config.dimension_priors`, which a yaml
#: override may change. See mapping_config.DimensionPriorsConfig.
DIMENSION_PRIORS = MappingConfig().dimension_priors.priors


def _fits_prior(extent, prior) -> bool:
    """Does an oriented box fit its class prior?

    Horizontals are compared SORTED, because get_bbox_3d_oriented returns [edge1, edge2, z] in
    whatever order the hull produced. Comparing positionally made acceptance depend on hull
    ordering, so the same box rotated a quarter turn could pass or fail. Z is never permuted."""
    horizontal = sorted(extent[:2], reverse=True)
    prior_horizontal = sorted(prior[:2], reverse=True)
    return (horizontal[0] <= prior_horizontal[0]
            and horizontal[1] <= prior_horizontal[1]
            and extent[2] <= prior[2])


def percentile_index_search_binary(sorted_weights, percentile):
    """

First index (into ascending-sorted weights) where cumulative weight passes `percentile`."""
    total_weight = np.sum(sorted_weights)
    percentile_weight = total_weight * percentile
    current_weight = 0
    i = 0
    while i < len(sorted_weights) and current_weight < percentile_weight:
        current_weight += sorted_weights[i]
        i += 1
    return i


def get_box_3d(points, weights=None, voxel_size=0.0):
    """

World-axis-aligned box over `points`, as (centre, extent, identity quaternion).

    `voxel_size` is a FLOOR on each extent, not a margin added to it. A voxel is a CELL, not
    a point, so min/max over centres collapses a one-voxel-thick object to exactly zero
    volume (windows published [0.0, 0.52, 0.78]) — clamping fixes that. Adding the cell to
    every axis instead fixed it by inflating every object: at voxel_size 0.05 a 0.20 x 0.15 x
    0.03 book was published 0.25 x 0.20 x 0.08, 4.4x its true volume, which caps its IoU
    against the ground truth at 0.23 and puts a perfectly-detected book below the 0.25 the
    challenge scores. Objects a metre across never noticed; the small ones category 2 asks
    about are the whole population.

    `weights` is accepted and ignored — weighting the faces was measured 4x worse."""
    xyz = points[:, :3]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    return (lo + hi) / 2, np.maximum(hi - lo, voxel_size), [0.0, 0.0, 0.0, 1.0]


def get_bbox_3d_oriented(points, voxel_size=0.0):
    """Yaw-aligned box from a minimum-area rectangle over the XY footprint.

    Falls back to an axis-aligned footprint when ConvexHull degenerates — a window is 4 cm
    thick, so its projection is collinear and the hull raises. That fallback is why planar
    objects publish at all. `voxel_size` is a floor on every extent (see get_box_3d)."""
    bbox2d, _ = minimum_bounding_rectangle(points[:, :2])
    if bbox2d is None:
        if points.shape[0] == 0:
            return None, None, None
        bbox2d = _axis_aligned_footprint(points)
    center2d = np.mean(bbox2d, axis=0)
    edge1, edge2 = bbox2d[1] - bbox2d[0], bbox2d[2] - bbox2d[1]
    edge1_length, edge2_length = np.linalg.norm(edge1), np.linalg.norm(edge2)
    longest_edge = edge1 if edge1_length > edge2_length else edge2
    q = Rotation.from_euler("z", math.atan2(longest_edge[1], longest_edge[0])).as_quat()
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    extent = np.maximum(np.array([edge1_length, edge2_length, z_hi - z_lo]), voxel_size)
    # Expansion is symmetric, so the centre is the midpoint either way.
    center = np.array([center2d[0], center2d[1], (z_lo + z_hi) / 2])
    return center, extent, q


def _axis_aligned_footprint(points):
    """

4 corners of the XY axis-aligned rectangle, in the winding get_bbox_3d_oriented
    expects (edge1 = corner1-corner0 along x, edge2 = corner2-corner1 along y)."""
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def minimum_bounding_rectangle(points):
    """

Rotating-calipers minimum-area rectangle around a 2D point set. Returns (None, None) on
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
    """

Per-voxel observation count + which viewing-angle bins have seen it (confidence signal —
    see docs/M2_perception.md 2.6, "voxel voting")."""

    def __init__(self, voxels: np.ndarray, voxel_size: float, odom_R, odom_t, num_angle_bin=15):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.num_angle_bin = num_angle_bin
        self.tree = cKDTree(voxels)
        self.vote = np.ones(voxels.shape[0])
        self.observation_angles = np.zeros([voxels.shape[0], num_angle_bin])
        # World frame, consistently. The original rotated ONLY this first batch into the
        # body frame (apply_rotation=True) while update() and reproject_obs_angle() used the
        # world frame, so an object's first observation landed in a different bin convention
        # from every subsequent one. Angle-bin diversity is the confidence signal behind
        # voxel selection, the centroid and (now) the box faces, so a mixed convention
        # inflates apparent diversity for the founding voxels and understates it after.
        obs_angles = self._obs_angle_bins(voxels, odom_t)
        self.observation_angles[np.arange(voxels.shape[0]), obs_angles] = 1
        self.regularized_voxel_mask = np.zeros(voxels.shape[0], dtype=bool)

    def _obs_angle_bins(self, voxels, odom_t):
        """

Which viewing-direction bin each voxel was seen from, in WORLD frame.

        The `apply_rotation` switch this used to carry is gone deliberately: it existed
        only so the constructor could bin in the body frame while every other caller binned
        in the world frame, which is the inconsistency described in __init__.
        """
        voxel_to_odom = voxels - odom_t
        angles = np.arctan2(voxel_to_odom[:, 1], voxel_to_odom[:, 0])
        return discretize_angles(normalize_angles_to_pi(angles), self.num_angle_bin)

    def update(self, voxels, odom_R, odom_t):
        obs_angles = self._obs_angle_bins(voxels, odom_t)
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
        """

Keep only the voxels where `mask` is True — used by SingleObject.pop() (dormant
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

        odom_t = -R_w2b.T @ t_w2b
        obs_angles = self._obs_angle_bins(self.voxels[voxels_mask], odom_t)
        self.observation_angles[voxels_mask, obs_angles] = 1

    def angle_bin_coverage(self):
        """Which viewing-azimuth bins have EVER seen this object — the OR over all its voxels.

        Deliberately over every voxel rather than the diversity-trimmed survivors that
        retrieve_valid_voxel_indices returns. Those two answer different questions: the trim
        asks "is this voxel's shape trustworthy", this asks "which directions have we still
        never looked from", which is what an exploration planner needs in order to pick where
        to stand next. Using the trimmed set would also let the answer *shrink* whenever
        regularize_shape rejects a cluster, and a coverage signal that goes backwards makes a
        planner oscillate between goals it thinks it has just un-satisfied.

        Bin k spans [-pi + k*2pi/n, -pi + (k+1)*2pi/n) in the WORLD frame, matching
        _obs_angle_bins, so bin k inverts to a standing position at
        center - r * (cos theta_k, sin theta_k).
        """
        if self.observation_angles.shape[0] == 0:
            return np.zeros(self.num_angle_bin, dtype=bool)
        return np.any(self.observation_angles > 0, axis=0)

    def retrieve_valid_voxel_indices(self, diversity_percentile=0.3, regularized=True):
        obs_angles = self.observation_angles[self.regularized_voxel_mask] if regularized else self.observation_angles
        if len(obs_angles) == 0:
            return np.empty(0, dtype=int)

        angle_diversity = np.sum(obs_angles, axis=1)
        sorted_indices = np.argsort(angle_diversity)  # smaller to larger
        percentile_index = percentile_index_search_binary(angle_diversity[sorted_indices], 1 - diversity_percentile)
        return sorted_indices[percentile_index:]


class SingleObject:
    """

One tracked 3D instance — a voxel-voted point cluster plus its class/id/lifecycle state."""

    def __init__(self, class_id, obj_id, voxels, voxel_size, odom_R, odom_t, mask, stamp, num_angle_bin=15,
                 config=None):
        self.class_id = {class_id: 1}
        self.obj_id = [obj_id]
        # Defaults match the module-level constants, so a SingleObject built without a
        # config (tests, ad-hoc tooling) behaves exactly as before.
        self.config = config if config is not None else MappingConfig()
        self.vote_stat = VoteStatistics(voxels, voxel_size, odom_R, odom_t, num_angle_bin)
        #: Which world azimuths the ROBOT has stood in, relative to this object as a whole.
        #: Distinct from vote_stat.observation_angles, which is per VOXEL: standing 1.8 m from
        #: a 2 m sofa spans 71 degrees of voxel azimuth, so the per-voxel OR marks four of the
        #: twenty bins from a single pose and an exploration planner reads that as "circled".
        #: One bin per observation is the honest answer to "which side have I looked from".
        self.view_bins = np.zeros(num_angle_bin, dtype=bool)
        self._mark_view_bin(odom_t)

        self.life = 0
        self.inactive_frame = -1
        self.latest_stamp = stamp
        self.info_frames_cnt = 1

        self.valid_indices = None
        self.valid_indices_regularized = None
        self.clustering_labels = None
        self.regularize_rejections = {"weight": 0, "hull_failed": 0, "exceeds_prior": 0,
                                      "accepted": 0, "trimmed_voxels": 0, "trim_failed": 0}

        self.req_clustering = True
        self.req_shape_regularization = True
        self.req_recompute_indices = True
        #: Points refused by the merge-time class-prior gate, cumulative. Surfaced through
        #: describe_objects so the replay bench can see how much bleed it is catching.
        self.prior_gate_dropped = 0

    def _prior_gate(self, voxels):
        """Which incoming points can belong to an object of this class, given what it is.

        D3 already holds every class to a size cap, but only ever as a whole-cluster verdict:
        `regularize_shape` accepts a cluster or rejects it entire, so a bled object is either
        kept with its bleed or thrown away with its body. This is the same cap applied to the
        individual points instead, at the moment they would join.

        Per axis, the object's span may not exceed its class prior. With the current body
        spanning [lo, hi], a point is admissible only in `[hi - prior, lo + prior]`: it may
        extend the box in either direction, but never past what an object of this class can
        physically measure. The bound is monotone, so however long a run goes on, an object
        cannot grow past its cap -- which is what stops a 5 cm-resolution voxel chain walking
        from a picture into the wall behind it and taking the box with it.

        Anchored on the REGULARIZED body when there is one: that is the clean core DBSCAN and
        the prior already agreed on, so the gate measures from good geometry rather than from
        a span that bleed may already have inflated. Falls back to the raw voxels before the
        first successful regularization, where the cap still bounds the damage.

        NOTE the cap is a p90 UPPER bound across every VLA-3D instance of the class, so it is
        loose for classes with a wide size range -- `sofa` allows 5.39 m because sectionals
        exist. This gate bites hardest exactly where the targets are smallest, which is where
        a stray point moves the centroid most.
        """
        cfg = self.config.dimension_priors
        if not (cfg.enabled and cfg.gate_merge) or voxels.shape[0] == 0:
            return np.ones(voxels.shape[0], dtype=bool)
        body = self.vote_stat.voxels
        if self.vote_stat.regularized_voxel_mask.any():
            body = body[self.vote_stat.regularized_voxel_mask]
        if body.shape[0] == 0:
            return np.ones(voxels.shape[0], dtype=bool)
        prior = np.asarray(cfg.for_label(self.get_dominant_label()), dtype=float)
        lo, hi = body.min(axis=0), body.max(axis=0)
        return np.all((voxels >= hi - prior) & (voxels <= lo + prior), axis=1)

    def _mark_view_bin(self, odom_t):
        """Record the azimuth bin the robot occupied for THIS observation.

        Binned on the vector robot -> object centre, i.e. the same convention as
        VoteStatistics._obs_angle_bins, so bin k inverts to a standing position at
        `centre - r * (cos theta_k, sin theta_k)` exactly as angle_bin_coverage documents.

        Skipped beyond range_filter.max_distance: past that no lidar point is ever assigned to
        this object's mask, so the frame cannot improve its geometry and calling that side
        "inspected" would retire a viewpoint the planner still needs.
        """
        centre = self.provisional_centroid()
        if centre is None or odom_t is None:
            return
        offset = np.asarray(centre)[:2] - np.asarray(odom_t)[:2]
        rf = self.config.range_filter
        if rf.enabled and float(np.linalg.norm(offset)) > rf.max_distance:
            return
        angle = normalize_angles_to_pi(np.arctan2(offset[1], offset[0]))
        self.view_bins[int(discretize_angles(angle, self.vote_stat.num_angle_bin))] = True

    def merge(self, voxels, odom_R, odom_t, label, stamp):
        """Fold one frame's points into this object. Returns how many the prior gate refused.

        The gate runs BEFORE vote_stat.update so refused points never reach the accumulation
        at all -- they are not merely excluded from the box, they never vote, never seed a
        DBSCAN cluster, and never drag the provisional centroid. Filtering after the fact
        would leave every one of those effects in place.
        """
        keep = self._prior_gate(voxels)
        dropped = int(voxels.shape[0] - np.count_nonzero(keep))
        self.prior_gate_dropped += dropped
        voxels = voxels[keep]
        # Still count the frame even if the gate emptied it: the object WAS seen, and
        # info_frames is what admission and the exploration planner read as "seen across
        # frames". Only the geometry is refused, not the observation.
        if voxels.shape[0]:
            self.vote_stat.update(voxels, odom_R, odom_t)
            self.req_clustering = self.req_shape_regularization = True
            self.req_recompute_indices = True
        self._mark_view_bin(odom_t)
        self.info_frames_cnt += 1
        self.latest_stamp = stamp
        self.class_id[label] = self.class_id.get(label, 0) + 1
        return dropped

    def merge_object(self, other: "SingleObject"):
        self.obj_id.extend(other.obj_id)
        self.vote_stat.update_through_vote_stat(other.vote_stat)
        # Union, like the per-voxel bins: a world merge means both id's observations were of
        # the same physical object, so the sides the loser was seen from are sides we have
        # genuinely stood in and must not have to walk to again.
        self.view_bins = np.logical_or(self.view_bins, other.view_bins)
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

    #: Rejects strays attached to ONE object, not instance separation — so err generous.
    #: Values live in mapping_config.DbscanConfig.

    def dbscan_cluster_params(self):
        cfg = self.config.dbscan
        return self.vote_stat.voxel_size * cfg.eps_voxels, cfg.min_points

    def cal_clusters(self):
        if self.req_clustering:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.vote_stat.voxels)
            eps, min_points = self.dbscan_cluster_params()
            self.clustering_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
            self.req_clustering = False

    def _trim_to_prior(self, valid_mask, cluster_mask, dim_prior):
        """Shave a cluster's outermost voxels until it fits its class cap. D3b2.

        Returns `(kept_mask, n_trimmed)`, or `(None, n)` when nothing survives.

        The alternative -- what this replaces -- is discarding the cluster whole, and whole
        means the object can disappear: regularized voxels reach zero, `infer_centroid` returns
        None, and `serialize_map_to_dict` skips it. hotel_room_1 lost one of its two bedside
        tables exactly so, because a lamp standing 3 cm from the table's centre pushed the
        cluster past the 1.14 m `bedsidetable` z cap. A box 0.2 m too big scores the
        constraint; an absent object cannot be answered about at all.

        Voxels are dropped furthest-first from the body already accepted -- bleed is what
        reaches away from the object, so distance from its trusted core is the ordering that
        sheds bleed before geometry. Before there is a trusted core the cluster anchors on
        itself, which still sheds its own outliers.

        Binary search over the kept prefix rather than dropping one voxel at a time: each test
        is a convex hull, and a linear walk would be thousands of them per cluster. The search
        is sound because the predicate is monotone -- adding voxels to a box can only grow it,
        never shrink it.
        """
        voxels = self.vote_stat.voxels
        order_src = np.flatnonzero(cluster_mask)
        if order_src.size == 0:
            return None, 0
        anchored = valid_mask if valid_mask.any() else cluster_mask
        weights = np.sum(self.vote_stat.observation_angles[anchored], axis=1)
        body = voxels[anchored]
        centre = (np.average(body, axis=0, weights=weights)
                  if float(np.sum(weights)) > 0.0 else body.mean(axis=0))
        order = order_src[np.argsort(np.linalg.norm(voxels[order_src] - centre, axis=1))]

        def fits(k):
            trial = valid_mask.copy()
            trial[order[:k]] = True
            if not trial.any():
                return False
            _, extent, _ = get_bbox_3d_oriented(voxels[trial])
            return extent is not None and _fits_prior(extent, dim_prior)

        # A single voxel fits any positive cap, so with no accepted body yet the floor is 1.
        low = 0 if valid_mask.any() else 1
        high = order.size
        if not fits(low):
            return None, order.size
        while high - low > 1:
            mid = (low + high) // 2
            if fits(mid):
                low = mid
            else:
                high = mid
        if low == 0:
            return None, order.size
        kept = np.zeros_like(cluster_mask)
        kept[order[:low]] = True
        return kept, int(order.size - low)

    def regularize_shape(self, percentile=None):
        """

Cluster voxels (DBSCAN), then keep clusters — largest observation-weight first — that
        fit the class's DIMENSION_PRIORS, up to `percentile` of total weight."""
        if not self.req_shape_regularization:
            return
        self.cal_clusters()
        unique_labels = np.unique(self.clustering_labels)
        priors_cfg = self.config.dimension_priors
        dim_prior = priors_cfg.for_label(self.get_dominant_label())

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
        # Why clusters get rejected. An object can hold a clean DBSCAN cluster and still end
        # up with zero regularized voxels — which makes infer_centroid return None and the
        # object silently unpublished. The three causes need different fixes, so count them
        # rather than infer: a planar object (a window is 4 cm thick) fails ConvexHull, a
        # bleeding one exceeds the class prior, a sparse one falls under the weight floor.
        self.regularize_rejections = {"weight": 0, "hull_failed": 0, "exceeds_prior": 0,
                                      "accepted": 0}
        for weight_index in reversed(np.argsort(cluster_weights)):
            # drop tiny clusters (odom noise, bleed-through)
            if cluster_weights[weight_index] < self.config.cluster_weight_min:
                self.regularize_rejections["weight"] += 1
                continue
            attempt = np.logical_or(valid_mask, cluster_masks[weight_index])
            _, extent, _ = get_bbox_3d_oriented(self.vote_stat.voxels[attempt])
            if extent is None:
                self.regularize_rejections["hull_failed"] += 1
                continue
            if priors_cfg.enabled and not _fits_prior(extent, dim_prior):
                # Counted whether or not the trim rescues it: this stays the measure of how
                # often bleed pushes a cluster past its cap.
                self.regularize_rejections["exceeds_prior"] += 1
                if not priors_cfg.trim_to_fit:
                    continue
                kept, trimmed = self._trim_to_prior(
                    valid_mask, cluster_masks[weight_index], dim_prior)
                # "Nothing here" has to stay reachable: a cluster that is all bleed should
                # still be refused, and the weight floor is the same one the loop opens with.
                if kept is None or (np.sum(self.vote_stat.observation_angles[kept])
                                    < self.config.cluster_weight_min):
                    self.regularize_rejections["trim_failed"] += 1
                    continue
                self.regularize_rejections["trimmed_voxels"] += trimmed
                attempt = np.logical_or(valid_mask, kept)
            self.regularize_rejections["accepted"] += 1
            valid_mask = attempt
            if percentile is not None:
                current_weight += cluster_weights[weight_index]
                if current_weight > percentile * total_weight:
                    break

        self.vote_stat.regularized_voxel_mask = valid_mask
        self.req_recompute_indices = True
        self.req_shape_regularization = False

    def pop(self, mask):
        """

Split off the voxels NOT in `mask`, for the dormant IoU-overlap voxel-exchange path
        (docs backlog §6 item 4, object_mapper.py) — not currently called."""
        voxels_pop = self.vote_stat.voxels[~mask]
        obs_angles_pop = self.vote_stat.observation_angles[~mask]
        votes_pop = self.vote_stat.vote[~mask]
        self.vote_stat.update_through_mask(mask)
        self.req_clustering = self.req_shape_regularization = self.req_recompute_indices = True
        return voxels_pop, obs_angles_pop, votes_pop

    def add(self, voxels, obs_angles, votes):
        """

Counterpart to pop() — also dormant, same feature."""
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

    def observed_angle_bins(self):
        """Length-num_angle_bin bool vector: which world azimuths this object has been seen
        from. See VoteStatistics.angle_bin_coverage for why it is not diversity-trimmed."""
        return self.vote_stat.angle_bin_coverage()

    def observed_view_bins(self):
        """Length-num_angle_bin bool vector: which world azimuths the ROBOT has stood in.

        The planner-facing counterpart of observed_angle_bins. That one answers "has this
        voxel's shape been triangulated"; this one answers "have I looked at this thing from
        that side", which is the only question a viewpoint planner can act on.
        """
        return self.view_bins.copy()

    def provisional_centroid(self):
        """A position for an object infer_centroid cannot place.

        infer_centroid returns None when the surviving voxels carry zero observation-angle
        weight, and such an object is skipped by serialize_map_to_dict AND by world merge
        (both the source and the target branch test for a non-None centroid) -- so it is
        invisible to every consumer and never deduplicated. It is still a real cluster of
        lidar points, and it is precisely the object exploration needs to go look at again,
        so the plain voxel mean is enough to aim at. Never used when infer_centroid succeeds.
        """
        voxels = self.vote_stat.voxels
        if voxels.shape[0] == 0:
            return None
        return np.mean(voxels, axis=0)

    def infer_bbox(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        voxels, obs_angles = self.vote_stat.voxels, self.vote_stat.observation_angles
        if regularized:
            mask = self.vote_stat.regularized_voxel_mask
            idx = self.valid_indices_regularized
            voxels, obs_angles = voxels[mask][idx], obs_angles[mask][idx]
        else:
            voxels, obs_angles = voxels[self.valid_indices], obs_angles[self.valid_indices]
        if not len(voxels):
            return None
        # Same weight infer_centroid uses: a voxel seen from many directions is more
        # trustworthy than one glimpsed once, so it should carry more say over the faces.
        return get_box_3d(voxels, weights=np.sum(obs_angles, axis=1),
                          voxel_size=self.vote_stat.voxel_size)

    def infer_bbox_oriented(self, diversity_percentile, regularized=True):
        self.compute_valid_indices(diversity_percentile)
        voxels = self.vote_stat.voxels
        if regularized:
            voxels = voxels[self.vote_stat.regularized_voxel_mask][self.valid_indices_regularized]
        else:
            voxels = voxels[self.valid_indices]
        return (get_bbox_3d_oriented(voxels, voxel_size=self.vote_stat.voxel_size)
                if len(voxels) else None)
