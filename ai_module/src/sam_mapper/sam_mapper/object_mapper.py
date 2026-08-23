"""3D instance mapper: fuses per-frame 2D detections + lidar into persistent tracked objects.

sam_mapper calls. Every threshold is configurable — see mapping_config.MappingConfig
and the `mapping:` block of the node yaml.

Kept dormant: the `captioner` param and its branches. map_node passes captioner=None."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from sam_mapper import box_geometry
from sam_mapper.challenge_marker import quaternion_to_yaw
from sam_mapper.mapping_config import MappingConfig
from sam_mapper.ros_markers import create_colored_point_cloud, create_text_marker, create_wireframe_marker
from sam_mapper.single_object import SingleObject

VERTICAL_OBJECTS = ["door", "painting"]  # merge candidates must agree on this (see the loop below)


class AdjacencyGraph:
    """Co-visibility: which object ids were seen in the same frame. Read by the D8 guard."""

    def __init__(self):
        self.adjacency_list: dict[int, list[int]] = {}

    def add_vertex(self, vertex):
        self.adjacency_list.setdefault(vertex, [])

    def add_edge(self, v1, v2):
        self.add_vertex(v1)
        self.add_vertex(v2)
        if v2 not in self.adjacency_list[v1]:
            self.adjacency_list[v1].append(v2)
        if v1 not in self.adjacency_list[v2]:
            self.adjacency_list[v2].append(v1)

    def is_adjacent(self, v1, v2):
        # .get, not [] — an id can reach the merge check without ever having been added as a
        # vertex (a detection dropped before the adjacency update still carries its id).
        return v2 in self.adjacency_list.get(v1, ())

    def is_set_adjacent(self, v_set1, v_set2):
        return any(self.is_adjacent(id1, id2) for id1 in v_set1 for id2 in v_set2)


class ObjMapper:
    """Every threshold below is configurable — see `mapping_config.MappingConfig` and the
    `mapping:` block of the node yaml. `just map3d-replay` loads the same file, so tuning
    is a config edit, not a code edit."""

    #: px per radian on the equirect panorama; mirrors cloud_image_fusion._scan2pixels.
    _PIXELS_PER_RADIAN = 1920 / (2 * np.pi)

    def _overlap(self, obj_a, obj_b) -> float:
        """Intersection over the SMALLER box's volume, for the D8 co-visibility guard.

        Not IoU: a 0.2 m fragment inside a 2 m sofa scores 0.01 IoU but 1.0 here, and
        "contained in" is the question. Unfittable boxes return 0.0, which keeps the merge
        blocked — a wrong merge is unrecoverable, a missed one is retried next frame.
        """
        box_a = self._published_box(obj_a)
        box_b = self._published_box(obj_b)
        if box_a is None or box_b is None or box_a[1] is None or box_b[1] is None:
            return 0.0
        # Yaw matters: with publish_oriented_box on, comparing axis-aligned footprints of
        # rotated boxes would silently change what the 0.7 threshold means.
        return box_geometry.overlap_fraction(
            box_a[0], box_a[1], box_b[0], box_b[1],
            quaternion_to_yaw(box_a[2]), quaternion_to_yaw(box_b[2]))

    def _claim_table(self):
        """(labels, lo, hi, volume) for every tracked object with enough voxels to claim.

        Built ONCE per frame and reused across the frame's detections: update_map loops over
        detections, and rebuilding this per detection would make the cost quadratic in objects.

        The raw accumulated-voxel AABB, deliberately, not `_published_box` -- that one runs
        regularize_shape, which runs DBSCAN, and this is consulted per object per detection.
        """
        cfg = self.config.claimed_volume
        labels, lows, highs, volumes = [], [], [], []
        for obj in self.single_obj_list:
            voxels = obj.vote_stat.voxels
            if voxels.shape[0] < cfg.min_voxels:
                continue
            lo, hi = voxels.min(axis=0), voxels.max(axis=0)
            centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * cfg.shrink
            labels.append(obj.get_dominant_label())
            lows.append(centre - half)
            highs.append(centre + half)
            # The TRUE volume, not the shrunk one. `shrink` decides how much of the box
            # claims points; it must not also make the claimant look smaller than it is, or
            # the "claimant must be smaller" test quietly loosens by shrink**3 (0.73x at 0.9).
            volumes.append(float(np.prod(hi - lo)))
        if not labels:
            return None
        return labels, np.asarray(lows), np.asarray(highs), np.asarray(volumes)

    def _claimed_volume_cut(self, obj_cloud, label, obj_id, table, stats):
        """D2b. Drop points that lie inside a SMALLER, differently-labelled object's box.

        See ClaimedVolumeConfig for the evidence and for why each condition is there. The
        starvation floor at the end is the important part: a bled object is acceptable, a
        missing one is not.
        """
        cfg = self.config.claimed_volume
        if not cfg.enabled or table is None or obj_cloud.shape[0] == 0:
            return obj_cloud
        labels, lo, hi, volume = table

        # The object being filtered must itself be established -- a young object needs its
        # founding points more than a neighbour needs its boundary respected. Its own volume
        # is what "smaller" is measured against; with no accumulation yet there is nothing to
        # compare, so the guard simply does not apply.
        mine = next((o for o in self.single_obj_list if obj_id in o.obj_id), None)
        if mine is None or mine.vote_stat.voxels.shape[0] < cfg.min_voxels:
            return obj_cloud
        my_voxels = mine.vote_stat.voxels
        my_volume = float(np.prod(my_voxels.max(axis=0) - my_voxels.min(axis=0)))

        claimant = np.array([lbl != label and vol < my_volume
                             for lbl, vol in zip(labels, volume)])
        if not claimant.any():
            return obj_cloud

        xyz = obj_cloud[:, :3]
        # (N, M) containment against every claiming box at once.
        inside = np.all((xyz[:, None, :] >= lo[None, claimant, :])
                        & (xyz[:, None, :] <= hi[None, claimant, :]), axis=2)
        keep = ~inside.any(axis=1)
        dropped = int((~keep).sum())
        if not dropped:
            return obj_cloud

        # Unconditional, not a knob. Taking this detection under B8 would cost the OBJECT,
        # not just its bleed: below min_points it never creates, never clusters, never gets a
        # centroid, and serialize_map_to_dict drops it. A bled object is an acceptable
        # outcome; a missing one is not, and no measurement would justify trading one for the
        # other -- so there is nothing here to tune.
        if int(keep.sum()) < self.config.min_points_per_detection:
            stats["claim_declined_would_starve"] += 1
            return obj_cloud

        stats["claimed_by_other_object"] += dropped
        return obj_cloud[keep]

    def _range_gap_cut(self, obj_cloud, robot_xyz, mask, stats):
        """B7 - drop points sitting far behind the object their mask claimed."""
        cfg = self.config.range_gap
        if not cfg.enabled or obj_cloud.shape[0] == 0:
            return obj_cloud

        ranges = np.linalg.norm(obj_cloud[:, :3] - robot_xyz, axis=1)
        order = np.argsort(ranges)
        sorted_ranges = ranges[order]
        front = sorted_ranges[0]
        cut = np.inf

        # Geometric ceiling: an object is not much deeper than it is wide. Works at any
        # point count, so it is the fallback when the gap search has too little data.
        if cfg.max_depth_scale > 0 and mask is not None:
            cols = np.flatnonzero(mask.any(axis=0))
            rows = np.flatnonzero(mask.any(axis=1))
            if cols.size and rows.size:
                span_px = max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1)
                apparent_width = front * (span_px / self._PIXELS_PER_RADIAN)
                cut = front + cfg.max_depth_scale * apparent_width

        # Largest depth discontinuity above min_gap is the object/background boundary.
        # `>= 2` is not implied by min_points (configurable to 1) and np.diff of one range
        # is empty, which argmax cannot take.
        if obj_cloud.shape[0] >= max(cfg.min_points, 2):
            gaps = np.diff(sorted_ranges)
            widest = int(np.argmax(gaps))
            if gaps[widest] >= cfg.min_gap:
                cut = min(cut, sorted_ranges[widest])

        if not np.isfinite(cut):
            return obj_cloud
        keep = ranges <= cut
        dropped = int(obj_cloud.shape[0] - keep.sum())
        if dropped:
            stats["points_dropped_range_gap"] += dropped
            stats["detections_trimmed_range_gap"] += 1
        return obj_cloud[keep]

    def _erosion_iterations(self, mask) -> int:
        """How hard to shrink one mask, from its own size. 10% of the smaller side.

        Reads the BOUNDING-BOX short side, not local thickness, so a sliver can score
        "gentle, 1 iteration" and still be annihilated (N iterations leave w - 2N). Counted
        as `erased_by_erosion`.
        """
        cfg = self.config.erosion
        if not cfg.enabled:
            return 0
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if not rows.size or not cols.size:
            return 0
        short_side = min(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
        return int(np.clip(round(cfg.fraction * short_side),
                           cfg.min_iterations, cfg.max_iterations))

    def __init__(self, cloud_image_fusion, label_template, captioner=None, log_info=print,
                 config=None):
        self.single_obj_list: list[SingleObject] = []
        self.background_obj_list: list[SingleObject] = []
        self.adjacency_graph = AdjacencyGraph()

        # Per-label funnel: every stage that can discard a detection counts here, so
        # "where did the pillows go" is a lookup rather than an argument.
        self.funnel = defaultdict(lambda: defaultdict(int))

        self.cloud_image_fusion = cloud_image_fusion
        self.captioner = captioner
        self.log_info = log_info

        # Defaults reproduce the values the 13-scene bench was last measured at, so
        # constructing without a config (tests, ad-hoc tooling) is unchanged behaviour.
        self.config = config if config is not None else MappingConfig()

        # Kept as plain attributes because they are read on nearly every line below; the
        # config object remains the single source of truth for their values.
        self.voxel_size = self.config.voxel_size
        self.confidence_thres = self.config.confidence_threshold
        self.cloud_to_odom_dist_thres = self.config.range_filter.max_distance
        self.num_angle_bin = self.config.num_angle_bins
        self.percentile_thresh = self.config.percentile_threshold

        instance_labels = [label for label, val in label_template.items() if val["is_instance"]]
        self.log_info(f"Instance level objects: {instance_labels}")
        self.log_info(f"label template: {label_template}")

    def update_map(self, detections, detection_stamp, detection_odom, cloud, image=None):
        R_b2w = Rotation.from_quat(detection_odom["orientation"]).as_matrix()
        t_b2w = np.array(detection_odom["position"])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        cloud_body = cloud @ R_w2b.T + t_w2b

        for label in detections["labels"]:
            self.funnel[str(label)]["detections_in"] += 1

        keep = np.array(detections["confidences"]) >= self.confidence_thres
        for label, k in zip(detections["labels"], keep):
            if not k:
                self.funnel[str(label)]["dropped_low_confidence"] += 1
        masks = [m for m, k in zip(detections["masks"], keep) if k]
        labels = [l for l, k in zip(detections["labels"], keep) if k]
        obj_ids = [i for i, k in zip(detections["ids"], keep) if k]
        bboxes = [b for b, k in zip(detections["bboxes"], keep) if k]
        if not obj_ids:
            return

        # B2. Off by default — see ErosionConfig. When on, the shrink is proportional to
        # the mask: a fixed 5 px left 79% of a sofa mask but only 21% of a pillow's.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = []
        for label, mask in zip(labels, masks):
            before = int(np.count_nonzero(mask))
            iterations = self._erosion_iterations(mask)
            shrunk = (cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
                      if iterations else mask)
            after = int(np.count_nonzero(shrunk))
            stats = self.funnel[str(label)]
            stats["mask_px_before_erosion"] += before
            stats["mask_px_after_erosion"] += after
            stats["erosion_iterations"] += iterations
            if before and not after:
                stats["erased_by_erosion"] += 1
            eroded.append(shrunk)
        masks = eroded

        if len(obj_ids) == 1:
            self.adjacency_graph.add_vertex(obj_ids[0])
        else:
            for i in range(len(obj_ids)):
                for j in range(i + 1, len(obj_ids)):
                    self.adjacency_graph.add_edge(obj_ids[i], obj_ids[j])

        # B3 before the projection. Identical output — ||cloud_body|| == ||cloud_world -
        # t_b2w|| since rotation preserves norm — but generate_seg_cloud runs one pass over
        # the cloud PER MASK, so shrinking it first saves N x M. Pinned by
        # test_projection.py::test_range_filter_commutes_with_projection.
        self.funnel["_frame"]["cloud_points_in"] += int(cloud_body.shape[0])
        if self.config.range_filter.enabled:
            in_range = np.linalg.norm(cloud_body[:, :3], axis=1) < self.cloud_to_odom_dist_thres
            self.funnel["_frame"]["cloud_points_beyond_range"] += int((~in_range).sum())
            cloud_body = cloud_body[in_range]

        obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, R_b2w, t_b2w)

        # D2b, once per frame rather than once per detection -- see _claim_table.
        claim_table = self._claim_table() if self.config.claimed_volume.enabled else None

        for i, obj_cloud in enumerate(obj_clouds_world):
            stats = self.funnel[str(labels[i])]
            # No per-label range-drop count any more: B3 runs before masks exist, so it
            # cannot be attributed to a class. The frame-level pair above replaces it.
            stats["points_in_mask"] += int(obj_cloud.shape[0])

            # B7, before the min-points gate: a detection whose points are ALL background
            # should fail the gate, not pass it on the strength of the background.
            obj_cloud = self._range_gap_cut(obj_cloud, t_b2w, masks[i], stats)

            # D2b, here rather than in the merge/create branches below: those are asymmetric
            # (merge takes the RAW cloud, create the downsampled one), so filtering once at the
            # single point they share is the only way to keep one implementation.
            obj_cloud = self._claimed_volume_cut(obj_cloud, labels[i], obj_ids[i],
                                                 claim_table, stats)

            if obj_cloud.shape[0] < self.config.min_points_per_detection:
                # ZERO points is a coverage problem, a HANDFUL is a resolution problem.
                # Opposite fixes, so count them apart.
                stats["dropped_zero_points" if obj_cloud.shape[0] == 0
                      else "dropped_sparse_points"] += 1
                continue

            class_id, obj_id = labels[i], obj_ids[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_cloud[:, :3])
            pcd_downsampled = (pcd.voxel_down_sample(voxel_size=self.voxel_size)
                               if self.config.voxel_downsample.enabled else pcd)
            if pcd_downsampled.is_empty():
                stats["dropped_empty_after_voxel"] += 1
                continue
            outlier_cfg = self.config.outlier_removal
            if outlier_cfg.enabled:
                pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(
                    nb_neighbors=outlier_cfg.nb_neighbors, std_ratio=outlier_cfg.std_ratio)
            if pcd_downsampled.is_empty():
                stats["dropped_empty_after_outlier"] += 1
                continue
            stats["voxels_at_creation"] += len(pcd_downsampled.points)

            merged = False
            if obj_id >= 0:
                for single_obj in self.single_obj_list:
                    if obj_id in single_obj.obj_id:
                        # RAW pcd, deliberately. Create uses the downsampled cloud, merge
                        # uses the raw one — asymmetric, and it looks like a bug. Both
                        # obvious fixes measured 4x WORSE (bestIoU 0.031 -> 0.007): voxel
                        # downsampling starves the accumulation, because every downstream
                        # threshold is tuned to raw-merge point counts.
                        #
                        # So defect 5 cannot be fixed in isolation — it needs those
                        # thresholds retuned together, or the global voxel hash (Tier 2.1)
                        # where the raw/downsampled distinction stops existing. Growth is
                        # tolerable meanwhile: update_map p50 is 194 ms against SAM 3's
                        # ~18 s/frame, so the mapper is ~1% of the run budget.
                        dropped = single_obj.merge(np.array(pcd.points), R_b2w, t_b2w,
                                                   class_id, detection_stamp)
                        # D3b. Points refused for pushing the object past its class cap —
                        # the bleed that used to end up inside the box. A count that climbs
                        # with no drop in voxels_valid is the gate doing its job; one that
                        # eats real geometry shows up as voxels_valid falling instead.
                        stats["prior_gate_dropped"] += dropped
                        single_obj.reproject_obs_angle(R_w2b, t_w2b, masks[i], self.cloud_image_fusion.scan2pixels)
                        single_obj.inactive_frame = -1
                        merged = True
                        stats["merged_into_existing"] += 1
                        break
            # else: background merge is intentionally not attempted — background objects are
            # re-created every frame (no persistent identity), matching current behavior.

            if not merged:
                stats["created_new"] += 1
                target_list = self.single_obj_list if obj_id >= 0 else self.background_obj_list
                target_list.append(SingleObject(class_id, obj_id, np.array(pcd_downsampled.points),
                    self.voxel_size, R_b2w, t_b2w, masks[i], detection_stamp,
                    num_angle_bin=self.num_angle_bin, config=self.config))

        # ===== dormant: captioner crop update (see module docstring) =====
        if self.captioner is not None and image is not None:
            obj_ids_updated, bboxes_2d, centroids_3d, bboxes_3d, class_names = [], [], [], [], []
            for i, obj_id in enumerate(obj_ids):
                if obj_id < 0:
                    continue
                for single_obj in self.single_obj_list:
                    if obj_id in single_obj.obj_id:
                        cent_3d = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                        if cent_3d is not None:
                            obj_ids_updated.append(single_obj.obj_id[0])
                            bboxes_2d.append(bboxes[i])
                            centroids_3d.append(cent_3d)
                            class_names.append(single_obj.get_dominant_label())
                            bboxes_3d.append(single_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True))
                        break
            self.captioner.update_object_crops(
                rgb=torch.from_numpy(image).cuda().flip((-1)),
                bboxes_2d=bboxes_2d, obj_ids_global=obj_ids_updated,
                centroids_3d=centroids_3d, class_names=class_names, bboxes_3d=bboxes_3d,
            )

        # ===== associate objects in world space (merge duplicates, drop stale ones) =====
        i = 0
        while i < len(self.single_obj_list):
            single_obj = self.single_obj_list[i]
            if single_obj.obj_id[0] < 0:  # background object, no persistent lifecycle
                i += 1
                continue

            single_obj.life += 1

            # No age-based eviction, deliberately. max_age is a tracking idea and this is
            # a static scene with a moving robot — an object unseen for 30 frames has not
            # moved, we just looked away. Measured: evicting cost 20 objects -> 12.
            # Stale objects instead fall through to the merge below, where a re-detection
            # under a new id can rejoin them. Degenerate ones go to the voxel-count prune.
            merged_obj = False
            if 5 < single_obj.life < 1000:
                # valid_indices_regularized is lazily computed and starts as None. Every
                # other reader goes through retrieve_valid_voxels/infer_centroid, which
                # force it. This used to rely on the caller having published the map first
                # to populate it as a side effect, so any caller that does not publish every
                # frame (the offline replay) hit AttributeError on frame 6.
                single_obj.compute_valid_indices(self.percentile_thresh)
                prune_cfg = self.config.prune
                if (prune_cfg.enabled
                        and single_obj.valid_indices_regularized.shape[0] < prune_cfg.min_valid_voxels
                        and single_obj.inactive_frame > prune_cfg.min_inactive_frames):
                    self.funnel[single_obj.get_dominant_label()]["pruned_too_few_voxels"] += 1
                    self.single_obj_list.remove(single_obj)
                    if self.captioner is not None:
                        self.captioner.remove_object(single_obj.obj_id[0])
                    continue

                # centroid is None means this object's voxels don't currently regularize to a
                # valid shape (e.g. too sparse/diverse) — skip merge-matching for it *this frame*
                # only. Falling through to `if not merged_obj:` below (rather than `continue`)
                # matters: `continue` here would re-enter this same `i` next iteration without
                # advancing, re-running infer_centroid on the same object every spin until
                # `life` ages past 1000 on its own.
                centroid = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                merge_cfg = self.config.world_merge
                if centroid is not None and merge_cfg.enabled:
                    source_is_vertical = single_obj.get_dominant_label() in VERTICAL_OBJECTS
                    target_obj, target_index, minimum_dist = None, -1, 1e6
                    for j, same_class_obj in enumerate(self.single_obj_list):
                        if same_class_obj.obj_id[0] < 0 or j == i:
                            continue
                        if same_class_obj.get_dominant_label() != single_obj.get_dominant_label():
                            continue
                        if (same_class_obj.get_dominant_label() in VERTICAL_OBJECTS) != source_is_vertical:
                            continue
                        # Co-visible AND disjoint -> different objects, never merge.
                        # Co-visible AND overlapping -> one object split across two ids by
                        # the tracker, so merge anyway. See WorldMergeConfig for why the
                        # overlap guard is needed and how 0.7 was chosen.
                        if merge_cfg.block_covisible and self.adjacency_graph.is_set_adjacent(
                                single_obj.obj_id, same_class_obj.obj_id):
                            if self._overlap(single_obj, same_class_obj) < merge_cfg.covisible_overlap:
                                self.funnel[single_obj.get_dominant_label()]["merge_blocked_covisible"] += 1
                                continue
                            self.funnel[single_obj.get_dominant_label()]["merge_covisible_overlapping"] += 1
                        target_centroid = same_class_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                        if target_centroid is None:
                            continue
                        dist = np.linalg.norm(target_centroid - centroid)
                        if dist < minimum_dist:
                            minimum_dist, target_obj, target_index = dist, same_class_obj, j

                    if target_obj is not None:
                        _, extent_object, _ = single_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                        _, extent_target, _ = target_obj.infer_bbox_oriented(diversity_percentile=self.percentile_thresh, regularized=True)
                        if extent_object is not None and extent_target is not None:
                            dist_thresh = (np.linalg.norm((extent_object / 2 + extent_target / 2) / 2)
                                           * merge_cfg.extent_scale)
                            if minimum_dist < dist_thresh or minimum_dist < merge_cfg.absolute_distance:
                                self.log_info(f"Merge {single_obj.class_id}:{single_obj.obj_id} to "
                                             f"{target_obj.class_id}:{target_obj.obj_id} with dist thresh {dist_thresh}")
                                merged_obj = True

                            # TODO (docs backlog §6 item 4): when not merged_obj, a commented-out path
                            # here checked 3D IoU (pytorch3d.ops.box3d_overlap on
                            # get_corners_from_box3d_torch(...)) plus AdjacencyGraph.is_set_adjacent()
                            # to catch cases plain distance can't: high IoU + never co-visible -> merge
                            # anyway (probably one object split across ids); high IoU + co-visible ->
                            # exchange only the ambiguous overlapping voxels via
                            # single_obj.pop()/target_obj.add() rather than merging whole objects.
                            # semantic_map.py (pre-split) for the original, fuller version.

                            if merged_obj:
                                if target_index < i:
                                    single_obj, target_obj = target_obj, single_obj
                                single_obj.merge_object(target_obj)
                                single_obj.inactive_frame = -1
                                # Two distinct tracked instances collapsed into one. For a
                                # class whose real objects sit closer together than the
                                # merge radius — pillows are 0.44 m apart against an
                                # unconditional `dist < 0.5` — this is where they are lost.
                                self.funnel[single_obj.get_dominant_label()]["world_merged"] += 1
                                self.single_obj_list.remove(target_obj)
                                if self.captioner is not None:
                                    centroid_target = target_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                                    bbox_3d_target = target_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
                                    self.captioner.merge_objects(single_obj.obj_id[0], target_obj.obj_id[0], centroid_target, bbox_3d_target)

            if not merged_obj:
                single_obj.inactive_frame += 1
                single_obj.regularize_shape(self.percentile_thresh)
                i += 1


    def _published_box(self, single_obj):
        """The box that goes into /obj_map_json and the answer Marker.

        Falls back to the axis-aligned box whenever the oriented fit is unavailable, so a
        degenerate object still publishes something rather than vanishing.
        """
        if self.config.publish_oriented_box:
            oriented = single_obj.infer_bbox_oriented(
                diversity_percentile=self.percentile_thresh, regularized=True)
            if oriented is not None and oriented[1] is not None:
                return oriented
        return single_obj.infer_bbox(diversity_percentile=self.percentile_thresh,
                                     regularized=True)

    def describe_objects(self) -> list[dict]:
        """Per-object internals for every TRACKED object, published or not.

        The funnel counts detections; this counts what became of the objects they built.
        An object can survive every stage and still never appear in the output, because
        serialize_map_to_dict skips anything whose centroid comes back None — so without
        this the last step of the pipeline is invisible.

        Two consumers: offline replay diagnostics, and /exploration/object_targets. The
        planner-facing fields are `center`, `extent` and `view_bins` (where to go and which
        directions are still unseen) plus `life`, `info_frames`, `voxels_total` and
        `published`, which are its admission gate — an unpublished object is exempt from world
        merge (that path needs a non-None centroid on BOTH sides), so duplicates of one real
        object can persist and the consumer has to deduplicate them itself.
        """
        out = []
        for obj in self.single_obj_list:
            obj.compute_valid_indices(self.percentile_thresh)
            regularized = int(np.count_nonzero(obj.vote_stat.regularized_voxel_mask))
            valid = (int(obj.valid_indices_regularized.shape[0])
                     if obj.valid_indices_regularized is not None else 0)
            centroid = obj.infer_centroid(diversity_percentile=self.percentile_thresh,
                                          regularized=True)
            clusters = (obj.clustering_labels if obj.clustering_labels is not None
                        else np.empty(0, dtype=int))
            center, extent = self._exploration_geometry(obj, centroid)
            bins = obj.observed_angle_bins()
            view_bins = obj.observed_view_bins()
            out.append({
                "label": obj.get_dominant_label(),
                "ids": [int(x) for x in obj.obj_id],
                "voxels_total": int(obj.vote_stat.voxels.shape[0]),
                "voxels_regularized": regularized,
                "voxels_valid": valid,
                "n_clusters": int(len(set(clusters.tolist()) - {-1})),
                "n_noise": int(np.count_nonzero(clusters == -1)),
                "life": int(obj.life),
                "inactive_frame": int(obj.inactive_frame),
                "info_frames": int(obj.info_frames_cnt),
                "published": centroid is not None,
                # Where to aim, and from which world azimuths it has already been seen.
                # `center` falls back to the provisional voxel mean precisely when
                # `published` is false, so an exploration consumer always has a position.
                "center": None if center is None else [float(v) for v in center],
                "extent": None if extent is None else [float(v) for v in extent],
                "angle_bins": [bool(b) for b in bins],
                "n_angle_bins": int(np.count_nonzero(bins)),
                # Which sides the robot has actually STOOD on -- one bin per observation,
                # against angle_bins' per-voxel OR. A planner must read this one; see
                # SingleObject.observed_view_bins.
                "view_bins": [bool(b) for b in view_bins],
                "n_view_bins": int(np.count_nonzero(view_bins)),
                # Points the class-prior gate refused at merge time. Large next to a healthy
                # voxels_valid means it is catching bleed; large next to a starved one means
                # the cap is too tight for this class and is eating the object.
                "prior_gate_dropped": int(obj.prior_gate_dropped),
                "reject": dict(obj.regularize_rejections),
            })
        return out

    def _exploration_geometry(self, obj, centroid):
        """Position and size for an exploration consumer, for ANY tracked object.

        Both fall back for objects the normal path cannot describe: the centroid to the
        provisional voxel mean, the extent to the raw voxel spread. Without the fallbacks an
        unpublished object carries no position at all and cannot be a navigation goal -- which
        is the whole reason it is under-observed in the first place.
        """
        center = centroid if centroid is not None else obj.provisional_centroid()
        box = obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
        if box is not None and box[1] is not None:
            return center, box[1]
        voxels = obj.vote_stat.voxels
        if voxels.shape[0] == 0:
            return center, None
        return center, np.ptp(voxels, axis=0)

    def serialize_map_to_dict(self):
        objects_dict = {}
        for single_obj in self.single_obj_list:
            center = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
            if center is None:
                # A fully-tracked object that never reaches the output. infer_centroid
                # returns None when the surviving voxels carry zero total observation-angle
                # weight — i.e. regularize_shape accepted no cluster, or the diversity trim
                # removed everything. This was previously a silent `continue`, which is why
                # three complete pillow objects vanished with no trace in any counter.
                self.funnel[single_obj.get_dominant_label()]["unpublished_no_centroid"] += 1
                continue
            box_center, box_extent, box_rotation = self._published_box(single_obj)
            obj_id = [int(x) for x in single_obj.obj_id]
            objects_dict[obj_id[0]] = {
                "label": single_obj.get_dominant_label(),
                "id": obj_id,
                "center": center.tolist(),
                "bbox3d": {"center": box_center.tolist(), "extent": box_extent.tolist(), "rotation": list(box_rotation)},
            }
        return objects_dict

    def to_ros2_msgs(self, stamp):
        seconds = int(stamp)
        nanoseconds = int((stamp - seconds) * 1e9)

        points_list, colors_list, bbox_msgs, text_msgs = [], [], [], []
        colors = _generate_colors(len(self.single_obj_list))
        for idx, single_obj in enumerate(self.single_obj_list):
            obj_points = single_obj.retrieve_valid_voxels(diversity_percentile=self.percentile_thresh, regularized=True)
            if len(obj_points) == 0:
                continue

            color = colors[idx]
            points_list.append(obj_points)
            colors_list.append(np.array([color] * obj_points.shape[0]))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            aabb = pcd.get_axis_aligned_bounding_box()

            bbox_msgs.append(create_wireframe_marker(
                center=aabb.get_center(), extent=aabb.get_extent(), yaw=0.0,
                ns=f"{single_obj.class_id}", box_id=f"{single_obj.obj_id[0]}", color=color,
                seconds=seconds, nanoseconds=nanoseconds, frame_id="map"))
            text_msgs.append(create_text_marker(
                center=aabb.get_center(), marker_id=single_obj.obj_id[0], text=single_obj.get_dominant_label(),
                color=color, text_height=0.2, seconds=seconds, nanoseconds=nanoseconds, frame_id="map"))

        ros_pcd = None
        if points_list:
            ros_pcd = create_colored_point_cloud(
                points=np.concatenate(points_list, axis=0), colors=np.concatenate(colors_list, axis=0),
                seconds=seconds, nanoseconds=nanoseconds, frame_id="map")
        return bbox_msgs, text_msgs, ros_pcd


def _generate_colors(n, is_int=False):
    """Evenly-spaced hues so consecutive objects get visually distinct colors."""
    import colorsys
    colors = [colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 0.9) for i in range(n)]
    if is_int:
        colors = [[int(round(c * 255)) for c in rgb] for rgb in colors]
    return colors
