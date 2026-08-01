"""3D instance mapper: fuses per-frame 2D detections + lidar into persistent tracked objects.

Owned copy of semantic_mapping/semantic_map.py's ObjMapper (docs/M2_perception.md 3.6-split),
trimmed to what sam_mapper calls: __init__, update_map, to_ros2_msgs, serialize_map_to_dict,
.single_obj_list.

Dropped — superseded, not on the roadmap: `tracker`/`visualize` constructor params (dead: their
only readers, track_objects/predict_new_tracklet_state — ByteTrack+spaCy, SAM3 replaces this —
and open3d_vis/rerun_vis/print_obj_info — ROS/Foxglove already covers this need — are gone too),
self.valid_cnt/self.clear_outliers_cycle (write-only, never read), and the module globals
INSTANCE_LEVEL_OBJECTS/OMIT_OBJECTS/BACKGROUND_OBJECTS (track_objects-only).

Kept dormant — real planned improvements, not dead code (docs backlog §6):
  * `captioner` param + every `self.captioner is not None` branch in update_map. map_node.py still
    passes captioner=None; wiring a real Captioner (ai_module/src/captioner/) is separate work.
  * `AdjacencyGraph` (which object ids were seen together in the same frame) + the commented-out
    IoU-overlap check it feeds, in the merge loop below. Co-visibility disambiguates "two close
    real objects" (seen together -> don't merge) from "one object split across ids by occlusion"
    (never seen together -> do merge) — the duplicate-instance-rate metric M6 scores directly.
    Blocked on pytorch3d.ops.box3d_overlap (+ get_corners_from_box3d_torch), not installed here.
  * `infer_bbox` (axis-aligned) vs. `infer_bbox_oriented` (already used for merge-distance checks
    below) — serialize_map_to_dict/to_ros2_msgs keep the axis-aligned box, matching this session's
    validated output. Switching is a real fidelity win but changes /obj_boxes /obj_map_json shape
    and needs the ConvexHull-failure case handled first.
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from sam_mapper.ros_markers import create_colored_point_cloud, create_text_marker, create_wireframe_marker
from sam_mapper.single_object import SingleObject

VERTICAL_OBJECTS = ["door", "painting"]  # merge candidates must agree on this (see the loop below)


class AdjacencyGraph:
    """Co-visibility: which object ids were seen in the same frame. Dormant — see module docstring."""

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
        return v2 in self.adjacency_list[v1]

    def is_set_adjacent(self, v_set1, v_set2):
        return any(self.is_adjacent(id1, id2) for id1 in v_set1 for id2 in v_set2)


class ObjMapper:
    def __init__(self, cloud_image_fusion, label_template, captioner=None, log_info=print):
        self.single_obj_list: list[SingleObject] = []
        self.background_obj_list: list[SingleObject] = []
        self.adjacency_graph = AdjacencyGraph()

        self.cloud_image_fusion = cloud_image_fusion
        self.captioner = captioner
        self.log_info = log_info

        self.voxel_size = 0.05
        self.confidence_thres = 0.30
        self.cloud_to_odom_dist_thres = 6.0
        self.num_angle_bin = 20
        self.percentile_thresh = 0.8

        instance_labels = [label for label, val in label_template.items() if val["is_instance"]]
        self.log_info(f"Instance level objects: {instance_labels}")
        self.log_info(f"label template: {label_template}")

    def update_map(self, detections, detection_stamp, detection_odom, cloud, image=None):
        R_b2w = Rotation.from_quat(detection_odom["orientation"]).as_matrix()
        t_b2w = np.array(detection_odom["position"])
        R_w2b = R_b2w.T
        t_w2b = -R_w2b @ t_b2w
        cloud_body = cloud @ R_w2b.T + t_w2b

        keep = np.array(detections["confidences"]) >= self.confidence_thres
        masks = [m for m, k in zip(detections["masks"], keep) if k]
        labels = [l for l, k in zip(detections["labels"], keep) if k]
        obj_ids = [i for i, k in zip(detections["ids"], keep) if k]
        bboxes = [b for b, k in zip(detections["bboxes"], keep) if k]
        if not obj_ids:
            return

        # Shrink masks so lidar points near a silhouette edge don't bleed onto background —
        # the main knob for point-cloud bleed-through (docs/M2_perception.md 2.6 step 3).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        masks = [cv2.erode(m.astype(np.uint8), kernel, iterations=5).astype(bool) for m in masks]

        if len(obj_ids) == 1:
            self.adjacency_graph.add_vertex(obj_ids[0])
        else:
            for i in range(len(obj_ids)):
                for j in range(i, len(obj_ids)):
                    self.adjacency_graph.add_edge(obj_ids[i], obj_ids[j])

        obj_clouds_world = self.cloud_image_fusion.generate_seg_cloud(cloud_body, masks, R_b2w, t_b2w)

        for i, obj_cloud in enumerate(obj_clouds_world):
            dist_mask = np.linalg.norm(obj_cloud[:, :3] - t_b2w, axis=1) < self.cloud_to_odom_dist_thres
            obj_cloud = obj_cloud[dist_mask]
            if obj_cloud.shape[0] < 5:
                continue

            class_id, obj_id = labels[i], obj_ids[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_cloud[:, :3])
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            if pcd_downsampled.is_empty():
                continue

            merged = False
            if obj_id >= 0:
                for single_obj in self.single_obj_list:
                    if obj_id in single_obj.obj_id:
                        single_obj.merge(np.array(pcd.points), R_b2w, t_b2w, class_id, detection_stamp)
                        single_obj.reproject_obs_angle(R_w2b, t_w2b, masks[i], self.cloud_image_fusion.scan2pixels)
                        single_obj.inactive_frame = -1
                        merged = True
                        break
            # else: background merge is intentionally not attempted — background objects are
            # re-created every frame (no persistent identity), matching current behavior.

            if not merged:
                target_list = self.single_obj_list if obj_id >= 0 else self.background_obj_list
                target_list.append(SingleObject(class_id, obj_id, np.array(pcd_downsampled.points),
                    self.voxel_size, R_b2w, t_b2w, masks[i], detection_stamp, num_angle_bin=self.num_angle_bin))

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
            if single_obj.inactive_frame > 20:
                i += 1
                continue

            merged_obj = False
            if 5 < single_obj.life < 1000:
                if single_obj.valid_indices_regularized.shape[0] < 20 and single_obj.inactive_frame > 5:
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
                if centroid is not None:
                    source_is_vertical = single_obj.get_dominant_label() in VERTICAL_OBJECTS
                    target_obj, target_index, minimum_dist = None, -1, 1e6
                    for j, same_class_obj in enumerate(self.single_obj_list):
                        if same_class_obj.obj_id[0] < 0 or j == i:
                            continue
                        if same_class_obj.get_dominant_label() != single_obj.get_dominant_label():
                            continue
                        if (same_class_obj.get_dominant_label() in VERTICAL_OBJECTS) != source_is_vertical:
                            continue
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
                            dist_thresh = np.linalg.norm((extent_object / 2 + extent_target / 2) / 2) * 0.5
                            if minimum_dist < dist_thresh or minimum_dist < 0.5:
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
                            # Needs pytorch3d installed to bring back — see semantic_mapping's
                            # semantic_map.py (pre-split) for the original, fuller version.

                            if merged_obj:
                                if target_index < i:
                                    single_obj, target_obj = target_obj, single_obj
                                single_obj.merge_object(target_obj)
                                single_obj.inactive_frame = -1
                                self.single_obj_list.remove(target_obj)
                                if self.captioner is not None:
                                    centroid_target = target_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
                                    bbox_3d_target = target_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
                                    self.captioner.merge_objects(single_obj.obj_id[0], target_obj.obj_id[0], centroid_target, bbox_3d_target)

            if not merged_obj:
                single_obj.inactive_frame += 1
                single_obj.regularize_shape(self.percentile_thresh)
                i += 1

    def serialize_map_to_dict(self, stamp):
        objects_dict = {}
        for single_obj in self.single_obj_list:
            center = single_obj.infer_centroid(diversity_percentile=self.percentile_thresh, regularized=True)
            if center is None:
                continue
            box_center, box_extent, box_rotation = single_obj.infer_bbox(diversity_percentile=self.percentile_thresh, regularized=True)
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
