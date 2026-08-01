"""Top-N best-view image collector for a set of target object labels.

Called once per processed frame from `sam_node._process`.

The unit of selection is a spatial cluster of objects within a frame, not the frame
itself: instances are grouped left-to-right and split on a gap wider than
`roi_cluster_gap_px`, so one distant object cannot stretch the crop across the panorama.
Each cluster is cropped to its own ROI.

Per tracked instance the collector keeps the cluster where it was most visible (mask
coverage x confidence). The top-N are picked by greedy max-coverage: new-object COUNT
first, so several objects beat a sharper shot of fewer. Only objects inside the chosen
cluster count as covered.

Output is written through on every change, so killing the node mid-bag still leaves a
correct selection on disk.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from sam_mapper.annotate import annotate_frame
from sam_mapper.detections import PromptTable


@dataclass(frozen=True)
class BestViewConfig:
    targets: tuple[str, ...]
    top_n: int
    output_dir: str
    save_annotated_copy: bool
    min_instance_score: float
    crop_to_roi: bool
    roi_padding_frac: float
    roi_min_size_px: int
    roi_cluster_gap_px: float

    @staticmethod
    def from_dict(raw: dict, table: PromptTable) -> "BestViewConfig":
        # Every instance: true label is tracked automatically, so there is no separate
        # target list to keep in sync. Background labels share one id that is recreated
        # each frame, so they have no per-instance coverage to maximize.
        targets = tuple(sorted({s.label for s in table.specs if s.instance}))
        if not targets:
            raise ValueError("save_best_target_view_images needs at least one "
                              "instance: true entry in objects: to track")

        top_n = int(raw.get("top_n", 3))
        if top_n < 1:
            raise ValueError("save_best_target_view_images.top_n must be >= 1")

        return BestViewConfig(
            targets=targets,
            top_n=top_n,
            output_dir=raw.get("output_dir", "/data/bags/_best_views"),
            save_annotated_copy=bool(raw.get("save_annotated_copy", True)),
            # An object covers well under 1% of a 1920x640 panorama, so real scores land
            # around 1e-3 to 1e-2, not near 1. This floor only rejects near-empty masks.
            min_instance_score=float(raw.get("min_instance_score", 0.0005)),
            crop_to_roi=bool(raw.get("crop_to_roi", True)),
            roi_padding_frac=float(raw.get("roi_padding_frac", 0.2)),
            roi_min_size_px=int(raw.get("roi_min_size_px", 300)),
            roi_cluster_gap_px=float(raw.get("roi_cluster_gap_px", 250)),
        )


@dataclass(frozen=True)
class _Inst:
    """One target instance that passed the filters, within a single frame."""
    tid: int
    score: float
    bbox: tuple          # (x0,y0,x1,y1) in FRAME coordinates
    row: int             # index back into the frame's detections arrays


@dataclass
class Candidate:
    """One spatial cluster of target objects, from one frame, already cropped.

    Keeps the crop rather than the full frame and only this cluster's masks: the pool is
    keyed by track id and never evicted, so pinning whole panoramas would grow unbounded.
    """
    seq: int                   # monotonic identity for dedup/change detection (never id())
    stamp: float
    crop: np.ndarray           # BGR, already cut to `roi`
    crop_detections: dict      # 5-key dict, crop-relative, cluster members only
    roi: tuple                 # (x0,y0,x1,y1) in the ORIGINAL frame
    instance_scores: dict      # cluster member track_id -> coverage*confidence
    instance_bboxes: dict      # cluster member track_id -> bbox in FRAME coordinates

    @property
    def cluster_score(self) -> float:
        return sum(self.instance_scores.values())


class BestViewCollector:
    """One instance per node run. Call `consider()` once per processed frame."""

    # Equirect 360 panorama: column 0 and the last column are the same azimuth. SAM 3 has
    # no wrap-around awareness, so an object on the seam comes back as two separate
    # instances, one hugging each edge -- neither spans both, which is why rejection is on
    # EITHER edge. The blind band this creates is safe here: the robot passes every object,
    # so a seam-clipped view is always available cleanly in some other frame.
    SEAM_MARGIN_PX = 200

    @classmethod
    def _touches_border(cls, bbox, width: int) -> bool:
        x0, _, x1, _ = bbox
        return x0 <= cls.SEAM_MARGIN_PX or x1 >= width - cls.SEAM_MARGIN_PX

    def __init__(self, config: BestViewConfig, log=print):
        self.config = config
        self.log = log
        self.targets = set(config.targets)
        self._target_tag = "+".join(sorted(self.targets))

        run_name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{self._target_tag}"
        self.run_dir = os.path.join(config.output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        if config.save_annotated_copy:
            os.makedirs(os.path.join(self.run_dir, "annotated"), exist_ok=True)

        self.best_for_id: dict[int, Candidate] = {}
        self._selected_key: tuple = ()   # seqs of the last-written selection
        self._seq = itertools.count()
        self._frozen = False

        self.log(f"best-view collector: targets={sorted(self.targets)} "
                 f"top_n={config.top_n} -> {self.run_dir}")

    # -- bag-loop handling ----------------------------------------------------

    def on_time_jump(self) -> None:
        """Freeze on bag loop: the loop re-numbers track ids, so the same furniture would
        come back as new objects and inflate coverage. One pass has already seen it all."""
        if self._frozen:
            return
        self._frozen = True
        self.log("best-view collector: bag looped — freezing selection after the first "
                 "pass (later laps re-number the same objects)")

    # -- per-frame ------------------------------------------------------------

    @staticmethod
    def _cluster(instances: list, gap_px: float) -> list:
        """Split instances into left-to-right groups on a large horizontal gap.

        Measured against the running right-edge maximum, not the previous box's right
        edge, so a wide object overlapping several others is not split away from them.
        """
        clusters: list[list[_Inst]] = []
        current: list[_Inst] = []
        right = None
        for inst in sorted(instances, key=lambda i: i.bbox[0]):
            if current and inst.bbox[0] - right > gap_px:
                clusters.append(current)
                current, right = [], None
            current.append(inst)
            right = inst.bbox[2] if right is None else max(right, inst.bbox[2])
        if current:
            clusters.append(current)
        return clusters

    def consider(self, image_bgr: np.ndarray, detections: dict, stamp: float) -> None:
        if self._frozen:
            return

        labels = detections["labels"]
        if len(labels) == 0:
            return

        ids = detections["ids"]
        masks = detections["masks"]
        confidences = detections["confidences"]
        bboxes = detections["bboxes"]
        height, width = image_bgr.shape[:2]
        frame_area = float(height * width)

        surviving = []
        for row, (label, obj_id, mask, conf, bbox) in enumerate(
                zip(labels, ids, masks, confidences, bboxes)):
            if label not in self.targets or obj_id < 0:
                continue
            if self._touches_border(bbox, width):
                continue
            score = float(mask.sum()) / frame_area * float(conf)
            if score >= self.config.min_instance_score:
                surviving.append(_Inst(tid=int(obj_id), score=score,
                                       bbox=tuple(float(v) for v in bbox), row=row))

        if not surviving:
            return

        changed = False
        for cluster in self._cluster(surviving, self.config.roi_cluster_gap_px):
            # Only pay for the crop if this cluster is somebody's new best.
            improved = [i for i in cluster
                        if i.tid not in self.best_for_id
                        or i.score > self.best_for_id[i.tid].instance_scores[i.tid]]
            if not improved:
                continue

            candidate = self._build_candidate(cluster, image_bgr, detections, stamp,
                                              width, height)
            for inst in improved:
                self.best_for_id[inst.tid] = candidate
            changed = True

        if changed:
            self._select_and_flush()

    def _build_candidate(self, cluster: list, image_bgr, detections: dict, stamp: float,
                         width: int, height: int) -> Candidate:
        """Crop the frame to this cluster's ROI and keep only what the output needs."""
        bboxes = [i.bbox for i in cluster]
        if self.config.crop_to_roi:
            roi = self._compute_roi(bboxes, width, height)
        else:
            roi = (0, 0, width, height)
        x0, y0, x1, y1 = roi

        # Crop-relative, so annotate_frame works unchanged on the cropped image.
        crop_detections = {
            "masks": np.asarray([detections["masks"][i.row][y0:y1, x0:x1] for i in cluster],
                                dtype=bool),
            "bboxes": np.asarray([[i.bbox[0] - x0, i.bbox[1] - y0,
                                   i.bbox[2] - x0, i.bbox[3] - y0] for i in cluster],
                                 dtype=float),
            "labels": np.asarray([detections["labels"][i.row] for i in cluster], dtype=object),
            "ids": np.asarray([i.tid for i in cluster], dtype=int),
            "confidences": np.asarray([detections["confidences"][i.row] for i in cluster],
                                      dtype=float),
        }
        return Candidate(
            seq=next(self._seq),
            stamp=stamp,
            crop=image_bgr[y0:y1, x0:x1].copy(),
            crop_detections=crop_detections,
            roi=roi,
            instance_scores={i.tid: i.score for i in cluster},
            instance_bboxes={i.tid: i.bbox for i in cluster},
        )

    # -- geometry -------------------------------------------------------------

    @staticmethod
    def _spread(cand: "Candidate", ids: set) -> float:
        """Max pairwise pixel distance between bbox centers; 0 for a single instance.

        Not wrap-aware on purpose: this is about how the saved image reads, and
        seam-touching instances are already excluded (SEAM_MARGIN_PX).
        """
        if len(ids) < 2:
            return 0.0
        centers = []
        for tid in ids:
            x0, y0, x1, y1 = cand.instance_bboxes[tid]
            centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        return max(math.hypot(ax - bx, ay - by)
                   for (ax, ay), (bx, by) in itertools.combinations(centers, 2))

    @staticmethod
    def _fit_in_bounds(lo: float, hi: float, dim: int) -> tuple:
        """Slide [lo,hi) inside [0,dim) rather than clipping it.

        Clipping would return a window smaller than asked for near an edge, silently
        defeating roi_min_size_px. Only a window larger than the frame gets shrunk.
        """
        size = min(hi - lo, dim)
        lo = min(max(lo, 0.0), dim - size)
        return lo, lo + size

    def _compute_roi(self, bboxes, img_w: int, img_h: int) -> tuple:
        """Padded union of one cluster's boxes, floored to a minimum size, fitted to the
        frame. Passing a single cluster rather than the whole frame is what keeps it tight."""
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)

        pad_x = (x1 - x0) * self.config.roi_padding_frac
        pad_y = (y1 - y0) * self.config.roi_padding_frac
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y

        min_size = self.config.roi_min_size_px
        if (x1 - x0) < min_size:
            cx = (x0 + x1) / 2.0
            x0, x1 = cx - min_size / 2.0, cx + min_size / 2.0
        if (y1 - y0) < min_size:
            cy = (y0 + y1) / 2.0
            y0, y1 = cy - min_size / 2.0, cy + min_size / 2.0

        x0, x1 = self._fit_in_bounds(x0, x1, img_w)
        y0, y1 = self._fit_in_bounds(y0, y1, img_h)
        return int(x0), int(y0), int(x1), int(y1)

    # -- selection ------------------------------------------------------------

    def _gain(self, cand: Candidate, covered: set) -> tuple:
        """Greedy key: new-object count, their grouping (negated), their visibility, then
        the cluster total.

        That last term matters -- once everything is covered the first three are 0 for
        every candidate, and max() would otherwise return whichever came first in dict
        order, filling the spare slots at random.
        """
        new_ids = set(cand.instance_scores) - covered
        return (len(new_ids), -self._spread(cand, new_ids),
                sum(cand.instance_scores[t] for t in new_ids),
                cand.cluster_score)

    def _select_and_flush(self) -> None:
        # Several track ids can share one cluster candidate.
        candidates = list({c.seq: c for c in self.best_for_id.values()}.values())

        covered: set[int] = set()
        selected: list[tuple] = []      # (candidate, objects it newly covered)
        remaining = candidates
        for _ in range(min(self.config.top_n, len(remaining))):
            best = max(remaining, key=lambda c: self._gain(c, covered))
            selected.append((best, len(set(best.instance_scores) - covered)))
            remaining.remove(best)
            # Cluster members only -- never claim coverage for an object the crop left out.
            covered |= set(best.instance_scores)

        # Greedy order is the rank order; sorting by score would invert the "more objects
        # beats fewer clearer ones" priority. The manifest's new_objects shows the reason.
        selection_key = tuple(c.seq for c, _ in selected)
        if selection_key == self._selected_key:
            return
        self._selected_key = selection_key
        self._flush(selected)

    def _flush(self, selected: list[tuple]) -> None:
        written: list[tuple] = []   # (rank, cand, new_count) -- only images on disk
        for rank, (cand, new_count) in enumerate(selected, start=1):
            name = f"best_rank{rank}_{self._target_tag}.png"
            if not cv2.imwrite(os.path.join(self.run_dir, name), cand.crop):
                self.log(f"best-view collector: failed to write {name}")
                continue
            written.append((rank, cand, new_count))
            if self.config.save_annotated_copy:
                overlay = annotate_frame(cand.crop, cand.crop_detections)
                cv2.imwrite(os.path.join(self.run_dir, "annotated", name), overlay)

        # The pool can shrink (two ids' bests can converge on one cluster), so drop any
        # file past the current selection rather than leaving a stale rank behind.
        for stale in glob.glob(os.path.join(self.run_dir, f"best_rank*_{self._target_tag}.png")):
            rank = int(os.path.basename(stale).split("_")[1].removeprefix("rank"))
            if rank > len(selected):
                os.remove(stale)
                annotated = os.path.join(self.run_dir, "annotated", os.path.basename(stale))
                if os.path.exists(annotated):
                    os.remove(annotated)

        def _crop_relative_bbox(cand: Candidate, tid: int) -> list:
            rx0, ry0, _, _ = cand.roi
            x0, y0, x1, y1 = cand.instance_bboxes[tid]
            return [round(v, 1) for v in (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)]

        # From `written`, so the manifest never names a file that failed to write.
        manifest = {
            "targets": sorted(self.targets),
            "top_n": self.config.top_n,
            "selected": [
                {
                    "rank": rank,
                    "file": f"best_rank{rank}_{self._target_tag}.png",
                    "stamp": cand.stamp,
                    "cluster_score": cand.cluster_score,
                    "new_objects": new_count,   # 0 = added no coverage, kept as best remaining
                    "roi": list(cand.roi),      # in ORIGINAL frame coords
                    "instances": [
                        # bbox is crop-relative: matches pixels in the saved PNG.
                        {"track_id": tid, "score": score,
                         "bbox": _crop_relative_bbox(cand, tid)}
                        for tid, score in sorted(cand.instance_scores.items(),
                                                 key=lambda kv: -kv[1])
                    ],
                }
                for rank, cand, new_count in written
            ],
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)

        covered_ids = {tid for _, cand, _ in written for tid in cand.instance_scores}
        self.log(f"best-view collector: wrote {len(written)}/{self.config.top_n} images, "
                 f"covering {len(covered_ids)} instance(s) -> {self.run_dir}")
