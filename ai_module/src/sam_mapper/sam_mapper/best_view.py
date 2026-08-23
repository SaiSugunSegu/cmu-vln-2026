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

Writes happen on a background thread, coalesced to the latest selection (see
`_FlushWriter`): the encode is ~130 ms and used to run on sam_node's worker, between SAM 3
inference and the publish. Killing the node mid-bag can therefore lose at most the one
selection still in flight, falling back to the previous complete one on disk.

The overlay copies are the exception: `finalize()` draws them once at the end of the run,
because their labels carry map_node's object ids and world merges keep renaming those. That
is what makes a crop readable against `obj_map.json` alone — `chair [3]` names the entry
keyed `3`, no manifest lookup in between.

`save_full_views` mirrors the whole output under `full/`, uncropped: same filenames, same
overlays, same captions, the difference being only how much room is in shot.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np

from sam_mapper.annotate import annotate_frame, silhouette_frame
from sam_mapper.detections import PromptTable

_UNSAFE_RUN_ID_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_run_id(run_id: str, fallback: str = "run") -> str:
    """Reduce a caller-supplied run id to a safe relative path under the output dir.

    Run ids reach us over ROS (`/sam3/set_prompts`) carrying question text, so they can
    hold spaces, punctuation and slashes. Callers that build paths from a run id
    (sam_mapper here, smart_vlm's category-1 reasoner) must agree on the transform or
    they compute different directories for the same run.

    A slash is kept as a level, which is what gives the eval sweep its
    `<scene>/<question>/` layout, but only as a separator: every component is scrubbed
    to `[a-zA-Z0-9._-]`, and a component made only of dots becomes the fallback, so
    `../..` cannot walk out of the crops root.
    """
    parts = []
    for raw in str(run_id).split("/"):
        part = _UNSAFE_RUN_ID_CHARS.sub("_", raw.strip()).strip("_")
        # ".." survives the character filter above — dots are legal in a name.
        if not part or set(part) == {"."}:
            continue
        parts.append(part)
    return "/".join(parts) or fallback


@dataclass(frozen=True)
class BestViewConfig:
    targets: tuple[str, ...]
    top_n: int
    output_dir: str
    save_annotated_copy: bool
    save_silhouette_copy: bool
    save_full_views: bool
    min_instance_score: float
    crop_to_roi: bool
    roi_padding_frac: float
    roi_min_size_px: int
    roi_cluster_gap_px: float
    finalize_obj_map_wait_s: float

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
            output_dir=raw.get("output_dir", "/data/crops"),
            save_annotated_copy=bool(raw.get("save_annotated_copy", True)),
            save_silhouette_copy=bool(raw.get("save_silhouette_copy", False)),
            # Opt-in: it roughly doubles the images on disk, and every overlay with them.
            save_full_views=bool(raw.get("save_full_views", False)),
            # An object covers well under 1% of a 1920x640 panorama, so real scores land
            # around 1e-3 to 1e-2, not near 1. This floor only rejects near-empty masks.
            min_instance_score=float(raw.get("min_instance_score", 0.0005)),
            crop_to_roi=bool(raw.get("crop_to_roi", True)),
            roi_padding_frac=float(raw.get("roi_padding_frac", 0.2)),
            roi_min_size_px=int(raw.get("roi_min_size_px", 300)),
            roi_cluster_gap_px=float(raw.get("roi_cluster_gap_px", 250)),
            # finalize()'s wait for map_node's obj_map.json. Same file, same settling time
            # as object_reference_reasoner.MAP_WAIT_S.
            finalize_obj_map_wait_s=float(raw.get("finalize_obj_map_wait_s", 5.0)),
        )


@dataclass(frozen=True)
class _Inst:
    """One target instance that passed the filters, within a single frame."""
    tid: int
    score: float
    bbox: tuple          # (x0,y0,x1,y1) in FRAME coordinates
    row: int             # index back into the frame's detections arrays


class _FlushWriter:
    """Single-slot, latest-wins writer thread for best-view flushes.

    Two problems, one mechanism. The encode is expensive -- with save_full_views a flush
    writes every rank's crop AND its uncropped frame, so top_n=3 is six PNGs, one pair of
    them a full panorama -- and it used to run inline on sam_node's worker thread, where it
    measured ~130 ms of a ~445 ms frame. And it runs far more often than it needs to: while
    scores keep improving, the selection changes on nearly every frame, so the same images
    were rewritten dozens of times per run.

    Moving the write here takes it off the critical path; the single slot fixes the rest.
    Same drop-to-latest shape as sam_node/map_node's frame slots, for the same reason --
    a queue would grow without bound and spend the time writing selections nobody reads.
    """

    def __init__(self, write, log):
        self._write = write
        self._log = log
        self._cond = threading.Condition()
        self._pending = None            # latest selection awaiting a write, or None
        self._writing = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, selected: list) -> None:
        """Replace whatever was queued. Dropping the previous one is the point."""
        with self._cond:
            self._pending = selected
            self._cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                # Drain before exiting: a selection queued just before stop() still lands,
                # so a clean shutdown loses nothing.
                if self._pending is None:
                    return
                selected, self._pending = self._pending, None
                self._writing = True
            try:
                self._write(selected)
            except Exception:  # noqa: BLE001 -- a write fault must not kill the thread and
                               # silently stop every later flush with it
                self._log("best-view collector: flush failed:\n"
                          f"{traceback.format_exc()}")
            finally:
                with self._cond:
                    self._writing = False
                    self._cond.notify_all()

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until nothing is queued and nothing is being written. False on timeout."""
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._cond:
            while self._pending is not None or self._writing:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(remaining)
            return True

    def stop(self, timeout: float = 10.0) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=timeout)


@dataclass
class Candidate:
    """One spatial cluster of target objects, from one frame, already cropped.

    Keeps the crop and only this cluster's masks: the pool is keyed by track id and never
    evicted, so pinning a panorama per candidate would grow unbounded.

    `frame` is that rule's one exception, for `save_full_views`, and is affordable because it
    is a REFERENCE — nothing downstream mutates the decoded image, so every cluster of one
    frame shares one array and the cost is per distinct frame retained, not per candidate.
    Full-frame masks are still never held: `_full_detections` rebuilds them on demand.
    """
    seq: int                   # monotonic identity for dedup/change detection (never id())
    stamp: float
    crop: np.ndarray           # BGR, already cut to `roi`
    crop_detections: dict      # 5-key dict, crop-relative, cluster members only
    roi: tuple                 # (x0,y0,x1,y1) in the ORIGINAL frame
    instance_scores: dict      # cluster member track_id -> coverage*confidence
    instance_bboxes: dict      # cluster member track_id -> bbox in FRAME coordinates
    frame: np.ndarray | None = None   # uncropped BGR, only when save_full_views

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

    def __init__(self, config: BestViewConfig, log=print, run_id: str | None = None):
        self.config = config
        self.log = log
        self.targets = set(config.targets)
        self._target_tag = "+".join(sorted(self.targets))

        if run_id:
            # Verbatim: the caller owns the layout, and an eval sweep needs the same
            # question to land in the same directory every time it is rebuilt. The
            # target tag stays in the filenames, where it costs no predictability.
            run_name = sanitize_run_id(run_id)
        else:
            run_name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{self._target_tag}"
        self.run_dir = os.path.join(config.output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self._clear_stale_crops()

        # subdir -> renderer, for every overlay copy this run writes next to the raw crops.
        # One dict drives the mkdir, the per-flush write and the stale-file cleanup.
        self._overlays = {name: render for name, render, enabled in (
            ("annotated", annotate_frame, config.save_annotated_copy),
            ("silhouette", silhouette_frame, config.save_silhouette_copy),
        ) if enabled}

        # The geometries each best view is saved in: "" is the crop, "full" the frame it was
        # taken from. One list drives the mkdirs, both writes and both cleanups, so the two
        # can never drift apart.
        self._save_full = config.save_full_views and config.crop_to_roi
        if config.save_full_views and not config.crop_to_roi:
            # roi is the whole frame, so full/ would be byte-identical duplicates.
            self.log("best-view collector: save_full_views ignored — crop_to_roi is off, so "
                     "the crops are already full frames")
        self._geometries = ("", "full") if self._save_full else ("",)
        for geometry in self._geometries:
            os.makedirs(os.path.join(self.run_dir, geometry), exist_ok=True)
            for name in self._overlays:
                os.makedirs(os.path.join(self.run_dir, geometry, name), exist_ok=True)

        self.best_for_id: dict[int, Candidate] = {}
        self._selected_key: tuple = ()   # seqs of the last-written selection
        self._seq = itertools.count()
        self._frozen = False
        # _written / _rendered_with / _on_disk are set on the writer thread and read by
        # finalize(), which runs on yet another thread (sam_node spawns it) -- one lock
        # over all three. finalize() also drain()s first, so it never reads a half-flush.
        self._state_lock = threading.Lock()
        # What is on disk right now, so finalize() can re-render exactly those ranks.
        self._written: list[tuple[int, Candidate]] = []
        # rank -> Candidate.seq for images CONFIRMED on disk. _flush diffs against this
        # rather than against the previous selection: coalescing means selections in
        # between were never written, and diffing against one of those would skip a rank
        # whose file is still two selections stale.
        self._on_disk: dict[int, int] = {}
        # The id lookup the overlays currently show. `False`, not None, for "never
        # rendered": None is itself a valid lookup, meaning no 3D map at all.
        self._rendered_with: object = False
        self._writer = _FlushWriter(self._flush, self.log)

        self.log(f"best-view collector: targets={sorted(self.targets)} "
                 f"top_n={config.top_n} -> {self.run_dir}")

    def _clear_stale_crops(self) -> None:
        """Empty a reused run directory before writing into it.

        A stable run id means a rebuild lands on top of the previous attempt, and the
        per-flush cleanup only drops ranks of the CURRENT target tag. Different targets
        for the same question would otherwise leave both sets side by side, and a reader
        taking the top few files would mix two runs.
        """
        # Recursive, so the overlay copies go too without naming their subdirectories
        # here — one of which may not even be enabled on this run.
        for stale in glob.glob(os.path.join(self.run_dir, "**", "best_rank*.png"),
                               recursive=True):
            os.remove(stale)
        # The manifest too, or the merge on flush would carry the previous attempt's
        # answer over onto this attempt's images.
        manifest = os.path.join(self.run_dir, "manifest.json")
        if os.path.exists(manifest):
            os.remove(manifest)

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
            # Deliberately not copied — see Candidate. Every cluster of this frame gets the
            # same array, so a frame with six clusters still costs one frame.
            frame=image_bgr if self._save_full else None,
        )

    @staticmethod
    def _full_detections(cand: Candidate) -> dict:
        """`crop_detections` mapped back onto the whole frame.

        The masks are rebuilt rather than stored: one is 1.2 MB at 1920x640, so a handful of
        instances would outweigh the image. Pasting loses nothing — a mask never extends
        beyond its own bbox, and the roi is the padded union of the cluster's bboxes.
        """
        x0, y0, x1, y1 = cand.roi
        crop_masks = cand.crop_detections["masks"]
        masks = np.zeros((len(crop_masks), *cand.frame.shape[:2]), dtype=bool)
        masks[:, y0:y1, x0:x1] = crop_masks
        return {
            **cand.crop_detections,
            "masks": masks,
            "bboxes": np.asarray([cand.instance_bboxes[int(tid)]
                                  for tid in cand.crop_detections["ids"]], dtype=float),
        }

    def _views(self, cand: Candidate):
        """(subdir, image, detections thunk) once per geometry this run saves.

        The detections are a thunk because `_flush` writes raw images only and must not pay
        to rebuild masks it will not draw.
        """
        yield "", cand.crop, lambda: cand.crop_detections
        if cand.frame is not None:
            yield "full", cand.frame, lambda: self._full_detections(cand)

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
        # Off the caller's thread (sam_node's SAM worker) and coalesced -- see _FlushWriter.
        self._writer.submit(selected)

    # -- write lifecycle ------------------------------------------------------

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until every submitted selection has reached disk. False on timeout.

        Public because the write is asynchronous: finalize() needs it (it renders overlays
        for whatever _flush last wrote), and so does any caller that inspects the run
        directory straight after consider().
        """
        return self._writer.drain(timeout)

    def stop(self, timeout: float = 10.0) -> None:
        """Land any queued flush and retire the writer thread.

        A collector is replaced per /sam3/set_prompts and dropped at shutdown; without this
        each one leaves a thread parked on its condition for the life of the process.
        """
        self._writer.stop(timeout)

    def _flush(self, selected: list[tuple]) -> None:
        """Runs on the writer thread. Postcondition: the run directory holds exactly this
        selection — whatever it held before, and however many selections were coalesced
        away getting here."""
        written: list[tuple] = []   # (rank, cand, new_count) -- only images on disk
        encoded = 0
        for rank, (cand, new_count) in enumerate(selected, start=1):
            name = f"best_rank{rank}_{self._target_tag}.png"
            # Skip only a rank whose file on disk is ALREADY this candidate. The filename
            # carries the rank, so a candidate that merely moved rank still fails this test
            # at both ranks and is rewritten at both — which is what a 1<->2 swap needs.
            if self._on_disk.get(rank) == cand.seq:
                written.append((rank, cand, new_count))
                continue
            # A list, not all(...): short-circuiting would skip the remaining geometries and
            # leave the rank half written. A rank counts only if every geometry landed.
            ok = [cv2.imwrite(os.path.join(self.run_dir, geometry, name), image)
                  for geometry, image, _ in self._views(cand)]
            if not all(ok):
                self.log(f"best-view collector: failed to write {name}")
                # Half a rank is on disk at best, so forget it: the next flush must retry
                # rather than trust this seq.
                self._on_disk.pop(rank, None)
                continue
            self._on_disk[rank] = cand.seq
            encoded += 1
            written.append((rank, cand, new_count))

        # No overlays here: they carry map ids, and obj_map.json is still moving while frames
        # arrive — see finalize(). Raw crops stay per-flush, so a kill mid-bag loses nothing.
        with self._state_lock:
            self._written = [(rank, cand) for rank, cand, _ in written]
            self._rendered_with = False

        # The pool can shrink (two ids' bests can converge on one cluster), so drop any
        # file past the current selection rather than leaving a stale rank behind.
        for geometry in self._geometries:
            for stale in glob.glob(os.path.join(self.run_dir, geometry,
                                                f"best_rank*_{self._target_tag}.png")):
                rank = int(os.path.basename(stale).split("_")[1].removeprefix("rank"))
                if rank > len(selected):
                    os.remove(stale)
        # Forget them here too, or a later selection that grows back to this rank with the
        # same candidate would be skipped as "already written" against a file just deleted.
        for rank in [r for r in self._on_disk if r > len(selected)]:
            del self._on_disk[rank]

        def _crop_relative_bbox(cand: Candidate, tid: int) -> list:
            rx0, ry0, _, _ = cand.roi
            x0, y0, x1, y1 = cand.instance_bboxes[tid]
            return [round(v, 1) for v in (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)]

        def _label_for(cand: Candidate, tid: int) -> str:
            ids = cand.crop_detections.get("ids", [])
            labels = cand.crop_detections.get("labels", [])
            for obj_id, label in zip(ids, labels):
                if int(obj_id) == tid:
                    return str(label)
            return ""

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
                        {"track_id": tid, "label": _label_for(cand, tid),
                         "score": score, "bbox": _crop_relative_bbox(cand, tid)}
                        for tid, score in sorted(cand.instance_scores.items(),
                                                 key=lambda kv: -kv[1])
                    ],
                }
                for rank, cand, new_count in written
            ],
        }
        # Merge, never replace: the reasoner adds the question, the prompts it armed us
        # with and its answer to this same file, and frames keep arriving after it does
        # that — a plain overwrite silently threw all of it away on every later flush.
        path = os.path.join(self.run_dir, "manifest.json")
        if os.path.exists(path):
            try:
                with open(path) as handle:
                    foreign = {k: v for k, v in json.load(handle).items()
                               if k not in manifest}
            except (OSError, json.JSONDecodeError):
                foreign = {}
            manifest = {**foreign, **manifest}
        with open(path, "w") as handle:
            json.dump(manifest, handle, indent=2)

        covered_ids = {tid for _, cand, _ in written for tid in cand.instance_scores}
        # "encoded" is the work this flush actually did; the ranks it skipped were already
        # on disk carrying the right candidate. This used to fire once per frame, which is
        # why it had been commented out — coalescing is what makes it readable again.
        # self.log(f"best-view collector: wrote {len(written)}/{self.config.top_n} images "
        #          f"({encoded} encoded), covering {len(covered_ids)} instance(s) "
        #          f"-> {self.run_dir}")

    # -- finalize -------------------------------------------------------------

    @staticmethod
    def _caption(label: str, track_id: int, track_to_map: dict | None) -> str:
        """What one instance is called on the overlay.

        The bracketed number is a key of `obj_map.json` — that join is the whole point.
        No map at all: fall back to the track id, the only id there is. A map but no entry
        for this instance: no brackets, because the object never reached a 3D box (too few
        lidar points, pruned, no centroid) and an id there would resolve to nothing.
        """
        if track_to_map is None:
            return f"{label} [{track_id}]"
        map_id = track_to_map.get(track_id)
        return label if map_id is None else f"{label} [{map_id}]"

    def _resolve(self, cand: Candidate, track_to_map: dict | None) -> tuple[list, list]:
        """(captions, ids) for one crop, both keyed on the 3D map wherever it knows the object.

        The ids matter as much as the captions: an outline's colour is `_color_for(id)`, so
        leaving track ids here drew one object in two colours across crops while every
        caption named it the same. Map object 1 is track 1 in two of a run's three crops and
        track 3 in the other.

        An instance the map has no entry for keeps its track id, which cannot collide with a
        map key: every key is the `id[0]` of its own entry, hence itself a resolved track id.

        A map id seen twice in one crop keeps only the first caption — two tabs naming one
        object is noise — but both outlines, which are separate mask regions.
        """
        captions, ids, seen = [], [], set()
        for label, raw in zip(cand.crop_detections["labels"], cand.crop_detections["ids"]):
            track_id = int(raw)
            map_id = track_id if track_to_map is None else track_to_map.get(track_id)
            ids.append(track_id if map_id is None else map_id)
            captions.append("" if map_id in seen else self._caption(str(label), track_id,
                                                                    track_to_map))
            if map_id is not None:
                seen.add(map_id)
        return captions, ids

    def finalize(self, track_to_map: dict | None, drain_timeout: float = 30.0) -> bool:
        """Render the overlay copies for the crops on disk. True if it drew anything.

        Deferred rather than per-flush because the ids are map_node's, and it rewrites
        obj_map.json on every publish until exploration closes. Re-runnable on purpose —
        bag loop, /pipeline/explore_done and shutdown all call it, and the last call with
        better data wins; an unchanged lookup does nothing.
        """
        # The raw crops are written asynchronously now, so without this the overlays would
        # be drawn over whatever _flush happened to have finished — an older selection.
        if not self._writer.drain(drain_timeout):
            self.log("best-view collector: timed out waiting for pending crop writes; "
                     "overlays may lag the final selection")
        with self._state_lock:
            if not self._overlays or self._rendered_with == track_to_map:
                return False
            self._rendered_with = dict(track_to_map) if track_to_map is not None else None
            pending = list(self._written)

        names = set()
        for rank, cand in pending:
            name = f"best_rank{rank}_{self._target_tag}.png"
            names.add(name)
            # Once per candidate, not per geometry: the crop and the full frame hold the same
            # instances, so they carry the same captions and the same colours.
            captions, ids = self._resolve(cand, track_to_map)
            for geometry, image, detections in self._views(cand):
                labelled = dict(detections())
                labelled["labels"] = np.asarray(captions, dtype=object)
                labelled["ids"] = np.asarray(ids, dtype=int)
                for subdir, render in self._overlays.items():
                    overlay = render(image, labelled)
                    path = os.path.join(self.run_dir, geometry, subdir, name)
                    if not cv2.imwrite(path, overlay):
                        self.log(f"best-view collector: failed to write {path}")

        # A previous pass may have left a rank the current selection no longer has.
        for geometry in self._geometries:
            for subdir in self._overlays:
                for stale in glob.glob(os.path.join(self.run_dir, geometry, subdir,
                                                    f"best_rank*_{self._target_tag}.png")):
                    if os.path.basename(stale) not in names:
                        os.remove(stale)

        resolved = sum(1 for _, cand in pending for tid in cand.crop_detections["ids"]
                       if track_to_map is None or int(tid) in track_to_map)
        self.log(f"best-view collector: finalized {len(pending)} image(s), "
                 f"{resolved} instance label(s) carrying a map id -> {self.run_dir}")
        return True
