# M2 — Perception (open-vocab 2D → 3D instances)

**Task:** Convert 360° images + lidar into a persistent 3D object instance memory. Concretely: open-vocab detect+segment+track on the panorama; sync frames with accumulated lidar cloud; project masked pixels → cluster → fit 3D boxes; associate instances across frames (merge, confidence, best-crop storage); extract attributes (color name, size).

**Status:** 🟡 in progress — `sam_mapper` package being built on SAM 3
**Owner:**
**Depends on:** M0 (bags), M1 (viewpoints)

> This doc is both the **milestone tracker** (sections above the first `---`) and the
> **engineering reference** for the two perception packages (everything below).
> Start at [§1 System context](#1-system-context) if you are new to the pipeline.

## Interfaces
- In: `/camera/image` (10 Hz, 1920×640 equirect, 360°×120°), `/registered_scan`, `/state_estimation`, camera↔lidar extrinsics (repo), prompt sets from M4
- Out: `Instance {id, label, bbox3d, color, size, conf, siglip_feat, obs_count, best_crop, last_seen}` stream → M3

## Plan
| Stage | What to try |
|---|---|
| Baseline | **SAM 3** text-prompt concept segmentation (detect+segment+track in one model) on the **whole panorama**, single pass, streaming; lidar mask-projection + voxel voting → 3D boxes. Replaces the Grounding DINO + spaCy + ByteTrack + SAM2 chain wholesale. |
| Upgrade | **SAM 3.1** (Object Multiplex — shared-memory joint multi-object tracking); non-square input resolution to kill aspect distortion; **SigLIP 2** crop embeddings for re-ID/merging; HSV-median color naming; size from box dims |
| Stretch | Panorama-seam handling (tiling or perspective reprojection); **SAM 3D** for lidar-sparse/thin objects; fine-tune on Unity renders if synthetic domain gap appears |

### Decision record — model bake-off
Resolved in favour of SAM 3 before running the bake-off: it collapses detection, segmentation **and**
tracking into one model, which removes two whole subsystems (spaCy meta-class extraction, ByteTrack)
rather than just swapping a detector. Table kept for the record; fill in if we ever need to justify
revisiting.

| Candidate | Recall | Precision | ms/frame | Notes |
|---|---|---|---|---|
| **SAM 3** | | | | **chosen** — detect+segment+track in one pass |
| YOLOE (+SAM 3 masks) | | | | fallback if latency-bound |
| DINO-X | | | | accuracy reference only |

## Metrics (via M6)
- Per-class instance recall/precision vs VLA-3D GT (target ≥80% recall)
- Duplicate instance rate (target <10%) — the counting killer
- Bbox IoU vs GT; attribute (color) accuracy
- End-to-end perception latency per frame

## Progress checklist
- [x] Docs: this reference reconciled and written
- [x] `sam_mapper` package scaffolded, builds under colcon
- [x] Spike: `facebook/sam3.1` — does not load directly, needs conversion (§4.5, parked)
- [x] Backend probe: IDs persist across frames (23/30 frames) — ByteTrack drop validated
- [x] Resolution sweep: non-square unsupported; square 1008/672/336 are the options
- [x] Throughput measured: 7.3 s/frame — optimisation deferred
- [x] `kernels` warning fixed: version pin (§3.6 lever 5) — mask NMS/hole-fill/sprinkle now active
- [ ] Square 672/336 presets: propagation bug found + one-line fix applied, not yet build-verified (§3.6 backlog) — deferred, default 1008 unaffected
- [x] **Full node end-to-end on `scene_0`** — 2D detect+track, lidar fusion, 3D boxes all confirmed
  live via `bag-play` + `sam-map` (see Log). Throughput is worse than expected — next up.
- [x] **Throughput root cause found**: GIL contention between SAM3's worker thread and
  `/state_estimation` (50 Hz) + `/registered_scan` callbacks — confirmed via `camera_only_debug`
  (returned to 1.3-2.7 s/frame) and `nvidia-smi dmon` (GPU busy only in brief bursts).
- [x] **Split into `sam_node` + `map_node`** (§3.6-split) — separate processes, separate
  GILs; instance-ID-map + JSON wire format. Implemented and confirmed working (Log).
- [x] **Decoupled from `semantic_mapping`** (§3.1) — own trimmed `ObjMapper`/`CloudImageFusion`/
  `SingleObject`; `compat/` shims gone. Captioner/adjacency-graph/oriented-box capability kept
  dormant, not lost (§6 items 3-5). **Not yet re-run end-to-end since the port** ← current.
- [ ] Same node(s), live sim, no code change
- [ ] Duplicate rate measured on identical-furniture scene
- [ ] Tuning guide (§5) filled with real numbers
- [ ] Attributes (color, size) extracted and validated

## Suggestions (still current)
- **Read the organizer's panorama+lidar→RGB-D bridge**: [Navigation-Physical-Experiment](https://github.com/Yuxin916/Navigation-Physical-Experiment) converts pano images + lidar into registered RGB-D — a working reference for image↔lidar projection plumbing. Dev tip: `without_360_camera` env variant renders faster for non-perception work.
- **Checking it works.** `just sam-status` reports the rate on every output topic and summarises the 3D map (object count by class, plus centroids — obviously-wrong positions show up immediately). For visuals, `just foxglove` and watch `/annotated_image` (masks labelled `label#id score`, coloured per SAM 3 id so an id switch shows as a colour flip) alongside `/obj_boxes` and `/obj_points` in 3D.
- Keep a per-instance "best crop" (largest, most frontal) — M4 uses it for VLM verification of borderline attributes.
- Color words: use VLA-3D's exact 15-color LAB/CSS3 mapping (`3d_data_preprocess/utils/dominant_colors_new_lab.py`) — the questions' color vocabulary comes from it; sample interior 70% of mask. VLA-3D was extended/filtered as IRef-VLA — download data from there.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
| Jul 27 | SAM 3 chosen; whole-panorama single-pass replaces the 6-crop plan; expanded into the full perception reference below |
| Jul 27 | `sam_mapper` package built; container builds clean. **Probe on `scene_0`: tracking CONFIRMED** (23 ids stable across all 30 frames) — validates dropping ByteTrack + spaCy. **Throughput 0.14 Hz**, ~40× worse than estimated; ~33 objects/frame, per-object memory attention is the leading suspect. Non-square `image_size` **does not work** (square-grid reshape hardcoded); `facebook/sam3.1` needs checkpoint conversion (§4.5). |
| Jul 28 | **`kernels` warning fixed**: `transformers.is_kernels_available()` needs `0.15.2 <= version < 0.16.0`; unpinned Dockerfile install grabbed 0.16.0, one patch too new. Pinned + live-downgraded; `cv-utils` kernel now loads (mask NMS/hole-fill/sprinkle active). Also found (but deferred, see §3.6 backlog): a real propagation bug behind the square `672`/`336` preset failures, a `sam_mapper` build-not-picking-up-source-edit issue, and a new `flash-attn` kernelization `KeyError` on repeat model reload in one process — none block the default `square_1008` path. Added `runtime.verbose_objects` (§3.5) for per-frame 2D+3D terminal detail ahead of the end-to-end run. |
| Jul 28 | **Full node end-to-end on `scene_0`: CONFIRMED working** via `bag-play` + the (then combined) mapper node — correct 2D detections, persistent SAM 3 ids, real 3D boxes/centroids, correct bag-loop resync. **But throughput root-caused**: GIL contention between the worker thread and `/state_estimation` (50 Hz) + `/registered_scan` callbacks — `camera_only_debug` (drop both, keep only `/camera/image`) returned the same node to 1.3-2.7 s/frame, and `nvidia-smi dmon` confirmed the GPU was busy only in brief scattered bursts otherwise. **Fix: split into `sam_node` (2D, own process) + `map_node` (3D)** (§3.6-split), wired through a new instance-ID-map + JSON-sidecar format so `ObjMapper` needed no changes. Each node now has its own launch file (`sam_node.launch`, `map_node.launch`) and `just` recipe (`just sam_node`, `just map_node`); end-to-end re-verification of the split is next. |
| Jul 28 | **Full node end-to-end on `scene_0`: CONFIRMED working.** `bag-play` + `sam-map` (default `square_1008`, `verbose_objects: true`) produced correct 2D detections, persistent SAM 3 ids across frames, real 3D boxes with plausible centers/extents (near-zero extent on wall-mounted flats like `door`/`painting` is expected, not a bug), and correct bag-loop resync. **Throughput is worse than the offline probe**: 30-79 s/frame in-node vs ~2.3-7 s/frame standalone (§3.6's GIL-starvation hypothesis, `executor_threads: 2` apparently insufficient on its own) — next thing to dig into. `tracked` vs `published` ran ~1.5-2x (e.g. 37 vs 20), within the documented "thin/noisy cloud" range, not alarming. Also: `just sam-frames`' default `out` was `/data/frames`, contradicting its own comment about `/data` being root-owned — fixed to `/data/bags/_frames`. Added `sam3_backend --verbose`/`--save-annotated` (writes `annotate_frame()` overlay PNGs) for lidar-free 2D-only visual sanity checks on dumped frames before committing to a bag run. |
| Jul 28 | **`sam_mapper` decoupled from `semantic_mapping` entirely** (§3.1, §3.6-split). Removed `use_lidar_odom`/`/aft_mapped_to_init_incremental` (real on the physical rig, but `/state_estimation` alone was judged sufficient on both platforms). Ported a trimmed `CloudImageFusion`/`ObjMapper`/`SingleObject`/`VoteStatistics` into `sam_mapper/{cloud_image_fusion,object_mapper,single_object,ros_markers}.py`, dropping genuinely dead code (ByteTrack/spaCy `track_objects` path — superseded by SAM3, not deferred; `open3d_vis`/`rerun_vis`/`print_obj_info` — ROS/Foxglove already covers this; several never-called `SingleObject` methods; two duplicate `find_neighbouring_stamps` definitions and a stray module-scope spaCy load in `semantic_mapping/utils.py`, now avoided by inlining the one function actually needed). **Kept deliberately dormant, not trimmed** — real planned improvements, not dead ends: the captioner hook, `AdjacencyGraph` co-visibility (+ its commented-out IoU-merge consumer, ported verbatim), and `infer_bbox_oriented` (still available, not yet wired into `serialize_map_to_dict`/`to_ros2_msgs`) — see §6 items 3-5. Found and fixed two small bugs while porting: `compute_valid_indices` ran DBSCAN clustering twice per object per frame in some cases (redundant `cal_clusters()` call), and `retrieve_valid_voxel_indices` computed an unused weighted-total from mismatched (filtered vs. unfiltered) array indices. `compat/` (bytetrack/rerun import shims) and its Dockerfile `PYTHONPATH` line are gone — nothing needs them anymore. |
| Jul 28 | **`map_node` crash fixed** (§3.6-split "Wire format"): `_take_detection_frame` compared `header.stamp` on both `/sam3/instance_map` (a real `sensor_msgs/Image`) and `/sam3/detections` (a `std_msgs/String`, which has no `header` at all) — `AttributeError` every time, thrown outside `_worker_loop`'s try/except, silently killing the worker thread on the first successful pairing. ROS callbacks and the heartbeat kept running, so the node just looked permanently stuck "waiting on sync" with climbing counts rather than visibly crashed. Fix: embed the stamp in the `/sam3/detections` JSON payload itself (`{stamp, entries}`) and compare that instead. |

---
---

# Reference

## 0. How to read this

Two packages live under `ai_module/src/`:

| Package | What it is | State |
|---|---|---|
| `semantic_mapper/` | The **existing**, upstream 3D mapping module (CMU, Guofei Chen). Grounding DINO + spaCy + ByteTrack + SAM2 front-end, voxel-voting 3D back-end. | works; **not modified** by us |
| `sam_mapper/` | The **new** package. Own, trimmed port of the 3D back-end (§3.1 — no longer depends on `semantic_mapping` at all) plus SAM 3 replacing the entire four-stage front-end. | end-to-end confirmed working (2026-07-28); throughput unoptimised |

Stability of what follows: §2 and §3 document shipped code. §4 is verified against transformers
source and measurement. §5 still carries default values rather than measured ones.

---

## 1. System context

`sam_mapper` is two processes, split 2026-07-28 (§3.6-split) so SAM3's GPU-heavy inference has its
own GIL, immune to `semantic_mapper`'s (or its own) high-frequency ROS callbacks:

```
                    ┌──────────────┐  /annotated_image  (debug overlay)
 /camera/image ────►│  sam_node    │  /sam3/instance_map (mono16 id map)      ┌──────────────────────┐
   10 Hz, 1920×640  │              │──────────────────────────────────────►  │  map_node             │──► /obj_points
   equirect 360×120 │  detect      │  /sam3/detections   (JSON sidecar)      │  (or older             │──► /obj_boxes
                    └──────────────┘                                        │   semantic_mapper)     │──► /obj_labels
 /registered_scan ──────────────────────────────────────────────────────────►│  project → fuse →      │──► /obj_map_json
   world-frame lidar                                                        │  3D instances          │
 /state_estimation ─────────────────────────────────────────────────────────►│                        │
   ~50 Hz odometry                                                          └──────────────────────┘
                                                                                        │
                                                                                        ▼  M3 scene graph → M4 reasoner
```

### `wait_for_prompts` — booting unarmed

`sam_node` normally starts with the prompts in its config's `objects:` list. Pass
`wait_for_prompts:=true` (`sam_node.launch`, set by `smart_vlm.launch`) and it instead boots
**unarmed**: weights still load — the slow part, ~60 s — but `prompt_table` stays `None`, every
`/camera/image` is dropped without buffering, and no best-view run directory is created.
`/sam3/status` reports `awaiting_prompts` rather than `ready`.

The first `/sam3/set_prompts` arms it through the existing path: fresh `PromptTable`, fresh SAM 3
session, fresh `BestViewCollector` with its own `run_dir`. The category-1 pipeline needs this,
because its prompts are derived from the question — anything detected before they arrive would be
against the config's placeholder objects and would pollute the run's crops. A rejected
`set_prompts` rolls `armed` back to whatever it was, so a bad request can never leave the node
armed with no prompt table.

**Bag and live sim are interchangeable.** Both publish the same three source topics; the nodes
subscribe and cannot tell them apart. `just bag-play scene_0` and `just sim-noviz` are drop-in
alternatives. There is no bag-specific code path anywhere in perception.

**Coordinate frames.** `/registered_scan` arrives already in the **world** frame.
`/state_estimation` gives the body pose in world. Projection into the image needs the cloud in the
**body** frame, so every frame does world→body, projects, masks, then body→world again to accumulate.

---

## 2. `semantic_mapper` — the existing package

Upstream: [semantic_mapping_with_360_camera_and_3d_lidar](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar) (ros2 branch), vendored here.

### 2.1 Layout and how it runs

```
semantic_mapper/
├── ros2_system.py            # near-duplicate of the node's __main__ (entrypoint variant)
├── config/                   # 3× mapping_*.yaml, 2× objects_*.yaml, obj_vis.rviz
├── semantic_mapping/         # the package
│   ├── mapping_ros2_node.py  # 735 L — the only ROS2 node
│   ├── semantic_map.py       # 753 L — ObjMapper: tracking + 3D association
│   ├── single_object.py      # 635 L — SingleObject / VoteStatistics / AdjacencyGraph
│   ├── cloud_image_fusion.py # 406 L — equirect lidar↔image projection
│   ├── visualizer.py         # 361 L — VisualizerRerun (unused; rerun is shimmed)
│   └── utils.py              # 178 L — incl. extract_meta_class (spaCy)
├── external/{Grounded-SAM-2, byte_track, cython_bbox}
└── tests/                    # 16 demo scripts, not a pytest suite
```

**It is not a colcon package.** Unlike every sibling under `ai_module/src/`, it has no `package.xml`
and no `setup.py` — just `setup.cfg` with `packages = semantic_mapping`. It is pip-installed and run
as a module:

```bash
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml
```

not `ros2 run`. This matters for the Dockerfile: it must be `pip install -e`'d, not colcon-built.

### 2.2 ROS interface

**Subscriptions** (`mapping_ros2_node.py:192-246`)

| Topic | Type | Callback | QoS | Purpose |
|---|---|---|---|---|
| `/camera/image` | `sensor_msgs/Image` | `image_callback` | 10 | panorama → `rgb_stack` (max 10) |
| `/registered_scan` | `PointCloud2` | `cloud_callback` | 10 | world-frame lidar |
| `/aft_mapped_to_init_incremental` | `nav_msgs/Odometry` | `lidar_odom_callback` | 10 | only if `use_lidar_odom` |
| `/state_estimation` | `nav_msgs/Odometry` | `odom_callback` | 50 | high-rate odometry |
| `/explored_areas` | `PointCloud2` | `global_cloud_callback` | 10 | Rerun backdrop |
| `/terrain_map_ext` | `PointCloud2` | `generate_freespace` | 10 | `intensity < 0.05` → traversable |
| `/object_query` | `std_msgs/String` | `handle_object_query` | 1 | JSON query → captioner CLIP search |

**Publishers** (`mapping_ros2_node.py:248-257`)

| Topic | Type | Note |
|---|---|---|
| `/obj_points` | `PointCloud2` | xyz + packed rgb float32 |
| `/obj_boxes` | `MarkerArray` | LINE_LIST wireframes, prefixed by a `DELETEALL` marker |
| `/obj_labels` | `MarkerArray` | TEXT_VIEW_FACING |
| `/queried_captions` | `String` | JSON |
| `/traversable_area` | `PointCloud2` | |
| `/annotated_image` | `Image` | **declared but never published** — publish is commented out at `mapping_ros2_node.py:472-476` |

**Timers:** `mapping_timer` @ 0.5 s → `mapping_callback`; `caption_pub_timer` @ 0.1 s.

**Threading.** Every subscription declares a `MutuallyExclusiveCallbackGroup`, but the entrypoint runs
plain `rclpy.spin` — the `MultiThreadedExecutor` is commented out (`ros2_system.py:119-124`), so the
callback groups do nothing. `mapping_callback` spawns a bare `threading.Thread` per frame
(`mapping_ros2_node.py:598`) guarded by `mapping_processing_lock`.

### 2.3 The pipeline, stage by stage

1. **Ingest** — per-topic stacks under mutexes. Images capped at 10; the timer consumes
   `rgb_stack[-2]`, not `[-1]`.
2. **Time sync** (`mapping_callback`, `:498-602`) — applies configurable linear/angular time biases,
   linearly interpolates position/velocity and **SLERPs orientation** between the two odom samples
   bracketing the image stamp, then gathers all clouds in `[stamp-0.5, stamp+0.1]` (`:584`).
3. **Detect** — Grounding DINO (`inference`, `:264`), `box_threshold=0.35`, `text_threshold=0.35`.
4. **Meta-class + track** (`ObjMapper.track_objects`, `semantic_map.py:157`) — maps free-form DINO
   phrases to config meta-classes via spaCy, then ByteTrack assigns persistent ids, with map-derived
   3D centroids reprojected into the panorama to correct each tracklet's Kalman state
   (`compensate_with_3d`).
5. **Segment** — SAM2 `predict(box=tracked_bboxes, multimask_output=False)` (`:393`).
6. **Fuse** — `CloudImageFusion.generate_seg_cloud` (§2.5).
7. **Map update** — `ObjMapper.update_map` (§2.6).
8. **Publish / visualize** — MarkerArrays + coloured PointCloud2, plus Rerun.

### 2.4 Data structures

**`det_result`** — raw detector output (`mapping_ros2_node.py:294`):
```python
{"bboxes": (N,4) float xyxy pixels, "labels": (N,) str, "confidences": (N,) float}
```

**`detections_tracked`** — **the contract into the 3D stage.** Built at `semantic_map.py:186-232`,
gains `masks` at `mapping_ros2_node.py:407`:
```python
{'bboxes':      np.ndarray (M,4)   float xyxy,
 'confidences': np.ndarray (M,)    float,
 'labels':      np.ndarray (M,)    str,    # meta-class
 'ids':         np.ndarray (M,)    int,    # >=0 instance, <0 background
 'masks':       np.ndarray (M,H,W) bool}   # H,W = 640,1920
```
Background ids are `-BACKGROUND_OBJECTS.index(label) - 1`. A background object reaching the tracker
raises `ValueError` (`semantic_map.py:221`).

**odom dict** — used everywhere as `camera_odom` / `detection_odom` (`:357-366`):
```python
{'position': [x,y,z], 'orientation': [x,y,z,w],   # scipy xyzw
 'linear_velocity': [...], 'angular_velocity': [...]}
```

**`VoteStatistics`** (`single_object.py:51`) — the heart of the 3D representation:
```python
self.voxels             # (V,3) world coords
self.tree               # cKDTree(voxels)
self.vote               # (V,)   times observed
self.observation_angles # (V,B)  one-hot per viewing-angle bin, B = num_angle_bin
self.regularized_voxel_mask  # (V,) DBSCAN-kept
```

**`SingleObject`** (`single_object.py:239`) — one 3D instance:
```python
self.class_id   # {label: vote_count} — dominant label wins
self.obj_id     # list of track ids merged into this instance
self.vote_stat  # VoteStatistics
self.life, self.inactive_frame, self.key_frames, self.key_pose, self.latest_stamp
```

**`DIMENSION_PRIORS`** (`single_object.py:9`) — per-class size caps used by shape regularization:
`default (5,5,2)`, `table (5,3,2)`, `chair (1.5,1.5,2)`, `sofa (3,3,2)`, `pottedplant (1,1,1)`,
`fireextinguisher (0.5,0.5,0.5)`.

**`AdjacencyGraph`** (`single_object.py:595`) — co-visibility graph. Maintained every frame, but its
only real consumer (the IoU split/merge path) is commented out.

### 2.5 Equirectangular projection

All projection is equirectangular, never pinhole (except the ScanNet variant). Canonical size
**1920×640**, `hfov 360°`, `vfov 120°` (30° cropped top and bottom).

`scan2pixels` (`cloud_image_fusion.py:7-47`) — lidar→camera extrinsics, then:
```python
horiDis     = sqrt(x² + y²)
horiPixelID = (-W/(2π) · atan2(y, x) + W/2 + 1).astype(int) - 1
vertPixelID = (-W/(2π) · atan2(z, horiDis) + H/2 + 1 + vertPixelOffset).astype(int)
PixelDepth  = horiDis
return np.array([horiPixelID, vertPixelID, PixelDepth]).T.astype(int)
```

**Why both axes scale by `W/(2π)`** — that is not a typo. In an equirect image, radians-per-pixel is
identical horizontally and vertically; the horizontal axis spans 2π over `W` pixels, so `W/(2π)` is
*the* rad→pixel constant for both. `H` only sets the vertical centre.

**Quirk:** the final `.astype(int)` also truncates the depth channel, so `PixelDepth` is integer
metres. Harmless in practice — it is used only for the debug colour overlay (`maxRange = 6.0`, so 6
discrete bands). The real 3D points come from `cloud[cloud_mask]` at full precision.

Five platform variants, each hardcoding its own extrinsics:

| Function | camera x,y,z | roll,pitch,yaw |
|---|---|---|
| `scan2pixels_wheelchair` (`:49`) | 0, 0, 0.235 | 0, 0, 0 |
| `scan2pixels_mecanum_sim` (`:67`) | 0, 0, 0.1 | −π/2, 0, −π/2 |
| `scan2pixels_mecanum` (`:106`) | −0.12, −0.075, 0.265 | −π/2, 0, −π/2 |
| `scan2pixels_diablo` (`:144`) | 0, 0, 0.185 | −π/2, 0, −π/2 |
| `scan2pixels_scannet` (`:182`) | pinhole, fx 1169.62 fy 1167.11, 1296×968 | — |

`generate_seg_cloud` (`:276`) filters out-of-bounds pixels, then per mask indexes
`obj_mask[v, u]` to select points, and returns them transformed back to world.

### 2.6 The 3D mapping stage — `ObjMapper.update_map`

`semantic_map.py:241-557`. Parameters at `:138-144`: `voxel_size 0.05`, `confidence_thres 0.30`,
`cloud_to_odom_dist_thres 6.0`, `ground_height -0.5`, `num_angle_bin 20`, `percentile_thresh 0.8`.

1. World→body via `R_w2b = R_b2w.T`, `t_w2b = -R_w2b @ t_b2w`.
2. Confidence gate at `0.30`, applied to masks/labels/ids/bboxes in lockstep.
3. **Mask erosion** — `cv2.erode(mask, 3×3, iterations=5)` (`:267-269`). Shrinks masks so lidar points
   near a silhouette edge don't bleed onto background. This is the main knob for bleed-through.
4. Adjacency graph edges for every co-visible pair.
5. Fuse → per-object world clouds.
6. Per object: drop points >6 m from the sensor, skip if <5 points, voxel downsample 0.05, then
   `remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)`.
7. **Associate by track id** — if `obj_id in single_obj.obj_id`, merge and re-project observation
   angles; else create a new `SingleObject`. Negative ids go to `background_obj_list` (the background
   merge branch is commented out at `:312-318`, so background objects are recreated every frame).
8. **World-space association** (`:376-554`) — for objects with `5 < life < 1000`: prune if
   `valid_indices_regularized.shape[0] < 20 and inactive_frame > 5`; else find the nearest
   same-dominant-label object with matching verticality and merge if
   ```python
   dist_thresh = np.linalg.norm((extent_object/2 + extent_target/2)/2) * 0.5   # :434
   if minimum_dist < dist_thresh or minimum_dist < 0.5:                        # :437
   ```
   **This is the duplicate-instance safety net** — it is what lets us tolerate a tracker that
   occasionally splits one object into two ids.

**Voxel voting, explained.** Each voxel records *which directions it has been seen from*, quantized
into `num_angle_bin` bins. A voxel's confidence is its **angular diversity** — `sum(obs_angles)`,
i.e. how many distinct viewpoints have seen it. Points seen from one angle only are likely
projection artefacts (bleed-through, mis-registration); points seen from many angles are really
there. `retrieve_valid_voxel_indices(diversity_percentile)` keeps the top-`percentile` mass, and the
centroid is a diversity-weighted average. `regularize_shape` then runs DBSCAN and clips against
`DIMENSION_PRIORS`.

### 2.7 Config schema

```yaml
platform: mecanum_sim        # selects scan2pixels_* extrinsics
use_lidar_odom: false
detection_linear_state_time_bias: 0.0
detection_angular_state_time_bias: 0.0
image_processing_interval: 0.5   # parsed; throttle code is commented out
visualize: true
vis_interval: 0.5
annotate_image: true
prompts:
  <meta_class>:
    prompts: ["phrase 1", "phrase 2"]   # fed to Grounding DINO
    is_instance: true|false             # false -> background, negative ids
```
All prompt strings are flattened into one DINO text prompt joined by `" . "`
(`mapping_ros2_node.py:134-138`). DINO's free-form output phrase is mapped back to a meta-class by
`utils.extract_meta_class` (`utils.py:61-93`): exact phrase → spaCy NOUN/VERB token match
(`en_core_web_sm`) → whole-word equality. Returns `None` on no match.

Files: `mapping_mecanum_sim.yaml`, `mapping_mecanum_real.yaml`, `mapping_wheelchair.yaml`; plus
`objects_cic.yaml` / `objects_sqh.yaml` in an older flat format used only by tests.

### 2.8 `external/`

All three are vendored source copies (regular git files, not submodules).

- **`Grounded-SAM-2/`** — full `sam2/` package + demos. **`grounding_dino/groundingdino/models/` is
  absent**, so the local-GroundingDINO path cannot work; only the HuggingFace path is used.
  No demo script here is imported by `semantic_mapping/`.
- **`byte_track/`** — a *customized* ByteTrack: `STrack` carries `class_name` and `curr_bbox`;
  `BYTETracker(dimension_constraints=[1920, 640])` bakes the panorama size into the state clamp;
  adds `compensate_with_3d()` and `compensate_orientation()`.
- **`cython_bbox/`** — the IoU helper ByteTrack's `matching.py` needs.

**None of `external/` is installed in the container.** That needed care while `sam_mapper` still
imported `semantic_mapping.semantic_map` directly, because that module does heavy imports at module
scope regardless of whether they're actually used:

| Import | Where | Used by the original `ObjMapper`? |
|---|---|---|
| `from bytetrack.byte_tracker import BYTETracker` | `semantic_map.py:11`, annotation at `:108` | only by `track_objects` |
| `cython_bbox`, `lap` | `external/byte_track/bytetrack/matching.py:4,7` | transitively, via ByteTrack |
| `spacy` + `en_core_web_sm` | `utils.py:59-60` (runs `spacy.load()` on import) | only by `track_objects` (`extract_meta_class`) |
| `line_profiler` | `semantic_map.py:25`, `single_object.py:7` | decorator only |
| `import rerun as rr` | `visualizer.py:2`, reached via `semantic_map.py:17` | only by `rerun_vis` |
| `sam2` / Grounded-SAM-2 | old node only | Grounding-DINO+SAM2 front end |

**Consequence:** the original `semantic_mapping.mapping_ros2_node` cannot run in this image — it
needs the real ByteTrack *and* Grounded-SAM-2's `sam2`. The regression comparison in the milestone
checklist therefore needs a separate environment.

**Historical note (resolved 2026-07-28, §3.6-split):** `sam_mapper` used to import `ObjMapper`/
`CloudImageFusion` straight from `semantic_mapping`, so `bytetrack`/`rerun` needed import shims
(`sam_mapper/compat/`) purely to satisfy those two module-scope imports above, even though the parts
of `ObjMapper` that actually use them (`track_objects`, `rerun_vis`) were never called. Once the
table above made clear that everything sam_mapper *actually* needed was tracker/rerun-free, the
cleaner fix was porting a trimmed `ObjMapper` into `sam_mapper` itself (§3.1) — `compat/` is gone,
and `sam_mapper` no longer depends on `semantic_mapping` at all.

### 2.9 Known bugs and quirks

Each verified against source. These are the concrete "improve later" items.

| # | Issue | Where |
|---|---|---|
| 1 | `semantic_mapping/scannet_utils.py` does not exist → 3 test scripts unimportable | `tests/test_scannet_mapping*.py`, `verify_scannet_projection_offline.py` |
| 2 | Duplicate/empty `prompts:` key — YAML keeps the last, so it works by luck | `config/mapping_mecanum_sim.yaml:10-11` |
| 3 | RViz config references `/semantic_cloud` and `/bbox3d`, which are never published | `config/obj_vis.rviz` |
| 4 | `cloud_stack_left` / `cloud_stack_right` are never read; window hardcoded | `mapping_mecanum_real.yaml`, vs `mapping_ros2_node.py:584` |
| 5 | `generate_seg_cloud` returns a bare `list` normally but `(None, None)` when masks are empty | `cloud_image_fusion.py:281` |
| 6 | `horDis = horDis` before assignment → `UnboundLocalError` if `image_src` is passed | `cloud_image_fusion.py:398` (v2 only; v1's copy at `:312` is a harmless no-op) |
| 7 | `INSTANCE_LEVEL_OBJECTS` / `BACKGROUND_OBJECTS` are **module globals** mutated per `ObjMapper` instance — state leaks across instances | `semantic_map.py:74-97`, mutated at `:149-152` |
| 8 | `extract_meta_class` returns Python `None`; callers compare against the **string** `'None'` | `utils.py:93` vs `semantic_map.py:169,178` |
| 9 | `OMIT_OBJECTS = ["window","door"]` silently drops `door` even though the sim config declares `door: is_instance: true` | `semantic_map.py:91` vs `mapping_mecanum_sim.yaml` |
| 10 | `/annotated_image` publisher declared, publish commented out | `mapping_ros2_node.py:472-476` |
| 11 | `MultiThreadedExecutor` commented out despite per-subscription callback groups | `ros2_system.py:119-124` |
| 12 | `image_processing_interval` parsed but its throttle is commented out | `mapping_ros2_node.py:309-323` |
| 13 | `ros2_system.py`'s `load_models` closes over a `device` global only defined under `__main__` | `ros2_system.py` |
| 14 | `CloudImageFusion.__init__` sets `self.scan2pixels` twice (once by `eval`, once by if/elif); the `else: raise` is dead | `cloud_image_fusion.py:260-274` |
| 15 | `tests/` has no pytest config, no fixtures, and hardcoded absolute dataset paths | `tests/` |

---

## 3. `sam_mapper` — the new package

### 3.1 Why it exists

SAM 3 does detection, segmentation **and** tracking in one pass from noun-phrase prompts. That
collapses four stages into one and removes the two most brittle links in the old chain:

| Old stage | Fate under SAM 3 |
|---|---|
| Grounding DINO | replaced |
| spaCy `extract_meta_class` | **deleted** — SAM 3 returns `prompt_to_obj_ids`, so the label is known exactly |
| ByteTrack (+ `cython_bbox`) | **deleted** — SAM 3 object ids are already stable across frames |
| SAM2 | replaced |

**The enabling fact:** `ObjMapper.update_map()` never references `self.tracker` anywhere in
`semantic_map.py:241-557`. It uses only `cloud_image_fusion`, `single_obj_list` and `captioner`. So
`ObjMapper` can be driven purely through `update_map`, `to_ros2_msgs` and `serialize_map_to_dict`,
with no tracker at all.

**The 3D stage was originally imported unmodified from `semantic_mapping`** (hence the `compat/`
bytetrack/rerun import shims, needed only because `semantic_map.py`/`visualizer.py` import those at
module scope). Ported into `sam_mapper` as its own, trimmed copy on 2026-07-28 (§3.6-split) — the
shims were satisfying imports for code paths (`track_objects`, `open3d_vis`/`rerun_vis`, ByteTrack
tracking) sam_mapper never used, and going through them precisely was also the occasion to find and
keep three things that ARE still on the roadmap (captioner hook, adjacency-graph co-visibility for
duplicate-instance disambiguation, oriented boxes) rather than losing them along with the dead code.
`sam_mapper` now depends on nothing from `semantic_mapping` — `compat/` is gone.

| Symbol | Now lives at |
|---|---|
| `CloudImageFusion` | `sam_mapper/cloud_image_fusion.py` (trimmed: only `mecanum_sim`/`mecanum`) |
| `ObjMapper` | `sam_mapper/object_mapper.py` (trimmed — see its module docstring for the full list) |
| `SingleObject`, `VoteStatistics` | `sam_mapper/single_object.py` (trimmed) |
| marker/point-cloud builders | `sam_mapper/ros_markers.py` (5 functions, avoids a `rosbag2_py` import) |

### 3.2 Layout

```
ai_module/src/sam_mapper/          # ament_python, unlike semantic_mapper
├── package.xml, setup.py, setup.cfg, resource/sam_mapper
├── config/{sam3_mecanum_sim.yaml, sam3_mecanum_real.yaml}
├── launch/{sam_node.launch, map_node.launch}   # one node each (§3.6-split)
└── sam_mapper/
    ├── sam3_backend.py      # backend protocol + HF impl (+ native 3.1 impl if needed)
    ├── detections.py        # SAM3 result -> the 5-key dict; label/ID assignment; id-map codec
    ├── sam_node.py          # ROS2 node: SAM3 only — /camera/image in, detections out
    ├── map_node.py          # ROS2 node: lidar fusion / 3D mapping — the old "one node" (§3.6-split)
    └── annotate.py          # debug overlay -> /annotated_image
```

**Two nodes, not one — reversed 2026-07-28 (§3.6-split).** The original design kept SAM3 and the 3D
stage in one process, reasoning that splitting them would serialize `(N, 640, 1920)` bool masks across
ROS — ~1.2 MB/object/frame, ~120 MB/s at 10 objects × 10 Hz. That estimate assumed 10 Hz throughput,
which profiling showed was never real (§3.6), and it missed a bigger cost of staying in one process:
SAM3's GPU work shares a GIL with the 3D node's high-frequency ROS callbacks, which turned out to
dominate wall-clock time far more than any serialization would. See §3.6-split for the fix and the
wire format that keeps per-frame data small regardless of object count.

### 3.3 Backend interface

```python
@dataclass
class Sam3FrameResult:
    object_ids: np.ndarray                    # (N,) int
    scores: np.ndarray                        # (N,) float
    boxes: np.ndarray                         # (N,4) xyxy absolute pixels
    masks: np.ndarray                         # (N,H,W) bool
    prompt_to_obj_ids: dict[str, list[int]]

class Sam3VideoBackend(Protocol):
    def set_prompts(self, prompts: list[str]) -> None: ...
    def process_frame(self, rgb: np.ndarray) -> Sam3FrameResult: ...
    def reset(self) -> None: ...
```

Everything downstream depends only on this, which is why the SAM 3 → SAM 3.1 swap is contained.

**Output mapping** — SAM 3 fills every key of the existing contract:

| SAM 3 output | → mapper key |
|---|---|
| `boxes` (N,4) xyxy absolute | `bboxes` — already in panorama pixels, no conversion |
| `scores` | `confidences` — `ObjMapper.confidence_thres = 0.30` still applies |
| `masks` at original resolution | `masks` — `.cpu().numpy().astype(bool)` |
| `object_ids` | `ids` |
| `prompt_to_obj_ids` | `labels` |

### 3.4 Label and ID assignment (`detections.py`)

1. Build `prompt → (meta_label, is_instance)` from config.
2. Per frame, invert `prompt_to_obj_ids` into `obj_id → prompt`.
3. If one `obj_id` appears under several prompts, keep the prompt whose detection scored highest.
4. For `instance: false` prompts, **overwrite** the SAM 3 id with the negative background index
   `-idx - 1`, preserving the mapper's convention.
5. Emit the 5-key dict.

`ObjMapper.__init__` wants `label_template` shaped
`{label: {'is_instance': bool, 'prompts': [...]}}` and mutates it plus the `INSTANCE_LEVEL_OBJECTS`
global (quirk #7). Build that adapter shape here so `ObjMapper` is constructed unmodified.

`OMIT_OBJECTS` never applies to us — it is read only inside `track_objects`, which we bypass. If you
don't want a class, don't prompt for it.

`encode_instance_id(obj_id)` (added §3.6-split) maps a real `obj_id` (`>=0` instance, `<0`
background) to a positive uint16 pixel value for the `/sam3/instance_map` wire format —
`obj_id + 1` for instances, `65536 + obj_id` for background (top of the range, never collides).
`sam_node` encodes with it, `map_node` decodes with it.

### 3.5 Config

```yaml
platform: mecanum_sim        # -> CloudImageFusion extrinsics
detection_linear_state_time_bias: 0.0
detection_angular_state_time_bias: 0.0

sam3:
  backend: hf_sam3                        # hf_sam3 | native_sam3.1
  model_id: facebook/sam3
  device: cuda
  dtype: bfloat16
  attn_implementation: flash_attention_2  # falls back to sdpa
  max_vision_features_cache_size: 1
  image_size: [1008, 1008]                # [h,w] — see §4.3
  backbone_feature_sizes: null            # must be set when image_size is non-square
  score_threshold_detection: 0.5
  new_det_thresh: 0.7
  det_nms_thresh: 0.1
  recondition_every_nth_frame: 16

objects:
  - {prompt: "chair", instance: true}
  - {prompt: "wall",  instance: false}
  # ...

runtime:
  max_inference_hz: 10.0
  publish_annotated: true
  verbose_objects: false   # print every 2D detection + 3D map object per frame (validation)
visualize: true
vis_interval: 0.5
```

Sim vs. real differ only in the platform block:

| key | `_sim` | `_real` |
|---|---|---|
| `platform` | `mecanum_sim` (camera at z=0.1) | `mecanum` (camera at −0.12,−0.075,0.265) |
| `detection_*_time_bias` | `0.0` | tune against the rig |

`use_lidar_odom` (real-rig lidar SLAM odometry, `/aft_mapped_to_init_incremental`) was removed
2026-07-28 — `/state_estimation` alone covers both platforms, and the code path was untested
dead weight. `semantic_mapper` (§2) still has it; unrelated to this package.

### 3.6 Real-time design

**Superseded 2026-08-11.** The 7339 ms/frame figure below was measured at 13 prompts / 33 objects
with `cv-utils` silently unloaded and every yaml threshold silently ignored (see the two bugs at the
end of this section). Current measurements, `sam_mapper` stage profiler (`just sam-profile`):

| regime | ms/frame | breakdown |
|---|---|---|
| offline, 1 prompt, 5-6 obj | 449 | encoder 178, tracker 208, detection 28 |
| offline, 5 prompts, 6.0 obj | 563 | encoder 187, tracker 233, detection 127 |
| **live `eval-cat1`, 3 prompts, 12.2 obj** | **663 (1.5 Hz)** | encoder 173, tracker 285, detection 97, node 50, off-seam 43 |

Fitted over those points, and predicting the 5-prompt/12-object case to within 8 ms:

```
SAM 3 model cost ~ 175 (vision encoder, FIXED)
                 + ~30 per PROMPT   (run_detection loops prompts over shared vision features)
                 + 37 + 20.4 * N    (tracker, per OBJECT)
```

**Tracking works** — the property the design rests on. 23 of ~33 ids held across all 30 frames in the
original run; re-check it after any prompt or threshold change.

**The old per-object hypothesis was directionally right.** The tracker *does* scale linearly with
object count, but at 20.4 ms/object, not the ~200 ms guessed here. `run_detection` also loops per
PROMPT (`modeling_sam3_video.py:588`), so prompt count is a separate linear cost — "N classes cost
~1 forward pass" is true only of the vision encoder.

**Two bugs found 2026-08-11, both silent, both now fixed:**

1. **No yaml threshold ever reached the model.** `Sam3VideoModel.__init__` copies each knob onto the
   module and the forward path reads the copy; `Sam3Backend` set them only on `model.config`.
   Verified in transformers 5.15: `self.<key>` is read in the model body, `self.config.<key>` **zero**
   times, for all five knobs. So `score_threshold_detection: 0.7` ran at the checkpoint's 0.5 and
   `det_nms_thresh` did nothing. Fixed in `Sam3Backend._apply_override` (writes both), logged as
   `[sam3] effective: ...` at startup, guarded by `tests/test_backend_config.py`.
2. **The `cv-utils` kernel was not loading**, so mask NMS was skipped entirely (`generic_nms` returns
   `is_valid`, keeping every detection) and hole filling was off. Two causes: torch 2.5.1 had no
   matching prebuilt variant, and transformers' `is_kernels_available()` window moved with an
   unpinned transformers bump. The torch pin and the `kernels` version are now both derived from what
   the kernel and transformers actually require — see `docker/requirements_captioner.txt`.

**Every map3d bench number predating 2026-08-11 was therefore produced by a different detector than
its config describes.**

Levers, re-ranked by measurement:
1. **`image_size: [672, 672]`** — 0.44× tokens against a fixed 175 ms encoder. The largest untouched
   lever at low object counts; weak at 20 objects where the encoder is only 14% of the frame. Gate on
   mask IoU (`--save-masks` / `--compare-baseline`).
2. **Cut tracked objects** — ~20 ms per object removed, now that thresholds actually apply.
3. **Prompt count** — ~30 ms each; cat1 already sends the minimum.
4. **flash attention: do NOT.** Measured: `kernels-community/flash-attn2` returns **0.0 objects/frame**
   against sdpa's 2.0, and is slower — `Sam3Attention` falls back to SDPA for relative-position
   cross-attention anyway. transformers 5.15 substitutes the hub kernel for a `flash_attention_2`
   request automatically, so `Sam3Backend.ATTN_FALLBACKS` is `("sdpa", "eager")` to keep it
   unreachable by accident. FA3 is Hopper-only; both deploy targets are Ada sm_89.
5. **`fill_hole_area: 0`** — disables cv-utils hole filling / sprinkle removal (mask NMS still runs).
   Cheap, but the latency saving attributed to it was thermally confounded and is unproven.
6. **SAM 3.1** (§4.5) — now measured, and viable for multi-concept.

**Square 672/336 backlog (2026-07-28, deferred).** Root cause of the reshape failure:
`sam3_backend.py` set `config.detector_config.image_size` directly instead of the top-level
`config.image_size` property, which is the one that fans out to `tracker_config` too
(`configuration_sam3_video.py:166-174`) — left at its 1008 default, the tracker's prompt-encoder grid
stays hardcoded 72×72. One-line fix applied (`config.image_size = ...`), confirmed correct in an
isolated script (produces a working 48×48 grid), but three loose ends before it can be called done:
1. the container's installed `sam_mapper` didn't pick up the source edit after `colcon build
   --packages-select sam_mapper` in-session — cause not yet found;
2. even past that, `--sweep-image-size` now also hits a `KeyError` from `transformers`' new
   `kernels`-based `flash-attn2` auto-substitution (kicks in now that lever 5 is fixed) on the
   *second* model reload in one process — only affects the multi-preset sweep, not the live node
   (single `Sam3Backend` per process);
3. mask-quality impact of smaller presets (post lever-5 fix) is unmeasured.
Default `square_1008` is unaffected by any of this — it's what ships and what end-to-end validation
uses meanwhile.

- `image_callback` writes a **single-slot** `latest_frame` under a lock, dropping any unconsumed
  frame. Lidar and odom keep ring buffers, because `update_map` needs the cloud window and the
  bracketing odom samples around the *image* stamp.
- One worker thread: grab latest frame → interpolate odom → gather clouds `[t-0.5, t+0.1]` →
  `backend.process_frame` → `detections.py` → `ObjMapper.update_map` → publish.
- `MultiThreadedExecutor` with per-subscription callback groups, so the GPU worker never blocks
  callbacks. (Doing properly what `semantic_mapper` set up but left commented out — quirk #11.)
  **Cap its thread count.** `MultiThreadedExecutor()` defaults to `cpu_count()` threads, and each
  spins in rclpy's Python-level wait loop holding the GIL, starving the worker that feeds the GPU.
  Measured on an L4: the offline probe (single-threaded, no ROS) ran inference at **7 s/frame with
  the GPU at 100%**, while the node ran the same backend at **20–37 s/frame with the GPU at 0%**.
  `runtime.executor_threads: 2` — there are four short subscriptions and one timer; the real work
  is on the worker thread. **Update 2026-07-28: this alone was not enough** — see §3.6-split; the
  actual fix was moving SAM3 to its own process.
- **Keep callbacks trivial.** The image callback stores the raw `sensor_msgs/Image` and defers
  `imgmsg_to_cv2` to the worker: ~99% of frames are dropped, so decoding on arrival burns CPU on
  images that are discarded (measured 565 decoded to use 8) and steals GIL time from inference.
- Log achieved Hz and dropped-frame count periodically. That number decides the `image_size` setting.

**Bag looping.** `bag_replay.launch` plays with `--loop` by default, so stamps jump backwards every
lap. Three things must reset together or the node breaks silently:

1. the odom/cloud ring buffers — otherwise they hold the previous lap's newer stamps and *every*
   frame fails sync with "frame older than oldest odom", a permanent stall;
2. the SAM 3 session — its object ids are only meaningful within one session;
3. the **id namespace** — SAM 3 restarts ids at 0 after a session reset, and `update_map` associates
   purely on `obj_id in single_obj.obj_id` (`semantic_map.py:321`), so a fresh id 0 would merge into
   whatever object held id 0 last lap. `_handle_time_jump` offsets new ids past the high-water mark
   so they can never collide.

   (Written when this was one node. Since §3.6-split, #1 is `map_node._handle_time_jump` and #2-3 are
   `sam_node._handle_time_jump` — see that section for exactly how the state divides.)

**Rerun is removed.** Everything is inspected through ROS 2 topics in Foxglove. Since the 2026-07-28
port (§3.1), `sam_mapper`'s own `ObjMapper` has no `visualize` param, `rerun_vis`, or rerun dependency
at all — not shimmed, just not there.

**Dropping frames is safe; reordering them is not.** SAM 3's tracker is memory-based, so skipping
frames just lowers the effective tracking rate. But frames must be fed **monotonically** to the one
session.

### 3.6-split — two nodes (2026-07-28)

**Diagnosis.** End-to-end validation on `scene_0` measured 20-80 s/frame against a live bag — worse
than the 20-37 s/frame above, and `runtime.executor_threads: 2` didn't fix it. A `camera_only_debug`
flag (drop `/registered_scan` + `/state_estimation`, keep only `/camera/image`) brought the same node,
same model, back to 1.3-2.7 s/frame — matching the standalone probe. `nvidia-smi dmon` sampled through
several live frames and confirmed the GPU was busy only in brief scattered bursts (spikes to 18-74%,
mostly 0%), not genuinely saturated the whole time. Conclusion: SAM3's forward pass is many small
Python-level steps per object; `/state_estimation` alone arrives at 50 Hz, so the executor threads are
constantly asking for the GIL, fragmenting the worker's progress. Thread-count capping helps but can't
remove this — only a separate process (separate GIL) can.

**Fix.** Split into two nodes, each with its own launch file so they run as separate processes:

| Node | Executable | Launch | Subscribes | Publishes |
|---|---|---|---|---|
| `sam_node` | `sam_node` | `sam_node.launch` | `/camera/image` | `/annotated_image`, `/sam3/instance_map`, `/sam3/detections` |
| `map_node` | `map_node` | `map_node.launch` | `/sam3/instance_map`, `/sam3/detections`, `/registered_scan`, `/state_estimation` | `/obj_points`, `/obj_boxes`, `/obj_labels`, `/obj_map_json` |

Both take the same `config:=` arg. `just sam_node` and `just map_node` run them in two terminals
(replaces the old single `just sam-map`).

**Wire format.** Sending `(N, 640, 1920)` bool masks separately was the original one-node design's
objection (§3.2) — but that assumed 10 Hz throughput, never real in practice. Instead `sam_node`
publishes one **instance-ID map** (`sensor_msgs/Image`, mono16, pixel = `encode_instance_id(obj_id)`,
`detections.py`) — a single H×W image regardless of object count, and directly viewable/colorizable in
RViz/Foxglove, unlike a packed blob — plus a small JSON sidecar (`/sam3/detections`, same pattern as
`/obj_map_json`): `{stamp: {sec, nanosec}, entries: [{id, label, confidence, bbox}]}`. `map_node`
reconstructs each mask via `id_map == encode_instance_id(id)`, reproducing the exact 5-key
`to_detections()` dict — `ObjMapper.update_map()` needed no changes. The two messages are matched by
stamp — `/sam3/instance_map`'s real `header.stamp` against the stamp *embedded in the JSON*, since
`std_msgs/String` has no header of its own to compare against. **Bug, found and fixed 2026-07-28:**
the first version compared `header.stamp` on both sides, which threw `AttributeError` on `String`
every time and silently killed `map_node`'s worker thread on the very first successful pairing (ROS
callbacks and the heartbeat kept running, so the node looked like it was "waiting on sync" forever
rather than visibly crashed) — embedding the stamp in the JSON is the actual fix.

**State split.** Bag-loop handling (the three things listed above) now lives in two places: `sam_node`
owns the SAM3 session reset + id-offset (its own state, gone from `map_node` entirely — ids arrive
already non-colliding), and `map_node` owns only its odom/cloud ring-buffer reset (unchanged logic,
just no id bookkeeping anymore).

---

## 4. SAM 3 / SAM 3.1 reference

### 4.1 What it is

**Promptable Concept Segmentation (PCS):** given short noun phrases ("yellow school bus"), image
exemplars, or both, return segmentation masks *and unique identities* for every matching instance.
Architecturally: an image-level detector plus a memory-based video tracker sharing one backbone, with
a **presence head** that decouples recognition from localization.

Two model families in transformers:

| Class | Use |
|---|---|
| `Sam3Model` / `Sam3Processor` | single images; text, box, and exemplar prompts |
| `Sam3VideoModel` / `Sam3VideoProcessor` | video; persistent object ids — **what we use** |

**Streaming vs. pre-loaded.** Pre-loaded (`propagate_in_video_iterator`) runs "hotstart" heuristics
that remove unmatched and duplicate tracklets — but those need *future* frames, so streaming disables
them. Streaming therefore yields more false positives and duplicate tracks. We accept that because
`ObjMapper`'s world-space merge (§2.6 step 8) absorbs duplicate ids into one 3D instance. If we ever
need best-possible quality on a bag, an offline pre-loaded mode is the answer (§6).

Key streaming call:
```python
session = processor.init_video_session(inference_device=dev, processing_device="cpu",
                                       video_storage_device="cpu", dtype=torch.bfloat16)
processor.add_text_prompt(session, all_prompts)     # ALL prompts, ONE session
out = model(inference_session=session, frame=inputs.pixel_values[0], reverse=False)
res = processor.postprocess_outputs(session, out, original_sizes=inputs.original_sizes)
```

**Two rules that matter most:**
- **One session, all prompts.** Vision features are reused across prompts, so N classes cost ~1
  forward pass, not N. Never run one session per class.
- **One session for the whole run.** Object ids are only stable within a session.

### 4.2 Optimization knobs (verified against transformers source)

| Knob | Status | Evidence |
|---|---|---|
| `attn_implementation="flash_attention_2"` | **supported** | `_supports_sdpa` / `_supports_flash_attn` / `_supports_flex_attn` / `_supports_attention_backend` all `True` — `modeling_sam3.py:776-779`, `modeling_sam3_video.py:500-503`. Propagates to geometry encoder, DETR encoder/decoder, mask decoder — `modeling_sam3.py:2181-2185`. Fall back to `sdpa`. |
| `dtype=torch.bfloat16` | standard | |
| non-square `image_size` | experiment | §4.3 |
| `max_vision_features_cache_size` | session param | on `init_video_session` |
| `torch.compile` | **not declared** | neither model sets `_can_compile_fullgraph`. Try it; don't count on it. |
| SAM 3.1 Object Multiplex | see §4.4 | shared-memory joint multi-object tracking, ~7× faster at 128 objects |
| flash-attn-3, `cc_torch` | native path only | optional deps of `facebookresearch/sam3` |

Streaming-quality knobs, all `Sam3VideoConfig` fields: `score_threshold_detection` (0.5),
`new_det_thresh` (0.7), `det_nms_thresh` (0.1), `assoc_iou_thresh` (0.1), `trk_assoc_iou_thresh`
(0.5), `recondition_every_nth_frame` (16), `init/max/min_trk_keep_alive` (30/30/−1),
`hotstart_delay` (15), `fill_hole_area` (16).

### 4.3 Input resolution — squash, pad, or non-square

`Sam3ImageProcessor` defaults (`image_processing_sam3.py:406-415`):
```python
size = {"height": 1008, "width": 1008}
do_resize = True
do_pad = None      # no padding
pad_size = None
```

So the default **squashes** 1920×640 into 1008×1008 — horizontally compressed ~1.9×, vertically
stretched ~1.6×, a 3× aspect distortion in what the model sees. Output geometry is unaffected:
`original_sizes` is captured at preprocess and used to map masks and boxes back to 1920×640.

**Padding is the wrong fix.** Pad to 1920×1920, resize to 1008², and real content occupies just
1008×336 — two-thirds of the token budget on grey bars, *and* worse vertical resolution than
squashing.

**Non-square looked like the right fix. It does not work — measured 2026-07-27.**
`Sam3ViTConfig.image_size` is typed `int | list[int] | tuple[int, int]`, which suggested 3:1 input
was a designed-for case. Both attempts died identically:

```
[672, 2016] -> RuntimeError: shape '[1, 72, 72, -1]' is invalid for input of size 7077888
[336, 1008] -> RuntimeError: shape '[1, 72, 72, -1]' is invalid for input of size 1769472
```

Read the numbers: `7077888 = 48×144×1024`, so the ViT *did* produce the correct non-square token
grid. But the reshape target is `72×72` — the **square** 1008/14 grid. A reshape downstream of the
ViT hardcodes a square token grid regardless of `image_size` or `backbone_feature_sizes`. (Setting
`backbone_feature_sizes` remains necessary; it is just not sufficient.) Fixing this means patching
transformers modeling code — not worth it.

**Consequence: the panorama stays squashed 3× into a square, and the only resolution lever is
_which_ square.** Valid sizes are doubly constrained — divisible by `patch_size` 14, *and* the
resulting grid divisible by `window_size` 24 — leaving grid ∈ {24, 48, 72}:

| `image_size` | grid | tokens | vs 1008² |
|---|---|---|---|
| `[1008, 1008]` | 72×72 | 5184 | 1.00× (default) |
| `[672, 672]` | 48×48 | 2304 | 0.44× |
| `[336, 336]` | 24×24 | 576 | 0.11× |

Aspect distortion is therefore permanent for v1, which promotes the tiling / perspective-reprojection
work in §6 from "nice to have" to the only route to undistorted input.

### 4.4 Version landscape

| Model | Status here |
|---|---|
| `facebook/sam3` | Full transformers integration (`Sam3Model`, `Sam3VideoModel`). **Ships first.** |
| `facebook/sam3.1` | Released 2026-03-27. Adds **Object Multiplex** — objects packed into buckets of `multiplex_count` (16) and tracked jointly, ~7× at 128 objects. **Loads and runs (2026-08-11)** via the NATIVE `facebookresearch/sam3` package; no conversion. `transformers` has none of it. See §4.5. |
| SAM3-LiteText | **Dead end here — do not re-investigate.** Distills the 353M text encoder to MobileCLIP (42.5M, −88%). Disqualified twice over: there is no `sam3_lite_text_video` module (image only, no tracking), and our prompts are fixed per session so the text encoder runs *once* — the saving is static memory, not per-frame latency. |

### 4.5 SAM 3.1 — measured, 2026-08-11

**This section previously said "the gap is a file format, not a missing integration" and that a
checkpoint conversion was the cheap path. That was wrong and cost a day.** `transformers` contains
**no Object Multiplex at all** (`grep -ril multiplex` over `transformers/models/` returns nothing),
so converting `sam3.1_multiplex.pt` into `Sam3VideoModel` yields 3.1's *weights* running SAM 3's
*algorithm* — the speedup is in the code. The conversion tooling has been deleted.

**How it is actually used.** `pip install --no-deps git+.../sam3.git@<sha>` (pinned in
`ai_module/docker/Dockerfile`) plus `einops`, `iopath`, `pycocotools`. `--no-deps` is load-bearing:
sam3 pins `ftfy==6.1.1`, which would downgrade the one `open_clip` needs, and an unconstrained
resolve could move torch or numpy (the 1.26 ABI is what `cv_bridge`/`rclpy` are built against).
`einops` is *not* in sam3's `pyproject` but is imported at `sam3/sam/rope.py:15`. `hydra` and
`skimage` are reachable only from paths we do not use. `build_sam3_multiplex_video_predictor` loads
`sam3.1_multiplex.pt` directly via `just hf-fetch sam3.1`.

**Findings, all verified against source at `96914d2` and measured on 40 `livingroom_1` frames (L4):**

| | finding |
|---|---|
| **Streaming** | Possible. `_run_single_frame_inference(state, frame_idx, reverse)` is a per-frame primitive with all memory in `inference_state`; `assert img_ids.numel() == 1`; `feature_cache.pop(frame_idx-1)` self-evicts. Only the session bootstrap wants a whole video, and `resource_path` accepts a list of PIL Images (`io_utils.py:44`) — no disk. |
| **Multi-concept** | Works, but not through `add_prompt`, which holds ONE caption and calls `reset_state()` on entry (a second call replaces the first). `find_text_batch` is an arbitrary caption list and any slot is selectable: `text_ids` and `img_ids` are **parallel batches** (`sam3_image.py:180-184`, *"the batch size of txt_feats is always the number of prompts"*). `img_ids=[t]*N` with `text_ids=[0..N-1]` grounds N concepts against one image, and `_get_img_feats` dedupes img_ids (`:139-143`) so it costs **one backbone pass**. Measured **3.1× faster** than N separate passes, with identical per-concept detections. |
| **Upstream's own multi-prompt** | `Sam3MultiplexTracking.forward()` loops `add_prompt` → full `propagate_in_video` → `reset_state` per concept, and `reset_state` does `feature_cache.clear()`. Measured: prompts 2–5 cost the same as prompt 1 — every frame re-encoded. 1788 ms/frame for 5 concepts. **This is the slow way; do not copy it.** |
| **Attribution** | Exists — `scores_labels[obj_id + start_obj_id] = (score, prompt_id)`, the `prompt_to_obj_ids` equivalent `detections.py` needs. Nothing to build. |
| **Tracker** | Nearly flat in object count: **`50 + 1.28·N` ms** against SAM 3's `37 + 20.4·N`. Object Multiplex pays off even at 43.75% bucket fill; it is concept-agnostic (it tracks masklets), so the fit should hold for a merged multi-concept object set. |
| **Fixed cost** | Higher than SAM 3: ~330 ms `detect(+backbone)` for one concept, ~447 ms for six batched, vs SAM 3's 175 ms encoder + ~30/prompt. |
| **`use_fa3`** | Builder default is **True** and imports `flash_attn_interface` with `float8_e4m3fn` — Hopper only. Must be `False` on Ada; falls back cleanly to SDPA (`model_misc.py:397-410`). |
| **Memory** | `batched_grounding_batch_size=16` and `postprocess_batch_size=16` are hardcoded (`model_builder.py:1190-1191`, exposed by neither) and OOM a 22 GB L4 at 640×1920. Plain attributes; set to 4. Both are offline lookahead and worthless in streaming. |
| **Upstream bug** | `Sam3BasePredictor.start_session` always passes `offload_state_to_cpu`, which no `init_state` in the multiplex chain accepts. Filter kwargs by the real signature — `offload_video_to_cpu` IS supported and must keep working. |
| **State dict** | The builder loads twice (`model_builder.py:1120` then `:1222`): first into the bare tracker BEFORE the `sam3_model.`→`detector.` remap (misses ~900 keys, means nothing), then into the assembled model. Only the second is authoritative — 64 missing, all benign `freqs_cis` register-buffers recomputed in `__init__` (`vitdet.py:552-555`), 0 unexpected. |
| **Lost in streaming** | hotstart(15), masklet confirmation, batched grounding — all need future frames. SAM 3 streaming already runs without hotstart, so this is parity, not a regression. It does mean 3.1's published benchmark numbers are offline numbers. |

**Where it stands.** For 1 concept SAM 3.1 already beats SAM 3 above ~6 objects (374 vs 449 ms
measured). For 3–6 concepts it should win by a widening margin — 6 concepts found 21.2 objects/frame,
where SAM 3 pays 20.4 ms each and SAM 3.1 pays 1.28 — but that requires the one thing still unbuilt:
**wiring batched multi-concept detection into the per-frame tracker.** Detection is proven
(`pred_logits (N,200,1)`, reproducing every per-concept baseline exactly); the tracker path still
computes image features at batch 1 against a batch-N stage. The merge and attribution it would feed
already exist.

Benchmark it with `just sam31-probe` (`scripts/eval/sam31_probe.py`). **Run it on a cool card** — the
L4 throttles from 2040 to ~1150 MHz within one sweep, which is enough to make per-concept times track
run order instead of object count; the probe now refuses to report a tracker fit when clocks move
more than 5% across a run.

**SAM3-LiteText remains a dead end** — see §4.4.

## 5. Tuning guide — symptom → knob

> Numbers here are defaults, not measurements. Milestone 3 replaces them with real ones.

| Symptom | First knob | Where |
|---|---|---|
| Too many false objects | `score_threshold_detection` ↑, `new_det_thresh` ↑ | sam3 config |
| One object appears as several 3D instances | check for panorama-seam split first (§6); then `ObjMapper` merge `dist_thresh` | `semantic_map.py:434` |
| Objects missed entirely | reword the prompt; `score_threshold_detection` ↓; try `[672,2016]` | config / §4.3 |
| 3D points bleed onto background | mask erosion iterations (currently 5) | `semantic_map.py:269` |
| Objects at wrong 3D position | wrong `platform` extrinsics; `detection_*_time_bias`; odom interpolation | §2.5 |
| Object drifts / smears across the map | `recondition_every_nth_frame` ↓ | sam3 config |
| Too slow, GPU util low | `runtime.executor_threads` (GIL starvation — §3.6); fewer prompts | §3.6 |
| Too slow, GPU util high | `image_size` ↓; confirm `attn_implementation` took effect; SAM 3.1 | §4.2 |
| Inference time grows with object count | expected — per-object memory attention; cut prompts | §3.6 |
| Track ids churn | `max_trk_keep_alive` ↑, `min_trk_keep_alive` | sam3 config |
| Sparse/hollow 3D objects | `percentile_thresh` ↓, `num_angle_bin` | `semantic_map.py:142-143` |
| Objects vanish from the map | `inactive_frame > 20` skip, and the `<20 voxel && inactive_frame > 5` prune | `semantic_map.py:382`, `:393` |

---

## 6. Improvement backlog

1. **Panorama seam.** An object spanning `u=0`/`u=1920` splits into two detections with two ids; SAM 3
   has no wrap-around awareness. The world-space merge usually fuses the fragments, but not always.
   Fixes: overlapping equirect tiles (cheap coordinate remap, but each tile is its own session so ids
   must be merged across tiles) or perspective reprojection (best detection accuracy, matches SAM 3's
   training distribution, but needs forward+inverse warps and cross-view association). **Either fix
   costs SAM 3's built-in cross-frame ids** — that is the real price, not the compute.
2. **Offline pre-loaded mode** for best-quality bag evaluation — drain the bag, run
   `propagate_in_video_iterator` with hotstart heuristics enabled, then replay the mapping stage.
   Gives a quality ceiling to measure the streaming path against.
3. **Re-enable the captioner** (`ai_module/src/captioner/`, not yet installed in this image) — CLIP
   features per instance for M4 queries. Kept dormant in `sam_mapper/object_mapper.py`'s `ObjMapper`:
   the `captioner` param and every `self.captioner is not None` branch in `update_map` are ported and
   ready, `map_node.py` just passes `captioner=None` for now.
4. **The disabled IoU split/merge path** (`sam_mapper/object_mapper.py`, in `update_map`'s world-space
   merge loop — commented out, needs `pytorch3d.ops.box3d_overlap` + `get_corners_from_box3d_torch`,
   neither installed) — would split objects that were wrongly merged, using `AdjacencyGraph`
   co-visibility (also kept dormant in the same file) to tell "two close real objects" from "one
   object split across ids" — the duplicate-instance-rate metric M6 scores directly.
5. **Oriented boxes.** `infer_bbox_oriented` exists (`sam_mapper/single_object.py`) but `update_map`
   uses the axis-aligned `infer_bbox` for `serialize_map_to_dict`/`to_ros2_msgs`. Oriented boxes
   returns `(None, None, None)` on ConvexHull failure — handle that before switching, and note it
   changes `/obj_boxes`/`/obj_map_json` shape for M3/M4 consumers.
6. **Colour/size attributes** per instance for M4 (§Metrics) — not yet extracted.
7. **Fix the `semantic_mapper` quirks** in §2.9 if we ever upstream changes.
8. **Make `semantic_mapping`'s heavy imports lazy**, for `semantic_mapper`'s own sake (no longer
   relevant to `sam_mapper`, which dropped the dependency entirely on 2026-07-28 — §3.1). `spacy.load()`
   at `utils.py:59-60` and `from bytetrack... import BYTETracker` at `semantic_map.py:11` still run at
   module scope (§2.8) for anyone running `semantic_mapping.mapping_ros2_node` directly. Moving both
   behind function-level imports would drop a ~500 MB model download for anyone who does not need
   `language_planner`. Best fixed upstream.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **odom / odometry** | The robot's pose over time. `/state_estimation` is `nav_msgs/Odometry`: position (xyz), orientation (quaternion), and linear/angular velocity in the world frame, ~50 Hz. Needed to know exactly where the camera was when a given image was taken. Since odom (50 Hz) and images (10 Hz) never share timestamps, the pose is interpolated — linear for position, **SLERP** for rotation — between the two odom samples bracketing the image stamp. |
| **SLERP** | Spherical linear interpolation — the correct way to interpolate between two rotations; naive component-wise averaging of quaternions is wrong. |
| **equirectangular** | The 360° panorama projection: longitude → x, latitude → y. Constant radians-per-pixel on both axes, which is why `scan2pixels` scales both by `W/(2π)`. |
| **ring buffer** | The bounded list of recent odom / lidar messages kept so a frame can find its bracketing samples when it is finally processed. |
| **meta-class** | The canonical class name (`chair`) that many free-form prompts (`"chair"`, `"office chair"`) collapse onto. Needed by Grounding DINO; **obsolete under SAM 3**, which reports the originating prompt directly. |
| **instance vs. background** | Instance classes (chair, table) get tracked ids `>= 0` and become `SingleObject`s. Background classes (wall, floor, ceiling) get negative ids and are re-created each frame — they have no persistent identity. |
| **voxel voting** | Each voxel records which viewing directions it has been observed from. Confidence = **observation-angle diversity** (how many distinct bins), so points seen from one angle only — typically projection artefacts — are filtered out. |
| **PCS** | Promptable Concept Segmentation — SAM 3's task: given a noun phrase, return masks *and identities* for every matching instance. |
| **tracklet** | One tracked object's state over time. In ByteTrack, a Kalman filter state; in SAM 3, a memory-conditioned masklet. |
| **hotstart** | SAM 3's start-up heuristics that drop unmatched and duplicate tracklets. Needs future frames, so it is **disabled in streaming mode**. |
| **Object Multiplex** | SAM 3.1's shared-memory approach to joint multi-object tracking; ~7× faster at 128 objects with no accuracy loss. |
| **presence head** | SAM 3's decoupling of "is this concept present at all" from "where is it", which improves detection accuracy. |
