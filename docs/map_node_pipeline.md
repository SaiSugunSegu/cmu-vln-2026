# `map_node` — pipeline reference

Every processing and filtering unit in the 2D→3D mapper: what goes in, what it does, and
every threshold it applies. Written to be argued with — if a stage looks wrong, it
probably is; several were.

Companion: [`map3d_bench.md`](map3d_bench.md) — how to measure a change.

Code: `ai_module/src/sam_mapper/sam_mapper/` — `map_node.py`, `frame_sync.py`,
`cloud_image_fusion.py`, `object_mapper.py`, `single_object.py`, `mapping_config.py`,
`box_geometry.py`.

---

## Tuning without touching code

Every stage below has an on/off switch and its thresholds in the `mapping:` block of the
node yaml (`config/sam3_mecanum_sim.yaml`, `..._real.yaml`). Defaults live in
`mapping_config.py`; the yaml carries overrides. An unknown key raises rather than being
ignored, because "that knob did nothing" and "I typo'd the knob" are otherwise the same
observation.

`just map3d-replay` loads the same file the node does, so an A/B is:

```bash
$EDITOR ai_module/src/sam_mapper/config/sam3_mecanum_sim.yaml
just map3d-replay all && just map3d-score            # ~2 min, 13 scenes
```

| stage | yaml key | current |
|---|---|---|
| B1 confidence | `confidence_threshold` | 0.30 |
| B2 erosion | `erosion.{enabled,fraction,min_iterations,max_iterations}` | **off** |
| B3 range | `range_filter.{enabled,max_distance}` | on, **5.0 m** |
| B5 bounds | `bounds_mode` | `clip` |
| B7 range gap | `range_gap.{enabled,min_points,min_gap,max_depth_scale}` | **on**, 6 / 0.35 m / 0.0 |
| B8 min points | `min_points_per_detection` | 5 |
| B9 voxel | `voxel_downsample.{enabled,voxel_size}` | on, 0.05 m |
| B10 outlier | `outlier_removal.{enabled,nb_neighbors,std_ratio}` | on, 20 / 1.0 |
| C4 angle bins | `num_angle_bins` | 20 |
| D1 DBSCAN | `dbscan.{eps_voxels,min_points}` | 3.0 / 3 |
| D2 cluster weight | `cluster_weight_min` | 10.0 |
| D3 priors | `dimension_priors.{enabled,priors}` | on, **264 derived classes** |
| D4/D5 percentile | `percentile_threshold` | 0.8 |
| D6 prune | `prune.{enabled,min_valid_voxels,min_inactive_frames}` | on, 20 / 5 |
| D8 world merge | `world_merge.{enabled,extent_scale,absolute_distance,block_covisible,covisible_overlap}` | on, 0.5 / 0.5 m / **true** / **0.7** |
| E oriented box | `publish_oriented_box` | **off** |

---

## The shape of the problem

SAM 3 gives 2D masks with track ids. Lidar gives 3D points. The mapper's whole job is to
decide which points belong to which mask, accumulate that across frames, and fit a box.
Everything below is either that decision or a filter protecting it.

One idea explains most historical failures: nearly every gate was an **absolute threshold**
applied across a 35× range of object volumes (a 0.03 m³ pillow to a 4.9 m³ sofa). A sofa
clears every gate; a pillow failed several in sequence and vanished with no error.

---

## A · Frame assembly — `map_node.py` + `frame_sync.py`

**In:** `/sam3/instance_map` (mono16) · `/sam3/detections` (JSON) · `/registered_scan` ·
`/state_estimation`. **Out:** one frame — `{masks, labels, ids, confidences}` + interpolated
pose + a merged cloud.

| # | Unit | Rule |
|---|---|---|
| A1 | **Topic pairing** | Exact `(sec, nanosec)`. `std_msgs/String` has no header, so `sam_node` embeds a stamp in the payload. A mismatch means the pair hasn't landed — wait rather than process a torn pair |
| A2 | **Odom interpolation** | Linear for position, SLERP for orientation, at the image stamp. Frame older than the buffer, or odom not caught up → skip frame |
| A3 | **Cloud gather** | Window `[stamp − 0.5 s, stamp + 0.1 s]`; buffer holds 40 scans. Asymmetric on purpose: lidar gathered *before* the image is already-observed geometry, lidar after it mostly is not |
| A4 | **Mask reconstruction** | `id_map == encode_instance_id(id)`. An id in the JSON but not painted in the image yields a silently empty mask |

> `frame_sync.py` exists so the offline bench runs this exact arithmetic rather than a
> reimplementation. Don't fork it.

---

## B · Point→mask assignment — `cloud_image_fusion.py` + `object_mapper.update_map`

**In:** frame from A. **Out:** one world-frame point cloud per surviving detection.
Listed in **execution order**, which is what the stage numbers mean.

| # | Unit | Rule |
|---|---|---|
| B1 | **Confidence filter** | `confidence ≥ 0.30`. Dead in sim — `sam_node` already gates at 0.7 |
| B2 | **Mask erosion** | **OFF.** When on: `clip(round(0.10 × mask_short_side), 1, 5)` iterations of 3×3 |
| B3 | **Range filter** | `‖point − robot‖ < 5.0 m`, applied to the whole cloud **before** the projection |
| B4 | **Equirect projection** | `u = W/2π·atan2(x,z) + W/2 + 1`, `v = W/2π·atan(y/horiDis) + H/2 + 1`; 305.58 px/rad; VFOV 120° |
| B5 | **Bounds handling** | `clip` (default) pins out-of-FOV points to row 0/639 where an edge mask swallows them; `reject` drops them |
| B6 | **Mask lookup** | `cloud[mask[v,u]]`. **No z-buffer** — points behind an object are claimed by its mask |
| B7 | **Range-gap cut** | **ON.** Largest depth gap ≥ 0.35 m, keep the near side; skipped below 6 points |
| B8 | **Min-points gate** | `< 5` points → detection discarded |
| B9 | **Voxel downsample** | `voxel_down_sample(0.05)`, **on creation only** — merges use the raw cloud |
| B10 | **Outlier removal** | `remove_statistical_outlier(nb=20, std=1.0)`, on creation only |

> **B3 runs before the projection and that is free.** `‖cloud_body‖ ≡ ‖cloud_world − t_b2w‖`
> since rotation preserves norm, so it selects exactly the points that filtering after B6
> would have kept — while shrinking the cloud ahead of `generate_seg_cloud`, which runs one
> pass over it **per mask**. Pinned by
> `test_projection.py::test_range_filter_commutes_with_projection`.

> **B3 and B7 are both depth filters answering different questions.** B3 is *too far from
> the robot*; B7 is *too far behind its own object*. B3 cannot do B7's job — a sofa at 2 m
> and the wall 5 m behind it are both in range.

> **B5 `clip` is a defect, not a design.** `_scan2pixels` clips `u`/`v` into range, which
> makes the `in_bounds` guard in `generate_seg_cloud` unreachable. With the sensor 0.75 m up
> under a 2.78 m ceiling, every ceiling return within 1.17 m horizontal radius lands on row
> 0 — so although the global impact is 0.1% of points, it concentrates exactly where tall
> objects live. `reject` is implemented and off until measured.

> **B2 is off, and that is measured.** Thin masks were losing most of themselves to it
> (`book` kept 38% of its pixels at *one* iteration, `easel` 25%) because the rule reads the
> mask's bounding-box short side, not its local thickness. Turning it off gained
> recall@0.50 0.138 → 0.186. The centroid accuracy it was buying is now bought by B7, which
> filters on depth rather than geometry and costs no pixels.

---

## C · Association — `object_mapper.update_map`

| # | Unit | Rule |
|---|---|---|
| C1 | **Track-id lookup** | Exact id match only — no geometric or semantic fallback |
| C2 | **Merge** | Adds the **raw** cloud to the existing object |
| C3 | **Create** | New `SingleObject` from the **downsampled + outlier-filtered** cloud |
| C4 | **Voxel accumulation** | cKDTree; a point within `voxel_size` of an existing voxel votes on it, else becomes a new voxel. 20 azimuth bins |

> **C2/C3 are asymmetric and it looks like a bug.** It is — but both obvious fixes measured
> **4× worse** (bestIoU 0.031 → 0.007). Voxel downsampling alone starves the accumulation,
> because every downstream threshold is implicitly tuned to raw-merge point counts. It
> cannot be fixed in isolation.

---

## D · Per-object geometry — `single_object.py`

Runs for every object every frame, gated on `5 < life < 1000`.

| # | Unit | Rule |
|---|---|---|
| D1 | **DBSCAN** | `eps = 3 × voxel = 0.15 m`, `min_points = 3`. Rejects strays attached to one object — not instance separation |
| D2 | **Cluster weight** | `Σ observation_angles < 10` → dropped |
| D3 | **Prior acceptance** | Accepts clusters heaviest-first while the combined box still fits the class cap. 264 caps derived from VLA-3D GT |
| D4 | **Percentile stop** | Stops once accepted weight `> 0.8 × total` |
| D5 | **Diversity trim** | Drops the lowest-angle-diversity voxels holding the bottom 20% of weight |
| D6 | **Prune** | `valid voxels < 20` **and** `inactive_frame > 5` |
| D7 | **Centroid** | Diversity-weighted mean of surviving voxels; `None` if total weight is 0 |
| D8 | **World merge** | Same label, matching verticality, and `dist < ‖(ext_a/2 + ext_b/2)/2‖ × 0.5` or `dist < 0.5 m` — **unless** the pair is co-visible and overlaps < 0.7 |

> **D5 trims edges specifically.** An object's edges are seen from the fewest angles, so the
> lowest-diversity voxels *are* its boundary. This erodes every box inward and remains a
> suspect for residual under-sizing.

> **D8's co-visibility guard.** Two ids SAM 3 saw in the *same frame* are different objects
> at any distance — something distance cannot know, and why 0.44 m-spaced pillows fused. But
> SAM 3 also splits one object across two ids, so blocking on co-visibility alone took
> duplicate_rate 0.029 → 0.189. The guard adds spatial overlap (intersection-over-*minimum*,
> via `box_geometry`): co-visible **and** disjoint → distinct objects; co-visible **and**
> overlapping → one object fragmented, merge anyway.

> **D3 is usually the messenger, not the culprit.** A cluster losing 97–99% of its voxels to
> `exceeds_prior` is nearly always a bled multi-instance blob D8 produced. Checked against
> GT: each cap sits ~2× above the real object in that scene.

---

## E · Output — `serialize_map_to_dict` / `to_ros2_msgs`

| Field | Source |
|---|---|
| `center` | diversity-weighted voxel centroid (D7) |
| `bbox3d.center` / `.extent` | min/max over surviving voxels **+ `voxel_size`** |
| `bbox3d.rotation` | identity — the oriented box is computed but not published |

Topics: `/obj_points` · `/obj_boxes` · `/obj_labels` · `/obj_map_json`. With
`runtime.save_obj_map`, the same map is written to `<run_dir>/obj_map.json` on every publish
(atomically, so a kill mid-write leaves the previous complete map).

> **`center` and `bbox3d.center` are different points** and disagree whenever voxels are
> unevenly distributed — i.e. always, for a one-sided view. `map3d-score` reports both.

> **`+ voxel_size` matters**: a voxel is a *cell*, not a point. Min/max over centres
> under-reported every extent by 0.05 m and collapsed one-voxel-thick objects (windows) to
> exactly zero volume.

> **Never answer category 2 with the wireframe.** `create_wireframe_marker` leaves `pose` at
> identity and uses `scale.x` as a line width, so a pose+scale reader sees a 5 cm × 0 × 0
> box at the origin — a structural 0.0. Use `ros_markers.create_selected_object_marker`.

---

## Next

Ordered by expected value. Baseline and how to measure: [`map3d_bench.md`](map3d_bench.md).

| # | Item | Why |
|---|---|---|
| 1 | **Publish `/selected_object_marker`** | Nothing does. Category 2 has no answer path at all — `TODO(M4)` at `smart_vlm.py:297`. The Marker builder already exists |
| 2 | **`min_points_per_detection` 5 → 1** (B8) | Discards 8% of all detections, concentrated on the starving classes (stool +11%, tvcabinet +15%). B7 now removes the noise that made a low threshold risky |
| 3 | **Zero-point loss** | ~50% of detections get no lidar. Size-structured: floor-level classes starve (stool 82%, pillow 73%), tall and wall-mounted ones do not (column 5%). Consistent with a lidar blind cone below the sensor — likely not fixable here. `just map3d-zeropoints` |
| 4 | **`percentile_threshold` 0.8 → 1.0** (D5) | One key, and D5 trims exactly the edges that leave boxes under-sized |
| 5 | **`bounds_mode: reject`** (B5) | Correctness fix, small and unmeasured |
| 6 | **Class size priors as *anchors*** (D3) | Caps only reject; nothing supplies the unseen side of a one-sided object. The amodal residual |
| 7 | **Support-plane snapping** | Snap an object's base to the surface it rests on. The questions are saturated with "on the …" |
| 8 | **Unify C2/C3 accumulation** | Needs the downstream thresholds retuned together, or a global voxel hash where the distinction stops existing |
| 9 | **Superpoint voting** (Open3DIS / SAI3D / MaskClustering) | The SOTA answer: over-segment by geometry, then let masks vote for *superpoints*, which cannot straddle a depth discontinuity. Needs density we lack per frame, so it converges with 8 |

**Known limits, not bugs.** Thin wall-mounted objects (a photo 4 cm off its wall) share
voxels with the wall at 5 cm resolution; the cluster measures 3–5× oversize and D3 rejects
it. No threshold fixes that.

**Explicitly rejected.** Training a 3D detector (no labels, wrong sensor, open vocabulary) ·
Mask3D-style 3D-native segmentation (needs density we don't have) · a Kalman 3D-MOT layer
and `max_age` eviction (the scene is static and the robot moves — measured, cost 20 objects
→ 12) · replacing SAM 3 (verified sound: 0.93 confidence, 45-frame tracks) · publishing the
oriented box (loses on every metric, cat2 0.473 → 0.366).
