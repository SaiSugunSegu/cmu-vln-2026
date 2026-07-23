# M2 — Perception (open-vocab 2D → 3D instances)

**Task:** Convert 360 images + lidar into a persistent 3D object instance memory. Concretely: equirect→pinhole crop remapping (precomputed LUTs); open-vocab segmentation with two prompt sets (general VLA-3D vocab + question-conditioned phrases); sync frames with accumulated lidar cloud; project masked pixels → depth-cluster → fit 3D boxes; associate detections across frames (merge, confidence, best-crop storage); extract attributes (color name, size).

**Status:** ⚪ not started
**Owner:**
**Depends on:** M0 (bags), M1 (viewpoints)

## Interfaces
- In: `/camera/image` (10 Hz, 1920×640 equirect, 360°×120°), `/registered_scan`, `/state_estimation`, camera↔lidar extrinsics (repo), prompt sets from M4
- Out: `Instance {id, label, bbox3d, color, size, conf, siglip_feat, obs_count, best_crop, last_seen}` stream → M3

## Plan
| Stage | What to try |
|---|---|
| Baseline | **SAM 3** (text-prompt concept segmentation: detect+segment+track in one model), 6×75° pinhole crops w/ 15° overlap, 1–2 Hz processing; lidar mask-projection + DBSCAN depth clustering → 3D boxes |
| Upgrade | If latency-bound on NUC-class HW: **YOLOE** always-on + SAM 3 only on question-relevant prompts; **SigLIP 2** crop embeddings for re-ID/merging; HSV-median color naming; size from box dims |
| Stretch | **SAM 3D** for lidar-sparse/thin objects; fine-tune on Unity renders if synthetic domain gap appears |

## W1 bake-off (do before building the full pipeline)
Run on Unity renders from 3–4 training scenes, score vs VLA-3D object lists:
| Candidate | Recall | Precision | ms/frame (L4 box) | Notes |
|---|---|---|---|---|
| SAM 3 | | | | |
| YOLOE (+SAM 3 masks) | | | | |
| DINO-X (accuracy ref) | | | | |

(Eval GPU is unknown NUC-class — leave headroom vs L4 numbers.)

## Metrics (via M6)
- Per-class instance recall/precision vs VLA-3D GT (target ≥80% recall)
- Duplicate instance rate (target <10%) — the counting killer
- Bbox IoU vs GT; attribute (color) accuracy
- End-to-end perception latency per frame

## Design notes
- Dedupe crop-overlap detections in 3D (same location), not 2D — simpler and robust.
- Bleed-through rejection: inside-mask points clustered by depth; keep nearest cluster consistent with box size.
- Identical furniture (6 same chairs) is the association stress test — gate by 3D distance first, features second.
- Defer box finalization until obs_count ≥ 2 when time allows; ghosts (1 obs, low conf) culled.
- Color words: use VLA-3D's exact 15-color LAB/CSS3 mapping (`3d_data_preprocess/utils/dominant_colors_new_lab.py`) — the questions' color vocabulary comes from it; sample interior 70% of mask. Note VLA-3D was extended/filtered as IRef-VLA — download data from there.

## Progress checklist
- [ ] Equirect→pinhole remap LUTs; visual sanity check
- [ ] Bake-off complete; model chosen (record in README decision log)
- [ ] Mask → lidar projection working on bag data
- [ ] 3D boxes visualized in RViz/Foxglove overlay
- [ ] Instance memory w/ association + merging
- [ ] Duplicate rate measured on identical-furniture scene
- [ ] Attributes (color, size) extracted and validated
- [ ] Latency measured on L4 box; within budget at 1–2 Hz with headroom

## Suggestions
- **Read the organizer's panorama+lidar→RGB-D bridge first**: [Navigation-Physical-Experiment](https://github.com/Yuxin916/Navigation-Physical-Experiment) converts pano images + lidar into registered RGB-D (Habitat-style) — a working reference for our image↔lidar projection plumbing; clone and study `src/` before writing our own. Dev tip: `without_360_camera` env variant renders faster for non-perception work.
- **Instrument with Rerun from line one:** `rr.log()` crops, masks, projected points, 3D boxes, instance merges at every pipeline stage. Time-scrubbing replays make association bugs (double-counting) visible in seconds; `.rrd` recordings are shareable with teammates who have no ROS installed.
- SAM 3's built-in cross-frame tracking may replace most association logic — test tracking IDs across viewpoint jumps before writing a custom tracker.
- Keep a per-instance "best crop" (largest, most frontal) — M4 uses it for VLM verification of borderline attributes.
- If SAM 3 is too slow even on x86: SAM 3 distilled/edge variants exist (see Edge AI ports) — check before falling back to YOLOE.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created |
