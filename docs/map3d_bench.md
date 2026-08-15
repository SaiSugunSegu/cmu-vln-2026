# M2 · 3D map benchmark (`map3d`)

Measures how good `map_node`'s 3D instance boxes actually are, against IRef-VLA ground
truth, deterministically and without a GPU.

Nothing in the repo measured 3D box accuracy before this, and every experiment had to
re-run SAM 3 on GPU — whose streaming tracker assigns different ids run to run, so two runs
of identical code produced different maps. Record once, replay deterministically on CPU.

---

## The loop

```bash
# 1. once per scene — the ONLY GPU step
just map3d-record all

# 2. every experiment — CPU only, ~30 s/scene, deterministic
just map3d-replay all 8

# 3. score against IRef-VLA ground truth
just map3d-score
```

Supporting:

```bash
just map3d-determinism arabic_room   # two replays must give the same map digest
just map3d-audit                     # which prompts/labels fail to resolve to a GT label
just map3d-zeropoints livingroom_1   # why masked detections receive no lidar
just test sam_mapper                 # 83 tests, 7 xfail (see "Defects as tests")
```

All of these run in the container. Prompt sets are **hand-curated data**, not generated:
`data/benchmark/bench_prompts.json` lists, per scene, the objects that scene's five
official questions actually name — target or anchor, **capped at 10 per scene** (129 across
13 scenes). Edit it directly when the questions change. wall/floor/ceiling are deliberately
absent: questions reference them only as spatial datums, and they are the most expensive
prompts SAM 3 can be given.

---

## How it fits together

```
data/bags/<scene>/<scene>_0.mcap          8 raw sensor topics (given)
              │
              │  map3d-record  ── SAM 3, GPU, once ──▶  <scene>_sam3/
              │                                          + <scene>_sam3.manifest.json
              ▼                                                │
        ┌───────────────────────────────────────────────────┐  │
        │  map3d-replay   (no GPU, no ROS graph, no DDS)    │◀─┘
        │  frame_sync ─▶ ObjMapper.update_map ─▶ obj map    │
        └───────────────────────────────────────────────────┘
              │  data/runs/map3d/<run-id>/<scene>.json
              ▼
        ┌───────────────────────────────────────────────────┐
        │  map3d-score   vs iref_vla_metadata/*_object_data │
        └───────────────────────────────────────────────────┘
              │  summary.json + a table on stdout
```

### Why an offline driver rather than `ros2 bag play` + a live `map_node`

The live node is latest-frame-wins and drops under load — correct for a robot on a clock,
useless for a benchmark. `replay_map3d.py` reads both bags and calls `ObjMapper.update_map`
directly, so **every** frame is processed, in order, with no DDS and no wall-clock
dependence. It also uses whole-bag odom/cloud arrays instead of ring buffers, so no frame
is skipped for "odom has not caught up".

It runs the **same** frame-assembly code the node does: that arithmetic lives in
`sam_mapper/frame_sync.py`, which `map_node` also calls. If the harness reimplemented it,
the bench would be measuring a different algorithm than the one that ships.

### VRAM: why the recorder restarts SAM 3 periodically

SAM 3's video tracker keeps a per-object memory bank that grows every frame, so VRAM
scales with **(tracked objects × frames)**. This recorder is a much heavier load than
`sam_node` ever sees in production — ~20 prompts instead of 2–5, and *every* frame instead
of latest-frame-wins. On a 22 GB card that produced ~70 detections/frame and OOM'd around
frame 25.

So the session restarts every `--session-frames` frames (default 20), shifting subsequent
ids past the high-water mark so ids from different sessions can never collide — the same
mechanism `sam_node` uses on a bag loop. A CUDA OOM is also caught: the frame is dropped,
VRAM reclaimed, the session restarted, and recording continues.

**This costs fidelity, and you need to know which way.** Each restart breaks track
continuity, so one physical object can appear under several ids across a scene. That
**inflates the duplicate-instance rate the bench reports** — it is a property of the
recording, not of the mapper. `session_restarts` and `oom_recoveries` are stamped into the
manifest so you can see how fragmented a given recording is.

It does not undermine the bench's main job: every A/B runs against the *same* companion
bags, so relative movement is still valid. Absolute duplicate-rate is pessimistic. Raise
`--session-frames` if your card has headroom; a longer session is strictly more faithful.

### Frame rate: why stride 5 is the default, not 1

Measured over the 13 scenes `bench_prompts.json` covers — the training scenes that have
both a bag and IRef-VLA metadata (3194 camera frames, 495 s, ~6.5 Hz). `office_building_1`
and `office_building_2` are excluded: they are the held-out test scenes.

| stride | frames | spacing | GPU @ ~11 s/frame |
|---|---|---|---|
| 1 | 3194 | 0.15 s | ~10 h |
| 2 | 1597 | 0.31 s | ~5 h |
| 5 (default) | 638 | 0.77 s | ~2 h |
| **10** | **319** | **1.55 s** | **~1 h** |
| production `sam_node` | ~248 | ~2.0 s | — |

⚠️ **Cost scales steeply with prompt count**, not with the 1.3–2.7 s/frame
`docs/M2_perception.md` quotes for a 2–5 prompt set. Measured on `arabic_room`:
2 prompts → 1.8 s/frame · 4 prompts → 6.5 s/frame · 19 prompts → 18 s/frame. At the
curated <=10 prompts/scene, budget ~10 s/frame. This is the most expensive step in the
whole workflow. **stride 5 is the default** (justfile and `record_companion.py` agree); it
is still 2.6x denser than production, and stride 10 halves the GPU cost again if needed.

`sam_node` runs at 1.3–2.7 s/frame against a 6.5 Hz camera, so in production the mapper
sees roughly **every 13th frame, 2 s apart**. Stride 5 is still 2.6× denser than anything
the real system achieves. **Stride 10 is 1.55 s spacing — still denser than production, at
half the cost**, and is the sensible choice if 3.2 h is too much; just keep it consistent
across scenes, since observation density affects every metric.

Recording at stride 1 would not be "more faithful" — it would benchmark the mapper in a
regime it never operates in, at 5× the cost. If you want to confirm the bench is not
stride-limited, re-record a single scene at stride 1 and compare; that is a ~3 min check,
not a reason to change the default.

Stride is stamped into each manifest, so a bench run can always be traced to the sampling
it was recorded at.

### The companion bag

`data/bags/` records only the 8 allowed sensor topics, so `/sam3/instance_map` and
`/sam3/detections` have to be regenerated. `map3d-record` drives SAM 3 frame-by-frame and
writes them into `<scene>_sam3/` with `log_time` taken from the source camera frame,
so the two bags share a timeline. A sidecar manifest stamps the prompt set, platform and
full `sam3:` config — a companion bag is only valid for the prompts it was recorded with,
and replay reads the platform back from the manifest so it cannot project with the wrong
extrinsics.

Prompt sets come from `data/benchmark/bench_prompts.json` — **hand-curated per scene**
from the objects that scene's five official questions name, target or anchor — **capped at
10 per scene** (129 across 13 scenes, mean 9.9). When cutting, everything the NUMERICAL and
OBJECT_REFERENCE questions name is kept first, since those are the question types this
bench scores; instruction-following anchors fill the rest. Per-scene rather than one global
union because SAM 3 cost grows steeply with prompt count.

---

## Reading the score

```
scene              pred   gt  P@.25  R@.25  R_obs   dup  bestIoU  claimed  near  cat2/2
arabic_room           3   56   0.00   0.00   0.00  0.00    0.005    0.041     1    0.02
```

**Read A/Bs against `bestIoU` and `claimed`, not `P@.25`/`R@.25`.** Precision and recall at
a fixed IoU are step functions. On the first real run *every* prediction scored 0.00 at
0.25 while the actual best IoUs were 0.068 and 0.216 — a fix moving 0.07 → 0.24 would have
read as "no change at all".

| Column | Meaning |
|---|---|
| `bestIoU` | mean best IoU per askable GT object, including ones nothing claimed — mixes in recall |
| `claimed` | same but only over GT some prediction claims — **isolates pure box quality** |
| `near` | GT objects sitting at IoU 0.1–0.25, i.e. nearly matched |
| `dup` | predictions whose best GT was already taken — the duplicate-instance rate |
| `R_obs` | recall over GT within 6 m of the robot path, so M1's coverage gaps don't score against M2 |
| `cat2/2` | the category-2 marker score, out of 2.0 — see below |

`gt` counts only **askable** objects: GT whose label is in the prompt set. Scoring against
all 85 objects when 16 classes were prompted would report a recall governed by the prompt
set, not by the mapper.

Also in `summary.json`: oriented-IoU variants, over-merge rate (measured separately —
greedy association can never reveal one box swallowing two GT objects), and centroid /
per-axis extent error split at 0.5 m max-dim, because averaging small and large objects
together hides the regime that matters.

### The category-2 marker score

The challenge scores object-reference by publishing a `visualization_msgs/Marker` on
`/selected_object_marker` and measuring overlap with the GT box. The bench therefore scores
**the actual Marker we would publish**, built by `sam_mapper/challenge_marker.py` — the same
module `ros_markers.create_selected_object_marker` builds the real message from, so the
bench cannot drift from what ships.

The organizer's overlap function is unpublished, so all plausible readings are reported:

```
category-2 marker score (oracle selection, out of 2.0):
  scorer honours orientation   0.020
  scorer ignores orientation   0.020
  if we published the wireframe 0.000   <- structural zero, not a low score
```

That third line is a permanent regression guard. `create_wireframe_marker` emits a
`LINE_LIST` with `scale.x = 0.05` (a *line width*) and never sets `pose`, so any reader
interpreting a Marker box as pose+scale — including this repo's own `qa_recorder.py` — sees
a 5 cm × 0 × 0 box at the world origin. **Answer with `CUBE`; keep the wireframe for RViz.**

Selection is **oracle** (best label-compatible prediction per target) because M4 does not
exist yet. So this is the ceiling the reasoner will work under, and it separates "our boxes
are bad" from "our reasoner picked the wrong object".

The target set is the category-2 benchmark's answer objects — up to 10 per scene, each the
single object a reference question must be pointed at (see
[docs/cat2_benchmark.md](cat2_benchmark.md)). Those answers are themselves gated on visibility:
an IRef box the robot's own camera never resolved is not asked about, so a miss here is a
mapping failure rather than a question about geometry the sensors never saw. Scenes without a
category-2 file fall back to the category-1 `object_ids`, which are *numerical*-question
objects: a proxy that over-counts, since a counting question touches many objects and singles
out none. Only `home_building_1/2` are on the fallback.

Targets are then intersected with the askable set, as everything here is, and **that is where
most of them go**: only 81 of the 122 category-2 answers have a label in their scene's prompt
set, because the sets are curated from the five official questions and capped at 10 while the
generated questions name whatever the scene contains (`books`, `ceiling lamp`, `towel rack`,
`tv remote`). Nothing prompted them, so SAM 3 never looked for them and they cannot be found
by construction. Widening the prompt sets is the fix; it costs SAM 3 time per
scene, which is why they are capped. Do not instead bias category-2 selection toward objects
we happen to prompt — that grades the bench against itself.

### Label mapping

Predicted labels are `default_label(prompt)` — the prompt lowercased with spaces stripped
(`potted plant` → `pottedplant`) — so the mapping back to GT `raw_label` is mechanical. The
only judgement is a short `SYNONYMS` table in `score_map3d.py`. **Keep it short**: every
wrong entry inflates recall. Extend it only from `just map3d-audit` output.

---

## Determinism

`just map3d-determinism <scene>` replays twice and requires an identical map digest. It
passes today (`80fb73d09089b748` on arabic_room). Keep it passing: `ObjMapper` uses Open3D
DBSCAN and dict iteration, either of which can order nondeterministically, and an A/B
across 13 scenes is noise otherwise.

---

## Defects as tests

`ai_module/src/sam_mapper/tests/` — 83 tests with 7 `xfail(strict=True)`. The xfails encode
behaviour the design calls for but the code lacks. Strict means they *fail the suite* if
someone fixes a defect without removing the marker, so the suite is a live checklist.

| File | Tests | Covers |
|---|---|---|
| `test_frame_sync.py` | 14 | odom interpolation (SLERP, not lerp), cloud windowing, mask reconstruction |
| `test_projection.py` | 18 | equirect geometry, B3/B4 commutation, bounds modes; **4 xfail** |
| `test_single_object.py` | 13 | box fitting, derived dimension priors; **1 xfail** |
| `test_object_mapper.py` | 19 | lifecycle, merging, B7 range-gap, D8 co-visibility; **2 xfail** |

Named `test_projection.py`, not `test_cloud_image_fusion.py`, because a file of that
basename exists in the vendored `semantic_mapper/tests/` and pytest aborts on same-named
modules in directories without `__init__.py`.

`test_single_object.py` and `test_object_mapper.py` need `open3d` and `importorskip` on the
host — run them with `just test sam_mapper`.

---

## Measurements so far

13 scenes, `just map3d-replay all && just map3d-score`. Only the current baseline is kept
here; per-change history lives in the config docstrings that own each threshold.

| | start of work | now |
|---|---|---|
| cat2 (oracle selection) | 0.473 | **0.637** |
| recall @0.25 / @0.50 | 0.419 / 0.138 | **0.531 / 0.293** |
| precision @0.25 | — | 0.426 |
| bestIoU per GT | 0.244 | **0.308** |
| duplicate / over-merge | — | 0.028 / **0.042** |
| centre err small / large | 0.069 / 0.105 m | **0.047 / 0.095 m** |
| targets with no candidate | 3/205 | 3/205 |

What moved it, in order of size: the **B7 range-gap cut**, the **D8 co-visibility guard**,
**derived dimension priors**, **erosion off**, and range 6 -> 5 m.

**cat2 is an ORACLE number** — it assumes M4 always picks the best candidate for a target.
`just map3d-score` also prints a naive-selection column; the gap between them is what
duplicates cost, and it is the honest figure for the finished pipeline.

### Still open

- **~50% of detections receive zero lidar points** and it is the largest remaining loss.
  Size-structured, not random: floor-level classes starve (arabic_room `stool` 82%, `table`
  75%, `pillow` 73%) while tall and wall-mounted ones do not (`column` 5%). Consistent with
  a lidar blind cone below the sensor rather than anything in this node. `just map3d-zeropoints`.
- **Thin wall-mounted objects are unmappable at 5 cm voxels.** A photo 4 cm off its wall
  shares voxels with the wall; the cluster measures 3-5x oversize and D3 rejects it. A
  resolution limit, not a bug.

---

## Gotchas

- **`docker/compose.yml` does not bind-mount `ai_module`** — only `compose_dev.yml`
  (`just up-dev`) does. Confirm which compose is running before trusting any container
  measurement; a frozen copy would silently invalidate all of it.
- **Companion bags are prompt-set specific.** Re-record after changing
  `bench_prompts.json`, or the manifest and the detections disagree.
- **`--force` overwrites** an existing companion bag; without it, recording refuses.
  `--skip-existing` instead resumes an interrupted `--scene all` sweep.
- **A `--scene all` run no longer dies on one bad scene** — failures are collected and
  reported at the end, with a non-zero exit.
- **The companion bag is a rosbag2 *directory*** (`<scene>_sam3/`), not a single file.
  The manifest is written only on success, so a crashed run never leaves a partial bag
  paired with a manifest describing a different one.
- **`data/benchmark/bench_prompts.json` is hand-curated** from questions/questions.json,
  and read-only in the container — edit it on the host.

## See also

**`docs/map_node_pipeline.md`** — every stage and threshold in the mapper, and the Tier
1/2/3 backlog · `docs/M2_perception.md` (the wider perception module) · `docs/cat1_bag_benchmark.md` (the category-1
loop, which bypasses `map_node`) · [`docs/cat2_benchmark.md`](cat2_benchmark.md) (where the
question-target set and the cat-2 answers come from) · `docs/M6_eval_harness.md`
