# Architecture & Proposals — CMU VLN 2026

Design doc: end-to-end baseline, per-component proposals (baseline → top candidate), and the weekly plan.
Research-backed; sources at the bottom. Companion trackers: `M0…M6` in this folder.

---

## 1. Design principles

1. **Walking skeleton first.** A complete, scoring pipeline by end of W2 — weak everywhere, broken nowhere. Every upgrade then replaces one box at a time, measured by M6.
2. **Answers come from a structured scene graph, not pixels.** Counting = counting graph nodes; grounding = symbolic filtering. This is exactly the zero-shot approach the organizers themselves published (SORT3D) and the direction 2025-26 grounding SOTA converged on (SceneGraphGrounder, View-on-Graph).
3. **Points-per-effort ordering.** Instruction-following = 6/9 pts per question mix → gets the most iteration. Numerical = 1 pt but binary → cheap wins from dedup quality.
4. **Every question must produce an answer** (T-30s fallback). Partial credit on object-ref/instruction is free.
5. **10-min clock is a feature:** early finish = tiebreaker bonus → early-stopping logic is a scoring component, not an optimization.

---

## 2. System overview

```mermaid
flowchart TB
  subgraph BASE["Base autonomy (provided — do not touch)"]
    SIM["Unity sim / robot"] --> SE["State estimation"]
    SE --> TA["Terrain analysis"]
    TA --> NAV["Waypoint nav + collision avoidance"]
  end

  subgraph AI["AI module (ours, ai_module/)"]
    subgraph PERC["M2 Perception"]
      REMAP["360° → pinhole crops"] --> DET["Open-vocab detect+segment"]
      DET --> LIFT["Lidar lifting → 3D boxes"]
      LIFT --> MEM["Instance memory (re-ID, merge)"]
    end
    subgraph GRAPH["M3 Scene graph"]
      MEM --> SG["Objects · attributes · relations · regions"]
    end
    subgraph REASON["M4 Reasoner"]
      QP["Question parser"] --> GATE["Decision gate"]
      SG --> GATE
      GATE -->|"ambiguous"| REOBS["Reobserve request"]
      GATE -->|"unknown area"| EXPL
      GATE -->|"confident"| ANS["Answer heads"]
    end
    subgraph ACT["M1 Exploration"]
      EXPL["Frontier scoring + viewpoint select"]
    end
    subgraph PLAN["M5 Instruction planner"]
      ANS -->|"instruction Q"| IF["Constraint-ordered waypoints + costmap"]
    end
  end

  Q["/challenge_question"] --> QP
  SIM -->|"/camera/image 10Hz"| REMAP
  SE -->|"/registered_scan 5Hz"| LIFT
  TA -->|"/terrain_map_ext 5Hz"| EXPL
  EXPL -->|"/way_point_with_heading"| NAV
  IF -->|"/way_point_with_heading"| NAV
  REOBS --> EXPL
  ANS -->|"numerical"| OUT1["/numerical_response Int32"]
  ANS -->|"object ref"| OUT2["/selected_object_marker Marker"]
```

### Runtime sequence (one question, 10-min budget)

```mermaid
sequenceDiagram
  participant E as Eval node
  participant R as Reasoner (M4)
  participant X as Explorer (M1)
  participant P as Perception (M2/M3)
  participant B as Base autonomy

  E->>R: /challenge_question (1 Hz)
  R->>R: parse → type + target spec (~5 s)
  loop until decision gate fires
    R->>X: explore_next() / reobserve(id)
    X->>B: waypoint
    B-->>P: images + registered scan (while driving)
    P->>P: detect → lift → merge instances
    P-->>R: scene graph update
    R->>R: gate: confident? covered? T-60s?
  end
  alt numerical
    R->>E: Int32 on /numerical_response
  else object reference
    R->>E: Marker on /selected_object_marker
  else instruction
    R->>B: ordered dense waypoints (M5), monitor + replan
  end
  Note over R: T-30s hard fallback: publish best guess no matter what
```

---

## 3. End-to-end baseline (walking skeleton — done by end of W2)

The simplest complete system that scores non-zero on all three question types. Everything local except two LLM calls.

| Stage | Baseline implementation | Est. effort |
|---|---|---|
| Question parsing | One API-LLM call → JSON `{type, targets[], attributes{}, relations[], order[]}` | 0.5 d |
| Exploration | Occupancy grid from `/terrain_map_ext`; nearest-frontier; stop at 90% coverage or 5 min | 2 d |
| Perception | YOLOE (prompted with VLA-3D vocab + question nouns) on 4×100° crops @ 1 Hz; lidar points inside box → median-depth cluster → AABB; merge by centroid dist < 0.5 m + same label | 3 d |
| Attributes | Median HSV of box interior → nearest color word; size = bbox volume | 0.5 d |
| Scene graph | Flat instance list JSON (no relation precompute — LLM gets raw centroids and reasons directly) | 0.5 d |
| Answering | One API-LLM call with instance JSON → count / pick instance / order constraint objects | 1 d |
| Instruction path | One waypoint at each ordered constraint object (offset 1 m toward free space), then goal; publish on arrival radius 0.5 m | 1 d |
| Fallbacks | T-30s: publish count=modal class count, marker=best label match, waypoint=goal object | 0.5 d |

Expected baseline scoring (against 75 training Qs): numerical ~50-60% (dedup errors), object-ref partial overlap on most, instruction partial credit from order-correct waypoints without avoid-zone handling. That's a mid-table 2025 result — from there, targeted upgrades.

---

## 4. Per-component proposals: baseline → top candidate

### M1 Exploration
| | Choice | Why / research |
|---|---|---|
| Baseline | Nearest-frontier on terrain map occupancy grid | Trivial, deterministic, good enough for single rooms |
| **Top candidate** | **VLFM-style language-scored frontiers**: score each frontier by SigLIP 2 similarity between its image direction and the question text; add lidar **doorway detection** for multi-room (LLM-MCoX technique); targeted `reobserve(id)` poses with line-of-sight check; confidence-based early stop | [VLFM](https://arxiv.org/pdf/2312.03275) (zero-shot semantic nav SOTA recipe), [LGR](https://arxiv.org/pdf/2503.20241) (LLM frontier ranking), [LLM-MCoX](https://arxiv.org/html/2509.26324v1) (lidar frontier + doorway extraction) |
| Fallback | Coverage-lawnmower over traversable area | If frontier extraction from terrain PointCloud2 proves noisy |

### M2 Perception
| | Choice | Why / research |
|---|---|---|
| Baseline | YOLOE (real-time open-vocab, prompt = vocab list) + box-interior lidar lifting | Fast, simple, runs anywhere ([YOLOE](https://learnopencv.com/yoloe-tutorial-real-time-open-vocabulary-detection/)) |
| **Top candidate** | **SAM 3** promptable concept segmentation: text-phrase → all instance masks + built-in tracking (one model replaces detect+segment+associate); mask-based lidar lifting w/ depth clustering; **SigLIP 2** crop embeddings for cross-view re-ID; VLM-on-best-crop for hard attributes | [SAM 3](https://github.com/facebookresearch/sam3) (2× prior SOTA on concept segmentation, 30 ms/frame on H200), [SigLIP 2](https://arxiv.org/abs/2502.14786) (better localization/dense features than CLIP/SigLIP) |
| Fallback | Grounding DINO 1.6 + MobileSAM | If SAM 3 too slow on NUC-class GPU (check distilled/edge variants first) |

### M3 Scene graph
| | Choice | Why / research |
|---|---|---|
| Baseline | Flat instance JSON, LLM reasons over raw centroids | Zero code, works for small scenes |
| **Top candidate** | **VLA-3D's 8 relation heuristics ported verbatim** (Above/Below/Closest/Farthest/Between/Near/In/On — the exact functions that generated the questions) + SORT3D toolbox for LLM sequencing; VLA-3D 15-color LAB mapping for attributes; validate by diffing our relations vs shipped `_scene_graph.json` | [SORT3D](https://arxiv.org/abs/2504.18684) — CMU's own zero-shot system; **SORT3D-Nav `humble-mecanum` branch runs on our same autonomy stack with a semantic mapping module + Qwen2.5-VL captioner (10 GB VRAM) — study/adapt before writing anything** ([repo](https://github.com/nzantout/SORT3D)); [VLA-3D format](https://github.com/HaochenZ11/VLA-3D) |
| Fallback | Subset of relations (near, closest) hand-written | Relations are unit-testable in isolation |

### M4 Reasoner
| | Choice | Why / research |
|---|---|---|
| Baseline | Single LLM call: question + instance JSON → answer | One prompt, surprisingly strong with good perception |
| **Top candidate** | **Tool-calling agent**: `query_graph` (SORT3D filters), `reobserve(id)`, `verify_crop(id, question)` (VLM check on stored crop), `answer()`; SceneGraphGrounder-style query-graph→subgraph matching for complex references; calibrated decision gate (confidence τ + margin δ + coverage); provider fallback + cached parses | [SceneGraphGrounder](https://arxiv.org/html/2605.21788), [View-on-Graph](https://arxiv.org/abs/2512.09215) (zero-shot 3DVG SOTA '26), [SPAZER](https://arxiv.org/pdf/2506.21924) (agentic spatial reasoning) |
| Fallback | Local Qwen-class LLM for parse + rules for counting | Removes network dependency for 2 of 3 question types |

### M5 Instruction planner
| | Choice | Why / research |
|---|---|---|
| Baseline | One waypoint per ordered constraint + goal | Earns order-based partial credit immediately |
| **Top candidate** | **Costmap + dense trajectory**: attraction bands near pass-constraints, inflated repulsion polygons for avoid-zones, waypoints every ~1 m so base planner can't shortcut through forbidden space; execution monitor (odometry vs plan) with replan-on-violation; pre-execution self-scoring against our own M6 trajectory scorer | [Beyond Waypoints](https://arxiv.org/pdf/2606.07244) (trajectory-centric waypointing), [NavOne](https://arxiv.org/pdf/2605.06317) (global top-down-map planning); scoring rules: order sacred, penalties for forbidden areas |
| Fallback | Denser sampling along straight-line homotopy avoiding zones | If costmap tuning runs out of time |

### M6 Eval harness (not optional — the compass)
| | Choice |
|---|---|
| Baseline | Runner over 75 training Qs; exact-match + IoU scorers; results table per run |
| **Top candidate** | + Own trajectory scorer replicating constraint-order/penalty logic vs provided `.ply` GT; per-module metrics (recall, dupes, grounding acc on GT objects vs live); 10-question smoke set for fast iteration; nightly full run |

---

## 5. Weekly plan

| Week | Theme | What to try | Exit criteria |
|---|---|---|---|
| **W1** Jul 21–27 | It runs | Register (Jul 25!). M0 done (✅ sim+RViz Jul 21). Bags for 3 scenes. M6 skeleton + dummy-model floor score. Frontier explorer v0. **SAM 3 vs YOLOE bake-off on Unity renders** | Skeleton explores a room <3 min; perception model picked with L4 latency numbers; harness scores the dummy model |
| **W2** Jul 28–Aug 3 | Baseline e2e | Wire walking skeleton (§3): parse → explore → perceive → answer, all 3 types. First full 75-Q run | **Non-zero score on every question type**; instance recall ≥70%, dupes <15% on 4 scenes |
| **W3** Aug 4–10 | It answers well | M2 top candidate (SAM 3 + SigLIP 2 re-ID); SORT3D toolbox in; tool-calling agent + decision gate + reobserve | ≥70% numerical exact + object-ref IoU on training Qs; dupes <10% |
| **W4** Aug 11–17 | It follows | M5 top candidate (costmap, dense waypoints, monitor); M6 trajectory scorer; language-scored frontiers for multi-room; mid-point model-watch check | Majority of instruction Qs scoring >3/6 on our scorer; multi-room scenes covered <6 min |
| **W5** Aug 18–25 | It survives | Timing/early-stop tuning; fallback paths forced-tested; API retry + provider fallback; clean-machine rebuild of amd64 image; **submit ~Aug 20, iterate to Aug 25** | Full 75-Q run with zero unanswered; submission evaluated same way organizers will run it |

---

## 6. Top risks & mitigations

| Risk | Mitigation |
|---|---|
| SAM 3/YOLOE weak on Unity synthetic textures | W1 bake-off catches it; fine-tune on training-scene renders w/ VLA-3D labels |
| Duplicate instances wreck counting | Re-ID stress test on identical-furniture scene in M6; SAM 3 tracking; obs_count≥2 rule |
| API outage/latency at eval | Cached parses, retries, second provider, local-LLM fallback for parse+count |
| Trajectory scoring semantics guessed wrong | Build our scorer W4 vs provided `.ply` GT; ask organizers via GitHub issue early |
| Eval GPU weaker than L4 | Keep 2× latency headroom; SAM 3 edge/distilled variants pre-tested |
| 10-min timeout on multi-room scenes | Doorway-aware exploration; budget manager: hard answer-mode at T-60s |

---

## Sources

[SORT3D](https://arxiv.org/abs/2504.18684) · [SORT3D code](https://github.com/nzantout/SORT3D) · [SAM 3](https://github.com/facebookresearch/sam3) · [SigLIP 2](https://arxiv.org/abs/2502.14786) · [YOLOE](https://learnopencv.com/yoloe-tutorial-real-time-open-vocabulary-detection/) · [VLFM](https://arxiv.org/pdf/2312.03275) · [LGR](https://arxiv.org/pdf/2503.20241) · [LLM-MCoX](https://arxiv.org/html/2509.26324v1) · [SceneGraphGrounder](https://arxiv.org/html/2605.21788) · [View-on-Graph](https://arxiv.org/abs/2512.09215) · [SPAZER](https://arxiv.org/pdf/2506.21924) · [Beyond Waypoints](https://arxiv.org/pdf/2606.07244) · [NavOne](https://arxiv.org/pdf/2605.06317) · [Org thesis (object-centric grounding)](https://www.ri.cmu.edu/app/uploads/2025/09/HaochenZhang_MSR_Thesis.pdf) · [Challenge repo](https://github.com/Yuxin916/CMU-VLA-Challenge-2026) · [2025 3rd-place note (CMU MRSD)](https://labs.ri.cmu.edu/mrsd-news/articles/)
