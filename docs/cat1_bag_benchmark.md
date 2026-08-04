# Category-1 Bag Benchmark (SAM + Qwen)

Offline evaluation of **category 1 (numerical)** questions against recorded scene bags.
Mirrors the challenge I/O contract while replacing live exploration with bag replay.

QA data lives under [`data/benchmark/`](../data/benchmark/).

## Layout

```
data/benchmark/
  <scene>/category_1/<scene>_category1_qa.json   # 30 questions per scene
```

Each QA file includes `question`, integer `answer`, `difficulty`, and `target_objects`
(for audit / scoring only — the live pipeline extracts targets with Qwen).

Scene bags live under `data/bags/<scene>/` (auto-fetched via `data/bags/scenes.yaml` if missing).

## How it works

For each question:

1. Publish the question on `/challenge_question` (1 Hz, like the eval node).
2. **Qwen** extracts short object noun phrases from the question text.
3. Those phrases are sent to **SAM** as text prompts (`/sam3/set_prompts`).
4. The driver waits until `/sam3/status` is `ready`, then plays the scene bag once.
5. **SAM** detects/tracks objects and writes top-N best-view crops + `manifest.json`.
6. After the bag (plus a drain wait), the driver signals `/pipeline/explore_done`.
7. **Qwen** answers the numerical question from `best_rank1` + the manifest.
8. Qwen captions instance crops and appends `attributes` into the manifest.
9. The answer is published once on `/numerical_response` and scored against GT.

`map_node` is **not** required for this path (best views are produced inside `sam_node`).

## Data flow

```
┌────────────────────┐  1 Hz String   ┌─────────────────────┐
│ cat1_bag_bench     │───────────────▶│ category1_reasoner  │
│ (driver)           │                │                     │
└─────────┬──────────┘                └──────────┬──────────┘
          │                                      │
          │ /pipeline/explore_done               │ extract / VQA
          │                                      ▼
          │                           ┌─────────────────────┐
          │                           │ qwen_vqa_server     │
          │                           │ (/qwen_vqa/*)       │
          │                           └──────────┬──────────┘
          │                                      │
          │            /sam3/set_prompts ◀───────┘
          │            /sam3/prompts_ack
          │            /sam3/status (ready)
          ▼
┌────────────────────┐   /camera/image    ┌─────────────────────┐
│ bag_replay         │───────────────────▶│ sam_node            │
│ (ros2 bag play)    │                    │ best-view collector │
└────────────────────┘                    └──────────┬──────────┘
                                                     │
                                                     ▼
                              data/crops/<run_id>_<targets>/
                                best_rank*.png
                                manifest.json  (+ labels, attributes)
                                                     │
category1_reasoner ◀── VQA on best_rank1 ────────────┘
        │
        ▼
 /numerical_response  →  driver scores vs GT
```

### Challenge-facing topics

| Topic | Type | Role |
|-------|------|------|
| `/challenge_question` | `std_msgs/String` | Question in (1 Hz until answered) |
| `/numerical_response` | `std_msgs/Int32` | Integer answer out (once) |

### Internal topics

| Topic | Role |
|-------|------|
| `/sam3/set_prompts` | JSON `{"prompts": [...], "run_id": "..."}` |
| `/sam3/prompts_ack` | Applied prompts + `run_dir` |
| `/sam3/status` | Latched `loading` \| `setting_prompts` \| `ready` |
| `/sam3/best_view_dir` | Latched path to current best-view run |
| `/pipeline/explore_done` | Bag finished; reasoner may answer |
| `/qwen_vqa/request` | VQA / extract / attribute requests |
| `/qwen_vqa/response` | Model replies |
| `/qwen_vqa/status` | Latched `loading` \| `ready` |

## Nodes involved

| Process | How to start | Responsibility |
|---------|--------------|----------------|
| `qwen_vqa_server` | `just vqa-up` | Persistent Qwen3-VL (extract, numerical VQA, attributes) |
| `sam_node` | `just run-sam` | SAM 3 detection + best-view crops + manifest |
| `category1_reasoner` | `just cat1-reasoner` | Orchestrates extract → prompts → answer |
| `cat1_bag_bench` | `just cat1-bag-bench …` | Per-question driver: question → wait SAM → bag → score |
| `bag_replay` | started by the driver | Single-pass bag play for the scene |

## Commands

Run from the repo root. Use **four terminals**.

### 1. Containers + Qwen (once)

```bash
cd /home/ubuntu/myspace/cmu-vln-2026
just up-dev-fast
just vqa-up
```

Wait for `VQA server ready.`

### 2. Terminal A — SAM

```bash
just run-sam
```

Wait for `SAM 3 ready` / `sam_node started`. Optional check:

```bash
docker exec iros2026_ai_module bash -lc \
  'source /home/docker/ai_module/install/setup.bash && \
   timeout 5 ros2 topic echo /sam3/status std_msgs/msg/String --once'
```

Expect `data: ready`.

### 3. Terminal B — reasoner

```bash
just cat1-reasoner
```

Wait for `category1_reasoner ready`.

### 4. Terminal C — benchmark

Smoke (first 3 questions):

```bash
just cat1-bag-bench arabic_room 3
```

Specific question IDs:

```bash
just cat1-bag-bench arabic_room 0 "Q01 Q02 Q03"
```

Full scene (all 30 questions):

```bash
just cat1-bag-bench arabic_room 0
```

Other scenes (must have a bag in `data/bags/scenes.yaml` and a QA JSON under `data/benchmark/`):

```bash
just cat1-bag-bench japanese_room 3
just cat1-bag-bench livingroom_1 0 "Q01"
```

### Direct driver (inside the AI container)

```bash
docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  python3 /home/ubuntu/myspace/cmu-vln-2026/scripts/eval/run_cat1_bag_bench.py \
    --qa /data/benchmark/arabic_room/category_1/arabic_room_category1_qa.json \
    --scene arabic_room \
    --out /data/runs/cat1_arabic_room \
    --ids Q01 Q02 Q03 \
    --speed 1.0 \
    --post-bag-wait 45
'
```

Useful flags: `--speed`, `--post-bag-wait` (SAM drain after bag), `--sam-ready-timeout`, `--pre-bag-settle`.

## Outputs

| Output | Host path | Container path |
|--------|-----------|----------------|
| Best-view images + `manifest.json` | `data/crops/cat1_<run>_…/` | `/data/crops/…` |
| Per-question scores | `data/runs/cat1_<scene>/` | `/data/runs/cat1_<scene>/` |

Default score dir for `just cat1-bag-bench`: `data/runs/cat1_<scene>/`.

### `manifest.json` (after a successful answer)

- `targets`, `selected[].instances[]` with `track_id`, **`label`**, `score`, `bbox`
- Top-level `question`, `extracted_targets`, `predicted_answer`
- Per-instance `attributes` (Qwen caption: color / material / shape)

### Result JSON

Each `arabic_room_Q01.json` (example) contains `gt`, `predicted`, `correct`,
`extracted_targets`, `best_view_dir`, and `prompts_ack`.  
`summary.json` aggregates accuracy over the run.

## Per-question ordering (driver)

1. Wait `/sam3/status == ready`
2. Publish `/challenge_question`
3. Wait `/sam3/prompts_ack`
4. Wait `/sam3/status == ready` again (prompts applied)
5. Play bag (single pass)
6. Post-bag wait (SAM frame drain)
7. Publish `/pipeline/explore_done`
8. Wait `/numerical_response` and score
