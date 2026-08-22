# `smart_vlm`

The team AI module: mission supervision, the three category answer heads, and the
evaluation harness. It composes `sam_mapper` (perception), `captioner` (Qwen VQA) and
the vendored `tare_planner` (exploration) — it does not reimplement any of them.

**Where the flow is documented** — deliberately in two places, not repeated here:

- **Startup ordering and the two gates** (`/pipeline/ready`, `/pipeline/armed`): the header
  comment of [`launch/smart_vlm.launch`](launch/smart_vlm.launch).
- **The end-to-end evaluation sequence**: [repo README §3.5](../../../README.md#35--end-to-end-orchestrated-evaluation).
- **Who publishes what**: the topic-contract table in the `SmartVLM` class docstring
  region of [`smart_vlm/smart_vlm.py`](smart_vlm/smart_vlm.py).

## Launch files

| File | Purpose |
|---|---|
| `launch/smart_vlm.launch` | **The per-question unit.** `sam_node` + supervisor + three reasoners + TARE. Starts no scene source: it consumes the six allowed topics from whatever publishes them. Official eval starts this via `ros2 launch dummy_vlm dummy_vlm.launch`. |
| `launch/eval_bag.launch` | **Eval only, never submitted.** Wraps `smart_vlm.launch` and adds `bag_replay.launch` as the scene source, so offline sweeps do not require the launch under test to know bags exist. |
| `launch/bag_replay.launch` | Replays a recorded scene. Included by the above with `wait_for_armed:=true` (held until SAM has the question's prompts); also usable standalone via `just bag-play`, where the gate is off. |

## Modules

### In `smart_vlm.launch` — these run for every question

| Module | Executable | Role |
|---|---|---|
| `smart_vlm.py` | `smart_vlm` | Supervisor. Owns the mission clock, publishes the `/pipeline/{ready,armed,explore_done}` gates, and guarantees an answer at T-30. Spawns no processes and writes no files. |
| `numerical_reasoner.py` | `numerical_reasoner` | Category-1 answer head: question → target nouns → `/sam3/set_prompts` → wait for `explore_done` → count from SAM's top few best-view crops. Both model steps go through `captioner.vlm_backends`. |
| `object_reference_reasoner.py` | `object_reference_reasoner` | Category-2 answer head: same arming chain → pick one 3D box → `/selected_object_marker`. |
| `instruction_reasoner.py` | `instruction_reasoner` | Category-3 answer head: same arming chain → hold TARE → ordered `/way_point_with_heading` toward each named object. |

### Pure logic — no ROS imports, unit-tested on the host

| Module | Role |
|---|---|
| `question.py` | `QuestionType` enum + `classify()`. Lives apart from the nodes so the heads can route a question without importing rclpy. |
| `mission_clock.py` | The 10-minute budget as arithmetic: `MissionBudget`, `MissionClock`, `Phase`. All deadline decisions live here so they can be tested without a ROS graph. |
| `numerical_utils.py` | Target-noun cleanup, integer parsing, and the shared extract prompt. Re-exports `extract_integer` from `captioner.text_utils`. |
| `cat2_utils.py` | Object-reference selection (`select_object`) shared with `cat2_bench`. |
| `cat3_utils.py` | Instruction waypoint list from `obj_map.json` + SAM prompts. |
| `reasoner_common.py` | Shared VQA transport, noun extract, obj-map wait, and the extract → arm-SAM lifecycle for the three reasoners. |

Covered by `tests/test_question.py`, `tests/test_mission_clock.py`,
`tests/test_numerical_reasoner.py`, `tests/test_cat3_utils.py` — these run on a bare
host, no container needed:

```bash
python3 -m pytest ai_module/src/captioner/tests ai_module/src/smart_vlm/tests -q
```

### Standalone tools — never started by `smart_vlm.launch`

| Module | Executable | Role |
|---|---|---|
| `eval_orchestrator.py` | `eval_orchestrator` | Drives `just eval-cat1`. Iterates every scene × question, spawning and tearing down `smart_vlm.launch` per question, and writes `/data/runs/challenge_report.json`. |
| `bag_fetch.py` | `bag_fetch` | Fetch-if-missing for scene bags, reading `data/bags/scenes.yaml`. Called by `bag_replay.launch`. |
| `wait_ready.py` | `wait_ready` | Blocks until a latched topic fires, then exits 0/1. Used as the `/pipeline/armed` barrier in `bag_replay.launch`; `ros2 topic echo --once` cannot do this, as its VOLATILE subscriber misses an already-latched sample. |
| `qwen_numerical.py` | `qwen_numerical` | SAM-independent fallback head — asks Qwen about the whole 360° panorama rather than a detection crop. Kept for the case where the detector finds nothing; run by hand, not wired into any launch. |

## Dependencies

Nothing here loads a model checkpoint of its own.

On the local backend `smart_vlm.launch` starts `captioner`'s VQA server itself, so
nothing has to be running first — at evaluation there is nobody to start it. Every VQA
request goes to that one server, so a single copy of the weights serves the reasoner and
the target extractor. `just vqa-up` is an optional dev speed-up that keeps a server
resident across the per-question relaunches; do not run it alongside the launch's own. On the cloud backend nothing local is
needed — the reasoner posts the same views to whatever OpenAI-compatible endpoint
`VLM_PROVIDER` names, and the harness drops the readiness wait on `/qwen_vqa/status`
accordingly.
