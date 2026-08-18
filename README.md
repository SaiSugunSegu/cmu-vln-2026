## CMU-VLN-Challenge Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                 SIMULATOR + BASE AUTONOMY  (provided)                 │
│         Unity scene · SLAM · terrain analysis · waypoint nav          │
└───────────────────────────────────────────────────────────────────────┘
   │ /camera/image (360°, 10 Hz)                                ▲
   │ /registered_scan (5 Hz)            /way_point_with_heading │
   │ /state_estimation (100–200 Hz)      ┌─────────────────────┴┐
   │ /terrain_map_ext (5 Hz) ───────────▶│   M1 EXPLORATION     │
   ▼                                     │ frontier · coverage  │
┌──────────────────────────┐             │ · reobserve poses    │
│      M2 PERCEPTION       │             └──────────────────────┘
│ SAM 3 masks + track IDs  │                        ▲
│ lidar lifting → 3D box   │                        │ explore /
│ SigLIP 2 re-ID · caption │                        │ reobserve
└──────────────────────────┘                        │
   │ 3D instances                                   │
   │ (label · box · color · caption)                │
   ▼                                                │
┌──────────────────────────┐  queryable  ┌──────────┴───────────┐
│     M3 SCENE GRAPH       │    graph    │     M4 REASONER      │◀── /challenge_question
│ VLA-3D 8 relations       │────────────▶│ LLM over graph       │        (String, 1 Hz)
│ regions · attributes     │             │ + decision gate      │
└──────────────────────────┘             └──────────────────────┘
                              ┌────────────────┼──────────────────┐
                              ▼                ▼                  ▼
                    /numerical_response  /selected_object  ┌──────────────────────┐
                    (Int32 — count)      _marker           │  M5 INSTR. PLANNER   │
                                         (Marker — 3D box) │ costmap · ordered    │
                                                           │ dense waypoints      │
                                                           └──────────────────────┘
                                                                      │
                                                          /way_point_with_heading
                                                          (→ back to base autonomy)
```

**Simulator + base autonomy (provided)** — untouched by us
- In: `Pose2D` waypoints · Out: 360° camera, registered lidar (map frame), terrain maps, odometry
- Handles SLAM, obstacle avoidance, low-level path following

**M1 · Exploration** — see every object fast
- In: terrain map, odometry, requests from M4 · Out: waypoints
- Frontier coverage sweep, doorway detection, `reobserve(id)` viewpoints, early stop for time bonus

**M2 · Perception** — pixels + points → labeled 3D instances
- In: 360° images, lidar scans, odometry · Out: instances (label, 3D box, color, caption)
- SAM 3 text-prompted masks carry track IDs across frames (no separate re-ID stage) → lidar projected through masks → 3D box; Qwen3-VL captions (Qwen2.5-VL and PaliGemma selectable via `--captioning_model`); CLIP/SigLIP embeddings match crops to query text

**M3 · Scene graph** — instances → queryable spatial facts
- In: instance stream · Out: `filter(label, color, relation, anchor)` query API
- VLA-3D's exact 8 relation heuristics (Above/Below/Closest/Farthest/Between/Near/In/On) — same functions that generated the questions

**M4 · Reasoner** — owns the question from parse to answer
- In: question, graph · Out: `Int32` count / `Marker` bbox / constraints to M5 / explore requests to M1
- LLM → symbolic graph queries; decision gate; T-30s fallback (always answer)

**M5 · Instruction planner** — "go near X, avoid Y, stop at Z" → trajectory
- In: ordered grounded constraints, terrain map · Out: dense waypoint sequence
- Costmap with avoid zones, ~1 m waypoint spacing so base planner can't shortcut, replan on violation

**M6 · Eval bench** (`scripts/eval/`) — 15 scenes × 5 questions, challenge-style scoring; every change measured. `just cat1-bag-bench` scores category-1 offline against recorded bags

Full plan: [TEAM_PLAN.md](TEAM_PLAN.md) · deep dives: [docs/](docs/)

## Team quickstart

Everything runs through [`just`](https://github.com/casey/just). Run `just` with no
arguments for the full grouped command list.

### 1 · First run — once per machine

```bash
git clone --recurse-submodules git@github.com:SaiSugunSegu/cmu-vln-2026.git && cd cmu-vln-2026
# existing clone:  git pull && git submodule update --init --recursive

xhost +local:
just up          # build + start containers (the `init` one-shot fixes mount perms; Exited (0) = ok)
just hf-fetch    # ONE-TIME ~15-20 GB weights — nothing loads until this finishes
```

`hf-fetch` is the only command that goes online. It needs `HF_TOKEN=hf_…` in the
repo-root `.env` (**not** `hf auth login`), plus the gated `facebook/sam3` licence
accepted once at <https://huggingface.co/facebook/sam3> with that same account.
[docker/README.md](docker/README.md) · troubleshooting [docs/M0_infra.md](docs/M0_infra.md)

### 2 · Live sim — three terminals

```bash
just sim                                     # A — or sim-noviz (Foxglove) / challenge (6-topic firewall)
just ai                                      # B — starts its own Qwen server (local backend default)
just ask "How many books are on the sofa"    # C — add `42` as a 2nd arg in challenge mode
```

### 3 · Offline bags — no Unity, no sim GPU

Replays the 15 recorded scenes in `data/bags/`. Plain replay: `just bag-play scene_0`.
The category-1 loop — the one that runs today — needs four terminals:

```bash
just vqa-up                          # A — load Qwen once, stays resident
just run-sam                         # B — SAM 3 detector
just cat1-reasoner                   # C — extract targets → prompt SAM → answer
just cat1-bag-bench arabic_room 3    # D — score 3 questions; results in data/runs/
```
[docs/cat1_bag_benchmark.md](docs/cat1_bag_benchmark.md) · [docs/M0.5_rosbag_infra.md](docs/M0.5_rosbag_infra.md)

### 3.5 · End-to-End Orchestrated Evaluation

Iterates every scene × every question. `smart_vlm.launch` is a **disposable unit** — SAM,
supervisor, reasoner and the scene source all start and die together, once per question —
so nothing carries over, exactly as the challenge relaunches the system per command. Model
load therefore counts against the same 10-minute budget it will on the real evaluation,
which makes the reported `time_taken_s` honest.

```bash
just vqa-up                    # A — OPTIONAL: keeps Qwen resident so the per-question
                               #     relaunch does not reload 8.3 GB each time
just eval-cat1 arabic_room 2   # B — dev loop: one scene, two questions
just eval-cat1 all 0           # B — full sweep (75 questions, several hours)
```

Results are rewritten atomically after every question to `/data/runs/challenge_report.json`,
so an interrupted sweep is still valid JSON with a per-scene summary.

Ordering is enforced by two latched gates, not by sleeps. `sam_node` boots **unarmed**: it
loads weights but processes no frame until the reasoner sends this question's prompts, and
the bag is held until then — so the detector never burns the front of a scene on the config's
placeholder objects.

```text
┌────────────────────────┐        ┌────────────────────────────────────────────────┐
│    EVAL ORCHESTRATOR   │        │   smart_vlm.launch  (spawned PER QUESTION)     │
│   [load QA + targets]  │        │                                                │
│                        │ ── /gt_target_objects (latched) ──────────┐             │
│                        │        │  ┌──────────────────────────┐    │             │
│  [spawn launch] ──────────────▶ │  │  sam_node (UNARMED)      │    │             │
│                        │        │  │  loads weights ~60s      │    │             │
│                        │        │  └────────────┬─────────────┘    │             │
│                        │        │   /sam3/status: awaiting_prompts │             │
│                        │        │               ▼                  ▼             │
│                        │ ◀── /pipeline/ready ── ┌───────────────────────────┐    │
│  [publish Q @ 1 Hz] ─────────── /challenge_question ─▶│ TARGET EXTRACTION   │    │
│                        │        │               │  GT targets, else Qwen    │    │
│                        │        │               └─────────────┬─────────────┘    │
│                        │        │                             │ /sam3/set_prompts│
│                        │        │                             ▼                  │
│                        │        │               ┌───────────────────────────┐    │
│                        │        │               │  sam_node ARMS            │    │
│                        │        │               │  fresh session + run_dir  │    │
│                        │        │               └─────────────┬─────────────┘    │
│                        │ ◀── /pipeline/armed ─────────────────┤                  │
│                        │        │                             ▼   releases ──▶   │
│                        │        │               ┌───────────────────────────┐    │
│                        │        │               │  SCENE PLAYS              │    │
│                        │        │               │  bag replay │ TARE explore│    │
│                        │        │               └─────────────┬─────────────┘    │
│                        │        │      /pipeline/explore_done │  (bag end, or    │
│                        │        │       from smart_vlm        │   T-90 / timeout)│
│                        │        │                             ▼                  │
│                        │        │               ┌───────────────────────────┐    │
│                        │        │               │  VQA ANSWERING            │    │
│                        │        │               │  Qwen(question + crops)   │    │
│  [score vs GT]         │ ◀── /numerical_response ─────────────┘                  │
│  [write report.json]   │        │        (or smart_vlm's T-30 fallback guess)    │
│                        │        │                                                │
│  [SIGINT process group]│ ─────▶ │  every node dies together — clean slate         │
│  [next question]       │        │                                                │
└────────────────────────┘        └────────────────────────────────────────────────┘
```

### 4 · Full challenge-style bench (sim)

```bash
python3 scripts/eval/run_bench.py --repo . --scenes-dir ~/vln_scenes --out ~/vln_eval/$(date +%Y%m%d_%H%M) --smoke
python3 scripts/eval/score.py --results ~/vln_eval/<run> --gt scripts/eval/gt/gt.json
```
`--ai-launch` still defaults to `dummy_vlm`. [scripts/eval/README.md](scripts/eval/README.md)

### 5 · Foxglove over SSH (camera + clouds on your laptop, no X11 lag)

```bash
just sim-noviz                                  # remote — or `just challenge`
just foxglove                                   # remote — `just foxglove 42` in challenge mode
ssh -N -L 8765:localhost:8765 <user>@<remote>   # laptop
```
Connect Foxglove desktop to `ws://localhost:8765` (**Open connection → Foxglove
WebSocket**); import `scripts/foxglove/vln_layout.json` via **Layouts → Import from
file**. Add `just teleop` in a third terminal to drive (needs keyboard focus).

### 6 · Command reference

| Group | Commands |
|---|---|
| **setup** | `up` (build + start; also how source edits reach the container) · `down` · `hf-fetch` · `test` |
| **sim** | `sim` · `sim-noviz` · `challenge` · `ask "…"` · `teleop` |
| **bags** | `bag-play <scene>` · `bag <name>` (record) · `list-scenes` |
| **run** | `ai` · `run-sam` · `run-map` |
| **perception** | `sam-status` · `sam-map-json` · `sam-frames` · `sam-probe` · `sam-prompts` |
| **debug** | `foxglove` · `topics` · `shell-ai` · `shell-sys` |
| **vqa** | `vqa-up` · `vqa-ask "…" <img>` · `caption` |
| **cat1** | `cat1-reasoner` · `cat1-bag-bench <scene> [limit]` |
| **cat2** | `gen-cat2` · `verify-cat2` · `pdf-assets` |
| **eval** | `eval-cat1 <scene> [limit] [target_source] [speed]` |
| **map3d** | `map3d-record <scene> [stride]` · `map3d-replay <scene> [jobs]` · `map3d-score` · `map3d-determinism` · `map3d-audit` |

Run `just` for this list with descriptions and default arguments.

### 3.6 · 3D box accuracy — measured, deterministic, no GPU

`map_node`'s 3D boxes had no accuracy metric of any kind, and every experiment re-ran SAM 3
nondeterministically. The `map3d` group fixes both: record SAM 3 output once per scene
(the only GPU step), then replay the 3D mapper on CPU in ~30 s/scene with byte-identical
output, and score it against IRef-VLA ground truth.

```bash
# prompt sets are hand-curated in data/benchmark/bench_prompts.json
just map3d-record all              # ONE-TIME GPU pass, ~21 min for all 13 scenes
just map3d-replay all 8            # CPU, deterministic, all scenes in parallel
just map3d-score                   # vs IRef-VLA GT, incl. the category-2 marker score
```

Full guide, metric definitions and current numbers: [docs/map3d_bench.md](docs/map3d_bench.md)
Mapper internals — every stage, threshold and the Tier 1/2/3 backlog: [docs/map_node_pipeline.md](docs/map_node_pipeline.md)

### 3.7 · Category-2 ground truth — 122 questions, seconds to rebuild

Object reference needs a *box* as the answer, so the benchmark is a join over the IRef-VLA
metadata: up to 10 questions per scene across the 13 scenes with referential statements. No
SAM, no VLM, no bag — the whole thing regenerates in ~15 s, and every answer is re-derived
from the boxes rather than trusted from the statement that produced it.

```bash
just gen-cat2       # -> data/benchmark/<scene>/category_2/<scene>_category2_qa.json
just verify-cat2    # independent audit; non-zero exit on any mismatch
```

Half the questions are distance comparisons (`closest`, `farthest`, `near`) and half are
spatial predicates (`on`, `in`, `supports`, `above`, `below`, `between`). IRef's statements
are almost entirely the former, so the latter are synthesised from the boxes and held to the
same verification — with a per-relation quota so a scene cannot fill up on "farthest from".

Every question is also gated on what the robot **actually saw**: `just visibility` projects
each IRef box into the recorded `/camera/image` frames with the mapper's own camera model and
keeps only the objects the sensors resolved. It turned out to disqualify 47 of the first 130
questions — recessed downlights, a book 270 px² across, geometry behind a wall — and its
committed per-scene report is what lets `gen-cat2` stay bag-free.

Corrections go in `scripts/bench/category2_overrides.json` (pin / reword / drop / hide), never into the
generated JSON — the next `gen-cat2` overwrites it. These answers are also the target set
`just map3d-score` measures its category-2 marker score over.

Full guide, verification rules and the review loop: [docs/cat2_benchmark.md](docs/cat2_benchmark.md)

-----------------------


# CMU-VLN-Challenge Progress

**Finished Basic setup**

<img src="videos/basic_system_setup.jpeg" alt="Basic Setup of System Simulation" width="600" height="350">


-----------------------


# CMU-VLN-Challenge

## Table of Contents
[Introduction](#introduction)  
[Objective](#objective)  
[Task Specification](#task-specification)

[Setting Up](#setting-up)
- [Challenge Scenes](#challenge-scenes)
- [Challenge Questions](#challenge-questions)
- [System](#system)
- [Simulator](#simulator)
- [Object-Referential Dataset](#object-referential-dataset-vla-3d)

[Real-Robot Challenge](#real-robot-challenge-2026)
- [Real-Robot Data](#real-robot-data)

[Submission](#submission)

[Evaluation](#evaluation)
- [Question Types and Initial Scoring](#question-types-and-initial-scoring)
- [Timing](#timing)

[Challenge FAQ](#challenge-faq)

## Introduction
The CMU Vision-Language-Navigation Challenge leverages computer vision and natural language understanding in navigation autonomy. The challenge aims at pushing the limit of embodied AI in real environments and on real robots - providing a robot platform and a working autonomy system to bring everybody's work a step closer to real-world deployment. The challenge provides a real-robot system equipped with a 3D lidar and a 360 camera. The system has base autonomy onboard that can estimate the sensor pose, analyze the terrain, avoid collisions, and navigate to waypoints. Teams will set up software on the robot's onboard computer to interface with the system and navigate the robot. For 2026, the challenge will be done in a custom simulation environment and move to the real-robot system in the second phase. 

To register for the challenge, please see our [Challenge Website](https://www.ai-meets-autonomy.com/cmu-vln-challenge).


## Objective 
Teams are expected to come up with a vision-language model that can take a natural language navigation query and navigate the vehicle system by generating a waypoint or path based on the query.


## Task Specification
In the challenge, teams are provided with a set of natural language questions/statements for scenes from Unity [1]. The team is responsible for developing software that processes the questions together with onboard data of the scene provided by the system. The questions/statements all contain a spatial reasoning component that requires semantic spatial understanding of the objects in the scene. The environment is initially unknown and the scene data is gathered by navigating to appropriate viewpoints and exploring the scene by sending waypoints to the system. 5 questions/statements are provided for each of 15 Unity scenes and 3 scenes are held out for test evaluation.

The natural language questions are separated into three categories: numerical, object reference, and instruction following, which are further described below.

**Numerical**

Numerical questions asks about the quantity of an object that fits certain attributes or spatial relations. The response is expected to be an integer number.

Examples:

    How many blue chairs are between the table and the wall?

    How many black trash cans are near the window? 

**Object Reference**

Object reference statements asks the system to find a certain object located in the scene that is referred to by spatial relations and/or attributes. The response is expected to be a bounding box around the object and there exists only one correct answer in the scene (the referred object is unique). The center point of the bounding box marker will be used as a waypoint to navigate the robot system.

Examples:

    Find the potted plant on the kitchen island that is closest to the fridge.

    Find the orange chair between the table and sink that is closest to the window.

**Instruction-Following**

Instruction following statements ask the system to take a certain path, using objects to specify the trajectory of the path. The response is expected to be a sequence of waypoints.

Examples:

    Take the path near the window to the fridge.

    Avoid the path between the two tables and go near the blue trash can near the window.


## Setting Up
First, clone this repo and place it under your local folder.

```
git clone --recurse-submodules git@github.com:Yuxin916/CMU-VLN-Challenge-2026.git
```

### Challenge Scenes
A total of 18 Unity scenes are used for the challenge. 15 scenes are provided for model development while 3 are held out for testing. The majority of these scenes are single rooms while a few are multi-room buildings.  A set of the training environment models can be downloaded from [here](https://drive.google.com/drive/folders/1nki_xoFKX1bYr8m7qiGRQelwnQ7EKVYc?usp=drive_link). For all of the 15 training scenes, we also provide a processed point cloud of the scene, object and region information including color and size attributes, and referential language statements (please see [Object-Referential Dataset](#object-referential-dataset-vla-3d) for more details). 

![image](figures/scenes.png)

### Challenge Questions
A set of challenge questions for each Unity scene is provided in the pdf files for each of the 15 training scenes under the [questions](questions/) folder. Images of the correct answer in each scene are also provided for visualization purposes and a .ply file of the target trajectory is provided as well. All questions for all training scenes can also be found in JSON format under [questions/questions.json](questions/questions.json).

### System

Our system runs on Ubuntu 24.04 and uses ROS Jazzy in both simulation and onboard the real robot. Follow the instructions in the [docker/](docker/) folder to try the simulator by pulling the docker image provided and launching the system.

The system uses Unity environments by default and has two parts:
- The base navigation system is in the [autonomy_stack_mecanum_wheel_platform](https://github.com/Yuxin916/End2end-ObjectNav-Physical-Experiment/tree/81035e9e4190826b7458c711f08cb64f8f9e64ac) folder. This system can be launched by itself without the AI module running. For the base navigation system, you may change the scene used by placing it in the [autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/](https://github.com/Yuxin916/End2end-ObjectNav-Physical-Experiment/tree/81035e9e4190826b7458c711f08cb64f8f9e64ac/src/base_autonomy/vehicle_simulator/mesh/unity/) directory.
- The vision-language model should be in the [ai_module](ai_module/) folder. The model currently in the folder under [ai_module/src](ai_module/src) is a "dummy model" that demonstrates how to read input questions and produces arbitrary examples of the different types of output responses which are to be used by the system and the evaluation node. **Teams are expected to come up with a model to replace this one.**

#### Dummy Model

The dummy model will read a question as a ROS String message on the `/challenge_question` topic. The dummy model will then either publish an integer as an Int32 message, send bounding box visualization markers for object reference, or waypoints to guide vehicle navigation. The three types of messages are listed below. To integrate the a model with the system, please modify the system startup script.
- Numerical response: ROS Int32 message on topic `/numerical_response`, containing an integer answering a numerical question.
- Visualization marker: ROS Marker message on topic `/selected_object_marker`, containing object label and bounding box of the selected object.
- Waypoint: ROS Pose2D message on topic `/way_point_with_heading` (neglect the heading for this year’s challenge).

#### System Outputs
The system provides onboard data to the AI module as shown in the table below:

| Message | Description | Frequency | Frame | ROS Topic Name |
|-|-|-|-|-|
| Image | ROS Image message from the 360 camera. The image is at 1920/640 resolution with 360 deg HFOV and 120 VFOV. | 10Hz | camera | `/camera/image` |
| Registered Scan | ROS PointCloud2 message from the 3D lidar and registered by the state estimation module. | 5Hz | map | `/registered_scan` |
| Sensor Scan | ROS PointCloud2 message from the 3D lidar. | 5Hz | sensor_at_scan | `/sensor_scan` |
| Local Terrain Map | ROS PointCloud2 message from the terrain analysis module around the vehicle. | 5Hz | map | `/terrain_map` (5m around vehicle) <br> `/terrain_map_ext` (20m around vehicle) |
| Sensor Pose| ROS Odometry message from the state estimation module. | 100-200Hz | from map to sensor | `/state_estimation` |


**IMPORTANT NOTE**: While more topics may be available from the system, these are the only ones allowed to be used during test time. During training/development, you are free to use whatever information the system simulator provides. One thing is different that Traversable Area and Ground-truth Semantics will not be provided this year.

#### System Inputs

The system takes waypoints output from the AI module to navigate the robot. Waypoints located in the traversable area (listed above) are accepted directly, and waypoints out of the traversable area are adjusted and moved into the traversable area. The system also takes visualization markers output by the module to highlight selected objects. Int32 messages indicating a numerical response are not directly used by the system to navigate the robot, and are read instead by the evaluation node detailed in the [evaluation](#evaluation) section.

The table below lists the ROS topics to use. The waypoints should be used for Instruction-Following questions, the visualization marker should be the output for the Object Reference questions, and the integers for Numerical questions.

| Message | Description | ROS Topic Name |
|-|-|-|
| Waypoint with Heading | ROS Pose2D message with position and orientation. | `/way_point_with_heading` |
| Selected Object Marker | ROS Marker message containing object label and bounding box of the selected object. | `/selected_object_marker` |
| Numerical Response | ROS Int32 message with an integer as an answer to a numerical question. | `/numerical_response` |

The coordinate frames used by the physical system are shown below. The camera position (camera frame) with respect to the lidar (sensor frame) is measured based on a CAD model. The orientation is calibrated and the images are remapped to keep the camera frame and lidar frame aligned. 

<p align="center">
  <img src="figures/system.png" alt="system" width="30%"/>
</p>


### Object-Referential Dataset (VLA-3D)

To help with the subtask of referential object-grounding, the VLA-3D dataset containing 7.6K indoor 3D scenes with over 11K regions and 9M+ statements is provided. The dataset includes processed scene point clouds, object and region labels, a scene graph of semantic relations, and generated language statements for each 3D scene from a diverse set of data sources and includes the 15 training scenes in Unity. For access to the data and more details on the format, please see our [VLA-3D repository](https://github.com/HaochenZ11/VLA-3D).

## Real-Robot Challenge (2026)

Starting in 2025, the final round of challenge evaluation will be done on the real-robot system while initial evaluation rounds are still done in simulation. Similar to the simulator, the system provides onboard data as described below and takes waypoints in the same way as the simulator. The software developed in the AI module is only able to send waypoints to explore the scene. Manually sending waypoints or teleoperation is not allowed. During the final evaluation phase, each team will remotely login to the robot's onboard computer (16x i9 CPU cores, 32GB RAM, RTX 4090 GPU), and set up software in a Docker container that interfaces with the autonomy modules. The Docker container is used by each team alone and not shared with other teams. We will schedule time slots for teams who pass the simulation round to set up the software and test the robot during that phase. The teams can also record data on the robot's onboard computer and this data will be made available to participants afterwards.

### Real-Robot Data

Example scene data collected from the real system is provided [here](https://drive.google.com/drive/folders/1xaatyLeIKLTh_oRzkyd7F1G6tkRPbFtm?usp=sharing) with some differences in the object layout. The following can be found in the sample data:

- `data_view.rviz2`: An RVIZ configuration file provided for viewing the data
- `system_ros2.zip`: Zipped bagfile with ROS messages provided by the system in the same format as during the challenge

Here, the ground truth map and the object list are not provided files during the challenge but shown as a sample of what information can be obtained and processed from the system. The camera pose (camera frame) with respect to the lidar (sensor frame) can be found in the README file included. Further details about the files can be found in the README text file as well.


## Submission
Submissions will be made as a github repository link to a public repository. The easiest way would be to fork this repository and make changes there, as the repository submitted will need to be structured in the same way. The only files/folders that should be changed are what's under [ai_module](ai_module/). If changes were made to the docker image to install packages, push the updated image to [Docker Hub](https://hub.docker.com/) and submit the link to the image as well.

Prior to submitting, please download the docker image and test it with the simulator as the submission will be evaluated in the same way. Please also make sure that waypoints and visualization markers sent match the types in the example dummy model and are on the same ROS topics so that the base navigation system can correctly receive them.

Please fill out the [Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScdZAcw5S2nbfSKn8qB-kmNC3PEEQHTK64dU9Hqb5iKg0_jtA/viewform) with a link to your Github repo.


## Evaluation
The submitted code will be pulled and evaluated with 3 Unity environment models which have been held from the released data. Each scene will be unknown and the module has a set amount of time to explore and answer the question (see [timing](#timing) for more details). The test scenes are of similar style to the provided training scenes. **The system will be relaunched for each language command tested such that information collected from previously exploring the scene is not retained.** Note that the information onboard the system that is allowed to be used at test time is limited to what is listed in [System Outputs](#system-outputs).

Evaluation is performed by a `challenge_evaluation_node` whose source code is not made public. The evaluation node will be started along with the team-provided AI module and the system at the same time, and publishes a single question each startup as a ROS String message on the following topic at a rate of 1Hz:

| Message | Description | Frequency | ROS Topic Name |
|-|-|-|-|
| Challenge Question | ROS String message with detailed question to solve. | 1Hz | `/challenge_question` |

### Question Types and Initial Scoring

For each scene, 5 questions similar to those provided will be tested and a score will be given to each response. The question types will be scored as follows:
- **Numerical** (/1): Exact number must be published on `/numerical_response` as an `std_msgs/msg/Int32` message. Score of 0 or 1.
- **Object Reference** (/2): ROS `visualization_msgs/msg/Marker` message must be published on `/selected_object_marker`, and is scored based on its degree of overlap with the ground truth object bounding box. Score between 0 and 2.
- **Instruction-Following** (/6): A series of `geometry_msgs/msg/Pose2D` waypoints must be published on `/way_point_with_heading` to guide the vehicle. The score will be calculated based on the actual trajectory followed by the robot based on whether it follows the path constraints in the command and in the correct order. Penalties are imposed upon the score if the followed path deviates from the correct order of constraints, does not achieve the desired constraints, or passes through areas it is forbidden to go through in the command. Score between 0 and 6, with possibility for partial points. 

The scores from all questions across the 3 test scenes will be totaled for each team's final score. 


### Timing

For each question, both re-exploration on system launch and question answering will be timed. Timing will begin immediately at system startup. Each question has a total time limit of **10 minutes** for exploration and question answering combined, regardless of the test scene. Exceeding the time limit for a certain question incurs a penalty on the initial score calculated for the question. Finishing before the allotted time for a question earns bonus points on that question, which will be used to break ties between teams with similar initial scores.


## Challenge FAQ
Any questions regarding the challenge can be asked by opening a Github issue with the "question" label. We encourage you to use this feature so that multiple members of the team can see the question. Questions specific to your team situation can be emailed to jingfant@andrew.cmu.edu or other challenge organizers. Frequently asked questions will be posted here.

1. Are multiple submissions allowed?

    Yes, there is no limit to the number of submissions allowed during the competition. The submission form is set up to allow multiple submissions and we will take your highest scoring one.

2. What are the time constraints for completing the task?

    Please check the [timing](#timing) section.

3. Any restrictions on the usage of LLMs/VLMs/APIs?

    There are no restrictions on using LLMs, VLMs, or online APIs. Any model can be used, however, keep in mind that we will need to be able to run your code and if it needs to query online APIs during runtime, you will have to provide your access token in your code.

4. What is the docker size limit?

    The size limitation depends on the machine we use to run evaluation. The specs for the machine can be found [HERE](https://simplynuc.com/product/nuc13rngi9-full/?gad_source=1&gclid=CjwKCAjwiaa2BhAiE[%E2%80%A6]g4P7AnhLOZQVIoVC9croO7-i74DfuezIOztALzi5RVJ3jv3bxoCxmEQAvD_BwE).

5. How will real-robot evaluation work?
   
   All submissions will first be evaluated in simulation first. Valid submissions will then be evaluated on the real-robot system and teams will be invited to schedule a timeslot and connect remotely to assist with the integration and evaluation.

6. Will ground-truth semantics be provided in the simulation and real-robot evaluation?

   No, we are sorry that this year we will not provide ground-truth semantics in both phases.

7. How will presentation at the IROS workshop work?

   All evaluation will be conducted prior to the IROS conference. The top 3 teams will be contacted with the opportunity to present their method either in-person or remotely.

## Acknowledgements
Thank you to [AlphaZ](https://alpha-z.ai/) for sponsoring the challenge for 2026! Their generous support enables us to provide the top three teams with a cash prize.

## References

[1] J. Haas. "A history of the unity game engine," in Diss. Worcester Polytechnic Institute, vol. 483, no. 2014, pp. 484, 2014.
