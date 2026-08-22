# Common ops for CMU-VLN. Run from anywhere in the repo: `just <recipe>`
# Requires https://github.com/casey/just  (e.g. `cargo install just` or `sudo snap install just`)
#
# Run `just` with no arguments for the grouped recipe list.
#
# First run, in order:
#   just up          # build + start containers
#   just hf-fetch    # ONE-TIME ~15-20 GB weight download (needs HF_TOKEN in .env)
#
# Then pick a flow:
#   scored eval   just eval-cat1 arabic_room 2                # one command, per-question relaunch
#                 just eval-cat2 chinese_room 2               # the same for object reference
#   switch VLM    edit vqa.yaml `vlm_backend` / `target_extract_backend`  # then just up
#   compare VLMs  just cache-cat1   (once, hours) then just bench-cat1  (per model, minutes)
#                 just cache-cat2                then just bench-cat2  (per selection mode)
#   live sim      just sim          | just ai | just ask "How many …"
#   manual bags   just vqa-up ; just run-sam ; just cat1-reasoner ; just cat1-bag-bench

vgl := "cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl"
# The workspace path inside the image.
ai_src := "/home/docker/ai_module"
# `bash -lc` does not source ~/.bashrc for non-interactive shells, so recipes that
# invoke a captioner console script must put its install dir on PATH themselves.
capt_env := "source /home/docker/ai_module/install/setup.bash && export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH"

# --unsorted keeps groups in justfile order (setup first, then the flows) instead
# of alphabetical, which would bury [setup] in the middle.
default:
    @just --list --unsorted

# The only way to start the stack. ai_module is never bind-mounted: the container
# always runs the source baked into the image, which is exactly what CI and the eval
# harness run. The corollary is that an ai_module edit does nothing until you re-run
# this — the rebuild IS how source lands in the container.
[group('setup')]
[doc('Build + start both containers (GPU)')]
[working-directory: 'docker']
up:
    docker compose -f compose_gpu.yml up --build -d

[group('setup')]
[doc('Stop and remove both containers')]
[working-directory: 'docker']
down:
    docker compose -f compose_gpu.yml down

# Pre-seeds the HF cache so the first real run is not also a ~20 GB download, and warms
# SAM 3's cv-utils kernel, whose absence silently disables mask NMS. Re-run on a new machine.
# Gated sam3 needs HF_TOKEN in .env and the licence accepted on the model page.
#   just hf-fetch                # all defaults
#   just hf-fetch "qwen3vl sam3" # a subset;  just hf-fetch --list  to see them
[group('setup')]
[doc('ONE-TIME ~15-20 GB weight download (sam3, Qwen3-VL, CLIP) + SAM 3 kernels')]
hf-fetch models="":
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "{{capt_env}} && fetch_weights {{models}}"

# Runs in the container because sam_mapper/smart_vlm tests need cv2 + rclpy;
# the captioner/ ones are stdlib-only and also run on the host:
#   python3 -m pytest ai_module/src/captioner/tests ai_module/src/smart_vlm/tests -q
[group('setup')]
[doc('Run unit tests in the container')]
test pkgs="captioner sam_mapper smart_vlm":
    #!/usr/bin/env bash
    set -euo pipefail
    dirs=""
    for p in {{pkgs}}; do dirs="$dirs {{ai_src}}/src/$p/tests"; done
    # -p no:cacheprovider: the source tree is root-owned in the container, so pytest's
    # .pytest_cache write fails and emits three Permission-denied warnings per run. The
    # cache buys nothing here (no --lf/--sw usage), so turn it off rather than chmod.
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      python3 -m pytest $dirs -q -p no:cacheprovider
    "

# Override display: just sim :0
[group('sim')]
[doc('Simulator + base autonomy + rviz2 (blocks; terminal A)')]
sim sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./system_simulation.sh"

[group('sim')]
[doc('Simulator without rviz2 — pair with `just foxglove`')]
sim-noviz sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./system_simulation_noviz.sh"

[group('sim')]
[doc('Simulator behind the 6-topic eval firewall (domain 42, no rviz)')]
challenge sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./challenge_simulation.sh --noviz"

# Single pass by default; loop:=true to loop (e.g. just bag-play livingroom_1 1.0 true)
[group('bags')]
[doc('Replay a recorded scene bag instead of the live sim')]
bag-play scene="livingroom_1" speed="1.0" loop="false":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm bag_replay.launch scene:={{scene}} speed:={{speed}} loop:={{loop}}"

# Self-contained: when vqa.yaml has a local backend the launch starts its own Qwen
# server, so `just vqa-up` is not a prerequisite — and must not be running at the
# same time, or two servers collide on the node name and the /qwen_vqa topics.
# Use vqa-up only to keep a server resident across per-question relaunches.
# Brings up sam_node too (unarmed until a question supplies prompts).
[group('run')]
[doc('Official entry: dummy_vlm.launch → SAM + supervisor + reasoners + TARE')]
ai:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch dummy_vlm dummy_vlm.launch"

# Exploration on its own, with no perception or reasoning attached -- the way to tell a
# TARE problem from a pipeline problem. Needs the sim already up in the other terminal
# (`just challenge` for the firewalled 6-topic view, or `just sim` for everything).
# Runs the real pipeline config, so the robot HOLDS STILL until the gate opens -- that
# is cmu_challenge.yaml's kAutoStart: false, normally released by the supervisor.
# Nothing here plays that role, so open it yourself from a third terminal:
#   just tare      # terminal B: waits, logging "Waiting for start signal"
#   just tare-go   # terminal C: the robot starts moving
# Pass scenario=indoor_small instead for the upstream tuning, which auto-starts.
[group('run')]
[doc('TARE exploration alone (blocks; holds until `just tare-go`)')]
tare scenario="cmu_challenge":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch tare_planner explore.launch scenario:={{scenario}}"

# Stands in for the supervisor, which publishes this alongside /pipeline/armed once SAM
# holds the question's prompts. Only needed for a bare `just tare` -- in the full
# pipeline (`just ai`) the supervisor sends it.
[group('run')]
[doc('Release the TARE start gate (supervisor stand-in; for a bare `just tare`)')]
tare-go:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic pub --once /start_exploration std_msgs/msg/Bool '{data: true}'"

# /camera/image in -> /annotated_image, /sam3/instance_map, /sam3/detections out.
[group('run')]
[doc('SAM 3 2D detector node (blocks)')]
run-sam config="sam3_mecanum_sim.yaml":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch sam_mapper sam_node.launch config:={{config}}"

[group('run')]
[doc('Lidar fusion / 3D mapping node (blocks)')]
run-map config="sam3_mecanum_sim.yaml":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch sam_mapper map_node.launch config:={{config}}"

# Use out=/data/bags/_frames to see them on the host.
[group('sam')]
[doc('Dump /camera/image frames from a bag to PNGs, for the offline probe')]
sam-frames scene="livingroom_1" out="/data/bags/_frames" limit="40":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.dump_frames --bag /data/bags/{{scene}} --out {{out}} --limit {{limit}}"

# Are object IDs stable across frames, and which image_size is fastest?
# Answers whether dropping ByteTrack was sound.
[group('sam')]
[doc('Offline SAM 3 probe on dumped frames: ID stability + image_size sweep')]
sam-probe frames="/data/bags/_frames" config="sam3_mecanum_sim.yaml" args="--sweep-image-size":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.sam3_backend --frames {{frames}} --config /home/docker/ai_module/src/sam_mapper/config/{{config}} {{args}}"

# Splits the frame into vision encoder / per-prompt detection / tracker / mask transfer, and
# fits ms ~ fixed + per_object * N per stage. Dump frames first with `just sam-frames`.
#   just sam-profile <frames> <cfg> "--prompts tv,cabinet,chair"   one prompt regime
#   just sam-profile <frames> <cfg> --torch-profile                 when residual is >10%
[group('sam')]
[doc('Where does a SAM 3 frame actually go? Per-stage breakdown')]
sam-profile frames="/data/bags/_frames" config="sam3_mecanum_sim.yaml" args="":
    just sam-probe {{frames}} {{config}} "--profile {{args}}"

# Profiles SAM 3.1 the way sam_node runs it: `just hf-fetch sam3.1` first.
[group('sam')]
[doc('SAM 3.1 benchmark: all concepts batched through one backbone pass')]
sam31-probe args="--bench":
    docker exec -it -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && \
       python3 /home/docker/scripts/eval/sam31_probe.py {{args}}"

# Challenge mode uses domain 42: just ask "…" 42
[group('sim')]
[doc('Publish a one-shot question on /challenge_question')]
ask q domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '{{q}}'}\""

# Default domain 0 matches sim-noviz. Challenge mode: just foxglove 42
[group('debug')]
[doc('Foxglove bridge on host port 8765 (blocks; tunnel from laptop)')]
foxglove domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8090 address:=0.0.0.0"

# Default domain 0 matches sim-noviz. Challenge mode: just teleop 42
[group('sim')]
[doc('Keyboard drive via /joy (needs TTY focus; domain must match sim)')]
teleop domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_system bash -lc \
      'source /opt/ros/jazzy/setup.bash && \
       source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && \
       python3 /home/docker/scripts/keyboard_teleop.py'

[group('debug')]
[doc('List all ROS topics visible in the AI container')]
topics:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic list"

[group('bags')]
[doc('Record the 6 allowed topics + tf into data/bags/<name>')]
bag name:
    docker exec -it iros2026_ai_module bash -c "ros2 bag record /camera/image /registered_scan /sensor_scan /terrain_map /terrain_map_ext /state_estimation /tf /tf_static /challenge_question -o /data/bags/{{name}}"

# Static list -- the Drive folder is a public link, not in this account's own
# Drive, so it can't be queried live.
[group('bags')]
[doc('List the 18 Unity scenes: 15 training + 3 held-out test')]
list-scenes:
    #!/usr/bin/env bash
    echo "Training scenes (15, have questions/ + IRef-VLA metadata):"
    for s in arabic_room chinese_room home_building_1 home_building_2 hotel_room_1 hotel_room_2 japanese_room livingroom_1 livingroom_2 livingroom_3 livingroom_4 loft office_1 office_2 studio; do
      echo "  $s"
    done
    echo
    echo "Held-out test scenes (3, per README -- no IRef-VLA metadata published):"
    echo "  office_building_1"
    echo "  office_building_2"
    echo "  office_building_2_without_360_cam  (dev-speed variant of office_building_2, no camera data -- skip for real collection)"
    echo
    echo "Source: https://drive.google.com/drive/folders/1nki_xoFKX1bYr8m7qiGRQelwnQ7EKVYc"

[group('debug')]
[doc('Interactive shell in the system container')]
shell-sys:
    docker exec -it iros2026_system bash

[group('debug')]
[doc('Interactive shell in the AI module container')]
shell-ai:
    docker exec -it iros2026_ai_module bash

# ---------- 3D map benchmark (map3d) -----------------------------------
# Record SAM 3 output once per scene on GPU, then replay the 3D mapper on CPU
# deterministically in ~30 s/scene. Guide: docs/map3d_bench.md

# The ONLY GPU step. stride=5 is still 2.6x denser than sam_node achieves in production.
# Resume an interrupted sweep with:  just map3d-record all 5 "--skip-existing"
[group('map3d')]
[doc('Record companion /sam3/* bag for a scene (GPU, one-time). scene=all for every scene')]
map3d-record scene="arabic_room" stride="5" args="":
    docker exec -it -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.record_companion --scene {{scene}} --stride {{stride}} {{args}}"
# Splits no-coverage-where-the-mask-sits (sensor geometry) from coverage-beside-it
# (alignment). Opposite fixes.
[group('map3d')]
[doc('Why do masked detections receive zero lidar points?')]
map3d-zeropoints scene="livingroom_1" variant="" args="":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 /home/docker/scripts/eval/diagnose_zero_points.py --scene {{scene}} {{ if variant != '' { '--variant ' + variant } else { '' } }} {{args}}"

# Writes /data/runs/map3d/<run-id>/<scene>.json. run-id is the HOST git sha (the container
# has no .git), so A/B diffs are a git-keyed table.
[group('map3d')]
[doc('Replay the 3D mapper offline against recorded SAM 3 output (CPU, deterministic)')]
map3d-replay scene="arabic_room" jobs="1" args="":
    docker exec -it -e MAP3D_RUN_ID="$(git rev-parse --short HEAD 2>/dev/null || echo dev)$(git diff --quiet 2>/dev/null || echo -dirty)" iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 /home/docker/scripts/eval/replay_map3d.py --scene {{scene}} --jobs {{jobs}} {{args}}"

# Two replays must give an identical digest, or every later A/B is measuring noise.
[group('map3d')]
[doc('Assert the offline replay is byte-reproducible')]
map3d-determinism scene="arabic_room":
    just map3d-replay {{scene}} 1 "--determinism-check --quiet"

# In the container so replay and score write /data/runs as the same uid. Defaults to the
# newest run. Read A/Bs against bestIoU/claimed — precision-recall at a fixed IoU is a step
# function and reads 0.00 until something crosses it.
[group('map3d')]
[doc('Score a replayed 3D map against IRef-VLA ground truth')]
map3d-score run="" args="":
    docker exec -it -e MAP3D_RUN_ID="$(git rev-parse --short HEAD 2>/dev/null || echo dev)$(git diff --quiet 2>/dev/null || echo -dirty)" iros2026_ai_module bash -c "python3 /home/docker/scripts/eval/score_map3d.py {{ if run != '' { '--run ' + run } else { '' } }} {{args}}"

# A sloppy prompt->GT-label map silently invalidates every number above.
[group('map3d')]
[doc('Which prompts / predicted labels fail to resolve to a GT label?')]
map3d-audit run="":
    just map3d-score "{{run}}" --audit-labels

# Regenerates sam_mapper/dimension_priors.json (D3 caps) from VLA-3D ground truth. Needed
# only when new scene GT lands. Runs on the HOST — it reads ../IRef-VLA.
[group('map3d')]
[doc('Regenerate the per-class size caps from VLA-3D ground truth')]
map3d-priors args="":
    python3 scripts/eval/build_dimension_priors.py {{args}}

# ---------- Category-3 (instruction following) benchmark -------------------
# The GT is a reading of the organizers' demo trajectories in questions/<scene>/*.ply.
# Both recipes run on the HOST — they read ../IRef-VLA, which no container mounts, same
# reason as map3d-priors above. Guide: docs/cat3_benchmark.md
#
# cat3-verify is the gate: it replays every demo through score.py::score_instruction and
# requires 6/6. If a radius, an order or an avoid zone is wrong, this is what says so.
[group('bench')]
[doc('Audit the category-3 GT: every demo trajectory must score 6/6')]
cat3-verify args="":
    python3 scripts/eval/verify_category3.py {{args}}

# --dry-run reports without writing; --write regenerates all 15 files. Hand decisions live
# in scripts/bench/category3_overrides.json, so a rerun reproduces them exactly.
[group('bench')]
[doc('Rebuild the category-3 QA files from the demo trajectories')]
cat3-build args="--dry-run":
    python3 scripts/bench/generate_category3_qa.py {{args}}

[group('map3d')]
[doc('One-shot JSON dump of every 3D instance the mapper has built')]
sam-map-json:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic echo /obj_map_json --once"

# ---------- Persistent Qwen VQA (model stays loaded) --------------------
#   just vqa-up
#   just vqa-ask "How many pillows?" /data/img.png
#   just vqa-down

# Starts the Qwen VQA server process. Run it before `just ai` or the evaluation.
[group('vqa')]
[doc('Load Qwen once and keep it resident (~60s first time)')]
vqa-up model="qwen3vl" quantization="int4":
    docker exec -it -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && ros2 launch captioner vqa_server.launch model:={{model}} quantization:={{quantization}}"

# Image path: container-absolute under /data (host data/x is /data/x).
[group('vqa')]
[doc('Ask the running server a question about an image (fast; no reload)')]
vqa-ask q image:
    #!/usr/bin/env bash
    set -euo pipefail
    q="{{q}}"
    image="{{image}}"
    q="${q#q=}"
    image="${image#image=}"
    # Rewrite repo-relative data/ paths to the container mount.
    if [[ "$image" != /* ]]; then
      image="/data/${image#data/}"
    else
      image="${image/#$PWD\/data//data}"
    fi
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      if ! pgrep -f '[q]wen_vqa_server' >/dev/null; then
        echo 'VQA server is not running. Start it with: just vqa-up' >&2
        exit 1
      fi
      {{capt_env}} &&
      qwen_vqa_ask --question $(printf '%q' "$q") --image $(printf '%q' "$image")
    "

# Host data/ <-> /data, so `just caption crops captions` reads data/crops and
# writes data/captions. Weights come from the mounted ~/.cache/huggingface.
[group('vqa')]
[doc('Caption a folder of crops offline with Qwen (data/crops -> data/captions)')]
caption input="crops" output="captions" batch_size="8" quantization="int4" model="qwen3vl":
    #!/usr/bin/env bash
    set -euo pipefail
    # Older just versions pass "model=qwen3vl" as a positional string instead of
    # binding the keyword; strip key= prefixes so both styles work.
    in="{{input}}"
    out="{{output}}"
    batch_size="{{batch_size}}"
    quantization="{{quantization}}"
    model="{{model}}"
    batch_size="${batch_size#batch_size=}"
    quantization="${quantization#quantization=}"
    model="${model#model=}"
    # Host data/ is bind-mounted 1:1 at /data, so a bare name or a host path under
    # the repo's data/ both become the same container path.
    [[ "$in"  == /* ]] || in="/data/${in}"
    [[ "$out" == /* ]] || out="/data/${out}"
    in="${in/#$PWD\/data//data}"
    out="${out/#$PWD\/data//data}"
    # No host-side mkdir/chmod: the init one-shot in docker/compose.yml makes
    # /data writable by the container's uid 1001, and these are container paths —
    # creating them on the host would make a root-level /data.
    # PYTHONUTF8: native libs loaded during model init can reset the C locale, which
    # makes Python's default text encoding ascii and breaks writing captions that
    # contain curly quotes. UTF-8 mode ignores the locale entirely.
    docker exec -it -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      {{capt_env}} &&
      mkdir -p '${out}' &&
      caption_crops '${in}' \
        --output_dir '${out}' \
        --captioning_model '${model}' \
        --quantization '${quantization}' \
        --batch_size ${batch_size}
    "

# ---------- Category-1 bag bench (SAM best-views + Qwen VQA) ---------------
# Terminals: just vqa-up | just run-sam | just cat1-reasoner | just cat1-bag-bench
# Bench waits for /sam3/status=ready before (and after) prompts, then starts the bag.
#
# Which model answers is pinned in ai_module/src/captioner/config/vqa.yaml
# (`vlm_backend` / `provider` / `model`), not passed here -- see
# captioner/vlm_backends/constants.py.
# views is how many best-view ranks it answers from at once (the VQA server caps at 4).
# Which image of a best-view crop the model sees (silhouette, the mask-outline + label
# copy, vs. the plain crop) is vqa.yaml's `view_source`.
[group('cat1')]
[doc('Rebuild smart_vlm/sam_mapper and launch the category-1 reasoner (blocks)')]
cat1-reasoner views="3":
    #!/usr/bin/env bash
    set -euo pipefail
    # Rebuild the image so this reasoner runs the current source: captioner counts
    # too, since smart_vlm imports captioner.paths / .text_utils / .ros_utils /
    # .vlm_backends.
    just up
    docker exec -it iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      ros2 run smart_vlm numerical_reasoner --ros-args -p max_context_views:={{views}}
    "

# The whole pipeline (SAM and, when either vqa.yaml backend is local, Qwen) is relaunched
# per question, so model load lands inside the measured budget exactly as it will on
# the real evaluation -- a full sweep is 75 questions and takes hours. `just vqa-up`
# beforehand is an optional speed-up: a resident server survives the relaunches and
# saves reloading 8.3 GB per question. Use a scene + limit as the dev loop:
#   just eval-cat1 arabic_room 2
# target_source=vlm exercises the model target-extraction path instead of benchmark GT.
# Cloud vs local is `vlm_backend` / `target_extract_backend` in vqa.yaml. Give a separate report=
# when A/B-ing two configurations, or the second sweep overwrites the first.
# Crops land in data/crops/<report name>/<scene>/<question id>-<question>/ and the report
# records the directory, so any sweep's report doubles as the cache index that
# `just bench-cat1` replays from -- and a second sweep with its own report= keeps its
# own crops rather than overwriting the first one's question by question.
# Which image of a best-view crop the model sees (silhouette vs. plain crop) is set once
# in ai_module/src/captioner/config/vqa.yaml's `view_source`, the same switch category-2
# and the offline benches read -- edit it there rather than per-invocation here. See
# captioner/vlm_backends/constants.py.
[group('eval')]
[doc('Orchestrated end-to-end category-1 eval; relaunches the pipeline per question')]
eval-cat1 scene="all" limit="0" target_source="gt" speed="0.1" report="/data/runs/challenge_report.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p report_file:={{report}}"

# The same driver against the object-reference questions: same per-question relaunch, same
# gates, but the answer arrives as a Marker on /selected_object_marker and is graded on
# overlap -- twice the axis-aligned 3D IoU against the answer's box, the challenge's own
# formula -- so a question scores partial credit rather than pass/fail. 122 questions at
# roughly 3 minutes each is about 6 hours, so use a scene + limit as the dev loop:
#   just eval-cat2 chinese_room 2
# mode= is how the reasoner chooses: hybrid (ships: solver, model only where the geometry
# is not decisive), solver (no model call), vlm, naive. See cat2_utils.select_object.
# A row records the score, not what was reachable: whether a zero is selection or perception
# takes the run's obj_map.json against the answer's box. bench-cat2 carries that ceiling.
# view_source is the same vqa.yaml setting category-1 reads -- see eval-cat1 above.
[group('eval')]
[doc('Orchestrated end-to-end category-2 eval; relaunches the pipeline per question')]
eval-cat2 scene="all" limit="0" mode="hybrid" target_source="gt" speed="0.1" report="/data/runs/cat2_report.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p category:=2 -p scene:={{scene}} -p question_limit:={{limit}} -p cat2_mode:={{mode}} -p target_source:={{target_source}} -p speed:={{speed}} -p report_file:={{report}}"

# The same two sweeps against the LIVE SIM instead of bags, with TARE driving. Unlike
# the bag recipes these run on the HOST, not in a container: switching Unity scenes
# means docker cp-ing a mesh into the system container and restarting the sim, which
# nothing inside the AI module can do. The driver handles the scene loop, so `all`
# works here the way it does for bags -- do NOT leave `just challenge` running, it
# starts and stops the sim itself.
#   just eval-cat1-sim arabic_room 2
#   just eval-cat2-sim "arabic_room chinese_room"
[group('eval')]
[doc('Orchestrated category-1 eval against the live sim (TARE explores; host-side)')]
eval-cat1-sim scene="all" limit="0" target_source="gt" report="/data/runs/challenge_report_sim_cat1.json":
    python3 scripts/eval/run_sim_sweep.py --category 1 --scenes {{scene}} \
      --limit {{limit}} --target-source {{target_source}} \
      --report "{{report}}"

[group('eval')]
[doc('Orchestrated category-2 eval against the live sim (TARE explores; host-side)')]
eval-cat2-sim scene="all" limit="0" mode="hybrid" target_source="gt" report="/data/runs/challenge_report_sim_cat2.json":
    python3 scripts/eval/run_sim_sweep.py --category 2 --scenes {{scene}} \
      --limit {{limit}} --mode {{mode}} --target-source {{target_source}} \
      --report "{{report}}"

# Phase 1 of the two-phase VLM comparison: eval-cat1 minus the counting call, so it
# costs one cheap text-only extraction per question instead of a 3-image one.
# Hours for all 15 scenes -- run it in tmux. Accuracy in the report it writes is
# meaningless (every prediction is a placeholder); the report is a cache index.
# target_source=vlm is the point: the model picks the SAM prompts, as it must on a
# scored run where there is no ground truth to hand it.
# Resumable: re-running keeps every question whose crops are already on disk, so an
# interruption costs one question rather than the whole sweep. To force a full rebuild,
# delete the cache= report first.
# vqa.yaml's view_source only matters here insofar as it decides what lands under
# crops/silhouette/ for bench-cat1 to later replay against; crops_only skips the
# answering call either way.
[group('eval')]
[doc('Generate and save best-view crops per question, without answering (cache builder)')]
cache-cat1 scene="all" limit="0" speed="0.1" target_source="vlm" cache="/data/runs/views_cache.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p crops_only:=true -p resume:=true -p report_file:={{cache}}"

# Phase 2: answer-only replay over those crops. No TARE, no SAM, no bag -- minutes per
# model instead of hours, and every model sees byte-identical images, which is what
# makes the comparison about the model. cache= is any sweep's report, cache-cat1 or not.
#   just bench-cat1 /data/runs/views_cache.json 3 all 0 /data/runs/bench_gemini.json
# Which model answers comes from `provider` / `model` in
# ai_module/src/captioner/config/vqa.yaml, so comparing two providers is: edit vqa.yaml,
# re-run with a different report=. The summary records which model produced the numbers.
# Only the counting step is replayed -- a model that would have extracted different SAM
# targets needs a full `just cache-cat1` of its own.
# The cache paths still say `views`: they name crops already on disk from earlier sweeps,
# and renaming the default would hide 14 scenes of them from the recipe that reads them.
# vqa.yaml's view_source picks which of the crop's saved copies is replayed -- silhouette
# (default) or the plain crop -- independent of which one the original cache-cat1 sweep
# answered with, since crops_only never calls the model either way.
[group('eval')]
[doc('Benchmark a VLM against the cached best views (no SAM, no bag)')]
bench-cat1 cache="/data/runs/views_cache.json" views="3" scene="all" limit="0" report="/data/runs/cat1_bench_report.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm cat1_bench --cache {{cache}} --views {{views}} --scene {{scene}} --limit {{limit}} --report {{report}}"

# The category-2 half of the same two-phase split, and the same warning: 122 questions at
# roughly 2.5 minutes each is about 5.5 hours, so run it in tmux. Resumable question by
# question. What it caches is the 3D map as much as the crops -- a reference question is
# answered by choosing one entry of obj_map.json, so a run whose map never landed is a hole
# in the cache and gets replayed rather than counted as done.
[group('eval')]
[doc('Cache maps + crops for every category-2 question, without answering')]
cache-cat2 scene="all" limit="0" speed="0.1" target_source="vlm" cache="/data/runs/cat2_cache.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p category:=2 -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p crops_only:=true -p resume:=true -p report_file:={{cache}}"

# Phase 2 for category 2, the same shape as bench-cat1: replay the selection step over
# those cached maps and crops. Seconds per mode for solver and naive, which ask no model,
# so this is the loop a change to selection is measured in.
#   just bench-cat2 hybrid    # what ships: solver, model only where the geometry is not decisive
#   just bench-cat2 solver    # no model call at all
#   just bench-cat2 naive     # the floor: largest instance of the named class
# Every row carries ceiling_score -- twice the best IoU reachable against the cached boxes --
# so one run says whether a low score is selection or perception. Give a separate report=
# when A/B-ing two modes, or the second run overwrites the first.
[group('eval')]
[doc('Score category-2 object selection over the cached maps (no SAM, no bag)')]
bench-cat2 mode="hybrid" scene="all" limit="0" cache="/data/runs/cat2_cache.json" report="/data/runs/cat2_bench_report.json":
    docker exec -it iros2026_ai_module bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm cat2_bench --mode {{mode}} --scene {{scene}} --limit {{limit}} --cache {{cache}} --report {{report}}"

# Text-only extract call the live reasoners use to arm SAM, scored against each
# question's target_objects. `flags` go straight to extract_bench (see --help).
#   just eval-target-extract
#   just eval-target-extract --category 3 --scene arabic_room
#   just eval-target-extract --category 1 --scene arabic_room --limit 2
# Which model answers is vqa.yaml's `target_extract_backend` + `provider` / `model_lite`,
# not a flag here. Give a separate --report when A/B-ing two configurations.
[group('eval')]
[doc('Score target extraction against benchmark target_objects (text only, no SAM)')]
eval-target-extract *flags:
    docker exec -it iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && \
       export PYTHONPATH={{ai_src}}/src/smart_vlm:{{ai_src}}/src/captioner:\$PYTHONPATH && \
       python3 -m smart_vlm.extract_bench {{flags}}"

# Requires vqa-up + run-sam + cat1-reasoner already running in other terminals.
#   just cat1-bag-bench arabic_room 3
#   just cat1-bag-bench arabic_room 0 "Q01 Q02 Q03"
# `tag` suffixes the output dir. The bench does not start the reasoner, so nothing in the
# results says which backend answered -- tag the run or an A/B overwrites its own baseline.
[group('cat1')]
[doc('Score category-1 questions against a recorded scene bag')]
cat1-bag-bench scene="arabic_room" limit="0" ids="" speed="1.0" tag="":
    #!/usr/bin/env bash
    set -euo pipefail
    # Support both `just cat1-bag-bench arabic_room 3` and key=value forms.
    scene="{{scene}}"
    limit="{{limit}}"
    ids="{{ids}}"
    speed="{{speed}}"
    tag="{{tag}}"
    scene="${scene#scene=}"
    limit="${limit#limit=}"
    ids="${ids#ids=}"
    speed="${speed#speed=}"
    tag="${tag#tag=}"
    # If a caller passed ids=... as the positional limit slot, recover.
    if [[ "$limit" == ids=* ]]; then
      ids="${limit#ids=}"
      limit="0"
    fi
    qa="/data/benchmark/${scene}/category_1/${scene}_category1_qa.json"
    out="/data/runs/cat1_${scene}${tag:+_$tag}"
    # Repo scripts/ is bind-mounted read-only at /home/docker/scripts; /data/runs
    # is created and made writable by the init one-shot (docker/compose.yml).
    script="/home/docker/scripts/eval/run_cat1_bag_bench.py"
    extra=""
    if [[ -n "$ids" ]]; then
      extra="--ids ${ids}"
    fi
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && python3 $(printf '%q' "$script") --qa $(printf '%q' "$qa") --scene $(printf '%q' "$scene") --out $(printf '%q' "$out") --limit $(printf '%q' "$limit") --speed $(printf '%q' "$speed") ${extra}"

# ---------- Category-2 (object reference) benchmark ---------------------
# gen/verify/pdf-assets run on the HOST: they only read bags/<scene>/iref_vla_metadata
# and questions/, and write data/benchmark + data/pdf_assets. Seconds, not hours --
# unlike the category-1 view cache there is no SAM or VLM in this loop, so
# regenerating the whole benchmark is cheap and is the intended way to change it.
# `just visibility` is the exception: it needs the bags and ros2, so it runs in the
# container, and its committed report is what lets gen-cat2 stay host-only.
#   just gen-cat2                     # all 13 scenes with referential statements
#   just gen-cat2 "--scenes loft -v"
# Hand corrections belong in scripts/bench/category2_overrides.json (pin / reword /
# drop / hide), never in the generated QA files: the next gen-cat2 would overwrite them.
[group('cat2')]
[doc('Generate the category-2 object-reference benchmark from IRef-VLA metadata')]
gen-cat2 args="":
    python3 scripts/bench/generate_category2_qa.py {{args}}

# Re-derives every answer box from the scene metadata, re-checks every relation, and
# re-solves every question -- generated as well as official -- from its own text. Non-zero
# exit on any mismatch, so it is safe to gate on. Run it after gen-cat2 and after any
# metadata refresh.
[group('cat2')]
[doc('Audit the category-2 benchmark against the scene metadata')]
verify-cat2 args="":
    python3 scripts/eval/verify_category2.py {{args}}

# The screenshots the challenge PDFs show next to each question, which are the only
# visual reference for reviewing category-2 answers (the expected object is outlined).
# Output is untracked; re-run after questions/ changes.
[group('cat2')]
[doc('Extract questions.pdf images and text into data/pdf_assets')]
pdf-assets args="":
    python3 scripts/bench/extract_pdf_assets.py {{args}}

# Measures which IRef-VLA boxes the robot's camera actually resolved, by projecting them
# into the recorded /camera/image frames (see docs/cat2_benchmark.md "Visibility gate").
# ~35 s/scene, needs the scene bag. The JSON reports are committed -- gen-cat2 and
# verify-cat2 read them and must not need a bag -- while the annotated crops under
# data/crops/visibility are untracked review material. Re-run after a camera-model or
# threshold change, then re-run gen-cat2: dropping a scene's report silently disables
# the gate, and both tools say so when it is missing.
[group('cat2')]
[doc("Measure which objects the robot's camera saw, per scene")]
visibility scene="all" args="":
    #!/usr/bin/env bash
    set -euo pipefail
    docker exec iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && python3 /home/docker/scripts/eval/object_visibility.py --scene $(printf '%q' "{{scene}}") {{args}}"
    for report in data/runs/visibility/*_visibility.json; do
      s="$(basename "$report" _visibility.json)"
      mkdir -p "data/benchmark/${s}/visibility"
      cp "$report" "data/benchmark/${s}/visibility/"
    done
    echo "reports copied to data/benchmark/<scene>/visibility/ -- now run: just gen-cat2"

# Host-only, no docker/ROS2 -- reads the mcap directly (`pip install mcap mcap-ros2-support`,
# see scripts/utils/mcap_io.py) and re-projects each category-2 target/anchor's own 8
# corners onto the camera frame object_visibility.py already picked as its best view.
# Catches a wrong yaw or a frame mixup that the axis-aligned crops under
# data/crops/visibility/ would not show. Output: data/crops/bbox_check/<scene>/*_{full,crop}.png.
[group('cat2')]
[doc('Draw each category-2 GT box on its best camera frame, for visual/numeric review')]
check-bboxes scene="arabic_room" args="":
    python3 scripts/eval/project_bboxes.py --scene {{scene}} {{args}}

# Also host-only. Copies the scene bag's camera/lidar/odometry through untouched and adds
# a /gt_boxes MarkerArray (green = visibility gate passed, orange = did not), republished
# at every lidar frame so scrubbing Foxglove's timeline always shows the boxes. Default
# --objects qa keeps it to what the category-2 QA file names; pass "--objects all" for a
# full-scene dump. Output is untracked (data/runs/foxglove/<scene>_gt_boxes.mcap).
[group('cat2')]
[doc('Build a scene mcap + GT-box MarkerArray to open in Foxglove')]
bbox-mcap scene="arabic_room" args="":
    python3 scripts/eval/make_bbox_mcap.py --scene {{scene}} {{args}}
