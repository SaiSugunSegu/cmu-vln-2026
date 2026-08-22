# Common ops for CMU-VLN. Run from anywhere in the repo: `just <recipe>`
# Requires https://github.com/casey/just  (e.g. `cargo install just` or `sudo snap install just`)
#
# Run `just` with no arguments for the grouped recipe list.
#
# First run, in order:
#   just up          # build + start; bakes weights, HF_TOKEN, OpenRouter key
#
# Then pick a flow:
#   scored eval   just eval-cat1 arabic_room 2                # one command, per-question relaunch
#                 just eval-cat2 chinese_room 2               # the same for object reference
#   live-sim eval just eval-cat1-sim arabic_room 2            # TARE explores; host-side
#                 just eval-cat2-sim chinese_room 2
#   switch VLM    edit vqa.yaml `vlm_backend` / `target_extract_backend`  # then just up
#   compare VLMs  just cache-cat1   (once, hours) then just bench-cat1  (per model, minutes)
#                 just cache-cat2                then just bench-cat2  (per selection mode)
#   live sim      just sim          | just ai | just ask "How many …"
#   submit        just trial-submission-image                 # arabic_room Q01 in live sim
#                 just build-submission-image                 # bake AI image + checks
#                 just push-submission-image USER/cmu-vln-ai:v1

vgl := "cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl"
# The workspace path inside the image.
ai_src := "/home/docker/ai_module"
# `bash -lc` does not source ~/.bashrc for non-interactive shells, so recipes that
# invoke a captioner console script must put its install dir on PATH themselves.
capt_env := "source /home/docker/ai_module/install/setup.bash && export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH"
# just up writes this tag. just push-submission-image USER/name:tag retags and pushes.
submit_image := "iros2026_odyssey:submission"

# --unsorted keeps groups in justfile order (setup first, then the flows) instead
# of alphabetical, which would bury [setup] in the middle.
default:
    @just --list --unsorted

# The only way to start the stack. ai_module is never bind-mounted: the container
# always runs the source baked into the image, which is exactly what CI and the eval
# harness run. The corollary is that an ai_module edit does nothing until you re-run
# this — the rebuild IS how source lands in the container.
# Tags iros2026_odyssey:submission and bakes HF_TOKEN + the vqa.yaml provider key
# into image ENV so the running container is the Hub artifact.
[group('setup')]
[doc('Build + start both containers (GPU); bake weights and API keys')]
[working-directory: 'docker']
up:
    #!/usr/bin/env bash
    set -euo pipefail
    extra=()
    if [ -f ../.env ]; then extra+=(--env-file ../.env); fi
    docker compose "${extra[@]}" -f compose_gpu.yml build
    ../scripts/submit/wrap_image_keys.sh --tag {{submit_image}}
    docker compose "${extra[@]}" -f compose_gpu.yml up -d

[group('setup')]
[doc('Stop and remove both containers')]
[working-directory: 'docker']
down:
    docker compose -f compose_gpu.yml down

# Optional: pull a new checkpoint or warm SAM 3's cv-utils kernel in the running
# container. Setup itself is `just up` (HF_TOKEN in .env bakes sam3 / Qwen into
# the image). A checkpoint you want in the Hub image has to land via just up.
#   just hf-fetch                # all defaults
#   just hf-fetch "qwen3vl sam3" # a subset;  just hf-fetch --list  to see them
[group('setup')]
[doc('Refresh weights / SAM 3 kernels in the running container')]
hf-fetch models="":
    docker exec -e PYTHONUTF8=1 iros2026_odyssey bash -lc \
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
    docker exec -e PYTHONUTF8=1 iros2026_odyssey bash -lc "
      source {{ai_src}}/install/setup.bash &&
      python3 -m pytest $dirs -q -p no:cacheprovider
    "

# Bakes iros2026_odyssey:submission only (compose service odyssey). Does not
# build cmu-vln-system or start containers — that is `just up`.
[group('submit')]
[doc('Bake the AI image and run preflight checks (no compose up)')]
build-submission-image:
    ./scripts/submit/build_submission_image.sh --tag {{submit_image}}

# arabic_room Q01: "How many sofas are below a window?" (GT 2). Live sim with
# TARE exploring. Fetches the Unity mesh only if data/scenes/arabic_room is
# missing. just up first.
[group('submit')]
[doc('arabic_room Q01 in the live sim (TARE explores)')]
trial-submission-image:
    ./scripts/submit/trial_submission_image.sh --tag {{submit_image}}

[group('submit')]
[doc('Tag the local image and push to a registry (no rebuild)')]
push-submission-image tag:
    docker tag {{submit_image}} {{tag}}
    docker push {{tag}}

# Headless default: Xvfb :99 inside iros2026_system. Override: just sim :0
[group('sim')]
[doc('Simulator + base autonomy + rviz2 (blocks; terminal A)')]
sim sim_display="":
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{sim_display}}"
    if [ -z "$d" ]; then d="$(scripts/eval/ensure_xvfb.sh)"; fi
    docker exec -it -e DISPLAY="$d" -e XDG_RUNTIME_DIR=/tmp/runtime-docker \
      iros2026_system bash -c "{{vgl}} ./system_simulation.sh"

[group('sim')]
[doc('Simulator without rviz2 — pair with `just foxglove`')]
sim-noviz sim_display="":
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{sim_display}}"
    if [ -z "$d" ]; then d="$(scripts/eval/ensure_xvfb.sh)"; fi
    docker exec -it -e DISPLAY="$d" -e XDG_RUNTIME_DIR=/tmp/runtime-docker \
      iros2026_system bash -c "{{vgl}} ./system_simulation_noviz.sh"

[group('sim')]
[doc('Simulator behind the 6-topic eval firewall (domain 42, no rviz)')]
challenge sim_display="":
    #!/usr/bin/env bash
    set -euo pipefail
    d="{{sim_display}}"
    if [ -z "$d" ]; then d="$(scripts/eval/ensure_xvfb.sh)"; fi
    docker exec -it -e DISPLAY="$d" -e XDG_RUNTIME_DIR=/tmp/runtime-docker \
      iros2026_system bash -c "{{vgl}} ./challenge_simulation.sh --noviz"

# Challenge mode uses domain 42: just ask "…" 42
[group('sim')]
[doc('Publish a one-shot question on /challenge_question')]
ask q domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_odyssey bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '{{q}}'}\""

# Default domain 0 matches sim-noviz. Challenge mode: just teleop 42
[group('sim')]
[doc('Keyboard drive via /joy (needs TTY focus; domain must match sim)')]
teleop domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_system bash -lc \
      'source /opt/ros/jazzy/setup.bash && \
       source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && \
       python3 /home/docker/scripts/keyboard_teleop.py'

# Self-contained: when vqa.yaml has a local backend the launch starts its own Qwen
# server, so `just vqa-up` is not a prerequisite — and must not be running at the
# same time, or two servers collide on the node name and the /qwen_vqa topics.
# Use vqa-up only to keep a server resident across per-question relaunches.
# Brings up sam_node too (unarmed until a question supplies prompts).
[group('run')]
[doc('Official entry: dummy_vlm.launch → SAM + supervisor + reasoners + TARE')]
ai:
    docker exec -it iros2026_odyssey bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch dummy_vlm dummy_vlm.launch"

# Single pass by default; loop:=true to loop (e.g. just bag-play livingroom_1 1.0 true)
[group('bags')]
[doc('Replay a recorded scene bag instead of the live sim')]
bag-play scene="livingroom_1" speed="1.0" loop="false":
    docker exec -it iros2026_odyssey bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm bag_replay.launch scene:={{scene}} speed:={{speed}} loop:={{loop}}"

# Default domain 0 matches sim-noviz. Challenge mode: just foxglove 42
[group('debug')]
[doc('Foxglove bridge on host port 8765 (blocks; tunnel from laptop)')]
foxglove domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_odyssey bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8090 address:=0.0.0.0"

[group('debug')]
[doc('List all ROS topics visible in the AI container')]
topics:
    docker exec -it iros2026_odyssey bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic list"

[group('bags')]
[doc('Record the 6 allowed topics + tf into data/bags/<name>')]
bag name:
    docker exec -it iros2026_odyssey bash -c "ros2 bag record /camera/image /registered_scan /sensor_scan /terrain_map /terrain_map_ext /state_estimation /tf /tf_static /challenge_question -o /data/bags/{{name}}"

[group('debug')]
[doc('Interactive shell in the system container')]
shell-sys:
    docker exec -it iros2026_system bash

[group('debug')]
[doc('Interactive shell in the AI module container')]
shell-ai:
    docker exec -it iros2026_odyssey bash

# Optional resident Qwen server. `just ai` and the eval recipes start their own on
# the local backend — do not run this at the same time as those, or two servers
# collide on the node name and the /qwen_vqa topics. Use it to keep one server
# loaded across per-question relaunches (~60s first time).
[group('vqa')]
[doc('Load Qwen once and keep it resident (~60s first time)')]
vqa-up model="qwen3vl" quantization="int4":
    docker exec -it -e PYTHONUTF8=1 iros2026_odyssey bash -lc \
      "source {{ai_src}}/install/setup.bash && ros2 launch captioner vqa_server.launch model:={{model}} quantization:={{quantization}}"

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
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p report_file:={{report}}"

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
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p category:=2 -p scene:={{scene}} -p question_limit:={{limit}} -p cat2_mode:={{mode}} -p target_source:={{target_source}} -p speed:={{speed}} -p report_file:={{report}}"

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
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p crops_only:=true -p resume:=true -p report_file:={{cache}}"

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
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm cat1_bench --cache {{cache}} --views {{views}} --scene {{scene}} --limit {{limit}} --report {{report}}"

# The category-2 half of the same two-phase split, and the same warning: 122 questions at
# roughly 2.5 minutes each is about 5.5 hours, so run it in tmux. Resumable question by
# question. What it caches is the 3D map as much as the crops -- a reference question is
# answered by choosing one entry of obj_map.json, so a run whose map never landed is a hole
# in the cache and gets replayed rather than counted as done.
[group('eval')]
[doc('Cache maps + crops for every category-2 question, without answering')]
cache-cat2 scene="all" limit="0" speed="0.1" target_source="vlm" cache="/data/runs/cat2_cache.json":
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm eval_orchestrator --ros-args -p category:=2 -p scene:={{scene}} -p question_limit:={{limit}} -p target_source:={{target_source}} -p speed:={{speed}} -p crops_only:=true -p resume:=true -p report_file:={{cache}}"

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
    docker exec -it iros2026_odyssey bash -lc "source {{ai_src}}/install/setup.bash && ros2 run smart_vlm cat2_bench --mode {{mode}} --scene {{scene}} --limit {{limit}} --cache {{cache}} --report {{report}}"
