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
#   live sim      just sim          | just vqa-up && just ai | just ask "How many …"
#   offline bags  just vqa-up ; just run-sam ; just cat1-reasoner ; just cat1-bag-bench

vgl := "cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl"
# Container path (compose_dev mounts host ai_module here).
ai_src := "/home/docker/ai_module"
# `bash -lc` does not source ~/.bashrc for non-interactive shells, so recipes that
# invoke a captioner console script must put its install dir on PATH themselves.
capt_env := "source /home/docker/ai_module/install/setup.bash && export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH"

# --unsorted keeps groups in justfile order (setup first, then the flows) instead
# of alphabetical, which would bury [setup] in the middle.
default:
    @just --list --unsorted

# Pure image — ai_module is NOT bind-mounted, so this matches the CI/eval path.
[group('setup')]
[doc('Build + start both containers (GPU)')]
[working-directory: 'docker']
up:
    docker compose -f compose_gpu.yml up --build -d

[group('setup')]
[doc('Like up, but bind-mount host ai_module for live code edits (daily dev)')]
[working-directory: 'docker']
up-dev:
    docker compose -f compose_gpu.yml -f compose_dev.yml up --build -d

[group('setup')]
[doc('Start containers if needed, without forcing an image rebuild')]
[working-directory: 'docker']
up-dev-fast:
    docker compose -f compose_gpu.yml -f compose_dev.yml up -d

[group('setup')]
[doc('Stop and remove both containers')]
[working-directory: 'docker']
down:
    docker compose -f compose_gpu.yml -f compose_dev.yml down

# THE ONLY recipe that goes online — everything else runs with HF_HUB_OFFLINE=1 from
# the image. Passing it here via -e means the repo-root .env is never edited.
# Gated sam3 also needs HF_TOKEN in .env and the licence accepted on the model page.
#   just hf-fetch                # all defaults
#   just hf-fetch "qwen3vl sam3" # a subset;  just hf-fetch --list  to see them
[group('setup')]
[doc('ONE-TIME ~15-20 GB weight download (sam3, Qwen3-VL, CLIP). Run once after up')]
hf-fetch models="":
    docker exec -e HF_HUB_OFFLINE=0 -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "{{capt_env}} && fetch_weights {{models}}"

# Run this after `up-dev`, after pulling, or whenever a launch reports "package '<name>'
# not found": with compose_dev bind-mounting the host tree, install/ is whatever your last
# --packages-select left behind, so a package present in src/ can be missing from install/.
# src/semantic_mapper is a plain pyproject package, not ament — colcon skips it, as intended.
# setuptools<80 (ament_python --symlink-install needs --editable) is pinned in
# docker/requirements_captioner.txt and baked into the image — no runtime install.
[group('setup')]
[doc('colcon build the ai_module workspace (or `just build sam_mapper` for one pkg)')]
build pkgs="":
    docker exec iros2026_ai_module bash -lc "\
      source /opt/ros/jazzy/setup.bash && \
      cd {{ai_src}} && \
      colcon build --symlink-install {{ if pkgs != '' { '--packages-select ' + pkgs } else { '' } }}"

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
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      python3 -m pytest $dirs -q
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

# Single pass by default; loop:=true to loop (e.g. just bag-play scene_0 1.0 true)
[group('bags')]
[doc('Replay a recorded scene bag instead of the live sim')]
bag-play scene="scene_0" speed="1.0" loop="false":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm bag_replay.launch scene:={{scene}} speed:={{speed}} loop:={{loop}}"

# Needs the VQA server up first (`just vqa-up`): the numerical head is a client of it.
[group('run')]
[doc('smart_vlm: supervisor + answer heads + TARE (blocks; terminal B)')]
ai:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"

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
[group('perception')]
[doc('Dump /camera/image frames from a bag to PNGs, for the offline probe')]
sam-frames scene="scene_0" out="/data/bags/_frames" limit="40":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.dump_frames --bag /data/bags/{{scene}} --out {{out}} --limit {{limit}}"

# Are object IDs stable across frames, and which image_size is fastest?
# Answers whether dropping ByteTrack was sound.
[group('perception')]
[doc('Offline SAM 3 probe on dumped frames: ID stability + image_size sweep')]
sam-probe frames="/tmp/frames" config="sam3_mecanum_sim.yaml" args="--sweep-image-size":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.sam3_backend --frames {{frames}} --config /home/docker/ai_module/src/sam_mapper/config/{{config}} {{args}}"

# Times all prompts vs instances-only vs 3 classes. If ms/frame falls with fewer objects
# while ms/object stays flat, per-object memory attention is the bottleneck -> cut
# prompts, and SAM 3.1 Object Multiplex is the real fix.
[group('perception')]
[doc('Does SAM 3 runtime scale with object count? Prompt-count sweep')]
sam-prompts frames="/tmp/frames" config="sam3_mecanum_sim.yaml":
    just sam-probe {{frames}} {{config}} --sweep-prompts

# Challenge mode uses domain 42: just ask "…" 42
[group('sim')]
[doc('Publish a one-shot question on /challenge_question')]
ask q domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '{{q}}'}\""

# Default domain 0 matches sim-noviz. Challenge mode: just foxglove 42
[group('debug')]
[doc('Foxglove bridge on host port 8765 (blocks; tunnel from laptop)')]
foxglove domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 address:=0.0.0.0"

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

[group('perception')]
[doc('One-shot JSON dump of every 3D instance the mapper has built')]
sam-map-json:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic echo /obj_map_json --once"

[group('perception')]
[doc('Is sam_mapper working? Topic rates + a summary of the 3D map')]
sam-status seconds="15":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.status --seconds {{seconds}}"

# ---------- Persistent Qwen VQA (model stays loaded) --------------------
#   just vqa-up
#   just vqa-ask "How many pillows?" /data/img.png
#   just vqa-down

# Starts compose (dev mount), rebuilds captioner, then loads Qwen once. Every other
# component that needs a VLM talks to this one process, so run it before `just ai`
# and before the cat1 flow.
[group('vqa')]
[doc('Load Qwen once and keep it resident (~60s first time)')]
vqa-up model="qwen3vl" quantization="int4":
    #!/usr/bin/env bash
    set -euo pipefail
    model="{{model}}"
    quantization="{{quantization}}"
    model="${model#model=}"
    quantization="${quantization#quantization=}"
    just up-dev-fast
    just build captioner
    # [q]wen… so pkill's cmdline does not match itself.
    docker exec iros2026_ai_module bash -lc \
      "pkill -f '[q]wen_vqa_server --ros-args' || true" >/dev/null 2>&1 || true
    sleep 1
    docker exec -d -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      {{capt_env}} &&
      : > /tmp/qwen_vqa_server.log &&
      setsid nohup qwen_vqa_server --ros-args \
        -p captioning_model:=${model} \
        -p quantization:=${quantization} \
        >> /tmp/qwen_vqa_server.log 2>&1 < /dev/null &
    "
    echo "Waiting for Qwen VQA server (loading weights)…"
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      {{capt_env}} &&
      qwen_vqa_wait_ready --timeout 600
    "

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
      if ! pgrep -f '[q]wen_vqa_server --ros-args' >/dev/null; then
        echo 'VQA server is not running. Start it with: just vqa-up' >&2
        exit 1
      fi
      {{capt_env}} &&
      qwen_vqa_ask --question $(printf '%q' "$q") --image $(printf '%q' "$image")
    "

[group('vqa')]
[doc('Is the VQA server up? Shows ready / loading without reloading')]
vqa-status:
    #!/usr/bin/env bash
    set -euo pipefail
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      if ! pgrep -f '[q]wen_vqa_server --ros-args' >/dev/null; then
        echo 'VQA server is not running. Start it with: just vqa-up' >&2
        echo 'Last log lines:' >&2
        tail -n 20 /tmp/qwen_vqa_server.log 2>/dev/null >&2 || true
        exit 1
      fi
      source {{ai_src}}/install/setup.bash
      timeout 5 ros2 topic echo /qwen_vqa/status std_msgs/msg/String --once
    "

[group('vqa')]
[doc('Stop the VQA server (container keeps running)')]
vqa-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker exec iros2026_ai_module bash -lc "pkill -f '[q]wen_vqa_server --ros-args' || true"
    echo "VQA server stopped."

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
[group('cat1')]
[doc('Rebuild smart_vlm/sam_mapper and launch the category-1 reasoner (blocks)')]
cat1-reasoner:
    #!/usr/bin/env bash
    set -euo pipefail
    # captioner too: smart_vlm imports captioner.paths / .text_utils / .ros_utils.
    just build "captioner smart_vlm sam_mapper"
    docker exec -it iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      ros2 launch smart_vlm category1_reasoner.launch
    "

# Requires vqa-up + run-sam + cat1-reasoner already running in other terminals.
#   just cat1-bag-bench arabic_room 3
#   just cat1-bag-bench arabic_room 0 "Q01 Q02 Q03"
[group('cat1')]
[doc('Score category-1 questions against a recorded scene bag')]
cat1-bag-bench scene="arabic_room" limit="0" ids="" speed="1.0":
    #!/usr/bin/env bash
    set -euo pipefail
    # Support both `just cat1-bag-bench arabic_room 3` and key=value forms.
    scene="{{scene}}"
    limit="{{limit}}"
    ids="{{ids}}"
    speed="{{speed}}"
    scene="${scene#scene=}"
    limit="${limit#limit=}"
    ids="${ids#ids=}"
    speed="${speed#speed=}"
    # If a caller passed ids=... as the positional limit slot, recover.
    if [[ "$limit" == ids=* ]]; then
      ids="${limit#ids=}"
      limit="0"
    fi
    qa="/data/benchmark/${scene}/category_1/${scene}_category1_qa.json"
    out="/data/runs/cat1_${scene}"
    # Repo scripts/ is bind-mounted read-only at /home/docker/scripts; /data/runs
    # is created and made writable by the init one-shot (docker/compose.yml).
    script="/home/docker/scripts/eval/run_cat1_bag_bench.py"
    extra=""
    if [[ -n "$ids" ]]; then
      extra="--ids ${ids}"
    fi
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && python3 $(printf '%q' "$script") --qa $(printf '%q' "$qa") --scene $(printf '%q' "$scene") --out $(printf '%q' "$out") --limit $(printf '%q' "$limit") --speed $(printf '%q' "$speed") ${extra}"
