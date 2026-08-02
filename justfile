# Common ops for CMU-VLN. Run from anywhere in the repo: `just <recipe>`
# Requires https://github.com/casey/just  (e.g. `cargo install just` or `sudo snap install just`)
#
# Examples:
#   just up                 # GPU compose (baked ai_module image)
#   just up-dev             # + bind-mount host ai_module for live code edits
#   just sim-noviz
#   just foxglove
#   just teleop
#   just ask "How many books are on the sofa"
#   just bag scene01_run1
#   just caption
#   just caption crops captions
#   just vqa-up             # load Qwen once inside the AI container
#   just vqa-ask "How many pillows are on the bed?" "/abs/path/image.png"

vgl := "cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl"
# Container path (compose_dev mounts host ai_module here).
ai_src := "/home/docker/ai_module"

default:
    @just --list

# Build + start both containers (GPU). Pure image — ai_module is NOT bind-mounted.
[working-directory: 'docker']
up:
    docker compose -f compose_gpu.yml up --build -d

# Same as up, but mount host ../ai_module → /home/docker/ai_module (daily dev).
[working-directory: 'docker']
up-dev:
    docker compose -f compose_gpu.yml -f compose_dev.yml up --build -d

# Start containers if needed without forcing an image rebuild.
[working-directory: 'docker']
up-dev-fast:
    docker compose -f compose_gpu.yml -f compose_dev.yml up -d

[working-directory: 'docker']
down:
    docker compose -f compose_gpu.yml -f compose_dev.yml down

# Run this after `up-dev`, after pulling, or whenever a launch reports "package '<name>'
# not found": with compose_dev bind-mounting the host tree, install/ is whatever your last
# --packages-select left behind, so a package present in src/ can be missing from install/.
# src/semantic_mapper is a plain pyproject package, not ament — colcon skips it, as intended.
# Build the whole ai_module workspace in the container (or `just build sam_mapper` for one).
build pkgs="":
    docker exec iros2026_ai_module bash -lc "\
      pip install 'setuptools>=68,<80' --break-system-packages -q && \
      source /opt/ros/jazzy/setup.bash && \
      cd {{ai_src}} && \
      colcon build --symlink-install {{ if pkgs != '' { '--packages-select ' + pkgs } else { '' } }}"

# Simulator + base autonomy + rviz2 (blocks; terminal A)
# Override display: just sim :0
sim sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./system_simulation.sh"

# Same as sim but without rviz2 (use with Foxglove)
sim-noviz sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./system_simulation_noviz.sh"

# Sim behind the 6-topic domain firewall, no rviz2
challenge sim_display=":1":
    docker exec -it -e DISPLAY={{sim_display}} iros2026_system bash -c "{{vgl}} ./challenge_simulation.sh --noviz"

# Play Ros Bag from a scene. Single pass by default; loop:=true to loop (e.g. just bag-play scene_0 1.0 true)
bag-play scene="scene_0" speed="1.0" loop="false":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm bag_replay.launch scene:={{scene}} speed:={{speed}} loop:={{loop}}"
    
# smart_vlm (blocks; terminal B)
ai:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"

# SAM 3 2D detector. /camera/image in -> /annotated_image, /sam3/instance_map, /sam3/detections out.
run-sam config="sam3_mecanum_sim.yaml":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch sam_mapper sam_node.launch config:={{config}}"

# Lidar fusion / 3D mapping node 
run-map config="sam3_mecanum_sim.yaml":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch sam_mapper map_node.launch config:={{config}}"

# Dump /camera/image frames from a bag to PNGs, for the offline backend probe. Use out=/data/bags/_frames to see them on the host.
sam-frames scene="scene_0" out="/data/bags/_frames" limit="40":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.dump_frames --bag /data/bags/{{scene}} --out {{out}} --limit {{limit}}"

# Offline SAM 3 probe on dumped frames: are object IDs stable across frames, and which
# image_size is fastest? Answers whether dropping ByteTrack was sound.
sam-probe frames="/tmp/frames" config="sam3_mecanum_sim.yaml" args="--sweep-image-size":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.sam3_backend --frames {{frames}} --config /home/docker/ai_module/src/sam_mapper/config/{{config}} {{args}}"

# Does runtime scale with OBJECT COUNT? Times all prompts vs instances-only vs 3 classes.
# If ms/frame falls with fewer objects while ms/object stays flat, per-object memory
# attention is the bottleneck -> cut prompts, and SAM 3.1 Object Multiplex is the real fix.
sam-prompts frames="/tmp/frames" config="sam3_mecanum_sim.yaml":
    just sam-probe {{frames}} {{config}} --sweep-prompts

# Offline Qwen captioner (host data/ <-> /data/workspace). Ex: just caption crops captions
# Weights: host ~/.cache/huggingface mounted at /home/docker/.cache/huggingface
# Optional knobs (env or key=value positionals for older just):
#   just caption crops captions
#   BATCH_SIZE=8 QUANTIZATION=int4 just caption crops captions
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
    [[ "$in"  == /* ]] || in="/data/workspace/${in}"
    [[ "$out" == /* ]] || out="/data/workspace/${out}"
    # Host paths under the repo data/ mount are rewritten to the container path.
    in="${in/#$PWD\/data//data/workspace}"
    out="${out/#$PWD\/data//data/workspace}"
    # Container user is uid 1001 and cannot create dirs in host-owned 755 trees.
    # Prep the output dir on the host and make it writable by the container user.
    # (Running as host uid fails because /home/docker is mode 750.)
    mkdir -p "${out}"
    chmod a+rwx "${out}"
    # PYTHONUTF8: native libs loaded during model init can reset the C locale, which
    # makes Python's default text encoding ascii and breaks writing captions that
    # contain curly quotes. UTF-8 mode ignores the locale entirely.
    docker exec -it -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      source /home/docker/ai_module/install/setup.bash &&
      export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
      caption_crops '${in}' \
        --output_dir '${out}' \
        --captioning_model '${model}' \
        --quantization '${quantization}' \
        --batch_size ${batch_size}
    "

# Publish a one-shot challenge question (challenge: just ask "…" 42)
ask q domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '{{q}}'}\""

# Foxglove bridge on host port 8765 (blocks; tunnel from laptop)
# Default domain 0 matches sim-noviz. Challenge mode: just foxglove 42
foxglove domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_ai_module bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 address:=0.0.0.0"

# Keyboard drive via /joy (needs TTY focus; domain must match sim)
# Default domain 0 matches sim-noviz. Challenge mode: just teleop 42
teleop domain="0":
    docker exec -it -e ROS_DOMAIN_ID={{domain}} iros2026_system bash -lc \
      'source /opt/ros/jazzy/setup.bash && \
       source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && \
       python3 /home/docker/scripts/keyboard_teleop.py'

# Check the list of topics available
topics:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic list"

# Record a bag under /data/bags/<name>
bag name:
    docker exec -it iros2026_ai_module bash -c "ros2 bag record /camera/image /registered_scan /sensor_scan /terrain_map /terrain_map_ext /state_estimation /tf /tf_static /challenge_question -o /data/bags/{{name}}"

# List all 18 Unity scenes in the Drive source folder: 15 training (in questions/)
# + 3 held-out test scenes (README.md:224). Static list -- the folder is a public
# link, not in this account's own Drive, so it can't be queried live.
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

# Enter into iros2026_system container
shell-sys:
    docker exec -it iros2026_system bash

# Enter into iros2026_ai_module container
shell-ai:
    docker exec -it iros2026_ai_module bash

# What has the mapper actually built? One-shot JSON dump of every 3D instance.
sam-map-json:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 topic echo /obj_map_json --once"

# Is sam_mapper working? Rates on every output topic + a summary of the 3D map.
sam-status seconds="15":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && python3 -m sam_mapper.tools.status --seconds {{seconds}}"

# ---------- Persistent Qwen VQA (model stays loaded) --------------------
#   just vqa-up
#   just vqa-ask "How many pillows?" /data/workspace/img.png
#   just vqa-down

# Start compose (dev mount), rebuild captioner, load Qwen once (~60s first time).
vqa-up model="qwen3vl" quantization="int4":
    #!/usr/bin/env bash
    set -euo pipefail
    model="{{model}}"
    quantization="{{quantization}}"
    model="${model#model=}"
    quantization="${quantization#quantization=}"
    just up-dev-fast
    # ament_python --symlink-install needs setuptools<80 (83 drops --editable).
    docker exec iros2026_ai_module bash -lc "
      pip install 'setuptools>=68,<80' --break-system-packages -q
      source /opt/ros/jazzy/setup.bash &&
      cd {{ai_src}} &&
      colcon build --symlink-install --packages-select captioner
    "
    # [q]wen… so pkill's cmdline does not match itself.
    docker exec iros2026_ai_module bash -lc \
      "pkill -f '[q]wen_vqa_server --ros-args' || true" >/dev/null 2>&1 || true
    sleep 1
    docker exec -d -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      export PATH={{ai_src}}/install/captioner/lib/captioner:\$PATH &&
      : > /tmp/qwen_vqa_server.log &&
      setsid nohup qwen_vqa_server --ros-args \
        -p captioning_model:=${model} \
        -p quantization:=${quantization} \
        >> /tmp/qwen_vqa_server.log 2>&1 < /dev/null &
    "
    echo "Waiting for Qwen VQA server (loading weights)…"
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      export PATH={{ai_src}}/install/captioner/lib/captioner:\$PATH &&
      qwen_vqa_wait_ready --timeout 600
    "

# Image path: container-absolute under $HOME or /data/workspace.
vqa-ask q image:
    #!/usr/bin/env bash
    set -euo pipefail
    q="{{q}}"
    image="{{image}}"
    q="${q#q=}"
    image="${image#image=}"
    # Rewrite repo-relative data/ paths to the container mount.
    if [[ "$image" != /* ]]; then
      if [[ -f "$PWD/data/$image" ]]; then
        image="/data/workspace/$image"
      elif [[ -f "$image" ]]; then
        image="$(cd "$(dirname "$image")" && pwd)/$(basename "$image")"
      fi
    fi
    image="${image/#$PWD\/data//data/workspace}"
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc "
      if ! pgrep -f '[q]wen_vqa_server --ros-args' >/dev/null; then
        echo 'VQA server is not running. Start it with: just vqa-up' >&2
        exit 1
      fi
      source {{ai_src}}/install/setup.bash &&
      export PATH={{ai_src}}/install/captioner/lib/captioner:\$PATH &&
      qwen_vqa_ask --question $(printf '%q' "$q") --image $(printf '%q' "$image")
    "

# Show server status (ready / loading) without reloading the model.
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

# Stop the persistent VQA server (keeps the container running).
vqa-down:
    #!/usr/bin/env bash
    set -euo pipefail
    docker exec iros2026_ai_module bash -lc "pkill -f '[q]wen_vqa_server --ros-args' || true"
    echo "VQA server stopped."

# ---------- Category-1 bag bench (SAM best-views + Qwen VQA) ---------------
# Terminals: just vqa-up | just run-sam | just cat1-reasoner | just cat1-bag-bench
# Bench waits for /sam3/status=ready before (and after) prompts, then starts the bag.
#
# Rebuilds smart_vlm (symlink) and launches category1_reasoner.
cat1-reasoner:
    #!/usr/bin/env bash
    set -euo pipefail
    docker exec iros2026_ai_module bash -lc "
      pip install 'setuptools>=68,<80' --break-system-packages -q
      source /opt/ros/jazzy/setup.bash &&
      cd {{ai_src}} &&
      colcon build --symlink-install --packages-select smart_vlm sam_mapper
    "
    docker exec -it iros2026_ai_module bash -lc "
      source {{ai_src}}/install/setup.bash &&
      ros2 launch smart_vlm category1_reasoner.launch
    "

# Loop category-1 QA over a scene bag. Requires vqa-up + run-sam + cat1-reasoner.
# Ex: just cat1-bag-bench arabic_room 3
# Ex: just cat1-bag-bench arabic_room 0 "Q01 Q02 Q03"
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
    qa="/data/workspace/benchmark/${scene}/category_1/${scene}_category1_qa.json"
    out="/data/workspace/runs/cat1_${scene}"
    mkdir -p "{{justfile_directory()}}/data/runs"
    chmod a+rwx "{{justfile_directory()}}/data/runs" 2>/dev/null || true
    # Repo scripts/ is not bind-mounted into ai_module; $HOME is.
    script="{{justfile_directory()}}/scripts/eval/run_cat1_bag_bench.py"
    extra=""
    if [[ -n "$ids" ]]; then
      extra="--ids ${ids}"
    fi
    docker exec -e PYTHONUTF8=1 iros2026_ai_module bash -lc \
      "source {{ai_src}}/install/setup.bash && python3 $(printf '%q' "$script") --qa $(printf '%q' "$qa") --scene $(printf '%q' "$scene") --out $(printf '%q' "$out") --limit $(printf '%q' "$limit") --speed $(printf '%q' "$speed") ${extra}"
