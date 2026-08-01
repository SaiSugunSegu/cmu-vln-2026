# Common ops for CMU-VLN. Run from anywhere in the repo: `just <recipe>`
# Requires https://github.com/casey/just  (e.g. `cargo install just` or `sudo snap install just`)
#
# Examples:
#   just up
#   just sim-noviz
#   just foxglove
#   just teleop
#   just ask "How many books are on the sofa"
#   just bag scene01_run1

vgl := "cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl"

default:
    @just --list

# Build + start both containers (GPU)
[working-directory: 'docker']
up:
    docker compose -f compose_gpu.yml up --build -d

[working-directory: 'docker']
down:
    docker compose -f compose_gpu.yml down

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

# Play Ros Bag from a scene
bag-play scene="scene_0":
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm bag_replay.launch scene:={{scene}}"
    
# smart_vlm (blocks; terminal B)
ai:
    docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"

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
