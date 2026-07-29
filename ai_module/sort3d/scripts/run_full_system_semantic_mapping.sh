#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE="$( cd "$SCRIPT_DIR/../.." && pwd )"
REPO_ROOT="$( cd "$AI_MODULE/.." && pwd )"
AUTONOMY="$REPO_ROOT/autonomy_stack_mecanum_wheel_platform"
SEMANTIC_MAPPER="$AI_MODULE/src/semantic_mapper"

source /opt/ros/humble/setup.bash
source "$AUTONOMY/install/setup.bash"
source "$AI_MODULE/install/setup.bash"

cd "$AUTONOMY"
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3
ros2 launch vehicle_simulator system_simulation_with_route_planner.launch > /dev/null 2>&1 &
sleep 1
cd "$AI_MODULE"
ros2 run rviz2 rviz2 -d src/language_planner/rviz/vehicle_simulator.rviz &
sleep 1

ros2 run language_planner language_planner_node --platform mecanum &
sleep 5

cd "$SEMANTIC_MAPPER"
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml --captioner_batch_size 16 --captioning_model qwen3vl
