#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE="$( cd "$SCRIPT_DIR/../.." && pwd )"
SEMANTIC_MAPPER="$AI_MODULE/src/semantic_mapper"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1

source "$AI_MODULE/install/setup.bash"
ros2 run language_planner language_planner_node --platform mecanum &
sleep 5

cd "$SEMANTIC_MAPPER"
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_real.yaml --captioner_batch_size 16
