#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE="$( cd "$SCRIPT_DIR/../.." && pwd )"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1

source "$AI_MODULE/install/setup.bash"
ros2 run language_planner language_planner_node --platform mecanum &
sleep 5
