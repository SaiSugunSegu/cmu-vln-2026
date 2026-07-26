#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE="$( cd "$SCRIPT_DIR/../.." && pwd )"

source /opt/ros/humble/setup.bash
source "$AI_MODULE/install/setup.bash"
ros2 launch language_planner sort3d_gt_semantics_launch.xml
