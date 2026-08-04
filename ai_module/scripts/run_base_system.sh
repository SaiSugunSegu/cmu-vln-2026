#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AI_MODULE="$( cd "$SCRIPT_DIR/../.." && pwd )"
REPO_ROOT="$( cd "$AI_MODULE/.." && pwd )"
AUTONOMY="$REPO_ROOT/autonomy_stack_mecanum_wheel_platform"

source /opt/ros/humble/setup.bash

cd "$AUTONOMY"
source ./install/setup.bash
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3
ros2 launch vehicle_simulator system_simulation_with_route_planner.launch > /dev/null 2>&1 &
sleep 1
cd "$AI_MODULE"
ros2 run rviz2 rviz2 -d src/language_planner/rviz/vehicle_simulator.rviz
sleep 1
