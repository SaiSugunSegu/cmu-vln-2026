#!/bin/bash
# system_simulation.sh without the foreground rviz2 — for Foxglove or headless
# runs, where the onboard rviz2 window is redundant and costs a CPU core.
# Launch through VirtualGL so Unity renders on the GPU:
#   vglrun -d egl ./system_simulation_noviz.sh

# This is a deliberate rviz-less variant of the submodule's system_simulation.sh.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ./install/setup.bash

./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
UNITY_PID=$!
trap 'kill -TERM ${UNITY_PID} 2>/dev/null || true' EXIT SIGINT SIGTERM

sleep 3
ros2 launch vehicle_simulator system_simulation.launch
