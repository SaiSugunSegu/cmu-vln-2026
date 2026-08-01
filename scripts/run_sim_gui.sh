#!/bin/bash
# Launch the CMU-VLN simulator for headless / VNC (browser) viewing.
#
# Rendering split:
#   * Unity environment (Model.x86_64) renders on the NVIDIA GPU via VirtualGL's
#     EGL backend (vglrun -d egl0).
#   * RVIZ runs on the VNC display's software GL directly and is deliberately
#     NOT wrapped with vglrun -- routing RVIZ through VirtualGL to TigerVNC
#     stalls its continuous repaint (the window freezes while ROS keeps running).
#
# Usage (inside the iros2026_system container, on the VNC display):
#   DISPLAY=:1 bash /home/docker/run_sim_gui.sh
set -e

export DISPLAY=${DISPLAY:-:1}
cd /home/docker/autonomy_stack_mecanum_wheel_platform
source /opt/ros/jazzy/setup.bash
source ./install/setup.bash

# Make sure VirtualGL selects the NVIDIA EGL device (not Mesa/software).
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 1) Unity environment on the GPU.
vglrun -d egl0 ./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3

# 2) Autonomy + simulator nodes.
ros2 launch vehicle_simulator system_simulation.launch &
sleep 2

# 3) RVIZ on software GL (foreground; Ctrl-C stops it). NOT under vglrun.
exec ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
