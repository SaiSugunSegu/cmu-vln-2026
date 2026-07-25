#!/usr/bin/env bash
# challenge_simulation.sh — run the sim under EVAL-LIKE conditions.
#
# Design: at eval, the organizers run the FULL system (it publishes extra topics
# like /overall_map, camera/semantic_image/*). The restriction is on what OUR
# module CONSUMES. Stripping sim publishers is fragile; instead we build a hard
# firewall with ROS domains:
#
#   domain 42 : full simulator + base autonomy (publishes everything)
#   bridge    : relays ONLY the challenge topics (both directions)
#   domain 0  : our AI module — physically cannot see anything else
#
# Prereq: ros-jazzy-domain-bridge, installed by docker/system/Dockerfile.
# Usage:
#   Terminal A (system container):  vglrun -d egl ./challenge_simulation.sh [--noviz]
#                                   --noviz / -n  skip rviz2 (use with Foxglove)
#   Terminal B (ai container):      export ROS_DOMAIN_ID=0
#                                   ros2 launch smart_vlm smart_vlm.launch
#   Question:  publish /challenge_question in DOMAIN 42 (it gets bridged in),
#              mimicking the eval node living on the system side:
#              ROS_DOMAIN_ID=42 ros2 topic pub --once /challenge_question ...

set -e
cd "$(dirname "$0")"

SIM_DOMAIN=42
BRIDGE_CFG="$(pwd)/challenge_topics_bridge.yaml"

SIM=./system_simulation.sh
case "${1:-}" in
  "") ;;
  --noviz|-n) SIM=./system_simulation_noviz.sh ;;
  *) echo "usage: $(basename "$0") [--noviz|-n]" >&2; exit 1 ;;
esac

# The sim script sources ROS in its own process, so do it here too for the
# bridge below. install/setup.bash chains the /opt/ros/jazzy underlay.
source ./install/setup.bash

echo "[challenge_sim] starting ${SIM} in ROS_DOMAIN_ID=${SIM_DOMAIN}"
ROS_DOMAIN_ID=${SIM_DOMAIN} "${SIM}" &
SIM_PID=$!
trap 'kill -TERM "${SIM_PID}" 2>/dev/null || true' EXIT SIGINT SIGTERM
sleep 8

echo "[challenge_sim] starting topic firewall (domain ${SIM_DOMAIN} <-> 0)"
ros2 run domain_bridge domain_bridge "${BRIDGE_CFG}"
