FROM zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation
USER root
RUN apt-get update && apt-get install -y ros-jazzy-domain-bridge && rm -rf /var/lib/apt/lists/*
USER docker

# Copy the challenge simulation scripts into the autonomy stack
COPY challenge_simulation.sh /home/docker/autonomy_stack_mecanum_wheel_platform/
COPY challenge_topics_bridge.yaml /home/docker/autonomy_stack_mecanum_wheel_platform/

# Ensure the script is executable
USER root
RUN chmod +x /home/docker/autonomy_stack_mecanum_wheel_platform/challenge_simulation.sh
USER docker
