FROM zhangjicmu/ubuntu24_ros:ai_module

# Install Foxglove bridge for remote visualization
USER root
RUN apt-get update && apt-get install -y ros-jazzy-foxglove-bridge && rm -rf /var/lib/apt/lists/*
USER docker

# Copy updated dummy_vlm, smart_vlm, and tare_planner source into the existing workspace
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/smart_vlm /home/docker/ai_module/src/smart_vlm
COPY src/tare_planner /home/docker/ai_module/src/tare_planner

# Rebuild packages against the already-built workspace dependencies
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm smart_vlm tare_planner"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
