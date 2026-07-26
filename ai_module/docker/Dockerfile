FROM zhangjicmu/ubuntu24_ros:ai_module

# Install Foxglove bridge for remote visualization
USER root
RUN apt-get update && apt-get install -y ros-jazzy-foxglove-bridge && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-yaml \
    && python3 -m pip install --no-cache-dir --break-system-packages gdown \
    && rm -rf /var/lib/apt/lists/*
USER docker

# Copy and compile TARE first (takes 90s, but is heavily cached and rarely changes)
COPY src/tare_planner /home/docker/ai_module/src/tare_planner
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select tare_planner"

# Copy and compile active dev packages (takes <5s and updates instantly on edit)
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/smart_vlm /home/docker/ai_module/src/smart_vlm
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm smart_vlm"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
