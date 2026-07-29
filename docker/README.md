## Install Docker

### 1) For computers without a Nvidia GPU

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install additional packages.
```
sudo apt update && sudo apt install mesa-utils libgl1-mesa-dri libgl1 libglx-mesa0
```

### 2) For computers with Nvidia GPUs

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install Nvidia Container Toolkit (Nvidia GPU Driver
should be installed already).

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```
sudo apt update && sudo apt install nvidia-container-toolkit
```
Configure Docker runtime and restart Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run Docker Containers

Clone the workshop repository to your home folder.
```
cd /path/to/desired/directory
git clone --recurse-submodules git@github.com:Yuxin916/CMU-VLN-Challenge-2026.git && cd ./CMU-VLN-Challenge-2026
```
Allow remote X connection.
```
xhost +
```
Go inside the docker folder.
```
cd docker
```
For computers **without a Nvidia GPU**, build and start both containers.
```bash
docker compose -f compose.yml up --build -d
```
For computers **with Nvidia GPUs**, use the GPU compose file instead.
```bash
docker compose -f compose_gpu.yml up --build -d
```
This starts two containers:
- `iros2026_system` — the base autonomy system (simulator + autonomy stack)
- `iros2026_ai_module` — the AI module development environment, with `dummy_vlm`, `smart_vlm`, and `captioner` built in

## Launch base autonomy system

Access the system container.
```bash
docker exec -it iros2026_system bash
```
Inside the container, launch the base autonomy system.
```bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
On a machine with an Nvidia GPU but no hardware-accelerated X server (e.g. a headless
box running Xvnc), prefix the launch with VirtualGL so Unity renders on the GPU instead
of falling back to Mesa `llvmpipe`:
```bash
cd /home/docker/autonomy_stack_mecanum_wheel_platform
vglrun -d egl ./system_simulation.sh              # ./system_simulation_noviz.sh to skip RVIZ
```
See "GPU rendering" in the [repo README](../README.md) for details and how to verify.

## Launch dummy VLM

Access the AI module container.
```bash
docker exec -it iros2026_ai_module bash
```
Inside the container, launch the dummy VLM.
```bash
ros2 launch dummy_vlm dummy_vlm.launch
```
The dummy VLM listens on `/challenge_question` (std_msgs/String) and responds based on the question type:
- Questions starting with **"Find"** or **"find"**: publishes a bounding box marker on `/selected_object_marker` and sends a waypoint to the object on `/way_point_with_heading`.
- Questions starting with **"How many"** or **"how many"**: publishes a random integer (1–10) on `/numerical_response`.
- All other questions (navigation): publishes a sequence of waypoints on `/way_point_with_heading`, advancing as the vehicle reaches each one.

To send example questions, open a new terminal, exec into either container, and use `ros2 topic pub`. Both containers share the same ROS2 network via `--network=host`.

Object reference question (triggers marker + object waypoint):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find teal pillow on the sofa farthest from the window'}"
```
Numerical question (triggers random integer response):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many books are on the sofa'}"
```
Navigation question (triggers sequential waypoint following):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door.'}"
```

You should see the vehicle following waypoints and the selected object being highlighted in RVIZ.

## Run the captioner (offline crop CLI)

The AI image installs CUDA PyTorch, transformers, and the `captioner` ROS package.
Host folder `data/` is mounted at `/data/workspace`, and your Hugging Face cache is
mounted so model weights persist across rebuilds.

Put instance-crop folders (each with `crop.png` or `rgb.png`) under `data/crops`, then:

```bash
docker exec -it iros2026_ai_module bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH &&
  caption_crops /data/workspace/crops \
    --output_dir /data/workspace/captions \
    --captioning_model qwen3vl \
    --quantization int4 \
    --batch_size 8
'
```

Or with the helper script from the repo root:

```bash
./ai_module/docker/run_captioner.sh /data/workspace/crops /data/workspace/captions
```

Or with explicit host paths already visible via the `data/` mount:

```bash
# from repo root on the host
mkdir -p data/crops data/captions
# copy/symlink a scene of crops into data/crops, then:
docker exec -it iros2026_ai_module bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH &&
  caption_crops /data/workspace/crops --output_dir /data/workspace/captions
'
```

After editing captioner Python sources with the [dev overlay](compose_dev.yml), rebuild only the package inside the container (ML wheels stay in the image):

```bash
docker compose -f compose_gpu.yml -f compose_dev.yml up -d
docker exec -it iros2026_ai_module bash -lc '
  source /opt/ros/jazzy/setup.bash &&
  cd /home/docker/ai_module &&
  colcon build --symlink-install --packages-select captioner
'
```

## Integrate your AI model

To replace the dummy VLM with your own model, modify `ai_module/src/dummy_vlm/src/dummyVLM.cpp` and rebuild the Docker image.
```bash
cd <path-to-repo>/docker
docker compose -f compose.yml up --build -d
```
Your model must subscribe to `/challenge_question` (std_msgs/msg/String) and publish on the appropriate response topic based on the question type.
