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

plus a one-shot `init` container that fixes permissions on the bind mounts and then
exits — `Exited (0)` is success, not an error.

## First run: download model weights

Everything runs offline (`HF_HUB_OFFLINE=1` is baked into the image), so the models
must be in the cache before anything will load. This is a **one-time** ~15-20 GB
download:

```bash
just up          # build + start; `init` makes /data and the HF cache writable
just hf-fetch    # one-time download: facebook/sam3, Qwen3-VL-4B, DFN5B-CLIP
just vqa-up      # loads Qwen from the now-populated cache
```

`just hf-fetch --list` shows what will be pulled; `just hf-fetch "qwen3vl sam3"`
pulls a subset. It resumes and skips what is already cached, so re-running is cheap.

**You do not need `hf auth login`.** Put your token in the repo-root `.env`:

```
HF_TOKEN=hf_...
```

`huggingface_hub` reads `HF_TOKEN` automatically on every `from_pretrained()`.
`facebook/sam3` is a **gated** repo, so besides a valid token you must accept its
licence once at <https://huggingface.co/facebook/sam3> with the same account —
otherwise `hf-fetch` reports `GATED` and tells you which of the two is missing.

`just hf-fetch` is the only command that goes online; it passes `HF_HUB_OFFLINE=0`
for that one invocation via `docker exec -e`, so your `.env` is never modified. If
you ever need a different model online, set `HF_HUB_OFFLINE=0` in `.env` instead.

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
Numerical question (triggers random integer response from dummy alone):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many books are on the sofa'}"
```
Navigation question (triggers sequential waypoint following):
```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door.'}"
```

You should see the vehicle following waypoints and the selected object being highlighted in RVIZ.

### Qwen numerical answers (smart_vlm)

`ros2 launch smart_vlm smart_vlm.launch` starts `qwen_numerical`. It answers
**How many / Count** questions from `/camera/image` and publishes an `Int32` on
`/numerical_response`. Dummy’s random numerical publisher is disabled in that
launch (`dummy_answer_numerical:=false`).

**Start `qwen_vqa_server` first.** `qwen_numerical` does not load a checkpoint of
its own — it sends the frame to the shared server, so one copy of the weights
serves this head, the category-1 reasoner, and `just vqa-ask`. Without the server
running, the head waits `server_wait_s` (default 600 s) and then errors.

```bash
# after colcon build --packages-select captioner smart_vlm dummy_vlm
just vqa-up                      # loads Qwen once; blocks until ready
ros2 launch smart_vlm smart_vlm.launch
# elsewhere:
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many pillows are on the bed?'}"
ros2 topic echo /numerical_response --once
```

Model and quantization are the **server's** parameters now:
`just vqa-up qwen2_5vl int8`.

Disable the Qwen head and restore dummy random ints with:
`use_qwen_numerical:=false dummy_answer_numerical:=true`.

## Run the captioner (offline crop CLI)

The AI image installs CUDA PyTorch, transformers, and the `captioner` ROS package.
Host folder `data/` is mounted at `/data`, and your Hugging Face cache is
mounted so model weights persist across rebuilds.

Put instance-crop folders (each with `crop.png` or `rgb.png`) under `data/crops`, then:

```bash
docker exec -it iros2026_ai_module bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH &&
  caption_crops /data/crops \
    --output_dir /data/captions \
    --captioning_model qwen3vl \
    --quantization int4 \
    --batch_size 8
'
```

Or with the helper script from the repo root:

```bash
./ai_module/docker/run_captioner.sh /data/crops /data/captions
```

## Run Qwen VQA (offline image + question CLI)

**Preferred (persistent server — load once, ask many):** from the repo root:

```bash
just vqa-up          # compose + load Qwen int4; blocks until ready
just vqa-ask "How many pillows are on the bed?" /data/pillow_bed.png
just vqa-ask "How many lamps are there?" /data/pillow_bed.png   # fast
just vqa-status
just vqa-down
```

Image paths must be under `/data/…`. Host `data/` is bind-mounted 1:1, so host
`data/pillow_bed.png` is container `/data/pillow_bed.png`. Paths outside the
mount are rejected (`captioner/paths.py`) — copy the file into `data/` first.

One-shot CLI (reloads weights every call — slow):

```bash
./ai_module/docker/run_qwen_vqa.sh /data/pillow_bed.png \
  "How many pillows are on the bed?"
```

Or directly, with paths under the `data/` mount (the `init` one-shot in
`compose.yml` creates `crops/`, `captions/`, `runs/` and makes them writable by
the container's uid, so no host-side `mkdir`/`chmod` is needed):

```bash
# copy/symlink a scene of crops into data/crops on the host, then:
docker exec -it iros2026_ai_module bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH &&
  caption_crops /data/crops --output_dir /data/captions
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
