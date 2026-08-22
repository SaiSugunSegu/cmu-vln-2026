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
From the repo root (GPU). `.env` must contain `HF_TOKEN` and the VLM key:
```bash
just up
```
This builds `iros2026_odyssey:submission` (weights + keys) and starts two containers:
- `iros2026_system` — the base autonomy system (simulator + autonomy stack)
- `iros2026_odyssey` — the submission image (`iros2026_odyssey:submission`)

plus a one-shot `init` container that fixes permissions on the bind mounts and then
exits — `Exited (0)` is success, not an error.

## First run: bake the AI image

`just up` builds `iros2026_odyssey:submission`, bakes `facebook/sam3` and Qwen3-VL
into a layer, and writes `HF_TOKEN` plus the vqa.yaml provider key (OpenRouter)
into image ENV. First build is 15–20 GB; later `just up` only rebuilds layers
that changed. `init` makes `/data` writable (`Exited (0)` is success).

```bash
just up          # build + start; weights and keys come from .env
just vqa-up      # OPTIONAL: keeps Qwen resident across relaunches; the
                 # pipeline starts its own server if you skip this
```

**You do not need `hf auth login`.** Put the tokens in the repo-root `.env`:

```
HF_TOKEN=hf_...
OPENROUTER_API_KEY=sk-or-...
```

`huggingface_hub` reads `HF_TOKEN` automatically on every `from_pretrained()`.
`facebook/sam3` is a **gated** repo, so besides a valid token you must accept its
licence once at <https://huggingface.co/facebook/sam3> with the same account.

`just hf-fetch` is optional: it pulls a new checkpoint or warms SAM 3's `cv-utils`
kernel in the running container. A checkpoint you want in the Hub image has to
land via `just up`. The kernel does mask **NMS** as well as hole filling; it is
fetched lazily on first use and its absence degrades **silently** — NMS is skipped
entirely, so duplicate detections are never suppressed and `det_nms_thresh` does
nothing. Confirm with `just sam-profile /data/bags/_frames`, which prints
`cv-utils kernel: ...`, `[sam3] attn effective: ...` and the thresholds actually
in force.

Flash attention is deliberately **not** used: on SAM 3 it returns zero detections.
Re-check with `just sam-probe <frames> <cfg> "--attn kernels-community/flash-attn2"`.

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

## Launch the AI module

Official evaluation starts the module with this command inside `iros2026_odyssey`:

```bash
ros2 launch dummy_vlm dummy_vlm.launch
```

That includes `smart_vlm.launch`: SAM 3, the 3D mapper, the supervisor, the numerical /
object-reference / instruction reasoners, TARE, and the local Qwen server. `just ai`
runs the same launch. `just up` bakes `facebook/sam3` and Qwen3-VL (and the API keys)
into `iros2026_odyssey:submission` from the repo-root `.env`.

## Numerical answers (smart_vlm)

`ros2 launch dummy_vlm dummy_vlm.launch` brings up the whole per-question pipeline:
`sam_node` (booted unarmed — it loads weights but detects nothing until the question
supplies prompts), the `smart_vlm` supervisor, the three reasoners, and TARE
exploration. It starts **no scene source** — it is the submission artifact, and consumes
the six allowed topics from whatever is publishing them. For offline replay use
`eval_bag.launch`, which wraps it and adds a bag.

`numerical_reasoner` answers **How many / Count** questions and publishes an `Int32` on
`/numerical_response`. It works from SAM's best-view crops rather than raw frames:
question → target nouns → `/sam3/set_prompts` → wait for `/pipeline/explore_done` →
ask Qwen about the best crop.

**The VQA server starts itself.** On the default `local` backend the launch brings up
`qwen_vqa_server` as part of the module and the supervisor gates `/pipeline/ready` on
`/qwen_vqa/status` reaching `ready` — the same way it gates on `/sam3/status`. That is
deliberate: at evaluation nobody runs a setup step for you.

`just vqa-up` is therefore optional. It keeps one server resident *across* relaunches,
which saves the ~8.3 GB reload each question costs — useful in a long sweep, but do not
run it while the launch is also starting one, or the two collide on the node name and
the `/qwen_vqa` topics. With both `vlm_backend` and `target_extract_backend` set to
`cloud` in vqa.yaml no server is started or waited for.

```bash
# after colcon build --packages-select captioner smart_vlm sam_mapper
ros2 launch smart_vlm eval_bag.launch bag:=arabic_room
# elsewhere — the bag is held at /pipeline/armed until this arrives:
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many pillows are on the bed?'}"
ros2 topic echo /numerical_response --once
```

Model and quantization are the **server's** parameters: `just vqa-up qwen2_5vl int8`.

To score this across the benchmark instead of asking by hand, use `just eval-cat1`, which
relaunches the pipeline per question (README §3.5).

## Run the captioner (offline crop CLI)

The AI image installs CUDA PyTorch, transformers, and the `captioner` ROS package.
Host folder `data/` is mounted at `/data`. Model weights live in the image.

Put instance-crop folders (each with `crop.png` or `rgb.png`) under `data/crops`, then:

```bash
docker exec -it iros2026_odyssey bash -lc '
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
docker exec -it iros2026_odyssey bash -lc '
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:$PATH &&
  caption_crops /data/crops --output_dir /data/captions
'
```

After editing captioner Python sources, rebuild the image — `ai_module` is never
bind-mounted, so that is what carries the edit into the container:

```bash
just up
```

The ML wheels (torch, transformers, …) live in an earlier layer and stay cached, so
only the source-copy and colcon layers re-run.
