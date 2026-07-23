# M0 — Infra & Integration

**Task:** Stand up the dev environment and ROS plumbing everything else depends on: Docker containers, sim launch, visualization, topic verification, bag record/replay, and our AI-module node skeleton in the submission-compatible repo structure.

**Status:** 🟢 baseline done (setup verified Jul 21)
**Owner:**

---

## Hardware

**Dev machine: the L4 box** — i9-13900 (32 threads), 64 GB RAM, NVIDIA L4 24 GB, Ubuntu 24.04 LTS, X11 GUI, 3 TB disk. Near-identical CPU to the eval NUC13 i9 and same OS as the challenge stack — a single-box setup that mirrors exactly how evaluation runs (sim + AI module containers on one x86_64 machine).

| Machine | Arch | Role |
|---|---|---|
| **L4 box (i9-13900 + L4 24GB)** | x86_64 | Everything: sim + RViz + AI module, full runbook, `compose_gpu.yml` |
| Eval machine (theirs) | x86_64 NUC13 (i9) | Where the submission actually runs |

Notes:

- Containers bundle their own CUDA; host driver 595 is backward-compatible — nothing to install beyond the driver.
- L4 is passively cooled (datacenter card) — watch `nvidia-smi` temps under sustained SAM 3 load; ensure chassis airflow.
- 24 GB VRAM comfortably fits SAM 3 + SigLIP 2 + a local LLM simultaneously.
- Final submission image must be **linux/amd64** — native builds on this box already are; still pin with `docker buildx --platform linux/amd64` in CI.
- Eval GPU is unknown (NUC-class) — keep latency margins; don't assume L4-level throughput at eval time.

---

## Setup runbook (from repo `docker/README.md`)

### 1. Install Docker + NVIDIA container toolkit (on the sim host)

**Check first — skip installs already done:**
```bash
docker --version && systemctl is-active docker   # docker present + running?
nvidia-ctk --version                              # toolkit installed?
# definitive test — if this prints the GPU, skip ALL of step 1:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi
```
- Container test passes → step 1 done, go to step 2.
- `nvidia-ctk` ok but container test fails → only run: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
- Repo already added? `ls /etc/apt/sources.list.d/nvidia-container-toolkit.list` → if present, skip the curl/gpg lines, go straight to `apt install`.
- Nothing installed → full steps below.

```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
# reboot, then:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Clone + start containers
```bash
git clone --recurse-submodules git@github.com:Yuxin916/CMU-VLN-Challenge-2026.git
cd CMU-VLN-Challenge-2026
xhost +local:                # allow container GUI (RViz); safer than bare 'xhost +'
# Over SSH: no DISPLAY is set — point at the box's local X session first:
#   export DISPLAY=:1        # find it via: ls /tmp/.X11-unix/  (X0 → :0, X1 → :1)
#   export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority   # if xhost says "unable to open display"
# If /tmp/.X11-unix is empty or Xorg runs as gdm: nobody is logged in at the monitor —
# log in once, and enable auto-login for a lab box (/etc/gdm3/custom.conf:
# AutomaticLoginEnable=true, AutomaticLogin=<user>) so reboots don't kill the display.
# NOTE: the Unity sim needs the display too (renders camera images) — X must be
# alive before system_simulation.sh. RViz opens on the box's monitor; for remote
# viewing use Foxglove or x11vnc (see Remote visualization) — never ssh -X.
# GOTCHA: compose captures $DISPLAY at 'compose up' time. If containers were started
# from SSH (empty DISPLAY), Qt fails with 'could not connect to display' (no value).
# Fix per-run:  docker exec -it -e DISPLAY=:1 iros2026_system bash
# Fix for good: re-run 'docker compose ... up -d' from a shell with DISPLAY set.
# If GNOME logged into Wayland (echo $XDG_SESSION_TYPE), pick "Ubuntu on Xorg" at login.
cd docker
docker compose -f compose_gpu.yml up --build -d   # (compose.yml if no GPU)
```
Two containers start (shared host network):
- `iros2026_system` — simulator + base autonomy
- `iros2026_ai_module` — our dev environment (dummy_vlm inside)

### 3. Launch sim + visualization
```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
RViz should open with the scene. To change scenes: drop scene files into
`autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/`
(training scenes download: link in repo README).

### 3.5 Daily dev loop (smart_vlm — current AI module)
Workspace path in the ai container: `/home/docker/ai_module`. **Daily driver = dev mode** — original files untouched, all our tooling in the NEW `docker/compose_dev.yml`, which mounts host `ai_module/` over the container workspace:
`docker compose -f compose_gpu.yml -f compose_dev.yml up -d`, then build inside:
```bash
# after every git pull (dev mode):
docker exec -it iros2026_ai_module bash -c "source /opt/ros/jazzy/setup.bash && \
  cd /home/docker/ai_module && colcon build --symlink-install --packages-select dummy_vlm smart_vlm"
# run it:
docker exec -it iros2026_ai_module bash -c "source install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"
```
`smart_vlm.launch` = supervisor (mission clock, question intake, topic health) + dummy_vlm answer heads + TARE (once vendored — see launch file comments). Watch the supervisor heartbeat: it logs per-topic rates and warns on dead topics.
Eval-realistic mode: `scripts/challenge_simulation.sh` runs the sim in ROS domain 42 with a domain-bridge firewall passing only the 6 allowed inputs + question in and 3 answers out; launch the AI side with `ROS_DOMAIN_ID=0`. Needs `ros-jazzy-domain-bridge` in the system container (add to Dockerfile once validated).

### 4. Launch dummy VLM + send test questions
```bash
docker exec -it iros2026_ai_module bash
ros2 launch dummy_vlm dummy_vlm.launch
```
In another terminal (either container):
```bash
# object reference → marker + waypoint
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find teal pillow on the sofa farthest from the window'}"
# numerical → random Int32
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many books are on the sofa'}"
# instruction following → waypoint sequence
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door.'}"
```
Expected: vehicle follows waypoints, selected object highlighted in RViz.

### 5. Verify subscriber data (checklist below)
```bash
ros2 topic list
ros2 topic hz /camera/image          # expect ~10 Hz, 1920x640
ros2 topic hz /registered_scan       # expect ~5 Hz
ros2 topic hz /terrain_map           # ~5 Hz (5 m)
ros2 topic hz /terrain_map_ext      # ~5 Hz (20 m)
ros2 topic hz /state_estimation      # 100–200 Hz
ros2 topic echo /state_estimation --once
ros2 topic echo /challenge_question  # while publishing a question
```
View the camera stream: add an Image display in RViz on `/camera/image`, or `ros2 run rqt_image_view rqt_image_view`.
Note from first run: sim publishes `camera/image/compressed` + `camera/semantic_image/compressed`; a `sim_image_repub` node exists — confirm whether raw `/camera/image` appears in `ros2 topic list`.

### 6. Record a dev bag (enables offline perception work without the sim)

**Good-bag protocol:** drive a slow full lap covering every region of the scene; pass within 1–2 m of most objects; revisit 3–4 objects from a second, different viewpoint (this is what tests instance re-ID/merging later); include a doorway transit if multi-room; keep it 3–6 min. Name it `<scene>_lap1` and note the scene + date in a bags/README. Record ALL topics below — including `camera/semantic_image/compressed` if present (GT semantics: dev-only, great for debugging M2, banned at test time).
```bash
ros2 bag record /camera/image /registered_scan /sensor_scan \
  /terrain_map /terrain_map_ext /state_estimation /challenge_question -o scene01_run1
```

---

## Remote visualization (optional)

The L4 box has a GUI — RViz runs natively per the runbook. This section is only for watching runs from a laptop elsewhere.

**Foxglove:** run a websocket bridge on the L4 box, connect from a browser or the Foxglove desktop app. No host installs needed — run it inside the AI container:
```bash
docker exec -it iros2026_ai_module bash
sudo apt update && sudo apt install -y ros-jazzy-foxglove-bridge
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
# laptop: Foxglove → Open connection → ws://<l4-box-ip>:8765
```
Tips: best-effort QoS + decay for `/registered_scan`; Image panel for `/camera/image`. Save the layout as `vln_dev.json` in the repo so the team shares one view.

**Remote RViz via DDS** also works (laptop on same LAN, same `ROS_DOMAIN_ID`; `ROS_STATIC_PEERS=<l4-box-ip>` if multicast discovery fails), but expect lag on raw images over Wi-Fi — prefer Foxglove.

**VNC of the box's display** (see the actual RViz window): `sudo apt install x11vnc; x11vnc -display :1 -localhost -nopw -forever &`, then `ssh -L 5900:localhost:5900 tester03@<box>` and VNC-view `localhost:5900`.

**Rerun vs Foxglove:** Foxglove = live ops viz (turnkey ROS bridge, supports Markers — Rerun's ROS bridge is PoC-level and doesn't). Rerun = perception debugging in M2/M3: log masks, boxes, instances from our own Python via `rr.log()`, time-scrub replays, share `.rrd` recordings — teammates need no ROS to open them. Use both for their strengths.

---

## Progress checklist

- [ ] **Register for challenge (deadline Jul 25!)**
- [x] Docker + toolkit installed on sim host (Jul 21)
- [x] Decide sim host: **L4 box, single-box setup** (Jul 21, see decision log)
- [x] Containers build and start (Jul 21)
- [x] Sim launches, RViz shows scene + robot; teleop verified, sensor data visible (Jul 21)
- [x] Dummy VLM responds to all 3 question types (Jul 22 — behaves as documented, numerical is random as expected)
- [x] All 6 input topics verified (Jul 22 — see table; camera at 5 Hz not 10)
- [ ] Camera stream visualized; image size/format confirmed
- [ ] Camera↔lidar extrinsics located in repo and noted here
- [ ] Bag recorded and replayable (`ros2 bag play`)
- [ ] Training scenes downloaded; scene-swap procedure tested
- [ ] questions.json + answer PDFs + .ply trajectories downloaded and organized
- [ ] AI-module node skeleton (python) replaces dummy_vlm: subscribes all inputs, publishes all 3 outputs
- [ ] amd64 image pinned via buildx; clean-machine rebuild tested
- [ ] **W5 submission prep:** update `ai_module/docker/Dockerfile` to COPY + build `smart_vlm` (kept stock during dev — dev mode mounts code instead); test the pure-image build end-to-end
- [ ] (Optional) Foxglove bridge up; laptop connects; shared layout saved

### Topic verification table (measured Jul 22)
| Topic | Type | Expected | Measured | Notes |
|---|---|---|---|---|
| /camera/image | Image | 10 Hz, 1920×640 | **5 Hz** | ⚠️ below README's 10 Hz — likely render-rate-bound; fine for 1–2 Hz perception, but confirm resolution + whether eval machine differs |
| /registered_scan | PointCloud2 | 5 Hz, map frame | 5 Hz | ✓ |
| /sensor_scan | PointCloud2 | 5 Hz | 5 Hz | ✓ |
| /terrain_map | PointCloud2 | 5 Hz, 5 m | 5 Hz | ✓ |
| /terrain_map_ext | PointCloud2 | 5 Hz, 20 m | 5 Hz | ✓ |
| /state_estimation | Odometry | 100–200 Hz | ~200 Hz | ✓ |

Many extra topics also publish (e.g. /overall_map, camera/semantic_image/*) — dev-only, firewalled at eval-mimic time by `scripts/challenge_simulation.sh` (domain bridge, only the 6+question in / 3 answers out).

---

## Suggestions / better things to try

- **Foxglove Studio alongside RViz** — free, runs in browser, connects via `foxglove_bridge` websocket; handy for teammates watching the same run remotely. See "Remote visualization" above.
- **Write the AI module in Python (rclpy)** even though dummy is C++ — iteration speed matters more than runtime here; keep C++ only if a profiler says so.
- **Wrap heavy models behind a local HTTP/gRPC service** inside the AI module container (e.g., a small FastAPI server for SAM 3). Decouples model runtime from ROS node lifecycle and makes model swaps/restarts independent of the ROS graph; runs on localhost in both dev and eval.
- **Bag-first development:** record one good exploration bag per training scene in W1; M2/M3 development then runs offline against bags (fast, deterministic, no sim launch needed).
- **Pin everything now** (base image digest, apt/pip versions) — avoids eval-day surprises when they rebuild our image.
- **Makefile/justfile targets** for the 5 common ops: `up`, `sim`, `ai`, `ask q="..."`, `bag`. Removes tribal knowledge. (See `scripts/Makefile`.)
- **CI later:** a GitHub Action that builds the amd64 image weekly catches bit-rot before submission week.

## Log
| Date | Update |
|---|---|
| Jul 21 | Doc created; runbook from repo docker/README.md |
| Jul 21 | Setup complete on L4 box: containers up, sim + RViz working, teleop verified, all sensor data visible. Gotchas: DISPLAY empty in container when compose'd from SSH (fix: `docker exec -e DISPLAY=...`); X session must be alive (auto-login recommended) |
| Jul 21 | Folder accidentally deleted; fully restored from Claude session |
