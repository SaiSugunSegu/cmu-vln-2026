# Remote GPU Machine Setup — Browser GUI + Keyboard Teleop

Guide for running the CMU-VLN simulator on a **headless remote server** (e.g. an
EC2 GPU instance with no monitor) and viewing/driving it from your **local
browser**. This complements [`README.md`](./README.md) — do the Docker install
and container build/start there first, then follow this.

> **Verified environment (what actually worked):** Ubuntu 22.04, NVIDIA A10G
> (driver 535), Docker 25.0 + Compose v2, headless. GUI viewed with VNC + noVNC
> in a local browser over an SSH tunnel; GPU rendering via VirtualGL.

## Prerequisites

- Docker + NVIDIA Container Toolkit installed, and the GPU visible to Docker:
  ```bash
  docker run --rm --gpus all ubuntu nvidia-smi -L   # should list the GPU
  ```
- Both containers built and running with the **GPU** compose file:
  ```bash
  cd docker
  docker compose -f compose_gpu.yml up --build -d
  docker compose -f compose_gpu.yml ps               # both Up
  ```

The compose files use `--network=host` and mount `/tmp/.X11-unix`, so the
containers share the ROS 2 graph and can reach a local X/VNC display.

---

## 1. Virtual display + noVNC (host, one-time)

Install the display/VNC tooling on the host:
```bash
sudo apt-get update
sudo apt-get install -y \
  tigervnc-standalone-server tigervnc-common openbox \
  mesa-utils dbus-x11 novnc websockify python3-websockify
```

Start a virtual display, a minimal window manager, and the noVNC web bridge.
`setsid` detaches them so they survive your shell/SSH session.
```bash
# Virtual display :1 (localhost-only, no VNC password)
setsid bash -c 'Xvnc :1 -geometry 1920x1080 -depth 24 -SecurityTypes None \
  -localhost yes -rfbport 5901 >/tmp/xvnc.log 2>&1' </dev/null &

# Minimal window manager on that display
DISPLAY=:1 setsid bash -c 'openbox >/tmp/openbox.log 2>&1' </dev/null &

# Allow the containers' X clients to connect
DISPLAY=:1 xhost +

# Serve the display to the browser (http 127.0.0.1:6080 -> VNC 5901)
setsid bash -c 'websockify --web /usr/share/novnc 127.0.0.1:6080 localhost:5901 \
  >/tmp/novnc.log 2>&1' </dev/null &
```

Sanity check:
```bash
ss -tlnp | grep -E '5901|6080'                                            # both LISTEN
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:6080/vnc.html   # 200
```

---

## 2. Open it from your local machine

Forward the noVNC port over SSH (no public ports / security-group changes needed):
```bash
# On your LOCAL machine:
ssh -L 6080:localhost:6080 <user>@<server-host>
```
Then open in your local browser:
```
http://localhost:6080/vnc.html?autoconnect=true
```

---

## 3. GPU-accelerated rendering in the container (one-time)

The Unity simulator needs real OpenGL. On a headless box we render it on the GPU
with **VirtualGL** (EGL backend). The base image ships the NVIDIA GL/EGL
libraries but is **missing the NVIDIA EGL vendor file**, so we add that too.
```bash
docker exec -u root iros2026_system bash -lc '
  curl -fsSL -o /tmp/vgl.deb \
    https://github.com/VirtualGL/virtualgl/releases/download/3.1.2/virtualgl_3.1.2_amd64.deb
  apt-get update && apt-get install -y /tmp/vgl.deb
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<JSON
{
    "file_format_version" : "1.0.0",
    "ICD" : { "library_path" : "libEGL_nvidia.so.0" }
}
JSON
'
```
Verify (should print `OpenGL Renderer: NVIDIA A10G/...`):
```bash
docker exec -u root -e DISPLAY=:1 iros2026_system \
  bash -lc 'timeout 5 vglrun -d egl0 /opt/VirtualGL/bin/glxspheres64 | head'
```

> **Persistence:** VirtualGL, the EGL vendor file, and the helper scripts below
> are installed into the *running* container. They **survive `docker restart`**
> but are **lost on `docker compose down` / recreate** — re-run this step and the
> `docker cp` commands after recreating the container.

---

## 4. Launch the simulator (GUI)

Copy the launch helper in and start it on display `:1`:
```bash
# from the repo root
docker cp scripts/run_sim_gui.sh iros2026_system:/home/docker/run_sim_gui.sh
docker exec -d -e DISPLAY=:1 iros2026_system bash /home/docker/run_sim_gui.sh
```
Watch it come up in the browser: RVIZ with the point cloud, camera image, and
semantic image panels.

`run_sim_gui.sh` renders the **Unity environment on the GPU** (`vglrun -d egl0`)
but runs **RVIZ on the display's software GL** on purpose.

> **Gotchas learned the hard way**
> - **Do NOT run RVIZ under `vglrun`.** Routing RVIZ through VirtualGL to
>   TigerVNC freezes its repaint (window looks stuck while ROS keeps running).
>   `run_sim_gui.sh` already avoids this.
> - **Run only ONE simulator instance.** Two Unity clients fighting over the same
>   ROS-TCP bridge port crashes it, and the camera/semantic images disappear
>   ("No Image"). Check before launching:
>   ```bash
>   docker exec iros2026_system pgrep -af system_simulation.launch   # empty = safe
>   ```
>   Cleanest reset: `docker restart iros2026_system`, then relaunch.

---

## 5. Drive the robot with the keyboard (teleop testing)

The autonomy stack has a manual holonomic drive mode (mecanum base) that
activates via the `/joy` topic. `scripts/keyboard_teleop.py` emulates a joystick
from the keyboard — no physical controller needed.

Copy it in and run it in its **own** terminal (while the simulator runs):
```bash
docker cp scripts/keyboard_teleop.py iros2026_system:/home/docker/keyboard_teleop.py
docker exec -it iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && \
   python3 /home/docker/keyboard_teleop.py'
```

Controls:

| Key           | Action                              |
|---------------|-------------------------------------|
| `w` / `s`     | forward / backward                  |
| `a` / `d`     | strafe left / right                 |
| `q` / `e`     | rotate left / right                 |
| `space` / `k` | stop                                |
| `z` / `x`     | speed scale down / up               |
| `Ctrl-C`      | quit (stops + releases to autonomy) |

Quick test:
1. Launch the simulator (section 4) and open the browser view.
2. Run the teleop command above in a second terminal and **click that terminal**
   so it has keyboard focus.
3. Tap `w` — the robot drives forward in the browser view. Press `space` to stop.

> **How to use it**
> - Type in the **teleop terminal** (it needs keyboard focus). You *watch* the
>   robot in the browser, but you *type* in the terminal — keys pressed in the
>   browser window won't reach it.
> - It's **cruise-style**: a key keeps the robot moving until you change it or
>   press `space` (max ~0.875 m/s, ~80°/s). This is why it can "run away" if you
>   forget to stop it.
> - On `Ctrl-C` it stops the robot and hands control back to the autonomy stack,
>   so waypoint / autonomous navigation works again afterward.

You can also verify the manual command reaches the base without the keyboard UI:
```bash
docker exec iros2026_system bash -lc '
  source /opt/ros/jazzy/setup.bash
  source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash
  # manual mode ON (axes[5]=-1), autonomy OFF (axes[2]=1), forward=0.5
  ros2 topic pub -r 20 /joy sensor_msgs/msg/Joy \
    "{axes: [0.0,0.0,1.0,0.0,0.5,-1.0,0.0,0.0], buttons: [0,0,0,0,0,0,0,0,0,0,0]}"'
# In another terminal, watch /cmd_vel respond, then Ctrl-C the publisher.
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| RVIZ window frozen (ROS still running) | RVIZ launched under `vglrun` | Use `run_sim_gui.sh` (RVIZ on software GL) |
| Camera/semantic panels show "No Image" | Sim launched twice → ROS-TCP bridge crashed | `docker restart iros2026_system`, launch once |
| Robot "runs away" / won't stop | Cruise-style key latched | Press `space`/`Ctrl-C` in the teleop terminal |
| `vglrun` shows Mesa/`llvmpipe`, not the GPU | Missing NVIDIA EGL vendor file | Re-run section 3 (`10_nvidia.json`) |
| Browser can't connect | SSH tunnel down / services not running | Recheck section 1 sanity checks and the `-L 6080` tunnel |
