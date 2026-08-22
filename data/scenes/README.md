# Unity scenes

The Unity scene builds used by the **live simulator** (`just eval-cat1-sim`, `just sim`,
`just challenge`) — the counterpart to [`../bags`](../bags), which replays recordings
instead.

- **Layout:** one folder per scene — `<repo>/data/scenes/<scene>/`, bind-mounted into the
  containers at `/data/scenes`. Each holds the whole Unity build plus its metadata:

  ```
  <scene>/environment/Model.x86_64     the simulator itself — this is what runs
  <scene>/environment/UnityPlayer.so   ~32 MB
  <scene>/environment/Model_Data/
  <scene>/map.ply                      point cloud of the scene
  <scene>/traversable_area.ply         where the robot may drive
  <scene>/object_list.txt              id, pose, size, label per object
  ```

  **The scene is the simulator.** `system_simulation_noviz.sh` runs
  `mesh/unity/environment/Model.x86_64`, so there is no simulator to fall back on when a
  scene is missing — which is why fetching matters.

- **Not in git:** ~300 MB per scene, ~16 GB for all 18. Only `README.md` and `scenes.yaml`
  (the manifest) are tracked.
- **Fetched on demand:** `scripts/eval/run_sim_sweep.py` downloads any scene a sweep needs
  and does not find, so a new machine needs no manual download.
- **Drive folder** (the organizers'):
  https://drive.google.com/drive/folders/1nki_xoFKX1bYr8m7qiGRQelwnQ7EKVYc?usp=drive_link

## Quick start

```bash
just up                                                    # creates + chmods /data/scenes
docker exec iros2026_odyssey ros2 run smart_vlm scene_fetch arabic_room
```

Idempotent — a second run says "already present". `just eval-cat1-sim <scene>` calls it for
you, so usually you never run it by hand.

## How a scene reaches the simulator

The system container's mesh slot is **not** bind-mounted, so it can only be written with
`docker cp`. `run_sim_sweep.py` does that per scene:

```
data/scenes/<scene>/  --docker cp-->  iros2026_system:
    /home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/
        vehicle_simulator/mesh/unity/
```

Consequences worth knowing: the loaded scene is **container state, not repo state** — it
survives until something overwrites it, and is lost on `docker compose down`. Editing the
repo's own `.../mesh/unity/` (which holds only a `readme.txt`) has no effect at all.

## Adding or refreshing scenes

`scenes.yaml` maps each scene to one Drive zip id. Its header carries the one-liner that
regenerates every id from the folder link, so nothing has to be looked up by hand.
