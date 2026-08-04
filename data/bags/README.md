# Scene bags

Recorded ros2 bags used to replay the challenge inputs **instead of the live Unity
simulator**, so the AI module can run offline (no Unity, no GPU).

- **Layout:** one folder per scene — `<repo>/bags/<scene>/` (each = `metadata.yaml` + `*.mcap`),
  named by scene (`scene_0`, `office`, …). Bind-mounted into the containers at `/data/bags`.
- **Not in git:** bag data is gitignored and fetched on demand. Only `README.md` and
  `scenes.yaml` (the manifest) are tracked.
- **Drive folder:**
  https://drive.google.com/drive/folders/1dgTn2f_oSGkJFlgA1jHcikDdjIYqZxE4?usp=drive_link

## Quick start

```bash
# inside the ai_module container — auto-downloads scene_0 if missing, then plays
ros2 launch smart_vlm bag_replay.launch scene:=scene_0            # bag only
ros2 launch smart_vlm smart_vlm.launch use_bag:=true bag:=scene_0 # smart_vlm + bag
```

Downloading a scene is one-time per host (bags persist here across container restarts). `gdown`
ships in the ai_module image; each scene just needs a real `drive_id` in `scenes.yaml`.

Full workflow — recording, publishing (zip + sha + Drive ID), playing options, and caveats:
see [`docs/M0.5_rosbag_infra.md`](../docs/M0.5_rosbag_infra.md).
