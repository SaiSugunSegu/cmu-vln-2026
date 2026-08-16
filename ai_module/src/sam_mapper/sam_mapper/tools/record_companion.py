"""Record a companion /sam3/* bag: drive SAM 3 over a scene bag once, offline.

The only GPU step in the map3d bench. Writes <scene>_sam3/ + a manifest so replay_map3d
can rebuild the 3D map on CPU deterministically, instead of re-running a nondeterministic
detector for every experiment.

    just map3d-record <scene> [stride] ["--objects a,b --variant tag"]"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time

import numpy as np
import rosbag2_py
import torch
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_mapper.detections import PromptTable, encode_instance_id, to_detections
from sam_mapper.node_base import load_config, resolve_config_path
from sam_mapper.sam3_backend import Sam3Backend, make_backend
from sam_mapper.tools.dump_frames import open_reader

INSTANCE_MAP_TOPIC = "/sam3/instance_map"
DETECTIONS_TOPIC = "/sam3/detections"
CAMERA_TOPIC = "/camera/image"


def open_writer(uri: str) -> rosbag2_py.SequentialWriter:
    """Open an mcap writer with the two companion topics registered."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    for name, type_name in ((INSTANCE_MAP_TOPIC, "sensor_msgs/msg/Image"),
                            (DETECTIONS_TOPIC, "std_msgs/msg/String")):
        # TopicMetadata gained a leading `id` field in newer rosbag2; accept both so this
        # doesn't break on a distro bump.
        try:
            meta = rosbag2_py.TopicMetadata(
                id=0, name=name, type=type_name, serialization_format="cdr")
        except TypeError:
            meta = rosbag2_py.TopicMetadata(
                name=name, type=type_name, serialization_format="cdr")
        writer.create_topic(meta)
    return writer


def _free_vram() -> None:
    """Hand cached blocks back to the driver. Worth doing on every session restart: after
    an OOM a large share of the reserved pool is fragmented rather than genuinely in use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def scene_prompts(prompts_path: str, scene: str, override: str | None = None) -> list[dict]:
    """The prompt set for a scene: the curated per-scene list, or an explicit override.

    `override` is a comma-separated prompt list, for narrowing a recording to a handful of
    classes so each can be debugged against ground truth without ~70 detections per frame
    to reason through. Every entry becomes an instance class; prefix one with '-' to make
    it a background class (negative id, re-created per frame, never tracked).
    """
    if override:
        objects = []
        for raw in override.split(","):
            prompt = raw.strip()
            if not prompt:
                continue
            instance = not prompt.startswith("-")
            objects.append({"prompt": prompt.lstrip("-").strip(), "instance": instance})
        if not objects:
            raise SystemExit(f"--objects {override!r} parsed to an empty prompt set")
        return objects

    entry = json.load(open(prompts_path))["scenes"].get(scene)
    if entry is None:
        raise SystemExit(f"{scene} not in {prompts_path} — that file is hand-curated from "
                         f"questions/questions.json; add the scene's target objects to it")
    return entry["objects"]


def companion_paths(bags_dir: str, scene: str, variant: str | None = None) -> tuple[str, str]:
    """(bag uri, manifest path) for a scene, optionally suffixed by an experiment variant.

    A variant keeps a narrowed recording from clobbering the full-prompt one, so both stay
    replayable and comparable: `<scene>_sam3` vs `<scene>_sam3_focus4`.
    """
    stem = f"{scene}_sam3" + (f"_{variant}" if variant else "")
    scene_dir = os.path.join(bags_dir, scene)
    return os.path.join(scene_dir, stem), os.path.join(scene_dir, f"{stem}.manifest.json")


def load_backend(args) -> tuple[Sam3Backend, dict]:
    # Return type stays Sam3Backend for the common case; make_backend() below may hand
    # back a Sam31Backend instead (backend: native_sam31), which implements the same
    # set_prompts/reset/release/process_frame interface used everywhere below.
    """Load SAM 3 ONCE, for however many scenes follow.

    A `--scene all` sweep used to construct a Sam3Backend per scene, reloading 1797 weight
    shards 13 times. Besides the waste, re-registering the `kernels` flash-attn2 provider
    on every construction failed after the first, so scenes 2..13 all died with
    `KeyError: '...flash-attn2 is not a valid attention implementation...'`.

    Prompts are per-scene, but `set_prompts` already resets the video session, so one
    model serves every scene with no state carried across.

    Goes through `make_backend()` (not a direct `Sam3Backend(...)` call) so `sam3.backend:
    native_sam31` in the config actually takes effect here — measured 1.65x faster than
    hf_sam3 (sam3_backend.py). This recorder had bypassed that switch entirely: every
    `map3d-record` run paid the slower backend regardless of the yaml.
    """
    config = load_config(resolve_config_path(args.config), required_keys=("sam3",))
    backend = make_backend(config["sam3"], log=lambda m: print(f"  [sam3] {m}", flush=True))
    return backend, config


def record_scene(scene: str, args, backend: Sam3Backend, config: dict) -> dict:
    scene_dir = os.path.join(args.bags_dir, scene)
    if not os.path.isdir(scene_dir):
        raise SystemExit(f"no bag directory {scene_dir}")
    # rosbag2 treats the uri as a bag DIRECTORY, not a file — passing "<scene>_sam3.mcap"
    # produces a directory of that name containing "<scene>_sam3.mcap_0.mcap". Name it
    # without the extension so the layout is the ordinary rosbag2 one.
    out_uri, manifest_path = companion_paths(args.bags_dir, scene, args.variant)
    if os.path.exists(out_uri):
        if not args.force:
            raise SystemExit(f"{out_uri} exists — pass --force to overwrite")
        shutil.rmtree(out_uri) if os.path.isdir(out_uri) else os.remove(out_uri)
    # A crashed run leaves a partial bag behind. Drop any manifest from a PREVIOUS run now,
    # so an interrupted recording can never be paired with a manifest describing a
    # different one — the manifest is only rewritten on success, at the end.
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    objects = scene_prompts(args.prompts, scene, args.objects)
    table = PromptTable(objects)

    # `objects:` is the one config key overridden per scene — it comes from the curated
    # prompt set, not the yaml's placeholder list. set_prompts resets the video session,
    # so a backend shared across scenes carries no state between them.
    cfg = config["sam3"]
    backend.set_prompts(table.prompts)

    reader = open_reader(scene_dir)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CAMERA_TOPIC]))
    writer = open_writer(out_uri)
    bridge = CvBridge()

    # SAM 3's video tracker keeps a per-object memory bank that grows with every frame, so
    # VRAM scales with (tracked objects x frames). This recorder is a far heavier load than
    # sam_node ever sees in production: ~20 prompts instead of 2-5, and EVERY frame instead
    # of latest-frame-wins. On a 22 GB card that OOMs around frame 25 with ~70 detections.
    #
    # Bound it the way sam_node already bounds it on a bag loop: restart the session
    # periodically and shift subsequent ids past the high-water mark, so ids from different
    # sessions can never collide. Cost is a broken track at each boundary, which shows up
    # as a duplicate instance — the same trade the streaming-mode docs describe, and
    # something ObjMapper's world-space merge is meant to fold back together.
    id_offset, max_seen_id = 0, -1
    session_frames = 0
    oom_recoveries = restarts = 0

    def restart_session(why: str) -> None:
        nonlocal id_offset, session_frames, restarts
        id_offset = max_seen_id + 1
        session_frames = 0
        restarts += 1
        backend.reset()          # drops the old session, then reclaim its blocks
        _free_vram()
        print(f"  {scene}: session restart ({why}); ids now offset by {id_offset}", flush=True)

    seen = written = 0
    started = time.perf_counter()
    while reader.has_next():
        _topic, data, log_time_ns = reader.read_next()
        seen += 1
        if (seen - 1) % args.stride:
            continue
        if args.limit and written >= args.limit:
            break

        image_msg = deserialize_message(data, Image)
        bgr = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        rgb = bgr[:, :, ::-1].copy()          # cv_bridge gives BGR; SAM 3 wants RGB

        if args.session_frames and session_frames >= args.session_frames:
            restart_session(f"{args.session_frames} frames")

        try:
            result = backend.process_frame(rgb)
        except torch.OutOfMemoryError:
            # Drop this frame, reclaim, and carry on from a clean session rather than
            # losing the whole scene. A companion bag missing one frame is fine; the
            # replay harness skips gaps. Losing 13 scenes to frame 25 is not.
            oom_recoveries += 1
            print(f"  {scene}: CUDA OOM at frame {written} — recovering "
                  f"({oom_recoveries})", flush=True)
            restart_session("OOM")
            continue
        session_frames += 1

        detections = to_detections(result, table)

        # Keep ids unique across sessions, exactly as sam_node does after a bag loop.
        instance = detections["ids"] >= 0
        if id_offset:
            detections["ids"][instance] += id_offset
        if instance.any():
            max_seen_id = max(max_seen_id, int(detections["ids"][instance].max()))

        # Exact integer stamps, not sam_node's float round-trip: map_node pairs the two
        # topics by (sec, nanosec) equality, and a float detour can perturb the low bits.
        sec = int(image_msg.header.stamp.sec)
        nanosec = int(image_msg.header.stamp.nanosec)

        height, width = bgr.shape[:2]
        id_map = np.zeros((height, width), dtype=np.uint16)
        for obj_id, mask in zip(detections["ids"], detections["masks"]):
            id_map[mask] = encode_instance_id(int(obj_id))
        map_msg = bridge.cv2_to_imgmsg(id_map, encoding="mono16")
        map_msg.header.stamp = image_msg.header.stamp
        map_msg.header.frame_id = "camera"

        payload = {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "entries": [
                {"id": int(i), "label": str(l), "confidence": float(c),
                 "bbox": [float(v) for v in b]}
                for i, l, c, b in zip(detections["ids"], detections["labels"],
                                      detections["confidences"], detections["bboxes"])
            ],
        }

        # log_time = the source camera frame's log_time, so a merged `ros2 bag play` emits
        # each detection at the instant its image appeared. The offline replay driver keys
        # off the header stamp instead and ignores this.
        writer.write(INSTANCE_MAP_TOPIC, serialize_message(map_msg), log_time_ns)
        writer.write(DETECTIONS_TOPIC,
                     serialize_message(String(data=json.dumps(payload))), log_time_ns)

        written += 1
        if written == 1 or written % 25 == 0:
            rate = written / max(time.perf_counter() - started, 1e-9)
            print(f"  {scene}: {written} frames written "
                  f"({len(detections['ids'])} dets this frame, {rate:.2f} fps)", flush=True)

    del writer                                 # flush and close before stamping the manifest
    elapsed = time.perf_counter() - started

    manifest = {
        "scene": scene,
        "companion_bag": os.path.basename(out_uri),
        "source_bag_dir": scene_dir,
        "camera_msgs_seen": seen,
        "frames_written": written,
        "stride": args.stride,
        "seconds": round(elapsed, 1),
        # Track continuity is broken at each session boundary, so a single physical object
        # can appear under several ids. That inflates the duplicate-instance rate the bench
        # reports; it is a property of the RECORDING, not of the mapper. Read it alongside
        # the score.
        "session_frames": args.session_frames,
        "session_restarts": restarts,
        "oom_recoveries": oom_recoveries,
        "max_object_id": max_seen_id,
        "prompts": table.prompts,
        "objects": objects,
        "variant": args.variant,
        "objects_override": args.objects,
        "background_ids": table.background_ids,
        # Stamped so replay_map3d uses the matching platform/extrinsics and window, and
        # so a companion bag can never be silently reused under a changed detector config.
        "config_path": resolve_config_path(args.config),
        "platform": config.get("platform"),
        "sam3_config": cfg,
        "runtime": config.get("runtime", {}),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"{scene}: wrote {written}/{seen} frames to {out_uri} in {elapsed / 60:.1f} min"
          + (f"  [{oom_recoveries} OOM recoveries]" if oom_recoveries else ""))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="arabic_room", help="scene name, or 'all'")
    ap.add_argument("--bags-dir", default="/data/bags")
    ap.add_argument("--prompts", default="/data/benchmark/bench_prompts.json")
    ap.add_argument("--config", default="sam3_mecanum_sim.yaml",
                    help="node config yaml (package config/ name, or an absolute path)")
    ap.add_argument("--stride", type=int, default=5,
                    help="process every Nth camera frame. Default 5 = 638 frames over the "
                         "13 benched scenes at 0.77 s spacing, still 2.6x denser than "
                         "the ~2 s spacing sam_node achieves in production, for 21 min "
                         "of GPU instead of 106. Use 1 only to check the bench is not "
                         "stride-limited")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing companion bag")
    ap.add_argument("--session-frames", type=int, default=20,
                    help="restart the SAM 3 session every N frames to bound tracker VRAM "
                         "(0 = never; raise it if your card has headroom, since each "
                         "restart breaks track continuity)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave scenes that already have a companion bag alone — makes a "
                         "'--scene all' run resumable after an interruption")
    ap.add_argument("--objects", default=None,
                    help="comma-separated prompt override, e.g. 'sofa,window,pillow,stool'. "
                         "Narrows a recording to a few classes so each can be debugged "
                         "against GT. Prefix a prompt with '-' for a background class. "
                         "Default: the curated per-scene set from --prompts")
    ap.add_argument("--variant", default=None,
                    help="suffix for the bag and manifest, e.g. 'focus4' -> "
                         "<scene>_sam3_focus4/. Keeps a narrowed recording from clobbering "
                         "the full-prompt one so both stay replayable")
    args = ap.parse_args(argv)
    if args.stride < 1:
        ap.error('--stride must be >= 1')

    if args.scene == "all":
        scenes = sorted(json.load(open(args.prompts))["scenes"])
    else:
        scenes = [args.scene]

    pending = []
    for scene in scenes:
        out_uri, _ = companion_paths(args.bags_dir, scene, args.variant)
        if args.skip_existing and os.path.exists(out_uri):
            print(f"=== {scene} === already recorded, skipping", flush=True)
            continue
        pending.append(scene)
    if not pending:
        print("nothing to record")
        return 0

    backend, config = load_backend(args)

    # One bad scene must not cost the others: this pass is tens of minutes of GPU.
    failures = {}
    for scene in pending:
        print(f"=== {scene} ===", flush=True)
        try:
            record_scene(scene, args, backend, config)
        except Exception as err:                      # noqa: BLE001 - report and continue
            failures[scene] = f"{type(err).__name__}: {err}"
            print(f"{scene}: FAILED — {failures[scene]}", flush=True)
        finally:
            # Fresh SAM 3 session per scene, unconditionally. Object ids are only
            # meaningful within a session, and the tracker's memory bank — the largest
            # thing on the GPU — grows with (objects x frames). Dropping it here means a
            # scene never inherits VRAM or tracker state from the one before, including
            # after a failure. The model stays loaded; only the session goes.
            backend.release()
            _free_vram()

    if failures:
        print(f"\n{len(failures)}/{len(scenes)} scene(s) failed:")
        for scene, err in failures.items():
            print(f"  {scene}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
