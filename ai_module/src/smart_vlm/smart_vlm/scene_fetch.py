#!/usr/bin/env python3
"""Download one Unity scene from Google Drive if it's missing.

The twin of bag_fetch, for the live sim instead of offline replay. Runs inside the
ai_module container (scenes are bind-mounted at /data/scenes, which persists to the host);
scripts/eval/run_sim_sweep.py calls it for any scene a sweep needs and cannot find, so
`just eval-cat1-sim <scene>` works on a machine that has never seen the scene data.

It runs in the container rather than on the host for two reasons: gdown lives here, and
writing as this uid into the world-writable /data avoids handing the other side a
directory it cannot write into.

Idempotent: no-op if <scenes_dir>/<scene>/environment/Model.x86_64 already exists. That
file is the check because it is literally what the simulator executes — the scene IS the
Unity build (system_simulation_noviz.sh runs mesh/unity/environment/Model.x86_64), so a
scene folder without it is useless even if everything else unpacked.

  ros2 run smart_vlm scene_fetch <scene> [--scenes-dir /data/scenes]

The manifest <scenes_dir>/scenes.yaml maps each scene to a single Google Drive ZIP
(drive_id) + optional sha256, and documents how to regenerate the ids.
"""
import argparse
import hashlib
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

# What the simulator actually runs. Used both to detect an already-present scene and to
# find the payload root inside the zip.
SIM_BINARY = "Model.x86_64"


def die(msg: str) -> None:
    print(f"[scene_fetch] error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Unity scene if missing.")
    ap.add_argument("scene", help="scene name (a key under `scenes:` in scenes.yaml)")
    ap.add_argument("--scenes-dir", default="/data/scenes",
                    help="directory holding scenes.yaml and <scene>/ folders "
                         "(default: /data/scenes)")
    args = ap.parse_args()

    scenes_dir = Path(args.scenes_dir)
    scene = args.scene
    dest = scenes_dir / scene
    if (dest / "environment" / SIM_BINARY).is_file():
        print(f"[scene_fetch] {scene} already present at {dest} — nothing to do")
        return

    try:
        import yaml
    except ImportError:
        die("pyyaml not installed in the image (add it to ai_module/docker/Dockerfile)")
    try:
        import gdown
    except ImportError:
        die("gdown not installed in the image (add it to ai_module/docker/Dockerfile)")

    manifest = scenes_dir / "scenes.yaml"
    if not manifest.is_file():
        die(f"manifest not found: {manifest} (is <repo>/data mounted at /data?)")
    scenes = (yaml.safe_load(manifest.read_text()) or {}).get("scenes") or {}
    if scene not in scenes:
        available = ", ".join(sorted(scenes)) or "(none)"
        die(f"scene '{scene}' not in {manifest}. Known scenes: {available}")

    entry = scenes[scene] or {}
    drive_id = str(entry.get("drive_id") or "").strip()
    if not drive_id or drive_id.startswith("<"):
        die(f"scene '{scene}' has no drive_id yet in {manifest} "
            f"(regenerate the ids — the recipe is in that file's header)")
    expected_sha = str(entry.get("sha256") or "").strip()

    scenes_dir.mkdir(parents=True, exist_ok=True)
    # Stage inside scenes_dir so the final move is a rename on one filesystem rather than a
    # 300 MB copy across the bind mount, and so a failure leaves no half-scene behind.
    with tempfile.TemporaryDirectory(dir=scenes_dir) as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / f"{scene}.zip"

        print(f"[scene_fetch] downloading {scene} (drive_id={drive_id}) ~300 MB ...")
        out = gdown.download(id=drive_id, output=str(zip_path), quiet=False)
        if not out or not zip_path.is_file():
            die("download failed — check the drive_id and that the file is shared "
                "'Anyone with the link'")

        if expected_sha and not expected_sha.startswith("<"):
            actual = sha256_of(zip_path)
            if actual != expected_sha:
                die(f"sha256 mismatch for {scene}: expected {expected_sha}, got {actual}")
            print("[scene_fetch] sha256 ok")
        else:
            print("[scene_fetch] no sha256 in manifest — skipping integrity check")

        stage = tmp_dir / "unzipped"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(stage)
            # ZipFile.extractall does NOT restore Unix permissions, so everything lands
            # 0644 and the sim dies with "Permission denied" on ./Model.x86_64. Replay the
            # modes the archive recorded in its external attributes.
            for info in zf.infolist():
                mode = info.external_attr >> 16
                if mode:
                    target = stage / info.filename
                    if target.exists():
                        target.chmod(mode)

        # Find the payload by its marker rather than assuming the zip has a top-level
        # <scene>/ directory: these are the organizers' archives and the nesting is not
        # ours to rely on. The scene root is the parent of environment/.
        binary = next(iter(stage.rglob(SIM_BINARY)), None)
        if binary is None:
            die(f"no {SIM_BINARY} found inside {scene}.zip — is it a Unity scene build?")
        root = binary.parent.parent

        # Belt and braces: the sim binary MUST be executable, whatever the archive said
        # (a zip built on Windows records no Unix mode at all). Failing here beats a sim
        # that starts, prints "Permission denied" and leaves the sweep timing out on a
        # scene that never rendered.
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if not binary.stat().st_mode & stat.S_IXUSR:
            die(f"could not make {binary} executable")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(root), str(dest))

    print(f"[scene_fetch] ready: {dest}")


if __name__ == "__main__":
    main()
