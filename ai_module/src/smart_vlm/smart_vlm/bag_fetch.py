#!/usr/bin/env python3
"""Download one recorded scene bag from Google Drive if it's missing.

Runs inside the ai_module container (bags are bind-mounted at /data/bags, which
persists to the host). Used by `bag_replay.launch` so a plain
`ros2 launch smart_vlm bag_replay.launch scene:=<name>` auto-fetches the scene.

Idempotent: no-op if <bags_dir>/<scene>/metadata.yaml already exists.

  ros2 run smart_vlm bag_fetch <scene> [--bags-dir /data/bags]

The manifest <bags_dir>/scenes.yaml maps each scene to a single Google Drive ZIP
(drive_id) + optional sha256. See docs/M0.5_rosbag_infra.md.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def die(msg: str) -> None:
    print(f"[bag_fetch] error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a scene bag if missing.")
    ap.add_argument("scene", help="scene name (a key under `scenes:` in scenes.yaml)")
    ap.add_argument("--bags-dir", default="/data/bags",
                    help="directory holding scenes.yaml and <scene>/ folders (default: /data/bags)")
    args = ap.parse_args()

    bags_dir = Path(args.bags_dir)
    scene = args.scene
    dest = bags_dir / scene
    if (dest / "metadata.yaml").is_file():
        print(f"[bag_fetch] {scene} already present at {dest} — nothing to do")
        return

    try:
        import yaml
    except ImportError:
        die("pyyaml not installed in the image (add it to ai_module/docker/Dockerfile)")
    try:
        import gdown
    except ImportError:
        die("gdown not installed in the image (add it to ai_module/docker/Dockerfile)")

    manifest = bags_dir / "scenes.yaml"
    if not manifest.is_file():
        die(f"manifest not found: {manifest} (is <repo>/bags mounted at {bags_dir}?)")
    scenes = (yaml.safe_load(manifest.read_text()) or {}).get("scenes") or {}
    if scene not in scenes:
        available = ", ".join(sorted(scenes)) or "(none)"
        die(f"scene '{scene}' not in {manifest}. Known scenes: {available}")

    entry = scenes[scene] or {}
    drive_id = str(entry.get("drive_id") or "").strip()
    if not drive_id or drive_id.startswith("<"):
        die(f"scene '{scene}' has no drive_id yet in {manifest} "
            f"(see 'Publishing a new scene' in docs/M0.5_rosbag_infra.md)")
    expected_sha = str(entry.get("sha256") or "").strip()

    bags_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bags_dir) as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / f"{scene}.zip"

        print(f"[bag_fetch] downloading {scene} (drive_id={drive_id}) ...")
        out = gdown.download(id=drive_id, output=str(zip_path), quiet=False)
        if not out or not zip_path.is_file():
            die("download failed — check the drive_id and that the file is shared "
                "'Anyone with the link'")

        if expected_sha and not expected_sha.startswith("<"):
            actual = sha256_of(zip_path)
            if actual != expected_sha:
                die(f"sha256 mismatch for {scene}: expected {expected_sha}, got {actual}")
            print("[bag_fetch] sha256 ok")
        else:
            print("[bag_fetch] no sha256 in manifest — skipping integrity check")

        stage = tmp_dir / "unzipped"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(stage)
        meta = next(iter(stage.rglob("metadata.yaml")), None)
        if meta is None:
            die(f"no metadata.yaml found inside {scene}.zip — is it a ros2 bag?")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(meta.parent), str(dest))

    print(f"[bag_fetch] ready: {dest}")


if __name__ == "__main__":
    main()
