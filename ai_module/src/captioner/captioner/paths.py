"""Path validation for user-supplied file paths (ROS payloads, CLI args).

The VQA server and the category-1 reasoner both accept image paths from other
processes, so a path has to be confined to the mounts we actually publish before
it reaches `open()`. Keeping the roots in one place means the compose mounts and
the guard cannot drift apart.

Deliberately dependency-free (stdlib only) so smart_vlm's pure-python helpers and
their tests can import it without pulling in torch or rclpy.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote

# Writable roots the pipeline may read/write. Official compose mounts none of
# these; /home/docker is the image workspace and always exists. Dev compose also
# bind-mounts ../data -> /data.
_CANDIDATE_ROOTS = (
    Path("/data"),
    Path("/home/docker"),
    Path("/tmp"),
)

# Resolved once at import: the mount set is fixed for the life of the container,
# and re-running this per request had every VQA call stat the filesystem.
ALLOWED_ROOTS: tuple[Path, ...] = tuple(
    p.resolve() for p in _CANDIDATE_ROOTS if p.exists()
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def secure_path(user_path: str | os.PathLike, roots: tuple[Path, ...] | None = None) -> Path:
    """Resolve `user_path` and confirm it lands inside an allowed mount.

    Raises ValueError on a traversal attempt and PermissionError when the
    resolved path escapes every root. Symlinks are followed by resolve() first,
    so a link pointing out of /data is rejected too.
    """
    decoded = unquote(str(user_path))
    if ".." in Path(decoded).parts:
        raise ValueError(f"Path traversal rejected: {user_path}")

    resolved = Path(decoded).expanduser().resolve()
    allowed = roots if roots is not None else ALLOWED_ROOTS
    if not allowed:
        # Every path would be rejected below, with a message that blames the path
        # rather than the real cause: this is not the container, or /data is unmounted.
        raise RuntimeError(
            f"No allowed data roots exist ({[str(p) for p in _CANDIDATE_ROOTS]}). "
            "Expected /home/docker (image workspace) or /data (dev mount).")
    if not any(_is_under(resolved, root) for root in allowed):
        raise PermissionError(
            f"Path is not under an allowed mount {[str(r) for r in allowed]}: {resolved}")
    return resolved


def secure_image_path(user_path: str | os.PathLike,
                      roots: tuple[Path, ...] | None = None) -> Path:
    """secure_path() plus an existence check, for paths we are about to read."""
    path = secure_path(user_path, roots)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def secure_output_path(user_path: str | os.PathLike,
                       roots: tuple[Path, ...] | None = None) -> Path:
    """secure_path() for a file we are about to write; parents are created."""
    path = secure_path(user_path, roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
