"""Unit tests for the shared path guard (no ROS / GPU / torch)."""
from __future__ import annotations

from pathlib import Path

import pytest

from captioner.paths import secure_image_path, secure_path


@pytest.fixture()
def roots(tmp_path):
    """Stand in for the /data mount so tests don't depend on the container."""
    (tmp_path / "inside").mkdir()
    return (tmp_path.resolve(),)


def test_accepts_path_inside_root(tmp_path, roots):
    target = tmp_path / "inside" / "crop.png"
    target.write_bytes(b"x")
    assert secure_path(target, roots) == target.resolve()


def test_rejects_traversal_segment(roots):
    with pytest.raises(ValueError):
        secure_path("/data/../etc/passwd", roots)


def test_rejects_path_outside_roots(roots):
    with pytest.raises(PermissionError):
        secure_path("/etc/hostname", roots)


def test_rejects_sibling_with_shared_prefix(tmp_path, roots):
    """`/tmp/xyz` must not pass just because `/tmp/xy` is a root prefix."""
    sibling = Path(str(tmp_path) + "_evil") / "f.png"
    with pytest.raises(PermissionError):
        secure_path(sibling, roots)


def test_rejects_symlink_escaping_root(tmp_path, roots):
    outside = tmp_path.parent / "outside_secret.png"
    outside.write_bytes(b"x")
    link = tmp_path / "inside" / "link.png"
    link.symlink_to(outside)
    # resolve() follows the link first, so the escape is caught.
    with pytest.raises(PermissionError):
        secure_path(link, roots)


def test_image_path_requires_existing_file(tmp_path, roots):
    with pytest.raises(FileNotFoundError):
        secure_image_path(tmp_path / "inside" / "missing.png", roots)
