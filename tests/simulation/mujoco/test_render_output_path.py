"""Security regression tests for render(output_path=...) path hardening.

Exercises the pure helpers behind ``MuJoCoSimEngine.render(output_path=...)``
(:func:`_validate_render_output_path`, :func:`~strands_robots.simulation.safe_output.atomic_write_bytes`,
:func:`_save_render_png`). These are GL-free so they run in CI without an
OpenGL context and on both Linux and macOS.
"""

import os
import sys

import pytest

from strands_robots.simulation.mujoco import rendering
from strands_robots.simulation.mujoco.rendering import (
    _save_render_png,
    _validate_render_output_path,
)
from strands_robots.simulation.safe_output import atomic_write_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the render sandbox at a temp dir; ensure absolute opt-out is unset."""
    root = tmp_path / "renders"
    root.mkdir()
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", raising=False)
    return root


def test_accepts_path_inside_sandbox(sandbox):
    """A path inside the sandbox resolves and writes with 0o600 perms."""
    target = sandbox / "sub" / "ok.png"
    saved = _save_render_png(str(target), PNG)
    assert os.path.realpath(saved) == os.path.realpath(str(target))
    assert target.read_bytes() == PNG
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    # the directory we created is 0o700 (owner-only)
    assert oct((sandbox / "sub").stat().st_mode & 0o777) == "0o700"


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "../../etc/passwd",
        "sub/../../../etc/passwd",
        "/etc/cron.d/x",
        "~/../../etc/passwd",
    ],
)
def test_rejects_traversal_and_absolute_escape(sandbox, bad):
    """`..` escapes and absolute paths outside the sandbox are rejected."""
    with pytest.raises(ValueError):
        _validate_render_output_path(bad)


def test_rejects_backslash_separators(sandbox):
    """Windows-style backslash traversal slips a `/`-only check; reject it."""
    with pytest.raises(ValueError, match="backslash"):
        _validate_render_output_path("..\\..\\etc\\passwd")


@pytest.mark.parametrize("meta", [";", "|", "$", "`", ">", "<", "\n", "\r", "\x00"])
def test_rejects_shell_metacharacters(sandbox, meta):
    """Shell metacharacters and NULs are rejected up front."""
    with pytest.raises(ValueError, match="metacharacters"):
        _validate_render_output_path(f"shot{meta}.png")


def test_rejects_symlink_target(sandbox):
    """A symlink planted at the target is refused (arbitrary-write vector)."""
    real = sandbox / "real.png"
    real.write_bytes(b"secret")
    link = sandbox / "link.png"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        _validate_render_output_path(str(link))


def test_absolute_opt_in_allows_outside_sandbox(tmp_path, monkeypatch):
    """STRANDS_ROBOTS_RENDER_ALLOW_ABS=1 permits paths outside the sandbox."""
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", "1")
    target = tmp_path / "elsewhere" / "shot.png"
    saved = _save_render_png(str(target), PNG)
    assert os.path.realpath(saved) == os.path.realpath(str(target))
    assert target.read_bytes() == PNG


def test_rejects_oversized_payload(sandbox, monkeypatch):
    """A PNG larger than the configured cap is refused without writing."""
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "16")
    target = sandbox / "big.png"
    with pytest.raises(ValueError, match="exceeds limit"):
        _save_render_png(str(target), b"y" * 64)
    assert not target.exists()


def test_invalid_max_bytes_env_rejected(sandbox, monkeypatch):
    """A malformed size cap surfaces as an error rather than silently defaulting."""
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "not-a-number")
    with pytest.raises(ValueError, match="STRANDS_ROBOTS_RENDER_MAX_BYTES"):
        _save_render_png(str(sandbox / "x.png"), PNG)


def test_atomic_write_preserves_existing_on_failure(sandbox, monkeypatch):
    """If os.replace fails mid-write, the existing target file is untouched."""
    target = sandbox / "keep.png"
    target.write_bytes(b"ORIGINAL")

    def boom(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(rendering.os, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_bytes(target, b"NEW-DATA-THAT-SHOULD-NOT-LAND")

    # Original content preserved; no stray temp files left behind.
    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
def test_created_file_and_dir_permissions(sandbox):
    """Created files are 0o600 and freshly created dirs are 0o700 (owner-only)."""
    target = sandbox / "deep" / "shot.png"
    _save_render_png(str(target), PNG)
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    assert oct((sandbox / "deep").stat().st_mode & 0o777) == "0o700"
