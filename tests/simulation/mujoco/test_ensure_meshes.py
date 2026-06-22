"""Behavior tests for ``MuJoCoSimEngine._ensure_meshes``.

``_ensure_meshes`` checks whether the mesh files (``.stl``/``.obj``) referenced
by a model XML are present on disk and, if any are missing, triggers a one-time
Menagerie auto-download. Its contract (see the method docstring) is:

* return ``None`` when every referenced mesh is already present (or downloads
  cleanly), so ``add_robot`` proceeds;
* return a standard ``{"status": "error", ...}`` dict when the auto-download
  fails, so the caller can propagate a clear message to the agent instead of
  letting MuJoCo raise a cryptic 'mesh not found'.

These paths were previously unexercised. The download-failure branch is the
one callers MUST propagate, so it is pinned here explicitly.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_ensure_meshes = MuJoCoSimEngine._ensure_meshes


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_no_mesh_references_returns_none(tmp_path):
    """A model that references no mesh files needs no download."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>',
    )
    assert _ensure_meshes(model, "robot") is None


def test_all_meshes_present_returns_none(tmp_path, monkeypatch):
    """When every referenced mesh exists on disk, no download is attempted."""
    (tmp_path / "arm.stl").write_bytes(b"solid\n")
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="arm.stl"/></asset></mujoco>',
    )

    def _boom(*a, **k):
        raise AssertionError("download must not be called when meshes are present")

    monkeypatch.setattr("strands_robots.assets.download.download_robots", _boom)
    assert _ensure_meshes(model, "robot") is None


def test_meshdir_is_honored_when_resolving_mesh_paths(tmp_path, monkeypatch):
    """A ``meshdir`` attribute is joined onto the mesh path before existence check."""
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "arm.stl").write_bytes(b"solid\n")
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><compiler meshdir="meshes"/><asset><mesh file="arm.stl"/></asset></mujoco>',
    )

    def _boom(*a, **k):
        raise AssertionError("mesh resolves under meshdir; no download expected")

    monkeypatch.setattr("strands_robots.assets.download.download_robots", _boom)
    assert _ensure_meshes(model, "robot") is None


def test_missing_mesh_triggers_successful_download(tmp_path, monkeypatch):
    """A missing mesh triggers auto-download; a clean download yields ``None``."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )
    calls = {}

    def _ok(names, force):
        calls["names"] = names
        calls["force"] = force

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _ok)

    assert _ensure_meshes(model, "so100") is None
    assert calls == {"names": ["so100"], "force": True}


def test_missing_mesh_download_failure_returns_error_dict(tmp_path, monkeypatch):
    """When auto-download fails, an error dict (not ``None``) is returned."""
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )

    def _fail(names, force):
        raise OSError("network down")

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _fail)

    result = _ensure_meshes(model, "so100")
    assert isinstance(result, dict)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "so100" in text
    assert "network down" in text


def test_missing_mesh_in_included_file_is_detected(tmp_path, monkeypatch):
    """Mesh refs inside an ``<include>``d file are checked, not just the top XML."""
    _write(
        tmp_path / "parts.xml",
        '<mujoco><asset><mesh file="absent.stl"/></asset></mujoco>',
    )
    model = _write(
        tmp_path / "robot.xml",
        '<mujoco><include file="parts.xml"/></mujoco>',
    )
    seen = {}

    def _ok(names, force):
        seen["called"] = True

    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda n: n)
    monkeypatch.setattr("strands_robots.assets.download.download_robots", _ok)

    assert _ensure_meshes(model, "robot") is None
    assert seen.get("called") is True


def test_unreadable_model_path_is_tolerated(tmp_path):
    """A model path that cannot be opened is skipped, not raised on.

    Both the top-level include scan and the per-file mesh scan swallow read
    errors so a transient/odd path never crashes ``add_robot``; with nothing
    readable there is nothing to download, so ``None`` is returned.
    """
    missing_path = str(tmp_path / "does_not_exist.xml")
    assert _ensure_meshes(missing_path, "robot") is None
