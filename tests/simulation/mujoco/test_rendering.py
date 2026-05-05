"""Tests for multi-camera snapshot + background recording."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_requires_mujoco = pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)


@_requires_mujoco
def test_render_all_returns_every_camera(tmp_path: Path) -> None:
    """render_all() should return one image block per camera."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    # add 3 cameras
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_c", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.step(n_steps=5)

    r = sim.render_all(width=64, height=48)
    assert r["status"] == "success", r
    image_blocks = [c for c in r["content"] if isinstance(c, dict) and "image" in c]
    # Should include at least the 3 user-added cameras (plus any default)
    assert len(image_blocks) >= 3, f"expected >=3 image blocks, got {len(image_blocks)}"

    # Subset mode
    r2 = sim.render_all(cameras=["cam_a", "cam_c"], width=48, height=32)
    assert r2["status"] == "success"
    imgs = [c for c in r2["content"] if isinstance(c, dict) and "image" in c]
    assert len(imgs) == 2

    sim.destroy()


@_requires_mujoco
def test_start_stop_cameras_recording_writes_one_mp4_per_camera(tmp_path: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("top", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])
    sim.add_camera("side", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.step(n_steps=5)

    r = sim.start_cameras_recording(
        cameras=["top", "side"],
        output_dir=str(tmp_path),
        fps=20,
        width=64,
        height=48,
        name="integ_test",
    )
    assert r["status"] == "success", r

    # Let it record for ~0.4s of wall time
    time.sleep(0.4)

    status = sim.get_cameras_recording_status()
    assert status["status"] == "success"
    assert "🟢" in status["content"][0]["text"]

    stop = sim.stop_cameras_recording()
    assert stop["status"] == "success"

    # Two MP4 files should exist
    files = sorted(tmp_path.glob("*.mp4"))
    names = [f.name for f in files]
    assert any("top" in n for n in names), names
    assert any("side" in n for n in names), names
    for f in files:
        assert f.stat().st_size > 0, f"empty file: {f}"

    sim.destroy()


@_requires_mujoco
def test_stop_without_start_is_idempotent() -> None:
    """T16: idempotent - stop_cameras_recording without a running recording
    returns success with 'Was not recording' instead of erroring."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.stop_cameras_recording()
    assert r["status"] == "success"
    assert "Was not recording" in r["content"][0]["text"]
    sim.destroy()


@_requires_mujoco
def test_status_when_idle_is_success() -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    r = sim.get_cameras_recording_status()
    assert r["status"] == "success"
    assert "⚪" in r["content"][0]["text"]
    sim.destroy()
