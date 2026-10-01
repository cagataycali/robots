"""``Robot(..., backend="mjlab")`` + ``tools.run_policy`` + the LeRobot recorder round-trip.

Requires mjlab + CUDA + the ``lerobot`` extra; skips otherwise.
Log: ~/.tiny/mjlab-20260928/logs/robot_smoke.log (first run of the smoke this test froze).
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile

import pytest

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None and importlib.util.find_spec("mujoco_warp") is not None
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False


pytestmark = [
    pytest.mark.skipif(not _HAS_MJLAB, reason="mjlab not installed (pip install 'strands-robots[sim-mjlab]')"),
    pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot extra not installed"),
    pytest.mark.skipif(not _cuda(), reason="mjlab backend needs a CUDA device"),
]


def test_robot_factory_returns_mjlab_engine():
    from strands_robots import Robot
    from strands_robots.simulation.mjlab import MjlabEngine

    sim = Robot("so101", backend="mjlab")
    try:
        assert isinstance(sim, MjlabEngine)
        assert sim.list_robots() == ["so101"]
        assert sim.describe()["backend"] == "mjlab"
    finally:
        sim.cleanup()


def test_run_policy_tool_records_n_distinct_episodes_with_camera():
    """Three episodes in ONE run_policy call land as three parquet episodes (parquet-truth)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots import Robot
    from strands_robots.tools.run_policy import run_policy

    sim = Robot("so101", backend="mjlab")
    root = tempfile.mkdtemp(prefix="mjlab_rp_")
    try:
        sim.add_camera("front", position=[0.8, -0.8, 0.6], target=[0, 0, 0.15], width=160, height=120)
        res = run_policy(
            sim,
            robot_name="so101",
            policy_provider="mock",
            n_episodes=3,
            n_steps=20,
            dataset_root=root,
            dataset_repo_id="local/mjlab_roundtrip",
            dataset_task="roundtrip",
            seed=0,
        )
        assert res["status"] == "success", res["content"][0]["text"]
        info = json.load(open(os.path.join(root, "meta", "info.json")))
        assert info["total_episodes"] == 3
        assert info["total_frames"] == 60
        assert info["features"]["observation.state"]["names"] == sim.robot_joint_names("so101")
        assert info["features"]["action"]["names"] == sim.robot_action_keys("so101")
        assert "observation.images.front" in info["features"]
        assert os.path.exists(os.path.join(root, "videos", "observation.images.front", "chunk-000", "file-000.mp4"))
    finally:
        sim.cleanup()


def test_reset_closes_the_open_episode():
    """reset() is an episode boundary, as on the MuJoCo/Newton backends."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots import Robot

    sim = Robot("so101", backend="mjlab")
    root = tempfile.mkdtemp(prefix="mjlab_reset_")
    try:
        assert sim.start_recording(repo_id="local/x", root=root, task="t", fps=30)["status"] == "success"
        r = sim.run_policy(
            robot_name="so101", policy_provider="mock", n_steps=10, control_frequency=30.0, fast_mode=True
        )
        assert r["status"] == "success", r["content"][0]["text"]
        note = sim.reset()["content"][0]["text"]
        assert "Reset 1 world(s)" in note
        status = sim.get_recording_status()
        assert status["status"] == "success"
        stop = sim.stop_recording()
        assert stop["status"] == "success", stop["content"][0]["text"]
        info = json.load(open(os.path.join(root, "meta", "info.json")))
        assert info["total_episodes"] == 1 and info["total_frames"] == 10
    finally:
        sim.cleanup()
