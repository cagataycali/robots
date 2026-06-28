"""Newton dataset-recording correctness (start/stop/save_episode -> parquet).

Verifies that the Newton backend records a LeRobotDataset that satisfies the
canonical parquet-correctness contract: a session of N rollouts, each flushed
with ``save_episode``, must produce a dataset with ``total_episodes == N``,
exactly ``N`` rows in the episode metadata parquet, and ``N`` distinct
``episode_index`` values (no merged ``episode_index=0`` mega-episode).

The engine is exercised through a hand-built ``SimWorld`` so the recording
lifecycle runs without the optional Newton/Warp physics stack (the recorder and
episode bookkeeping are engine-independent). The per-step capture hook is the
real ``NewtonSimEngine._make_run_policy_hook`` closure.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("pyarrow")

from strands_robots.dataset_recorder import read_dataset_episode_indices
from strands_robots.simulation.models import SimCamera, SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _make_engine(world: SimWorld) -> NewtonSimEngine:
    """Build a NewtonSimEngine bound to ``world`` without the Warp stack.

    ``NewtonSimEngine.__init__`` imports Newton/Warp, which are optional; the
    recording lifecycle does not touch physics, so the engine is constructed
    via ``__new__`` and given just the attributes the recording path reads.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


def _world_with_robot(name: str = "so100") -> SimWorld:
    world = SimWorld()
    world.robots[name] = SimRobot(
        name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
    )
    return world


def _drive_episode(engine: NewtonSimEngine, robot_name: str, instruction: str, n_frames: int) -> None:
    """Run one mock rollout: call the real on_frame hook ``n_frames`` times."""
    hook = engine._make_run_policy_hook(robot_name, instruction)
    assert hook is not None
    for step in range(n_frames):
        obs = {j: float(step) * 0.01 for j in _SO100_JOINTS}
        action = {j: float(step) * 0.01 + 0.001 for j in _SO100_JOINTS}
        hook(step, obs, action)


def test_three_episode_rollout_parquet_correctness(tmp_path):
    """3 rollouts -> 3 distinct episodes; parquet is the source of truth."""
    root = str(tmp_path / "newton_ds")
    engine = _make_engine(_world_with_robot())

    started = engine.start_recording(repo_id="local/newton_rec", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    n_episodes = 3
    frames_per_episode = 5
    for ep in range(n_episodes):
        _drive_episode(engine, "so100", f"episode {ep}", frames_per_episode)
        saved = engine.save_episode()
        assert saved["status"] == "success", saved

    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    # verify_dataset_episodes reads the parquet ground truth.
    verified = engine.verify_dataset_episodes(n_episodes)
    assert verified["status"] == "success", verified

    # Canonical correctness contract: total_episodes == N AND episode-parquet
    # num_rows == N AND len(unique(episode_index)) == N.
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == n_episodes
    assert len(set(info["episode_indices"])) == n_episodes
    assert sorted(info["episode_indices"]) == list(range(n_episodes))
    assert info["total_frames"] == n_episodes * frames_per_episode

    from pathlib import Path

    import pyarrow.parquet as pq

    parquets = sorted((Path(root) / "meta" / "episodes").glob("**/*.parquet"))
    num_rows = sum(pq.read_table(p).num_rows for p in parquets)
    assert num_rows == n_episodes


def test_stop_recording_flushes_trailing_episode(tmp_path):
    """stop_recording flushes the final unsaved rollout (no explicit save)."""
    root = str(tmp_path / "newton_trailing")
    engine = _make_engine(_world_with_robot())

    engine.start_recording(repo_id="local/newton_trail", root=root, fps=30, overwrite=True)
    _drive_episode(engine, "so100", "ep0", 4)
    engine.save_episode()
    _drive_episode(engine, "so100", "ep1", 4)
    # No save_episode here: stop_recording must flush the trailing episode.
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8


def test_camera_frames_recorded(tmp_path, monkeypatch):
    """A registered named camera is declared in the schema and captured."""
    root = str(tmp_path / "newton_cam")
    world = _world_with_robot()
    world.cameras["front"] = SimCamera(name="front", width=64, height=48)
    engine = _make_engine(world)

    # Stub render so the capture path runs without the Warp ray tracer.
    fake_img = np.zeros((48, 64, 3), dtype=np.uint8)

    def _fake_render(camera_name="default", width=None, height=None):
        return {
            "status": "success",
            "content": [{"image": {"format": "png", "source": {"bytes": _encode_png(fake_img)}}}],
        }

    monkeypatch.setattr(engine, "render", _fake_render)

    started = engine.start_recording(repo_id="local/newton_cam", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    assert "observation.images.front" in recorder.dataset.features

    _drive_episode(engine, "so100", "with camera", 4)
    engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped

    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 4


def test_multi_robot_namespaced_schema(tmp_path):
    """Two robots -> joint ids are namespaced ``robot__joint`` in the schema."""
    root = str(tmp_path / "newton_multi")
    world = SimWorld()
    for name in ("alice", "bob"):
        world.robots[name] = SimRobot(
            name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
        )
    engine = _make_engine(world)

    started = engine.start_recording(repo_id="local/newton_multi", root=root, fps=30, overwrite=True)
    assert started["status"] == "success", started

    recorder = world._backend_state["dataset_recorder"]
    state_names = recorder.dataset.features["observation.state"]["names"]
    assert "alice__Rotation" in state_names
    assert "bob__Jaw" in state_names
    assert len(state_names) == 2 * len(_SO100_JOINTS)

    # Drive both robots and confirm two episodes record cleanly.
    for ep, rname in enumerate(("alice", "bob")):
        _drive_episode(engine, rname, f"ep {ep}", 3)
        engine.save_episode()
    stopped = engine.stop_recording()
    assert stopped["status"] == "success", stopped
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 2


def test_start_recording_without_world_errors():
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = None
    engine._model = None
    result = engine.start_recording(repo_id="local/nope")
    assert result["status"] == "error"
    assert "No world" in result["content"][0]["text"]


def test_save_episode_without_recording_errors():
    engine = _make_engine(_world_with_robot())
    result = engine.save_episode()
    assert result["status"] == "error"
    assert "not recording" in result["content"][0]["text"].lower()


def test_stop_recording_when_idle_is_graceful():
    engine = _make_engine(_world_with_robot())
    result = engine.stop_recording()
    assert result["status"] == "success"
    assert "not recording" in result["content"][0]["text"].lower()


def _encode_png(img: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()
