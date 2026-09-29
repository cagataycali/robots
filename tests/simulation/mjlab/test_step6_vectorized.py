"""Step 6: per-world domain randomization and the vectorized evaluator on the mjlab backend."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from strands_robots.simulation import create_simulation
from strands_robots.simulation.base import SimEngine

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None and importlib.util.find_spec("mujoco_warp") is not None


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


pytestmark = [
    pytest.mark.skipif(not _HAS_MJLAB, reason="mjlab not installed (pip install 'strands-robots[sim-mjlab]')"),
    pytest.mark.skipif(not _cuda(), reason="mjlab backend needs a CUDA device"),
]


class _HoldPolicy:
    """Non-batched Policy stand-in: hold the first observed pose (exercises the gather path)."""

    def __init__(self, joints: list[str]) -> None:
        self.joints = joints
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.calls += 1
        return [{j: float(observation_dict[j]) for j in self.joints}]


class _BatchedHoldPolicy(_HoldPolicy):
    async def get_actions_batch(self, observations, instruction, kwargs_per_world=None):
        self.calls += 1
        return [{j: float(o[j]) for j in self.joints} for o in observations]


@pytest.fixture(scope="module")
def eng():
    sim = create_simulation("mjlab", num_envs=6)
    sim.create_world()
    sim.add_robot("so101")
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.25, 0.0, 0.02], mass=0.05)
    sim.reset()
    yield sim
    sim.cleanup()


def test_randomize_physics_draws_per_world_and_is_reproducible(eng: SimEngine) -> None:
    r1 = eng.randomize(randomize_physics=True, seed=3)
    assert r1["status"] == "success", r1
    j1 = r1["content"][1]["json"]
    fric = np.asarray(j1["friction_scales"])
    mass = np.asarray(j1["mass_scales"])
    assert fric.shape[0] == 6 and mass.shape[0] == 6
    # Different worlds got different draws.
    assert fric[:, 0].std() > 0.05 and mass[:, 1].std() > 0.05
    # The GPU model carries them, derived from the compiled defaults (no compounding on repeat).
    m = eng.sim.model
    bid = j1["body_ids"][3]
    default = float(eng.sim.get_default_field("body_mass")[bid])
    got = m.body_mass[:, bid].cpu().numpy()
    assert np.allclose(got, mass[:, 3] * default, atol=1e-6)
    sub = m.body_subtreemass[:, j1["body_ids"][1]].cpu().numpy()
    assert sub.std() > 1e-3, "set_const must run per world (subtree mass differs across worlds)"
    r2 = eng.randomize(randomize_physics=True, seed=3)
    assert np.allclose(np.asarray(r2["content"][1]["json"]["friction_scales"]), fric)
    assert np.allclose(m.body_mass[:, bid].cpu().numpy(), got)
    json.dumps(j1)  # tool-envelope friendly


def test_randomize_positions_moves_objects_not_robots(eng: SimEngine) -> None:
    r = eng.randomize(randomize_positions=True, position_noise=0.03, seed=5)
    assert r["status"] == "success", r
    off = np.asarray(r["content"][1]["json"]["position_offsets"]["cube"])
    assert off.shape == (6, 3) and np.all(off[:, 2] == 0) and off[:, :2].std() > 0.005
    cube = eng.scene["cube"].data.root_link_pos_w.cpu().numpy() - eng.scene.env_origins.cpu().numpy()
    assert np.allclose(cube[:, :2], np.array([0.25, 0.0]) + off[:, :2], atol=1e-4)
    assert "so101" not in r["content"][1]["json"]["position_offsets"]


def test_randomize_refuses_unsupported_axes_and_unknown_keywords(eng: SimEngine) -> None:
    assert eng.randomize(randomize_colors=True)["status"] == "error"
    assert eng.randomize(randomize_lighting=True)["status"] == "error"
    bad = eng.randomize(randomize_physics=True, frictoin_range=(0.5, 1.5))
    assert bad["status"] == "error" and "frictoin_range" in bad["content"][0]["text"]
    assert eng.randomize(randomize_physics="false")["status"] == "error"
    assert eng.randomize(randomize_physics=True, mass_range=(0.0, 1.0))["status"] == "error"
    noop = eng.randomize()
    assert noop["status"] == "success" and "nothing randomized" in noop["content"][0]["text"]


@pytest.mark.parametrize("batched", [False, True])
def test_vec_rollout_drives_every_world_and_prices_the_batched_path(eng: SimEngine, batched: bool) -> None:
    from strands_robots.training.mjlab_tasks.vec_eval import vec_rollout

    joints = list(eng.robot_joint_names("so101"))
    policy = (_BatchedHoldPolicy if batched else _HoldPolicy)(joints)
    res = asyncio.run(vec_rollout(eng, policy, robot_name="so101", ticks=10, control_hz=50.0))
    assert res.num_envs == 6 and res.ticks == 10 and res.batched_policy is batched
    assert len(res.final_obs) == 6 and all(j in f for f in res.final_obs for j in joints)
    assert policy.calls == (10 if batched else 60)
    assert res.summary()["episodes_per_minute"] > 0


def test_batched_recorder_flushes_n_worlds_as_n_lerobot_episodes(eng: SimEngine, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from strands_robots.training.mjlab_tasks.vec_eval import BatchedLeRobotRecorder, open_recorder, vec_rollout

    joints = list(eng.robot_joint_names("so101"))
    buf = BatchedLeRobotRecorder(6, joints, eng.robot_action_keys("so101"), "hold")
    policy = _BatchedHoldPolicy(joints)
    res = asyncio.run(vec_rollout(eng, policy, robot_name="so101", ticks=12, control_hz=50.0, recorder=buf))
    assert buf.frames_buffered == 6 * 12
    recorder = open_recorder(eng, "test/vec", tmp_path / "ds", 50, "hold")
    flushed = buf.flush(recorder)
    recorder.finalize()
    assert flushed == {"episodes": 6, "frames": 72, "flush_s": flushed["flush_s"]}
    files = sorted((tmp_path / "ds" / "data").rglob("*.parquet"))
    rows = sum(pq.read_metadata(f).num_rows for f in files)
    assert rows == 72, files
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 6 and info["total_frames"] == 72 and info["fps"] == 50
    assert res.episodes_per_minute > 0
