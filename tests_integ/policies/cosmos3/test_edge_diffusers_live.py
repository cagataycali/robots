"""Live end-to-end test: ``nvidia/Cosmos3-Edge`` drives a MuJoCo Franka through ``run_policy``.

The in-process diffusers backend with ``ik=True`` and ``robot="franka"`` is the
route by which ``sim.run_policy(policy_provider="cosmos3", ...)`` closes the
loop without a RoboLab server. This test loads the real Edge weights, so it
needs a CUDA GPU, the checkpoint and ``diffusers>=0.41`` (Edge is built against
``0.40.0.dev0``; 0.39 leaves 112 transformer tensors unfilled and the backend
refuses the load). It is skipped unless enabled:

    COSMOS3_EDGE_LIVE=1 MUJOCO_GL=egl \\
    hatch run test-integ tests_integ/policies/cosmos3/test_edge_diffusers_live.py -v

Measured on a Jetson AGX Thor (torch 2.14.1+cu130, diffusers 0.41.0): load
17.9 s / 7.6 GiB resident, one ``[32, 10]`` chunk in ~20 s at 4 sampler steps
(the 33-frame world video is decoded on every chunk), peak 11.1 GiB.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

LIVE = os.environ.get("COSMOS3_EDGE_LIVE", "").lower() in ("1", "true", "yes")
MODEL = os.environ.get("COSMOS3_EDGE_MODEL", "nvidia/Cosmos3-Edge")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a CUDA GPU + nvidia/Cosmos3-Edge weights + diffusers>=0.41. Set COSMOS3_EDGE_LIVE=1 to enable.",
)

pytest.importorskip("diffusers", reason="diffusers not installed")
torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("mujoco", reason="cosmos3-sim extra (mujoco) not installed")
pytest.importorskip("mink", reason="cosmos3-sim extra (mink) not installed")

_CAMERAS = {
    "wrist": "observation/wrist_image_left",
    "front": "observation/exterior_image_1_left",
    "side": "observation/exterior_image_2_left",
}


def _scene():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot(name="arm", data_config="franka")
    sim.add_object(name="cube", shape="box", size=[0.02, 0.02, 0.02], position=[0.5, 0.0, 0.02], color=[1, 0, 0, 1])
    sim.add_camera(name="wrist", parent_body="arm/hand", position=[0.05, 0.0, 0.0], target=[0.0, 0.0, 0.3])
    sim.add_camera(name="front", position=[1.2, 0.0, 0.6], target=[0.5, 0.0, 0.1])
    sim.add_camera(name="side", position=[0.5, 1.0, 0.6], target=[0.5, 0.0, 0.1])
    sim.step(5)
    return sim


@pytest.fixture(scope="module")
def policy():
    """One Edge load for the module: zero meta tensors, the 2169 guard passed."""
    from strands_robots.policies.cosmos3 import Cosmos3Policy
    from strands_robots.policies.cosmos3.policy_diffusers import _unloaded_checkpoint_tensors

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    p = Cosmos3Policy(
        embodiment="droid",
        backend="diffusers",
        model=MODEL,
        robot="franka",
        ik=True,
        num_inference_steps=4,
        guidance_scale=3.0,
        observation_mapping=_CAMERAS,
    )
    assert p._diffusers is not None
    assert _unloaded_checkpoint_tensors(p._diffusers._pipeline) == []
    p.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    return p


def test_edge_chunk_decodes_to_finite_franka_joint_targets(policy):
    """A rendered franka observation -> raw [32, 10] chunk -> 32 joint-target dicts keyed by Panda actuators."""
    sim = _scene()
    obs = sim.get_observation("arm")
    steps = policy.get_actions_sync(obs, "pick up the red cube")
    assert policy.last_rollout is not None
    raw = np.asarray(policy.last_rollout["action"])
    assert raw.shape == (32, 10) and np.isfinite(raw).all() and np.any(raw != 0)
    assert len(steps) == 32
    expected = {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
    assert all(set(step) == expected for step in steps)
    targets = np.array([[s[f"joint{i}"] for i in range(1, 8)] for s in steps])
    assert np.isfinite(targets).all()
    q_now = np.array([obs[f"joint{i}"] for i in range(1, 8)])
    # The decoded trajectory departs from the current pose (a chunk that never moves is the silent-failure shape).
    assert np.abs(targets - q_now).max() > 1e-3
    ik = policy.last_rollout["ik"]
    assert ik["tracking_error"]["mean_mm"] < 45.0, ik["tracking_error"]
    fingers = np.array([s["finger_joint1"] for s in steps])
    assert (fingers >= 0.0).all() and (fingers <= 0.04).all()


def test_edge_drives_the_franka_closed_loop_through_run_policy(policy):
    """sim.run_policy with the real Edge policy object: the arm moves, no NaN, every action key resolves."""
    from strands_robots.simulation import RunPolicyStep

    sim = _scene()
    q_start = np.array([sim.get_observation("arm")[f"joint{i}"] for i in range(1, 8)])
    seen: list[dict] = []

    def observer(ev):
        if isinstance(ev, RunPolicyStep):
            seen.append(
                {
                    "q": [float(ev.observation[f"joint{i}"]) for i in range(1, 8)],
                    "unresolved": list(ev.unresolved_action_keys),
                }
            )

    result = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        instruction="pick up the red cube",
        n_steps=24,
        control_frequency=15.0,
        observer=observer,
    )
    assert result["status"] == "success", result
    assert len(seen) >= 24
    qs = np.array([s["q"] for s in seen])
    assert not np.isnan(qs).any()
    assert not any(s["unresolved"] for s in seen), "an action key no actuator consumed"
    # The step events carry the observation the CHUNK was computed from (one
    # inference serves all 24 steps, ``observation_is_chunk_reused``), so the
    # arm's motion is read back from the simulator, not from the events.
    # Measured on Thor: 0.38-0.67 rad of joint excursion per 24-step episode.
    q_end = np.array([sim.get_observation("arm")[f"joint{i}"] for i in range(1, 8)])
    assert np.isfinite(q_end).all()
    assert np.abs(q_end - q_start).max() > 1e-2, ("the arm did not move", q_start, q_end)
