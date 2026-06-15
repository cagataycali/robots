"""Live integration test for the Cosmos 3 in-process diffusers backend.

Unlike tests/policies/cosmos3/test_policy_diffusers.py (fully mocked), this test
actually loads the Cosmos3OmniPipeline weights via strands-diffusers and runs a
real in-process forward pass. It needs a CUDA GPU + the model weights, so it is
skipped by default. Parallel to tests_integ/groot/test_n17_live_server.py.

Enable with:

    COSMOS3_DIFFUSERS_LIVE=1 \
    hatch run test-integ tests_integ/policies/cosmos3/test_diffusers_backend_live.py -v

Optionally override the checkpoint with COSMOS3_MODEL (default nvidia/Cosmos3-Nano).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

LIVE = os.environ.get("COSMOS3_DIFFUSERS_LIVE", "").lower() in ("1", "true", "yes")
MODEL = os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a CUDA GPU + Cosmos 3 weights. Set COSMOS3_DIFFUSERS_LIVE=1 to enable.",
)

# Skip cleanly if the optional stack is missing.
pytest.importorskip("strands_diffusers", reason="strands-diffusers not installed")
pytest.importorskip("diffusers", reason="diffusers not installed")
pytest.importorskip("torch", reason="torch not installed")


def _obs() -> dict:
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs: dict[str, object] = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = 0.1 * i
    obs["gripper"] = 0.2
    return obs


@pytest.fixture(scope="module")
def policy():
    from strands_robots.policies.cosmos3 import Cosmos3Policy

    p = Cosmos3Policy(embodiment="droid", backend="diffusers", model=MODEL, robot="panda")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    return p


def test_policy_mode_returns_action_chunk_and_world_video(policy):
    """A real in-process policy run yields per-step actuator dicts AND surfaces
    the predicted world video on last_rollout."""
    out = policy.get_actions_sync(_obs(), "pick up the red cube")
    assert isinstance(out, list) and out
    step = out[0]
    assert set(step.keys()) == {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
    assert all(isinstance(v, float) for v in step.values())
    assert policy.last_rollout is not None
    assert policy.last_rollout["video"]  # an mp4/gif path
