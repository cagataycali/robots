"""Recording must capture camera frames even when the policy skips images.

``PolicyRunner`` sets ``skip_images = not policy.requires_images`` to avoid
rendering when the policy does not consume pixels. The default ``mock`` policy
(and any non-VLA, proprioceptive-only policy) reports ``requires_images=False``,
so the runner asks the backend for a pixel-free observation.

While a dataset recording is active, the runner must render anyway: the
recorder writes the observation's camera ndarrays into the dataset's declared
video features, so a pixel-free observation produces frames with correct
episode counts but no pixels - a silently corrupt behavioural-cloning dataset.
``PolicyRunner._observe`` asks for the cameras on every backend, and
``get_observation(skip_images=True)`` itself renders nothing. These tests pin
that contract on the real Newton engine.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine_with_camera():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    sim.add_robot("so101")
    sim.add_camera("front", position=[0.0, -0.6, 0.4], target=[0.0, 0.0, 0.1])
    yield sim
    sim.destroy()


def test_skip_images_drops_cameras_without_recording(engine_with_camera):
    """The render-skip optimization still applies when nothing is recording."""
    obs = engine_with_camera.get_observation(skip_images=True)
    assert "front" not in obs, "skip_images should drop camera frames when not recording"


def test_full_observation_includes_camera(engine_with_camera):
    """skip_images=False renders the camera into the observation (sanity)."""
    obs = engine_with_camera.get_observation(skip_images=False)
    assert isinstance(obs.get("front"), np.ndarray)
    assert obs["front"].ndim == 3 and obs["front"].shape[2] == 3


def test_only_the_recording_rollout_renders(engine_with_camera):
    """skip_images=True renders nothing; the rollout that records asks for the cameras.

    Pre-fix the Newton backend honored ``skip_images`` literally and the
    rollout never asked again, so the recorder wrote frames with no pixels.
    """
    from strands_robots.simulation.policy_runner import PolicyRunner

    sim = engine_with_camera
    assert sim._world is not None
    sim._world._backend_state["recording"] = True
    assert "front" not in sim.get_observation(skip_images=True)
    obs = PolicyRunner(sim)._observe(None, skip_images=True)
    assert isinstance(obs.get("front"), np.ndarray), (
        "an active recording must make the rollout render so declared camera features receive real pixels"
    )
    assert obs["front"].ndim == 3 and obs["front"].shape[2] == 3
