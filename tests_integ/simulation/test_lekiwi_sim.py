"""End-to-end: LeKiwi resolves from its GitHub asset source and simulates.

Network + MuJoCo integration. Exercises the full path that was broken when
``lekiwi`` was a hardware-only registry entry: registry lookup -> GitHub asset
auto-download (Ekumen-OS/lekiwi) -> MuJoCo compile -> step -> mock policy
rollout driving all 9 actuators -> render.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

# 6-DOF SO-ARM arm + 3 omniwheels.
_EXPECTED_ACTUATORS = 9


@pytest.fixture
def lekiwi_sim(tmp_path, monkeypatch):
    """Resolve + build LeKiwi in sim, downloading assets into an isolated cache."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    import strands_robots as sr
    from strands_robots.registry import reload

    reload()
    sim = sr.Robot("lekiwi", mode="sim")
    yield sim
    sim.destroy()


def test_lekiwi_builds_and_steps(lekiwi_sim) -> None:
    """LeKiwi compiles with 9 actuators and steps stably (no NaN)."""
    name = lekiwi_sim.list_robots()[0]
    assert name == "lekiwi"

    joints = lekiwi_sim.robot_joint_names(name)
    assert len(joints) == _EXPECTED_ACTUATORS, joints
    assert {"base_left_wheel_joint", "base_right_wheel_joint", "base_back_wheel_joint"} <= set(joints)

    lekiwi_sim.step(n_steps=300)
    state = lekiwi_sim.get_observation(robot_name=name, skip_images=True)
    assert all(np.isfinite(v) for v in state.values())


def test_lekiwi_mock_policy_rollout_and_render(lekiwi_sim) -> None:
    """A mock policy drives all 9 joints and the namespaced camera renders."""
    name = lekiwi_sim.list_robots()[0]

    result = lekiwi_sim.run_policy(
        robot_name=name,
        policy_provider="mock",
        instruction="drive forward and move the arm",
        duration=2.0,
        control_frequency=30.0,
    )
    assert result["status"] == "success", result

    render = lekiwi_sim.render(camera_name="lekiwi/front", width=320, height=240)
    assert render["status"] == "success", render
