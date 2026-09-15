"""Tests for ``strands_robots.policies.mock.MockPolicy``.

MockPolicy is the only non-ML policy provider - it generates smooth
sinusoidal actions and is the workhorse for every policy-runner / recording /
evaluate test in the suite.
"""

import asyncio
import logging

import pytest

from strands_robots.policies import (
    MockPolicy,
    create_policy,
)

# Detect groot-service availability for conditional test grouping.
try:
    import msgpack  # noqa: F401
    import zmq  # noqa: F401

    _groot_available = True
except ImportError:
    _groot_available = False


class TestMockPolicy:
    """MockPolicy should produce deterministic sinusoidal trajectories."""

    def test_full_lifecycle(self):
        """Create -> set keys -> get actions -> verify structure and determinism."""
        p = create_policy("mock")
        assert isinstance(p, MockPolicy)
        assert p.provider_name == "mock"

        p.set_robot_state_keys(["j0", "j1", "j2"])

        obs = {"observation.state": [0.0, 0.0, 0.0]}
        actions = asyncio.run(p.get_actions(obs, "pick up the block"))

        # 8-step horizon, each action has all 3 keys
        assert len(actions) == 8
        assert set(actions[0].keys()) == {"j0", "j1", "j2"}

        # Deterministic
        p2 = MockPolicy()
        p2.set_robot_state_keys(["j0", "j1", "j2"])
        actions2 = asyncio.run(p2.get_actions(obs, "different instruction"))
        assert actions == actions2

    def test_auto_generates_keys_from_observation(self):
        """When no keys are set, infers dimensionality from observation.state."""
        p = MockPolicy()
        obs = {"observation.state": [0.0] * 7}
        actions = p.get_actions_sync(obs, "test")
        assert len(actions[0]) == 7
        assert "joint_0" in actions[0] and "joint_6" in actions[0]

    def test_defaults_to_6dof(self):
        """With empty observation, defaults to 6-DOF."""
        p = MockPolicy()
        actions = p.get_actions_sync({}, "test")
        assert len(actions[0]) == 6

    def test_values_are_bounded_sinusoids(self):
        """All action values should stay within +/-0.6."""
        p = MockPolicy()
        p.set_robot_state_keys(["j0", "j1"])
        for _ in range(10):
            actions = p.get_actions_sync({"observation.state": [0, 0]}, "test")
            for a in actions:
                for v in a.values():
                    assert -0.6 <= v <= 0.6, f"Value {v} out of bounds"

    def test_get_actions_sync_works_from_sync_context(self):
        """get_actions_sync() should be usable from plain synchronous code."""
        p = MockPolicy()
        p.set_robot_state_keys(["a", "b"])
        actions = p.get_actions_sync({"observation.state": [0, 0]}, "move")
        assert len(actions) == 8
        assert all(isinstance(a, dict) for a in actions)


class TestMockPolicyStaysInsideTheActuatorRange:
    """The mock commands what the actuator can do, so the engine has nothing to warn about.

    ``examples/01_sim_hello_world.py`` printed three ctrlrange clamp warnings
    before its one line of expected output, because the mock's ±0.5 sinusoid
    did not fit the SO-100 ``Pitch`` / ``Jaw`` / ``Elbow`` ranges. The engine
    hands an opted-in policy the model through ``set_sim_context``; the mock
    now uses it to clip.
    """

    def test_hello_world_scene_emits_no_clamp_warning(self, caplog):
        pytest.importorskip("mujoco")
        from strands_robots import Robot

        sim = Robot("so100", mesh=False)
        try:
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
                result = sim.run_policy(robot_name="so100", policy_object=MockPolicy(), instruction="x", n_steps=50)
            assert result["status"] == "success"
            clamps = [r for r in caplog.records if "outside its ctrlrange" in r.getMessage()]
            assert clamps == [], [r.getMessage()[:80] for r in clamps]
        finally:
            sim.destroy()

    def test_values_are_clipped_to_the_learned_bounds(self):
        p = MockPolicy()
        p.set_robot_state_keys(["a", "b"])
        p._ctrl_bounds = {"a": (-0.1, 0.1)}
        actions = p.get_actions_sync({}, "x")
        assert all(-0.1 <= a["a"] <= 0.1 for a in actions)
        assert any(abs(a["b"]) > 0.1 for a in actions), "an unbounded key keeps the full sinusoid"

    def test_set_sim_context_without_a_matching_actuator_changes_nothing(self):
        p = MockPolicy()
        p.set_robot_state_keys(["nope"])
        p.set_sim_context(object(), "ghost/")  # not an MjModel: read fails, policy untouched
        assert p._ctrl_bounds == {}
