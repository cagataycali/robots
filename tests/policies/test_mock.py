"""Tests for ``strands_robots.policies.mock.MockPolicy``.

MockPolicy is the only non-ML policy provider - it generates smooth
sinusoidal actions and is the workhorse for every policy-runner / recording /
evaluate test in the suite.
"""

import asyncio

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
