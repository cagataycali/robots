"""Tests for strands_robots.policies — behavior-focused tests for the policy system."""

import asyncio

import pytest

from strands_robots.policies import MockPolicy, create_policy, list_providers, register_policy


class TestMockPolicy:
    """MockPolicy should produce deterministic sinusoidal trajectories."""

    def test_full_lifecycle(self):
        """Create → set keys → get actions → verify structure and determinism."""
        p = create_policy("mock")
        assert isinstance(p, MockPolicy)
        assert p.provider_name == "mock"

        p.set_robot_state_keys(["j0", "j1", "j2"])

        obs = {"observation.state": [0.0, 0.0, 0.0]}
        actions = asyncio.run(p.get_actions(obs, "pick up the block"))

        # 8-step horizon, each action has all 3 keys
        assert len(actions) == 8
        assert set(actions[0].keys()) == {"j0", "j1", "j2"}

        # Deterministic — calling again from fresh policy with same state gives same output
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
        """All action values should stay within ±0.6 (amplitude 0.5 + margin)."""
        p = MockPolicy()
        p.set_robot_state_keys(["j0", "j1"])
        # Run for multiple steps to exercise different phases
        for _ in range(10):
            actions = p.get_actions_sync({"observation.state": [0, 0]}, "test")
            for a in actions:
                for v in a.values():
                    assert -0.6 <= v <= 0.6, f"Value {v} out of bounds"


class TestCreatePolicy:
    """create_policy() should resolve shorthands, URLs, and custom registrations."""

    def test_register_and_create_custom_provider(self):
        """Runtime-registered providers should be creatable by name and alias."""
        register_policy("custom_test", loader=lambda: MockPolicy, aliases=["ct"])

        p1 = create_policy("custom_test")
        assert isinstance(p1, MockPolicy)

        p2 = create_policy("ct")
        assert isinstance(p2, MockPolicy)

    def test_list_providers_includes_json_and_runtime(self):
        """list_providers() should include both JSON-defined and runtime providers."""
        register_policy("runtime_only_provider", loader=lambda: MockPolicy)
        providers = list_providers()
        assert "mock" in providers
        assert "groot" in providers
        assert "runtime_only_provider" in providers

    def test_unknown_provider_raises(self):
        """Unknown provider should raise, not silently fail."""
        with pytest.raises(Exception):
            create_policy("nonexistent_provider_xyz_123")
