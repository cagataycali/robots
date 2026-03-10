"""Tests for strands_robots.policies — registry, MockPolicy, create_policy."""

import asyncio
import math

import pytest

from strands_robots.policies import (
    MockPolicy,
    Policy,
    PolicyRegistry,
    create_policy,
    list_providers,
    register_policy,
)


class TestPolicyABC:
    def test_policy_is_abstract(self):
        with pytest.raises(TypeError):
            Policy()

    def test_policy_has_required_methods(self):
        assert hasattr(Policy, "get_actions")
        assert hasattr(Policy, "set_robot_state_keys")
        assert hasattr(Policy, "provider_name")


class TestMockPolicy:
    def test_init(self):
        p = MockPolicy()
        assert p.provider_name == "mock"
        assert p.robot_state_keys == []

    def test_set_robot_state_keys(self):
        p = MockPolicy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        assert p.robot_state_keys == ["j0", "j1", "j2"]

    def test_get_actions_async(self):
        p = MockPolicy()
        p.set_robot_state_keys(["j0", "j1", "j2"])
        obs = {"observation.state": [0.0, 0.0, 0.0]}
        actions = asyncio.run(p.get_actions(obs, "test"))
        assert len(actions) == 8
        assert all(isinstance(a, dict) for a in actions)
        assert all("j0" in a and "j1" in a and "j2" in a for a in actions)

    def test_get_actions_sync(self):
        p = MockPolicy()
        p.set_robot_state_keys(["j0", "j1"])
        obs = {"observation.state": [0.0, 0.0]}
        actions = p.get_actions_sync(obs, "move")
        assert len(actions) == 8

    def test_actions_are_sinusoidal(self):
        p = MockPolicy()
        p.set_robot_state_keys(["j0"])
        obs = {"observation.state": [0.0]}
        actions = p.get_actions_sync(obs, "test")
        vals = [a["j0"] for a in actions]
        # Sinusoidal values should be bounded in [-0.5, 0.5]
        assert all(-0.6 <= v <= 0.6 for v in vals)

    def test_auto_generates_keys(self):
        p = MockPolicy()
        obs = {"observation.state": [0.0, 0.0, 0.0, 0.0]}
        actions = p.get_actions_sync(obs, "test")
        assert len(actions) == 8
        assert "joint_0" in actions[0]
        assert "joint_3" in actions[0]

    def test_default_6dof_when_no_state(self):
        p = MockPolicy()
        actions = p.get_actions_sync({}, "test")
        assert len(actions[0]) == 6


class TestPolicyRegistry:
    def test_register_and_get(self):
        reg = PolicyRegistry()
        reg.register("test_mock", loader=lambda: MockPolicy)
        cls = reg.get("test_mock")
        assert cls is MockPolicy

    def test_aliases(self):
        reg = PolicyRegistry()
        reg.register("test_full", loader=lambda: MockPolicy, aliases=["tf", "test_f"])
        assert reg.get("tf") is MockPolicy
        assert reg.get("test_f") is MockPolicy

    def test_unknown_raises(self):
        reg = PolicyRegistry()
        with pytest.raises(ValueError, match="Unknown policy provider"):
            reg.get("nonexistent_provider_xyz")

    def test_list_providers(self):
        reg = PolicyRegistry()
        reg.register("a", loader=lambda: MockPolicy)
        reg.register("b", loader=lambda: MockPolicy, aliases=["b_alias"])
        providers = reg.list_providers()
        assert "a" in providers
        assert "b" in providers
        assert "b_alias" in providers

    def test_contains(self):
        reg = PolicyRegistry()
        reg.register("x", loader=lambda: MockPolicy, aliases=["y"])
        assert "x" in reg
        assert "y" in reg
        assert "z" not in reg


class TestCreatePolicy:
    def test_create_mock(self):
        p = create_policy("mock")
        assert isinstance(p, MockPolicy)

    def test_list_providers_includes_builtins(self):
        providers = list_providers()
        assert "mock" in providers
        assert "groot" in providers
        assert "lerobot_local" in providers
        assert "lerobot" in providers  # alias

    def test_register_custom_provider(self):
        register_policy("my_test_provider", loader=lambda: MockPolicy, aliases=["mtp"])
        p = create_policy("my_test_provider")
        assert isinstance(p, MockPolicy)
        p2 = create_policy("mtp")
        assert isinstance(p2, MockPolicy)
