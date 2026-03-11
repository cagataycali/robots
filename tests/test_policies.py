"""Tests for strands_robots.policies — MockPolicy, create_policy, registry.

Tests the unified registry approach where policy provider definitions
live in ``registry/policies.json`` and runtime registration is still
supported via ``register_policy()``.
"""

import asyncio

import pytest

from strands_robots.policies import (
    MockPolicy,
    Policy,
    create_policy,
    list_providers,
    register_policy,
)
from strands_robots.registry import (
    get_policy_provider,
    list_policy_providers,
    resolve_policy_string,
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
    def test_get_provider_config(self):
        config = get_policy_provider("groot")
        assert config is not None
        assert "port" in config["config_keys"]

    def test_aliases(self):
        config = get_policy_provider("lerobot")
        assert config is not None
        assert config["class"] == "LerobotLocalPolicy"

    def test_unknown_returns_none(self):
        config = get_policy_provider("nonexistent_provider_xyz")
        assert config is None

    def test_list_providers(self):
        providers = list_policy_providers()
        assert "mock" in providers
        assert "groot" in providers
        assert "lerobot" in providers  # alias
        assert "cosmos" in providers  # alias

    def test_resolve_mock(self):
        p, kw = resolve_policy_string("mock")
        assert p == "mock"

    def test_resolve_hf_model(self):
        p, kw = resolve_policy_string("lerobot/act_aloha_sim")
        assert p == "lerobot_local"
        assert kw["pretrained_name_or_path"] == "lerobot/act_aloha_sim"

    def test_resolve_server_address(self):
        p, kw = resolve_policy_string("localhost:8080")
        assert p == "lerobot_async"
        assert kw["server_address"] == "localhost:8080"


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
