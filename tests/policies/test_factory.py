"""Tests for ``strands_robots.policies.factory.create_policy``.

* provider resolution (mock / groot / lerobot_local)
* ``trust_remote_code`` security gate for HF-backed providers
* kwargs forwarding to the chosen provider
"""

import pytest

from strands_robots.policies import (
    MockPolicy,
    Policy,
    UntrustedRemoteCodeError,
    create_policy,
    list_providers,
    register_policy,
)

# Detect groot-service availability for conditional test grouping.
try:
    import msgpack  # noqa: F401
    import zmq  # noqa: F401

    _groot_available = True
except ImportError:
    _groot_available = False


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

    def test_create_mock_by_shorthand(self):
        """All mock shorthands should produce a MockPolicy instance."""
        for name in ("mock", "random", "test"):
            p = create_policy(name)
            assert isinstance(p, MockPolicy), f"'{name}' did not create MockPolicy"

    def test_create_passes_kwargs_to_policy(self):
        """kwargs given to create_policy should reach the Policy constructor."""
        register_policy("kwarg_test", loader=lambda: _KwargCapture, aliases=[])
        p = create_policy("kwarg_test", some_key="some_val")
        assert p.captured == {"some_key": "some_val"}

    def test_create_via_hf_model_id_triggers_smart_resolution(self):
        """An org/model string should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("unknownorg/somemodel")

    def test_create_via_grpc_url_triggers_smart_resolution(self):
        """A grpc:// URL should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("grpc://localhost:50051")

    def test_create_via_ws_url_triggers_smart_resolution(self):
        """A ws:// URL should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("ws://localhost:8080")


@pytest.mark.skipif(not _groot_available, reason="groot-service extras not installed")
class TestFactoryGrootIntegration:
    """Factory tests that require groot-service extras (zmq, msgpack).

    Grouped into a single class with a class-level skip marker so future
    contributors don't need to remember per-test decorators.
    """

    def test_create_via_zmq_url_resolves_to_groot(self):
        """A zmq:// URL should resolve to a Gr00tPolicy via smart-string resolution."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = create_policy("zmq://localhost:5555")
        assert isinstance(p, Gr00tPolicy)

    def test_groot_strict_and_api_token_passthrough(self):
        """strict and api_token kwargs should reach Gr00tPolicy constructor."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = create_policy("zmq://localhost:5555", strict=True, api_token="test-token")
        assert isinstance(p, Gr00tPolicy)
        assert p._strict is True
        assert p._client.api_token == "test-token"

    def test_groot_defaults_strict_false(self):
        """strict should default to False for production use."""
        p = create_policy("zmq://localhost:5555")
        assert p._strict is False

    def test_groot_direct_construction_with_new_params(self):
        """Direct Gr00tPolicy() should accept strict and api_token."""
        from strands_robots.policies.groot import Gr00tPolicy

        p = Gr00tPolicy(host="localhost", port=5555, strict=True, api_token="s3cret")
        assert p._strict is True
        assert p._mode == "service"
        assert p._client.api_token == "s3cret"


class TestTrustRemoteCodeGate:
    """STRANDS_TRUST_REMOTE_CODE gate should block lerobot_local without opt-in."""

    def test_lerobot_local_blocked_without_env(self, monkeypatch):
        """create_policy('lerobot_local') should raise without STRANDS_TRUST_REMOTE_CODE."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        with pytest.raises(UntrustedRemoteCodeError):
            create_policy("lerobot_local")

    def test_lerobot_local_allowed_with_env(self, monkeypatch):
        """create_policy('lerobot_local') should succeed with STRANDS_TRUST_REMOTE_CODE=1."""
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
        p = create_policy("lerobot_local")
        assert p.provider_name == "lerobot_local"

    def test_mock_never_gated(self, monkeypatch):
        """Mock provider should never be blocked by trust gate."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        p = create_policy("mock")
        assert isinstance(p, MockPolicy)

    def test_runtime_registered_not_gated(self, monkeypatch):
        """Runtime-registered providers (not in HF list) should not be gated."""
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        register_policy("safe_custom", loader=lambda: MockPolicy, aliases=["sc"])
        p = create_policy("safe_custom")
        assert isinstance(p, MockPolicy)


class _KwargCapture(Policy):
    """Test helper -- captures kwargs for verification."""

    def __init__(self, **kwargs):
        self.captured = kwargs

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return []

    def set_robot_state_keys(self, robot_state_keys):
        pass

    @property
    def provider_name(self):
        return "kwarg_test"
