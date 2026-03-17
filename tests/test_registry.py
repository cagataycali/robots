"""Tests for strands_robots.registry — resolve_policy() and provider lookup."""

from strands_robots.registry import get_policy_provider, list_policy_providers, resolve_policy


class TestResolvePolicy:
    """resolve_policy() should handle shorthands, HF model IDs, and server URLs."""

    def test_shorthand_aliases(self):
        """All shorthand aliases for a provider should resolve to the same provider."""
        for alias in ("mock", "random", "test"):
            provider, _ = resolve_policy(alias)
            assert provider == "mock", f"'{alias}' should resolve to 'mock'"

        provider, _ = resolve_policy("groot")
        assert provider == "groot"

    def test_huggingface_model_id(self):
        """HF org-based model IDs should resolve to the correct provider."""
        provider, _ = resolve_policy("nvidia/gr00t-n1.5-3b")
        assert provider == "groot"

    def test_zmq_url_extracts_host_and_port(self):
        """ZMQ URLs should resolve to groot with parsed host/port."""
        provider, kwargs = resolve_policy("zmq://myhost:9999")
        assert provider == "groot"
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 9999

    def test_extra_kwargs_forwarded(self):
        """Extra kwargs should pass through resolve_policy unchanged."""
        _, kwargs = resolve_policy("mock", custom_param="hello")
        assert kwargs["custom_param"] == "hello"


class TestProviderLookup:
    """JSON-based provider config should be queryable."""

    def test_known_provider_returns_config(self):
        config = get_policy_provider("groot")
        assert config is not None
        assert "port" in config["config_keys"]
        assert config["class"] == "Gr00tPolicy"

    def test_unknown_provider_returns_none(self):
        assert get_policy_provider("nonexistent_xyz") is None

    def test_list_providers_includes_all_json_entries(self):
        providers = list_policy_providers()
        assert "mock" in providers
        assert "groot" in providers
