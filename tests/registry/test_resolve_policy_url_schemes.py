"""Coverage for the policy resolver's URL-scheme and kwargs-mapping branches.

``strands_robots.registry.policies.resolve_policy`` documents a five-step
resolution order (URL patterns -> shorthands -> HF model IDs -> registered
provider name -> lerobot_local fallback). The shipped ``policies.json`` only
declares ``zmq://`` and ``cosmos3://`` URL patterns, so the generic parser
branches for ``ws(s)://``, ``grpc://`` and bare ``host:port`` addresses - part
of the public resolution contract - had no exercising input.

These tests inject a synthetic provider registry (via ``monkeypatch``) so the
generic parser branches run, plus cover the HF-org routing, canonical-name
passthrough, and ``build_policy_kwargs`` defaults/extra-key paths against the
real registry. Behaviour is asserted on the returned (provider, kwargs), never
on internal state.
"""

import strands_robots.registry.policies as policies_mod
from strands_robots.registry.policies import build_policy_kwargs, resolve_policy


def _inject_registry(monkeypatch, providers: dict) -> None:
    """Force resolve_policy to see a synthetic provider registry."""
    real_load = policies_mod._load

    def fake_load(name: str):
        if name == "policies":
            return {"providers": providers}
        return real_load(name)

    monkeypatch.setattr(policies_mod, "_load", fake_load)


class TestUrlSchemeParsing:
    """The generic URL parser must populate host/port/server_address per scheme."""

    def test_websocket_url_without_port_defaults_to_8000(self, monkeypatch):
        """ws:// with no explicit port should default the port to 8000."""
        _inject_registry(monkeypatch, {"wsprov": {"url_patterns": ["^wss?://"]}})
        provider, kwargs = resolve_policy("ws://myhost")
        assert provider == "wsprov"
        assert kwargs["host"] == "myhost"
        assert kwargs["port"] == 8000

    def test_secure_websocket_url_parses_host_and_port(self, monkeypatch):
        """wss:// with an explicit port should parse both host and port."""
        _inject_registry(monkeypatch, {"wsprov": {"url_patterns": ["^wss?://"]}})
        provider, kwargs = resolve_policy("wss://gpu-box:1234")
        assert provider == "wsprov"
        assert kwargs["host"] == "gpu-box"
        assert kwargs["port"] == 1234

    def test_grpc_url_strips_scheme_into_server_address(self, monkeypatch):
        """grpc:// should drop the scheme and keep the bare address."""
        _inject_registry(monkeypatch, {"grpcprov": {"url_patterns": ["^grpc://"]}})
        provider, kwargs = resolve_policy("grpc://10.0.0.5:50051")
        assert provider == "grpcprov"
        assert kwargs["server_address"] == "10.0.0.5:50051"

    def test_bare_host_port_address_becomes_server_address(self, monkeypatch):
        """A bare host:port (no scheme, no slash) maps to server_address."""
        _inject_registry(monkeypatch, {"hostport": {"url_patterns": [r"^[^/]+:[0-9]+$"]}})
        provider, kwargs = resolve_policy("myserver:8080")
        assert provider == "hostport"
        assert kwargs["server_address"] == "myserver:8080"

    def test_url_scheme_match_forwards_extra_kwargs(self, monkeypatch):
        """Extra kwargs must survive URL-pattern resolution."""
        _inject_registry(monkeypatch, {"grpcprov": {"url_patterns": ["^grpc://"]}})
        _, kwargs = resolve_policy("grpc://host:1", timeout=5)
        assert kwargs["timeout"] == 5


class TestHuggingFaceOrgRouting:
    """HF model IDs route by hf_orgs when no model_id_override matches."""

    def test_allenai_org_routes_to_lerobot_local(self):
        """allenai/* is a lerobot_local hf_org (not a groot override)."""
        provider, kwargs = resolve_policy("allenai/MolmoAct2-SO100_101")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "allenai/MolmoAct2-SO100_101"

    def test_lerobot_org_routes_to_lerobot_local(self):
        """lerobot/* resolves to lerobot_local via hf_orgs."""
        provider, kwargs = resolve_policy("lerobot/act_aloha_sim")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim"


class TestCanonicalNamePassthrough:
    """A bare canonical provider name (step 4) resolves to itself."""

    def test_canonical_provider_name_resolves_to_itself(self):
        """lerobot_local is a canonical name, not a shorthand or alias."""
        provider, kwargs = resolve_policy("lerobot_local")
        assert provider == "lerobot_local"
        assert kwargs == {}

    def test_canonical_name_forwards_extra_kwargs(self):
        """Extra kwargs pass through canonical-name resolution."""
        _, kwargs = resolve_policy("lerobot_local", device="cuda")
        assert kwargs["device"] == "cuda"


class TestBuildPolicyKwargsDefaultsAndExtra:
    """build_policy_kwargs applies JSON defaults and filters extra by config_keys."""

    def test_cosmos3_applies_json_defaults(self):
        """cosmos3 declares host/port/embodiment defaults in policies.json."""
        kwargs = build_policy_kwargs("cosmos3")
        assert kwargs["host"] == "localhost"
        assert kwargs["port"] == 8000
        assert kwargs["embodiment"] == "droid"

    def test_extra_kwarg_in_config_keys_is_kept(self):
        """An allowed extra key (prompt) is retained alongside defaults."""
        kwargs = build_policy_kwargs("cosmos3", prompt="pick up the cube")
        assert kwargs["prompt"] == "pick up the cube"
        assert kwargs["embodiment"] == "droid"

    def test_explicit_value_overrides_default(self):
        """An explicit param must win over the JSON default for the same key."""
        kwargs = build_policy_kwargs("cosmos3", policy_host="gpu-box")
        assert kwargs["host"] == "gpu-box"


class TestAbsoluteHuggingFaceFallback:
    """An HF model ID with no matching org and no is_hf_default provider."""

    def test_unknown_org_with_no_hf_default_falls_back_to_lerobot_local(self, monkeypatch):
        """When no provider declares is_hf_default, slash IDs hit the absolute fallback."""
        _inject_registry(monkeypatch, {"x": {"hf_orgs": ["onlythis"]}})
        provider, kwargs = resolve_policy("unknownorg/some-model")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "unknownorg/some-model"


class TestImportPolicyClassAutoDiscovery:
    """import_policy_class falls back to submodule discovery when not in JSON."""

    def test_capitalized_class_name_is_discovered(self, monkeypatch):
        """With an empty registry, 'mock' is found via MockPolicy in the submodule."""
        from strands_robots.policies import MockPolicy

        _inject_registry(monkeypatch, {})
        assert policies_mod.import_policy_class("mock") is MockPolicy

    def test_policy_subclass_scan_finds_class_when_name_mismatches(self, monkeypatch):
        """When 'NamePolicy' does not exist, the module is scanned for a Policy subclass."""
        from strands_robots.policies import Policy

        _inject_registry(monkeypatch, {})
        cls = policies_mod.import_policy_class("lerobot_local")
        assert issubclass(cls, Policy) and cls is not Policy
