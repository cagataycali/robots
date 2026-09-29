"""Coverage for the policy resolver's URL-scheme and kwargs-mapping branches.

``strands_robots.registry.policies.resolve_policy`` documents a five-step
resolution order (URL patterns -> shorthands -> HF model IDs -> registered
provider name -> lerobot_local fallback). The shipped ``policies.json`` declares
two URL patterns - ``^cosmos3://`` and ``^wss?://`` - so only the
generic scheme-less ``host:port`` branch, part of the public resolution contract
and reachable by a provider that declares a scheme-less pattern, has no
exercising input from the shipped registry. A ``scheme://`` outside the declared
set is refused rather than resolved, which is pinned at the bottom of this
module.

These tests inject a synthetic provider registry (via ``monkeypatch``) so the
generic parser branches run, plus cover the HF-org routing, canonical-name
passthrough, and ``build_policy_kwargs`` defaults/extra-key paths against the
real registry. Behaviour is asserted on the returned (provider, kwargs), never
on internal state.
"""

import re

import pytest

import strands_robots.policies.factory as factory_mod
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

    def test_bare_host_port_address_becomes_server_address(self, monkeypatch):
        """A bare host:port (no scheme, no slash) maps to server_address."""
        _inject_registry(monkeypatch, {"hostport": {"url_patterns": [r"^[^/]+:[0-9]+$"]}})
        provider, kwargs = resolve_policy("myserver:8080")
        assert provider == "hostport"
        assert kwargs["server_address"] == "myserver:8080"

    def test_url_scheme_match_forwards_extra_kwargs(self, monkeypatch):
        """Extra kwargs must survive URL-pattern resolution."""
        _inject_registry(monkeypatch, {"wsprov": {"url_patterns": ["^wss?://"]}})
        _, kwargs = resolve_policy("ws://host:1", timeout=5)
        assert kwargs["timeout"] == 5


class TestHuggingFaceOrgRouting:
    """HF model IDs route by hf_orgs when no model_id_override matches."""

    def test_allenai_org_routes_to_lerobot_local(self):
        """allenai/* is a lerobot_local hf_org (not a model_id override)."""
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
    """import_policy_class falls back to submodule discovery when not in JSON.

    The funnel lives in ``policies.factory``; what it reads is this module's
    registry, which is what ``_inject_registry`` empties.
    """

    def test_capitalized_class_name_is_discovered(self, monkeypatch):
        """With an empty registry, 'mock' is found via MockPolicy in the submodule."""
        from strands_robots.policies import MockPolicy

        _inject_registry(monkeypatch, {})
        assert factory_mod.import_policy_class("mock") is MockPolicy

    def test_policy_subclass_scan_finds_class_when_name_mismatches(self, monkeypatch):
        """When 'NamePolicy' does not exist, the module is scanned for a Policy subclass."""
        from strands_robots.policies import Policy

        _inject_registry(monkeypatch, {})
        cls = factory_mod.import_policy_class("lerobot_local")
        assert issubclass(cls, Policy) and cls is not Policy


#: One concrete URL per scheme the shipped ``policies.json`` declares, spelled
#: the way the docs spell them (lowercase). ``TestEveryDeclaredSchemeHasASample``
#: below pins that this tuple still covers every declared pattern, so a scheme
#: added to the registry cannot slip past the case-invariance guard.
_SCHEME_URLS = (
    "ws://gpu-box:8765",
    "wss://gpu-box:8765",
    "cosmos3://prod-server:9000",
)


def _case_variants(url: str) -> tuple[str, ...]:
    """Spellings of ``url`` that differ from it only in the scheme's case."""
    scheme, rest = url.split("://", 1)
    return tuple(
        dict.fromkeys(
            f"{spelling}://{rest}"
            for spelling in (scheme.upper(), scheme.capitalize(), scheme[:-1] + scheme[-1].upper())
            if spelling != scheme
        )
    )


class TestSchemeCaseDoesNotChangeResolution:
    """A URL scheme is case-insensitive (RFC 3986 section 3.1).

    Stage 1 matched the raw string against ``url_patterns`` that are all spelled
    lowercase, while every later stage lowercased what it compared. So ``MOCK``
    resolved to ``mock`` and ``NVIDIA/GR00T`` routed by org, but ``COSMOS3://...``
    matched nothing, fell through to the HuggingFace fallback, and became
    ``lerobot_local(pretrained_name_or_path="COSMOS3://gpu-box:9000")`` -- a request
    to download a repo by that literal name instead of dialing the sidecar, with
    only a warning. The failure surfaced as a HuggingFace error, nowhere near
    the server the caller named.
    """

    @pytest.mark.parametrize("url", _SCHEME_URLS)
    def test_an_uppercase_scheme_resolves_exactly_as_the_lowercase_one(self, url):
        """Provider and kwargs must be identical for every spelling of a scheme."""
        expected = resolve_policy(url)
        for variant in _case_variants(url):
            assert resolve_policy(variant) == expected, (
                f"{variant!r} resolved to {resolve_policy(variant)!r}, but {url!r} resolves to {expected!r}"
            )

    def test_the_emitted_websocket_endpoint_keeps_ws_and_wss_apart(self):
        """Folding the scheme must not collapse the plain/secure distinction."""
        assert resolve_policy("WS://h:8765")[1]["endpoint"] == "ws://h:8765"
        assert resolve_policy("WSS://h:8765")[1]["endpoint"] == "wss://h:8765"


class TestOnlyTheSchemeIsFolded:
    """Controls: everything a case fold would damage is left exactly as given."""

    def test_a_huggingface_repo_id_is_forwarded_with_its_case_intact(self):
        """Repo ids are case-sensitive, so the id must survive byte for byte."""
        repo = "NVIDIA/GR00T-N1.7-3B"
        provider, kwargs = resolve_policy(repo)
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == repo

    def test_the_host_and_port_of_a_url_are_not_folded(self):
        """Only the scheme is folded - the authority keeps the caller's spelling."""
        _, kwargs = resolve_policy("COSMOS3://GPU-Box:9000")
        assert kwargs["host"] == "GPU-Box"
        assert kwargs["port"] == 9000

    def test_a_string_with_no_scheme_is_untouched(self, monkeypatch):
        """A bare host:port has no ``scheme://``, so nothing is rewritten."""
        _inject_registry(monkeypatch, {"hostport": {"url_patterns": [r"^[^/]+:[0-9]+$"]}})
        provider, kwargs = resolve_policy("MyServer:8080")
        assert provider == "hostport"
        assert kwargs["server_address"] == "MyServer:8080"

    def test_the_scheme_fold_leaves_non_url_strings_alone(self):
        """The helper rewrites a leading scheme and nothing else."""
        from strands_robots.registry.policies import _with_lowercase_url_scheme

        for unchanged in ("mock", "NVIDIA/GR00T-N1.7-3B", "MyServer:8080", "a/B://c"):
            assert _with_lowercase_url_scheme(unchanged) == unchanged
        assert _with_lowercase_url_scheme("ZMQ://Host/Path") == "zmq://Host/Path"


class TestEveryDeclaredSchemeHasASample:
    """Keeps ``_SCHEME_URLS`` honest against the shipped registry."""

    def test_every_declared_url_pattern_matches_a_sample(self):
        """A newly declared scheme must be added to _SCHEME_URLS to pass."""
        providers = policies_mod._load("policies").get("providers", {})
        declared = {pattern for info in providers.values() for pattern in info.get("url_patterns", []) or ()}
        assert declared, "premise: the shipped registry declares url_patterns"
        uncovered = {p for p in declared if not any(re.match(p, u) for u in _SCHEME_URLS)}
        assert not uncovered, f"url_patterns with no sample in _SCHEME_URLS: {sorted(uncovered)}"


class TestASchemeNoProviderDeclaresIsRefused:
    """Stage 1 is the whole transport vocabulary, so an address outside it stops there.

    Every later stage reads the string as a *name* - a shorthand, a repo id, a
    provider - so an address that reaches them is forwarded to ``lerobot_local``
    as a checkpoint id. Measured pre-fix on ``grpc://gpu-box:8080``, the scheme
    the retired ``lerobot_async`` provider declared:
    ``("lerobot_local", {"pretrained_name_or_path": "grpc://gpu-box:8080"})``
    under a warning, so the caller's next report was a HuggingFace lookup
    failure for a repo id nobody wrote, nowhere near the server they named.
    """

    @pytest.mark.parametrize(
        "url",
        ["grpc://gpu-box:8080", "http://gpu-box:8000", "https://gpu-box", "tcp://10.0.0.5:5555", "zzz://h:1"],
    )
    def test_it_is_refused_naming_the_schemes_that_do_resolve(self, url):
        """The refusal carries the scheme, the caller's string, and the alternatives."""
        with pytest.raises(ValueError) as exc:
            resolve_policy(url)
        text = str(exc.value)
        assert f"'{url.split('://', 1)[0]}://'" in text, text
        assert url in text, text
        for scheme in policies_mod._declared_url_schemes():
            assert f"{scheme}://" in text, text

    def test_the_refusal_folds_the_scheme_like_the_match_does(self):
        """``GRPC://`` is the same request as ``grpc://``, so it is refused too."""
        with pytest.raises(ValueError, match=re.escape("'grpc://'")):
            resolve_policy("GRPC://gpu-box:8080")

    @pytest.mark.parametrize("scheme", policies_mod._declared_url_schemes())
    def test_every_declared_scheme_still_resolves(self, scheme):
        """Non-vacuity, and the scope control: only undeclared schemes are refused."""
        provider, _ = resolve_policy(f"{scheme}://gpu-box:8080")
        declared = policies_mod._load("policies")["providers"][provider].get("url_patterns", [])
        assert any(re.match(pattern, f"{scheme}://gpu-box:8080") for pattern in declared), (
            f"{scheme}:// resolved to {provider}, which declares {declared}"
        )

    def test_the_advertised_set_is_exactly_what_the_patterns_match(self):
        """A spelling the expansion invents but no pattern matches must not be named."""
        schemes = policies_mod._declared_url_schemes()
        assert schemes == sorted(set(schemes)), schemes
        patterns = [
            pattern
            for info in policies_mod._load("policies")["providers"].values()
            for pattern in info.get("url_patterns", []) or ()
        ]
        for scheme in schemes:
            assert any(re.match(pattern, f"{scheme}://h:1") for pattern in patterns), scheme

    @pytest.mark.parametrize("policy", ["mock", "lerobot/act_aloha_sim", "gpu-box:8080", "a/B://c"])
    def test_a_string_that_is_not_an_address_is_not_read_as_one(self, policy):
        """The guard reads a *leading* scheme only, so a name still resolves."""
        provider, _ = resolve_policy(policy)
        assert provider in policies_mod._load("policies")["providers"]
