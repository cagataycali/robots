"""Tests for ``strands_robots.policies.factory.create_policy``.

* provider resolution (mock / remote / lerobot_local)
* the removed ``groot`` spelling is refused with its fixed sentence, never rerouted
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
    policy_overrides_preflight,
    preflight_policy,
    register_policy,
)
from strands_robots.policies.factory import policy_provider_error, provider_can_be_created
from strands_robots.registry.policies import REMOVED_PROVIDERS, policy_provider_resolves, resolve_policy


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
        assert "lerobot_local" in providers
        assert "groot" not in providers
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

    def test_create_via_grpc_url_triggers_smart_resolution(self):
        """A grpc:// URL should trigger smart-string resolution."""
        with pytest.raises(Exception):
            create_policy("grpc://localhost:50051")

    def test_create_via_ws_url_resolves_to_remote_policy(self):
        """A ws:// URL resolves to the remote-inference provider (RemotePolicy).

        Construction is lazy (the WebSocket connects on first use), so this
        succeeds without a running server and preserves the full endpoint URL.
        """
        from strands_robots.inference import RemotePolicy

        policy = create_policy("ws://localhost:8080")
        assert isinstance(policy, RemotePolicy)
        assert policy.uri == "ws://localhost:8080"


class TestRemovedGrootProviderIsRefused:
    """``groot`` was removed in 1.0: every spelling is refused with one sentence.

    The refusal must be a refusal, not a reroute. Without the table in
    ``registry.policies.REMOVED_PROVIDERS`` the name would fall through
    ``resolve_policy``'s last stage to ``lerobot_local`` as a checkpoint id.
    """

    SENTENCE = REMOVED_PROVIDERS["groot"]

    def test_the_sentence_is_the_documented_one(self):
        assert self.SENTENCE == (
            "policy_provider 'groot' was removed in 1.0: GR00T N1.7 runs through "
            "lerobot_local(policy_type='groot'); for a remote GPU host run "
            "strands_robots.inference.server.PolicyServer there and use policy_provider='remote'."
        )

    @pytest.mark.parametrize("spelling", ["groot", "GROOT", " groot "])
    def test_create_policy_refuses_every_spelling(self, spelling):
        with pytest.raises(ValueError) as excinfo:
            create_policy(spelling, port=5555)
        assert str(excinfo.value) == self.SENTENCE

    def test_resolve_policy_refuses_before_the_lerobot_local_fallback(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_policy("groot")
        assert str(excinfo.value) == self.SENTENCE

    def test_preflight_surfaces_report_the_same_sentence(self):
        assert provider_can_be_created("groot") is False
        assert policy_provider_resolves("groot") is False
        assert policy_provider_error("groot") == self.SENTENCE

    def test_a_zmq_url_is_an_undeclared_scheme(self):
        """No provider dials ZMQ from a URL any more; the scheme is refused as an address."""
        with pytest.raises(ValueError, match="zmq://"):
            resolve_policy("zmq://localhost:5555")

    def test_nvidia_checkpoints_route_to_lerobot_local(self):
        provider, kwargs = resolve_policy("nvidia/GR00T-N1.7-3B")
        assert provider == "lerobot_local"
        assert kwargs == {"pretrained_name_or_path": "nvidia/GR00T-N1.7-3B"}


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


class TestSmartResolutionFallThrough:
    """How create_policy treats a failure of smart-string resolution.

    A smart string (HF id, ``ws://``, ``zmq://`` ...) first goes through
    ``resolve_policy``. An ``ImportError`` there (an optional resolver backend
    is missing) is skipped silently and the string is tried as a registry
    name. Any other error is the resolver's answer and reaches the caller
    unchanged: no smart string is a registry name, so the static lookup could
    only replace a precise refusal with "Unknown policy provider".
    """

    def test_resolution_importerror_is_swallowed_not_propagated(self, monkeypatch):
        """The surfaced error comes from the static lookup, not the resolver."""

        def boom(provider, **kwargs):
            raise ImportError("optional resolver backend missing -- do not surface")

        monkeypatch.setattr("strands_robots.policies.factory.resolve_policy", boom)
        with pytest.raises(Exception) as ei:
            create_policy("unknownorg/doesnotexist")
        assert "do not surface" not in str(ei.value), "resolver ImportError leaked instead of falling through to lookup"

    def test_any_other_resolver_error_reaches_the_caller_unchanged(self, monkeypatch):
        """Pre-fix it was logged at WARNING and replaced by the generic 404."""

        def boom(provider, **kwargs):
            raise ValueError("resolver refused this address")

        monkeypatch.setattr("strands_robots.policies.factory.resolve_policy", boom)
        with pytest.raises(ValueError, match="^resolver refused this address$"):
            create_policy("unknownorg/doesnotexist")


class TestOnlyAddressAndCheckpointShapesAreSmartStrings:
    """Punctuation alone does not make a spelling an address or a checkpoint.

    A typo carrying a ``/`` or ``:`` used to be forwarded to ``lerobot_local``
    as a checkpoint id, so the caller got a trust-remote-code refusal naming a
    provider they never typed, and the pre-flight check reported it buildable.
    """

    @pytest.mark.parametrize(
        ("spelling", "suggestion"),
        [
            ("wbc/", "'wbc'"),
            ("protomotions:", "'protomotions'"),
            (":", None),
            ("/", None),
            ("C:\\path", None),
            ("myserver:8080", None),
        ],
    )
    def test_a_typo_is_an_unknown_provider(self, spelling, suggestion, monkeypatch):
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        assert provider_can_be_created(spelling) is False
        with pytest.raises(ValueError, match="^Unknown policy provider") as ei:
            create_policy(spelling)
        if suggestion:
            assert f"Did you mean: {suggestion}" in str(ei.value)

    @pytest.mark.parametrize(
        "spelling",
        ["unknownorg/somemodel", "/tmp/ft/checkpoints/last/pretrained_model", "outputs/train/act", "./ckpt", "~/ckpt"],
    )
    def test_a_checkpoint_still_reaches_lerobot_local(self, spelling, monkeypatch):
        monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
        assert provider_can_be_created(spelling) is True
        with pytest.raises(UntrustedRemoteCodeError, match="'lerobot_local'"):
            create_policy(spelling)


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


class _PreflightPolicy(MockPolicy):
    """Test helper -- a provider that overrides the class-level ``preflight``
    hook to reject a runtime observation missing a required camera key.

    Records every ``preflight`` invocation so tests can assert the observation
    keys and provider kwargs were forwarded verbatim.
    """

    preflight_calls: list[tuple[set[str], dict]] = []

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: object) -> None:
        cls.preflight_calls.append((set(observation_keys), dict(policy_config)))
        if "camera_top" not in observation_keys:
            raise ValueError("preflight: required image source 'camera_top' is absent")


class TestPreflightPolicy:
    """``preflight_policy`` is the fail-fast seam ``run_policy`` / ``eval_policy``
    call BEFORE ``create_policy`` downloads weights. It must: resolve a provider
    without instantiating it, invoke the class ``preflight`` hook only when the
    provider overrides it, propagate that hook's ``ValueError``, and swallow
    resolution failures (the authoritative error comes later from
    ``create_policy``).
    """

    @pytest.fixture(autouse=True)
    def _reset_preflight_calls(self):
        _PreflightPolicy.preflight_calls.clear()
        yield
        _PreflightPolicy.preflight_calls.clear()

    def test_noop_when_provider_does_not_override_preflight(self):
        """A provider using the default no-op ``preflight`` (e.g. ``mock``) must
        pass silently -- no exception, no hook side effects."""
        assert preflight_policy("mock", {"joint_0", "joint_1"}) is None

    def test_propagates_valueerror_from_overridden_preflight(self):
        """When the resolved provider overrides ``preflight`` and rejects the
        observation keys, ``preflight_policy`` must surface that ``ValueError``
        (this is the whole point of the fail-fast seam)."""
        register_policy("preflight_reject", loader=lambda: _PreflightPolicy)
        with pytest.raises(ValueError, match="camera_top"):
            preflight_policy("preflight_reject", {"joint_0"})

    def test_passes_when_overridden_preflight_accepts_keys(self):
        """A satisfied override returns ``None`` -- and was actually invoked."""
        register_policy("preflight_accept", loader=lambda: _PreflightPolicy)
        assert preflight_policy("preflight_accept", {"joint_0", "camera_top"}) is None
        assert len(_PreflightPolicy.preflight_calls) == 1

    def test_forwards_observation_keys_and_policy_config(self):
        """The runtime observation keys and provider kwargs must reach the hook
        unchanged -- a dropped kwarg would defeat config-dependent validation."""
        register_policy("preflight_capture", loader=lambda: _PreflightPolicy)
        preflight_policy(
            "preflight_capture",
            {"camera_top", "joint_0"},
            image_key="camera_top",
            temperature=0.5,
        )
        assert len(_PreflightPolicy.preflight_calls) == 1
        keys, config = _PreflightPolicy.preflight_calls[0]
        assert keys == {"camera_top", "joint_0"}
        assert config == {"image_key": "camera_top", "temperature": 0.5}

    def test_swallows_resolution_failure(self):
        """An unresolvable provider must NOT raise here: resolution errors are
        surfaced authoritatively by the subsequent ``create_policy`` call, so
        the preflight seam degrades to a no-op instead of masking that error."""
        assert preflight_policy("nonexistent_provider_xyz_123", {"joint_0"}) is None


class TestPolicyOverridesPreflight:
    """``policy_overrides_preflight`` answers whether ``preflight_policy`` will
    read its ``observation_keys`` argument, so a caller can decide whether to
    pay for producing it. The simulation's preflight sources those keys from a
    ``get_observation`` that renders every camera in the scene.
    """

    @pytest.mark.parametrize(
        ("provider", "overrides"),
        [
            ("mock", False),
            ("lerobot_local", True),
            ("nonexistent_provider_xyz_123", False),
        ],
    )
    def test_the_shipped_verdicts(self, provider, overrides):
        """``lerobot_local`` is the one shipped provider with a real hook. An
        unresolvable name reports no hook, matching ``preflight_policy``, which
        degrades to a no-op for a name it cannot resolve.
        """
        assert policy_overrides_preflight(provider) is overrides

    def test_a_registered_override_is_reported(self):
        register_policy("preflight_overrides_probe", loader=lambda: _PreflightPolicy)
        assert policy_overrides_preflight("preflight_overrides_probe") is True

    def test_the_answer_agrees_with_whether_the_hook_runs(self):
        """The two functions must never disagree: a provider reported as having
        no override must also leave ``_PreflightPolicy``'s recorder untouched.
        """
        register_policy("preflight_agreement_probe", loader=lambda: _PreflightPolicy)
        _PreflightPolicy.preflight_calls.clear()

        assert policy_overrides_preflight("preflight_agreement_probe") is True
        preflight_policy("preflight_agreement_probe", {"joint_0", "camera_top"})
        assert len(_PreflightPolicy.preflight_calls) == 1

        assert policy_overrides_preflight("mock") is False
        preflight_policy("mock", {"joint_0", "camera_top"})
        assert len(_PreflightPolicy.preflight_calls) == 1
