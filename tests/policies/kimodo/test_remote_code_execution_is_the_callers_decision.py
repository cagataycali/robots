"""Whether a Kimodo checkpoint runs its own code is the caller's decision.

``DiffusersKimodoAgent`` forwards ``KimodoConfig.trust_remote_code`` to
``DiffusionPipeline.from_pretrained``, and
``test_the_configured_model_id_cache_dir_and_trust_flag_reach_from_pretrained``
already pins that forwarding with the reason spelled out: the flag is
"security-relevant ... so the caller's decision has to be the one that reaches
diffusers rather than a value this agent picks".  That test builds a
``KimodoConfig`` directly, so it never exercised a route a caller actually has,
and two properties the sentence depends on went unchecked:

* the value the agent picks when the caller says nothing, and
* whether any supported keyword route can carry the caller's decision at all.

Both failed.  The field defaulted to ``True``, and it was neither a
``KimodoPolicy.__init__`` parameter nor a ``config_keys`` entry, so
``KimodoPolicy(trust_remote_code=False)`` and
``create_policy("kimodo", trust_remote_code=False)`` raised ``TypeError`` while
``build_policy_kwargs`` - the route a JSON ``policy_config`` travels - dropped
the key and loaded with ``True`` anyway.  A caller could only reach the flag by
hand-building the frozen dataclass, which an agent driving the tool surface
cannot do.

``STRANDS_TRUST_REMOTE_CODE`` does not cover the gap: it decides whether the
provider may be constructed at all and never sets this field, so opting in to
the provider silently opted in to executing repository code for every
``model_id`` the process went on to load.

Reachability arrives as an explicit parameter, not as a ``**kwargs`` absorber,
and it keeps the documented precedence: an explicit keyword beats a ``config=``
base.  The three controls at the bottom pass before and after the fix, pinning
that the coarse provider gate is untouched, that the dataclass route is
undisturbed, and that a typo is still refused rather than swallowed.

Reaching the flag is not the same as the flag carrying the decision.  The
field was declared ``bool`` and checked nowhere - ``__post_init__`` validated
every numeric knob, ``dtype``, ``model_id`` and ``seed`` around it - so a
value arrived at ``from_pretrained`` as given, and diffusers reads it by
truthiness.  Measured through ``build_policy_kwargs``, the route a JSON
``policy_config`` travels: ``"false"``, ``"no"`` and ``"0"`` each reached the
loader verbatim and truthy, so the caller who spelled the opt-out the way JSON
invites had the checkpoint's code executed.  The flag selects a posture rather
than scaling a quantity, so it is now checked with
:func:`~strands_robots.utils.boolean_flag_error` in ``__post_init__`` - the one
site every route re-validates through - and refused before any loader is
built.  ``None`` on the policy keyword stays the documented "not supplied"
sentinel: ``_resolve_config`` drops it before the dataclass sees it.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import UntrustedRemoteCodeError, create_policy
from strands_robots.policies.kimodo import KimodoPolicy
from strands_robots.policies.kimodo.config import KimodoConfig
from strands_robots.registry import build_policy_kwargs

_TRUST_ENV = "STRANDS_TRUST_REMOTE_CODE"


@pytest.fixture
def loader_kwargs(monkeypatch):
    """Return ``(policy) -> load_kwargs`` recorded from a stubbed diffusers.

    Registers a stub ``diffusers`` module carrying only ``DiffusionPipeline``,
    the single name :mod:`strands_robots.policies.kimodo._diffusers_agent`
    imports from it, then drives the policy's own lazy agent construction so the
    recorded kwargs are the ones production would send.
    """
    recorded: dict[str, Any] = {}

    class _StubPipe:
        def to(self, _device: str) -> _StubPipe:
            return self

    class _StubDiffusionPipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _StubPipe:
            recorded.clear()
            recorded.update({"model_id": model_id, **kwargs})
            return _StubPipe()

    stub = types.ModuleType("diffusers")
    stub.DiffusionPipeline = _StubDiffusionPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffusers", stub)

    def _load(policy: KimodoPolicy) -> dict[str, Any]:
        policy._build_real_agent()
        assert recorded, "the stub pipeline was never asked to load"
        return recorded

    return _load


def test_a_default_policy_does_not_let_the_checkpoint_run_its_own_code(loader_kwargs):
    """With no instruction from the caller, the loader is told not to execute code."""
    kwargs = loader_kwargs(KimodoPolicy(device="cpu"))

    assert kwargs["trust_remote_code"] is False, (
        "a caller who said nothing about remote code had the checkpoint's own code "
        f"executed: from_pretrained received trust_remote_code={kwargs['trust_remote_code']!r}"
    )


def test_the_factory_path_carries_the_callers_refusal_to_the_loader(monkeypatch, loader_kwargs):
    """``create_policy`` accepts the flag instead of raising on it."""
    monkeypatch.setenv(_TRUST_ENV, "1")
    policy = create_policy("kimodo", trust_remote_code=False, device="cpu")

    assert loader_kwargs(policy)["trust_remote_code"] is False


def test_a_policy_config_dict_carries_the_callers_refusal(monkeypatch, loader_kwargs):
    """The route a JSON ``policy_config`` travels must not drop the decision.

    ``build_policy_kwargs`` forwards only keys the provider declares in
    ``config_keys`` and drops the rest, so an undeclared security flag is
    discarded in silence and the load proceeds on the default.
    """
    monkeypatch.setenv(_TRUST_ENV, "1")
    forwarded = build_policy_kwargs("kimodo", trust_remote_code=False, device="cpu")

    assert "trust_remote_code" in forwarded, (
        "build_policy_kwargs dropped the caller's trust_remote_code; a policy_config "
        f"asking not to execute repository code forwarded only {sorted(forwarded)}"
    )
    assert loader_kwargs(create_policy("kimodo", **forwarded))["trust_remote_code"] is False


def test_the_caller_can_still_opt_in_to_running_repository_code(monkeypatch, loader_kwargs):
    """Opting in stays possible, which is what makes the default safe to flip."""
    monkeypatch.setenv(_TRUST_ENV, "1")
    policy = create_policy("kimodo", trust_remote_code=True, device="cpu")

    assert loader_kwargs(policy)["trust_remote_code"] is True


TRUTHY_OPT_OUT_SPELLINGS = ["false", "no", "off", "0"]
OTHER_NON_BOOLEAN = [1, 0, 1.0, "true", [], [1]]


@pytest.mark.parametrize("spelling", TRUTHY_OPT_OUT_SPELLINGS)
def test_a_truthy_spelling_of_the_opt_out_never_reaches_the_loader(monkeypatch, loader_kwargs, spelling):
    """A ``policy_config`` that spells the refusal as a string is refused, not obeyed inverted.

    Pre-fix this call constructed, and ``from_pretrained`` received the string
    verbatim - truthy, so the checkpoint's code ran for the caller who asked it
    not to. The refusal names the parameter and the value so a JSON author is
    sent to the right key.
    """
    monkeypatch.setenv(_TRUST_ENV, "1")
    forwarded = build_policy_kwargs("kimodo", trust_remote_code=spelling, device="cpu")

    with pytest.raises(ValueError, match=r"trust_remote_code must be a boolean") as refused:
        create_policy("kimodo", **forwarded)

    assert repr(spelling) in str(refused.value)


@pytest.mark.parametrize("value", OTHER_NON_BOOLEAN, ids=[repr(v) for v in OTHER_NON_BOOLEAN])
def test_every_route_to_the_dataclass_applies_the_same_domain(value):
    """Direct construction, ``from_dict`` and the policy keyword refuse identically.

    The check lives in ``__post_init__`` because ``_resolve_config`` merges an
    override back through the dataclass, so one site covers every route; this
    pins that none of them bypasses it.
    """
    for route in (
        lambda: KimodoConfig(trust_remote_code=value),
        lambda: KimodoConfig.from_dict({"trust_remote_code": value}),
        lambda: KimodoPolicy(trust_remote_code=value, device="cpu"),
    ):
        with pytest.raises(ValueError, match=r"KimodoConfig: trust_remote_code must be a boolean"):
            route()


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)], ids=repr)
def test_a_boolean_is_stored_as_given(value):
    """Control: the two declared spellings, Python or NumPy, are untouched."""
    assert KimodoConfig(trust_remote_code=value).trust_remote_code is value


def test_the_policy_keyword_none_still_means_not_supplied(loader_kwargs):
    """Control: ``KimodoPolicy(trust_remote_code=None)`` keeps the documented sentinel.

    ``None`` is dropped by ``_resolve_config`` before the dataclass sees it, so
    the default ``False`` reaches the loader and the new check never fires on
    the policy's own sentinel.
    """
    assert loader_kwargs(KimodoPolicy(trust_remote_code=None, device="cpu"))["trust_remote_code"] is False


def test_the_remote_code_flag_is_refused_ahead_of_every_other_knob():
    """A config that is wrong in two places names the security flag first.

    The flag whose misread runs someone else's code is the one a caller must
    not be sent past to fix a frame count.
    """
    with pytest.raises(ValueError, match=r"^KimodoConfig: trust_remote_code must be a boolean"):
        KimodoConfig(trust_remote_code="false", num_frames=0)


def test_every_config_field_is_reachable_through_a_keyword():
    """No ``KimodoConfig`` field may be settable only by building the dataclass.

    ``KimodoPolicy.__init__`` documents that it takes no ``**kwargs`` so an
    unknown knob raises rather than being swallowed.  That makes an unlisted
    field strictly unreachable: the keyword is a ``TypeError`` and, being absent
    from ``config_keys``, it is dropped before ``create_policy`` ever sees it.
    """
    params = set(inspect.signature(KimodoPolicy.__init__).parameters) - {"self", "config"}
    unreachable = sorted(set(KimodoConfig.__dataclass_fields__) - params)

    assert unreachable == [], (
        f"KimodoConfig fields {unreachable} can only be set by hand-building the "
        "frozen dataclass: they are neither KimodoPolicy parameters nor config_keys"
    )


def test_the_registry_advertises_every_field_the_constructor_accepts():
    """``config_keys`` is the filter, so an unlisted parameter is unreachable."""
    from strands_robots.registry import get_policy_provider

    advertised = set((get_policy_provider("kimodo") or {}).get("config_keys", []))
    for field in ("trust_remote_code", "cache_dir"):
        assert field in advertised, (
            f"'{field}' is a KimodoPolicy parameter the registry does not advertise, "
            "so build_policy_kwargs drops a caller who passes it"
        )


def test_an_explicit_override_still_beats_a_config_base(loader_kwargs):
    """Precedence is unchanged: an explicit keyword wins over the base config."""
    policy = KimodoPolicy(config={"trust_remote_code": False, "device": "cpu"}, trust_remote_code=True)

    assert loader_kwargs(policy)["trust_remote_code"] is True


# --- controls: these pass before and after the fix -------------------------


def test_a_config_object_that_opts_in_is_left_alone(loader_kwargs):
    """The dataclass route keeps working exactly as it did."""
    policy = KimodoPolicy(config={"trust_remote_code": True, "device": "cpu"})

    assert loader_kwargs(policy)["trust_remote_code"] is True


def test_an_unknown_knob_is_still_refused_rather_than_swallowed():
    """Reachability must not arrive as a ``**kwargs`` that absorbs typos.

    The rejected name is derived from the live signature rather than hard-coded,
    so this stays a statement about unknown keywords: it cannot rot into passing
    a real parameter if one is later named after the string a literal happened
    to hold.
    """
    signature = inspect.signature(KimodoPolicy.__init__)
    assert not any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())

    unknown = max(signature.parameters, key=len) + "_x"
    assert unknown not in signature.parameters

    with pytest.raises(TypeError):
        KimodoPolicy(**dict.fromkeys([unknown], False))


def test_the_provider_gate_still_requires_the_env_opt_in(monkeypatch):
    """The coarse provider gate is untouched by making the field reachable."""
    monkeypatch.delenv(_TRUST_ENV, raising=False)

    with pytest.raises(UntrustedRemoteCodeError):
        create_policy("kimodo", trust_remote_code=False)
