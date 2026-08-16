"""Every registry key that can reach a provider constructor must name a parameter of it.

``policies.json`` describes each provider with a ``config_keys`` list and an
optional ``defaults`` map, and ``build_policy_kwargs`` merges both into one dict
that ``create_policy`` splats into the provider class.  Nothing in the tree
compared either against a constructor signature: ``grep -rn config_keys tests/``
finds presence checks (``"port" in config["config_keys"]``) and filtering
behaviour, never an agreement check.  #2013 pinned the property for ``vera``
alone, deliberately, and named the general form #2022 - this is it.

The two sources are not equally guarded, which is why both are checked here.

``config_keys`` is a filter, so a stale entry only bites when a caller happens
to pass that key::

    for key, value in extra.items():
        if key in allowed_keys:          # a stale entry is reachable, not forced
            kwargs[key] = value

``defaults`` has no such test::

    for key, default_val in defaults.items():
        if key not in kwargs:            # no ``key in allowed_keys`` here
            kwargs[key] = default_val

So a ``defaults`` key is inserted on *every* call, with no caller involvement.
An orphaned one is therefore strictly worse than an orphaned ``config_keys``
entry: it makes ``create_policy(provider)`` raise for every caller, not only for
one that passes the key.  ``test_a_defaults_key_bypasses_the_config_keys_filter``
pins that asymmetry, so the reason these tests cover ``defaults`` cannot quietly
stop being true.

What an orphan costs splits by provider shape.  ``cosmos3`` and ``vera`` declare
no ``**kwargs``, so an orphan is a hard ``TypeError: __init__() got an unexpected
keyword argument`` on the factory path - the path the registry exists to serve -
while a caller constructing the class directly is unaffected.  The other ten
swallow it into a ``**kwargs`` nothing reads, which is the inert-public-knob
shape #2013 was filed about rather than a safer outcome.  The guard is the
strict form for that reason: it is what all twelve providers already satisfy,
and the lenient form (an entry is fine if the class takes ``**kwargs``) would
exempt ten of twelve and assert almost nothing.

**This is a drift guard, not a live defect.** All twelve providers are consistent
today, in both directions, so there is no pre-fix failure to show - only the
guard plus the planted-orphan meta-tests that keep it from passing vacuously.

Kept as a test rather than a runtime check in ``build_policy_kwargs``:
``policies.json`` ships in-tree and changes only when someone edits it, so
disagreement is a development-time property, and paying an ``inspect.signature``
per ``create_policy`` call would import every provider's optional dependencies
to answer a question the suite can settle once.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry.policies import build_policy_kwargs

# ─── registry-derived cases ───────────────────────────────────────────


def _registry_path() -> Path:
    """Locate policies.json from the module under test, never a path literal."""
    return Path(inspect.getfile(build_policy_kwargs)).parent / "policies.json"


def _providers() -> dict[str, Any]:
    return json.loads(_registry_path().read_text())["providers"]


def _signature_parameters(cfg: dict[str, Any]) -> Any:
    """Import a provider class and return its ``__init__`` parameters."""
    cls = getattr(importlib.import_module(cfg["module"]), cfg["class"])
    return inspect.signature(cls.__init__).parameters


def _resolve_signatures() -> tuple[dict[str, Any], dict[str, str]]:
    """Import every declared provider once, and record what failed.

    Resolved at module scope rather than per test: importing a provider pulls in
    its optional dependencies, and the per-key tests below ask for the same
    signature 100+ times.  Doing it once also makes the coverage test and the
    per-key tests read the *same* map, so they cannot disagree about which
    providers were available.
    """
    signatures: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, cfg in _providers().items():
        try:
            signatures[name] = _signature_parameters(cfg)
        except Exception as exc:  # noqa: BLE001 - surfaced by the coverage test
            errors[name] = f"{type(exc).__name__}: {exc}"
    return signatures, errors


_SIGNATURES, _IMPORT_ERRORS = _resolve_signatures()


def _orphans(keys: list[str], parameters: Any) -> list[str]:
    """The registry keys that name no explicit constructor parameter.

    The single predicate every assertion below routes through, so the
    planted-orphan meta-tests exercise the same code path as the real ones
    rather than a re-implementation that could drift from it.
    """
    return [key for key in keys if key not in parameters]


def _parameters_or_skip(provider: str) -> Any:
    """Parameters for ``provider``, skipping if its optional deps are absent.

    A skip here is *not* silent: ``test_every_declared_provider_was_read``
    fails naming the provider and the import error, which is the non-vacuity
    assertion decision 2 of #2022 asks for.  Skipping only keeps a missing
    dependency from presenting as a registry disagreement.

    Reads the map resolved at import rather than importing here, so this is a
    dict lookup with no ``try`` around it.  That is also what keeps the helper
    free of the two CodeQL rules the earlier shapes tripped: returning from
    inside a ``try`` whose ``except`` falls off the end is ``py/mixed-returns``,
    and assigning in the ``try`` to return afterwards is
    ``py/uninitialized-local-variable``.  Both are artefacts of hiding the
    control flow in an exception handler, because ``pytest.skip`` raises and no
    static analysis here can know that.  With the import already done there is
    no handler to hide anything in.
    """
    if provider in _IMPORT_ERRORS:
        pytest.skip(f"{provider} is not importable here: {_IMPORT_ERRORS[provider]}")
    return _SIGNATURES[provider]


def _config_key_cases() -> list[tuple[str, str]]:
    return [(name, key) for name, cfg in _providers().items() for key in cfg.get("config_keys") or []]


def _defaults_cases() -> list[tuple[str, str]]:
    return [(name, key) for name, cfg in _providers().items() for key in (cfg.get("defaults") or {})]


# ─── the guard ────────────────────────────────────────────────────────


@pytest.mark.parametrize(("provider", "key"), _config_key_cases())
def test_every_config_keys_entry_names_a_constructor_parameter(provider: str, key: str) -> None:
    """A ``config_keys`` entry the constructor does not accept is a factory-only crash."""
    parameters = _parameters_or_skip(provider)
    assert _orphans([key], parameters) == [], (
        f"policies.json advertises {provider}.config_keys entry {key!r}, which is not a "
        f"parameter of the provider constructor - a caller passing it through "
        f"create_policy/build_policy_kwargs gets a TypeError, or has it silently dropped "
        f"into **kwargs that nothing reads"
    )


@pytest.mark.parametrize(("provider", "key"), _defaults_cases())
def test_every_defaults_key_names_a_constructor_parameter(provider: str, key: str) -> None:
    """The same agreement for ``defaults``, which is forwarded unconditionally.

    Stricter in consequence than the ``config_keys`` case above: the defaults
    loop applies no ``allowed_keys`` filter, so an orphan here breaks every
    ``create_policy(provider)`` call rather than only one that passes the key.
    """
    parameters = _parameters_or_skip(provider)
    assert _orphans([key], parameters) == [], (
        f"policies.json gives {provider} a default for {key!r}, which is not a parameter "
        f"of the provider constructor - defaults are forwarded with no config_keys filter, "
        f"so this reaches the constructor on every create_policy({provider!r}) call"
    )


@pytest.mark.parametrize(("provider", "key"), _defaults_cases())
def test_every_defaults_key_is_also_declared_in_config_keys(provider: str, key: str) -> None:
    """A default outside ``config_keys`` is a value no caller can override.

    ``config_keys`` is the registry's only vocabulary for "a key a caller may
    set", and it gates the ``extra`` loop but not the defaults loop.  So a
    ``defaults`` entry missing from ``config_keys`` is still applied, while a
    caller's own value for that key is dropped by the filter - a forced value
    with no way to say so and no error when it happens.

    This is *not* the reverse direction #2022 defers.  That one asks whether
    every constructor parameter should be registry-settable, which needs a
    judgement about deliberately code-only parameters (``client``,
    ``server_runner``).  This asks only that the registry agree with itself
    about keys it already declares, so it needs no such judgement.  If a
    non-overridable default is ever wanted, it should be named explicitly
    rather than arise from an omission.
    """
    assert key in (_providers()[provider].get("config_keys") or []), (
        f"policies.json gives {provider} a default for {key!r} but does not list it in "
        f"config_keys, so the default is applied while a caller's own value for the same "
        f"key is filtered out - an override that silently does nothing"
    )


# ─── non-vacuity ──────────────────────────────────────────────────────


def test_every_declared_provider_was_read() -> None:
    """Coverage, asserted in one place so a skipped provider cannot hide.

    Deciding a provider is consistent because it did not import is the vacuity
    risk decision 2 of #2022 names: a local run missing ``torch`` would check a
    handful of providers and pass.  CI installs the full extras, so this is a
    statement about coverage rather than about the environment.
    """
    providers = _providers()
    assert _IMPORT_ERRORS == {}, (
        f"read {len(_SIGNATURES)} of {len(providers)} provider signatures; "
        f"the guard is vacuous for the rest, install the full extras: {_IMPORT_ERRORS}"
    )
    assert set(_SIGNATURES) == set(providers), "the resolved map and the registry disagree"


def test_the_guard_covers_every_provider_that_declares_keys() -> None:
    """The parametrised case sets are derived, so pin that they are non-empty.

    An empty set makes every test above pass by having no cases at all - the
    failure mode a derived parametrization has and a listed one does not.
    """
    providers = _providers()
    config_key_providers = {name for name, _ in _config_key_cases()}
    defaults_providers = {name for name, _ in _defaults_cases()}

    expected_config_keys = {name for name, cfg in providers.items() if cfg.get("config_keys")}
    expected_defaults = {name for name, cfg in providers.items() if cfg.get("defaults")}

    assert config_key_providers == expected_config_keys
    assert defaults_providers == expected_defaults
    assert len(_config_key_cases()) > 50, "config_keys cases collapsed - the registry read is wrong"
    assert defaults_providers, "no provider declares defaults - the defaults guard is vacuous"


@pytest.mark.parametrize(
    "planted",
    ["definitely_not_a_parameter", "n_action_steps"],
    ids=["unknown-key", "the-key-2013-deleted"],
)
def test_the_guard_would_catch_a_planted_orphan(planted: str) -> None:
    """Non-vacuity for the two agreement tests: an unknown key must be reported.

    ``n_action_steps`` is the second case because it is the real one - #2013
    removed it from ``VeraPolicy`` and from the registry, and had the registry
    spelling been missed, this is the shape that would have caught it.
    """
    parameters = _parameters_or_skip("vera")
    assert _orphans([planted], parameters) == [planted]


def test_the_guard_accepts_the_keys_the_registry_actually_declares() -> None:
    """The converse of the meta-test above: the predicate is not simply strict.

    A ``_orphans`` that reported everything would satisfy the planted-orphan
    test and fail the real ones, so pin that a declared key passes.
    """
    cfg = _providers()["vera"]
    parameters = _parameters_or_skip("vera")
    assert cfg["config_keys"], "vera declares no config_keys - this case is vacuous"
    assert _orphans(cfg["config_keys"], parameters) == []


# ─── executable premises ──────────────────────────────────────────────


def test_a_defaults_key_bypasses_the_config_keys_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why ``defaults`` is guarded at all, and guarded more strictly.

    If the defaults loop ever grows an ``allowed_keys`` test, an orphaned default
    becomes unreachable and this premise fails rather than the guard silently
    over-claiming.
    """
    monkeypatch.setattr(
        "strands_robots.registry.policies.get_policy_provider",
        lambda _name: {"config_keys": ["host"], "defaults": {"host": "localhost", "not_a_config_key": 7}},
    )
    kwargs = build_policy_kwargs("vera")
    assert kwargs["not_a_config_key"] == 7, "defaults are now filtered by config_keys"


def test_a_config_keys_entry_the_caller_omits_is_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the asymmetry: ``config_keys`` alone forwards nothing."""
    monkeypatch.setattr(
        "strands_robots.registry.policies.get_policy_provider",
        lambda _name: {"config_keys": ["host", "never_passed"], "defaults": {}},
    )
    assert "never_passed" not in build_policy_kwargs("vera")


def test_a_provider_without_var_keyword_rejects_an_unknown_kwarg() -> None:
    """The hard-failure half of the cost, and that it is still reachable.

    Pins that at least one provider declares no ``**kwargs``, so the strict form
    is protecting a real ``TypeError`` rather than an inert knob everywhere.  If
    every provider grows a ``**kwargs``, the guard is still wanted - the failure
    becomes silent instead of loud - but this premise should be revisited rather
    than deleted.
    """
    without_var_keyword = [
        name
        for name, parameters in _SIGNATURES.items()
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    ]

    assert without_var_keyword, "every provider now takes **kwargs, so an orphan is silent everywhere"

    provider = _providers()[without_var_keyword[0]]
    cls = getattr(importlib.import_module(provider["module"]), provider["class"])
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cls(definitely_not_a_parameter=7)
