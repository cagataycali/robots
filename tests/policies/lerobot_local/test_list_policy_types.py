"""Tests for the ``lerobot_local`` policy-type discovery surface.

``list_policy_types()`` lets a caller enumerate the ``policy_type`` strings the
installed lerobot can resolve, instead of reading lerobot internals to guess.
The same list also turns ``resolve_policy_class_by_name``'s previously
dead-end "could not resolve" error into an actionable one that names the valid
choices.
"""

from __future__ import annotations

import sys

import pytest

# Skip the whole module unless lerobot is importable (the policy-type registry
# is sourced from lerobot's own draccus choice registry).
pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local import list_policy_types  # noqa: E402
from strands_robots.policies.lerobot_local.resolution import (  # noqa: E402
    resolve_policy_class_by_name,
)


def test_list_policy_types_is_sorted_and_includes_core_families() -> None:
    """The discovery surface returns the resolvable types, sorted and deduped."""
    types = list_policy_types()
    assert types, "expected a non-empty list of policy types with lerobot installed"
    assert types == sorted(types), "policy types must be returned sorted"
    assert len(types) == len(set(types)), "policy types must be deduplicated"
    # ACT and Diffusion ship in every lerobot >= 0.4; assert the stable core.
    for core in ("act", "diffusion"):
        assert core in types, f"expected core policy type {core!r} in {types}"


def test_listed_types_actually_resolve() -> None:
    """Every advertised type resolves to a concrete policy class.

    A discovery surface that lists types which then fail to resolve would be
    worse than none; tie the two together so they cannot drift.
    """
    for policy_type in list_policy_types():
        cls = resolve_policy_class_by_name(policy_type)
        assert isinstance(cls, type)
        assert cls.__name__.endswith("Policy")


def test_unknown_type_error_enumerates_available_types() -> None:
    """The unresolvable-type error names the valid choices (actionable error).

    Pre-fix the message ended with a bare "Ensure lerobot is installed" hint
    and gave a user with a typo'd ``policy_type`` no way to discover the right
    spelling; this regression pins the enumerated remedy.
    """
    types = list_policy_types()
    with pytest.raises(ImportError) as excinfo:
        resolve_policy_class_by_name("definitely_not_a_real_policy_type")
    message = str(excinfo.value)
    assert "definitely_not_a_real_policy_type" in message
    # The actionable part: the error enumerates the resolvable policy types.
    assert all(t in message for t in types), f"error message should list every available type {types}, got: {message}"


def test_list_policy_types_empty_when_lerobot_config_unimportable(monkeypatch) -> None:
    """A missing dependency yields an empty list, never an exception.

    ``list_policy_types`` is a discovery surface, so it degrades gracefully:
    setting the config module to ``None`` in ``sys.modules`` makes the internal
    ``from lerobot.configs.policies import PreTrainedConfig`` raise ImportError,
    and the function must swallow it and return ``[]``.
    """
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", None)
    assert list_policy_types() == []
