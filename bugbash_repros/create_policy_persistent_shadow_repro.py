"""Repro: create_policy("persistent") always TypeErrors despite being advertised.

Public claim (docs/learn/policies/index.md:64):
    "`composite` and `persistent` resolve by module name, outside the registry."

Internal truth (strands_robots/policies/factory.py:97-99 docstring):
    "persistent resolves but cannot be built here: its first parameter is
     named `provider`, which create_policy has already bound, so it is
     constructed directly."

So the factory *knows* this doesn't work, but still:
  * `provider_can_be_created("persistent")` returns True
  * `_resolve_policy_class("persistent")` returns (persistent, PersistentPolicy, {})
  * `policy_kwargs_error(...)` returns None (blind to missing-required-positional)
  * `create_policy("persistent", ...)` surfaces a bare CPython TypeError that
    names neither the provider nor the fix.

Two spellings a user would try after reading the docs page:

    A) create_policy("persistent")
       -> TypeError: PersistentPolicy.__init__() missing 1 required positional argument: 'provider'
          (no mention of create_policy, no 'use PersistentPolicy() directly' hint)

    B) create_policy("persistent", provider="mock")
       -> TypeError: create_policy() got multiple values for argument 'provider'
          (classic Python shadowing error - naming create_policy, not the inner class)

Run on strands-robots @ 4735957 (v0.5.3):

    python bugbash_repros/create_policy_persistent_shadow_repro.py

Exits non-zero when the contract is still broken; 0 once the factory refuses
"persistent" with a message naming the provider and the correct construction path.
"""
from __future__ import annotations

import inspect
import sys

from strands_robots.policies.factory import (
    _resolve_policy_class,
    create_policy,
    policy_kwargs_error,
    provider_can_be_created,
)
from strands_robots.policies.persistent import PersistentPolicy


def main() -> int:
    failures: list[str] = []

    # 1. The factory treats 'persistent' as a resolvable provider.
    if not provider_can_be_created("persistent"):
        failures.append(
            "provider_can_be_created('persistent') is False; the contract check "
            "already refuses it. Nothing to see here."
        )
        return 0 if not failures else 1

    canonical, PolicyClass, resolved_kwargs = _resolve_policy_class("persistent")
    assert canonical == "persistent"
    assert PolicyClass is PersistentPolicy
    assert resolved_kwargs == {}

    # 2. The pre-build guard that is supposed to catch bad construction kwargs
    #    is blind: {} is "accepted" because no name is misspelled or unknown,
    #    yet construction cannot succeed because 'provider' is required.
    if policy_kwargs_error("persistent", PersistentPolicy, {}) is not None:
        print("GOOD: policy_kwargs_error now refuses 'persistent' with no inner provider.")
    else:
        failures.append(
            "policy_kwargs_error('persistent', PersistentPolicy, {}) returns None - "
            "the pre-build guard misses the required-positional collision."
        )

    sig = inspect.signature(PersistentPolicy.__init__)
    first = list(sig.parameters.values())[1]
    assert first.name == "provider" and first.kind.name == "POSITIONAL_OR_KEYWORD", (
        "If the first positional is no longer named 'provider' the collision is gone - "
        "update the repro accordingly."
    )

    # 3A. Spelling A: just the provider name.
    try:
        create_policy("persistent")
    except TypeError as exc:
        msg = str(exc)
        if "persistent" not in msg.lower() or "create_policy" not in msg.lower():
            failures.append(
                "create_policy('persistent') raised a bare CPython TypeError that "
                f"names neither the provider nor create_policy: {msg!r}"
            )
    else:
        failures.append("create_policy('persistent') returned instead of raising.")

    # 3B. Spelling B: user tries to pass through an inner provider.
    try:
        create_policy("persistent", provider="mock")
    except TypeError as exc:
        msg = str(exc)
        # The current bad error names create_policy, which is misleading - the
        # actual issue is the shadowing of PersistentPolicy's `provider`
        # positional. A good error names BOTH: "`provider` is reserved by
        # create_policy; construct PersistentPolicy directly".
        if "reserved" not in msg.lower() and "shadow" not in msg.lower() and "directly" not in msg.lower():
            failures.append(
                "create_policy('persistent', provider='mock') surfaces the raw "
                "Python shadowing error with no fix hint: " + repr(msg)
            )
    else:
        failures.append(
            "create_policy('persistent', provider='mock') returned instead of raising."
        )

    # 4. Baseline: direct instantiation works - proving this is purely a factory
    #    dispatch problem, not an inner class problem.
    direct = PersistentPolicy(provider="mock")
    assert type(direct).__name__ == "PersistentPolicy"

    if failures:
        print("FAIL - create_policy('persistent') contract is broken:")
        for f in failures:
            print("  -", f)
        print()
        print("Public docs claim it's a valid create_policy() provider:")
        print("  docs/learn/policies/index.md:64:")
        print("    '`composite` and `persistent` resolve by module name, outside the registry.'")
        print()
        print("Internal docstring admits the trap:")
        print("  strands_robots/policies/factory.py:97-99 (docstring of list_aliases).")
        return 1

    print("OK - create_policy('persistent') now refuses or succeeds with a clear contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
