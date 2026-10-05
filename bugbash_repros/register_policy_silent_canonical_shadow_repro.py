"""Repro for harness#TBD — register_policy() silently shadows canonical built-ins.

End-user path only (public `register_policy` + `create_policy` surface).

Symptom
-------
Calling ``register_policy("wbc", loader)`` succeeds silently and routes
every subsequent ``create_policy("wbc", ...)`` to the user's loader.  No
WARNING, no INFO, no refusal, no ``overwrite=`` kwarg.  Same shape for
aliases: ``register_policy("newname", loader, aliases=["random"])`` steals
the canonical ``random -> mock`` alias.

The sibling ``register_robot(name=<canonical>)`` *refuses* with
``ValueError`` unless ``overwrite=True`` is passed (see harness#709), so
the two user-facing factories give opposite answers to the same user
intent.

Run
---
    python bugbash_repros/register_policy_silent_canonical_shadow_repro.py

Exit 0 = DEFECT REPRODUCED (silent shadow succeeded).
Exit 1 = behaviour changed (shadow refused or warned at WARNING+).
"""

from __future__ import annotations

import io
import logging
import sys


def main() -> int:
    # What a normal user sees at the default stderr floor.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    from strands_robots.policies.base import Policy
    from strands_robots.policies.factory import (
        create_policy,
        list_aliases,
        register_policy,
    )
    from strands_robots.registry import (
        get_policy_provider,
        list_policy_providers,
    )

    canonical = sorted(list_policy_providers())
    assert "wbc" in canonical, "canonical 'wbc' provider expected in built-ins"
    pre = get_policy_provider("wbc")
    pre_module = pre.get("module") if isinstance(pre, dict) else None
    print("BEFORE register_policy('wbc', StubPolicy):")
    print(f"  canonical providers (16): {canonical}")
    print(f"  get_policy_provider('wbc').module = {pre_module!r}")
    print(f"  (expected: 'strands_robots.policies.wbc')")

    class StubPolicy(Policy):
        provider_name = "stub"

        def __init__(self, **_kwargs):
            pass

        def get_actions(self, state):  # noqa: ARG002
            return {}

        def set_robot_state_keys(self, keys):  # noqa: ARG002
            pass

        def reset(self):
            pass

    # --- CANONICAL-NAME SHADOW ---
    register_policy("wbc", lambda: StubPolicy)

    after = create_policy("wbc")
    print()
    print("AFTER register_policy('wbc', StubPolicy):")
    print(f"  create_policy('wbc') -> {type(after).__name__}")
    print("  (expected: WBCPolicy; got StubPolicy)")

    # --- ALIAS SHADOW ---
    # Pick an existing canonical alias (e.g. 'random' -> 'mock')
    aliases = list_aliases()
    alias_pick, canonical_target = next(iter(aliases.items()))
    print()
    print(f"ALIAS SHADOW test: {alias_pick!r} -> canonical {canonical_target!r}")
    register_policy("my_stub", lambda: StubPolicy, aliases=[alias_pick])
    after_alias = create_policy(alias_pick)
    print(f"  create_policy({alias_pick!r}) -> {type(after_alias).__name__}")
    print(f"  (expected: canonical {canonical_target!r} impl; got StubPolicy)")

    # --- SIBLING register_robot (same user intent, DIFFERENT answer) ---
    from strands_robots.registry.user_registry import register_robot

    print()
    print("SIBLING register_robot('<canonical built-in>', ...):")
    try:
        register_robot(
            "so101",
            description="hijack",
            category="arm",
            joints=0,
        )
        sibling_refused = False
        sibling_msg = ""
    except ValueError as exc:
        sibling_refused = True
        sibling_msg = str(exc).splitlines()[0]
    print(f"  refused? {sibling_refused}")
    print(f"  message: {sibling_msg[:160]}...")

    # --- Diagnostics the user saw at WARNING floor ---
    print()
    print("Stderr captured at WARNING+ during both register_policy calls:")
    captured = log_buf.getvalue().strip()
    if captured:
        for line in captured.splitlines():
            print(f"  | {line}")
    else:
        print("  <<EMPTY>>")

    # --- Verdict ---
    print()
    both_shadowed = (
        type(after).__name__ == "StubPolicy"
        and type(after_alias).__name__ == "StubPolicy"
    )
    print(f"DEFECT REPRODUCED: {both_shadowed and not captured.strip()}")
    print(f"  canonical name silently shadowed? {type(after).__name__ == 'StubPolicy'}")
    print(f"  canonical alias silently shadowed? {type(after_alias).__name__ == 'StubPolicy'}")
    print(f"  WARNING+ stderr empty? {not captured.strip()}")
    print(f"  sibling register_robot refuses same intent? {sibling_refused}")

    return 0 if (both_shadowed and not captured.strip() and sibling_refused) else 1


if __name__ == "__main__":
    sys.exit(main())
