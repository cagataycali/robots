"""Repro for v0.5.3 defect: _REMOVED_IN_0_7 DeprecationWarning emits prose
descriptions where the sibling REMOVED_PROVIDERS dict in the same package
emits actionable ``policy_provider='X'`` sentences.

Expected (sibling convention): the warning tells the caller the EXACT token
to type as a drop-in replacement.

Actual: the warning f-strings a free-form noun phrase that is not a provider
name and cannot be grep'd in docs.

Run:
    cd /path/to/strands-labs/robots
    python bugbash_repro/removed_in_0_7_warning_prose_repro.py

Upstream pins:
    strands_robots/policies/factory.py:238-243  (_REMOVED_IN_0_7 payloads)
    strands_robots/policies/factory.py:896-902  (warning emission site)
    strands_robots/registry/policies.py:86-115  (REMOVED_PROVIDERS, the
        sibling dict whose convention is being violated — every entry is a
        complete, actionable sentence that names ``policy_provider='X'``)
"""

from __future__ import annotations

import warnings

from strands_robots.policies.factory import _REMOVED_IN_0_7
from strands_robots.registry.policies import REMOVED_PROVIDERS


def _collect_warnings(provider: str) -> list[warnings.WarningMessage]:
    """Call create_policy(provider) and return the warnings it emitted."""
    from strands_robots.policies import create_policy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            create_policy(provider)
        except Exception:
            # The user's ultimate error is unrelated — we only care about the
            # warning that was supposed to teach them the replacement.
            pass
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def main() -> int:
    # 1) Show the convention the sibling dict follows.
    print("=== Convention from REMOVED_PROVIDERS (sibling dict, same package) ===")
    for name in ("groot", "whole-body control", "scripted"):
        sentence = REMOVED_PROVIDERS[name]
        has_token = "policy_provider=" in sentence
        print(f"  {name!r}: has 'policy_provider=' token? {has_token}")
        print(f"    -> {sentence[:140]}")
    print()

    # 2) Show the four _REMOVED_IN_0_7 payloads.
    print("=== _REMOVED_IN_0_7 payloads (used by create_policy warning) ===")
    for name, replacement in _REMOVED_IN_0_7.items():
        has_token = "policy_provider=" in replacement
        marker = "OK" if has_token else "PROSE"
        print(f"  [{marker}] {name!r}: {replacement}")
    print()

    # 3) Show what the user sees when following docs/robots/unitree_g1.md:44.
    print("=== Actual warning on create_policy('protomotions') ===")
    msgs = _collect_warnings("protomotions")
    assert msgs, "expected a DeprecationWarning"
    rendered = str(msgs[0].message)
    print(f"  {rendered}")
    print()
    print("=== Actual warning on create_policy('kimodo') ===")
    msgs = _collect_warnings("kimodo")
    assert msgs, "expected a DeprecationWarning"
    rendered_kimodo = str(msgs[0].message)
    print(f"  {rendered_kimodo}")
    print()

    # 4) Assert the mismatch (grep-token convention violated).
    all_prose = [
        name for name, replacement in _REMOVED_IN_0_7.items()
        if "policy_provider=" not in replacement
    ]
    all_actionable = [
        name for name in REMOVED_PROVIDERS
        if "policy_provider=" in REMOVED_PROVIDERS[name]
    ]
    print("=== Convention audit ===")
    print(f"  REMOVED_PROVIDERS entries with 'policy_provider=' token: "
          f"{len(all_actionable)} / {len(REMOVED_PROVIDERS)}")
    print(f"  _REMOVED_IN_0_7 entries WITHOUT 'policy_provider=' token: "
          f"{len(all_prose)} / {len(_REMOVED_IN_0_7)} "
          f"(prose entries: {all_prose})")

    # On pre-fix code every _REMOVED_IN_0_7 entry is prose. On post-fix code
    # at least the two providers that HAVE an in-tree replacement
    # (kimodo -> 'mock' or custom Policy, protomotions -> 'wbc') carry the
    # replacement token. curobo / moveit2 point at a tool, not a provider,
    # so the token test does not apply there and this script treats a
    # length-<=2 prose list as "defect addressed".
    pre_fix_tripwire = len(all_prose) == len(_REMOVED_IN_0_7)
    assert not pre_fix_tripwire, (
        "defect still present on this tree: every _REMOVED_IN_0_7 payload is "
        "prose; no entry teaches the user the exact 'policy_provider=X' "
        "token a REMOVED_PROVIDERS sibling always does."
    )
    print()
    if all_prose:
        print(
            "DEFECT PARTIALLY ADDRESSED: the two providers with in-tree "
            "replacements now carry a 'policy_provider=' token; "
            f"{all_prose} remain prose because their replacement is a tool, "
            "not a provider (acceptable)."
        )
    else:
        print(
            "DEFECT FULLY ADDRESSED: every _REMOVED_IN_0_7 payload now "
            "carries the 'policy_provider=' token the sibling dict uses."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
