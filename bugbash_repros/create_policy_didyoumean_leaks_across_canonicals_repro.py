"""Minimal repro for the create_policy() "Did you mean" cross-provider leak.

See cagataycali/robots-harness issue for context. In short:

* ``create_policy()`` builds its difflib suggestion pool by concatenating
  :func:`strands_robots.policies.factory.list_providers` (16 canonical names)
  and :func:`list_aliases` (19 aliases/shorthands) without collapsing to
  canonicals (``factory.py:399``).
* ``n=3, cutoff=0.6`` then returns three *strings* even when two or three of
  them route to the SAME canonical, and sometimes mixes in a shorthand that
  routes to a DIFFERENT canonical.

Observed: ``create_policy("protomotion")`` (missing trailing ``s``) suggests
``['protomotions', 'protomotions_g1', 'text2motion']`` -- the first two are
the same canonical ``protomotions``, the third is a shorthand for
``kimodo`` (a different provider, with its own trust-remote-code gate and
weight set). A caller who follows the third suggestion downloads kimodo.

Run from the repo root with the lerobot extra NOT required::

    python bugbash_repros/create_policy_didyoumean_leaks_across_canonicals_repro.py
"""
from __future__ import annotations

import difflib

from strands_robots.policies.factory import list_aliases, list_providers
from strands_robots.registry.policies import _canonical_provider_name, list_policy_providers


def _close(typo: str) -> list[str]:
    """Reproduce the suggestion pool construction from factory.py:399."""
    folded = typo.lower().replace("-", "_")
    return difflib.get_close_matches(
        folded,
        [*list_providers(), *list_aliases()],  # <-- the papercut: pool is not deduped
        n=3,
        cutoff=0.6,
    )


def _canonical_count(suggestions: list[str]) -> dict[str, list[str]]:
    """Group suggestions by the canonical provider each one resolves to."""
    groups: dict[str, list[str]] = {}
    for name in suggestions:
        canon = _canonical_provider_name(name)
        groups.setdefault(canon, []).append(name)
    return groups


def main() -> None:
    print(f"canonical providers (available=): {sorted(list_policy_providers())}\n")

    cases = [
        # (typo, expected behaviour described in prose)
        ("protomotion", "typo of 'protomotions' -- cross-canonical leak"),
        ("cumoton",     "typo of 'cumotion' (alias of curobo) -- two suggestions, same canonical"),
        ("moveit3",     "typo of 'moveit2' -- two suggestions, same canonical"),
        ("kimod0",      "typo of 'kimodo' -- two suggestions, same canonical"),
        ("GTP",         "uppercase 'gtp' -- two suggestions, same canonical"),
    ]

    failures = 0
    for typo, note in cases:
        suggestions = _close(typo)
        groups = _canonical_count(suggestions)
        print(f"typo: {typo!r:16} suggestions: {suggestions}")
        print(f"  grouped by canonical: {groups}")
        print(f"  distinct canonicals : {len(groups)}  ({note})")
        # The user-visible papercut: EITHER the suggestions collapse to one canonical
        # (user sees two names for the same thing) OR they span canonicals (user sees
        # 'text2motion' when they typed 'protomotion' and ends up on kimodo).
        if len(suggestions) > len(groups):
            failures += 1
            if typo == "protomotion" and "kimodo" in groups:
                print(f"  ^^ CROSS-CANONICAL LEAK: 'text2motion' routes to 'kimodo', "
                      "not 'protomotions'")
            else:
                print(f"  ^^ REDUNDANT: multiple suggestions route to {list(groups)[0]!r}")
        print()

    print(f"\nRESULT: {failures}/{len(cases)} typos produce misleading suggestion lists.")
    print("Fix: dedupe the suggestion pool by canonical name before difflib,")
    print("     or dedupe the output of get_close_matches by canonical.")
    print("Anchor: strands_robots/policies/factory.py:399")

    # --- Second half: show the user-visible error after the fix ---
    print("\n" + "=" * 60)
    print("Error strings produced by create_policy() with the fix applied:")
    print("=" * 60)
    from strands_robots.policies import create_policy
    for typo, _note in cases:
        try:
            create_policy(typo)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Did you mean" in msg:
                start = msg.index("Did you mean")
                end = msg.index(". Available") if ". Available" in msg else msg.index("Available") - 1
                clause = msg[start:end]
            else:
                clause = "(no suggestion)"
            print(f"  {typo!r:16} {clause}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
