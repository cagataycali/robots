"""Repro: create_policy()'s did-you-mean pool omits the two auto-discovered
built-in providers (`composite`, `persistent`), so a typo of either lands on a
far-away provider or on nothing at all.

Root cause: strands_robots/policies/factory.py:445 builds the search pool as
    [*list_providers(), *list_aliases()]
Neither surface reports `composite` or `persistent` by design -- they ship as
prose, see the `list_aliases` docstring at lines 117-157. That is a legitimate
enumeration choice (confirmed by `test_provider_alias_discovery.py`), but the
did-you-mean pool at line 445 inherits the gap: it offers suggestions only from
the enumeration surfaces, not from the full accepted set.

Observable symptoms:
  create_policy("composie")   -> suggests 'cosmos3' (edit distance 4!)
                                 instead of 'composite' (edit distance 1)
  create_policy("persistant") -> suggests NOTHING, even though 'persistent'
                                 is edit distance 1

The two spellings `create_policy` resolves outside the registry (prose-only)
are the two spellings its own "did you mean" cannot offer. A user who typos the
one wrapper they know about gets routed to a different one.

Fix sketch (<10 LOC): add a module-level `_AUTO_DISCOVERED_SPELLINGS` tuple
listing the auto-discovered wrappers, and union it into the search pool at
line 445. `list_providers()` and `list_aliases()` keep their current shape --
only the suggestion pool widens.

Expected after fix:
  create_policy("composie")   -> suggests 'composite'
  create_policy("persistant") -> suggests 'persistent'

Run:
  cd <robots repo root>
  python bugbash_repros/list_providers_hides_composite_persistent_repro.py
"""
from __future__ import annotations

import sys
import traceback

from strands_robots.policies.factory import (
    create_policy,
    import_policy_class,
    list_aliases,
    list_providers,
    provider_can_be_created,
)


def main() -> int:
    failures: list[str] = []

    # Setup: both are factory-resolvable even though neither is reported
    assert provider_can_be_created("composite"), "composite must be resolvable"
    assert provider_can_be_created("persistent"), "persistent must be resolvable"
    import_policy_class("composite")
    import_policy_class("persistent")
    assert "composite" not in list_providers()
    assert "persistent" not in list_providers()
    assert "composite" not in list_aliases()
    assert "persistent" not in list_aliases()

    print("premise OK: 'composite'/'persistent' resolve but no public surface reports them")
    print()

    # The defect: did-you-mean pool at factory.py:445 omits them
    cases = [
        ("composie", "composite"),  # 1-char typo (missing 't'), edit distance 1
        ("persistant", "persistent"),  # common misspelling, edit distance 1
    ]
    for typo, want in cases:
        try:
            create_policy(typo)
        except ValueError as exc:
            text = str(exc)
            if f"'{want}'" in text and "Did you mean" in text:
                print(f"OK   create_policy({typo!r}) suggests {want!r}")
            else:
                msg = (
                    f"BUG  create_policy({typo!r}) does NOT suggest {want!r}: "
                    f"{text[:220]}"
                )
                print(msg)
                failures.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"UNEX create_policy({typo!r}) raised {type(exc).__name__}: {exc}"
            print(msg)
            failures.append(msg)

    print()
    if failures:
        print(f"REPRO FAILED with {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("REPRO PASSED (defect fixed)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
