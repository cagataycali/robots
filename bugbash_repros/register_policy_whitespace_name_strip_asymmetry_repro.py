"""Repro: register_policy() accepts a name with surrounding/embedded whitespace,
then create_policy() with the natural (stripped) spelling can't find it.

Root cause (strands_robots/policies/factory.py:88):

    if not isinstance(spelling, str) or not spelling.strip():
        raise TypeError(
            f"register_policy(): {param} must be a non-empty str, "
            f"got {type(spelling).__name__}: {spelling!r}."
        )
    ...
    _runtime_registry[name] = loader     # <-- stores un-stripped `name`

The validator uses `spelling.strip()` to detect emptiness, acknowledging that
whitespace is noise, but then persists the raw string as the registry key.
`create_policy()` looks the user's input up with exact-string equality, so a
caller who types the natural spelling ('myprov') gets a 404 while the key
('myprov ', 'myprov\t', '\nmyprov') sits unreachable in _runtime_registry.

The did-you-mean cascade then surfaces the whitespace-decorated spelling verbatim --
    "Unknown policy provider: 'myprov'. Did you mean: 'myprov\\t'?"
-- a hint a human cannot reliably transcribe (does \\t mean "backslash-t" or a
literal tab?). This is the exact shape #811 asked to be closed (register_policy
input validation) and that #843 reported for the sibling resolve_name path
(whitespace typo in Robot() name): resolve_name strips before lookup, so Robot()
tolerates whitespace, while register_policy both-accepts-and-preserves it.

Impact: a user who copy-pastes a provider name from a doc / Jupyter cell /
chat message (any surface that may append a stray space, tab, or newline) sees
their custom provider "registered" successfully, then every subsequent
create_policy() call for the natural spelling fails. The did-you-mean hint
cannot resolve the mystery; the user flips to overwrite=True thinking there is
a conflict; nothing changes.

Fix (1 LOC): either normalize at the write seam (`_runtime_registry[name.strip()]`)
or reject whitespace-decorated spellings at the validator -- the same choice
resolve_name made on the robot side at registry/user_registry.py:270 (strip
before compare).

Run:
    python bugbash_repros/register_policy_whitespace_name_strip_asymmetry_repro.py
"""

from __future__ import annotations

from strands_robots.policies.factory import (
    _runtime_aliases,
    _runtime_registry,
    create_policy,
    policy_provider_error,
    provider_can_be_created,
    register_policy,
)
from strands_robots.policies.mock import MockPolicy


def _reset():
    _runtime_registry.clear()
    _runtime_aliases.clear()


def _show(spelling_stored: str, spelling_queried: str) -> dict:
    _reset()
    register_policy(spelling_stored, lambda: MockPolicy)
    stored_ok = spelling_stored in _runtime_registry
    can = provider_can_be_created(spelling_queried)
    err = policy_provider_error(spelling_queried)
    try:
        obj = create_policy(spelling_queried)
        built = type(obj).__name__
    except Exception as e:
        built = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
    return {
        "stored": spelling_stored,
        "queried": spelling_queried,
        "key_in_runtime_registry": stored_ok,
        "provider_can_be_created(queried)": can,
        "create_policy(queried)": built,
        "did_you_mean_fragment": (err or "").split("Available")[0].strip() if err else None,
    }


def main() -> int:
    cases = [
        (" myprov ", "myprov"),      # surrounding spaces
        ("myprov\t", "myprov"),      # trailing tab (invisible in logs)
        ("\nmyprov", "myprov"),      # leading newline
        ("myprov ", "myprov "),      # same whitespace both ends: works, but every peer surface strips
    ]

    import json

    print("=== register_policy() whitespace / strip-asymmetry repro ===")
    print()
    for stored, queried in cases:
        row = _show(stored, queried)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print()

    row = _show(" myprov ", "myprov")
    assert row["key_in_runtime_registry"] is True, "register_policy silently kept the raw key"
    assert row["provider_can_be_created(queried)"] is False, (
        "preflight should refuse the natural spelling if that's what create_policy does"
    )
    assert row["create_policy(queried)"].startswith("ValueError"), (
        "create_policy refuses the natural spelling - exactly the bug"
    )
    assert "' myprov '" in (row["did_you_mean_fragment"] or ""), (
        "did-you-mean surfaces the whitespace-decorated spelling verbatim; "
        "a human cannot transcribe it from a log"
    )
    print("DEFECT REPRODUCED on this HEAD.")
    print(
        "Fix the write seam at strands_robots/policies/factory.py:109 (or the "
        "validator at :88): strip once, then both store and compare the stripped form."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
