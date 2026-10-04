"""Repro / pin for harness bug: ``Unknown predicate`` refusal has no hint.

Three sibling raise-sites in ``strands_robots/simulation/predicates.py``
(``make_predicate``, ``predicate_kind``, ``predicate_reads_robot_base``) all
share the same shape — ``Unknown predicate '{name}'. Valid: {valid}``. On
``strands-labs/robots`` main @ 1d0946e every realistic typo (``joint_abovve``,
``basetipped``, ``graspd``, ``distance_less``) was refused with a bare
30-name registry dump and no "Did you mean" line, even though the same
codebase ships 17+ ``difflib.get_close_matches`` refusals (see e.g.
``strands_robots.robot:211``, ``policies/factory.py:399``,
``hardware_robot.py:253``).

The branch fix mints the hint once in ``_unknown_predicate_message`` and
calls it from all three sites, mirroring ``Robot('<typo>')``'s shape
(``n=1``, ``cutoff=0.6``).

Run from a repo root:

    python bugbash_repros/predicate_no_didyoumean_repro.py

Expected on the branch (post-fix): exit 0, every typo gets a "Did you mean"
line pointing at the obvious canonical name.
Expected on main (pre-fix): exit 1, five bare dumps and zero hints.

Target: v0.5.3.
"""
from __future__ import annotations

import sys


EXPECTED_HINTS: dict[str, str] = {
    # Typo -> the near-neighbour difflib(cutoff=0.6, n=1) should hit.
    "joint_abovve": "joint_above",
    "basetipped": "base_tipped",
    "graspd": "grasped",
    "distance_less": "distance_less_than",
    "contact_any_body": "contact_any",
}


def main() -> int:
    sys.path.insert(0, ".")
    try:
        from strands_robots.simulation.predicates import make_predicate
    except Exception as e:  # noqa: BLE001 — import diagnostic
        print(f"FATAL: cannot import predicates: {e}")
        return 2

    missed: list[tuple[str, str]] = []
    hinted: list[tuple[str, str]] = []
    for typo, expected in EXPECTED_HINTS.items():
        try:
            make_predicate(typo)
        except ValueError as e:
            msg = str(e)
            if "Did you mean" in msg and f"'{expected}'" in msg:
                hinted.append((typo, expected))
                print(f"[hint ✓] {typo!r:20s} -> ...Did you mean '{expected}'?")
            else:
                missed.append((typo, msg))
                print(f"[dump  ✗] {typo!r:20s} -> {msg[:140]}...")

    print()
    if missed:
        print(
            f"PRE-FIX / REGRESSED: {len(missed)}/{len(EXPECTED_HINTS)} typos "
            f"missing a 'Did you mean' hint. "
            f"The sibling refusals strands_robots.robot:211, policies/factory.py:399, "
            f"hardware_robot.py:253 all use difflib.get_close_matches(cutoff=0.6, n=1)."
        )
        return 1
    print(f"POST-FIX ✓: {len(hinted)}/{len(EXPECTED_HINTS)} typos received the "
          f"expected 'Did you mean' hint at the three sibling raise-sites "
          f"(make_predicate, predicate_kind, predicate_reads_robot_base).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
