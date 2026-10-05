"""Repro for harness bug: docs/learn/agents.md:34 scope promise is false.

The paragraph:
    "`strands_robots.tools` lazy-loads every tool; the ones mounted most:"
followed by a table that names `use_unitree`, `g1_*`, `reachy_*` as rows.

Reality: `strands_robots/tools/__init__.py` has 19 entries in `_LAZY_IMPORTS`.
None of the 23 g1 verbs (packaged in `strands_robots.tools.g1`) nor the 14
reachy verbs (`strands_robots.tools.reachy`) are re-exported, so a user who
believes the paragraph types `from strands_robots.tools import use_unitree`
and hits ImportError. The sibling pages `docs/learn/hardware/unitree.md:64`
and `docs/learn/hardware/reachy-mini.md:47` already import from the
sub-packages directly; only agents.md over-generalises the scope.

Run:
    python harness_repros/tools_lazy_loads_every_tool_false_repro.py
"""

from __future__ import annotations

import sys


def _show(label: str, exc: Exception | None) -> None:
    tag = "PASS" if exc is None else "FAIL"
    msg = "" if exc is None else f"  :: {type(exc).__name__}: {exc}"
    print(f"[{tag}] {label}{msg}")


def main() -> int:
    print("Doc paragraph:  docs/learn/agents.md:34")
    print('                "`strands_robots.tools` lazy-loads every tool; the ones mounted most:"')
    print()
    print("Rows in the table immediately below:")
    print("                use_unitree, g1_*, reachy_*  (plus 11 others)")
    print()
    print("What a user who trusts the paragraph types:")
    print("-" * 72)

    fail_count = 0
    for stmt in (
        "from strands_robots.tools import use_unitree",
        "from strands_robots.tools import g1_move_velocity",
        "from strands_robots.tools import reachy_look",
        "from strands_robots import use_unitree",
    ):
        try:
            exec(stmt, {})
            _show(stmt, None)
        except Exception as e:
            _show(stmt, e)
            fail_count += 1

    print()
    print("What the hardware pages actually tell them (and which works):")
    print("-" * 72)
    for stmt in (
        "from strands_robots.tools.g1 import use_unitree, g1_move_velocity",
        "from strands_robots.tools.reachy import reachy_look",
    ):
        try:
            exec(stmt, {})
            _show(stmt, None)
        except Exception as e:
            _show(stmt, e)

    print()
    print("Scope-mismatch count:")
    import strands_robots.tools as parent
    import strands_robots.tools.g1 as g1
    import strands_robots.tools.reachy as reachy
    print(f"  strands_robots.tools._LAZY_IMPORTS         = {len(parent._LAZY_IMPORTS)}")
    print(f"  strands_robots.tools.g1._LAZY_IMPORTS       = {len(g1._LAZY_IMPORTS)}  (NONE re-exported)")
    print(f"  strands_robots.tools.reachy._LAZY_IMPORTS   = {len(reachy._LAZY_IMPORTS)}  (NONE re-exported)")
    print(f"  Hidden from the parent namespace            = {len(g1._LAZY_IMPORTS) + len(reachy._LAZY_IMPORTS)}")
    print()

    print(f"User-typed-the-paragraph failures: {fail_count}/4")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
