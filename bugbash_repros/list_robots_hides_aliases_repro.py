"""Repro: list_robots() output hides the 139 aliases users navigate by.

bugbash-strands-robots v0.5.3 — rotation target: registry_404s

Symptom
-------
``Robot("g1")`` works (alias → ``unitree_g1``), and ``docs/reference/project/
design-wholebody-teleop.md:28`` and ``design-driver-composition.md:46`` teach
exactly that spelling. But ``list_robots()`` and ``format_robot_table()`` both
list the canonical ``unitree_g1`` only, with no ``aliases`` field on the row.
A user who ran ``list_robots()`` to see "what's registered" sees no ``g1`` and
reasonably concludes the design-doc incantation is stale.

The sibling listing ``list_discoverable()`` carries ``g1`` (as an MJCF long-tail
description name) and NOT ``unitree_g1``. The two listings therefore split
the same robot across them — a false dichotomy.

The table footer even reads "``Aliases: 139``" but no row in the body shows a
single one.

Expected
--------
``list_robots()`` entries SHOULD surface the ``aliases`` already declared in
``strands_robots/registry/robots.json`` so a reader of the table can map the
``Robot("g1")`` spelling in design-docs back to the ``unitree_g1`` row.

Actual
------
Entry fields are hard-coded in ``strands_robots/registry/robots.py:303-311`` to
``name, description, category, joints, has_sim, has_real, source`` — the
``aliases`` list from the registry is dropped on the floor.

Run
---
``python3 bugbash_repros/list_robots_hides_aliases_repro.py``
"""

from __future__ import annotations

import strands_robots
from strands_robots import Robot, list_discoverable, list_robots
from strands_robots.registry.robots import list_aliases


def main() -> int:
    print("--- Does Robot('g1') work? (docs teach this spelling) ---")
    r = Robot("g1")
    print(f"OK: built {type(r).__name__}")

    print("\n--- Does list_robots() surface 'g1'? ---")
    names = sorted(x["name"] for x in list_robots())
    print(f"'g1'          in list_robots(): {'g1' in names}")
    print(f"'unitree_g1'  in list_robots(): {'unitree_g1' in names}")

    print("\n--- Does list_discoverable() surface 'unitree_g1'? ---")
    disc = list_discoverable()
    print(f"'g1'          in list_discoverable(): {'g1' in disc}")
    print(f"'unitree_g1'  in list_discoverable(): {'unitree_g1' in disc}")

    print("\n--- The row for unitree_g1 (post-fix carries aliases) ---")
    for row in list_robots():
        if row["name"] == "unitree_g1":
            print(f"keys: {sorted(row.keys())}")
            print(f"row:  {row}")
            break

    print(f"\n--- But {len(list_aliases())} aliases do exist (list_aliases()) ---")
    print(f"unitree_g1 aliases (from robots.json): "
          f"{[a for a, t in list_aliases().items() if t == 'unitree_g1']}")

    # Regression-test shape: these assertions hold on this branch (fix
    # applied) and will fail on an unfixed upstream main. Flip to red by
    # reverting the three-line addition in
    # ``strands_robots/registry/robots.py:303-311``.
    assert "unitree_g1" in names, "unitree_g1 must stay in list_robots()"
    assert "unitree_g1" not in disc, (
        "list_discoverable splits the same robot across two listings"
    )
    row = next(r for r in list_robots() if r["name"] == "unitree_g1")
    assert "aliases" in row, "DEFECT (unfixed main): entries drop the 'aliases' field"
    assert "g1" in row["aliases"], "'g1' must appear in the entry's aliases"
    print("\nALL ASSERTIONS HOLD → fix is in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
