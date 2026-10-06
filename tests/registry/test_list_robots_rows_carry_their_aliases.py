"""Every alias ``list_aliases`` counts is listed on the row of the robot it names.

``format_robot_table`` footers ``Aliases: N`` from :func:`list_aliases`, and
``Robot("g1")`` resolves through that map, so a reader of :func:`list_robots`
must be able to find ``g1`` on the ``unitree_g1`` row without a second call.
"""

from __future__ import annotations

from strands_robots.registry import list_aliases, list_robots


def test_each_alias_appears_on_exactly_the_row_it_resolves_to() -> None:
    rows = list_robots()
    listed = {(alias, r["name"]) for r in rows for alias in r["aliases"]}
    assert listed == set(list_aliases().items())
    assert sum(len(r["aliases"]) for r in rows) == len(list_aliases())
    g1 = next(r for r in rows if r["name"] == "unitree_g1")
    assert "g1" in g1["aliases"]
    assert all(r["aliases"] == sorted(r["aliases"]) for r in rows)
