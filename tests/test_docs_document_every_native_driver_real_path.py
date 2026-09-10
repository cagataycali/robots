"""Every native driver ships with a documented ``mode="real"`` construction.

A robot whose registry entry points at a driver in
:mod:`strands_robots.drivers` has no lerobot robot type behind it, so this
project's own pages are the only place its hardware bring-up is written down -
the port to pass, which joints a host may command, which verbs the onboard
controller refuses to share. Losing that text loses the only copy, and it is
easy to lose silently: the sections sit on the family pages beside catalog
material that is generated, so a page edit can take an operating procedure with
it while the diff reads as catalog churn.

The roster comes from :func:`~strands_robots.drivers.list_native_drivers`, the
same registration the factory resolves ``mode="real"`` through, so a driver
added tomorrow is graded without editing this file. Spellings are resolved with
:func:`~strands_robots.registry.resolve_name`, so a page is free to use an alias
(``Robot("go2", ...)`` documents ``unitree_go2``) and there is no second list of
names to keep in step. One documented example per driver is the bar, not one per
robot: a driver covering a family (``ur5e``/``ur10e``) is documented once.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from strands_robots.drivers import list_native_drivers
from strands_robots.registry import resolve_name

REPO_ROOT = Path(__file__).resolve().parents[1]

#: ``Robot("<name>", mode="real")`` as the pages spell it, allowing the line
#: break and keyword spacing a formatted fence uses.
REAL_CONSTRUCTION = re.compile(r"""Robot\(\s*["'](?P<name>[A-Za-z0-9_.-]+)["']\s*,\s*mode\s*=\s*["']real["']""")


def _pages() -> dict[Path, str]:
    """Every prose page a reader can reach, by path."""
    pages = {p: p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "docs").rglob("*.md"))}
    readme = REPO_ROOT / "README.md"
    if readme.is_file():
        pages[readme] = readme.read_text(encoding="utf-8")
    return pages


def _documented_drivers() -> dict[str, set[Path]]:
    """Driver name -> the pages showing one of its robots built for hardware."""
    native = list_native_drivers()
    found: dict[str, set[Path]] = defaultdict(set)
    for path, text in _pages().items():
        for match in REAL_CONSTRUCTION.finditer(text):
            driver = native.get(resolve_name(match.group("name")))
            if driver is not None:
                found[driver].add(path)
    return found


def test_every_native_driver_has_a_documented_real_construction() -> None:
    """A driver a caller can reach is a driver the docs show how to reach."""
    native = list_native_drivers()
    documented = _documented_drivers()
    robots_of: dict[str, list[str]] = defaultdict(list)
    for robot, driver in sorted(native.items()):
        robots_of[driver].append(robot)

    undocumented = sorted(set(native.values()) - set(documented))
    assert not undocumented, (
        'native drivers with no documented mode="real" construction under docs/ or README.md: '
        + "; ".join(f"{d} (drives {', '.join(robots_of[d])})" for d in undocumented)
        + ". These robots have no lerobot robot type, so this tree holds the only bring-up "
        'documentation for them - show one Robot("<name>", mode="real") example per driver.'
    )


def test_the_roster_and_the_pattern_both_find_something() -> None:
    """Neither half of the rule above can pass by matching nothing."""
    native = list_native_drivers()
    assert native, "no native driver registered; the rule above would hold vacuously"
    documented = _documented_drivers()
    assert documented, f"no page documents any of {sorted(set(native.values()))}; the pattern matches nothing"
