#!/usr/bin/env python3
"""Repro: ``examples/`` has two files with the same ``17_`` prefix, so the
only unique numeric handle for either goes to the one listed first; the
README's intro paragraph claims the top-level walkthroughs run ``01_*..15_*``
when they actually run through ``18_*``; and ``18_so101_pick_and_lift.py`` is
on disk with no row in the README index.

The branch's doc edit fixes the two docs-only halves (intro number range,
row for ``18_so101_pick_and_lift.py``). The on-disk prefix collision at
``17_`` can only be fixed by renaming one of the two files on disk, which
is a cross-cutting rename (tests, strands_robots references, notebooks
README) and is left as a follow-up explicitly scoped on the harness issue.

Run from any directory inside the repo clone:
    python examples_readme_numbering.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
README = EXAMPLES / "README.md"

on_disk = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))

row_re = re.compile(r"^\|\s*(\d{2})\s*\|\s*\[`([^`]+)`\]")
listed: list[tuple[str, str]] = []
for line in README.read_text().splitlines():
    m = row_re.match(line)
    if m:
        listed.append((m.group(1), m.group(2)))

intro = next(
    line
    for line in README.read_text().splitlines()
    if "numbered" in line and "scripts" in line
)

numbers_on_disk = sorted({name.split("_", 1)[0] for name in on_disk})
numbers_in_readme = sorted({n for n, _ in listed})

counts: dict[str, int] = {}
for name in on_disk:
    prefix = name.split("_", 1)[0]
    counts[prefix] = counts.get(prefix, 0) + 1
on_disk_collisions = {n: c for n, c in counts.items() if c > 1}

readme_row_counts: dict[str, int] = {}
for n, _ in listed:
    readme_row_counts[n] = readme_row_counts.get(n, 0) + 1
readme_duplicate_rows = {n: c for n, c in readme_row_counts.items() if c > 1}

missing_rows = [name for name in on_disk if not any(name == f for _, f in listed)]

print("intro:          ", intro.strip())
print("on disk:        ", numbers_on_disk)
print("readme:         ", numbers_in_readme)
print("on-disk 17_x2:  ", on_disk_collisions)
print("readme 17 rows: ", readme_duplicate_rows)
print("files w/o row:  ", missing_rows)

# After this branch's doc edit:
#   * the intro string names the real range
#   * every file on disk has a row in the index
# The structural bug the doc edit cannot repair is the on-disk ``17_``
# prefix collision — two python modules can share a Python import name but
# a POSIX directory does not notice that two files are "the 17th example".
assert on_disk_collisions == {"17": 2}, (
    f"on-disk prefix collision changed shape: {on_disk_collisions!r}"
)
assert "01_*`..`18_*`" in intro, "intro number range still names 15_*"
assert not missing_rows, f"files still missing a README row: {missing_rows}"
# After the fix the README's two 17-rows are intentional and tracked by the
# follow-up rename, not a documentation bug — so this assertion flips.
assert readme_duplicate_rows == {"17": 2}, (
    f"readme duplicate rows changed: {readme_duplicate_rows!r}"
)
print(
    "\nREPRO PASS: doc-only halves fixed on this branch; "
    "the ``17_`` on-disk prefix collision is left for the harness-tracked rename."
)
