# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documentation page stays under 1,500 words, or is named as still owing a split.

A page that grows past a reading's worth of text stops being read: the reference
material a caller needs is buried under prose the code already states, and the
next writer appends rather than replaces. ``docs/recording.md`` reached 9,924
words across 58 headings - record, verify and replay in one scroll - before it
was split.

The budget is graded as a ratchet rather than a flat rule, because the site
still carries pages that owe the same treatment. :data:`_OVER_BUDGET` names
them, and the second test refuses a stale entry: a page that has been trimmed
must leave the list, so the exemption cannot outlive the page it excuses and the
list can only shrink.

Words are counted the way the budget is stated - ``str.split()`` over the whole
file, front matter and fences included - so the number here is the number
``wc -w`` prints for the same path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_DOCS = _REPO_ROOT / "docs"

#: The per-page ceiling, in words.
_BUDGET = 1500

#: Pages that still exceed :data:`_BUDGET` and owe a split or a cut. Shrink this
#: list by trimming a page, never by raising the budget; a generated reference
#: (a page whose body is a hook token) belongs here permanently.
_OVER_BUDGET = frozenset(
    {
        "api-reference.md",
        "data/episode-labels.md",
        "device-connect.md",
        "getting-started/robot-factory.md",
        "hardware/teleoperation.md",
        "hardware/tools.md",
        "inference/remote.md",
        "mesh.md",
        "policies/cosmos3.md",
        "policies/kimodo.md",
        "policies/lerobot-local.md",
        "policies/protomotions.md",
        "policies/wbc.md",
        "reference/configuration.md",
        "robots/arms.md",
        "robots/humanoids.md",
        "robots/mobile.md",
        "ros2-integration.md",
        "rosbridge-integration.md",
        "rtps-integration.md",
        "security.md",
        "simulation/domain-randomization.md",
        "simulation/isaac.md",
        "simulation/newton.md",
        "simulation/rollouts.md",
        "simulation/world-building.md",
        "training/overview.md",
        "training/rl.md",
        "troubleshooting.md",
    }
)


def _pages() -> list[Path]:
    return sorted(_DOCS.rglob("*.md"))


def _words(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def test_the_reader_finds_pages_to_grade() -> None:
    """Guard both rules below against silently scanning an empty tree."""
    assert len(_pages()) >= 50


@pytest.mark.parametrize("relpath", sorted(str(p.relative_to(_DOCS)) for p in _pages()))
def test_a_page_is_within_budget_or_named_as_owing_a_split(relpath: str) -> None:
    if relpath in _OVER_BUDGET:
        pytest.skip(f"{relpath} is a named exemption; see _OVER_BUDGET")
    words = _words(_DOCS / relpath)
    assert words <= _BUDGET, (
        f"docs/{relpath} is {words} words, over the {_BUDGET}-word budget. Split it at its H2s or "
        "cut it - an option list becomes a table, and prose that restates a docstring goes."
    )


@pytest.mark.parametrize("relpath", sorted(_OVER_BUDGET))
def test_an_exemption_still_names_a_page_over_budget(relpath: str) -> None:
    page = _DOCS / relpath
    assert page.is_file(), f"_OVER_BUDGET names docs/{relpath}, which no longer exists - drop the entry"
    words = _words(page)
    assert words > _BUDGET, (
        f"docs/{relpath} is now {words} words, within the {_BUDGET}-word budget - remove it from "
        "_OVER_BUDGET so the page cannot grow back unnoticed"
    )
