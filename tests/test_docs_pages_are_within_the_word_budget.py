# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documentation page stays under 1,500 words, and the whole site stays under its ceiling.

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

A page coming inside the ceiling does not mean the site got shorter. A split
pays a front matter, a nav row and a see-also block, so the per-page rule is
satisfied by moving words rather than removing them: over the 22 pages added
most recently the site grew from 113,472 words to 113,954 while the number of
pages owing a split fell from 25 to 2, and nothing graded the difference.
:data:`_SITE_BUDGET` grades the site the way the rule above grades the page,
and it only ever moves down - a diff that adds words pays for them with a cut,
and a diff that cuts words lowers the ceiling so the room it freed cannot be
spent unnoticed.

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
        "device-connect.md",
        "reference/configuration.md",
    }
)


#: The whole-site ceiling, in words, counted over every page :func:`_pages`
#: finds. Lower it whenever a change cuts words; never raise it to admit them.
_SITE_BUDGET = 113_336

#: How far :data:`_SITE_BUDGET` may sit above the real total before it is stale.
#: A cut larger than this has to be banked by lowering the ceiling.
_SITE_SLACK = 500


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


def test_the_site_total_is_within_budget() -> None:
    """The site as a whole stays under :data:`_SITE_BUDGET`."""
    total = sum(_words(p) for p in _pages())
    assert total <= _SITE_BUDGET, (
        f"docs/ is {total} words, over the {_SITE_BUDGET}-word site ceiling by {total - _SITE_BUDGET}. "
        "Splitting a page does not pay for new words - it adds a front matter, a nav row and a "
        "see-also block. Cut words elsewhere in this change, or delete a page that no longer earns "
        "its place, rather than raising the ceiling."
    )


def test_the_site_budget_is_not_stale() -> None:
    """A cut is banked by lowering the ceiling, so freed room is not spent unnoticed."""
    total = sum(_words(p) for p in _pages())
    assert total > _SITE_BUDGET - _SITE_SLACK, (
        f"docs/ is {total} words, {_SITE_BUDGET - total} under the {_SITE_BUDGET}-word site "
        f"ceiling - more than the {_SITE_SLACK} words of slack the ceiling may carry. Lower "
        "_SITE_BUDGET to the new total so the room this cut freed cannot be spent unnoticed."
    )
