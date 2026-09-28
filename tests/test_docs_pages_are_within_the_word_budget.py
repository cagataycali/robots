# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documentation page stays under 900 words, and the whole site stays under its ceiling.

A page that grows past a reading's worth of text stops being read: the reference
material a caller needs is buried under prose the code already states, and the
next writer appends rather than replaces. ``docs/reference/recording.md`` once
reached 9,924 words across 58 headings, record, verify and replay in one scroll,
before it was split. The rewritten site answers one question per page and holds
every page, generated ones included, under the budget ``docs/hooks/word_budget.py``
states; there is no exemption list, so a page over budget is split or cut.

A page coming inside the ceiling does not mean the site got shorter. A split
pays a front matter, a nav row and a see-also block, so the per-page rule is
satisfied by moving words rather than removing them: over 22 pages added to the
old site it grew from 113,472 words to 113,954 while the pages owing a split
fell from 25 to 2, and nothing graded the difference. :data:`_SITE_BUDGET`
grades the site the way the hook grades the page, and it only ever moves down:
a diff that adds words pays for them with a cut, and a diff that cuts words
lowers the ceiling so the room it freed cannot be spent unnoticed.

Words are counted the way the hook counts them, ``str.split()`` over the whole
file, front matter and fences included, so the number here is the number
``wc -w`` prints for the same path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_DOCS = _REPO_ROOT / "docs"
_HOOK = _DOCS / "hooks" / "word_budget.py"


def _hook():
    """The word-budget hook, loaded by path: the docs venv is not the test venv."""
    spec = importlib.util.spec_from_file_location("docs_word_budget_hook", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The per-page ceiling, in words: the one the build-time hook enforces.
_BUDGET: int = _hook().LIMIT

#: The whole-site ceiling, in words, counted over every page :func:`_pages`
#: finds. Lower it whenever a change cuts words; never raise it to admit them.
_SITE_BUDGET = 51_500

#: How far :data:`_SITE_BUDGET` may sit above the real total before it is stale.
#: A cut larger than this has to be banked by lowering the ceiling.
_SITE_SLACK = 500


def _pages() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "hooks" not in p.parts)


def _words(path: Path) -> int:
    return _hook().words(path)


def test_the_reader_finds_pages_to_grade() -> None:
    """Guard both rules below against silently scanning an empty tree."""
    assert len(_pages()) >= 50


def test_the_hook_states_the_budget_the_design_promises() -> None:
    """The design holds every page under 900 words; the hook is where that number lives."""
    assert _BUDGET == 900, f"docs/hooks/word_budget.py LIMIT is {_BUDGET}; the design's page budget is 900"


@pytest.mark.parametrize("relpath", sorted(str(p.relative_to(_DOCS)) for p in _pages()))
def test_a_page_is_within_budget(relpath: str) -> None:
    words = _words(_DOCS / relpath)
    assert words <= _BUDGET, (
        f"docs/{relpath} is {words} words, over the {_BUDGET}-word budget. Split it at its H2s or "
        "cut it: an option list becomes a table, and prose that restates a docstring goes."
    )


def test_the_site_total_is_within_budget() -> None:
    """The site as a whole stays under :data:`_SITE_BUDGET`."""
    total = sum(_words(p) for p in _pages())
    assert total <= _SITE_BUDGET, (
        f"docs/ is {total} words, over the {_SITE_BUDGET}-word site ceiling by {total - _SITE_BUDGET}. "
        "Splitting a page does not pay for new words: it adds a front matter, a nav row and a "
        "see-also block. Cut words elsewhere in this change, or delete a page that no longer earns "
        "its place, rather than raising the ceiling."
    )


def test_the_site_budget_is_not_stale() -> None:
    """A cut is banked by lowering the ceiling, so freed room is not spent unnoticed."""
    total = sum(_words(p) for p in _pages())
    assert total > _SITE_BUDGET - _SITE_SLACK, (
        f"docs/ is {total} words, {_SITE_BUDGET - total} under the {_SITE_BUDGET}-word site "
        f"ceiling, more than the {_SITE_SLACK} words of slack the ceiling may carry. Lower "
        "_SITE_BUDGET to the new total so the room this cut freed cannot be spent unnoticed."
    )
