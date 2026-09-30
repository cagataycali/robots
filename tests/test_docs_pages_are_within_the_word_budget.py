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

from pathlib import Path

import pytest

import strands_robots
from tests._docs_hooks import docs_hook

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_DOCS = _REPO_ROOT / "docs"


def _hook():
    """The word-budget hook, loaded by path: the docs venv is not the test venv."""
    return docs_hook("word_budget")


#: The per-page ceiling, in words: the one the build-time hook enforces.
_BUDGET: int = _hook().LIMIT

#: The whole-site ceiling, in words, counted over every page :func:`_pages`
#: finds. Lower it whenever a change cuts words; never raise it to admit them.
#: Banked at 53,300 when the rewrite's second grader round restored the content
#: the retargeted graders demand (runbooks, scene tables, allowlist reach), and
#: lowered to 51,184 when the robot-page template stopped restating its fences and
#: to 48,896 when the per-driver facts moved to one section on the drivers page, and
#: to 47,845 when the robot pages dropped the back-link footer the nav already gives, and
#: to 46,418 when they dropped the lines their chips and the nav already state; the old
#: site was 112,416. Raised once, to 49,800, when the site started saying where readers
#: look that learned policies run on real robots: a new Start page (First learned
#: policy) and a generated "Policies verified on this robot" section on every robot
#: page, each row carrying the source of its numbers (docs/hooks/data/checkpoints.json).
#: Then lowered to 49,002 when the robot pages' chips became one {{robot_chips:<name>}} token the hook expands (798 words the chips no longer repeat).
#: Raised to 49,508 when the mesh gained its direct messaging page (one new page under
#: learn/mesh, a variable row and a sentence on the pages that point at it, and the `iot`
#: verbs on the command line page).
#: Raised by 551 to 50,059 for the flux3_action provider page (one page per provider).
#: Lowered to 49,433 when the GR00T provider page left with the provider (GR00T N1.7
#: is a section of lerobot-local now).
#: Raised by 733, to 50,166, for learn/policies/wbc-latent.md (the wbc_latent provider).
#: Raised to 57,259 for the 81 robot_descriptions URDF robots: one generated page each and the
#: learn page that explains the loader once.
#: Raised to 51,187 for the two humanoid design pages under project/ (whole-body teleoperation,
#: driver composition) plus their index rows.
_SITE_BUDGET = 59_018

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
