# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documentation page stays under 900 words.

A page that grows past a reading's worth of text stops being read: the reference
material a caller needs is buried under prose the code already states, and the
next writer appends rather than replaces. ``docs/reference/recording.md`` once
reached 9,924 words across 58 headings, record, verify and replay in one scroll,
before it was split. The rewritten site answers one question per page and holds
every page, generated ones included, under the budget ``docs/hooks/word_budget.py``
states; there is no exemption list, so a page over budget is split or cut.

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


def _pages() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "hooks" not in p.parts)


def _words(path: Path) -> int:
    return _hook().words(path)


def test_the_reader_finds_pages_to_grade() -> None:
    """Guard the rule below against silently scanning an empty tree."""
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
