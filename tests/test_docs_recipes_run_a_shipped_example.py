# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recipe page shows a shipped example, the command that runs it, and what it looks like.

``docs/recipes/`` is the lane a newcomer copies from, so a recipe's code is never
typed into the page: it is included from ``examples/`` with a
``--8<-- "examples/<script>.py"`` snippet, which ``check_paths: true`` makes the
strict build refuse when the script is gone. That leaves three ways a recipe can
still lie, each graded here per page: the console fence runs a different script
from the one the page shows, the lead image is missing, or the page links into no
reference page, so a reader who wants the contract has nowhere to go.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_RECIPES = _REPO / "docs" / "recipes"

_SNIPPET = re.compile(r'^--8<-- "(?P<path>examples/[\w/]+\.py)"$', re.M)
_COMMAND = re.compile(r"^\$ MUJOCO_GL=egl python (?P<path>examples/[\w/]+\.py)$", re.M)
_IMAGE = re.compile(r"!\[[^\]]+\]\((?P<src>[^)\s]+)\)")
_REFERENCE_LINK = re.compile(r"\]\(\.\./reference/[\w/-]+\.md(?:#[\w-]+)?\)")


def _recipes() -> list[Path]:
    return sorted(p for p in _RECIPES.glob("*.md") if p.name != "index.md")


def test_the_lane_carries_recipes() -> None:
    """Guard the per-page rules below against an empty directory."""
    assert len(_recipes()) >= 9


@pytest.mark.parametrize("page", _recipes(), ids=lambda p: p.stem)
def test_a_recipe_runs_the_example_it_shows(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    shown = _SNIPPET.findall(text)
    run = _COMMAND.findall(text)
    assert len(shown) == 1, f"{page.name} includes {shown}; a recipe shows exactly one example"
    assert (_REPO / shown[0]).is_file(), f"{page.name} includes {shown[0]}, which does not exist"
    assert run == shown, f"{page.name} shows {shown[0]} but its console fence runs {run}"

    image = _IMAGE.search(text)
    assert image and image.start() < text.index("--8<--"), f"{page.name} does not lead with an image"
    assert (page.parent / image["src"]).resolve().is_file(), f"{page.name}: {image['src']} is not on disk"

    assert _REFERENCE_LINK.search(text), f"{page.name} links into no reference page"


def test_the_index_lists_every_recipe() -> None:
    index = (_RECIPES / "index.md").read_text(encoding="utf-8")
    missing = [p.name for p in _recipes() if f"]({p.name})" not in index]
    assert not missing, f"docs/recipes/index.md does not list {missing}"
