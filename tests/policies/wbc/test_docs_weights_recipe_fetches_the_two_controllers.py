"""The WBC weights recipe fetches the two controllers, not the tree around them.

The checkpoint step on ``docs/policies/wbc.md`` is the first command a WBC user
runs, and the two G1 controllers it needs are 1.8 MB each: obtaining them by
cloning ``NVlabs/GR00T-WholeBodyControl`` downloads a 4.6 GB git-LFS tree for
3.6 MB of ONNX. Pin the recipe as a per-artifact fetch, and pin the filenames it
names to the constants :class:`~strands_robots.policies.wbc.WBCPolicy` resolves
(a transcription can drift from the loader; a constant cannot).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.policies.wbc.policy import (
    _MAIN_POLICY_CANONICAL,
    _MAIN_POLICY_FILENAME,
    _WALK_POLICY_CANONICAL,
    _WALK_POLICY_FILENAME,
)

_PAGE = Path(__file__).resolve().parents[3] / "docs" / "policies" / "wbc.md"
_FENCE = re.compile(r"^```[a-z]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_ONNX = re.compile(r"[\w.-]+\.onnx")
_CKPT_DIR = "grootwbc-g1"  # the page's placeholder checkpoint directory
_FETCH_VERBS = ("curl", "wget", "git ", "cp ")


def _weights_recipes() -> list[str]:
    """Every fence on the page that populates the checkpoint directory.

    Selected by the destination rather than by an artifact name, so a recipe
    that writes the files under another spelling (a brace expansion, a rename)
    is still graded instead of silently skipped.
    """
    fences = [m.group(1) for m in _FENCE.finditer(_PAGE.read_text(encoding="utf-8"))]
    return [f for f in fences if _CKPT_DIR in f and any(v in f for v in _FETCH_VERBS)]


@pytest.fixture(scope="module")
def recipes() -> list[str]:
    found = _weights_recipes()
    assert found, f"{_PAGE.name} states no shell recipe that fetches {_MAIN_POLICY_CANONICAL}"
    return found


def test_recipe_fetches_the_artifacts_not_the_repository(recipes: list[str]) -> None:
    """A whole-repo clone pulls the 4.6 GB LFS tree to obtain 3.6 MB of weights."""
    for recipe in recipes:
        assert "git clone" not in recipe, (
            f"{_PAGE.name} fetches the WBC weights by cloning the upstream repository; "
            "the two controllers are 1.8 MB each and the clone is a 4.6 GB git-LFS tree. "
            "Fetch the two artifacts directly."
        )


def test_recipe_names_both_controllers_the_loader_accepts(recipes: list[str]) -> None:
    """Both files, spelled as names ``_default_onnx_paths`` resolves."""
    accepted = {
        _MAIN_POLICY_FILENAME,
        _WALK_POLICY_FILENAME,
        _MAIN_POLICY_CANONICAL,
        _WALK_POLICY_CANONICAL,
    }
    for recipe in recipes:
        named = set(_ONNX.findall(recipe))
        assert {_MAIN_POLICY_CANONICAL, _WALK_POLICY_CANONICAL} <= named, (
            f"{_PAGE.name} recipe must fetch both the main and the walk controller, got {sorted(named)}"
        )
        unknown = named - accepted
        assert not unknown, f"{_PAGE.name} recipe fetches ONNX names the loader does not resolve: {sorted(unknown)}"
