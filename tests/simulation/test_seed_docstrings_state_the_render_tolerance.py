# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``seed`` contract of the rollout surfaces states what "the same trajectory" means on GPU rendering.

Measured on main 9e4f0a3d0 with ``MUJOCO_GL=egl``: eight ``get_observation()["wrist"]``
reads without stepping differed from frame 0 in 2 to 9 pixels each, by 1 LSB; a
seeded ``robotfuel/act_so101_t16b`` rollout run twice ended 0.006 to 0.3 rad apart
after 90 to 150 steps, while ``mock`` (no camera) re-ran bit-exact. The ``seed``
docstrings of ``run_policy`` and ``eval_policy`` promised "the same trajectory on
re-run" without that distinction, and the rollouts page said two seeded
evaluations "replay the same episodes".

The three texts now say the same thing: bit-exact for a state-only policy, to a
render tolerance for a camera policy on GPU rendering, naming ``MUJOCO_GL=egl``
as the case and telling the reader to compare outcomes rather than frames. This
grader holds all three to it, so a later rewording cannot drop the caveat from
one of them and leave the other two making a promise the renderer does not keep.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import strands_robots
from strands_robots.simulation.base import SimEngine

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PAGE = _REPO_ROOT / "docs" / "learn" / "simulation" / "predicates-and-rollouts.md"

#: What every seed contract must say, in these words.
REQUIRED = ("bit-exact", "render tolerance", "MUJOCO_GL=egl", "1 LSB")


def _seed_paragraph(text: str, opener: str) -> str:
    """The paragraph of ``text`` that opens on ``opener``; exactly one must exist."""
    paragraphs = [p for p in text.split("\n\n") if opener in p]
    assert len(paragraphs) == 1, f"expected one paragraph containing {opener!r}, found {len(paragraphs)}"
    return paragraphs[0]


class TestTheDocstrings:
    @pytest.mark.parametrize("phrase", REQUIRED)
    def test_run_policy_seed_states_it(self, phrase: str) -> None:
        doc = inspect.getdoc(SimEngine.run_policy) or ""
        seed_entry = doc.split("seed: Optional master RNG seed", 1)[1].split("policy_kwargs:", 1)[0]
        assert phrase in seed_entry, f"run_policy's seed entry no longer says {phrase!r}"

    @pytest.mark.parametrize("phrase", REQUIRED)
    def test_eval_policy_seed_states_it(self, phrase: str) -> None:
        doc = inspect.getdoc(SimEngine.eval_policy) or ""
        paragraph = _seed_paragraph(doc, "``seed`` pins the eval")
        assert phrase in paragraph, f"eval_policy's seed paragraph no longer says {phrase!r}"

    def test_the_state_only_case_is_the_exact_one(self) -> None:
        doc = inspect.getdoc(SimEngine.run_policy) or ""
        assert "bit-exact for" in doc and "state-only policy" in doc


class TestTheRolloutsPage:
    @pytest.mark.parametrize("phrase", REQUIRED)
    def test_the_seed_paragraph_states_it(self, phrase: str) -> None:
        paragraph = _seed_paragraph(_PAGE.read_text(encoding="utf-8"), "`seed` reseeds the client RNGs once")
        assert phrase in paragraph, f"{_PAGE.name}'s seed paragraph no longer says {phrase!r}"

    def test_the_page_tells_the_reader_what_to_compare(self) -> None:
        paragraph = _seed_paragraph(_PAGE.read_text(encoding="utf-8"), "`seed` reseeds the client RNGs once")
        assert "`success_rate`" in paragraph


def test_the_grader_reports_a_dropped_caveat() -> None:
    """A clean sweep means the texts say it, not that the grader accepts anything."""
    with pytest.raises(AssertionError):
        _seed_paragraph("no seed paragraph here\n\nnor here", "`seed` reseeds the client RNGs once")
