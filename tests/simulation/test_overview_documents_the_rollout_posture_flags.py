# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``docs/simulation/overview.md`` documents the posture-flag domain the facades enforce.

The page documents every numeric domain ``SimEngine.run_policy`` and
``SimEngine.eval_policy`` hold their knobs to - ``n_steps``, ``duration``,
``action_horizon``, ``n_episodes``, ``max_steps`` - each in a paragraph that
names the knob, the refusal and where it sits. The posture flags in the same
signatures (``fast_mode``, ``reset_between``, ``wbc_install_torque_control``,
``async_rtc``) gained their domain through ``SimEngine._validate_posture_flags``
with no paragraph beside those, so a reader of the page saw the numeric knobs
checked and had to infer the flags were not.

Pinned here: the paragraph exists and names every boolean parameter of both
facades, so a fifth flag bound to the domain cannot leave the page describing
four; and the paragraph's claim about *where* the check sits stays true - the
facades check and ``PolicyRunner.run`` does not repeat it - so a later decision
to check on the runner has to move this sentence rather than leave it stale.
The roster is read from the signatures, not spelled here.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation import policy_runner as runner_mod
from strands_robots.simulation.base import SimEngine

_PAGE = Path(inspect.getfile(SimEngine)).resolve().parents[2] / "docs" / "simulation" / "overview.md"

_BOOL_ANNOTATIONS = ("bool", "bool | None", bool, bool | None)


def _boolean_parameters(method: Callable[..., Any]) -> set[str]:
    return {
        name for name, param in inspect.signature(method).parameters.items() if param.annotation in _BOOL_ANNOTATIONS
    }


def _posture_paragraph() -> str:
    """The one paragraph of the page that opens on the posture flags."""
    paragraphs = [p for p in _PAGE.read_text(encoding="utf-8").split("\n\n") if "**posture** flags" in p]
    assert len(paragraphs) == 1, f"expected one posture-flag paragraph in {_PAGE.name}, found {len(paragraphs)}"
    return paragraphs[0]


class TestTheParagraphNamesEveryFlagTheFacadesCheck:
    """A boolean parameter of a rollout facade is named where its domain is documented."""

    def test_the_roster_is_being_read(self) -> None:
        """A broken signature read would make the rule below vacuously green."""
        declared = _boolean_parameters(SimEngine.run_policy) | _boolean_parameters(SimEngine.eval_policy)
        assert {"fast_mode", "reset_between", "wbc_install_torque_control", "async_rtc"} <= declared, declared

    @pytest.mark.parametrize("facade", [SimEngine.run_policy, SimEngine.eval_policy], ids=lambda f: f.__name__)
    def test_every_boolean_parameter_is_in_the_paragraph(self, facade: Callable[..., Any]) -> None:
        named = set(re.findall(r"`([a-z_]+)`", _posture_paragraph()))
        missing = _boolean_parameters(facade) - named
        assert not missing, (
            f"{facade.__name__} declares boolean parameter(s) {sorted(missing)} that the posture-flag "
            f"paragraph of docs/simulation/overview.md does not name"
        )

    def test_the_paragraph_names_the_binding_that_exists(self) -> None:
        paragraph = _posture_paragraph()
        assert "`SimEngine._validate_posture_flags`" in paragraph
        assert callable(getattr(SimEngine, "_validate_posture_flags", None))


class TestTheParagraphSaysWhereTheCheckSits:
    """The page's claim that the facades check and the runner does not is read off the code."""

    def test_the_facade_binds_the_domain(self) -> None:
        source = inspect.getsource(SimEngine._validate_posture_flags)
        assert "boolean_flag_error(" in source

    def test_the_runner_does_not_repeat_it(self) -> None:
        """The paragraph's last sentence.

        If ``PolicyRunner.run`` gains its own posture-flag check, this cell is
        the one that says the sentence "does not repeat it" is now false and
        has to move with the code.
        """
        assert "does not repeat it" in _posture_paragraph()
        source = inspect.getsource(runner_mod)
        assert "boolean_flag_error" not in source, (
            "PolicyRunner now checks posture flags; update the overview paragraph"
        )
