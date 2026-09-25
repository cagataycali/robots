# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Reachy Mini's page states the vocabulary its driver actually dispatches.

The Mini has no lerobot robot type, so ``docs/hardware/reachy-mini.md`` is the
only written account of what an agent can ask it for. That page said the native
tool "exposes only ``sensors``, ``status``, and ``stop``" and that camera
capture, audio playback, volume and pixel-directed look "are not implemented",
while :class:`~strands_robots.drivers.reachy.ReachyDriver` declares twenty-four
actions including all four - so a reader learned that four shipped surfaces did
not exist.

The roster is read from the driver's own ``tool_spec`` enum, the literal list a
model picks a verb from, so an action added tomorrow is graded without editing
this file. A name counts as documented when it appears as a code span on the
page, which is how the page spells every verb.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.drivers.reachy import ReachyDriver

_PAGE = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "hardware" / "reachy-mini.md"

#: Text inside backticks - the page's spelling for a verb, a parameter or a path.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def _declared_actions() -> tuple[str, ...]:
    """Every ``action`` the driver's published tool schema accepts."""
    schema = ReachyDriver().tool_spec["inputSchema"]["json"]
    return tuple(schema["properties"]["action"]["enum"])


def _documented_names() -> str:
    """The page's code spans, joined, for word-boundary lookups."""
    return " ".join(_CODE_SPAN.findall(_PAGE.read_text(encoding="utf-8")))


@pytest.mark.parametrize("action", _declared_actions())
def test_the_page_names_every_action_the_driver_declares(action: str) -> None:
    """A verb a model may send is a verb the driver's page writes down."""
    assert re.search(rf"\b{re.escape(action)}\b", _documented_names()), (
        f"{_PAGE.name} never names the {action!r} action, which "
        f"ReachyDriver.tool_spec declares - the page is this driver's only documentation"
    )


def test_the_roster_and_the_page_are_both_read() -> None:
    """Neither side of the rule above can pass by matching nothing."""
    actions = _declared_actions()
    assert len(actions) >= 20, f"only {len(actions)} actions declared; the rule above barely grades anything"
    assert "camera" in actions, "the camera action is the one the page denied; it must be in the roster"
    assert _documented_names(), f"{_PAGE.name} has no code spans; every lookup above would fail"
