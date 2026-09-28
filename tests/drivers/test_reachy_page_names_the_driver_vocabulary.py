# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Reachy Mini's page states the vocabulary its driver actually dispatches.

The Mini has no lerobot robot type, so ``docs/learn/hardware/reachy-mini.md``
plus the generated ``docs/robots/reachy_mini.md`` are the only written account
of what an agent can ask it for. The previous page said the native tool
"exposes only ``sensors``, ``status``, and ``stop``" and that camera capture,
audio playback, volume and pixel-directed look "are not implemented", while
:class:`~strands_robots.drivers.reachy.ReachyDriver` declares twenty-four
actions including all four - so a reader learned that four shipped surfaces did
not exist. The new page hands the driver itself to ``Agent(tools=[mini, ...])``,
so every verb in its ``tool_spec`` enum is one the model may send.

The roster is read from the driver's own ``tool_spec`` enum, the literal list a
model picks a verb from, so an action added tomorrow is graded without editing
this file. A name counts as documented when it appears as a code span on either
page, verbatim or as the ``reachy_<name>`` ``@tool`` the page's verb table
spells it with.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.drivers.reachy import ReachyDriver

_DOCS = Path(strands_robots.__file__).resolve().parent.parent / "docs"
_PAGES = (_DOCS / "learn" / "hardware" / "reachy-mini.md", _DOCS / "robots" / "reachy_mini.md")
_PAGE_NAMES = " + ".join(p.name for p in _PAGES)

#: Text inside backticks - the page's spelling for a verb, a parameter or a path.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def _declared_actions() -> tuple[str, ...]:
    """Every ``action`` the driver's published tool schema accepts."""
    schema = ReachyDriver().tool_spec["inputSchema"]["json"]
    return tuple(schema["properties"]["action"]["enum"])


def _documented_names() -> str:
    """Both pages' code spans, joined, for word-boundary lookups."""
    return " ".join(span for page in _PAGES for span in _CODE_SPAN.findall(page.read_text(encoding="utf-8")))


def _names_action(action: str, spans: str) -> bool:
    """``action`` appears as a span, or as the ``reachy_<action>`` verb the page spells."""
    return re.search(rf"\b(?:reachy_)?{re.escape(action)}\b", spans) is not None


@pytest.mark.parametrize("action", _declared_actions())
def test_the_page_names_every_action_the_driver_declares(action: str) -> None:
    """A verb a model may send is a verb the driver's page writes down."""
    assert _names_action(action, _documented_names()), (
        f"{_PAGE_NAMES} never name the {action!r} action (as `{action}` or `reachy_{action}`), which "
        f"ReachyDriver.tool_spec declares and Agent(tools=[mini]) exposes - these pages are this driver's only documentation"
    )


def test_the_roster_and_the_page_are_both_read() -> None:
    """Neither side of the rule above can pass by matching nothing."""
    actions = _declared_actions()
    assert len(actions) >= 20, f"only {len(actions)} actions declared; the rule above barely grades anything"
    assert "camera" in actions, "the camera action is the one the page denied; it must be in the roster"
    spans = _documented_names()
    assert spans, f"{_PAGE_NAMES} have no code spans; every lookup above would fail"
    assert "reachy_camera" in spans, "premise: the verb table spells camera as reachy_camera"
    assert not _names_action("not_a_reachy_action", spans), "the lookup must report a name the pages lack"
    assert _names_action("camera", "reachy_camera") and not _names_action("camera", "reachy_camera_status")
