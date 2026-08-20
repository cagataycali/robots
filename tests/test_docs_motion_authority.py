"""Documentation must state the two motion switches, with the DEFAULTS the code implements.

A permissions page and an env row tell an operator a variable exists. Neither tells them which way it
points when unset — and the two point OPPOSITE ways, on purpose:

* ``STRANDS_DASH_AGENT_PHYSICAL_MOTION`` unset means REFUSE (the agent may not start a real arm).
* ``STRANDS_DASH_TASK_REQUIRES_CONFIRM`` unset means ALLOW (any token holder may POST a real task).

Getting either default backwards in prose is worse than not documenting it. A reader who believes the
route is confirmed-only leaves a tunnelled dashboard open on purpose; a reader who believes the agent is
allowed by default goes looking for a switch to turn off and finds nothing, then concludes the guard is
broken. So the defaults are graded here against the functions that implement them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.dashboard.agent_motion import (
    MOTION_ENV,
    TASK_CONFIRM_ENV,
    agent_motion_allowed,
    task_confirm_required,
)

DOC = Path(__file__).resolve().parents[1] / "docs/dashboard/remote-access.md"
ARM = {"presence": {"hw": "so_follower"}}


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text()


def test_the_page_names_both_variables(text: str):
    assert MOTION_ENV in text
    assert TASK_CONFIRM_ENV in text


def test_the_documented_defaults_are_the_implemented_ones(text: str):
    """Read the behaviour out of the code, then require the page to say it in the same direction."""
    agent_refused = not agent_motion_allowed(
        action="task", peer=ARM, target="arm", env={})["allowed"]
    route_open = not task_confirm_required({})
    assert agent_refused and route_open, "the code changed; this page's prose is now wrong"

    row = next(ln for ln in text.splitlines() if MOTION_ENV in ln and "|" in ln)
    assert "refuse" in row.lower(), f"agent default must read as a refusal, got: {row}"
    row = next(ln for ln in text.splitlines() if TASK_CONFIRM_ENV in ln and "|" in ln)
    assert "allow" in row.lower(), f"route default must read as open, got: {row}"


def test_the_page_does_not_oversell_the_lock(text: str):
    """It is defeatable by any client willing to send the marker. Calling it security would earn it a
    trust it cannot carry — and would make a leaked token feel handled when it is not."""
    low = text.lower()
    assert "anti-accident" in low
    assert "not a security boundary" in low or "not the fix" in low
    assert "rotate the token" in low


def test_the_page_says_stopping_is_never_gated(text: str):
    assert "stopping is never gated" in text.lower()


def test_the_page_points_at_the_screen_not_only_the_shell(text: str):
    """Both switches are visible in Settings > permissions; a doc that only shows env vars teaches the
    operator to hand-edit .env, which is the one path the UI cannot keep consistent."""
    assert "Settings → permissions" in text or "Settings > permissions" in text
