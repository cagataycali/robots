"""The suite's own safety interlock: no test run may be able to reach a real fleet.

Q30 was a full pytest sweep that E-STOPPED cagatay's live arms for three hours; Q32
was a test run that joined the fleet as a `gateway-*` peer and hung on shutdown.
Both needed the same precondition - a process that could open a real mesh session -
and the only thing standing between the suite and that precondition is the kill
switch conftest sets. So assert it, here, as a test: if someone loosens conftest
again (it used to be `setdefault`, which any ambient `STRANDS_MESH=true` disarmed),
this fails immediately and by name instead of the fleet finding out first.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.mesh.core import mesh_kill_switch_engaged

ALLOW = os.environ.get("STRANDS_TEST_ALLOW_LIVE_MESH", "").strip().lower() in ("1", "true", "yes")


@pytest.mark.skipif(ALLOW, reason="STRANDS_TEST_ALLOW_LIVE_MESH: live mesh deliberately allowed")
def test_the_mesh_kill_switch_is_engaged_for_this_whole_run() -> None:
    assert os.environ.get("STRANDS_MESH") == "false", (
        "conftest must FORCE STRANDS_MESH=false. If this fails, the suite inherited a "
        "mesh-enabling environment and can publish to a real fleet."
    )
    assert mesh_kill_switch_engaged() is True


@pytest.mark.skipif(ALLOW, reason="STRANDS_TEST_ALLOW_LIVE_MESH: live mesh deliberately allowed")
def test_an_ambient_mesh_true_cannot_disarm_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin that matters: the OLD conftest (setdefault) would have left this 'true'."""
    monkeypatch.setenv("STRANDS_MESH", "true")
    # A test may of course opt itself back in - that is monkeypatch's job and it is
    # undone at teardown. What must never happen is the SUITE starting out that way,
    # which the test above pins. Here we only prove the predicate reads the live env,
    # so the check above is a real observation rather than a constant.
    assert mesh_kill_switch_engaged() is False
