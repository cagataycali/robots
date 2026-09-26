# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cmd_vel demo never reports a drive the command gate refused.

``/turtle1/cmd_vel`` is a gated command surface, and a script has no
``tool_context`` for an operator to approve one: ``drive`` then RETURNS
``{"status": "error", ...}`` naming the two remedies
(``STRANDS_ROS2_COMMAND_ALLOW`` / ``BYPASS_TOOL_CONSENT``). Both demos discarded
that result:

* ``examples/ros2/rtps_turtle_demo.py`` printed ``done - the turtle should have
  moved (check the turtlesim window)`` and exited 0 against a real ``turtlesim``
  that had not moved at all;
* ``examples/ros2/turtlebot_demo.py`` printed an ``after:`` pose identical to its
  ``before:`` one, as if that were the drive's outcome.

Graded here on one rule, one row per demo: a refused command raises and no claim
is printed, an accepted one is unchanged. No DDS, no ROS 2 and no turtle - the
bridge class each script imports is stubbed at its own seam.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import strands_robots.mesh as mesh

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ros2"

_REFUSED: dict[str, Any] = {
    "status": "error",
    "content": [
        {
            "text": (
                "use_rtps: '/turtle1/cmd_vel' is a safety-critical command surface, blocked for "
                "publish. No tool_context available for operator approval."
            )
        }
    ],
}
_ACCEPTED: dict[str, Any] = {"status": "success", "content": [{"text": "published 15 message(s)"}]}


class _StubBridge:
    """The slice of the bridge classes the demos drive, with a scripted result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.drives = 0

    def advertise(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "advertised"}]}

    def drive(self, **_kwargs: Any) -> dict[str, Any]:
        self.drives += 1
        return self._result

    def stop(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "stopped"}]}

    def get_pose(self) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "x=5.5 y=5.5 theta=0.0"}]}


@dataclass(frozen=True)
class _Demo:
    """One script, the class it imports, and the claim its output must not carry."""

    script: str
    factory: str
    claim: str


DEMOS = [
    _Demo("rtps_turtle_demo.py", "RtpsRobot", "done - the turtle should have moved"),
    _Demo("turtlebot_demo.py", "RosBridgedRobot", "after: "),
]


@pytest.fixture
def stub_bridge(monkeypatch: pytest.MonkeyPatch):
    """Install a stubbed bridge class at the seam a demo imports it from."""

    def install(demo: _Demo, result: dict[str, Any]) -> _StubBridge:
        stub = _StubBridge(result)
        factory = type(
            "_StubFactory",
            (),
            {
                "from_rtps": classmethod(lambda _cls, **_k: stub),
                "from_ros": classmethod(lambda _cls, **_k: stub),
            },
        )
        monkeypatch.setattr(mesh, demo.factory, factory)
        return stub

    return install


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.script)
def test_a_refused_command_is_not_reported_as_the_drives_outcome(demo, stub_bridge, capsys) -> None:
    """The refusal raises, carrying the remedy the gate named, and nothing claims."""
    stub = stub_bridge(demo, _REFUSED)

    with pytest.raises(RuntimeError, match="safety-critical command surface"):
        runpy.run_path(str(_EXAMPLES / demo.script), run_name="__main__")

    assert stub.drives == 1
    assert demo.claim not in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.script)
def test_an_accepted_command_still_reports_what_it_did(demo, stub_bridge, capsys) -> None:
    """The guard reads the status only - an allowed drive is unchanged."""
    stub = stub_bridge(demo, _ACCEPTED)

    runpy.run_path(str(_EXAMPLES / demo.script), run_name="__main__")

    assert stub.drives == 1
    assert demo.claim in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.script)
def test_the_demo_documents_the_pre_approval_its_own_gate_needs(demo) -> None:
    """A script with no operator names the env var that stands in for one."""
    header = (_EXAMPLES / demo.script).read_text(encoding="utf-8")
    assert "STRANDS_ROS2_COMMAND_ALLOW" in header, f"{demo.script} leaves the gate unmentioned"
