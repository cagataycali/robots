# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A demo never reports a drive the command gate refused.

``cmd_vel`` is a gated command surface, and a script has no operator to approve
one, so the gate answers the two shapes of caller differently and both were
reported as a drive that happened:

* a programmatic caller gets ``{"status": "error", ...}`` back from ``drive``,
  naming the two remedies (``STRANDS_ROS2_COMMAND_ALLOW`` /
  ``BYPASS_TOOL_CONSENT``). ``examples/ros2/rtps_turtle_demo.py`` printed ``done
  - the turtle should have moved`` against a real ``turtlesim`` that had not
  moved, and ``examples/ros2/turtlebot_demo.py`` printed an ``after:`` pose
  identical to its ``before:`` one;
* an agent run is PAUSED instead of finished - ``result.stop_reason ==
  "interrupt"``, the question in ``result.interrupts`` carrying the resume form
  and the pre-approval line - and both agent demos printed it under ``Agent
  completed``.

Graded here on one rule per caller shape, one row per demo: a refused command
raises and carries the remedy, no claim is printed, and an accepted one is
unchanged. No DDS, no ROS 2, no model call and no turtle - the bridge class each
script imports is stubbed at its own seam, and so is ``strands.Agent``.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import strands

import strands_robots.mesh as mesh

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

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

#: The line the gate puts in a paused run's question - the remedy a demo that
#: stops on one has to carry, since the question is the only thing that names it.
_REMEDY = (
    "This call is paused, not done. For a script with no operator, pre-approve this command "
    "instead with STRANDS_ROS2_COMMAND_ALLOW=/turtle1/cmd_vel."
)


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

    @property
    def tools(self) -> list[Any]:
        return []


@dataclass
class _StubAgent:
    """An agent that hands back one scripted result, with no model behind it."""

    result: Any
    calls: list[Any] = field(default_factory=list)

    def __call__(self, prompt: Any = None, **_kwargs: Any) -> Any:
        self.calls.append(prompt)
        return self.result


@dataclass(frozen=True)
class _Result:
    """The two fields of an ``AgentResult`` a demo reads to tell paused from done."""

    stop_reason: str
    interrupts: tuple[Any, ...] = ()

    def __str__(self) -> str:
        return "the rover drove two legs"


@dataclass(frozen=True)
class _Demo:
    """One script, the class it imports, and the claim its output must not carry."""

    script: str
    factory: str
    claim: str


DEMOS = [
    _Demo("ros2/rtps_turtle_demo.py", "RtpsRobot", "done - the turtle should have moved"),
    _Demo("ros2/turtlebot_demo.py", "RosBridgedRobot", "after: "),
]

AGENT_DEMOS = [
    _Demo("rosbridge/curiosity_agent.py", "RosbridgeRobot", "Agent completed"),
    _Demo("ros2/deepracer_agent.py", "AckermannRosRobot", "Agent completed"),
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
                "from_curiosity": classmethod(lambda _cls, **_k: stub),
                "from_deepracer": classmethod(lambda _cls, **_k: stub),
            },
        )
        monkeypatch.setattr(mesh, demo.factory, factory)
        return stub

    return install


@pytest.fixture
def stub_agent(monkeypatch: pytest.MonkeyPatch, stub_bridge):
    """Install a scripted agent (and bridge) for a demo that hands over its tools."""

    def install(demo: _Demo, result: Any) -> _StubAgent:
        stub_bridge(demo, _ACCEPTED)
        agent = _StubAgent(result)
        monkeypatch.setattr(strands, "Agent", lambda *_a, **_k: agent)
        return agent

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


@pytest.mark.parametrize("demo", AGENT_DEMOS, ids=lambda d: d.script)
def test_a_paused_run_is_not_reported_as_a_completed_one(demo, stub_agent, capsys) -> None:
    """A run the gate paused stops the script, carrying the question's own remedy."""
    question = {"name": "use_rosbridge-command-approval", "reason": {"how_to_answer": _REMEDY}}
    agent = stub_agent(demo, _Result("interrupt", (question,)))

    with pytest.raises(RuntimeError, match="STRANDS_ROS2_COMMAND_ALLOW"):
        runpy.run_path(str(_EXAMPLES / demo.script), run_name="__main__")

    assert len(agent.calls) == 1
    assert demo.claim not in capsys.readouterr().out


@pytest.mark.parametrize("demo", AGENT_DEMOS, ids=lambda d: d.script)
def test_a_finished_run_still_reports_what_the_agent_did(demo, stub_agent, capsys) -> None:
    """The guard reads ``stop_reason`` only - a run that finished is unchanged."""
    agent = stub_agent(demo, _Result("end_turn"))

    runpy.run_path(str(_EXAMPLES / demo.script), run_name="__main__")

    assert len(agent.calls) == 1
    assert demo.claim in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS + AGENT_DEMOS, ids=lambda d: d.script)
def test_the_demo_documents_the_pre_approval_its_own_gate_needs(demo) -> None:
    """A script with no operator names the env var that stands in for one."""
    header = (_EXAMPLES / demo.script).read_text(encoding="utf-8")
    assert "STRANDS_ROS2_COMMAND_ALLOW" in header, f"{demo.script} leaves the gate unmentioned"
