"""Every mobile-base bridge that can start motion also exposes a stop tool.

All three transports publish a *latching* velocity command: with no ``duration``
a single ``cmd_vel`` message leaves the base moving until another command
arrives. :class:`~strands_robots.mesh.RosbridgeRobot`'s own drive tool
description states that contract - "without duration the last command latches
until stop" - and every bridge carries a public ``stop()`` that publishes a zero
``Twist``.

Two of the three exposed that halt as a named agent tool. The ROS 2 bridge did
not, so an agent handed ``RosBridgedRobot.tools`` could start motion and had no
tool to end it; its only halt was a ``drive`` at zero velocity, which is exactly
the idiom ``stop()`` exists to name. These tests pin the parity behaviourally on
every transport, and structurally require any future bridge owning a
``drive``/``stop`` pair to expose the halt too.

They also drive the remaining forwarding closures each ``tools`` property
builds. That is what kept the gap invisible: of the ten closures these three
bridges construct, four were never invoked through the tool, so nothing
observed which capabilities the surface actually reached.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

import strands_robots.mesh.ros_bridge as ros_mod
import strands_robots.mesh.rosbridge_robot as rbr_mod
import strands_robots.mesh.rtps_robot as rtps_mod
from strands_robots.mesh import RosBridgedRobot, RosbridgeRobot, RtpsRobot

ZERO_TWIST = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


class _Recorder:
    """Records the kwargs of each forwarded transport call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}


def _ros(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(ros_mod, "use_ros", rec)
    return RosBridgedRobot.from_ros(node_name="rover", cmd_vel_topic="/cmd_vel", odom_topic="/odom")


def _rosbridge(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(rbr_mod, "use_rosbridge", rec)
    return RosbridgeRobot(node_name="rover", cmd_vel_topic="/cmd_vel", odom_topic="/odom")


def _rtps(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(rtps_mod, "use_rtps", rec)
    return RtpsRobot.from_rtps(node_name="rover", cmd_vel_topic="/cmd_vel")


#: Every transport that turns a mobile base into strands agent tools.
BRIDGES = [
    pytest.param(_ros, id="ros2-rclpy"),
    pytest.param(_rosbridge, id="rosbridge-websocket"),
    pytest.param(_rtps, id="rtps-dds"),
]


def _tools(robot: Any) -> dict[str, Any]:
    return {t.tool_name: t for t in robot.tools}


# Cross-transport parity -------------------------------------------------------


@pytest.mark.parametrize("build", BRIDGES)
def test_a_bridge_exposing_a_drive_tool_also_exposes_a_stop_tool(build: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport that can start motion must offer the halt in the same set."""
    names = set(_tools(build(_Recorder(), monkeypatch)))
    assert "drive_rover" in names, f"no drive tool to pair a halt with: {sorted(names)}"
    assert "stop_rover" in names, f"drive without a halt: {sorted(names)}"


@pytest.mark.parametrize("build", BRIDGES)
def test_the_stop_tool_publishes_a_zero_velocity_twist(build: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The halt is a real zero command on the wire, not just a present name."""
    rec = _Recorder()
    stop_tool: Any = _tools(build(rec, monkeypatch))["stop_rover"]
    rec.calls.clear()
    result = stop_tool()
    assert result["status"] == "success"
    (call,) = rec.calls
    assert call["action"] == "publish"
    assert call["topic"] == "/cmd_vel"
    assert call["fields"] == ZERO_TWIST


@pytest.mark.parametrize("build", BRIDGES)
def test_a_drive_tool_with_no_duration_publishes_one_latching_command(
    build: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The premise: one message, non-zero, and no automatic stop after it.

    This is why the halt has to be reachable - the base keeps the last command
    until something else is published.
    """
    rec = _Recorder()
    drive_tool: Any = _tools(build(rec, monkeypatch))["drive_rover"]
    rec.calls.clear()
    drive_tool(linear=0.5)
    (call,) = rec.calls  # a single publish, with no trailing zero
    assert call["count"] == 1
    assert call["fields"]["linear"]["x"] == pytest.approx(0.5)


def test_the_ros2_stop_tool_is_present_without_any_optional_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The halt is unconditional, unlike get_scan and navigate."""
    robot = _ros(_Recorder(), monkeypatch)
    assert robot.scan_topic is None and robot.nav_action is None
    names = set(_tools(robot))
    assert "stop_rover" in names
    assert "get_scan_rover" not in names and "navigate_rover" not in names


def test_the_ros2_stop_tool_forwards_to_the_instance_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool is the documented ``stop()``, not a re-implementation."""
    rec = _Recorder()
    robot = _ros(rec, monkeypatch)
    stop_tool: Any = _tools(robot)["stop_rover"]

    rec.calls.clear()
    stop_tool()
    via_tool = list(rec.calls)
    rec.calls.clear()
    robot.stop()
    assert via_tool == rec.calls


# The forwarding closures that were never invoked through the tool -------------


def test_the_ros2_pose_and_scan_tools_forward_to_the_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder()
    monkeypatch.setattr(ros_mod, "use_ros", rec)
    robot = RosBridgedRobot("rover", "/cmd_vel", "/odom", scan_topic="/scan")
    tools = _tools(robot)

    pose_tool: Any = tools["get_pose_rover"]
    rec.calls.clear()
    pose_tool()
    assert rec.calls[0]["action"] == "echo" and rec.calls[0]["topic"] == "/odom"

    scan_tool: Any = tools["get_scan_rover"]
    rec.calls.clear()
    scan_tool()
    assert rec.calls[0]["action"] == "echo" and rec.calls[0]["topic"] == "/scan"


def test_the_rosbridge_stop_and_scan_tools_forward_to_the_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder()
    monkeypatch.setattr(rbr_mod, "use_rosbridge", rec)
    robot = RosbridgeRobot(node_name="rover", cmd_vel_topic="/cmd_vel", odom_topic="/odom", scan_topic="/scan")
    tools = _tools(robot)

    stop_tool: Any = tools["stop_rover"]
    rec.calls.clear()
    stop_tool()
    assert rec.calls[0]["fields"] == ZERO_TWIST

    scan_tool: Any = tools["get_scan_rover"]
    rec.calls.clear()
    scan_tool()
    assert rec.calls[0]["topic"] == "/scan"


# Structural: a future transport cannot ship a drive without a halt ------------

#: Classes that own a mobile base's drive/stop pair today. ``MobileBaseRobot``
#: owns the shared pair; ``RosbridgeRobot`` still defines its own until it moves
#: onto the base too.
EXPECTED_DRIVE_STOP_CLASSES = {"MobileBaseRobot", "RosbridgeRobot"}

#: Classes that inherit the pair instead of defining one. Their *absence* from
#: the scan is the consolidation, so it is asserted rather than tolerated: a
#: subclass that re-grows a hand-rolled drive would answer the fleet-standard
#: signature while opting out of every structural check in this file.
INHERIT_THE_DRIVE_STOP_PAIR = {"RosBridgedRobot", "RtpsRobot"}


def _mesh_dir() -> pathlib.Path:
    """Derive the scanned package from an imported symbol, not a path literal."""
    return pathlib.Path(inspect.getfile(ros_mod)).parent


def _declared_tool_name_prefix(decorator: ast.expr) -> str | None:
    """Return the literal head of a ``@tool(name=...)`` value, or None.

    The name is written as an f-string (``f"stop_{suffix}"``), so the leading
    literal is the first element of the ``JoinedStr``; a plain string name is
    read whole. Reading the keyword structurally keeps the survey independent
    of how the source spells its quotes.
    """
    if not isinstance(decorator, ast.Call):
        return None
    for kw in decorator.keywords:
        if kw.arg != "name":
            continue
        if isinstance(kw.value, ast.JoinedStr) and kw.value.values:
            head = kw.value.values[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                return head.value
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _stop_tool_survey(source: str) -> tuple[set[str], set[str]]:
    """Return (classes owning a drive/stop pair, those whose tools build a stop).

    A class qualifies when it defines public ``drive`` and ``stop`` methods and
    a ``tools`` property; it satisfies the rule when that property contains a
    ``@tool`` closure whose declared name starts with ``stop_``.
    """
    owning: set[str] = set()
    exposing: set[str] = set()
    for cls in (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)):
        methods = {m.name for m in cls.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)}
        if not {"drive", "stop", "tools"} <= methods:
            continue
        owning.add(cls.name)
        for member in cls.body:
            if not (isinstance(member, ast.FunctionDef) and member.name == "tools"):
                continue
            for closure in member.body:
                if not isinstance(closure, ast.FunctionDef):
                    continue
                for dec in closure.decorator_list:
                    prefix = _declared_tool_name_prefix(dec)
                    if prefix is not None and prefix.startswith("stop_"):
                        exposing.add(cls.name)
    return owning, exposing


def _survey_mesh() -> tuple[set[str], set[str]]:
    owning: set[str] = set()
    exposing: set[str] = set()
    for path in sorted(_mesh_dir().rglob("*.py")):
        own, exp = _stop_tool_survey(path.read_text())
        owning |= own
        exposing |= exp
    return owning, exposing


def test_every_bridge_owning_a_drive_stop_pair_builds_a_stop_tool() -> None:
    owning, exposing = _survey_mesh()
    assert owning - exposing == set(), f"drive/stop bridges with no stop tool: {sorted(owning - exposing)}"


def test_the_survey_finds_the_bridges_it_is_meant_to_cover() -> None:
    """Non-vacuity: an empty or mis-rooted scan must not read as compliant."""
    owning, _ = _survey_mesh()
    assert owning == EXPECTED_DRIVE_STOP_CLASSES, sorted(owning)
    redefined = INHERIT_THE_DRIVE_STOP_PAIR & owning
    assert not redefined, f"{sorted(redefined)} must inherit drive/stop from MobileBaseRobot, not redefine them"


def test_the_survey_flags_a_bridge_whose_tools_omit_the_stop() -> None:
    """Planted defect: the rule must fail for a drive/stop pair with no halt."""
    planted = """
class _PlantedBridge:
    def drive(self): ...
    def stop(self): ...
    def tools(self):
        @tool(name=f"drive_{suffix}", description="d")
        def drive(): ...
        return [drive]
"""
    owning, exposing = _stop_tool_survey(planted)
    assert owning == {"_PlantedBridge"}
    assert exposing == set()
