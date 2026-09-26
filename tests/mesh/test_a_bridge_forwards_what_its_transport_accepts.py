"""Every mesh bridge forwards a call its transport would accept.

A bridge owns no transport state: :class:`~strands_robots.mesh.RosBridgedRobot`,
:class:`~strands_robots.mesh.RosbridgeRobot`,
:class:`~strands_robots.mesh.ackermann_robot.AckermannRosRobot` and
:class:`~strands_robots.mesh.RtpsRobot` each resolve ``ros_action``,
``rosbridge_action`` or ``rtps_action`` through their own module and hand it the
whole command. Every test of that forwarding replaces the symbol with a recorder,
and a recorder that takes ``**kwargs`` accepts calls the transport would refuse:
dropping ``gate=never_gated`` from ``_RtpsTransport.echo`` - an argument all
three transports require and none defaults - left 5,265 mesh tests green, and the
first caller to reach the real transport a ``TypeError``.

Two halves, both over an inventory read from the tree rather than listed here, so
a fifth bridge is graded on arrival:

- the forwards, each bound against the signature of the transport it goes to;
- the shared stand-in those tests install, which binds the same way, so a
  forward built at runtime is refused there too.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import strands_robots.mesh as mesh_pkg
from strands_robots.ros import ros_action
from strands_robots.rosbridge import rosbridge_action
from strands_robots.rtps.participant import rtps_action
from tests.mesh._transport_stand_in import OFF_THE_WIRE, Transport, stands_in_for

#: The transport callables a bridge can forward to, by the name it imports.
_TRANSPORTS: dict[str, Any] = {
    "ros_action": ros_action,
    "rosbridge_action": rosbridge_action,
    "rtps_action": rtps_action,
}

_MESH_DIR = Path(mesh_pkg.__file__).resolve().parent

#: The forwards in the tree when this was written: publish/echo/service_call/
#: action_send_goal on the ROS 2 bridge, publish plus two echoes on rosbridge,
#: publish/echo/advertise on RTPS, publish/echo/service_call on the Ackermann car.
_KNOWN_FORWARDS = 13


@dataclass(frozen=True)
class _Forward:
    """One call site handing a command to a transport."""

    module: str
    symbol: str
    caller: str
    line: int
    positional: int
    keywords: tuple[str, ...]
    splatted: bool

    def __str__(self) -> str:
        return f"{self.module}:{self.line} {self.caller} -> {self.symbol}"


def _forward_sites() -> list[_Forward]:
    """Every call to a transport callable inside the mesh bridge modules.

    Returns:
        One :class:`_Forward` per call site, in file and line order.
    """
    found: list[_Forward] = []
    for path in sorted(_MESH_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
        } & set(_TRANSPORTS)
        if not imported:
            continue
        callers = {
            child: node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            for child in ast.walk(node)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported):
                continue
            found.append(
                _Forward(
                    module=path.name,
                    symbol=node.func.id,
                    caller=callers.get(node, "<module>"),
                    line=node.lineno,
                    positional=len([arg for arg in node.args if not isinstance(arg, ast.Starred)]),
                    keywords=tuple(kw.arg for kw in node.keywords if kw.arg is not None),
                    splatted=any(kw.arg is None for kw in node.keywords)
                    or any(isinstance(arg, ast.Starred) for arg in node.args),
                )
            )
    return found


_FORWARDS = _forward_sites()


def test_the_scan_finds_the_forwards_it_is_meant_to_cover() -> None:
    """Non-vacuity: an empty or mis-rooted scan must not read as compliant."""
    assert len(_FORWARDS) >= _KNOWN_FORWARDS, f"the scan lost a forward: {[str(f) for f in _FORWARDS]}"
    assert {forward.module for forward in _FORWARDS} >= {
        "ros_bridge.py",
        "rosbridge_robot.py",
        "rtps_robot.py",
        "ackermann_robot.py",
    }, f"the scan lost a bridge: {sorted({forward.module for forward in _FORWARDS})}"


@pytest.mark.parametrize("forward", _FORWARDS, ids=str)
def test_a_forward_names_arguments_its_transport_has_parameters_for(forward: _Forward) -> None:
    """The rule: the transport would accept this call, gate included.

    ``bind`` answers both halves at once - an argument the transport has no
    parameter for, and a required one the forward leaves out.
    """
    signature = inspect.signature(_TRANSPORTS[forward.symbol])
    arguments: dict[str, Any] = dict.fromkeys(forward.keywords)
    bind = signature.bind_partial if forward.splatted else signature.bind
    try:
        bind(*[None] * forward.positional, **arguments)
    except TypeError as refused:
        pytest.fail(f"{forward} forwards a call the transport refuses: {refused}")


class TestTheStandInHasTheShapeOfTheTransport:
    """A forward built at runtime is refused by the recorder the tests install."""

    def test_an_argument_the_transport_has_no_parameter_for_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stand_in = stands_in_for(monkeypatch, mesh_pkg.rtps_robot, "rtps_action")
        with pytest.raises(TypeError):
            stand_in(action="publish", topic="/cmd_vel", qos=1, gate=lambda _target: None)
        assert stand_in.calls == [], "a call the transport would refuse is not a call that happened"

    def test_a_forward_without_the_operator_gate_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No transport defaults the gate, so a forward that omits it raises."""
        stand_in = stands_in_for(monkeypatch, mesh_pkg.ros_bridge, "ros_action")
        with pytest.raises(TypeError):
            stand_in(action="echo", topic="/odom", count=1, timeout=5.0)

    def test_a_second_stand_in_keeps_the_real_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A test that patches one symbol per probe patches over a stand-in.

        Taking the previous one's own ``(*args, **kwargs)`` would make every
        later probe accept anything, and record it under ``kwargs``.
        """
        first = stands_in_for(monkeypatch, mesh_pkg.rtps_robot, "rtps_action")
        second = stands_in_for(monkeypatch, mesh_pkg.rtps_robot, "rtps_action")
        assert second.target is first.target is rtps_action
        with pytest.raises(TypeError):
            second(action="advertise", topic="/cmd_vel", qos=1, gate=lambda _target: None)

    def test_the_wire_view_drops_the_operator_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A gate is a fresh closure per call, and nothing in it reaches the robot."""
        stand_in = stands_in_for(monkeypatch, mesh_pkg.rtps_robot, "rtps_action")
        robot = mesh_pkg.RtpsRobot("rover", "/cmd_vel")
        assert robot.drive(linear=0.5)["status"] == "success"
        (recorded,) = stand_in.calls
        assert OFF_THE_WIRE & set(recorded), "the gate is part of what the bridge forwards"
        assert not OFF_THE_WIRE & set(stand_in.on_the_wire[0])

    def test_a_scripted_answer_is_returned_before_the_default_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bridge's failure paths are driven by the transport's answers."""
        stand_in = Transport(rtps_action, text="ok")
        stand_in.responses = [{"status": "error", "content": [{"text": "backend down"}]}]
        failed = stand_in(action="advertise", topic="/cmd_vel", type="T", gate=lambda _target: None)
        assert failed["status"] == "error"
        assert stand_in(action="advertise", topic="/cmd_vel", type="T", gate=lambda _target: None) == {
            "status": "success",
            "content": [{"text": "ok"}],
        }
