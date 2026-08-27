"""A :class:`RosbridgeRobot` command must reach the ``use_rosbridge`` operator gate.

The fourth of four. ``tests/mesh/test_ros_bridge_command_gate.py``,
``tests/mesh/test_rtps_robot_command_gate.py`` and
``tests/mesh/test_ackermann_command_gate.py`` each pin that their bridge's
commands reach the shared operator gate of
:mod:`strands_robots.tools._command_gate`; the rosbridge bridge had no such
suite, and it was the one class that could not carry an operator's decision at
all. Measured against the real ``use_rosbridge`` with an operator standing by to
approve, on the same blocklisted ``/cmd_vel`` surface::

    RtpsRobot.stop(tool_context=ctx)       -> success, 1 message published
    RosbridgeRobot.stop()                  -> error, "No tool_context available
                                              for operator approval", 0 published

``RosbridgeRobot.drive`` and ``.stop`` took no ``tool_context`` parameter and its
two commanding agent tools were not declared ``@tool(context=True)``, so there
was nothing for the gate to ask with: every command this bridge sent to a
blocklisted surface failed closed, and the only remedy it could offer was the
blanket ``BYPASS_TOOL_CONSENT`` or a standing pre-approval. The halt was the
worst case - the one control the ``tools`` contract guarantees ("a caller that
can start motion must be able to end it") was the one an operator present and
willing could not authorise.

``tests/mesh/test_rosbridge_robot.py`` cannot see any of that: it patches the
``use_rosbridge`` symbol at the mesh boundary, which is the boundary the gate
lives behind. These tests keep the real ``use_rosbridge`` and double the roslibpy
WebSocket client beneath it instead - the same boundary
``tests/tools/test_use_rosbridge.py`` doubles - so the gate and the transport
wiring both run unmodified while nothing reaches a real rosbridge server.

The structural half is derived over every drive-owning class rather than listed,
because "a command can carry an operator's decision" is a fleet property: a
fifth mobile base fails it on arrival instead of shipping unapprovable the way
this one did. ``tests/mesh/test_mobile_base.py`` grades only the classes that
inherit the shared base, which is exactly why it never covered this one.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.mesh as mesh_pkg
import strands_robots.tools.use_rosbridge as rb_mod
from strands_robots.mesh import RosbridgeRobot

# ``/turtle1/cmd_vel`` matches the ``/cmd_vel`` blocklist entry on the
# final-segment rule, so every command a robot on it sends is gated - which is
# what makes it the right topic to measure the gate with.
_BLOCKED_CMD_VEL = "/turtle1/cmd_vel"
# Not a blocklisted surface, so a command to it must never prompt. The control
# that keeps every assertion below about the gate rather than about publishing.
_UNBLOCKED_CMD_VEL = "/rover/waypoint_request"

_ZERO_TWIST = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}


class _FakeTopic:
    """Records advertise/publish/subscribe against a doubled rosbridge client."""

    def __init__(self, ros: _FakeRos, name: str, message_type: str) -> None:
        self.ros, self.name, self.message_type = ros, name, message_type
        ros.topics.append(self)

    def advertise(self) -> None:
        pass

    def unadvertise(self) -> None:
        pass

    def publish(self, message: dict[str, Any]) -> None:
        self.ros.published.append((self.name, dict(message)))

    def subscribe(self, callback: Any) -> None:
        for sample in self.ros.samples.get(self.name, []):
            callback(sample)

    def unsubscribe(self) -> None:
        pass


class _FakeRos:
    """A roslibpy client double that connects and records what was published."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host, self.port = host, port
        self.is_connected = False
        self.topics: list[_FakeTopic] = []
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.samples: dict[str, list[dict[str, Any]]] = {}
        _FakeRos.instances.append(self)

    instances: list[_FakeRos] = []

    def run(self, timeout: float | None = None) -> None:
        self.is_connected = True


def _install_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand the roslibpy client down for one test, with no ambient pre-approval.

    Both gate env vars short-circuit the operator prompt, so an ambient
    ``BYPASS_TOOL_CONSENT`` - common in agent and automation shells - would make
    every assertion here pass without the gate ever running.
    """
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_ROS2_COMMAND_ALLOW", raising=False)
    _FakeRos.instances = []
    module = types.ModuleType("roslibpy")
    module.Ros = _FakeRos  # type: ignore[attr-defined]
    module.Topic = _FakeTopic  # type: ignore[attr-defined]
    module.Message = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", module)
    monkeypatch.setattr(rb_mod._backend, "_available", None)
    monkeypatch.setattr(rb_mod._backend, "_connections", {})
    # publish sleeps for the advertise settle and for the inter-message period.
    monkeypatch.setattr(rb_mod.time, "sleep", lambda *_: None)


def _published() -> list[tuple[str, dict[str, Any]]]:
    return [entry for ros in _FakeRos.instances for entry in ros.published]


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _turtle(topic: str = _BLOCKED_CMD_VEL, **kwargs: Any) -> RosbridgeRobot:
    # ``odom_type`` is declared so a read needs no rosapi type lookup: the read
    # cases here are about whether the gate is consulted, not about resolution.
    kwargs.setdefault("odom_type", "nav_msgs/Odometry")
    return RosbridgeRobot(node_name="turtlesim", cmd_vel_topic=topic, odom_topic="/odom", **kwargs)


def _tool(robot: RosbridgeRobot, name: str) -> Any:
    return next(agent_tool for agent_tool in robot.tools if agent_tool.tool_name == name)


def _approving() -> MagicMock:
    context = MagicMock()
    context.interrupt.return_value = "y"
    return context


class TestRosbridgeCommandsReachTheGate:
    """The robot's agent tools must prompt the operator, not refuse outright."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_doubles(monkeypatch)

    def test_stop_tool_halts_the_robot_once_the_operator_approves(self) -> None:
        """The headline: the halt is gated, so it has to be reachable.

        A bridge that cannot forward an operator context turns ``stop`` into an
        unconditional refusal, removing the one control the ``tools`` contract
        guarantees for anything that can start motion.
        """
        context = _approving()
        result = _tool(_turtle(), "stop_turtlesim")(tool_context=context)
        assert context.interrupt.called, "the stop tool never reached the operator gate"
        assert context.interrupt.call_args[1]["reason"]["target"] == _BLOCKED_CMD_VEL
        assert context.interrupt.call_args[1]["reason"]["action"] == "publish"
        assert result["status"] == "success"
        assert _published() == [(_BLOCKED_CMD_VEL, _ZERO_TWIST)]

    def test_stop_tool_declined_publishes_nothing(self) -> None:
        context = MagicMock()
        context.interrupt.return_value = "n"
        result = _tool(_turtle(), "stop_turtlesim")(tool_context=context)
        assert result["status"] == "error"
        assert "declined by the operator" in _texts(result)
        assert _published() == []

    def test_drive_tool_prompts_the_operator_and_publishes_on_approval(self) -> None:
        context = _approving()
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=context)
        assert context.interrupt.called, "the drive tool never reached the operator gate"
        assert result["status"] == "success"
        assert _published() == [(_BLOCKED_CMD_VEL, {"linear": {"x": 1.0}, "angular": {"z": 0.0}})]

    def test_drive_tool_declined_publishes_nothing(self) -> None:
        context = MagicMock()
        context.interrupt.return_value = "n"
        result = _tool(_turtle(), "drive_turtlesim")(linear=1.0, tool_context=context)
        assert result["status"] == "error"
        assert "declined by the operator" in _texts(result)
        assert _published() == []

    def test_the_trailing_zero_of_a_timed_drive_reaches_the_same_operator(self) -> None:
        """A timed drive owns its own stop, and that stop is gated too.

        The trailing zero is a second publish to the same blocklisted surface, so
        it needs its own approval. One that could not reach the gate would leave
        the robot latched at the speed of a hold the operator did approve.
        """
        context = _approving()
        robot = _turtle(publish_rate=2.0)
        result = _tool(robot, "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert result["status"] == "success"
        # Two prompts: one for the hold, one for the trailing zero that ends it.
        assert context.interrupt.call_count == 2
        # round(1.0 * 2.0) held messages, then a single zero.
        published = _published()
        assert len(published) == 3
        assert published[0][1]["linear"]["x"] == 1.0
        assert published[-1] == (_BLOCKED_CMD_VEL, _ZERO_TWIST)

    def test_a_declined_hold_never_latches_a_velocity(self) -> None:
        """Refusing the hold refuses the motion, and its undo has nothing to undo."""
        context = MagicMock()
        context.interrupt.return_value = "n"
        result = _tool(_turtle(publish_rate=2.0), "drive_turtlesim")(linear=1.0, duration=1.0, tool_context=context)
        assert result["status"] == "error"
        assert _published() == []

    def test_without_an_operator_the_refusal_names_both_ways_through(self) -> None:
        """Fail-closed is correct with no operator - and must stay actionable.

        This is what every command on this bridge used to do even with an
        operator present, so the refusal is pinned as the no-context outcome
        rather than as the bridge's behaviour.
        """
        result = _turtle().stop()
        assert result["status"] == "error"
        text = _texts(result)
        assert "No tool_context available for operator approval" in text
        assert "STRANDS_ROS2_COMMAND_ALLOW" in text and "BYPASS_TOOL_CONSENT" in text
        assert _published() == []

    def test_carrying_a_context_does_not_start_gating_an_unblocked_surface(self) -> None:
        """Over-reach guard: the new parameter must not widen what is gated.

        Forwarding a context has to leave the gate's own rule alone - it is keyed
        on the surface, so a command to a topic the blocklist does not name still
        goes out with no operator asked.
        """
        context = _approving()
        result = _tool(_turtle(topic=_UNBLOCKED_CMD_VEL), "stop_turtlesim")(tool_context=context)
        assert not context.interrupt.called, "an unblocked surface reached the operator gate"
        assert result["status"] == "success"
        assert _published() == [(_UNBLOCKED_CMD_VEL, _ZERO_TWIST)]

    def test_a_pre_approved_surface_publishes_without_asking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_ROS2_COMMAND_ALLOW", "/cmd_vel")
        context = _approving()
        result = _turtle().stop(tool_context=context)
        assert result["status"] == "success"
        assert not context.interrupt.called, "a pre-approved surface still prompted"
        assert _published() == [(_BLOCKED_CMD_VEL, _ZERO_TWIST)]


class TestOnlyCommandsToABlockedSurfaceAreGated:
    """Controls, valid before and after the fix: reads and unblocked surfaces.

    None of these needs an operator context, which is what makes them controls:
    they hold identically on a bridge that cannot carry one, so a failure here
    means the gate's own rule moved rather than this bridge's wiring.
    """

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_doubles(monkeypatch)

    def test_a_command_to_an_unblocked_surface_needs_no_context_at_all(self) -> None:
        """The gate, not the transport, is what a missing context costs."""
        result = _turtle(topic=_UNBLOCKED_CMD_VEL).stop()
        assert result["status"] == "success"
        assert _published() == [(_UNBLOCKED_CMD_VEL, _ZERO_TWIST)]

    def test_reading_a_pose_never_prompts(self) -> None:
        robot = _turtle()
        result = robot.get_pose(timeout=0.01)
        assert result["status"] == "success"
        assert _published() == [], "a read published to the command surface"

    def test_the_read_tools_take_no_operator_context(self) -> None:
        """A read is never gated, so its tool must not ask for a decision."""
        read_tool = _tool(_turtle(), "get_pose_turtlesim")
        assert "tool_context" not in read_tool.tool_spec["inputSchema"]["json"].get("properties", {})


def _drive_owning_classes() -> list[type]:
    """Every shipped mobile base, derived from the public mesh modules.

    A class that owns ``drive`` and lives in a public module of
    ``strands_robots.mesh`` is a shipped platform bridge. Derived rather than
    listed so a fifth one is graded on arrival; the ``_``-prefixed modules are
    skipped because a shared base declares a contract but is not a platform
    anyone drives.
    """
    package_dir = pathlib.Path(mesh_pkg.__file__).parent
    owners: list[type] = []
    for path in sorted(package_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"strands_robots.mesh.{path.stem}")
        owners.extend(
            candidate
            for candidate in vars(module).values()
            if isinstance(candidate, type)
            and candidate.__module__ == module.__name__
            and callable(getattr(candidate, "drive", None))
        )
    return owners


_COMMANDING_METHODS = ("drive", "stop")


class TestEveryMobileBaseCanCarryAnOperatorDecision:
    """Fleet-wide: a command surface an operator cannot answer for is unusable.

    Both halves of the path are graded, because either one alone lets the
    reported defect back in: the method has to accept a context, and the agent
    tool that calls it has to be declared with one to receive it.
    """

    def test_the_survey_finds_the_bridges_it_is_meant_to_cover(self) -> None:
        """Non-vacuity: an empty or one-class survey would assert nothing."""
        names = {owner.__name__ for owner in _drive_owning_classes()}
        assert {"RosbridgeRobot", "RtpsRobot", "RosBridgedRobot", "AckermannRosRobot"} <= names, (
            f"the drive-owner survey found only {sorted(names)}"
        )

    @pytest.mark.parametrize("method", _COMMANDING_METHODS)
    def test_every_commanding_method_accepts_an_operator_context(self, method: str) -> None:
        missing = [
            owner.__name__
            for owner in _drive_owning_classes()
            if "tool_context" not in inspect.signature(getattr(owner, method)).parameters
        ]
        assert not missing, (
            f"{missing} cannot carry an operator decision into {method}(), so every command they send "
            "to a blocklisted surface fails closed even with an operator standing by"
        )

    def test_every_commanding_agent_tool_is_declared_with_the_context(self) -> None:
        """The tool has to ask for the context, or the method never receives one.

        ``tool_context`` is deliberately absent from a tool's input schema - the
        runtime injects it rather than the model supplying it - so the wiring is
        read from the source of the ``tools`` property, the same way the
        per-bridge suites read their own. Those grade one bridge each; this
        grades whichever set of bridges ships.
        """
        offenders: list[str] = []
        for owner in _drive_owning_classes():
            for func in _decorated_tools(_tools_property_ast(owner)):
                if not _forwards_a_commanding_method(func):
                    continue
                decorator = next(dec for dec in func.decorator_list if isinstance(dec, ast.Call))
                context_kwarg = next((kw for kw in decorator.keywords if kw.arg == "context"), None)
                enabled = context_kwarg is not None and getattr(context_kwarg.value, "value", None) is True
                params = [arg.arg for arg in func.args.args] + [arg.arg for arg in func.args.kwonlyargs]
                forwards = any(
                    kw.arg == "tool_context"
                    for node in ast.walk(func)
                    if isinstance(node, ast.Call)
                    for kw in node.keywords
                )
                if not (enabled and "tool_context" in params and forwards):
                    offenders.append(f"{owner.__name__}.{func.name}")
        assert not offenders, (
            f"{offenders} do not declare @tool(context=True), receive the injected context and forward it, "
            "so an operator's decision can never reach the command they carry"
        )


def _tools_property_ast(owner: type) -> ast.Module:
    """Parse the module that defines ``owner``'s ``tools`` property.

    A bridge inherits ``tools`` from the shared mobile base or declares its own,
    so the file to read is the one the property is defined in rather than the one
    the class is named in.
    """
    source_file = inspect.getsourcefile(owner.tools.fget)  # type: ignore[attr-defined]
    assert source_file is not None, f"cannot locate the source of {owner.__name__}.tools"
    return ast.parse(pathlib.Path(source_file).read_text(encoding="utf-8"))


def _decorated_tools(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "tool" for dec in node.decorator_list)
    ]


def _forwards_a_commanding_method(func: ast.FunctionDef) -> bool:
    """Whether this tool closure calls one of the commanding methods."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _COMMANDING_METHODS
        for node in ast.walk(func)
    )
