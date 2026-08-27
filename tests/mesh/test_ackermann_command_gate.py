"""The Ackermann bridge's commands must reach the ``use_ros`` operator gate.

Every :class:`AckermannRosRobot` command forwards to ``use_ros``, whose command
gate refuses a safety-critical surface when no operator context is reachable.
The surfaces this bridge drives are the DeepRacer's servo topic and the two mode
services that arm the vehicle, and all three are blocklisted - so a bridge that
forwards no context turns its whole command surface, ``stop`` included, into a
per-call refusal, while a bridge whose surfaces are *not* on the blocklist sends
throttle commands with no prompt, no allowlist check and no audit row.
``tests/mesh/test_ackermann_robot.py`` can see neither: it patches the
``use_ros`` symbol at the boundary the gate lives behind.

These tests keep the real ``use_ros`` and substitute the rclpy transport
instead (the same boundary ``tests/tools/test_use_ros.py`` doubles), so the gate
and the bridge wiring under test both run unmodified while no message can reach
a real DDS graph. They are the Ackermann half of
``tests/mesh/test_ros_bridge_command_gate.py``;
:class:`TestEveryCommandingMeshBridgeHasAGateSuite` derives the set of bridges
owing such a suite from the tree, so a fifth one is graded on arrival rather
than shipping ungated the way this one did.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools._command_gate as gate_mod
import strands_robots.tools.use_ros as ros_mod
from strands_robots.mesh import AckermannRosRobot

_MESH_DIR = Path(ros_mod.__file__).parent.parent / "mesh"
_BRIDGE_SOURCE = _MESH_DIR / "ackermann_robot.py"
_TESTS_DIR = Path(__file__).parent

_COMMAND_METHODS = frozenset({"drive", "stop", "enable", "_publish_servo"})
_COMMAND_ACTIONS = frozenset({"publish", "service_call", "action_send_goal"})

#: Every mesh bridge that sends a command through ``use_ros``, and the suite that
#: grades its gate wiring. Discovered modules are compared against these keys, so
#: a new commanding bridge fails until it is triaged into a gate suite of its own.
_GATE_SUITES: dict[str, str] = {
    "ros_bridge.py": "test_ros_bridge_command_gate.py",
    "ackermann_robot.py": "test_ackermann_command_gate.py",
}


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _car() -> AckermannRosRobot:
    """A stock DeepRacer bridge: every surface it commands is blocklisted.

    ``/webserver_pkg/manual_drive`` matches ``/manual_drive`` on the
    final-segment rule, and the handshake's ``/ctrl_pkg/vehicle_state`` and
    ``/ctrl_pkg/enable_state`` match ``/vehicle_state`` and ``/enable_state`` the
    same way - which is what makes the shipped wiring the right instance to test
    the gate with.

    Only ``scan_type`` departs from the stock wiring: an omitted type is resolved
    from the live graph, which is the one path in this class that needs a real
    rclpy, and it is a read the gate never sees.
    """
    return AckermannRosRobot.from_deepracer(node_name="deepracer", scan_type="sensor_msgs/msg/LaserScan")


def _tool(robot: AckermannRosRobot, name: str) -> Any:
    return next(t for t in robot.tools if t.tool_name == name)


@pytest.fixture(scope="module")
def bridge_ast() -> ast.Module:
    """Parse the bridge source once for the structural guards below."""
    return ast.parse(_BRIDGE_SOURCE.read_text(encoding="utf-8"))


class TestBridgeCommandsReachTheGate:
    """The bridge's agent tools must prompt the operator, not refuse outright."""

    published: list[tuple[Any, ...]]
    services: list[tuple[Any, ...]]
    echoed: list[tuple[Any, ...]]

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both env vars short-circuit the gate, so an ambient BYPASS_TOOL_CONSENT
        # (common in agent/automation shells) would make these assertions pass
        # without the gate ever running. Cases that need them opt in explicitly.
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv(gate_mod.COMMAND_ALLOW_ENV, raising=False)
        monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
        self.published, self.services, self.echoed = [], [], []

        def _record(sink: list[tuple[Any, ...]], outcome: Any) -> Any:
            def _fake(*args: Any) -> Any:
                sink.append(args)
                return outcome

            return _fake

        monkeypatch.setattr(ros_mod, "_publish", _record(self.published, None))
        monkeypatch.setattr(ros_mod, "_service_call", _record(self.services, {"success": True}))
        monkeypatch.setattr(ros_mod, "_echo", _record(self.echoed, [{"ranges": [1.0, 2.0]}]))

    def test_drive_tool_prompts_for_every_surface_and_commands_on_approval(self) -> None:
        """The handshake services and the servo publish are each gated."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_car(), "drive_deepracer")(linear=0.5, tool_context=ctx)
        assert ctx.interrupt.called, "the drive tool never reached the operator gate"
        prompted = [call[1]["reason"]["target"] for call in ctx.interrupt.call_args_list]
        assert prompted == [
            "/ctrl_pkg/vehicle_state",
            "/ctrl_pkg/enable_state",
            "/webserver_pkg/manual_drive",
        ]
        assert result["status"] == "success"
        assert len(self.services) == 2
        assert len(self.published) == 1

    def test_drive_tool_declined_at_the_servo_topic_publishes_nothing(self) -> None:
        ctx = MagicMock()
        ctx.interrupt.side_effect = ["y", "y", "n"]
        result = _tool(_car(), "drive_deepracer")(linear=0.5, tool_context=ctx)
        assert result["status"] == "error"
        assert "declined" in _texts(result)
        assert self.published == []

    def test_drive_tool_declined_at_the_handshake_never_reaches_the_servo_topic(self) -> None:
        """A refused arming service aborts the drive, as a failed one does."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "n"
        result = _tool(_car(), "drive_deepracer")(linear=0.5, tool_context=ctx)
        assert result["status"] == "error"
        assert ctx.interrupt.call_count == 1  # stops at the first refusal
        assert self.services == []
        assert self.published == []

    def test_stop_tool_halts_the_car_once_the_operator_approves(self) -> None:
        """The halt is gated like any other servo publish, and must be reachable.

        A bridge that cannot forward an operator context makes ``stop`` an
        unconditional refusal - removing the one control that makes an
        ungated-looking latching command safe to issue at all.
        """
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_car(), "stop_deepracer")(tool_context=ctx)
        assert ctx.interrupt.called, "the stop tool never reached the operator gate"
        assert result["status"] == "success"
        topic, _type, fields = self.published[0][:3]
        assert topic == "/webserver_pkg/manual_drive"
        assert fields == {"angle": 0.0, "throttle": 0.0}

    def test_read_tool_needs_no_context_because_reads_are_never_gated(self) -> None:
        result = _tool(_car(), "get_scan_deepracer")()
        assert result["status"] == "success"
        assert len(self.echoed) == 1

    def test_programmatic_command_without_a_context_names_the_headless_variables(self) -> None:
        """The documented decision: no operator context means the gate refuses."""
        result = _car().drive(linear=0.5)
        assert result["status"] == "error"
        assert gate_mod.COMMAND_ALLOW_ENV in _texts(result)
        assert "BYPASS_TOOL_CONSENT" in _texts(result)
        assert self.services == []
        assert self.published == []

    @pytest.mark.parametrize(
        "allow",
        [
            "/manual_drive,/vehicle_state,/enable_state",
            "/webserver_pkg/manual_drive,/ctrl_pkg/vehicle_state,/ctrl_pkg/enable_state",
        ],
    )
    def test_programmatic_drive_and_stop_run_under_the_headless_allowlist(
        self, monkeypatch: pytest.MonkeyPatch, allow: str
    ) -> None:
        """The pre-approval the docs and the example hand an operator works.

        Both spellings are pinned because the bare form is what they publish: it
        is the namespaced surfaces the bridge actually sends to that have to be
        covered by it.
        """
        monkeypatch.setenv(gate_mod.COMMAND_ALLOW_ENV, allow)
        car = _car()
        assert car.drive(linear=0.5)["status"] == "success"
        assert car.stop()["status"] == "success"
        assert len(self.published) == 2

    def test_a_timed_drive_gates_the_command_and_its_trailing_halt(self) -> None:
        """The halt publish is a command too, so it carries the same context."""
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        result = _tool(_car(), "drive_deepracer")(linear=0.5, duration=0.1, tool_context=ctx)
        assert result["status"] == "success"
        assert len(self.published) == 2  # the hold, then the trailing zero
        assert self.published[1][2] == {"angle": 0.0, "throttle": 0.0}


class TestCommandToolsDeclareTheOperatorContext:
    """Structural guards, so a command tool added later cannot ship contextless.

    The behavioural tests above cover the two command tools that exist today. A
    third one added without ``context=True`` would fail closed on every call with
    no test noticing, so the wiring itself is pinned here.
    """

    @staticmethod
    def _decorated_tools(tree: ast.Module) -> list[ast.FunctionDef]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "tool" for dec in node.decorator_list
            )
        ]

    @staticmethod
    def _forwards_a_command_method(func: ast.FunctionDef) -> bool:
        return any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _COMMAND_METHODS
            for node in ast.walk(func)
        )

    def test_every_command_tool_is_context_enabled_and_forwards_it(self, bridge_ast: ast.Module) -> None:
        command_tools = [f for f in self._decorated_tools(bridge_ast) if self._forwards_a_command_method(f)]
        assert {f.name for f in command_tools} == {"drive", "stop"}
        for func in command_tools:
            decorator = next(d for d in func.decorator_list if isinstance(d, ast.Call))
            context_kwarg = next((kw for kw in decorator.keywords if kw.arg == "context"), None)
            assert context_kwarg is not None and getattr(context_kwarg.value, "value", None) is True, (
                f"bridge command tool {func.name!r} is not declared @tool(context=True), "
                "so it can never reach the use_ros operator gate"
            )
            params = [a.arg for a in func.args.args] + [a.arg for a in func.args.kwonlyargs]
            assert "tool_context" in params, f"{func.name!r} does not receive the injected operator context"
            forwarded = [
                kw.arg
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
                for kw in node.keywords
                if kw.arg == "tool_context"
            ]
            assert forwarded, f"{func.name!r} receives the operator context but does not forward it"

    def test_read_only_tools_stay_contextless(self, bridge_ast: ast.Module) -> None:
        """``echo`` is never gated, so a read tool must not ask for an operator."""
        read_tools = [f for f in self._decorated_tools(bridge_ast) if not self._forwards_a_command_method(f)]
        assert {f.name for f in read_tools} == {"get_scan"}
        for func in read_tools:
            params = [a.arg for a in func.args.args]
            assert "tool_context" not in params, f"read-only tool {func.name!r} should not require an operator context"

    def test_every_bridge_command_call_forwards_the_context_to_use_ros(self, bridge_ast: ast.Module) -> None:
        """A bridge method that carries a command must not drop the context.

        This is the failure this suite exists for: the gate lives inside
        ``use_ros``, so a call site that omits ``tool_context`` silently becomes
        a fail-closed refusal for its whole method.
        """
        checked = 0
        for node in ast.walk(bridge_ast):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "use_ros"):
                continue
            action = next((kw.value for kw in node.keywords if kw.arg == "action"), None)
            if not (isinstance(action, ast.Constant) and action.value in _COMMAND_ACTIONS):
                continue
            checked += 1
            assert any(kw.arg == "tool_context" for kw in node.keywords), (
                f"use_ros(action={action.value!r}) at ackermann_robot.py:{node.lineno} does not forward tool_context"
            )
        assert checked == 2, f"expected the servo publish and the handshake service_call, found {checked}"

    def test_command_tools_do_not_expose_the_context_in_their_input_schema(self) -> None:
        """The operator context is injected, never a parameter the model fills."""
        for name in ("drive_deepracer", "stop_deepracer"):
            spec = _tool(_car(), name).tool_spec
            assert "tool_context" not in spec["inputSchema"]["json"].get("properties", {})


class TestTheCommandedSurfacesAreBlocklisted:
    """The gate only fires on a blocklisted surface, so coverage is half the fix.

    A bridge can forward a context perfectly and still be ungated if none of the
    surfaces it drives is on the blocklist - which is how this one shipped: the
    stock DeepRacer wiring matched no entry, so every publish and handshake call
    proceeded with no prompt and no audit row.
    """

    def test_every_surface_the_shipped_wiring_commands_is_gated(self) -> None:
        car = _car()
        commanded = [car.servo_topic] + [item["service"] for item in car.init_services]
        assert commanded, "the DeepRacer wiring commands nothing - the premise is gone"
        for name in commanded:
            assert gate_mod.command_block_message("publish", name) is not None, (
                f"{name} is a DeepRacer motion/arming surface but no blocklist entry covers it"
            )

    def test_the_bare_entries_are_what_covers_the_namespaced_spellings(self) -> None:
        """The entries are spelled bare, so the base-name rule is load-bearing."""
        for entry in ("/manual_drive", "/vehicle_state", "/enable_state"):
            assert entry in gate_mod.COMMAND_BLOCKLIST


class TestEveryCommandingMeshBridgeHasAGateSuite:
    """A mesh bridge that commands through ``use_ros`` owes a gate suite.

    Derived from the tree rather than from a list, because the defect this file
    fixes was a *new* bridge shipping with neither blocklist coverage nor context
    threading while the existing bridge's suite stayed green. A fifth bridge
    fails here until it is triaged, which is the only moment the question gets
    asked cheaply.
    """

    @staticmethod
    def _modules_sending_commands() -> set[str]:
        found: set[str] = set()
        for path in sorted(_MESH_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "use_ros"):
                    continue
                action = next((kw.value for kw in node.keywords if kw.arg == "action"), None)
                if isinstance(action, ast.Constant) and action.value in _COMMAND_ACTIONS:
                    found.add(path.name)
        return found

    def test_the_inventory_names_every_bridge_that_sends_a_command(self) -> None:
        assert self._modules_sending_commands() == set(_GATE_SUITES), (
            "a mesh module sends commands through use_ros without being triaged into "
            "_GATE_SUITES; add it with the suite that grades its gate wiring"
        )

    @pytest.mark.parametrize("module, suite", sorted(_GATE_SUITES.items()))
    def test_each_bridge_gate_suite_exists(self, module: str, suite: str) -> None:
        assert (_TESTS_DIR / suite).is_file(), f"{module} has no gate suite at tests/mesh/{suite}"
