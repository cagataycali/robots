"""``serial_tool`` stops for an operator before a write reaches the servo bus.

Finding F-009 (CWE-862): the four write actions - ``send``, ``send_read``,
``feetech_position``, ``feetech_velocity`` - reached ``ser.write`` with no gate
of their own. The dashboard's ``MotionInterruptHook`` lists them, but a hook is
something an ``Agent`` has to be built with, and every canonical
``Agent(tools=robot.tools)`` build has none, so the default wiring moved servos
unasked.

These tests grade the observable: whether ``serial.Serial`` was constructed and
whether anything was written to it. The gate runs before the port is opened, so
a denied call is measured as "no port, no bytes", not as which branch was taken.
The audit row is graded beside the other gates in
``tests/tools/test_hitl_operator_response_audit.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.serial_tool as serial_mod

PORT = "/dev/ttyFAKE0"


class _FakeSerial:
    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        pass


def _ctx(response: object) -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[_FakeSerial]:
    """Every port the tool opens, with a clean gate environment and a private audit dir."""
    ports: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        fake = _FakeSerial(port, baudrate, timeout)
        ports.append(fake)
        return fake

    monkeypatch.setattr(serial_mod.serial, "Serial", _ctor)
    monkeypatch.setattr(serial_mod.time, "sleep", lambda *_: None)
    for name in ("BYPASS_TOOL_CONSENT", serial_mod.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    return ports


WRITES: dict[str, dict[str, Any]] = {
    "send": {"data": "go"},
    "send_read": {"hex_data": "FF FF 01 02 01 FB"},
    "feetech_position": {"motor_id": 1, "position": 2048},
    "feetech_velocity": {"motor_id": 1, "velocity": 100},
}


class TestWithoutApprovalNothingTouchesTheBus:
    @pytest.mark.parametrize("action", sorted(WRITES))
    def test_a_declined_write_opens_no_port(self, opened: list[_FakeSerial], action: str) -> None:
        res = serial_mod.serial_tool(
            action=action, port=PORT, tool_context=_ctx("n - arm is in a person's hand"), **WRITES[action]
        )

        assert res["status"] == "error", res
        assert "declined by the operator" in res["content"][0]["text"]
        assert opened == [], f"the port was opened after the operator said no: {opened}"

    @pytest.mark.parametrize("action", sorted(WRITES))
    def test_no_tool_context_refuses_and_names_both_escape_hatches(
        self, opened: list[_FakeSerial], action: str
    ) -> None:
        """Headless with nothing pre-approved fails closed - the default agent wiring."""
        res = serial_mod.serial_tool(action=action, port=PORT, **WRITES[action])

        assert res["status"] == "error", res
        text = res["content"][0]["text"]
        assert serial_mod.COMMAND_ALLOW_ENV in text and "BYPASS_TOOL_CONSENT" in text
        assert opened == []

    def test_the_operator_is_shown_the_port_and_the_motion(self, opened: list[_FakeSerial]) -> None:
        ctx = _ctx("y")
        serial_mod.serial_tool(action="feetech_position", port=PORT, motor_id=3, position=1000, tool_context=ctx)

        assert ctx.interrupt.call_args.args[0] == "serial_tool-command-approval"
        reason = ctx.interrupt.call_args.kwargs["reason"]
        assert reason["action"] == "feetech_position"
        assert reason["target"] == PORT
        assert "motor_id=3" in reason["warning"] and "position=1000" in reason["warning"]

    def test_an_invalid_option_is_refused_before_the_operator_is_asked(self, opened: list[_FakeSerial]) -> None:
        """A human is not asked to approve a command the tool would refuse anyway."""
        ctx = _ctx("y")
        res = serial_mod.serial_tool(action="feetech_position", port=PORT, motor_id=1, position=70000, tool_context=ctx)

        assert res["status"] == "error"
        ctx.interrupt.assert_not_called()
        assert opened == []


class TestAnApprovalWritesOnce:
    @pytest.mark.parametrize("action", sorted(WRITES))
    def test_an_approved_write_reaches_the_port_once(self, opened: list[_FakeSerial], action: str) -> None:
        ctx = _ctx("y")
        res = serial_mod.serial_tool(action=action, port=PORT, tool_context=ctx, **WRITES[action])

        assert res["status"] == "success", res
        assert len(opened) == 1 and len(opened[0].writes) == 1
        ctx.interrupt.assert_called_once()


class TestPreApprovalIsSilent:
    def test_a_named_action_is_written_without_an_interrupt(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(serial_mod.COMMAND_ALLOW_ENV, "feetech_position, feetech_velocity")
        ctx = _ctx("n")

        res = serial_mod.serial_tool(action="feetech_velocity", port=PORT, motor_id=1, velocity=50, tool_context=ctx)

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()

    def test_the_allowlist_is_exact_so_a_sibling_action_still_asks(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(serial_mod.COMMAND_ALLOW_ENV, "feetech_position")

        res = serial_mod.serial_tool(action="send", port=PORT, data="x", tool_context=_ctx("n"))

        assert res["status"] == "error"
        assert opened == []

    def test_star_pre_approves_every_write(self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(serial_mod.COMMAND_ALLOW_ENV, "*")

        res = serial_mod.serial_tool(action="send", port=PORT, data="x")

        assert res["status"] == "success", res
        assert opened[0].writes == [b"x"]

    def test_bypass_tool_consent_lifts_the_gate(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        res = serial_mod.serial_tool(action="feetech_position", port=PORT, motor_id=1, position=1)

        assert res["status"] == "success", res
        assert len(opened[0].writes) == 1


class TestTheDashboardHookIsNotAskedTwice:
    """A yes the dashboard's ``MotionInterruptHook`` already recorded is spent, not re-asked."""

    def test_a_deposited_grant_lets_the_write_through_without_an_interrupt(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        tool_input = {"action": "feetech_position", "port": PORT, "motor_id": 2, "position": 512}
        agent_hitl.deposit_grant("serial_tool", tool_input)
        ctx = _ctx("n")
        try:
            res = serial_mod.serial_tool(
                action="feetech_position", port=PORT, motor_id=2, position=512, tool_context=ctx
            )
        finally:
            agent_hitl.consume_grant("serial_tool", tool_input)

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()
        assert len(opened[0].writes) == 1

    def test_a_grant_is_spendable_exactly_once(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        tool_input = {"action": "feetech_position", "port": PORT, "motor_id": 2, "position": 512}
        agent_hitl.deposit_grant("serial_tool", tool_input)
        serial_mod.serial_tool(action="feetech_position", port=PORT, motor_id=2, position=512, tool_context=_ctx("n"))

        res = serial_mod.serial_tool(
            action="feetech_position", port=PORT, motor_id=2, position=512, tool_context=_ctx("n")
        )

        assert res["status"] == "error", res
        assert len(opened) == 1

    def test_a_grant_for_another_motion_does_not_cover_this_one(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        other = {"action": "feetech_position", "port": PORT, "motor_id": 2, "position": 4095}
        agent_hitl.deposit_grant("serial_tool", other)
        try:
            res = serial_mod.serial_tool(
                action="feetech_position", port=PORT, motor_id=2, position=512, tool_context=_ctx("n")
            )
        finally:
            agent_hitl.consume_grant("serial_tool", other)

        assert res["status"] == "error", res
        assert opened == []

    def test_a_missing_dashboard_extra_means_no_grant_not_a_crash(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _no_dashboard(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("strands_robots.dashboard"):
                raise ImportError("No module named 'strands_robots.dashboard'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_dashboard)

        res = serial_mod.serial_tool(
            action="feetech_position", port=PORT, motor_id=1, position=1, tool_context=_ctx("n")
        )

        assert res["status"] == "error", res
        assert "declined" in res["content"][0]["text"]
        assert opened == []


class TestReadsAreNeverGated:
    @pytest.mark.parametrize(
        ("action", "kwargs"),
        [("read", {}), ("feetech_ping", {"motor_id": 1}), ("monitor", {"timeout": 0.0})],
    )
    def test_a_read_needs_no_context_and_no_approval(
        self, opened: list[_FakeSerial], action: str, kwargs: dict[str, Any]
    ) -> None:
        ctx = _ctx("n")
        res = serial_mod.serial_tool(action=action, port=PORT, tool_context=ctx, **kwargs)

        assert res["status"] in ("success", "error"), res
        assert "operator" not in res["content"][0]["text"]
        ctx.interrupt.assert_not_called()
        assert len(opened) == 1

    def test_list_ports_needs_no_port_and_no_approval(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(serial_mod.serial.tools.list_ports, "comports", lambda: [])
        res = serial_mod.serial_tool(action="list_ports")

        assert res["status"] == "success"
