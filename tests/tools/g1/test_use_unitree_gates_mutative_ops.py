"""``use_unitree`` stops for an operator before a mutative RPC reaches the robot.

Finding F-001 (CWE-862): the raw SDK escape hatch dispatched ``loco.SetVelocity``,
``loco.ZeroTorque``, ``loco.SetFsmId`` and ``motion_switcher.ReleaseMode`` with a
``logger.warning`` as its only rail. An agent steered by untrusted content could
walk a standing humanoid or drop its holding torque with nobody in the loop,
while the sibling ROS transports refused the same class of command without a
human's ``y``.

These tests grade the observable: whether the SDK method was called. The stand-in
client records every call, so "the robot did not move" is a measured claim and
not a reading of which branch the tool took. The gate itself is the shared one in
:mod:`~strands_robots.tools._command_gate`; its audit row is graded beside the
other gates in ``tests/tools/test_hitl_operator_response_audit.py``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.g1.use_unitree as uu


class _Recorder:
    """Stand-in SDK client that records every method call instead of moving a robot."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def SetVelocity(self, vx: float, vy: float, vyaw: float, duration: float = 1.0) -> int:
        self.calls.append(("SetVelocity", {"vx": vx, "vy": vy, "vyaw": vyaw, "duration": duration}))
        return 0

    def ZeroTorque(self) -> int:
        self.calls.append(("ZeroTorque", {}))
        return 0

    def SetFsmId(self, fsm_id: int) -> int:
        self.calls.append(("SetFsmId", {"fsm_id": fsm_id}))
        return 0

    def StopMove(self) -> int:
        self.calls.append(("StopMove", {}))
        return 0

    def GetFsmId(self) -> tuple[int, int]:
        self.calls.append(("GetFsmId", {}))
        return (0, 801)

    def ReleaseMode(self) -> int:
        self.calls.append(("ReleaseMode", {}))
        return 0

    def TtsMaker(self, text: str, speaker_id: int = 0) -> int:
        self.calls.append(("TtsMaker", {"text": text, "speaker_id": speaker_id}))
        return 0


def _ctx(response: object) -> MagicMock:
    """Stand-in ToolContext whose interrupt() returns *response*."""
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """A reachable bus with recording clients and a clean gate environment."""
    rec = _Recorder()
    monkeypatch.setattr(uu, "ensure_dds", lambda _iface: None)
    monkeypatch.setattr(uu, "_CLIENTS", {"loco": rec, "motion_switcher": rec, "audio": rec})
    for name in ("BYPASS_TOOL_CONSENT", uu.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    return rec


@pytest.fixture(autouse=True)
def _audit_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the gate's audit row off the developer's real ``~/.strands_robots`` trail."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


HIGH_DANGER = sorted(uu.HIGH_DANGER_OPS)


class TestWithoutApprovalNothingIsDispatched:
    @pytest.mark.parametrize(
        ("service", "op"), [("loco", "ZeroTorque"), ("loco", "SetFsmId"), ("motion_switcher", "ReleaseMode")]
    )
    def test_a_declined_high_danger_op_never_reaches_the_client(self, robot: _Recorder, service: str, op: str) -> None:
        params = {"fsm_id": 0} if op == "SetFsmId" else {}
        res = uu.use_unitree(service, op, params, tool_context=_ctx("n - the robot is standing"))

        assert res["status"] == "error", res
        assert res["dispatched"] is False
        assert "declined by the operator" in res["message"]
        assert robot.calls == [], f"the SDK method was called after the operator said no: {robot.calls}"
        # The classification the danger tests pin survives on the refusal too.
        assert res["high_danger"] is True and res["mutative"] is True

    def test_no_tool_context_refuses_and_names_both_escape_hatches(self, robot: _Recorder) -> None:
        """Headless with nothing pre-approved fails closed, not open."""
        res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0})

        assert res["status"] == "error", res
        assert res["dispatched"] is False
        assert uu.COMMAND_ALLOW_ENV in res["message"]
        assert "BYPASS_TOOL_CONSENT" in res["message"]
        assert robot.calls == []

    def test_a_merely_mutative_op_is_gated_too(self, robot: _Recorder) -> None:
        """``audio.TtsMaker`` is not high-danger but it is a write; a write asks."""
        res = uu.use_unitree("audio", "TtsMaker", {"text": "hi"}, tool_context=_ctx("no"))

        assert res["status"] == "error" and res["dispatched"] is False
        assert res["high_danger"] is False and res["mutative"] is True
        assert robot.calls == []

    def test_interrupts_unavailable_is_a_refusal_not_a_dispatch(self, robot: _Recorder) -> None:
        ctx = MagicMock(name="ToolContext")
        ctx.interrupt.side_effect = RuntimeError("no interrupt channel")

        res = uu.use_unitree("loco", "ZeroTorque", {}, tool_context=ctx)

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.calls == []


class TestAnApprovalDispatchesExactlyOnce:
    def test_an_approved_op_is_sent_once_with_its_parameters(self, robot: _Recorder) -> None:
        ctx = _ctx("y")
        res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.3, "vy": 0.0, "vyaw": 0.1}, tool_context=ctx)

        assert res["status"] == "success", res
        assert robot.calls == [("SetVelocity", {"vx": 0.3, "vy": 0.0, "vyaw": 0.1, "duration": 1.0})]
        ctx.interrupt.assert_called_once()

    def test_the_interrupt_names_the_tool_and_the_pair(self, robot: _Recorder) -> None:
        """The operator is told which tool asks and what exactly it wants to send."""
        ctx = _ctx("y")
        uu.use_unitree("loco", "ZeroTorque", {}, tool_context=ctx)

        interrupt_id = ctx.interrupt.call_args.args[0]
        reason = ctx.interrupt.call_args.kwargs["reason"]
        assert interrupt_id == "use_unitree-command-approval"
        assert reason["action"] == "ZeroTorque"
        assert reason["target"] == "loco.ZeroTorque"
        assert "loco.ZeroTorque" in reason["warning"]


class TestPreApprovalIsSilent:
    def test_an_allowlisted_pair_is_sent_without_an_interrupt(
        self, robot: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(uu.COMMAND_ALLOW_ENV, "audio.TtsMaker, loco.SetVelocity")
        ctx = _ctx("n")

        res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.1, "vy": 0.0, "vyaw": 0.0}, tool_context=ctx)

        assert res["status"] == "success", res
        assert [name for name, _ in robot.calls] == ["SetVelocity"]
        ctx.interrupt.assert_not_called()

    def test_the_allowlist_is_exact_so_a_sibling_pair_still_asks(
        self, robot: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-approving SetVelocity does not pre-approve ZeroTorque on the same service."""
        monkeypatch.setenv(uu.COMMAND_ALLOW_ENV, "loco.SetVelocity")

        res = uu.use_unitree("loco", "ZeroTorque", {}, tool_context=_ctx("n"))

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.calls == []

    def test_star_pre_approves_every_pair(self, robot: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(uu.COMMAND_ALLOW_ENV, "*")

        res = uu.use_unitree("motion_switcher", "ReleaseMode", {})

        assert res["status"] == "success", res
        assert robot.calls == [("ReleaseMode", {})]

    def test_bypass_tool_consent_lifts_the_gate(self, robot: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        res = uu.use_unitree("loco", "ZeroTorque", {})

        assert res["status"] == "success", res
        assert robot.calls == [("ZeroTorque", {})]


class TestReadsAndStopsAreNeverGated:
    def test_a_read_needs_no_context_and_no_approval(self, robot: _Recorder) -> None:
        res = uu.use_unitree("loco", "GetFsmId", {})

        assert res["status"] == "success", res
        assert robot.calls == [("GetFsmId", {})]

    def test_a_read_with_a_context_never_interrupts(self, robot: _Recorder) -> None:
        ctx = _ctx("n")
        res = uu.use_unitree("loco", "GetFsmId", {}, tool_context=ctx)

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()

    def test_stop_move_goes_out_without_asking(self, robot: _Recorder) -> None:
        """The emergency stop must not be behind the prompt it is meant to rescue."""
        ctx = _ctx("n")
        res = uu.use_unitree("loco", "StopMove", {}, tool_context=ctx)

        assert res["status"] == "success", res
        assert robot.calls == [("StopMove", {})]
        ctx.interrupt.assert_not_called()

    def test_meta_operations_need_no_context(self) -> None:
        res = uu.use_unitree("meta", "list_services")

        assert res["status"] == "success"


def test_the_tool_declares_it_receives_the_operator_context() -> None:
    """Without ``context=True`` the gate could only ever fail closed."""
    fn = getattr(uu.use_unitree, "__wrapped__", None) or uu.use_unitree
    params = inspect.signature(fn).parameters
    assert "tool_context" in params
    assert list(params)[-1] == "tool_context", (
        "tool_context must stay the last parameter so the model-facing order is unchanged"
    )
