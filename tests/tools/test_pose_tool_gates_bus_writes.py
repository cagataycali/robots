"""``pose_tool`` stops for an operator before a goal position reaches the arm.

Finding F-010 (CWE-862, follow-up to F-009): the five motion actions -
``move_motor``, ``move_multiple``, ``incremental_move``, ``load_pose``,
``reset_to_home`` - reached ``MotorController``'s writers with no gate of their
own. The dashboard's ``MotionInterruptHook`` lists them, but a hook is something
an ``Agent`` has to be built with, and every canonical
``Agent(tools=robot.tools)`` build has none, so the default wiring moved the arm
unasked.

These tests grade the observable: whether ``serial.Serial`` was constructed and
whether anything was written to it. The gate runs before the controller is
built, so a denied call is measured as "no port, no bytes", not as which branch
was taken. The audit row is graded beside the other gates in
``tests/tools/test_hitl_operator_response_audit.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.pose_tool as pose_mod

PORT = "/dev/ttyFAKE0"


def _status_packet(motor_id: int, position: int) -> bytes:
    """A verified Present_Position reply for ``motor_id``: ``FF FF ID LEN ERR LO HI CHK``."""
    lo, hi = position & 0xFF, (position >> 8) & 0xFF
    body = [motor_id, 4, 0, lo, hi]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


class _FakeSerial:
    """A bus whose every motor sits at mid-travel and answers any read."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.port = port
        self.is_open = True
        self.writes: list[bytes] = []
        self._last_id = 1

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        if len(data) > 2:
            self._last_id = data[2]

    def read(self, n: int = 1) -> bytes:
        return _status_packet(self._last_id, 2048)

    def close(self) -> None:
        self.is_open = False


def _ctx(response: object) -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[_FakeSerial]:
    """Every port the tool opens, with a clean gate environment, a private pose library and audit dir."""
    ports: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        fake = _FakeSerial(port, baudrate, timeout)
        ports.append(fake)
        return fake

    monkeypatch.setattr(pose_mod.serial, "Serial", _ctor)
    monkeypatch.setattr(pose_mod.time, "sleep", lambda *_: None)
    for name in ("BYPASS_TOOL_CONSENT", pose_mod.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.chdir(tmp_path)  # PoseManager stores under cwd/.strands_robots/poses
    pose_mod.PoseManager("so101_follower").store_pose("rest", {"shoulder_pan": 0.0, "elbow_flex": 10.0})
    return ports


MOTIONS: dict[str, dict[str, Any]] = {
    "move_motor": {"motor_name": "shoulder_pan", "position": 10.0},
    "move_multiple": {"positions": {"shoulder_pan": 10.0, "elbow_flex": 5.0}, "smooth": False},
    "incremental_move": {"motor_name": "shoulder_pan", "delta": 1.0},
    "load_pose": {"pose_name": "rest", "smooth": False},
    "reset_to_home": {"steps": 2, "step_delay": 0.001},
}


def _writes(ports: list[_FakeSerial]) -> int:
    return sum(len(p.writes) for p in ports)


class TestWithoutApprovalNothingTouchesTheBus:
    @pytest.mark.parametrize("action", sorted(MOTIONS))
    def test_a_declined_motion_opens_no_port(self, opened: list[_FakeSerial], action: str) -> None:
        res = pose_mod.pose_tool(
            action=action, port=PORT, tool_context=_ctx("n - someone is holding the arm"), **MOTIONS[action]
        )

        assert res["status"] == "error", res
        assert "declined by the operator" in res["content"][0]["text"]
        assert opened == [], f"the port was opened after the operator said no: {opened}"

    @pytest.mark.parametrize("action", sorted(MOTIONS))
    def test_no_tool_context_refuses_and_names_both_escape_hatches(
        self, opened: list[_FakeSerial], action: str
    ) -> None:
        """Headless with nothing pre-approved fails closed - the default agent wiring."""
        res = pose_mod.pose_tool(action=action, port=PORT, **MOTIONS[action])

        assert res["status"] == "error", res
        text = res["content"][0]["text"]
        assert pose_mod.COMMAND_ALLOW_ENV in text and "BYPASS_TOOL_CONSENT" in text
        assert opened == []

    def test_the_operator_is_shown_the_port_and_the_motion(self, opened: list[_FakeSerial]) -> None:
        ctx = _ctx("y")
        pose_mod.pose_tool(action="move_motor", port=PORT, motor_name="elbow_flex", position=25.0, tool_context=ctx)

        assert ctx.interrupt.call_args.args[0] == "pose_tool-command-approval"
        reason = ctx.interrupt.call_args.kwargs["reason"]
        assert reason["action"] == "move_motor"
        assert reason["target"] == PORT
        assert "motor_name=elbow_flex" in reason["warning"] and "position=25.0" in reason["warning"]

    def test_an_out_of_travel_target_is_refused_before_the_operator_is_asked(self, opened: list[_FakeSerial]) -> None:
        """A human is not asked to approve a command the tool would refuse anyway."""
        ctx = _ctx("y")
        res = pose_mod.pose_tool(
            action="move_motor", port=PORT, motor_name="shoulder_pan", position=1e4, tool_context=ctx
        )

        assert res["status"] == "error"
        ctx.interrupt.assert_not_called()
        assert opened == []


class TestAnApprovalMovesTheArm:
    @pytest.mark.parametrize("action", sorted(MOTIONS))
    def test_an_approved_motion_reaches_the_port(self, opened: list[_FakeSerial], action: str) -> None:
        ctx = _ctx("y")
        res = pose_mod.pose_tool(action=action, port=PORT, tool_context=ctx, **MOTIONS[action])

        assert res["status"] == "success", res
        assert len(opened) == 1 and _writes(opened) >= 1
        ctx.interrupt.assert_called_once()


class TestPreApprovalIsSilent:
    def test_a_named_action_moves_without_an_interrupt(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(pose_mod.COMMAND_ALLOW_ENV, "move_motor, load_pose")
        ctx = _ctx("n")

        res = pose_mod.pose_tool(
            action="move_motor", port=PORT, motor_name="shoulder_pan", position=5.0, tool_context=ctx
        )

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()

    def test_the_allowlist_is_exact_so_a_sibling_action_still_asks(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(pose_mod.COMMAND_ALLOW_ENV, "move_motor")

        res = pose_mod.pose_tool(action="reset_to_home", port=PORT, tool_context=_ctx("n"))

        assert res["status"] == "error"
        assert opened == []

    def test_star_pre_approves_every_motion(self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(pose_mod.COMMAND_ALLOW_ENV, "*")

        res = pose_mod.pose_tool(action="load_pose", port=PORT, pose_name="rest", smooth=False)

        assert res["status"] == "success", res
        assert _writes(opened) >= 1

    def test_bypass_tool_consent_lifts_the_gate(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        res = pose_mod.pose_tool(action="move_motor", port=PORT, motor_name="shoulder_pan", position=1.0)

        assert res["status"] == "success", res
        assert _writes(opened) == 1


class TestTheDashboardHookIsNotAskedTwice:
    """A yes the dashboard's ``MotionInterruptHook`` already recorded is spent, not re-asked."""

    def test_a_deposited_grant_lets_the_motion_through_without_an_interrupt(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        tool_input = {"action": "move_motor", "port": PORT, "motor_name": "shoulder_pan", "position": 12.0}
        agent_hitl.deposit_grant("pose_tool", tool_input)
        ctx = _ctx("n")
        try:
            res = pose_mod.pose_tool(
                action="move_motor", port=PORT, motor_name="shoulder_pan", position=12.0, tool_context=ctx
            )
        finally:
            agent_hitl.consume_grant("pose_tool", tool_input)

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()
        assert _writes(opened) == 1

    def test_a_grant_is_spendable_exactly_once(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        tool_input = {"action": "move_motor", "port": PORT, "motor_name": "shoulder_pan", "position": 12.0}
        agent_hitl.deposit_grant("pose_tool", tool_input)
        pose_mod.pose_tool(
            action="move_motor", port=PORT, motor_name="shoulder_pan", position=12.0, tool_context=_ctx("n")
        )

        res = pose_mod.pose_tool(
            action="move_motor", port=PORT, motor_name="shoulder_pan", position=12.0, tool_context=_ctx("n")
        )

        assert res["status"] == "error", res
        assert len(opened) == 1

    def test_a_grant_for_another_motion_does_not_cover_this_one(self, opened: list[_FakeSerial]) -> None:
        from strands_robots.dashboard import agent_hitl

        other = {"action": "move_motor", "port": PORT, "motor_name": "shoulder_pan", "position": 170.0}
        agent_hitl.deposit_grant("pose_tool", other)
        try:
            res = pose_mod.pose_tool(
                action="move_motor", port=PORT, motor_name="shoulder_pan", position=12.0, tool_context=_ctx("n")
            )
        finally:
            agent_hitl.consume_grant("pose_tool", other)

        assert res["status"] == "error", res
        assert opened == []

    def test_a_missing_dashboard_extra_means_no_grant_not_a_crash(
        self, opened: list[_FakeSerial], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _no_dashboard(name: str, *args: Any, **kwargs: Any) -> object:
            if name.startswith("strands_robots.dashboard"):
                raise ImportError("No module named 'strands_robots.dashboard'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_dashboard)

        res = pose_mod.pose_tool(
            action="move_motor", port=PORT, motor_name="shoulder_pan", position=1.0, tool_context=_ctx("n")
        )

        assert res["status"] == "error", res
        assert "declined" in res["content"][0]["text"]
        assert opened == []


class TestStopReadsAndTheLibraryAreNeverGated:
    @pytest.mark.parametrize(
        ("action", "kwargs"),
        [
            ("emergency_stop", {}),
            ("connect", {}),
            ("read_position", {"motor_name": "shoulder_pan"}),
            ("read_all", {}),
            ("store_pose", {"pose_name": "now"}),
        ],
    )
    def test_a_bus_action_that_moves_nothing_needs_no_approval(
        self, opened: list[_FakeSerial], action: str, kwargs: dict[str, Any]
    ) -> None:
        ctx = _ctx("n")
        res = pose_mod.pose_tool(action=action, port=PORT, tool_context=ctx, **kwargs)

        assert res["status"] in ("success", "error"), res
        assert "operator" not in res["content"][0]["text"]
        ctx.interrupt.assert_not_called()
        assert len(opened) == 1

    @pytest.mark.parametrize(
        ("action", "kwargs"),
        [("list_poses", {}), ("show_pose", {"pose_name": "rest"}), ("delete_pose", {"pose_name": "rest"})],
    )
    def test_the_pose_library_needs_no_port_and_no_approval(
        self, opened: list[_FakeSerial], action: str, kwargs: dict[str, Any]
    ) -> None:
        ctx = _ctx("n")
        res = pose_mod.pose_tool(action=action, port=None, tool_context=ctx, **kwargs)

        assert res["status"] == "success", res
        ctx.interrupt.assert_not_called()
        assert opened == []


def test_the_gated_set_is_exactly_the_dashboard_hooks_set() -> None:
    """One roster: what the hook asks about is what the tool gates, so a grant always has a spender."""
    from strands_robots.dashboard import agent_hitl

    assert pose_mod.MOTION_ACTIONS == agent_hitl.MOTION_ACTIONS["pose_tool"]


def test_the_audit_dir_env_is_the_one_the_gate_writes(opened: list[_FakeSerial]) -> None:
    pose_mod.pose_tool(action="move_motor", port=PORT, motor_name="shoulder_pan", position=1.0, tool_context=_ctx("n"))
    assert os.path.isdir(os.environ["STRANDS_MESH_AUDIT_DIR"])
