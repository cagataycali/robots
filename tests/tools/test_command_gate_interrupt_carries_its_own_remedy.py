"""The gate's interrupt tells a headless script how to answer it.

``agent("move the wrist")`` on a real arm returns a PAUSED result: the operator
gate raised an interrupt and the SDK handed it back for someone to answer. A
script that does ``print(agent(...))`` sees the interrupt list's repr - the
question, the action, the target - and nothing that says the run is paused or
how to resume it (the owner's four-line script in the README ends there). The
``reason`` dict is the one thing that script is certainly printing, so it now
carries ``how_to_answer``: the exact ``agent([{"interruptResponse": ...}])``
form, what 'y' means, and the pre-approval variable for a script with no
operator. Every tool on the shared gate gets the same line.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.tools import _command_gate as gate_mod
from strands_robots.tools._command_gate import HOW_TO_ANSWER, gate_motion


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_TEST_COMMAND_ALLOW", raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


def _ctx(response: object) -> MagicMock:
    ctx = MagicMock()
    ctx.interrupt.return_value = response
    return ctx


def _reason(tool: str = "robot", action: str = "set_joint_positions", target: str = "so101") -> dict:
    ctx = _ctx("n")
    gate_motion(tool, action, target, "it moves.", ctx, allow_env="STRANDS_TEST_COMMAND_ALLOW")
    return ctx.interrupt.call_args.kwargs["reason"]


class TestTheReasonCarriesTheRemedy:
    def test_how_to_answer_is_in_the_reason(self) -> None:
        reason = _reason()
        assert reason["how_to_answer"] == HOW_TO_ANSWER
        # the pre-existing fields are untouched: hosts that read them keep working
        assert reason["action"] == "set_joint_positions" and reason["target"] == "so101"
        assert reason["warning"].endswith("Reply 'y' to approve, anything else to deny.")

    def test_the_line_says_paused_how_to_resume_and_what_y_means(self) -> None:
        assert "paused, not done" in HOW_TO_ANSWER
        assert "result.interrupts" in HOW_TO_ANSWER
        assert 'agent([{"interruptResponse": {"interruptId":' in HOW_TO_ANSWER
        assert '"response": "y"' in HOW_TO_ANSWER
        assert "anything else denies and nothing moves" in HOW_TO_ANSWER
        assert "*_COMMAND_ALLOW" in HOW_TO_ANSWER

    @pytest.mark.parametrize("tool", ["robot", "serial_tool", "pose_tool", "use_unitree", "use_ros"])
    def test_every_tool_on_the_shared_gate_gets_the_same_line(self, tool: str) -> None:
        assert _reason(tool=tool)["how_to_answer"] == HOW_TO_ANSWER

    def test_the_headless_refusal_is_unchanged(self) -> None:
        """No operator, no interrupt: the refusal still names both escape hatches."""
        refusal = gate_motion("robot", "execute", "so101", "it moves.", None, allow_env="STRANDS_TEST_COMMAND_ALLOW")
        assert refusal is not None
        assert "STRANDS_TEST_COMMAND_ALLOW" in refusal and gate_mod.BYPASS_CONSENT_ENV in refusal


def test_the_real_robot_tool_interrupt_carries_it() -> None:
    """End to end through ``Robot.stream`` on a fake arm: the event a script prints has the line."""
    from strands.types._events import ToolInterruptEvent

    from tests.test_hardware_robot_direct_motion import _make_hw, _robot, _state, _stream

    hw = _make_hw(_robot(calibrated=False))
    events = _stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 95.0})
    assert isinstance(events[-1], ToolInterruptEvent)
    (interrupt,) = events[-1].interrupts
    assert interrupt.reason["how_to_answer"] == HOW_TO_ANSWER
    # ...and it survives the repr a script prints
    assert "interruptResponse" in repr(interrupt.to_dict())
