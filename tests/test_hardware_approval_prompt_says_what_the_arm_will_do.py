"""The approval prompt says how long the arm may move, and whether the words matter.

The interrupt an operator answers before a real arm moves read ``'execute'
drives the real robot 'so101' with 'Wave the arm' (policy mock at
localhost:None); it needs operator approval before it is dispatched``. Two facts
that decide whether to approve were missing: with ``mock`` the arm does not wave
- every joint follows a sinusoid whatever the task says - and the wall-clock
budget the control loop honours was unstated, so "y" bought an unknown length of
motion the words would not shape.

The notice is the one ``start`` and a RUNNING ``status`` already carry, so an
operator reads before dispatch exactly what the envelopes will report after.
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import TaskStatus
from strands_robots.policies.base import instruction_not_read_notice
from strands_robots.policies.mock import MockPolicy


@pytest.fixture
def arm():
    """A real-mode Robot; nothing here connects to the bus."""
    robot = Robot("so101", mode="real", port=os.devnull)
    yield robot
    robot.cleanup()


def warning_for(arm, monkeypatch, **tool_input) -> str:
    """The warning the gate is handed for one ``execute`` call."""
    seen: dict[str, str] = {}

    def fake_gate_motion(tool, action, target, warning, tool_context, **kwargs):
        seen["warning"] = warning
        return "refused for the test"

    monkeypatch.setattr("strands_robots.hardware_robot.gate_motion", fake_gate_motion)
    assert (
        arm._gate_motion(
            "execute",
            {"action": "execute", "instruction": "Wave the arm", **tool_input},
            {"toolUseId": "t1", "name": "so101", "input": {}},
            {},
        )
        == "refused for the test"
    )
    return seen["warning"]


class TestTheSentenceTheOperatorApproves:
    def test_the_prompt_for_a_mock_rollout_reads_whole(self, arm, monkeypatch) -> None:
        """Graded as one string: an operator reads one sentence, not three facts."""
        assert warning_for(arm, monkeypatch, policy_provider="mock", duration=3) == (
            "'execute' drives the real robot 'so101' for up to 3s with 'Wave the arm' "
            "(policy mock built in this process, no server); it needs operator approval before it is dispatched. "
            "Note: MockPolicy does not read the instruction. Its actions - a test motion on every "
            "joint - are commanded to the robot whatever the task says; no status or completion "
            "that follows will mean the task was performed."
        )

    @pytest.mark.parametrize(
        ("duration", "expected"),
        [(3, "for up to 3s"), (0.5, "for up to 0.5s"), (120.0, "for up to 120s")],
    )
    def test_the_budget_is_the_horizon_the_loop_will_honour(self, arm, monkeypatch, duration, expected: str) -> None:
        assert expected in warning_for(arm, monkeypatch, policy_provider="mock", duration=duration)

    def test_a_call_that_names_no_budget_is_told_the_default_one(self, arm, monkeypatch) -> None:
        """``duration`` defaults to 30.0 for ``execute`` and ``start``, so the
        prompt states that rather than leaving the horizon unsaid."""
        assert "for up to 30s" in warning_for(arm, monkeypatch, policy_provider="mock")

    def test_only_a_policy_that_ignores_the_instruction_carries_the_notice(self, arm, monkeypatch) -> None:
        """Keyed on the class the provider resolves to, not on a provider name:
        ``groot`` reads its instruction, so nothing is added."""
        assert "does not read the instruction" in warning_for(arm, monkeypatch, policy_provider="mock")
        assert "does not read the instruction" not in warning_for(
            arm, monkeypatch, policy_provider="groot", policy_port=5555
        )

    def test_the_notice_is_the_one_the_envelopes_carry(self, arm, monkeypatch) -> None:
        """One sentence, one source: the prompt must not describe the same policy
        in words of its own that could drift from what ``start`` then reports."""
        notice = instruction_not_read_notice(MockPolicy, pending=True)
        assert notice is not None
        assert notice in warning_for(arm, monkeypatch, policy_provider="mock")

    def test_the_budget_is_dropped_not_crashed_for_a_value_the_pre_gate_refuses(self, arm, monkeypatch) -> None:
        """A ``duration`` that is not a positive finite number never reaches the
        gate - :meth:`_pre_gate_error` refuses it first - so the clause is the
        only thing lost if a caller ever arrives without that check."""
        assert arm._pre_gate_error("execute", None, "mock", "soon") is not None
        text = warning_for(arm, monkeypatch, policy_provider="mock", duration="soon")
        assert "for up to" not in text
        assert text.startswith("'execute' drives the real robot 'so101' with 'Wave the arm' ")


class TestItReachesTheOperator:
    def test_the_headless_refusal_carries_the_same_sentence(self, arm, monkeypatch) -> None:
        """No agent means no operator to ask; the refusal a script reads is built
        from the same warning, so a headless run is told what it was denied."""
        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
        monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
        refusal = arm._gate_motion(
            "start",
            {"action": "start", "instruction": "Wave the arm", "policy_provider": "mock", "duration": 3},
            {"toolUseId": "t1", "name": "so101", "input": {}},
            {},
        )
        assert refusal is not None
        assert refusal.startswith("'start' drives the real robot 'so101' for up to 3s with 'Wave the arm' ")
        assert "MockPolicy does not read the instruction" in refusal
        assert "STRANDS_ROBOT_COMMAND_ALLOW" in refusal


class TestTheStatusEnvelopeCarriesTheNoticeOnce:
    def test_a_running_status_on_a_policy_that_ignores_the_words_says_so_once(self, arm) -> None:
        """The same sentence, once: ``status`` is parsed by agents and operators,
        and a merge that keeps two copies of the block that appends it doubles
        the notice on every non-idle report. Graded by count, on the envelope."""
        arm._task_state.status = TaskStatus.RUNNING
        arm._task_state.instruction = "Wave the arm"
        arm._task_state.policy = MockPolicy
        notice = instruction_not_read_notice(MockPolicy, pending=True)
        assert notice is not None
        text = arm.get_task_status()["content"][0]["text"]
        assert text.count(notice) == 1, text
