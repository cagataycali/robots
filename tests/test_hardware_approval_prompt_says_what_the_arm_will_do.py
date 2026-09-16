"""The operator-approval prompt for ``execute`` / ``start`` describes the motion, not the request.

Measured with an agent and a live gate: the interrupt the operator answered
read ``'execute' drives the real robot 'so101' with 'Wave the arm' (policy
mock at localhost:None); it needs operator approval before it is
dispatched``. With ``mock`` the arm does not wave - every joint follows a
sinusoid - and there is no server at ``localhost:None``. The one sentence an
operator reads before a real arm moves described a motion that would not
happen, at a server that did not exist, with no time budget.
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot
from strands_robots.tools import _command_gate


@pytest.fixture
def arm():
    robot = Robot("so101", mode="real", port=os.devnull)
    yield robot
    robot.cleanup()


class TestTheWarning:
    def test_mock_names_the_motion_that_will_happen(self, arm) -> None:
        text = arm._motion_warning("execute", {"instruction": "Wave the arm", "policy_provider": "mock", "duration": 3})
        assert text.startswith("'execute' drives the real robot 'so101' for up to 3s with 'Wave the arm' ")
        assert "(policy mock, built in process, no server)" in text
        assert "localhost:None" not in text
        assert "MockPolicy does not read the instruction: every joint will follow a sinusoidal test motion" in text

    def test_a_server_provider_names_the_server_and_nothing_about_sinusoids(self, arm) -> None:
        text = arm._motion_warning(
            "start", {"instruction": "pick up the cube", "policy_provider": "groot", "policy_port": 5555}
        )
        assert "'start' drives the real robot 'so101' for up to 30.0s with 'pick up the cube' " in text
        assert "(policy groot on localhost:5555)" in text
        assert "sinusoidal" not in text

    def test_default_provider_is_named(self, arm) -> None:
        text = arm._motion_warning("execute", {"instruction": "x"})
        assert "policy groot" in text

    def test_unknown_provider_still_produces_a_sentence(self, arm) -> None:
        text = arm._motion_warning("execute", {"instruction": "x", "policy_provider": "no-such"})
        assert "policy no-such, built in process, no server" in text
        assert "sinusoidal" not in text

    def test_non_numeric_duration_is_left_out_not_crashed(self, arm) -> None:
        text = arm._motion_warning("execute", {"instruction": "x", "duration": "soon", "policy_provider": "mock"})
        assert "for up to" not in text and "drives the real robot" in text


class TestItReachesTheOperator:
    def test_the_interrupt_reason_carries_the_sentence(self, arm, monkeypatch) -> None:
        seen: dict = {}

        def fake_gate_motion(tool, action, target, warning, tool_context, **kw):
            seen.update(tool=tool, action=action, target=target, warning=warning)
            return "refused for the test"

        monkeypatch.setattr(_command_gate, "gate_motion", fake_gate_motion)
        import strands_robots.hardware_robot as hr

        monkeypatch.setattr(hr, "gate_motion", fake_gate_motion)
        refusal = arm._gate_motion(
            "execute",
            {"action": "execute", "instruction": "Wave the arm", "policy_provider": "mock", "duration": 3},
            {"toolUseId": "t1", "name": "so101", "input": {}},
            {},
        )
        assert refusal == "refused for the test"
        assert seen["tool"] == "robot" and seen["target"] == "so101"
        assert "every joint will follow a sinusoidal test motion" in seen["warning"]
        assert "for up to 3s" in seen["warning"]
