"""``status`` on a real robot says whether the hardware is open, not only what the task state is.

Measured with the SO-101 unplugged and an agent asked "is the arm doing
anything right now?": the tool answered ``Robot Status: IDLE`` and the agent
told the operator "the arm is free and ready to receive new instructions".
Nothing in the reply said no arm was on the port.
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot


def _text(result) -> str:
    return " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))


@pytest.fixture
def arm(tmp_path):
    robot = Robot("so101", mode="real", port=str(tmp_path / "no-such-port"))
    yield robot
    robot.cleanup()


class TestBeforeAnyTask:
    def test_idle_names_the_missing_port(self, arm) -> None:
        text = _text(arm.get_task_status())
        assert "Robot Status: IDLE" in text
        assert "Connection: not connected (port '" in text
        assert "no-such-port' not found on this machine)" in text

    def test_a_present_but_unopened_port_reads_present(self, arm) -> None:
        arm.robot.config.port = os.devnull
        text = _text(arm.get_task_status())
        assert f"Connection: not connected (port {os.devnull!r} present)" in text

    def test_an_open_driver_reads_connected(self, arm, monkeypatch) -> None:
        monkeypatch.setattr(type(arm.robot), "is_connected", property(lambda self: True))
        assert "Connection: connected" in _text(arm.get_task_status())

    def test_a_probe_that_raises_reads_closed_not_crashed(self, arm, monkeypatch) -> None:
        def boom(self):
            raise RuntimeError("bus gone")

        monkeypatch.setattr(type(arm.robot), "is_connected", property(boom))
        text = _text(arm.get_task_status())
        assert "Connection: not connected" in text

    def test_the_status_action_carries_the_line(self, arm) -> None:
        import asyncio

        async def call():
            out = []
            async for ev in arm.stream({"toolUseId": "t", "name": "so101", "input": {"action": "status"}}, {}):
                out.append(ev)
            return out

        events = asyncio.run(call())
        text = _text(events[-1].tool_result)
        assert "Connection: not connected (port '" in text
