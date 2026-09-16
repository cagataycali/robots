"""A real robot tool refusing a verb it does not have names the tool that does.

``Robot(mode="real")`` publishes four actions: execute, start, status, stop.
The simulation tool also has ``teleoperate`` and ``start_recording`` /
``stop_recording``, and the quickstart once asked the real robot tool for all
three in one prompt - every call came back "Unknown action" with nothing about
where the verb went. The refusal now names ``lerobot_teleoperate`` (and the
Python ``attach_teleop().teleoperate()`` path) for those verbs; an unknown
spelling still gets the plain refusal. The quickstart itself is pinned here to
verbs the tool it hands the agent actually has.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor

QUICKSTART = Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "quickstart.md"
REAL_ROBOT_ACTIONS = {"execute", "start", "status", "stop"}


class _Arm:
    name = "so101"
    robot_type = "so_follower"
    is_connected = False
    config = type("Cfg", (), {"port": None, "cameras": {}})()


def _hw() -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "so101"
    hw.data_config = None
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="t")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = _Arm()
    return hw


def _call(hw: HwRobot, action: str) -> str:
    async def run() -> str:
        events = [e async for e in hw.stream({"toolUseId": "t", "input": {"action": action}}, {})]
        result = events[-1].tool_result
        assert result["status"] == "error"
        return result["content"][0]["text"]

    return asyncio.run(run())


@pytest.mark.parametrize("action", ["teleoperate", "start_teleop", "stop_teleoperate"])
def test_a_teleoperation_verb_is_sent_to_the_tool_that_teleoperates(action):
    text = _call(_hw(), action)
    assert text.startswith(f"Unknown action: {action}. Valid actions: execute, start, status, stop")
    assert "lerobot_teleoperate" in text
    assert "attach_teleop" in text
    assert "does not teleoperate from an agent" in text


@pytest.mark.parametrize("action", ["record", "start_recording", "stop_recording", "record_episode"])
def test_a_recording_verb_is_sent_to_the_tool_that_records(action):
    text = _call(_hw(), action)
    assert text.startswith(f"Unknown action: {action}. Valid actions: execute, start, status, stop")
    assert "lerobot_teleoperate" in text
    assert "dataset_repo_id" in text
    assert "simulation tool's actions" in text


def test_an_unknown_spelling_gets_the_plain_refusal():
    text = _call(_hw(), "bogus")
    assert text == "Unknown action: bogus. Valid actions: execute, start, status, stop"


def test_the_refusal_is_one_content_block_of_text():
    hw = _hw()

    async def run():
        events = [e async for e in hw.stream({"toolUseId": "t", "input": {"action": "teleoperate"}}, {})]
        return events[-1].tool_result

    result = asyncio.run(run())
    assert [set(block) for block in result["content"]] == [{"text"}]


class TestTheQuickstartAsksTheRealRobotToolOnlyForVerbsItHas:
    """The prompt handed to ``Agent(tools=[follower])`` may only use the four real actions."""

    def _agent_prompts_over_a_real_robot(self) -> list[str]:
        text = QUICKSTART.read_text()
        prompts = []
        for m in re.finditer(r"Agent\(tools=\[(?P<tools>[^\]]+)\]\)\(\s*(?P<prompt>(?:\"[^\"]*\"\s*)+)\)", text):
            if "follower" in m.group("tools").split(","):
                prompts.append("".join(re.findall(r'"([^"]*)"', m.group("prompt"))))
        return prompts

    def test_no_prompt_over_the_real_robot_asks_for_teleoperation_or_recording(self):
        for prompt in self._agent_prompts_over_a_real_robot():
            for verb in ("start_recording", "stop_recording", "teleoperate"):
                assert verb not in prompt, f"the quickstart asks the real robot tool for {verb!r}: {prompt!r}"

    def test_recording_a_real_arm_is_shown_through_lerobot_teleoperate(self):
        text = QUICKSTART.read_text()
        assert "from strands_robots import lerobot_teleoperate" in text
        assert "Agent(tools=[lerobot_teleoperate])" in text

    def test_the_real_robot_tool_still_has_exactly_the_four_verbs(self):
        enum = set(_hw().tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        assert enum == REAL_ROBOT_ACTIONS
