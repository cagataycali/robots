"""A real robot tool refusing a verb it does not have names where that verb lives.

``Robot(mode="real")`` publishes four actions: execute, start, status, stop. The
quickstart once asked it, in one prompt, for ``start_recording``, ``teleoperate``
and ``stop_recording`` - every call came back "Unknown action" with nothing about
where the verb went. Measured against the simulation tool's 77 published actions,
the three land in three different places: ``start_recording`` / ``stop_recording``
are its actions, ``teleoperate`` is not an action of either tool but a Python
method of the teleop mixin, and the recording of a real arm under a leader is the
``lerobot_teleoperate`` tool.

A fourth place is this tool itself. ``run_policy`` is both a simulation tool
action and a Python method of this class, and it is what this tool does - under
the names ``execute`` and ``start`` - so refusing it without saying so
contradicts the refusal's own opening ("so101 drives policies").

The refusal now names the destination for each group, an unknown spelling still
gets the plain refusal byte for byte, and building it cannot raise on the value
it reports. The quickstart itself is pinned here to verbs the tool it hands the
agent actually has.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pytest
from strands.types._events import ToolResultEvent

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


def _call(hw: HwRobot, action: object) -> str:
    """The one text block the tool answers ``action`` with.

    ``action`` is typed ``object`` rather than ``str`` because a Python caller
    may send anything, which is what
    :func:`test_building_the_refusal_cannot_raise_on_the_action_it_reports`
    sends.
    """

    async def run() -> str:
        events = [e async for e in hw.stream({"toolUseId": "t", "name": "so101", "input": {"action": action}}, {})]
        final = events[-1]
        assert isinstance(final, ToolResultEvent)
        result = final.tool_result
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


@pytest.mark.parametrize("action", ["run_policy", "start_policy", "stop_policy"])
def test_a_policy_verb_is_sent_back_to_this_tools_own_verbs(action):
    """The simulation tool's policy spellings resolve here, not elsewhere.

    The refusal's first sentence is that this tool drives policies, so the one
    thing it may not do with a policy verb is refuse it without naming the verb
    that runs it.
    """
    text = _call(_hw(), action)
    assert text.startswith(f"Unknown action: {action}. Valid actions: execute, start, status, stop")
    assert "does drive policies" in text
    for verb in ("action='execute'", "action='start'", "action='stop'"):
        assert verb in text
    assert "robot.run_policy(policy_object=...)" in text


def test_the_policy_verb_the_refusal_answers_is_one_this_class_has_in_python():
    """Why the policy group exists: the name is real here, just not in the enum."""
    assert hasattr(HwRobot, "run_policy")
    assert "run_policy" not in REAL_ROBOT_ACTIONS


def test_every_verb_the_refusal_attributes_to_the_simulation_tool_is_published_by_it():
    """Two readings of one repository may not disagree about who owns a verb.

    The claim under test is the refusal's own text, not a list kept beside it:
    the verbs are read out of the sentences that attribute them, so a new false
    attribution fails here rather than being carried to an agent. ``teleoperate``
    is asserted absent because it is the one the earlier telling of this fix got
    wrong - it is a Python method of the teleop mixin, not an action of either
    tool.
    """
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS

    attributed: set[str] = set()
    for action in sorted(HwRobot._ELSEWHERE_ACTIONS):
        text = _call(_hw(), action)
        for group in re.findall(r"([a-z_/]+) are the simulation tool's", text):
            attributed.update(group.split("/"))
    assert attributed == {"start_recording", "stop_recording", "run_policy", "start_policy", "stop_policy"}
    assert attributed <= set(_PUBLISHED_ACTIONS), (
        f"attributed to the simulation tool but not published by it: {attributed - set(_PUBLISHED_ACTIONS)}"
    )
    assert "teleoperate" not in _PUBLISHED_ACTIONS


def test_building_the_refusal_cannot_raise_on_the_action_it_reports():
    """A Python caller of ``stream`` supplies the action, and rendering can raise.

    The tool path decodes JSON, so it can only send a renderable value; a Python
    caller is under no such restriction, and the refusal is the only channel this
    door answers on. An object whose ``__str__`` raises must still be refused.
    """

    class _Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    text = _call(_hw(), _Unprintable())
    assert text.endswith("Valid actions: execute, start, status, stop")
    assert "Unknown action: <" in text


def test_an_unknown_spelling_gets_the_plain_refusal():
    text = _call(_hw(), "bogus")
    assert text == "Unknown action: bogus. Valid actions: execute, start, status, stop"


def test_the_refusal_is_one_content_block_of_text():
    hw = _hw()

    async def run():
        events = [
            e async for e in hw.stream({"toolUseId": "t", "name": "so101", "input": {"action": "teleoperate"}}, {})
        ]
        final = events[-1]
        assert isinstance(final, ToolResultEvent)
        return final.tool_result

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
