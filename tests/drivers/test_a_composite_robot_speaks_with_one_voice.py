"""Two simulated parts and a gripper pair answer as one robot.

The design (``docs/project/design-driver-composition.md``) promises one state
dict, one action dict, one ``stop()``, one refusal when any part cannot move, and
a latch after a partial halt. Each promise is one cell here, on a MuJoCo G1 body
plus two mock grippers, so the sentences a dashboard will quote are pinned
before any CAN or DDS code exists.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from strands.types.tools import ToolUse

from strands_robots import Robot
from strands_robots.drivers.composite import VERBS, CompositeDriver, MockGripperPart, SimPart, part_key
from strands_robots.teleop import G1_SIM_JOINTS, teleop_layout

#: Physics steps after a write before the body is read back: 20 x 2 ms = 40 ms.
SETTLE_STEPS = 20
#: Mock gripper settle time, seconds.
GRIPPER_SETTLE_S = 0.02


@pytest.fixture
def g1() -> Any:
    """A MuJoCo G1, torn down after the test."""
    robot = Robot("g1")
    try:
        yield robot
    finally:
        robot.cleanup()


def _composite(g1: Any, **gripper_kwargs: Any) -> tuple[CompositeDriver, MockGripperPart, MockGripperPart]:
    left = MockGripperPart("left_gripper", settle_s=GRIPPER_SETTLE_S)
    right = MockGripperPart("right_gripper", settle_s=GRIPPER_SETTLE_S, **gripper_kwargs)
    driver = CompositeDriver(
        "g1_openarm",
        parts=[SimPart(g1, "g1", "body"), left, right],
        primary="body",
        stop_order=["body", "left_gripper", "right_gripper"],
    )
    return driver, left, right


def test_the_key_space_is_the_body_then_the_grippers_by_part_name(g1: Any) -> None:
    """29 unprefixed body joints, then ``left_gripper`` and ``right_gripper``: the blog's 31-D state."""
    driver, _left, _right = _composite(g1)
    assert driver.joint_names == G1_SIM_JOINTS + ("left_gripper", "right_gripper")
    assert list(driver.observation_features) == list(driver.action_features) == list(driver.joint_names)
    units = driver.units()
    assert units["left_hip_pitch_joint"] == "rad"
    assert units["left_gripper"] == "norm01"
    assert part_key("head", "pan", primary=False) == "head_pan"
    assert part_key("left_gripper", "gripper", primary=False) == "left_gripper"
    assert part_key("body", "gripper", primary=True) == "gripper"
    # The 31 composite keys line up with the blog layout's 31 state columns, position for position.
    assert len(teleop_layout("blog_31_66").state_names) == len(driver.joint_names)


def test_one_send_action_reaches_every_part_and_one_read_returns_them_all(g1: Any) -> None:
    driver, left, right = _composite(g1)
    action = dict.fromkeys(G1_SIM_JOINTS, 0.0)
    action.update({"left_shoulder_pitch_joint": 0.3, "left_gripper": 0.3, "right_gripper": 0.7})

    result = driver.send_action(action)
    assert result["status"] == "success", result
    assert set(result["content"][1]["json"]) == {"body", "left_gripper", "right_gripper"}
    assert left.writes == right.writes == 1

    g1.step(SETTLE_STEPS)
    time.sleep(GRIPPER_SETTLE_S * 2)
    observation = driver.get_observation()
    assert observation["left_gripper"] == pytest.approx(0.3)
    assert observation["right_gripper"] == pytest.approx(0.7)
    assert observation["left_shoulder_pitch_joint"] > 0.05
    assert observation["parts"]["body"] == {"connected": True, "joints": 29}
    assert set(observation) == set(driver.joint_names) | {"parts"}


def test_a_key_no_part_owns_refuses_the_whole_write(g1: Any) -> None:
    driver, left, right = _composite(g1)
    result = driver.send_action({"left_grip": 0.3, "left_gripper": 0.4})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "['left_grip']" in text and "nothing was written" in text
    assert "['body', 'left_gripper', 'right_gripper']" in text
    assert left.writes == right.writes == 0

    assert driver.send_action({})["status"] == "error"
    other = driver.send_action({"left_gripper": 0.4}, robot_name="somebody_else")
    assert other["status"] == "error" and "one robot" in other["content"][0]["text"]


def test_a_partial_stop_latches_and_reset_estop_clears_it(g1: Any) -> None:
    driver, left, right = _composite(g1, fail_stop=True)
    halted = driver.stop(reason="test")
    assert halted["status"] == "error"
    assert halted["content"][0]["text"].startswith("stopped 2/3 parts: right_gripper: no halt acknowledgement")
    assert list(halted["content"][1]["json"]) == ["body", "left_gripper", "right_gripper"]
    assert driver.estop_latched
    assert left.stops == right.stops == 1

    refused = driver.send_action({"left_gripper": 0.1})
    assert refused["status"] == "error"
    assert refused["content"][0]["text"] == (
        "g1_openarm: e-stop latched by right_gripper; call reset_estop() after checking the robot."
    )
    assert left.writes == 0

    cleared = driver.reset_estop()
    assert cleared["status"] == "success"
    assert not driver.estop_latched
    assert driver.send_action({"left_gripper": 0.1})["status"] == "success"
    assert driver.reset_estop()["content"][0]["text"].endswith("no e-stop latched")


def test_a_failed_write_stops_every_part_and_refuses(g1: Any) -> None:
    driver, left, right = _composite(g1, fail_write=True)
    result = driver.send_action({"left_gripper": 0.2, "right_gripper": 0.2})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "part 'right_gripper' refused the write" in text
    assert "every part was stopped (stopped 3/3 parts)" in text
    assert left.writes == 1 and left.stops == 1 and right.stops == 1
    assert not driver.estop_latched


def test_a_disconnected_part_refuses_motion_but_reads_still_answer(g1: Any) -> None:
    driver, left, right = _composite(g1)
    right.disconnect()
    assert not driver.is_connected
    refused = driver.send_action({"left_gripper": 0.1})
    assert refused["content"][0]["text"] == (
        "g1_openarm: part(s) ['right_gripper'] not connected; the whole robot refuses motion."
    )
    assert left.writes == 0

    observation = driver.get_observation()
    assert "right_gripper" not in observation
    assert "left_gripper" in observation and "left_hip_pitch_joint" in observation
    assert observation["parts"]["right_gripper"] == {"connected": False, "joints": 1}

    driver.stop()
    assert driver.reset_estop()["status"] == "success"  # nothing latched: a clean stop
    driver.stop()


def test_the_build_refuses_a_bad_primary_or_stop_order_or_a_key_collision(g1: Any) -> None:
    body = SimPart(g1, "g1", "body")
    with pytest.raises(ValueError, match="primary 'torso' is not one of the parts"):
        CompositeDriver("x", parts=[body, MockGripperPart("left_gripper")], primary="torso")
    with pytest.raises(ValueError, match="stop_order names parts that do not exist"):
        CompositeDriver("x", parts=[body], primary="body", stop_order=["head"])
    with pytest.raises(ValueError, match="duplicate part names"):
        CompositeDriver("x", parts=[body, MockGripperPart("body")], primary="body")
    # A non-primary part named like a primary joint collides after prefixing rules.
    with pytest.raises(ValueError, match="is produced by both"):
        CompositeDriver(
            "x",
            parts=[body, MockGripperPart("left_elbow_joint", settle_s=0)],
            primary="body",
        )
    # A stop_order that names only some parts still stops all of them.
    driver = CompositeDriver(
        "x", parts=[body, MockGripperPart("left_gripper")], primary="body", stop_order=["left_gripper"]
    )
    assert driver.stop_order == ("left_gripper", "body")


def test_the_agent_verbs_dispatch_and_an_unknown_verb_is_refused(g1: Any) -> None:
    driver, _left, _right = _composite(g1)

    async def call(action: str, **payload: Any) -> dict[str, Any]:
        events = []
        tool_use: ToolUse = {"toolUseId": "t", "name": driver.tool_name, "input": {"action": action, **payload}}
        async for event in driver.stream(tool_use, {}):
            events.append(event)
        assert len(events) == 1
        return events[0]

    assert set(driver.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]) == set(VERBS)
    status = asyncio.run(call("status"))
    assert status["status"] == "success"
    assert status["content"][0]["text"] == "g1_openarm: 3 parts (3 connected), 31 joints, e-stop latched: no"
    features = asyncio.run(call("features"))
    assert features["content"][0]["json"]["right_gripper"] == "norm01"
    written = asyncio.run(call("send_action", targets={"left_gripper": 0.5}))
    assert written["status"] == "success"
    unknown = asyncio.run(call("jump"))
    assert unknown["status"] == "error"
    assert "unknown action 'jump'" in unknown["content"][0]["text"]
    assert asyncio.run(call("stop"))["status"] == "success"
