"""``Robot(tool_name=...)`` - two robots of one type can share an Agent."""

import pytest

from strands_robots.robot import Robot, _tool_name_error

pytest.importorskip("mujoco")


def test_two_same_type_robots_register_under_their_own_names():
    from strands import Agent

    left = Robot("so101", mode="sim", tool_name="left_arm")
    right = Robot("so101", mode="sim", tool_name="right_arm")
    try:
        agent = Agent(tools=[left, right])
        assert {"left_arm", "right_arm"} <= set(agent.tool_registry.registry)
        assert left.tool_name == "left_arm"
        assert right.tool_name == "right_arm"
    finally:
        left.destroy()
        right.destroy()


def test_default_tool_name_is_unchanged():
    arm = Robot("so101", mode="sim")
    try:
        assert arm.tool_name == "so101_sim"
    finally:
        arm.destroy()


@pytest.mark.parametrize("bad", ["", "left arm", "a/b", 3])
def test_invalid_tool_name_is_refused_before_the_backend_builds(bad):
    with pytest.raises(ValueError, match="not a valid tool name"):
        Robot("so101", mode="sim", tool_name=bad)


def test_tool_name_error_accepts_none_and_tokens():
    assert _tool_name_error(None) is None
    assert _tool_name_error("left-arm_2") is None
    assert "tool_name='x y'" in (_tool_name_error("x y") or "")


def test_real_mode_hardware_robot_carries_the_tool_name():
    # Construction opens no port (the bus is touched on first connect), so a
    # bogus path is enough to reach the tool-name plumbing.
    arm = Robot("so101", mode="real", port="/dev/does-not-exist", tool_name="real_left")
    assert arm.tool_name == "real_left"
