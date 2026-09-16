"""``Robot(tool_name=...)`` - two robots of one type can share an Agent."""

import pytest

from strands_robots.robot import _TOOL_NAME_MAX_LENGTH, Robot, _tool_name_error

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


def test_a_name_over_the_length_ceiling_is_refused_with_both_numbers():
    # Length is the second rule Strands validates and the pattern says nothing
    # about it, so an over-long name of legal characters used to reach the model
    # and fail on its first call instead.
    assert _tool_name_error("a" * _TOOL_NAME_MAX_LENGTH) is None
    refusal = _tool_name_error("a" * (_TOOL_NAME_MAX_LENGTH + 1)) or ""
    assert f"is {_TOOL_NAME_MAX_LENGTH + 1} characters" in refusal
    assert f"at most {_TOOL_NAME_MAX_LENGTH}" in refusal
    with pytest.raises(ValueError, match="characters"):
        Robot("so101", mode="sim", tool_name="a" * (_TOOL_NAME_MAX_LENGTH + 1))


def test_the_length_ceiling_is_the_one_strands_enforces():
    # Strands states the ceiling inline instead of exporting it, so pin it by
    # probing the validator: this fails the day Strands moves the number.
    from strands.tools.tools import InvalidToolUseNameException, validate_tool_use_name

    at_ceiling = {"name": "a" * _TOOL_NAME_MAX_LENGTH, "toolUseId": "1", "input": {}}
    validate_tool_use_name(at_ceiling)
    with pytest.raises(InvalidToolUseNameException):
        validate_tool_use_name({**at_ceiling, "name": "a" * (_TOOL_NAME_MAX_LENGTH + 1)})


def test_real_mode_hardware_robot_carries_the_tool_name():
    # Construction opens no port (the bus is touched on first connect), so a
    # bogus path is enough to reach the tool-name plumbing.
    arm = Robot("so101", mode="real", port="/dev/does-not-exist", tool_name="real_left")
    assert arm.tool_name == "real_left"
