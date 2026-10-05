"""``Robot(tool_name=...)`` - two robots of one type can share an Agent."""

import pytest

from strands_robots.robot import Robot

pytest.importorskip("mujoco")

# What a model provider accepts as a tool name, spelled out here so the table
# below reads as the rule rather than as the implementation: Bedrock's Converse
# API requires ``toolSpec.name`` to match ``[a-zA-Z0-9_-]+`` and to be at most
# 64 characters. The local Strands tool registry validates neither.
MAX_LEN = 64


class Unprintable:
    """A value that cannot be rendered - a refusal still has to answer it."""

    def __repr__(self) -> str:
        raise RuntimeError("this value cannot be rendered")


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


# Both provider constraints are graded, each naming its own reason: a message
# about the character set would misdescribe a name whose only fault is length.
@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        (None, None),
        ("left-arm_2", None),
        ("x" * MAX_LEN, None),  # the limit is inclusive
        ("", "Robot(tool_name='') is not a valid tool name"),
        ("left arm", "Robot(tool_name='left arm') is not a valid tool name"),
        ("a/b", "Robot(tool_name='a/b') is not a valid tool name"),
        (3, "Robot(tool_name=3) is not a valid tool name"),
        ("x" * (MAX_LEN + 1), "is 65 characters long, over the 64-character limit"),
        # Both faults at once: the character set is reported, because shortening
        # a name with a space in it would not make it usable.
        ("left arm " * 8, "is not a valid tool name"),
        # The pattern is anchored at the end of the string, not the end of a
        # line: a trailing newline is a character the provider's pattern
        # rejects, so it cannot be admitted here.
        ("left_arm\n", "is not a valid tool name"),
        # A value whose rendering raises is still answered, not re-raised.
        pytest.param(Unprintable(), "is not a valid tool name", id="unrenderable-object"),
        pytest.param(10**5000, "is not a valid tool name", id="int-of-5001-digits"),
    ],
)
def test_a_tool_name_is_graded_against_what_a_provider_accepts(tool_name, expected):
    from strands_robots.robot import _tool_name_error

    error = _tool_name_error(tool_name)
    if expected is None:
        assert error is None
    else:
        assert error is not None
        assert expected in error


def test_the_factory_limit_is_the_providers_limit():
    from strands_robots.robot import _TOOL_NAME_MAX_LEN

    assert _TOOL_NAME_MAX_LEN == MAX_LEN


@pytest.mark.parametrize(
    "bad",
    [
        "left arm",
        "left_arm\n",
        "x" * (MAX_LEN + 1),
        pytest.param(Unprintable(), id="unrenderable-object"),
    ],
)
def test_an_invalid_tool_name_is_refused_before_the_backend_builds(bad, monkeypatch):
    import strands_robots.simulation as simulation_pkg

    def _never_build(*args, **kwargs):
        raise AssertionError("a refused tool name must not reach the backend")

    monkeypatch.setattr(simulation_pkg, "create_simulation", _never_build)
    with pytest.raises(ValueError, match="tool name"):
        Robot("so101", mode="sim", tool_name=bad)


# Every dispatch path names the robot it built by the string the caller passed,
# whatever the tool name. Native and lerobot construction open no link and no
# port (the bus is touched on first connect), so a bogus path is enough.
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("so100", {"mode": "sim"}),
        ("panda", {"mode": "real", "driver": "strands"}),
        ("so101", {"mode": "real", "port": "/dev/does-not-exist"}),
        ("so101", {"mode": "real", "driver": "lerobot", "port": "/dev/does-not-exist"}),
    ],
    ids=["sim", "native-driver", "auto-driver", "lerobot-driver"],
)
def test_every_robot_names_itself_apart_from_its_tool_name(name, kwargs):
    robot = Robot(name, tool_name="left", **kwargs)
    try:
        assert robot.tool_name == "left"
        assert robot.robot_name == name
    finally:
        if kwargs["mode"] == "sim":
            robot.destroy()


def test_a_sim_names_its_robot_in_repr_and_its_methods_take_that_name():
    sim = Robot("so100")
    try:
        assert repr(sim) == "<MuJoCoSimEngine robot='so100' tool='so100_sim'>"
        assert sim.robot_joint_names(sim.robot_name)
        sim.add_robot(name="so101")
        # Two robots, so no single name answers - the repr lists both.
        assert sim.robot_name is None
        assert "robots=['so100', 'so101']" in repr(sim)
    finally:
        sim.destroy()
