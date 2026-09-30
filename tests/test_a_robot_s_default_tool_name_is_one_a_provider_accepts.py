"""``Robot(name)``'s default tool name is one a model provider accepts.

The robot name is looked up whitespace- and case-tolerantly, so
``Robot(" so100 ")`` builds an so100. The default tool name was built from the
raw string, ``" so100 _sim"``; it registered with ``Agent(tools=[...])`` and the
first Bedrock call failed with ``ValidationException: Value ' so100 _sim' at
'toolConfig.tools.1.member.toolSpec.name' failed to satisfy constraint ...
[a-zA-Z0-9_-]+``. An explicit ``tool_name=`` was already refused up front; a
default is the library's own to make valid.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot  # noqa: E402
from strands_robots.robot import _default_sim_tool_name  # noqa: E402

_PROVIDER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}\Z")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("so100", "so100_sim"),
        ("SO100", "SO100_sim"),
        (" so100 ", "so100_sim"),
        ("so100\n", "so100_sim"),
        ("my arm v2", "my_arm_v2_sim"),
        ("lab/left.arm", "lab_left_arm_sim"),
        ("x" * 100, "x" * 60 + "_sim"),
        ("---", "---_sim"),
        ("  ", "robot_sim"),
    ],
)
def test_the_default_is_the_name_made_valid(name, expected):
    tool_name = _default_sim_tool_name(name)

    assert tool_name == expected
    assert _PROVIDER_PATTERN.match(tool_name)


def test_a_padded_robot_name_registers_a_valid_tool():
    sim = Robot(" so100 ")
    try:
        assert sim.tool_name == "so100_sim"
    finally:
        sim.cleanup()


def test_an_explicit_tool_name_is_still_the_callers_to_fix():
    with pytest.raises(ValueError, match="is not a valid tool name"):
        Robot("so100", tool_name="left arm")
