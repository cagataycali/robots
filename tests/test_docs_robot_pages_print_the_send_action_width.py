"""A generated robot page prints the list a numeric ``send_action`` is sized by.

Each simulated robot page in ``docs/robots/`` opens with a fence that builds the
robot and prints one list of names. A reader sizes their first action vector by
that list, so its length must be the width ``send_action`` accepts. The joint
roster is wider on a floating base (the free joint), a gripper with mimic
fingers and a tendon hand; it is narrower on a quadrotor commanded by rotors.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

from tests._docs_hooks import docs_hook

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None

#: One robot per way the joint roster and the actuators disagree, plus an arm
#: where they agree.
_ROBOTS = (
    "microduck",  # floating base: trunk_base_freejoint has no scalar column
    "cassie",  # free joint plus passive leg joints
    "panda",  # mimic finger joint
    "crazyflie",  # no joints, four rotor actuators
    "so101",  # joints and actuators agree
)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="the fence runs against a MuJoCo sim robot")
@pytest.mark.parametrize("name", _ROBOTS)
def test_the_printed_list_is_as_wide_as_send_action(name: str) -> None:
    from strands_robots import Robot

    page = docs_hook("robot_pages").robot_page(name)
    printed = re.search(rf'print\(robot\.(\w+)\("{name}"\)\)', page)
    assert printed, f"the {name} page prints no robot list"

    robot = Robot(name)
    try:
        names = getattr(robot, printed.group(1))(name)
        result = robot.send_action(action=[0.0] * len(names))
    finally:
        robot.cleanup()

    assert result["status"] == "success", (printed.group(1), len(names), result["content"])
