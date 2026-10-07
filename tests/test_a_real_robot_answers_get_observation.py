"""``Robot(name, mode="real")`` answers ``get_observation()``, as ``mode="sim"`` does.

A loop written against the simulation (``obs = robot.get_observation()``) used
to raise ``AttributeError`` the moment ``mode="real"`` was set, on both real
paths: the lerobot wrapper held a device that could answer but did not forward
it, and the native Feetech driver had no read verb at all. Both now answer in
lerobot's shape, ``{"<motor>.pos": value, ...}``, and a port that will not open
raises rather than reading as an arm with no joints.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots import Robot
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
from tests._hardware_robot import hardware_robot_on
from tests.drivers.conftest import FakeServoPort

_SO_ARM_KEYS = {f"{motor}.pos" for motor in SO_ARM_MOTORS}


class _LeRobotArm:
    """The slice of a lerobot robot the wrapper reads: connect, then observe."""

    def __init__(self) -> None:
        self.is_connected = False
        self.connect_calls: list[bool] = []

    def connect(self, calibrate: bool = True) -> None:
        self.connect_calls.append(calibrate)
        self.is_connected = True

    def get_observation(self) -> dict[str, Any]:
        assert self.is_connected, "lerobot refuses a read before connect()"
        return {f"{motor}.pos": 0.0 for motor in SO_ARM_MOTORS}


def _strands_driver() -> Any:
    robot: Any = Robot("so101", mode="real", driver="strands", port="/dev/fake")
    robot.bus._conn = FakeServoPort(dict.fromkeys((1, 2, 3, 4, 5, 6), 2048))
    return robot


def _lerobot_wrapper() -> Any:
    return hardware_robot_on(_LeRobotArm(), tool_name="so101")


@pytest.mark.parametrize("build", [_strands_driver, _lerobot_wrapper], ids=["strands", "lerobot"])
def test_both_real_paths_answer_the_sim_call(build: Any) -> None:
    robot = build()
    try:
        assert set(robot.get_observation()) == _SO_ARM_KEYS
    finally:
        robot.cleanup()


def test_the_wrapper_connects_on_first_read_without_calibrating() -> None:
    arm = _LeRobotArm()
    robot = hardware_robot_on(arm, tool_name="so101")
    try:
        robot.get_observation()
        robot.get_observation()
    finally:
        robot.cleanup()
    assert arm.connect_calls == [False], "connected once, calibrate=False, as send_action does"


def test_a_port_that_will_not_open_raises_instead_of_reading_empty(tmp_path: Any) -> None:
    robot: Any = Robot("so101", mode="real", driver="strands", port=str(tmp_path / "ttyACM9"))
    try:
        with pytest.raises(Exception, match="ttyACM9"):
            robot.get_observation()
    finally:
        robot.cleanup()
