"""``rotate_wrist(target_yaw=...)`` drives the wrist's yaw joint when the robot has one.

The joint picker tried ``wrist_roll`` before ``wrist_yaw``, so on a wrist with
separate roll, pitch and yaw joints the yaw call turned the roll joint about a
different axis and left the yaw joint idle. Each row is a bundled robot whose
wrist has both; the yaw joint must be the one that moves.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


@pytest.mark.parametrize(
    ("data_config", "yaw_joint", "roll_joint"),
    [
        ("unitree_g1", "right_wrist_yaw_joint", "right_wrist_roll_joint"),
        ("stretch3", "joint_wrist_yaw", "joint_wrist_roll"),
    ],
)
def test_the_yaw_joint_moves_and_the_roll_joint_holds(data_config: str, yaw_joint: str, roll_joint: str) -> None:
    from strands_robots.simulation import Simulation

    sim = Simulation()
    try:
        sim.create_world(timestep=0.002)
        assert sim.add_robot("r", data_config=data_config)["status"] == "success"
        before = sim.get_observation("r")

        result = sim.rotate_wrist(robot_name="r", target_yaw=0.3, max_steps=200)

        payload = result["content"][1]["json"]
        assert payload["wrist_joint"] == yaw_joint, payload
        after = sim.get_observation("r")
        assert abs(after[yaw_joint] - before[yaw_joint]) > 0.1, (before[yaw_joint], after[yaw_joint])
        assert abs(after[roll_joint] - before[roll_joint]) < 0.05, (before[roll_joint], after[roll_joint])
    finally:
        sim.cleanup()
