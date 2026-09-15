"""The sim tool description tells the agent about the world it is joining.

``Robot("so101")`` creates the world and adds the robot before the agent sees
the tool. The description used to be a constant that said the session starts
with ``create_world``; an agent following it had its first call refused in
every such session. These tests pin the description to the live world.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot
from strands_robots.simulation import Simulation


@pytest.fixture
def ready_arm():
    sim = Robot("so101", mode="sim")
    try:
        yield sim
    finally:
        sim.destroy()


def test_description_with_no_world_still_starts_at_create_world() -> None:
    sim = Simulation(tool_name="empty_sim")
    try:
        description = sim.tool_spec["description"]
        assert "starting with create_world" in description
        assert "ALREADY CREATED" not in description
    finally:
        sim.destroy()


def test_description_after_robot_factory_names_the_loaded_robot_and_its_joints(ready_arm) -> None:
    description = ready_arm.tool_spec["description"]
    assert "ALREADY CREATED" in description
    assert "'so101' (6 joints: 1, 2, 3, 4, 5, 6)" in description
    assert "do not call create_world" in description
    assert "starting with create_world" not in description


def test_description_points_at_actions_that_build_on_a_ready_world(ready_arm) -> None:
    description = ready_arm.tool_spec["description"]
    for action in ("get_robot_state", "set_joint_positions", "move_to", "step", "render", "reset"):
        assert action in description.split("Actions (")[0], action


def test_description_follows_the_world_when_it_is_destroyed(ready_arm) -> None:
    assert "ALREADY CREATED" in ready_arm.tool_spec["description"]
    ready_arm.destroy()
    assert "starting with create_world" in ready_arm.tool_spec["description"]


def test_long_joint_lists_are_truncated_not_dumped() -> None:
    sim = Robot("unitree_g1", mode="sim")
    try:
        description = sim.tool_spec["description"]
        assert "ALREADY CREATED" in description
        assert "..." in description.split("Scene mutations")[0]
        # 8 shown joints + ellipsis: the description must stay short on the hot path
        assert len(description.split("Scene mutations")[0]) < 700
    finally:
        sim.destroy()
