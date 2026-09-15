"""The sim tool description tells the agent about the world it is joining.

``Robot("so101")`` creates the world and adds the robot before the agent sees
the tool. The description used to be a constant that said the session starts
with ``create_world``; an agent following it had its first call refused in
every such session. These tests pin the description to the live world.

The offered actions follow each robot's resolved actuator ownership, so a
session holding a model that compiles with no actuator is pointed at
``actuate_robot`` rather than at ``move_to``, which refuses such a robot
outright. A bare URDF is the fixture for that case because URDF has no
actuator concept, so it loads with ``nu == 0`` by construction and needs no
downloaded asset.
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


# URDF has no actuator element, so this compiles with nu == 0.
UNACTUATED_URDF = """<?xml version="1.0"?>
<robot name="stick">
  <link name="base">
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="0.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><geometry><box size="0.03 0.03 0.2"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
</robot>
"""

DRIVE_VERBS_OFFERED = "set_joint_positions, move_to, step or render"
REMEDY_OFFERED = "add a position servo per joint with actuate_robot"


def _prefix(sim) -> str:
    """The readiness sentence alone, before the constant part of the description."""
    return sim.tool_spec["description"].split("Scene mutations")[0]


@pytest.fixture
def unactuated_arm(tmp_path):
    """A world holding one robot that compiles with no actuator."""
    sim = Simulation(tool_name="unactuated_world", mesh=False)
    sim.create_world()
    path = tmp_path / "stick.urdf"
    path.write_text(UNACTUATED_URDF)
    assert sim.add_robot(name="stick", urdf_path=str(path))["status"] == "success"
    assert not sim._world.robots["stick"].actuator_ids, "fixture must own no actuator"
    try:
        yield sim
    finally:
        sim.destroy()


def test_a_robot_that_cannot_be_driven_is_offered_actuate_robot_not_a_drive_verb(unactuated_arm) -> None:
    """``move_to`` refuses a robot with no actuator, so it is not the way in."""
    prefix = _prefix(unactuated_arm)
    assert "[no actuators]" in prefix
    assert REMEDY_OFFERED in prefix
    assert DRIVE_VERBS_OFFERED not in prefix
    # The refusal the description would otherwise have walked into.
    refused = unactuated_arm.move_to(robot_name="stick", position=[0.1, 0.0, 0.2])
    assert refused["status"] == "error"
    assert "actuate_robot" in refused["content"][0]["text"]


def test_the_description_follows_the_remedy_it_offered(unactuated_arm) -> None:
    """Dialling the offered call makes the robot drivable, and the sentence says so."""
    assert REMEDY_OFFERED in _prefix(unactuated_arm)

    assert unactuated_arm.actuate_robot(robot_name="stick")["status"] == "success"
    assert unactuated_arm._world.robots["stick"].actuator_ids

    prefix = _prefix(unactuated_arm)
    assert DRIVE_VERBS_OFFERED in prefix
    assert REMEDY_OFFERED not in prefix
    assert "[no actuators]" not in prefix


def test_a_mixed_world_keeps_the_drive_verbs_and_flags_the_robot_that_has_none(unactuated_arm, tmp_path) -> None:
    """One actuated robot makes the drive verbs valid; the other is still flagged."""
    assert unactuated_arm.add_robot(name="so101", data_config="so101")["status"] == "success"
    owned = {name: len(robot.actuator_ids) for name, robot in unactuated_arm._world.robots.items()}
    assert owned["stick"] == 0 and owned["so101"] > 0, owned

    prefix = _prefix(unactuated_arm)
    assert DRIVE_VERBS_OFFERED in prefix
    assert REMEDY_OFFERED not in prefix
    assert "'stick' (1 joints: shoulder) [no actuators]" in prefix
    assert "[no actuators]" not in prefix.split("'so101'")[1].split(";")[0]
