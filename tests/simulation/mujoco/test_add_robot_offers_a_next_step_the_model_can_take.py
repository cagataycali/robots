"""``add_robot``'s closing line offers a step the robot it added can take.

The summary has always ended with ``Run policy: action='run_policy',
robot_name=...``. On a model that compiles with no actuators that invitation is
a dead end: ``send_action`` has no key to resolve, so a bound policy can only
emit an empty action, and both consumers of one already refuse that condition
(the all-unresolved abort in ``PolicyRunner.run``, ``uncommanded_eval_error``
for ``eval_policy``) -- but only after a whole rollout has run. Pinned here:

* an actuator-less model is NOT invited to ``run_policy``, and is pointed at
  ``actuate_robot``, the remedy that gives it actuators,
* an actuated model's invitation is unchanged and mentions no remedy,
* the offered remedy is real -- dialling it on the refused robot yields
  actuators and turns the same summary line into the ``run_policy`` invitation.

A bare URDF arm is the fixture because URDF has no actuator concept, so it
loads with ``nu == 0`` by construction and needs no downloaded asset.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

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

# The same one-hinge robot as MJCF, which CAN declare the position servo.
ACTUATED_MJCF = """<mujoco model="stick">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.025 0.025 0.025"/>
      <body name="arm">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="box" size="0.015 0.015 0.1" pos="0 0 0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="10"/></actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_robot_next_step", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _add(sim, tmp_path, body, suffix):
    path = tmp_path / f"stick{suffix}"
    path.write_text(body)
    result = sim.add_robot(name="stick", urdf_path=str(path))
    assert result["status"] == "success", result
    return result["content"][0]["text"]


@pytest.mark.parametrize(
    ("body", "suffix", "actuated"),
    [(UNACTUATED_URDF, ".urdf", False), (ACTUATED_MJCF, ".xml", True)],
    ids=["no-actuator", "actuated"],
)
def test_run_policy_is_offered_only_to_a_model_that_can_be_commanded(sim, tmp_path, body, suffix, actuated):
    summary = _add(sim, tmp_path, body, suffix)

    # Premise: the two fixtures really differ on the axis under test, and the
    # reported count agrees with the resolved ownership the line is derived
    # from -- so the assertions below are about actuators, not about MJCF vs URDF.
    owned = len(sim._world.robots["stick"].actuator_ids)
    assert (owned > 0) is actuated, f"fixture {suffix} owns {owned} actuator(s)"
    assert f"Actuators: {owned}\n" in summary

    offers_policy = "action='run_policy'" in summary
    offers_remedy = "action='actuate_robot'" in summary
    assert offers_policy is actuated, summary
    assert offers_remedy is not actuated, summary
    # Exactly one next step, so the caller is never handed both.
    assert offers_policy != offers_remedy, summary


def test_the_remedy_offered_to_an_unactuated_model_makes_it_commandable(sim, tmp_path):
    """The refused robot's offered call is dialled, and the refusal lifts."""
    refused = _add(sim, tmp_path, UNACTUATED_URDF, ".urdf")
    assert "action='actuate_robot', robot_name='stick'" in refused

    result = sim.actuate_robot(robot_name="stick")
    assert result["status"] == "success", result
    assert len(sim._world.robots["stick"].actuator_ids) > 0

    # Re-adding the same model is not how a caller lifts this, so re-derive the
    # line from the now-actuated robot the way the summary does.
    lifted = sim._next_step_after_add("stick", sim._world.robots["stick"])
    assert lifted == "Run policy: action='run_policy', robot_name='stick'"
