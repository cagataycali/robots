"""``get_robot_state`` says where the end effector is FROM THE BASE, and which way the arm extends.

Measured with an agent on a fresh so101 (v0.5.2 devx replay, scenario s06):
the state read ``end_effector … pos=[0.02, -0.38, 0.26]`` and nothing else
said which axis is the robot's front, so the model read "in front of the
base" as +X and placed the cube at ``[0.2, 0, 0.02]`` - beside the arm, whose
whole reach lies along -Y. Only the arm's spare reach saved the move. The
end-effector line now carries the base position, the EE offset from it and
the horizontal axis that offset lies along, and the ``json`` payload the same
three facts, so "in front of" means the same thing to the agent and the person.
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.ik import REACH_AXIS_MIN_M, reach_axis, reach_axis_label

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


class TestReachAxis:
    @pytest.mark.parametrize(
        ("offset", "axis"),
        [
            ((0.02, -0.38, 0.26), "-Y"),
            ((0.0, 0.3, 0.1), "+Y"),
            ((0.4, 0.1, 0.0), "+X"),
            ((-0.25, -0.2, 0.5), "-X"),
            ((0.3, 0.3, 0.0), "+X"),  # a tie goes to X, deterministically
        ],
    )
    def test_the_dominant_horizontal_component_names_the_axis(self, offset, axis) -> None:
        assert reach_axis(offset) == axis
        assert reach_axis_label(axis) == f"the arm currently extends along {axis}"

    @pytest.mark.parametrize("offset", [(0.0, 0.0, 0.4), (0.01, -0.01, 0.2), (REACH_AXIS_MIN_M * 0.7, 0.0, 0.0)])
    def test_a_pose_over_the_base_is_not_given_an_axis(self, offset) -> None:
        assert reach_axis(offset) is None
        assert reach_axis_label(None) == "the arm is currently over its base"

    def test_the_threshold_is_the_horizontal_norm(self) -> None:
        d = REACH_AXIS_MIN_M / 2**0.5
        assert reach_axis((d * 1.01, d * 1.01, 0.0)) == "+X"
        assert reach_axis((d * 0.99, d * 0.99, 5.0)) is None


_ARM_XML = """
<mujoco model="reach_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-3 3"/>
      <body name="link1" pos="0.3 0 0">
        <geom type="capsule" size="0.02" fromto="-0.3 0 0 0 0 0"/>
        <body name="gripper" pos="0.1 0 0">
          <geom type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="shoulder_act" joint="shoulder" kp="50"/></actuator>
</mujoco>
"""


@pytest.fixture
def arm_at(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "reach_arm.xml"
    path.write_text(_ARM_XML)
    sims = []

    def make(position):
        sim = Simulation()
        sim.create_world(timestep=0.002)
        assert sim.add_robot("arm", urdf_path=str(path), position=position)["status"] == "success"
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.destroy()


@requires_mujoco
class TestGetRobotStateNamesTheReachAxis:
    def test_text_carries_base_offset_and_axis(self, arm_at) -> None:
        text = arm_at([0.0, 0.0, 0.0]).get_robot_state("arm")["content"][0]["text"]
        line = next(ln for ln in text.splitlines() if ln.startswith("end_effector"))
        assert "from base [0.0000, 0.0000, 0.0000]: [+0.4000, +0.0000, +0.1000]" in line
        assert "(the arm currently extends along +X)" in line

    def test_the_offset_is_measured_from_the_spawn_pose_of_a_fixed_base(self, arm_at) -> None:
        res = arm_at([1.0, -2.0, 0.0]).get_robot_state("arm")
        ee = res["content"][1]["json"]["end_effector"]
        assert ee["base"] == [1.0, -2.0, 0.0]
        assert ee["position"] == pytest.approx([1.4, -2.0, 0.1], abs=1e-6)
        assert ee["from_base"] == pytest.approx([0.4, 0.0, 0.1], abs=1e-6)
        assert ee["extends_along"] == "+X"

    def test_the_axis_follows_the_arm_as_it_moves(self, arm_at) -> None:
        import math

        sim = arm_at([0.0, 0.0, 0.0])
        # Swing the shoulder 90 deg: the same arm now points along +Y.
        sim.send_action({"shoulder_act": math.pi / 2}, robot_name="arm", n_substeps=400)
        ee = sim.get_robot_state("arm")["content"][1]["json"]["end_effector"]
        assert ee["extends_along"] == "+Y"
        assert ee["from_base"][1] == pytest.approx(0.4, abs=0.05)  # servo still settling; the axis is the point

    def test_the_offset_plus_base_is_a_usable_move_to_target(self, arm_at) -> None:
        sim = arm_at([0.5, 0.5, 0.0])
        ee = sim.get_robot_state("arm")["content"][1]["json"]["end_effector"]
        target = [ee["base"][i] + ee["from_base"][i] for i in range(3)]
        assert target == pytest.approx(ee["position"], abs=1e-9)


@requires_mujoco
class TestOnTheBundledArms:
    @pytest.mark.parametrize("data_config", ["so100", "so101"])
    def test_the_so_arms_report_their_reach_along_minus_y(self, data_config) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("arm", data_config=data_config)["status"] == "success"
            res = sim.get_robot_state("arm")
            assert "(the arm currently extends along -Y)" in res["content"][0]["text"]
            assert res["content"][1]["json"]["end_effector"]["extends_along"] == "-Y"
        finally:
            sim.destroy()
