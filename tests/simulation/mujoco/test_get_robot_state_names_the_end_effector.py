"""``get_robot_state`` names the end-effector frame ``move_to`` drives, and where it is.

Measured with an agent on the bundled so100 (deepdive D-047): the model read
``get_body_state`` of the jaw body, asked ``move_to`` for a point 10 cm above
*that*, and got ``unreachable`` in 2 of 2 runs - because ``move_to`` drives
``so100/Wrist_Pitch_Roll``, a frame named nowhere the model could read before
acting. The state text now carries the frame and its world position, and the
``json`` payload an ``end_effector`` entry, so turn one can target the right frame.
"""

from __future__ import annotations

import importlib.util

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

_ROBOT_XML = """
<mujoco model="ee_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <body name="link1" pos="0 0 0.2">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="gripper" pos="0 0 0.2">
          <geom type="box" size="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_sim(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "ee_arm.xml"
    path.write_text(_ROBOT_XML)
    sim = Simulation()
    sim.create_world(timestep=0.002)
    assert sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])["status"] == "success"
    yield sim
    sim.destroy()


@requires_mujoco
class TestGetRobotStateNamesTheEndEffector:
    def test_text_names_the_frame_move_to_drives(self, arm_sim) -> None:
        text = arm_sim.get_robot_state("arm")["content"][0]["text"]
        assert "end_effector" in text and "arm/gripper" in text and "move_to" in text

    def test_json_carries_the_frame_and_its_world_position(self, arm_sim) -> None:
        res = arm_sim.get_robot_state("arm")
        ee = res["content"][1]["json"]["end_effector"]
        assert ee["name"] == "arm/gripper" and ee["type"] == "body"
        body = arm_sim.get_body_state(body_name="arm/gripper")["content"][1]["json"]["position"]
        assert ee["position"] == pytest.approx(body, abs=1e-9)
        # The reported position is a usable move_to target for the same frame.
        assert arm_sim.move_to(robot_name="arm", position=ee["position"], tol=0.02)["status"] == "success"

    def test_joint_state_contract_is_unchanged(self, arm_sim) -> None:
        state = arm_sim.get_robot_state("arm")["content"][1]["json"]["state"]
        assert set(state) == {"shoulder", "elbow"}
