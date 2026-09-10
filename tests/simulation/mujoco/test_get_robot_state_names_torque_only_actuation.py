"""``get_robot_state`` names torque-only actuation, so a settling robot is not read as a bug.

A Menagerie quadruped like ``go2`` is driven entirely by ``<motor>`` actuators:
``ctrl`` is a force in Nm, so a pose written there is a torque and nothing holds
the robot up. It sinks from base z 0.445 to 0.20 within 2 s of the first
``step`` - which the state read reported as a falling number with no word on its
cause, indistinguishable from a broken model. The note names the cause, so the
answer is "drive it with a controller" rather than a hunt for a defect.

The classification is unanimous and three-term (see
:func:`~strands_robots.simulation.mujoco.scene_ops.torque_only_actuation`): one
position servo is enough to make a robot's pose writes hold, and ``<damper>``
clears the other two terms while its ``ctrl`` is not a torque at all.
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.assets.manager import is_robot_asset_present

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

_LEG_XML = """
<mujoco model="drive_leg">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.05" mass="2"/>
      <body name="thigh" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.4" mass="0.2"/>
        <body name="shank" pos="0 0 -0.4">
          <joint name="knee" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""

# One row per MuJoCo actuator shortcut whose compiled fields overlap a motor's on
# some term, so the table pins the boundary rather than one happy case. Measured
# (gaintype, biastype, dyntype) on mujoco 3.13 in the comment.
_DRIVES = {
    # FIXED, NONE, NONE - ctrl is the force. The reported case.
    "motor": '<motor name="hip_act" joint="hip"/><motor name="knee_act" joint="knee"/>',
    # <general> defaults to exactly a motor, so it must read as one.
    "general": '<general name="hip_act" joint="hip"/><general name="knee_act" joint="knee"/>',
    # AFFINE bias (-kp) - the setpoint restores a pose, so the pose holds.
    "position": '<position name="hip_act" joint="hip" kp="50"/><position name="knee_act" joint="knee" kp="50"/>',
    # AFFINE bias with kp slot 0 - commands a rate, still not a raw force.
    "velocity": '<velocity name="hip_act" joint="hip" kv="5"/><velocity name="knee_act" joint="knee" kv="5"/>',
    # dyntype INTEGRATOR - an act state stands between ctrl and the force law.
    "intvelocity": (
        '<intvelocity name="hip_act" joint="hip" kp="10" actrange="-1 1"/>'
        '<intvelocity name="knee_act" joint="knee" kp="10" actrange="-1 1"/>'
    ),
    # AFFINE *gain* with no bias and no dynamics: clears a two-term test, yet
    # ctrl scales a damping coefficient, so "ctrl in Nm" would be false of it.
    "damper": (
        '<damper name="hip_act" joint="hip" kv="5" ctrlrange="0 1"/>'
        '<damper name="knee_act" joint="knee" kv="5" ctrlrange="0 1"/>'
    ),
    # Unanimous: one servo among motors means a pose write holds somewhere.
    "mixed": '<motor name="hip_act" joint="hip"/><position name="knee_act" joint="knee" kp="50"/>',
}
_TORQUE_ONLY = {"motor", "general"}


def _state(tmp_path, drive: str):
    from strands_robots.simulation import Simulation

    path = tmp_path / f"{drive}.xml"
    path.write_text(_LEG_XML.format(actuators=_DRIVES[drive]))
    sim = Simulation()
    sim.create_world(timestep=0.002)
    assert sim.add_robot("leg", urdf_path=str(path))["status"] == "success"
    try:
        res = sim.get_robot_state("leg")
        assert res["status"] == "success", res
        return res["content"][0]["text"], res["content"][1]["json"]
    finally:
        sim.destroy()


@requires_mujoco
class TestGetRobotStateNamesTorqueOnlyActuation:
    @pytest.mark.parametrize("drive", sorted(_DRIVES))
    def test_only_a_wholly_motor_driven_robot_is_reported_as_torque(self, tmp_path, drive) -> None:
        text, payload = _state(tmp_path, drive)
        if drive in _TORQUE_ONLY:
            assert "all 2 actuators are torque motors (ctrl in Nm)" in text
            assert "nothing holds the pose" in text
            assert payload["actuation"] == "torque"
        else:
            assert "torque motors" not in text
            assert "actuation" not in payload

    def test_the_note_does_not_disturb_the_joint_and_base_contract(self, tmp_path) -> None:
        text, payload = _state(tmp_path, "motor")
        assert set(payload["state"]) == {"hip", "knee"}
        assert payload["base"]["position"] == pytest.approx([0.0, 0.0, 0.5], abs=1e-9)
        assert text.startswith("'leg' state (t=0.000s):")

    @pytest.mark.skipif(not is_robot_asset_present("go2"), reason="go2 Menagerie asset not present")
    def test_go2_says_why_it_settles_within_two_seconds(self) -> None:
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        try:
            assert sim.add_robot("go2")["status"] == "success"
            before = sim.get_robot_state("go2")["content"][1]["json"]
            assert before["base"]["position"][2] == pytest.approx(0.445, abs=0.01)
            timestep = sim.physics_timestep()
            assert timestep is not None, "a created world reports its timestep"
            sim.step(int(2.0 / timestep))
            res = sim.get_robot_state("go2")
            text, after = res["content"][0]["text"], res["content"][1]["json"]
            # It collapsed, and now the same read says why.
            assert after["base"]["position"][2] < 0.25, after["base"]["position"]
            assert "all 12 actuators are torque motors (ctrl in Nm)" in text
            assert after["actuation"] == "torque"
        finally:
            sim.destroy()
