"""A kinematic pose write reports whether the actuators will hold the pose.

``set_joint_positions`` writes ``qpos`` and runs forward kinematics. On a robot
whose joints are held by position servos that is only half of what "teleport /
set an initial pose" needs: the servos are still commanded to their previous
setpoint, so the first ``mj_step`` drives the pose straight back toward it::

    sim.set_joint_positions(pose)      # -> "Set 6/6 joint positions, FK updated"
    sim.step(150)                      # so101: 2.75 rad away from `pose`, every
                                       #        joint back near zero

The pose read back correct and the call reported success because nothing had
stepped yet. That is the same failure the rest of the backend already guards:
``actuate_robot`` seeds every actuator it adds from its joint's current position
"so the arm doesn't snap to zero on the next step", and ``remove_robot`` carries
``ctrl`` across an eject because dropped setpoints read as zero and "an arm
parked mid-air sags to the floor while ``remove_robot`` reported success". The
one surface whose whole job is writing a pose was the one that neither moved the
setpoints nor said it had not.

It is not a niche shape: 42 of the 62 loadable registry robots drive at least one
of their joints with a position servo (so101, panda, aloha, unitree_g1, spot,
ur5e, kinova_gen3 among them).

These tests pin both halves:

* the default write is still kinematics-only - unchanged, because rendering a
  pose or replaying a planned trajectory frame by frame is a real use - but its
  success text now names the joints whose servo holds a different setpoint and
  quotes the remedy;
* ``hold=True`` moves those setpoints with the pose, which is what makes
  "teleport and stay there" expressible at all: ``send_action`` writes the
  setpoints but never ``qpos``, and always advances at least one step.

Only position servos are moved. A motor takes a torque, so writing a joint angle
into its ``ctrl`` would command a torque numerically equal to an angle in
radians. 14 registry robots are motor-driven throughout (unitree_go2, unitree_h1,
jvrc, cassie) and ``openarm`` carries 2 servos beside 16 motors, so the split has
to be per actuator - which is what
:func:`~strands_robots.simulation.mujoco.scene_ops.joint_drive_map` owns.
"""

import re

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Gravity is off so a settled servo lands exactly on its setpoint: the only force
# moving a joint here is an actuator, which is the quantity under test. Under
# gravity the same assertions would have to absorb the servo's steady-state droop.
DRIVE_MIX_XML = """
<mujoco model="drive_mix">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link1" pos="0 0 0.4">
      <joint name="served" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link1_geom" type="capsule" size="0.02 0.1" mass="0.4"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="motored" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom name="link2_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
      </body>
    </body>
    <body name="link3" pos="0.5 0 0.4">
      <joint name="undriven" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom name="link3_geom" type="capsule" size="0.015 0.08" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="served_servo" joint="served" kp="60" kv="6"/>
    <motor name="motored_motor" joint="motored"/>
  </actuator>
</mujoco>
"""

POSE = {"served": 0.5, "motored": 0.6, "undriven": 0.7}


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_pose_write_servo_hold", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(DRIVE_MIX_XML)["status"] == "success"
    yield s
    s.cleanup()


def _qpos(sim, joint: str) -> float:
    model, data = sim._world._model, sim._world._data
    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jnt_id >= 0, joint
    return float(data.qpos[model.jnt_qposadr[jnt_id]])


def _ctrl(sim, actuator: str) -> float:
    model, data = sim._world._model, sim._world._data
    act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator)
    assert act_id >= 0, actuator
    return float(data.ctrl[act_id])


def _text(result: dict) -> str:
    assert result["status"] == "success", result
    return str(result["content"][0]["text"])


class TestTheReportNamesAPoseTheNextStepUndoes:
    """The success text distinguishes a pose that survives from one that does not."""

    def test_a_servo_commanded_elsewhere_is_named_with_its_remedy(self, sim):
        """The joint whose servo pulls the pose back is named, and only that one.

        The measurement the report stands on: after the write the pose is exact,
        and stepping moves the served joint by nearly the whole request while the
        joints with no servo of their own stay put.
        """
        text = _text(sim.set_joint_positions(positions=POSE))
        assert _qpos(sim, "served") == pytest.approx(0.5)

        assert sim.step(n_steps=200)["status"] == "success"
        assert abs(_qpos(sim, "served") - 0.5) > 0.4, "premise: the servo pulls the pose back"
        assert _qpos(sim, "undriven") == pytest.approx(0.7, abs=1e-6)

        assert "served" in text, text
        assert "motored" not in text, "a motor's ctrl is not a setpoint the pose can be compared to"
        assert "undriven" not in text, "a joint with no actuator has no setpoint to disagree with"
        assert "hold=True" in text, text

    def test_the_remedy_the_report_quotes_makes_the_pose_survive_stepping(self, sim):
        """Parse the remedy out of the report, apply it, and step.

        Pinning the remedy rather than the wording: this fails both for a report
        that names no remedy and for one that names a remedy that does not work.
        """
        text = _text(sim.set_joint_positions(positions=POSE))
        offered = re.search(r"pass (\w+)=True", text)
        assert offered, f"the report offers no remedy: {text}"

        assert sim.reset()["status"] == "success"
        assert _text(sim.set_joint_positions(positions=POSE, **{offered.group(1): True}))
        assert sim.step(n_steps=200)["status"] == "success"
        assert _qpos(sim, "served") == pytest.approx(0.5, abs=1e-3)

    def test_a_servo_already_commanded_to_the_pose_is_not_reported(self, sim):
        """Writing the pose the servos already hold reports nothing extra.

        The report is about a disagreement, so it must not fire on a write that
        merely re-asserts where the servos are already pointed.
        """
        assert _text(sim.set_joint_positions(positions=POSE, hold=True))
        text = _text(sim.set_joint_positions(positions=POSE))
        assert "hold=True" not in text, text
        assert "served" not in text, text


class TestHoldMovesTheSetpointsWithThePose:
    """``hold=True`` writes the position-servo setpoints, and nothing else."""

    def test_the_servo_setpoint_becomes_the_pose_written(self, sim):
        text = _text(sim.set_joint_positions(positions=POSE, hold=True))
        assert _ctrl(sim, "served_servo") == pytest.approx(0.5)
        assert _qpos(sim, "served") == pytest.approx(0.5)
        assert "1 position-servo setpoint(s) moved" in text, text

    def test_a_motor_keeps_its_command_and_the_report_says_so(self, sim):
        """A joint angle must never land in a motor's ctrl.

        ``ctrl`` on a motor is a torque, so the pose value would be commanded as
        a torque numerically equal to an angle in radians. The report names the
        joints left alone so the caller is not told the whole pose is held.
        """
        text = _text(sim.set_joint_positions(positions=POSE, hold=True))
        assert _ctrl(sim, "motored_motor") == 0.0
        assert "motored" in text, text
        assert "torque" in text, text

    def test_hold_must_be_a_boolean(self, sim):
        """A flag read by truthiness inverts for the spellings an opt-out uses."""
        result = sim.set_joint_positions(positions=POSE, hold="false")
        assert result["status"] == "error", result
        assert "hold must be a boolean" in result["content"][0]["text"]
        assert _qpos(sim, "served") == pytest.approx(0.0), "a refused flag writes nothing"


class TestTheDefaultWriteIsUnchanged:
    """The kinematic write itself is untouched, so every existing caller is too."""

    def test_no_setpoint_moves_without_hold(self, sim):
        before = sim._world._data.ctrl.copy()
        assert _text(sim.set_joint_positions(positions=POSE))
        assert np.array_equal(sim._world._data.ctrl, before)

    def test_the_whole_pose_still_lands_in_qpos(self, sim):
        assert _text(sim.set_joint_positions(positions=POSE))
        for joint, value in POSE.items():
            assert _qpos(sim, joint) == pytest.approx(value)


class TestTheDriveSplitIsPerActuator:
    """``joint_drive_map`` classifies each actuator, not each robot.

    Imported inside each test so the module still collects against a tree without
    the helper, and the rest of the file reports its own verdict there.
    """

    def test_a_servo_and_a_motor_in_one_model_land_in_different_maps(self, sim):
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        model = sim._world._model
        servos, other = joint_drive_map(model, mj)
        served = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "served")
        motored = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "motored")
        undriven = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "undriven")

        assert servos == {served: mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "served_servo")}
        assert other == {motored: mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "motored_motor")}
        assert undriven not in servos and undriven not in other

    def test_the_two_maps_never_share_a_joint(self, sim):
        from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

        servos, other = joint_drive_map(sim._world._model, mj)
        assert not set(servos) & set(other)
