"""Regression tests for mobile-base (unnamed-freejoint) observation state.

A mobile manipulator (e.g. LeKiwi) carries its floating base on an UNNAMED
``<freejoint/>`` that is not enumerated in ``robot.joint_names`` (those are the
actuated wheel/arm joints). The floating-base IMU-style signals
(``base_quat`` + ``base_ang_vel``) must still be surfaced from that base free
joint, otherwise the mobile base is silently observed as a fixed-base arm and a
recorded dataset loses all base orientation/velocity state.

The base free joint is recovered from the kinematic tree (walk up from an
actuated joint to the ancestor base body), so a sibling free-jointed task
object (a cube) is never mistaken for the base.
"""

import os
import tempfile

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Mobile base: an UNNAMED free joint on ``base_plate`` (identity orientation),
# an actuated hinge ``shoulder`` on a child arm body, and a SIBLING free-jointed
# ``task_cube`` at a distinct 90-deg-about-z orientation. Mirrors LeKiwi, whose
# base freejoint is unnamed and whose scene carries free-jointed task cubes.
MOBILE_BASE_XML = """
<mujoco model="test_mobile">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base_plate" pos="0 0 0.1" quat="1 0 0 0">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
    <body name="task_cube" pos="1 0 0.05" quat="0.707 0 0 0.707">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: a single hinge, NO free joint anywhere. Must never surface
# base state (guards against a false-positive base pick).
FIXED_ARM_XML = """
<mujoco model="test_fixed">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_mobile_base", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def test_unnamed_base_freejoint_surfaces_base_state(sim):
    """A mobile base whose floating base is an unnamed free joint (not in
    ``robot.joint_names``) still surfaces ``base_quat`` + ``base_ang_vel``, and
    the base is resolved from the kinematic tree - never the sibling free-jointed
    task cube."""
    sim.add_robot("mob", urdf_path=_write(MOBILE_BASE_XML))

    # The unnamed base freejoint is not a named joint.
    assert sim._world.robots["mob"].joint_names == ["shoulder"]

    obs = sim.get_observation(robot_name="mob", skip_images=True)

    assert "base_quat" in obs, "mobile base must surface base_quat"
    assert len(obs["base_quat"]) == 4
    assert all(isinstance(x, float) for x in obs["base_quat"])
    assert "base_ang_vel" in obs, "mobile base must surface base_ang_vel"
    assert len(obs["base_ang_vel"]) == 3

    # base_quat is the base_plate (identity), NOT the task_cube (90 deg about z,
    # ~[0.707, 0, 0, 0.707]). If the sibling cube's free joint were picked, w
    # would be ~0.707 instead of ~1.0.
    assert obs["base_quat"][0] == pytest.approx(1.0, abs=1e-3)
    assert obs["base_quat"][3] == pytest.approx(0.0, abs=1e-3)

    # The actuated hinge still gets a scalar .vel; the free joint does not.
    assert "shoulder.vel" in obs and isinstance(obs["shoulder.vel"], float)


def test_base_quat_is_live_read_of_the_base_free_joint(sim):
    """``base_quat`` reflects the base free joint's CURRENT orientation (a live
    read), so a driven/rotated base is observed - it is not a static or
    wrong-joint value."""
    import mujoco

    sim.add_robot("mob", urdf_path=_write(MOBILE_BASE_XML))
    model, data = sim._world._model, sim._world._data

    # Locate the base_plate free joint and rotate it 90 deg about z.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mob/base_plate")
    jadr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jadr = int(model.jnt_qposadr[j])
    assert jadr is not None
    data.qpos[jadr + 3 : jadr + 7] = [0.70710678, 0.0, 0.0, 0.70710678]
    mujoco.mj_forward(model, data)

    obs = sim.get_observation(robot_name="mob", skip_images=True)
    assert obs["base_quat"][0] == pytest.approx(0.7071, abs=1e-3)
    assert obs["base_quat"][3] == pytest.approx(0.7071, abs=1e-3)


def test_fixed_base_arm_has_no_base_state(sim):
    """A fixed-base arm (no free joint) never surfaces base state - the mobile
    detection must not false-positive on a robot that has no floating base."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    obs = sim.get_observation(robot_name="arm", skip_images=True)
    assert "base_quat" not in obs
    assert "base_ang_vel" not in obs
    assert "j0" in obs and "j0.vel" in obs
