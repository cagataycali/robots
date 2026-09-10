"""``set_joint_velocities`` refuses a value MuJoCo would answer by resetting the world.

Measured (deepdive D-052): ``velocities={"Rotation": 1e300}`` on the bundled
so100 returned ``success``; the next ``step`` printed MuJoCo's ``WARNING: Nan,
Inf or huge value in QVEL ... The simulation is unstable`` to stderr and reset
every joint and object to its initial state - a silent success over a wiped
world. The write now runs mj_step's own test (finite and below ``mjMAXVAL`` on
``qvel`` and on the ``qacc`` it produces) under a checkpoint and refuses,
leaving the state exactly as it was: measured against a sim that never made the
call, ``time``, ``qpos``, ``qvel``, ``act`` and ``ctrl`` are byte-identical
after the refusal and so is the trajectory 50 steps later.
"""

from __future__ import annotations

import importlib.util

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

_ARM = """
<mujoco model="one_joint">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.1"/>
      <body name="tip" pos="0 0 0.2"><geom type="box" size="0.02 0.02 0.02"/></body>
    </body>
  </worldbody>
  <actuator><position name="hinge_act" joint="hinge" kp="50"/></actuator>
</mujoco>
"""


@pytest.fixture
def sim(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "one_joint.xml"
    path.write_text(_ARM)
    s = Simulation()
    s.create_world(timestep=0.002)
    assert s.add_robot("arm", urdf_path=str(path))["status"] == "success"
    assert s.set_joint_positions(robot_name="arm", positions={"hinge": 0.5})["status"] == "success"
    s.step(n_steps=1)
    yield s
    s.destroy()


def _hinge(sim):
    st = sim.get_robot_state("arm")["content"][1]["json"]["state"]["hinge"]
    return st["position"], st["velocity"]


# 1e300 and 1e10 trip on qvel itself (mjMAXVAL is 1e10); 1e9 is under that
# ceiling and trips only through the qacc it produces on this hinge (2.7e10),
# which is the arm of the check a qvel-only test would leave unpinned - and it
# resets the world on unfixed code exactly as the other two do.
@requires_mujoco
@pytest.mark.parametrize("value", [1e300, 1e10, 1e9])
def test_a_value_mujoco_would_reset_on_is_refused_and_the_state_is_unchanged(sim, value) -> None:
    pos_before, vel_before = _hinge(sim)
    res = sim.set_joint_velocities(robot_name="arm", velocities={"hinge": value})
    assert res["status"] == "error"
    text = res["content"][0]["text"]
    assert "unstable" in text and "reset" in text and "Nothing was written" in text
    assert f"{value:.3g}" in text
    pos_after, vel_after = _hinge(sim)
    assert (pos_after, vel_after) == pytest.approx((pos_before, vel_before), abs=1e-12)
    # And the world really is intact: a step later the joint is still up where
    # it was left (the servo pulls it toward 0 slowly), not at the reset pose 0.
    sim.step(n_steps=1)
    assert _hinge(sim)[0] > 0.3


@requires_mujoco
def test_an_ordinary_velocity_is_still_written(sim) -> None:
    res = sim.set_joint_velocities(robot_name="arm", velocities={"hinge": 2.0})
    assert res["status"] == "success"
    assert _hinge(sim)[1] == pytest.approx(2.0)
