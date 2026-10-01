"""A site-actuated free body (a MuJoCo quadrotor) is driven on Isaac as MuJoCo drives it.

``crazyflie`` and ``skydio_x2`` have no joints besides the free base, and their
actuators are ``<motor site=...>`` thrusters. PhysX builds no articulation from
such a body, so Isaac's articulation path failed on both ("'NoneType' object
has no attribute 'is_homogeneous'"). They now load as one rigid body whose
motors apply, every tick, the wrench MuJoCo's site transmission applies. These
cells pin that wrench against MuJoCo's own ``qfrc_actuator``.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.isaac.site_drives import (  # noqa: E402
    SiteDriveError,
    _quat_to_matrix,
    mjcf_site_drive,
    site_wrenches,
)

_QUAD = """<mujoco model="quad">
  <worldbody>
    <body name="frame" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .02" mass="1"/>
      <site name="r1" pos=".1 .1 .02"/>
      <site name="r2" pos="-.1 -.1 .02" quat="0.9659 0.2588 0 0"/>
      <site name="imu"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" site="r1" gear="0 0 1 0 0 -.02" ctrlrange="0 5"/>
    <motor name="m2" site="r2" gear="0 0 1 0 0 .02" ctrlrange="0 5"/>
    <motor name="roll" site="imu" gear="0 0 0 -.001 0 0" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def quad(tmp_path) -> str:
    path = tmp_path / "quad.xml"
    path.write_text(_QUAD)
    return str(path)


def test_a_quadrotor_is_read_as_one_body_and_its_motors(quad) -> None:
    drive = mjcf_site_drive(quad)
    assert drive is not None
    assert drive.body_name == "frame"
    assert [a.name for a in drive.actuators] == ["m1", "m2", "roll"]
    assert drive.actuators[0].ctrlrange == (0.0, 5.0)


def test_the_control_is_clipped_to_its_range_as_mujoco_clips_it(quad) -> None:
    drive = mjcf_site_drive(quad)
    assert drive is not None
    assert drive.set_ctrl("m1", 9.0) == 5.0
    assert drive.set_ctrl("roll", -3.0) == -1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_wrench_is_mujocos_site_transmission(quad, seed) -> None:
    """Summed about the body origin and put in the free joint's frame, it IS qfrc_actuator."""
    drive = mjcf_site_drive(quad)
    assert drive is not None
    model = mujoco.MjModel.from_xml_path(quad)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    data.qpos[:3] = rng.normal(size=3)
    data.qpos[3:7] = quat
    for i, act in enumerate(drive.actuators):
        assert act.ctrlrange is not None
        data.ctrl[i] = drive.set_ctrl(act.name, rng.uniform(*act.ctrlrange))
    mujoco.mj_forward(model, data)

    body_pos = data.qpos[:3].copy()
    force, torque = np.zeros(3), np.zeros(3)
    for f, t, point in site_wrenches(drive, body_pos, quat):
        force += f
        torque += t + np.cross(point - body_pos, f)
    ours = np.concatenate([force, _quat_to_matrix(quat).T @ torque])

    np.testing.assert_allclose(ours, data.qfrc_actuator[:6], atol=1e-12)


def test_a_motor_that_is_off_pushes_nothing(quad) -> None:
    drive = mjcf_site_drive(quad)
    assert drive is not None
    drive.set_ctrl("m1", 0.0)
    assert site_wrenches(drive, np.zeros(3), np.array([1.0, 0, 0, 0])) == []


# a joint: an articulated robot, the articulation path's to load
_ARTICULATED = (
    '<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>'
    '<actuator><motor joint="j"/></actuator></mujoco>'
)
# no actuators: nothing to drive
_UNDRIVEN = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
# a welded body with a thruster cannot fly
_WELDED_THRUSTER = (
    '<mujoco><worldbody><body><geom size=".1"/><site name="s"/></body></worldbody>'
    '<actuator><motor site="s" gear="0 0 1 0 0 0"/></actuator></mujoco>'
)


@pytest.mark.parametrize("xml", [_ARTICULATED, _UNDRIVEN, _WELDED_THRUSTER])
def test_anything_else_stays_on_the_articulation_path(tmp_path, xml) -> None:
    path = tmp_path / "m.xml"
    path.write_text(xml)
    assert mjcf_site_drive(str(path)) is None


# A site actuator whose force is NOT gear * ctrl: the backend drives only the
# <motor> shape (fixed gain 1, no bias, no activation dynamics). Anything else
# would be flown with the wrong force and no signal; it is refused by name.
_SERVO_SITE = (
    '<mujoco><worldbody><body><freejoint/><geom size=".1"/><site name="s"/></body></worldbody>'
    '<actuator><position name="hover" site="s" gear="0 0 1 0 0 0" kp="10"/></actuator></mujoco>'
)
_FILTERED_ROTOR = (
    '<mujoco><worldbody><body><freejoint/><geom size=".1"/><site name="s"/></body></worldbody>'
    '<actuator><general name="rotor" site="s" gear="0 0 1 0 0 0" dyntype="filter" dynprm="0.05"/>'
    "</actuator></mujoco>"
)
_SCALED_MOTOR = (
    '<mujoco><worldbody><body><freejoint/><geom size=".1"/><site name="s"/></body></worldbody>'
    '<actuator><general name="scaled" site="s" gear="0 0 1 0 0 0" gainprm="2.5"/></actuator></mujoco>'
)


@pytest.mark.parametrize(
    ("xml", "name", "why"),
    [
        (_SERVO_SITE, "hover", "bias"),
        (_FILTERED_ROTOR, "rotor", "dyntype"),
        (_SCALED_MOTOR, "scaled", "gainprm"),
    ],
    ids=["position-servo", "filtered-rotor", "scaled-gain"],
)
def test_a_site_actuator_the_backend_cannot_drive_faithfully_is_refused_by_name(tmp_path, xml, name, why) -> None:
    path = tmp_path / "m.xml"
    path.write_text(xml)
    with pytest.raises(SiteDriveError, match=rf"(?s){name}.*{why}.*gear \* ctrl") as info:
        mjcf_site_drive(str(path))
    assert "MuJoCo" in str(info.value)  # what the caller is told the sim would diverge from


def test_a_plain_motor_with_an_explicit_unit_gain_is_still_a_motor(tmp_path) -> None:
    path = tmp_path / "m.xml"
    path.write_text(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/><site name="s"/></body></worldbody>'
        '<actuator><general name="m" site="s" gear="0 0 1 0 0 0" gainprm="1"/></actuator></mujoco>'
    )
    drive = mjcf_site_drive(str(path))
    assert drive is not None and [a.name for a in drive.actuators] == ["m"]
