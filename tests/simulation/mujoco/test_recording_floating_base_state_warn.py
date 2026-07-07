"""Regression tests: start_recording warns when a floating base's orientation +
angular velocity are silently dropped from the recorded observation.state.

``get_observation`` surfaces ``base_quat`` (orientation, w,x,y,z) and
``base_ang_vel`` (rad/s) for a floating-base robot (a humanoid's named
``floating_base_joint`` or a mobile base's unnamed ``<freejoint>``). The
LeRobotDataset ``observation.state`` schema, however, is derived from the
robot's scalar joint names, so those base signals never reach the dataset - a
locomotion/whole-body-control policy trained on it would be base-blind. The
schema is intentionally left unchanged (existing datasets are stable); the
omission is surfaced at recording start instead of dropped silently, matching
the project's "no silent data loss" contract.
"""

import logging
import tempfile

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Humanoid-style floating base: a NAMED free root joint (enumerated in
# robot.joint_names, like a real ``floating_base_joint``) + one actuated hinge.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="torso" pos="0 0 0.5">
      <joint name="floating_base_joint" type="free"/>
      <geom type="box" size="0.1 0.1 0.2" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Mobile base: an UNNAMED free joint (not in robot.joint_names) + an actuated
# hinge. Exercises the kinematic-tree fallback detection (LeKiwi-style).
UNNAMED_BASE_XML = """
<mujoco model="test_unnamed_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: NO free joint anywhere. Must never trigger the base-drop
# warning (guards against a false positive).
FIXED_ARM_XML = """
<mujoco model="test_fixed_arm">
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

_WARN_MARK = "have a floating base"


def _write(xml: str) -> str:
    import os

    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def sim():
    pytest.importorskip("lerobot")  # start_recording produces a LeRobotDataset
    s = Simulation(tool_name="test_fb_warn", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    try:
        s.cleanup()
    except Exception:
        # Best-effort teardown: cleanup failures must not mask the test result.
        pass


def _start(sim, name, xml, caplog):
    sim.add_robot(name, urdf_path=_write(xml))
    root = tempfile.mkdtemp(prefix=f"fbwarn_{name}_")
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.recording"):
        res = sim.start_recording(repo_id=f"local/{name}_fbwarn", task="t", fps=20, root=root, cameras=[])
    sim.stop_recording()
    return res


def test_start_recording_warns_when_named_floating_base_state_dropped(sim, caplog):
    """A humanoid with a NAMED floating_base_joint: base_quat/base_ang_vel are
    dropped from observation.state -> start_recording must warn."""
    res = _start(sim, "humanoid", NAMED_BASE_XML, caplog)
    assert res["status"] == "success"
    base_warnings = [r for r in caplog.records if _WARN_MARK in r.getMessage()]
    assert base_warnings, "floating-base state-drop warning was not emitted"
    msg = base_warnings[0].getMessage()
    assert "humanoid" in msg
    assert "base_quat" in msg and "base_ang_vel" in msg


def test_start_recording_warns_for_unnamed_mobile_base(sim, caplog):
    """A mobile base on an UNNAMED freejoint (detected via the kinematic-tree
    fallback) must also warn."""
    res = _start(sim, "mob", UNNAMED_BASE_XML, caplog)
    assert res["status"] == "success"
    assert any(_WARN_MARK in r.getMessage() for r in caplog.records), (
        "unnamed-freejoint mobile base must trigger the base-drop warning"
    )


def test_start_recording_no_base_warning_for_fixed_arm(sim, caplog):
    """A fixed-base arm has no floating base -> no warning (no false positive)."""
    res = _start(sim, "arm", FIXED_ARM_XML, caplog)
    assert res["status"] == "success"
    assert not any(_WARN_MARK in r.getMessage() for r in caplog.records), (
        "fixed-base arm must NOT trigger the floating-base warning"
    )
