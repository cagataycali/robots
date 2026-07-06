"""``add_robot(keyframe=...)`` spawns a robot in a source ``<keyframe>`` pose.

MuJoCo Menagerie robots ship a canonical home pose in a ``<keyframe>`` (panda
``home``, ur5e/fr3/kuka ``home``, aloha ``neutral_pose``, quadrupeds/humanoids
a standing ``home``). ``add_robot`` historically ran ``mj_resetData`` (the
all-zero configuration) and ``reset()`` does the same, so that shipped home
pose was unreachable outside the LIBERO benchmark adapter. A policy trained
from the home pose then sees an out-of-distribution start (a folded/collapsed
arm), which measurably degrades its rollout.

``add_robot(keyframe=...)`` applies the named/indexed keyframe's qpos to the
robot's joints by name at spawn and stores it so ``reset()`` restores it (a
keyframe spawn is sticky across resets, matching how a benchmark restores its
canonical start each episode). ``keyframe=None`` (the default) keeps the
historical zero-pose spawn byte-for-byte.

These tests use a tiny inline two-hinge MJCF with a ``<keyframe>`` so they run
offline and GL-free in CI (no mesh download, no render).
"""

from __future__ import annotations

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two hinge joints; the ``home`` keyframe bends them to a non-zero pose that is
# distinct from the all-zero default and from what gravity/servos settle to.
_ARM_MJCF = """
<mujoco model="kf_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="l1" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1"/>
      <body name="l2" pos="0 0 0.3">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="s_act" joint="shoulder" kp="50"/>
    <position name="e_act" joint="elbow" kp="50"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.5 -1.2"/>
  </keyframe>
</mujoco>
"""

_HOME = [0.5, -1.2]


@pytest.fixture
def arm_xml(tmp_path):
    p = tmp_path / "kf_arm.xml"
    p.write_text(_ARM_MJCF)
    return str(p)


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_add_robot_keyframe", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _qpos(sim):
    return sim._world._data.qpos.copy()


class TestAddRobotKeyframe:
    def test_default_spawn_is_zero_pose(self, sim, arm_xml):
        # No keyframe -> historical all-zero spawn, and no home pose captured.
        sim.add_robot(name="a", urdf_path=arm_xml)
        assert np.allclose(_qpos(sim), [0.0, 0.0])
        assert sim._world.robots["a"].home_qpos == {}

    def test_keyframe_by_name_applies_home_pose(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)
        # Home pose captured under the namespaced joint names for reset().
        assert sim._world.robots["a"].home_qpos == {
            "a/shoulder": [0.5],
            "a/elbow": [-1.2],
        }

    def test_keyframe_by_index_applies_home_pose(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe=0)
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)

    def test_reset_restores_keyframe_home_pose(self, sim, arm_xml):
        sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        # Drive the arm off the home pose, then reset.
        sim.step(40)
        assert not np.allclose(_qpos(sim), _HOME)
        reset_result = sim.reset()
        assert reset_result["status"] == "success"
        # reset() must restore the keyframe home pose, not collapse to zeros.
        assert np.allclose(_qpos(sim), _HOME)

    def test_reset_without_keyframe_stays_zero(self, sim, arm_xml):
        # Guard the no-regression path: a robot added without a keyframe must
        # reset to the zero configuration exactly as before.
        sim.add_robot(name="a", urdf_path=arm_xml)
        sim.step(40)
        sim.reset()
        assert np.allclose(_qpos(sim), [0.0, 0.0])

    def test_unknown_keyframe_errors_and_leaks_nothing(self, sim, arm_xml):
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="does_not_exist")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "does_not_exist" in text
        # Names the available keyframe so the caller can fix the call.
        assert "'home'" in text
        # No half-added robot left behind; the same name is reusable.
        assert "a" not in sim._world.robots
        ok = sim.add_robot(name="a", urdf_path=arm_xml, keyframe="home")
        assert ok["status"] == "success"

    def test_bool_keyframe_rejected(self, sim, arm_xml):
        # bool is an int subclass; True/False must not be taken as index 1/0.
        result = sim.add_robot(name="a", urdf_path=arm_xml, keyframe=True)
        assert result["status"] == "error"
        assert "bool" in result["content"][0]["text"]
        assert "a" not in sim._world.robots

    def test_keyframe_via_tool_router(self, sim, arm_xml):
        # The agent-facing dispatch path forwards the keyframe param.
        result = sim._dispatch_action(
            "add_robot",
            {"action": "add_robot", "name": "a", "urdf_path": arm_xml, "keyframe": "home"},
        )
        assert result["status"] == "success"
        assert np.allclose(_qpos(sim), _HOME)
