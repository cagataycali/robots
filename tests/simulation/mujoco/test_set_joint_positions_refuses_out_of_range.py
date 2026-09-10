"""``set_joint_positions`` refuses a pose outside a limited joint's range.

``mj_forward`` does not clamp ``qpos``, so before this fix an out-of-range value
landed in the state and the call reported ``"Set n/n joint positions, FK
updated"``. The next ``step`` then drove the joint back through its limit at
whatever velocity the constraint solver produced - measured on the bundled
``so100``: ``positions=[99, 0, 0, 0, 0, 0]`` on a ``[-1.92, 1.92]`` joint left
it at -9.4 rad moving 23.8 rad/s after 100 steps, with a success result.

The house rule is refuse-not-clamp: the error names the joint, the value and
the range, and nothing is written. An unlimited joint (no ``range``) keeps
accepting any finite value - that is the over-reach control.
"""

from __future__ import annotations

import importlib.util

import pytest

requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed",
)

# Inline robot XML - no network dependency. ``elbow`` is range-limited,
# ``spinner`` deliberately has no range (jnt_limited == 0).
_ROBOT_XML = """
<mujoco model="range_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="spinner" type="hinge" axis="0 0 1"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="spinner_act" joint="spinner" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_sim(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "range_arm.xml"
    path.write_text(_ROBOT_XML)
    sim = Simulation()
    sim.create_world(timestep=0.002)
    result = sim.add_robot("arm", urdf_path=str(path), position=[0.0, 0.0, 0.0])
    assert result["status"] == "success", f"add_robot failed: {result}"
    yield sim
    sim.destroy()


@requires_mujoco
class TestSetJointPositionsRefusesOutOfRange:
    @pytest.mark.parametrize("value", [99.0, -1.58, 1.5701], ids=["far", "just-below", "just-above"])
    def test_out_of_range_is_refused_and_nothing_written(self, arm_sim, value: float) -> None:
        before = arm_sim._world._data.qpos.copy()
        res = arm_sim.set_joint_positions({"elbow": value})
        assert res["status"] == "error", f"{value!r} was written on a [-1.57, 1.57] joint: {res}"
        text = res["content"][0]["text"]
        assert "elbow" in text and f"{value:.4g}" in text and "[-1.57, 1.57]" in text
        assert "nothing written" in text
        assert (arm_sim._world._data.qpos == before).all(), "a refused pose must not touch qpos"

    def test_one_bad_value_refuses_the_whole_write(self, arm_sim) -> None:
        before = arm_sim._world._data.qpos.copy()
        res = arm_sim.set_joint_positions({"spinner": 0.5, "elbow": 3.0})
        assert res["status"] == "error"
        assert (arm_sim._world._data.qpos == before).all(), "a partial write is a silent success"

    def test_in_range_and_limits_still_accepted(self, arm_sim) -> None:
        for value in (0.0, 1.57, -1.57):
            res = arm_sim.set_joint_positions({"elbow": value})
            assert res["status"] == "success", f"{value!r} is inside the range: {res}"

    def test_unlimited_joint_accepts_any_finite_value(self, arm_sim) -> None:
        res = arm_sim.set_joint_positions({"spinner": 99.0})
        assert res["status"] == "success", f"a joint with no range has no limit to refuse on: {res}"
