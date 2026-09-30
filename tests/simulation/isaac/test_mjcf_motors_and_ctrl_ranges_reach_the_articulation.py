"""An MJCF ``<motor>`` is driven as a torque on Isaac, and ctrl ranges reach policies that read them.

Measured on one L40S (Isaac Sim 6.1):

* go2's twelve joints are ``<motor>`` actuators, converted to PhysX force drives
  with ``stiffness=0``; ``send_action`` wrote position targets to them, so +10
  or -10 on FL_calf moved it by exactly 0.0 rad, under ``status="success"``.
  After: -0.044 and +1.841 rad against the zero-torque run.
* ``MockPolicy`` learns each actuator's range from the model it is handed
  (``set_sim_context``); Isaac handed it nothing, so on so100 it commanded
  +-0.5 into Pitch [-3.32, 0.174] and Jaw/Elbow [-0.174, ...]. After: the same
  learned ranges and commanded values as the MuJoCo backend.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")
mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _split_joint_action  # noqa: E402

_MJCF = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="thigh">
      <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" size="0.02 0.1" mass="1"/>
      <body name="calf" pos="0 0 -0.2">
        <joint name="knee" type="hinge" axis="0 1 0" range="-2.7 -0.8"/>
        <geom type="capsule" size="0.02 0.1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip" joint="hip" gear="2" ctrlrange="-10 10"/>
    <position name="knee" joint="knee" kp="30"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def mjcf(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "leg.xml"
    path.write_text(_MJCF, encoding="utf-8")
    return str(path)


def _robot(mjcf: str) -> Any:
    return types.SimpleNamespace(name="leg", joint_names=["hip", "knee"], description_path=mjcf, articulation=None)


class TestTheMotorTable:
    def test_a_motor_is_listed_with_gear_and_range_and_a_servo_is_not(self, mjcf: str) -> None:
        assert mjcf_assets.mjcf_motor_joints(mjcf) == {"hip": (2.0, -10.0, 10.0)}

    def test_no_file_no_table(self, tmp_path: pathlib.Path) -> None:
        assert mjcf_assets.mjcf_motor_joints(str(tmp_path / "missing.xml")) == {}
        assert mjcf_assets.mjcf_motor_joints(None) == {}


class TestTheActionIsSplit:
    def test_a_motor_joint_gets_an_effort_clipped_to_ctrlrange_times_gear(self, mjcf: str) -> None:
        pos, pos_idx, eff, eff_idx = _split_joint_action(_robot(mjcf), {"hip": 25.0, "knee": -1.2})
        assert pos.tolist() == pytest.approx([-1.2]) and pos_idx.tolist() == [1]
        assert eff.tolist() == pytest.approx([20.0]) and eff_idx.tolist() == [0]  # clip 25 -> 10, x gear 2

    def test_unnamed_joints_are_not_commanded(self, mjcf: str) -> None:
        pos, pos_idx, eff, eff_idx = _split_joint_action(_robot(mjcf), {"knee": -1.0})
        assert pos_idx.tolist() == [1] and eff.size == 0 and eff_idx.size == 0

    def test_a_robot_without_an_mjcf_is_all_position_targets(self) -> None:
        robot = types.SimpleNamespace(joint_names=["a", "b"], description_path=None)
        pos, pos_idx, eff, _ = _split_joint_action(robot, {"a": 0.1, "b": 0.2})
        assert pos_idx.tolist() == [0, 1] and eff.size == 0


class TestAPolicyGetsTheCompiledModel:
    def test_set_sim_context_receives_the_robots_mjcf(self, mjcf: str) -> None:
        seen: dict[str, Any] = {}

        class _Policy:
            def set_sim_context(self, model: Any, namespace: str) -> None:
                seen["nu"], seen["ns"] = int(model.nu), namespace

        engine: Any = IsaacSimulation.__new__(IsaacSimulation)
        engine._robots = {"leg": _robot(mjcf)}
        engine.bind_policy_sim_context(_Policy(), "leg")
        assert seen == {"nu": 2, "ns": ""}

    def test_a_usd_robot_is_left_alone(self) -> None:
        class _Policy:
            def set_sim_context(self, model: Any, namespace: str) -> None:
                raise AssertionError("must not be called")

        engine: Any = IsaacSimulation.__new__(IsaacSimulation)
        engine._robots = {"arm": types.SimpleNamespace(description_path="/x/robot.usd")}
        engine.bind_policy_sim_context(_Policy(), "arm")

    def test_mock_policy_learns_the_ranges(self, mjcf: str) -> None:
        from strands_robots import MockPolicy

        engine: Any = IsaacSimulation.__new__(IsaacSimulation)
        engine._robots = {"leg": _robot(mjcf)}
        policy = MockPolicy()
        policy.set_robot_state_keys(["hip", "knee"])
        engine.bind_policy_sim_context(policy, "leg")
        bounds = policy._ctrl_bounds
        assert set(bounds) == {"hip", "knee"} and np.allclose(bounds["hip"], (-10.0, 10.0))
        assert np.allclose(bounds["knee"], (-2.7, -0.8))
