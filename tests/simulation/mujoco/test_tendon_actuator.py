"""Regression tests for issue #318: MuJoCo tendon-transmission actuators.

``_apply_action_by_name`` previously resolved a joint-name action key only to
*direct* joint-transmission actuators (``actuator_trnid[ai, 0] == jnt_id``).
Tendon-transmission actuators - most notably the Franka/Panda gripper, whose
``split`` actuator drives ``finger_joint1`` / ``finger_joint2`` through a
tendon - were silently dropped, so the gripper never actuated.

These tests build a tiny synthetic model (no asset download) with one direct
joint actuator and one tendon actuator wrapping two finger joints, and pin:

1. ``_actuator_for_joint`` resolves a finger joint to the tendon actuator.
2. ``_actuator_for_joint`` still resolves a direct joint to its actuator.
3. ``_scale_ctrl_for_actuator`` maps a logical [0, 1] command onto the
   tendon actuator's wide ctrlrange, and leaves direct-joint commands raw.
4. End-to-end ``_apply_action_by_name`` writes a non-zero ctrl for a finger
   joint key (the bug was a silent no-op).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402

# A minimal arm: one hinge "arm_joint" (direct actuator) + two finger slides
# wrapped by a "split" tendon driven by a single tendon actuator with a wide
# ctrlrange (mirrors the Panda gripper's [0, 255]).
_XML = """
<mujoco model="tendon_test">
  <worldbody>
    <body name="link">
      <joint name="arm_joint" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0 0 0.2"/>
      <body name="hand" pos="0 0 0.2">
        <geom type="box" size="0.03 0.03 0.02"/>
        <body name="finger1" pos="0.03 0 0">
          <joint name="finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.01 0.02"/>
        </body>
        <body name="finger2" pos="-0.03 0 0">
          <joint name="finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.01 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="1"/>
      <joint joint="finger_joint2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="arm_act" joint="arm_joint" ctrlrange="-3 3"/>
    <position name="grip_act" tendon="split" ctrlrange="0 255" kp="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def test_actuator_for_joint_resolves_tendon_gripper(model):
    """A finger joint must resolve to the tendon actuator (issue #318)."""
    fj = _jid(model, "finger_joint1")
    grip = _aid(model, "grip_act")
    assert RenderingMixin._actuator_for_joint(model, fj, mujoco) == grip


def test_actuator_for_joint_resolves_second_finger(model):
    """The other tendon-wrapped joint resolves to the same actuator."""
    fj2 = _jid(model, "finger_joint2")
    grip = _aid(model, "grip_act")
    assert RenderingMixin._actuator_for_joint(model, fj2, mujoco) == grip


def test_actuator_for_joint_resolves_direct_joint(model):
    """Direct joint transmission keeps working (no regression)."""
    aj = _jid(model, "arm_joint")
    arm = _aid(model, "arm_act")
    assert RenderingMixin._actuator_for_joint(model, aj, mujoco) == arm


def test_actuator_for_joint_unknown_returns_negative(model):
    """A joint that drives no actuator returns -1."""
    assert RenderingMixin._actuator_for_joint(model, 999, mujoco) == -1


def test_scale_maps_logical_open_to_full_range(model):
    """A logical 1.0 (fully open) maps to the tendon ctrlrange max."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 1.0, mujoco)
    assert out == pytest.approx(255.0)


def test_scale_maps_logical_close_to_range_min(model):
    """A logical 0.0 (closed) maps to the ctrlrange min."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 0.0, mujoco)
    assert out == pytest.approx(0.0)


def test_scale_passes_through_clear_in_range_command(model):
    """A clearly in-range tendon command (e.g. 128) is trusted verbatim."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 128.0, mujoco)
    assert out == pytest.approx(128.0)


def test_scale_leaves_direct_joint_value_raw(model):
    """Direct JOINT actuators are never rescaled (positions/torques)."""
    arm = _aid(model, "arm_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, arm, 0.5, mujoco)
    assert out == pytest.approx(0.5)


def test_apply_action_by_name_drives_tendon_gripper(model):
    """End-to-end: a finger-joint key writes a non-zero tendon ctrl.

    Before the fix this was a silent no-op (data.ctrl stayed 0).
    """
    data = mujoco.MjData(model)
    mixin = RenderingMixin()
    mixin._apply_action_by_name(model, data, {"finger_joint1": 1.0}, "", mujoco)
    grip = _aid(model, "grip_act")
    assert data.ctrl[grip] == pytest.approx(255.0)


def test_apply_action_by_name_direct_joint_unscaled(model):
    """End-to-end: a direct joint key writes the raw command."""
    data = mujoco.MjData(model)
    mixin = RenderingMixin()
    mixin._apply_action_by_name(model, data, {"arm_joint": 0.5}, "", mujoco)
    arm = _aid(model, "arm_act")
    assert data.ctrl[arm] == pytest.approx(0.5)
