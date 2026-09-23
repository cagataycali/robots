"""Regression: warn when an action value is outside the range its actuator holds it to.

The direct-actuator branch of ``_apply_action_by_name`` writes the action value
verbatim to ``data.ctrl``. MuJoCo then holds that command to a range by one of
two mechanisms, and the commanded trajectory is silently NOT reproduced for the
actuator either way while the call reports success:

* a ``ctrllimited`` actuator's ``ctrlrange``, which ``mj_step`` clamps ``ctrl``
  into - the failure mode of replaying a dataset whose action units differ from
  the robot's ctrl units (a normalized gripper action in ``[0.19, 1.12]``
  replayed onto a joint-position gripper whose ctrlrange is ``[0.002, 0.037]``
  pins the gripper open and destroys the grasp channel);
* the range of the joint an UNLIMITED position servo drives. ``ctrl`` is not
  clamped, but ``ctrl`` IS the joint target and the joint cannot leave its
  range. Every so101 actuator is in that case - its shipped MJCF authors
  neither ``ctrlrange`` nor ``inheritrange`` - so a degrees-valued chunk applied
  as radians pinned all six joints at a limit with nothing logged.

These tests build a tiny synthetic model (no asset download) covering both
mechanisms plus the shapes that must stay silent, and pin that
``_apply_action_by_name`` warns exactly when a value is meaningfully outside the
bounds its actuator is held to.
"""

from __future__ import annotations

import logging

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402
from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    actuator_joint_id,
    effective_ctrl_range,
)

_LOGGER = "strands_robots.simulation.mujoco.rendering"

# arm_act: wide ctrlrange; grip_act: small ctrlrange mirroring the ALOHA
# gripper ([0.002, 0.037]); free_act: no ctrlrange on an unlimited joint;
# wrist_act: no ctrlrange on a LIMITED joint (the so101 shape, where the joint
# range is the effective bound); spin_vel: a rate drive on a limited joint,
# whose ctrl is not a joint coordinate at all. ``angle="radian"`` so the
# declared joint ranges are the radians the assertions read (MuJoCo's default
# is degrees, which would compile range="-1.5 1.5" to +-0.026 rad).
_XML = """
<mujoco model="clamp_test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link">
      <joint name="arm_joint" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0 0 0.2"/>
      <body name="finger" pos="0 0 0.2">
        <joint name="grip_joint" type="slide" axis="1 0 0" range="0 0.041"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
      <body name="spin" pos="0.1 0 0">
        <joint name="free_joint" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
      <body name="wrist" pos="-0.1 0 0">
        <joint name="wrist_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
      <body name="turret" pos="0 0.1 0">
        <joint name="turret_joint" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="arm_act" joint="arm_joint" ctrlrange="-3 3"/>
    <position name="grip_act" joint="grip_joint" ctrlrange="0.002 0.037" kp="10"/>
    <position name="free_act" joint="free_joint" kp="1"/>
    <position name="wrist_act" joint="wrist_joint" kp="10"/>
    <velocity name="turret_vel" joint="turret_joint" kv="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def _apply(model, action, mixin=None):
    data = mujoco.MjData(model)
    mixin = mixin or RenderingMixin()
    mixin._apply_action_by_name(model, data, action, "", mujoco, "")
    return mixin, data


def _warnings_naming(caplog, key: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and key in r.getMessage()]


def test_warns_when_gripper_value_outside_ctrlrange(model, caplog):
    """A normalized gripper action (0.5) far above ctrlrange [0.002, 0.037] warns."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 0.5})
    hit = [m for m in _warnings_naming(caplog, "grip_act") if "ctrlrange" in m]
    assert hit, "expected a clamp warning naming grip_act"
    # Names the actuator's actual ctrlrange so the user can self-correct.
    assert "0.002" in hit[0] and "0.037" in hit[0]
    assert "MuJoCo will clamp it" in hit[0]


def test_warns_when_value_exceeds_the_driven_joint_range(model, caplog):
    """An unlimited position servo is bounded by the joint it drives, and says so.

    The so101 shape: no ``ctrlrange`` on the actuator, a limited joint behind it.
    ``ctrl`` is never clamped, so the pre-fix warning skipped this actuator
    entirely and a degrees-valued command (23.61 for a +-1.5 rad joint) reached
    ``data.ctrl`` with nothing logged while the joint sat pinned at its limit.
    """
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"wrist_act": 23.61})
    hit = _warnings_naming(caplog, "wrist_act")
    assert hit, "expected an out-of-range warning naming wrist_act"
    assert "-1.5" in hit[0] and "1.5" in hit[0]
    assert "the joint cannot leave that range" in hit[0]
    assert "NOT reproduced" in hit[0]


def test_the_reported_bounds_are_the_shared_effective_range(model, caplog):
    """The warning reports the bounds ``set_gripper`` resolves, not a second answer.

    Both read
    :func:`~strands_robots.simulation.mujoco.scene_ops.effective_ctrl_range`, so
    the numbers in the warning are the ones a set-point would be driven to.
    Reading them in two places is how the warning came to skip every so101
    actuator while ``set_gripper`` resolved its range.
    """
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_act")
    bounds, source = effective_ctrl_range(model, mujoco, act_id, actuator_joint_id(model, act_id, mujoco))
    assert bounds is not None and source == "driven joint range"
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"wrist_act": 23.61})
    hit = _warnings_naming(caplog, "wrist_act")
    assert hit and f"[{bounds[0]:.4g}, {bounds[1]:.4g}]" in hit[0]


@pytest.mark.parametrize(
    ("key", "value", "why"),
    [
        pytest.param("grip_act", 0.02, "inside the actuator's ctrlrange", id="in-range-gripper"),
        pytest.param("arm_act", 1.0, "inside a wide ctrlrange", id="in-range-arm"),
        pytest.param("wrist_act", 1.0, "inside the driven joint's range", id="in-range-driven-joint"),
        pytest.param(
            "free_act",
            999.0,
            "unlimited actuator whose joint is unlimited too: nothing bounds the command",
            id="unlimited-actuator-and-joint",
        ),
        pytest.param(
            "turret_vel",
            999.0,
            "a rate drive's ctrl is not a joint coordinate, so the joint's limits do not bound it",
            id="rate-drive-on-a-limited-joint",
        ),
    ],
)
def test_no_warning_when_the_command_is_not_held_out_of_range(model, caplog, key, value, why):
    """A value the actuator is not holding out of range must stay silent: {why}."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {key: value})
    assert not _warnings_naming(caplog, key), why


def test_warn_once_dedup(model, caplog):
    """Two out-of-range commands for the same key warn only once (no 50Hz spam)."""
    mixin = RenderingMixin()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 0.5}, mixin)
        _apply(model, {"grip_act": 0.8}, mixin)
    assert len(_warnings_naming(caplog, "grip_act")) == 1


def test_no_warning_for_degenerate_ctrlrange(model, caplog):
    """A ctrl-limited actuator whose range is degenerate ([v, v]) never clamps
    meaningfully, so an out-of-range command must not warn.

    A ``[0, 0]`` (or any ``lo >= hi``) ctrlrange is a sentinel, not a real
    limit: MuJoCo would pin every command to the single point, but the warning
    is about *unit mismatch*, not about a legitimately degenerate actuator.
    Emitting it here would be noise. The range is forced on the compiled model
    (the MJCF compiler auto-clears ``ctrllimited`` for a zero range) to mirror
    the sentinel state seen in the wild.
    """
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_act")
    model.actuator_ctrlrange[grip_id] = [5.0, 5.0]
    model.actuator_ctrllimited[grip_id] = 1
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 100.0})
    assert not _warnings_naming(caplog, "grip_act")


def test_no_crash_for_stale_actuator_id(model, caplog):
    """A stale/out-of-range actuator id must be swallowed, not crash the loop.

    After a scene recompile an actuator id can outlive the model it indexed
    (fewer actuators than before). Indexing ``actuator_ctrllimited`` with it
    raises ``IndexError``; the 50Hz control loop must survive that -- the warn
    is best-effort diagnostics, never a hard dependency. No warning is emitted
    and no exception escapes.
    """
    mixin = RenderingMixin()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        mixin._warn_ctrl_clamp(model, model.nu + 99, "", "grip_act", 100.0, mujoco)
    assert not _warnings_naming(caplog, "grip_act")
