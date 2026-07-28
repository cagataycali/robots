"""Joint targets must be addressed by DOF index, not coordinate index.

``Control.joint_target_q`` is DOF-shaped (``joint_dof_count``) unless
``newton.use_coord_layout_targets`` is enabled -- it defaults to False and this
backend never sets it. ``_write_targets`` indexed that buffer with
``_joint_coord_index``, and the two layouts diverge as soon as the model holds a
multi-coordinate joint: a FREE joint spans 7 coordinates but only 6 DOFs (a BALL
joint 4 and 3), so every joint after a floating base has
``coord_index == dof_index + 1``.

Measured on unitree_g1 before the fix (joint_coord_count=36, joint_dof_count=35,
target buffer length 35):

* commanding ``left_hip_pitch_joint`` (dof 6) wrote into dof 7, so
  ``left_hip_roll_joint`` tracked the target instead -- over 25 control steps the
  wrong joint moved +0.969 while the commanded one moved +0.032;
* the final joint's coordinate index (35) fell off the end of the 35-element
  buffer, so its command was silently discarded by the bounds guard;
* a plain SO-101 arm was corrupted merely by sharing a world with a floating-base
  robot added before it.

The engine already maintained the correct ``_joint_dof_index`` (built from
``builder.joint_qd_start``) and used it for READING ``joint_qd``; it was simply
never used on the write side.

Gated on Newton + Warp: every assertion here needs the real model layout.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")

_FLOATING_ROBOT = "unitree_g1"


def _make_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


def _nonzero_dofs(sim) -> list[int]:
    return [i for i, v in enumerate(sim._control.joint_target_q.numpy()) if abs(float(v)) > 1e-6]


def _reset_targets(sim) -> None:
    """Clear pending targets and both buffers they persist in.

    Three places hold a commanded target, and a test asserting on the exact
    nonzero set has to clear all of them:

    * ``_targets`` - the pending ``{(robot, joint): value}`` map;
    * ``_control.joint_target_q`` - the device buffer;
    * ``_target_host`` - the host staging array ``_write_targets`` reuses, which
      would otherwise re-push the previous value over a zeroed device buffer.

    The persistence itself is deliberate (it is what lets one joint be commanded
    without zeroing the others); only the test needs a clean slate.
    """
    sim._targets.clear()
    sim._control.joint_target_q.zero_()
    sim._target_host = None


@pytest.fixture(scope="module")
def g1():
    sim = _make_engine()
    sim.create_world()
    if sim.add_robot(_FLOATING_ROBOT)["status"] != "success":
        sim.destroy()
        pytest.skip(f"{_FLOATING_ROBOT} asset unavailable")
    sim.step(n_steps=1)
    yield sim
    sim.destroy()


class TestLayoutAssumption:
    def test_target_buffer_is_dof_shaped_not_coordinate_shaped(self, g1):
        """The premise: the two counts differ, and the buffer follows DOFs."""
        import newton  # type: ignore[import-not-found]

        assert not getattr(newton, "use_coord_layout_targets", False)
        assert g1._model.joint_coord_count != g1._model.joint_dof_count
        assert len(g1._control.joint_target_q.numpy()) == g1._model.joint_dof_count

    def test_indices_diverge_after_the_floating_base(self, g1):
        joints = g1._world.robots[_FLOATING_ROBOT].joint_names
        base = g1._robot_free_base_joint.get(_FLOATING_ROBOT)
        after_base = [j for j in joints if j != base]

        assert after_base, "expected joints after the floating base"
        for name in after_base:
            coord = g1._joint_coord_index[(_FLOATING_ROBOT, name)]
            dof = g1._joint_dof_index[(_FLOATING_ROBOT, name)]
            assert coord == dof + 1, f"{name}: coord={coord} dof={dof}"


class TestTargetLandsOnTheCommandedJoint:
    def test_command_writes_the_commanded_joints_dof(self, g1):
        """Regression: this used to write the NEXT joint's DOF."""
        name = "left_hip_pitch_joint"
        expected = g1._joint_dof_index[(_FLOATING_ROBOT, name)]

        _reset_targets(g1)
        g1.send_action({name: 1.0}, robot_name=_FLOATING_ROBOT, n_substeps=1)

        assert _nonzero_dofs(g1) == [expected]

    def test_last_joints_command_is_not_dropped(self, g1):
        """Its coordinate index was past the end of the DOF-shaped buffer."""
        name = g1._world.robots[_FLOATING_ROBOT].joint_names[-1]
        expected = g1._joint_dof_index[(_FLOATING_ROBOT, name)]

        _reset_targets(g1)
        g1.send_action({name: 0.7}, robot_name=_FLOATING_ROBOT, n_substeps=1)

        assert _nonzero_dofs(g1) == [expected]

    def test_floating_base_key_writes_nothing(self, g1):
        """A 6-DoF base is not a scalar target; one float there is a base command."""
        base = g1._robot_free_base_joint.get(_FLOATING_ROBOT)
        assert base is not None

        _reset_targets(g1)
        before = g1._control.joint_target_q.numpy().copy()
        g1.send_action({base: 5.0}, robot_name=_FLOATING_ROBOT, n_substeps=1)

        assert np.allclose(before, g1._control.joint_target_q.numpy())


class TestPhysicsFollowsTheCommandedJoint:
    def test_commanded_joint_moves_and_its_neighbour_does_not(self):
        """End to end: pre-fix the neighbour moved +0.969 and the target +0.032."""
        sim = _make_engine()
        try:
            sim.create_world()
            if sim.add_robot(_FLOATING_ROBOT)["status"] != "success":
                pytest.skip(f"{_FLOATING_ROBOT} asset unavailable")
            sim.step(n_steps=1)

            def q(name: str) -> float:
                return float(sim._state_0.joint_q.numpy()[sim._joint_coord_index[(_FLOATING_ROBOT, name)]])

            target, neighbour = "left_hip_pitch_joint", "left_hip_roll_joint"
            q0_target, q0_neighbour = q(target), q(neighbour)

            sim.send_action({target: 1.0}, robot_name=_FLOATING_ROBOT)
            sim.step(n_steps=25)

            moved_target = abs(q(target) - q0_target)
            moved_neighbour = abs(q(neighbour) - q0_neighbour)
            assert moved_target > 0.1, f"commanded joint barely moved ({moved_target:.4f})"
            assert moved_target > 3 * moved_neighbour, (
                f"neighbour moved comparably: target {moved_target:.4f} vs neighbour {moved_neighbour:.4f}"
            )
        finally:
            sim.destroy()


class TestFixedBaseArmInAMixedWorld:
    def test_arm_after_a_floating_base_robot_is_addressed_correctly(self):
        """A plain SO-101 was corrupted just by sharing a world with a humanoid."""
        sim = _make_engine()
        try:
            sim.create_world()
            if sim.add_robot(_FLOATING_ROBOT)["status"] != "success":
                pytest.skip(f"{_FLOATING_ROBOT} asset unavailable")
            assert sim.add_robot("so101")["status"] == "success"
            sim.step(n_steps=1)

            # The arm's own indices must have diverged for this to be meaningful.
            assert sim._joint_coord_index[("so101", "2")] != sim._joint_dof_index[("so101", "2")]

            _reset_targets(sim)
            sim.send_action({"2": 0.7}, robot_name="so101", n_substeps=1)

            assert _nonzero_dofs(sim) == [sim._joint_dof_index[("so101", "2")]]
        finally:
            sim.destroy()

    def test_plain_arm_alone_is_unaffected(self):
        """With no multi-coordinate joint the two layouts agree; no behaviour change."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            assert sim._joint_coord_index[("so101", "1")] == sim._joint_dof_index[("so101", "1")]
            sim.send_action({"1": 0.4}, n_substeps=1)

            assert _nonzero_dofs(sim) == [sim._joint_dof_index[("so101", "1")]]
        finally:
            sim.destroy()
