# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A scene mutation must not teleport the robot back to its rest pose.

``_rebuild`` finalizes a fresh ``Model`` then re-seeded ``_state_0`` from the MODEL
defaults (``eval_fk(model, model.joint_q, model.joint_qd, state_0)``), discarding
the live ``joint_q``/``joint_qd``. ``_rebuild`` is called by ``add_object``,
``remove_object``, ``move_object``, ``set_gravity``, ``add_robot`` and
``remove_robot`` - so any mid-rollout scene edit snapped every joint back to 0 and
zeroed all velocities. Only ``self._targets`` survived.

Measured pre-fix on so101 driven to joint "2" = 0.6393 rad::

    BEFORE add_object      2=0.6393 vel=0.1850 t=0.5167
    AFTER  add_object      2=0.0000 vel=0.0000 t=0.5167
    AFTER  move_object     2=0.0000 vel=0.0000 t=1.0333
    AFTER  set_gravity     2=0.0000 vel=0.0000 t=1.5500

``sim_time`` and ``step_count`` are NOT reset, so the engine reported a continuous
timeline over a discontinuous trajectory: a dataset recorded across an
``add_object`` contains a physically impossible jump with no marker.

It also contradicted the class's own documentation ("State is preserved across
rebuilds where joint names still exist") and the MuJoCo backend, the parity
reference, which preserves state exactly.

The fix snapshots per-(robot, joint) position and velocity through the OLD index
maps before they are cleared, then scatters them back through the NEW maps and
re-runs ``eval_fk`` so ``body_q`` matches. ``reset()`` and ``create_world()`` pass
``preserve_state=False`` - reset's contract IS the rest pose.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


def _driven_engine(target: float = 0.6):
    """An engine whose so101 is mid-motion, away from the rest pose."""
    sim = _engine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    joints = sim.robot_joint_names("so101")
    assert sim.send_action({joints[1]: target}, robot_name="so101")["status"] == "success"
    sim.step(40)
    observation = sim.get_observation("so101")
    position = float(observation[joints[1]])
    assert abs(position) > 0.1, f"fixture never left the rest pose (joint at {position})"
    return sim, joints, position


class TestSceneMutationsPreserveJointState:
    def test_add_object_does_not_reset_the_arm(self):
        """The regression: 0.6393 rad -> 0.0000 with sim_time unchanged."""
        sim, joints, before = _driven_engine()
        try:
            time_before = sim._world.sim_time

            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4), f"joint teleported {before:.4f} -> {after:.4f}"
            # The timeline must stay consistent with the trajectory.
            assert sim._world.sim_time == pytest.approx(time_before)
        finally:
            sim.destroy()

    def test_add_object_preserves_joint_velocity(self):
        """Position alone is not enough: a zeroed velocity is a discontinuity."""
        sim, joints, _ = _driven_engine()
        try:
            key = f"{joints[1]}.vel"
            before = float(sim.get_observation("so101")[key])
            assert abs(before) > 1e-3, f"fixture is not moving (vel {before})"

            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            after = float(sim.get_observation("so101")[key])
            assert after == pytest.approx(before, abs=1e-4), f"velocity zeroed {before:.4f} -> {after:.4f}"
        finally:
            sim.destroy()

    def test_move_object_does_not_reset_the_arm(self):
        sim, joints, before = _driven_engine()
        try:
            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            assert sim.move_object("box", position=[0.25, 0.0, 0.05])["status"] == "success"

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4)
        finally:
            sim.destroy()

    def test_remove_object_does_not_reset_the_arm(self):
        sim, joints, before = _driven_engine()
        try:
            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            assert sim.remove_object("box")["status"] == "success"

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4)
        finally:
            sim.destroy()

    def test_set_gravity_does_not_reset_the_arm(self):
        sim, joints, before = _driven_engine()
        try:
            assert sim.set_gravity([0.0, 0.0, -9.0])["status"] == "success"

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4)
        finally:
            sim.destroy()

    def test_a_surviving_robot_keeps_its_state_when_another_is_removed(self):
        """The index maps shift, so the scatter must go through the NEW ones."""
        sim = _engine()
        try:
            sim.create_world()
            assert sim.add_robot("so101")["status"] == "success"
            assert sim.add_robot("panda", position=[0.6, 0.0, 0.0])["status"] == "success"
            joints = sim.robot_joint_names("so101")
            assert sim.send_action({joints[1]: 0.5}, robot_name="so101")["status"] == "success"
            sim.step(40)
            before = float(sim.get_observation("so101")[joints[1]])

            assert sim.remove_robot("panda")["status"] == "success"

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4), f"{before:.4f} -> {after:.4f}"
        finally:
            sim.destroy()

    def test_adding_a_second_robot_keeps_the_first_ones_state(self):
        sim, joints, before = _driven_engine()
        try:
            assert sim.add_robot("panda", position=[0.6, 0.0, 0.0])["status"] == "success"

            after = float(sim.get_observation("so101")[joints[1]])
            assert after == pytest.approx(before, abs=1e-4)
        finally:
            sim.destroy()


class TestBodyPosesFollowTheRestoredJoints:
    def test_body_q_matches_the_restored_configuration(self):
        """eval_fk must re-run: body_q is what the renderer and pose queries read.

        Restoring joint_q without re-running forward kinematics would leave every
        link transform at the rest pose - the arm would look straight while
        reporting bent joints.
        """
        import numpy as np

        sim, _, _ = _driven_engine()
        try:
            before = sim._state_0.body_q.numpy().copy()

            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            after = sim._state_0.body_q.numpy().copy()
            shared = min(len(before), len(after))
            delta = float(np.abs(before[:shared] - after[:shared]).max())
            assert delta < 1e-5, f"body transforms moved by {delta}"
        finally:
            sim.destroy()

    def test_the_rendered_frame_is_not_the_rest_pose(self):
        """End-to-end: the pixels must show the same arm before and after."""
        import numpy as np

        sim, _, _ = _driven_engine()
        try:
            before = sim.get_frame("default", 96, 96)[0]

            assert sim.set_gravity([0.0, 0.0, -9.0])["status"] == "success"

            after = sim.get_frame("default", 96, 96)[0]
            # set_gravity adds no geometry, so with state preserved the frame is
            # identical; pre-fix the arm snapped to the rest pose.
            assert np.array_equal(before, after), (
                f"frame changed after a no-geometry rebuild (mean delta "
                f"{float(np.abs(before.astype(int) - after.astype(int)).mean()):.2f})"
            )
        finally:
            sim.destroy()


class TestResetStillDiscardsState:
    def test_reset_zeroes_position_and_velocity(self):
        """reset()'s contract IS the rest pose - it must NOT preserve."""
        sim, joints, _ = _driven_engine()
        try:
            assert sim.reset()["status"] == "success"

            observation = sim.get_observation("so101")
            assert float(observation[joints[1]]) == pytest.approx(0.0, abs=1e-4)
            assert float(observation[f"{joints[1]}.vel"]) == pytest.approx(0.0, abs=1e-4)
            assert sim._world.sim_time == pytest.approx(0.0)
        finally:
            sim.destroy()

    def test_reset_after_a_scene_mutation_still_zeroes(self):
        """Preservation must not leak into reset through a later rebuild."""
        sim, joints, _ = _driven_engine()
        try:
            assert (
                sim.add_object("box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1.0, 0.0, 0.0])[
                    "status"
                ]
                == "success"
            )

            assert sim.reset()["status"] == "success"

            assert float(sim.get_observation("so101")[joints[1]]) == pytest.approx(0.0, abs=1e-4)
        finally:
            sim.destroy()


class TestSnapshotEdgeCases:
    def test_a_jointless_world_snapshots_nothing(self):
        """Before the first add_robot the State's joint arrays are None."""
        sim = _engine()
        try:
            sim.create_world()

            assert sim._snapshot_joint_state() == {}
            # And a mutation on that world must not raise.
            assert (
                sim.add_object("box", position=[0.0, 0.0, 0.30], size=[0.02, 0.02, 0.02], mass=0.2)["status"]
                == "success"
            )
        finally:
            sim.destroy()

    def test_restoring_an_empty_snapshot_is_a_no_op(self):
        sim, joints, before = _driven_engine()
        try:
            sim._restore_joint_state({})

            assert float(sim.get_observation("so101")[joints[1]]) == pytest.approx(before, abs=1e-6)
        finally:
            sim.destroy()

    def test_a_snapshot_key_that_no_longer_exists_is_dropped(self):
        """A stale key must be ignored, not raise or corrupt an index."""
        sim, joints, before = _driven_engine()
        try:
            sim._restore_joint_state({("ghost_robot", "ghost_joint"): (1.23, 4.56)})

            assert float(sim.get_observation("so101")[joints[1]]) == pytest.approx(before, abs=1e-6)
        finally:
            sim.destroy()

    def test_the_snapshot_covers_every_scalar_joint(self):
        sim, joints, _ = _driven_engine()
        try:
            snapshot = sim._snapshot_joint_state()

            assert {key[1] for key in snapshot} == set(joints)
            assert all(key[0] == "so101" for key in snapshot)
        finally:
            sim.destroy()
