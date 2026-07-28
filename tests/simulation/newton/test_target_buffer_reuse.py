"""Joint targets must be written in place, not reallocated every control step.

``_write_targets`` used to do a device-to-host copy, mutate the host copy, then
bind a BRAND NEW ``wp.array`` onto ``Control.joint_target_q``. It runs once per
``send_action`` - i.e. once per control step of every policy rollout - so the
allocation churn was on the hot path (measured ~153us per call for a 6-DoF arm,
versus ~11us writing in place).

Rebinding is also what blocks CUDA-graph capture, the optimization Newton exists
to provide: a captured graph records the buffer addresses it was traced with, so
a pointer that changes every step can never be captured. Newton's
``solver.step(state_in, state_out, control, contacts, dt)`` takes ``control`` as
an argument on each call, so the old rebind was not producing WRONG physics -
this is a cost and churn defect, and the tests below pin the pointer stability
rather than claiming a correctness fix.

Gated on Newton + Warp.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _make_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco")


def _target_ptr(sim) -> int:
    return int(sim._control.joint_target_q.ptr)


class TestBufferIsReused:
    def test_repeated_actions_keep_one_device_allocation(self):
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            pointers = set()
            for i in range(25):
                sim.send_action({"1": 0.01 * i}, n_substeps=1)
                pointers.add(_target_ptr(sim))

            assert len(pointers) == 1, f"buffer was reallocated {len(pointers)} times"
        finally:
            sim.destroy()

    def test_host_staging_array_is_reused_not_rebuilt(self):
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            sim.send_action({"1": 0.1}, n_substeps=1)
            first = sim._target_host
            assert first is not None
            sim.send_action({"1": 0.2}, n_substeps=1)

            assert sim._target_host is first
        finally:
            sim.destroy()


class TestValuesStillLand:
    def test_written_target_reaches_the_device_buffer(self):
        """In-place must actually transfer: assign(), not a host-only mutation."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            sim.send_action({"1": 0.42}, n_substeps=1)

            idx = sim._joint_coord_index[("so101", "1")]
            assert sim._control.joint_target_q.numpy()[idx] == pytest.approx(0.42, abs=1e-6)
        finally:
            sim.destroy()

    def test_later_action_overwrites_the_earlier_value(self):
        """A stale staging array would leave the previous target in place."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)
            idx = sim._joint_coord_index[("so101", "1")]

            sim.send_action({"1": 0.30}, n_substeps=1)
            sim.send_action({"1": -0.30}, n_substeps=1)

            assert sim._control.joint_target_q.numpy()[idx] == pytest.approx(-0.30, abs=1e-6)
        finally:
            sim.destroy()

    def test_untouched_joints_keep_their_targets(self):
        """Writing one joint must not zero the others (the staging array persists)."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            sim.send_action({"2": 0.25}, n_substeps=1)
            sim.send_action({"3": -0.15}, n_substeps=1)

            buf = sim._control.joint_target_q.numpy()
            assert buf[sim._joint_coord_index[("so101", "2")]] == pytest.approx(0.25, abs=1e-6)
            assert buf[sim._joint_coord_index[("so101", "3")]] == pytest.approx(-0.15, abs=1e-6)
        finally:
            sim.destroy()

    def test_servo_still_converges_on_the_commanded_pose(self):
        """End to end: the optimization must not change the physics outcome."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)

            sim.send_action({"1": 0.5, "2": -0.4, "3": 0.4, "4": 0.0, "5": 0.0, "6": 0.0})
            sim.step(n_steps=1200)

            reached = sim._state_0.joint_q.numpy()
            assert float(reached[sim._joint_coord_index[("so101", "1")]]) == pytest.approx(0.5, abs=0.1)
        finally:
            sim.destroy()


class TestRebuildInvalidatesTheStagingArray:
    def test_adding_a_robot_resizes_the_staging_array(self):
        """A rebuild changes the DOF count; a stale host array would be wrong-sized."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)
            sim.send_action({"1": 0.2}, robot_name="so101", n_substeps=1)
            first_len = len(sim._target_host)

            # A second robot rebuilds the model -> a longer joint_target_q.
            assert sim.add_robot("so100")["status"] == "success"
            sim.send_action({"Rotation": 0.3}, robot_name="so100", n_substeps=1)

            assert len(sim._target_host) > first_len
            assert len(sim._target_host) == len(sim._control.joint_target_q.numpy())
        finally:
            sim.destroy()

    def test_targets_survive_a_rebuild(self):
        """_rebuild re-applies live targets through the new buffer."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.step(n_steps=1)
            sim.send_action({"1": 0.35}, robot_name="so101", n_substeps=1)

            sim.add_object("box", shape="box", position=[0.3, 0.0, 0.05], size=[0.02, 0.02, 0.02])

            idx = sim._joint_coord_index[("so101", "1")]
            assert sim._control.joint_target_q.numpy()[idx] == pytest.approx(0.35, abs=1e-6)
        finally:
            sim.destroy()
