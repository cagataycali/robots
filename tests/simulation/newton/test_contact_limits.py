"""Contact/constraint buffer sizing for the Newton MuJoCo-Warp solver.

The Newton backend previously built its solver as ``solver_cls(self._model)``,
leaving mujoco-warp to pick its own contact (``nconmax``) and constraint-row
(``njmax``) buffer sizes. For a 6-DoF arm that default is ``njmax=64``, and the
first actuated pose that brings links into contact overflowed the broadphase
("broadphase overflow - please increase nconmax to 49"). mujoco-warp cannot grow
those buffers, so the overflow was not a degraded step: the next constraint
kernel read past the end of the array and aborted with "CUDA error 700: an
illegal memory access was encountered", poisoning the CUDA context for the rest
of the process. Stepping an UNACTUATED world never tripped it, so the symptom
was "send_action kills the simulator".

These tests pin that the solver is built with real headroom, that the size
tracks the scene, that callers can override it, and - the actual regression -
that ``send_action`` followed by sustained stepping completes and tracks its
target.

Gated on Newton + Warp: the regression test steps the real GPU/CPU solver.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _make_engine(**kwargs):
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver="mujoco", **kwargs)


def _joint_q(engine, n: int) -> list[float]:
    """Return the first ``n`` generalized coordinates as floats."""
    return [float(v) for v in engine._state_0.joint_q.numpy()[:n]]


class TestContactLimitsApplied:
    def test_mujoco_solver_gets_headroom_far_above_the_default(self):
        """The solver must be built with far more than mujoco-warp's own default.

        mujoco-warp defaults to njmax=64 for a 6-DoF arm, which the SO-101
        overflows as soon as it is actuated into a self-contacting pose.
        """
        from strands_robots.simulation.newton.simulation import _CONTACT_LIMIT_FLOOR

        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            limits = sim._solver_contact_limits(type(sim._solver))
            assert limits["nconmax"] >= _CONTACT_LIMIT_FLOOR
            assert limits["njmax"] >= _CONTACT_LIMIT_FLOOR
        finally:
            sim.destroy()

    def test_limits_scale_with_the_scene_collision_pairs(self):
        """A scene with more collision pairs must not get a smaller budget."""
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            bare = sim._solver_contact_limits(type(sim._solver))["nconmax"]
            for i in range(4):
                sim.add_object(f"box{i}", shape="box", position=[0.2 + 0.1 * i, 0.0, 0.05], size=[0.02, 0.02, 0.02])
            crowded = sim._solver_contact_limits(type(sim._solver))["nconmax"]
            assert crowded >= bare
        finally:
            sim.destroy()

    def test_explicit_override_wins(self):
        sim = _make_engine(nconmax=7777, njmax=8888)
        try:
            sim.create_world()
            sim.add_robot("so101")
            limits = sim._solver_contact_limits(type(sim._solver))
            assert limits == {"nconmax": 7777, "njmax": 8888}
        finally:
            sim.destroy()

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "1024"])
    def test_invalid_limit_rejected_at_construction(self, bad):
        """A bad limit must fail next to the caller, not at the first add_robot."""
        with pytest.raises(ValueError, match="positive integer"):
            _make_engine(nconmax=bad)
        with pytest.raises(ValueError, match="positive integer"):
            _make_engine(njmax=bad)

    def test_non_mujoco_solver_gets_no_contact_kwargs(self):
        """Only MuJoCo-Warp accepts these kwargs; others must not receive them.

        Forwarding them blindly would raise TypeError inside the solver ctor.
        """
        import newton  # type: ignore[import-not-found]

        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            assert sim._solver_contact_limits(newton.solvers.SolverXPBD) == {}
        finally:
            sim.destroy()


class TestActuatedRolloutSurvives:
    def test_send_action_then_sustained_stepping_does_not_fault(self):
        """Regression: pre-fix this aborted a CUDA kernel and poisoned the context.

        Commands a pose that brings the SO-101's own links into contact, then
        steps far longer than the pre-fix failure point (which was within the
        first few hundred steps of the first actuated step).
        """
        sim = _make_engine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            target = {"1": 0.5, "2": -0.5, "3": 0.5, "4": 0.0, "5": 0.0, "6": 0.0}
            assert sim.send_action(target)["status"] == "success"
            for _ in range(6):
                assert sim.step(n_steps=200)["status"] == "success"
            # Physics must also be right, not merely non-crashing: a position
            # servo has to converge on the commanded target.
            reached = _joint_q(sim, 6)
            for index, name in enumerate(("1", "2", "3")):
                assert reached[index] == pytest.approx(target[name], abs=0.1)
        finally:
            sim.destroy()
