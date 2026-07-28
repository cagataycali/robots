"""Non-MuJoCo Newton solvers must actually see collisions.

``_advance`` called ``solver.step(state_in, state_out, control, None, dt)`` with the
4th argument - ``contacts`` - hard-coded ``None``. ``SolverMuJoCo`` runs the
MuJoCo-Warp collision pipeline internally and ignores it, so the default worked.
Every OTHER rigid solver in ``solver_registry()`` consumes the passed-in
``Contacts`` object and therefore saw no contacts at all, while
``create_simulation("newton", solver="featherstone")`` reported success, rendered
and stepped.

Reproduced - a 0.2 kg 6 cm cube dropped from z=0.30 onto the ground plane, 2.4 s::

    mujoco         z 0.3000 -> +0.0299   RESTS ON GROUND
    featherstone   z 0.3000 -> -27.9128  FELL THROUGH GROUND
    xpbd           z 0.3000 -> -27.9128  FELL THROUGH GROUND

-27.91 m is exactly free-fall for 2.4 s at 9.81 m/s^2, i.e. collision was entirely
absent. The engine now allocates ``model.contacts()`` for solvers that need it and
calls ``model.collide`` before each substep.

Gated on Newton + Warp: every assertion steps the real solver.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")

# The rigid-body solvers a user can select for an articulated scene.
_RIGID_SOLVERS = ("featherstone", "xpbd", "semi_implicit")
_DROP_HEIGHT = 0.30
_DROP_SECONDS = 2.4


def _make_engine(solver: str):
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return NewtonSimEngine(solver=solver)


#: Cube mass for the drop test. NOT 0.2 kg: the penalty-contact solvers
#: (featherstone / semi_implicit) use ``ShapeConfig.ke = 2500`` and over-impulse a
#: body that light, launching a 0.2 kg cube to z=+1.44 instead of resting it.
#: Verified to be Newton's own behaviour, not ours - reproduced with pure newton
#: calls and no strands code at all:
#:
#:     pure-newton featherstone mass=0.20 -> z=1.4440
#:     pure-newton featherstone mass=1.00 -> z=0.0290
#:
#: This test used to pass at 0.2 kg only by accident: ``add_object`` ACCUMULATED
#: the shape's density-derived mass on top of the requested one, so the body
#: really weighed 0.416 kg - just above the stability threshold measured here
#: (0.2 -> z=1.44, 0.416 -> z=0.031). Fixing that mass (D20) exposed the real
#: limitation. This test is about CONTACTS EXISTING for these solvers, so it uses
#: a mass they can integrate; the mass contract itself is pinned by
#: test_object_mass_fidelity.py.
_CUBE_MASS = 1.0


def _drop_cube(sim, mass: float = _CUBE_MASS) -> float:
    """Drop a cube onto the ground plane and return its settled world-z."""
    sim.create_world()
    sim.add_object("cube", shape="box", position=[0.0, 0.0, _DROP_HEIGHT], size=[0.03] * 3, mass=mass)
    sim.step(n_steps=int(_DROP_SECONDS / sim._world.timestep))
    return float(sim._state_0.body_q.numpy()[0][2])


class TestCollisionActuallyHappens:
    @pytest.mark.parametrize("solver", _RIGID_SOLVERS)
    def test_a_dropped_cube_rests_on_the_ground(self, solver):
        """Regression: these solvers let the cube fall to -27.91 m."""
        sim = _make_engine(solver)
        try:
            settled_z = _drop_cube(sim)

            assert settled_z > -0.5, f"{solver}: cube fell through the ground to z={settled_z:.4f}"
            # Resting on a 6 cm cube's half-height, not merely "slowed down".
            assert settled_z == pytest.approx(0.03, abs=0.02), f"{solver}: z={settled_z:.4f}"
        finally:
            sim.destroy()

    def test_the_mujoco_solver_still_rests_it(self):
        """The default path must be unchanged."""
        sim = _make_engine("mujoco")
        try:
            assert _drop_cube(sim) == pytest.approx(0.03, abs=0.02)
        finally:
            sim.destroy()

    def test_the_default_solver_rests_a_light_cube_too(self):
        """The mass the penalty solvers cannot hold: MuJoCo-Warp must be fine.

        Pins that ``_CUBE_MASS`` is a workaround for the penalty solvers only,
        not a mass the default backend needs.
        """
        sim = _make_engine("mujoco")
        try:
            assert _drop_cube(sim, mass=0.2) == pytest.approx(0.03, abs=0.02)
        finally:
            sim.destroy()

    @pytest.mark.parametrize("solver", _RIGID_SOLVERS)
    def test_free_fall_is_ruled_out_explicitly(self, solver):
        """Pin the exact pre-fix number so the failure mode cannot silently return."""
        sim = _make_engine(solver)
        try:
            free_fall_z = _DROP_HEIGHT - 0.5 * 9.81 * _DROP_SECONDS**2

            assert _drop_cube(sim) > free_fall_z + 1.0
        finally:
            sim.destroy()


class TestContactsBufferOwnership:
    def test_mujoco_supplies_its_own_contacts(self):
        """No buffer is allocated for the solver that collides internally."""
        sim = _make_engine("mujoco")
        try:
            sim.create_world()
            sim.add_robot("so101")

            assert sim._contacts is None
        finally:
            sim.destroy()

    @pytest.mark.parametrize("solver", _RIGID_SOLVERS)
    def test_other_solvers_get_a_buffer(self, solver):
        sim = _make_engine(solver)
        try:
            sim.create_world()
            sim.add_robot("so101")

            assert sim._contacts is not None, f"{solver} needs a Contacts buffer"
        finally:
            sim.destroy()

    def test_the_buffer_is_reallocated_on_rebuild(self):
        """A rebuild finalises a new Model, so the old buffer is stale."""
        sim = _make_engine("xpbd")
        try:
            sim.create_world()
            sim.add_robot("so101")
            first = sim._contacts

            sim.add_object("box", shape="box", position=[0.3, 0.0, 0.05], size=[0.02] * 3)

            assert sim._contacts is not None
            assert sim._contacts is not first
        finally:
            sim.destroy()

    def test_an_empty_world_needs_no_buffer(self):
        """No solver is built until a robot exists, so there is nothing to collide."""
        sim = _make_engine("xpbd")
        try:
            sim.create_world()

            assert sim._contacts is None
        finally:
            sim.destroy()


class TestActuationStillWorksWithContacts:
    @pytest.mark.slow
    def test_a_commanded_joint_still_tracks_under_a_rigid_solver(self):
        """Refreshing contacts per substep must not break the actuation path.

        Deliberately only a handful of steps. Featherstone on a 6-DOF arm with
        per-substep collision costs about 4.75 s PER CONTROL STEP on this host
        (10 substeps x 33 collision pairs), so a convergence-length rollout is not
        a test, it is a benchmark. Correctness of the tracking itself is covered by
        the MuJoCo-solver tests; this only asserts the contacts refresh does not
        break the actuation call path.
        """
        sim = _make_engine("featherstone")
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.send_action({"1": 0.4}, n_substeps=1)

            assert sim.step(n_steps=2)["status"] == "success"
        finally:
            sim.destroy()
