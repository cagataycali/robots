# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``set_timestep`` / ``set_gravity`` must survive the next scene recompile.

Both setters wrote only the COMPILED model (``model.opt``) and the Python-side
``world.timestep`` / ``world.gravity``. ``SpecBuilder.build`` sets
``spec.option.timestep`` / ``.gravity`` ONCE at ``create_world``, and every later
mutation (``add_robot``, ``add_object``, ``remove_object``, ``add_camera``,
``move_object`` on a static body, ``attach_bodies(weld)``, ``actuate_robot``)
recompiles from that spec - so the option block reverted to MuJoCo's defaults
dt=0.002 / g=-9.81.

The failure was silent in the worst way: ``world.timestep`` kept the requested
value, so ``get_state()``, ``describe()`` and ``physics_timestep()`` all went on
REPORTING it while ``mj_step`` integrated at the default. Downstream,
``PolicyRunner._control_substeps`` divides the control period by the REPORTED
timestep, so a 4x-wrong dt yields a 4x-wrong substep count and a nominal 1.0s
rollout integrates 4.0s of physics.

Both setters are in ``tool_spec.json``, i.e. an LLM can call them.

These tests pin the invariant at the level that matters - what ``mj_step``
actually uses after a mutation - rather than what the engine reports.

Gated on mujoco: every assertion compiles and steps a real model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_DT = 0.0005
_GZ = -1.62  # lunar gravity: distinctive, and nothing else in the stack uses it


def _sim_with_explicit_physics() -> MuJoCoSimEngine:
    """A world whose dt/gravity were set through the SETTERS, not create_world.

    create_world's own values reach the spec directly and were never the bug; the
    defect is specific to the setter path.
    """
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.set_timestep(_DT)["status"] == "success"
    assert sim.set_gravity([0.0, 0.0, _GZ])["status"] == "success"
    return sim


def _assert_physics_honoured(sim: MuJoCoSimEngine, label: str) -> None:
    model = sim._world._model  # type: ignore[union-attr]
    assert model.opt.timestep == pytest.approx(_DT), f"{label}: compiled dt reverted"
    assert model.opt.gravity[2] == pytest.approx(_GZ), f"{label}: compiled gravity reverted"
    # The reported value must agree with the compiled one; the old bug was
    # precisely that these two diverged.
    assert sim.physics_timestep() == pytest.approx(_DT), f"{label}: reported dt disagrees"


class TestOptionSurvivesEachMutation:
    def test_add_robot(self):
        sim = _sim_with_explicit_physics()
        try:
            assert sim.add_robot("so101")["status"] == "success"
            _assert_physics_honoured(sim, "add_robot")
        finally:
            sim.destroy()

    def test_add_object(self):
        sim = _sim_with_explicit_physics()
        try:
            assert sim.add_object("box", shape="box", position=[0.3, 0.0, 0.05], size=[0.02] * 3)["status"] == "success"
            _assert_physics_honoured(sim, "add_object")
        finally:
            sim.destroy()

    def test_remove_object(self):
        sim = _sim_with_explicit_physics()
        try:
            sim.add_object("box", shape="box", position=[0.3, 0.0, 0.05], size=[0.02] * 3)
            assert sim.remove_object("box")["status"] == "success"
            _assert_physics_honoured(sim, "remove_object")
        finally:
            sim.destroy()

    def test_add_camera(self):
        sim = _sim_with_explicit_physics()
        try:
            assert sim.add_camera("cam", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1])["status"] == "success"
            _assert_physics_honoured(sim, "add_camera")
        finally:
            sim.destroy()

    def test_several_mutations_in_sequence(self):
        """The compiled value must not oscillate depending on the last mutation."""
        sim = _sim_with_explicit_physics()
        try:
            sim.add_robot("so101")
            sim.add_object("box", shape="box", position=[0.3, 0.0, 0.05], size=[0.02] * 3)
            sim.add_camera("cam", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1])
            sim.remove_object("box")
            _assert_physics_honoured(sim, "mutation sequence")
        finally:
            sim.destroy()


class TestMeasuredPhysicsNotJustReportedValues:
    def test_stepping_advances_time_by_the_configured_timestep(self):
        """The assertion that cannot be satisfied by a stale report."""
        sim = _sim_with_explicit_physics()
        try:
            sim.add_robot("so101")
            start = sim._world._data.time
            sim.step(n_steps=10)

            measured = (sim._world._data.time - start) / 10
            assert measured == pytest.approx(_DT), f"mj_step used dt={measured}, not {_DT}"
        finally:
            sim.destroy()

    def test_gravity_actually_drives_the_dynamics(self):
        """A body must fall at the configured gravity, not the default."""
        sim = MuJoCoSimEngine()
        try:
            sim.create_world(ground_plane=False)
            assert sim.set_gravity([0.0, 0.0, 0.0])["status"] == "success"
            sim.add_object("ball", shape="sphere", position=[0.0, 0.0, 1.0], size=[0.05], mass=0.2)

            z_before = float(sim._world._data.qpos[2])
            sim.step(n_steps=100)

            # Zero gravity: the ball must not fall. Pre-fix add_object restored
            # g=-9.81 and it dropped.
            assert float(sim._world._data.qpos[2]) == pytest.approx(z_before, abs=1e-3)
        finally:
            sim.destroy()


class TestRolloutHorizonIsNotScaled:
    def test_run_policy_duration_matches_wall_clock_sim_time(self):
        """A wrong reported dt gives PolicyRunner a wrong substep count.

        ``_control_substeps`` derives its count from ``physics_timestep()``, so a
        reported dt 4x smaller than the compiled one made a nominal 1.0s rollout
        integrate 4.0s.
        """
        from strands_robots.policies.base import Policy

        joints = [str(i) for i in range(1, 7)]

        class _Hold(Policy):
            @property
            def provider_name(self) -> str:
                return "hold"

            def set_robot_state_keys(self, keys) -> None:
                pass

            async def get_actions(self, observation, instruction, **kwargs):
                return [dict.fromkeys(joints, 0.0)]

        sim = _sim_with_explicit_physics()
        try:
            sim.add_robot("so101")
            start = sim._world._data.time
            sim.run_policy(policy_object=_Hold(), robot_name="so101", duration=1.0, control_frequency=50.0)

            assert sim._world._data.time - start == pytest.approx(1.0, abs=0.05)
        finally:
            sim.destroy()
