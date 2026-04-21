"""Tests for NewtonSimulation class behaviour.

Tests the skeleton class — instantiation, cleanup, SimEngine conformance,
and that unimplemented methods raise NotImplementedError with clear messages.
No GPU required.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.simulation import NewtonSimulation


class TestNewtonSimulationInstantiation:
    """NewtonSimulation can be instantiated without GPU."""

    def test_default_config(self) -> None:
        sim = NewtonSimulation()
        assert isinstance(sim, SimEngine)
        assert sim._config.solver == "mujoco"
        assert sim._config.num_envs == 1

    def test_custom_config(self) -> None:
        config = NewtonConfig(num_envs=4096, solver="xpbd", device="cpu")
        sim = NewtonSimulation(config=config)
        assert sim._config.num_envs == 4096
        assert sim._config.solver == "xpbd"

    def test_kwargs_flow_to_config(self) -> None:
        """Factory kwargs should populate config."""
        sim = NewtonSimulation(num_envs=128, solver="semi_implicit")
        assert sim._config.num_envs == 128
        assert sim._config.solver == "semi_implicit"

    def test_initial_state_flags(self) -> None:
        sim = NewtonSimulation()
        assert sim._world_created is False
        assert sim._replicated is False
        assert sim._step_count == 0
        assert sim._sim_time == 0.0
        assert sim._pending_actions is None
        assert sim._robots == {}
        assert sim._objects == {}

    def test_is_simengine_subclass(self) -> None:
        assert issubclass(NewtonSimulation, SimEngine)


class TestNewtonSimulationContextManager:
    """Context manager protocol (__enter__/__exit__) works."""

    def test_context_manager(self) -> None:
        with NewtonSimulation() as sim:
            assert isinstance(sim, NewtonSimulation)
        # cleanup should have been called (no error)


class TestNewtonSimulationCleanup:
    """cleanup() should be safe to call multiple times."""

    def test_cleanup_idempotent(self) -> None:
        sim = NewtonSimulation()
        sim.cleanup()
        sim.cleanup()  # second call should not raise


class TestNewtonSimulationStubs:
    """All ABC methods must raise NotImplementedError with clear messages.

    These are skeleton stubs — real implementations come in subsequent PRs.
    Tests verify the fail-fast contract: callers get clear errors about
    what's not yet available.
    """

    @pytest.fixture()
    def sim(self) -> NewtonSimulation:
        return NewtonSimulation()

    # --- World lifecycle ---

    def test_create_world_not_implemented(self, sim: NewtonSimulation) -> None:
        """create_world requires warp, which is not installed in unit tests."""
        with pytest.raises((NotImplementedError, ImportError)):
            sim.create_world()

    def test_destroy_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="destroy"):
            sim.destroy()

    def test_reset_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="reset"):
            sim.reset()

    def test_step_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="step"):
            sim.step()

    def test_get_state_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="get_state"):
            sim.get_state()

    # --- Robot management ---

    def test_add_robot_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_robot"):
            sim.add_robot("so100")

    def test_remove_robot_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="remove_robot"):
            sim.remove_robot("so100")

    # --- Object management ---

    def test_add_object_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_object"):
            sim.add_object("cube")

    def test_remove_object_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="remove_object"):
            sim.remove_object("cube")

    # --- Observation / Action ---

    def test_get_observation_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="get_observation"):
            sim.get_observation("so100")

    def test_send_action_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="send_action"):
            sim.send_action({"joint_0": 0.5})

    # --- Rendering ---

    def test_render_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="render"):
            sim.render()

    # --- Optional overrides ---

    def test_load_scene_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="load_scene"):
            sim.load_scene("scene.usd")

    def test_run_policy_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="run_policy"):
            sim.run_policy("so100")

    def test_get_contacts_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="get_contacts"):
            sim.get_contacts()

    # --- Newton extensions ---

    def test_replicate_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="replicate"):
            sim.replicate(4096)

    def test_run_diffsim_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="run_diffsim"):
            sim.run_diffsim(100, lambda s: 0.0, "velocity")

    def test_solve_ik_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="solve_ik"):
            sim.solve_ik("so100", [0.3, 0, 0.2])

    def test_add_cloth_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_cloth"):
            sim.add_cloth("cloth_0")

    def test_add_cable_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_cable"):
            sim.add_cable("cable_0")

    def test_add_particles_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_particles"):
            sim.add_particles("fluid_0")

    def test_add_sensor_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="add_sensor"):
            sim.add_sensor("imu_0", "imu")

    def test_read_sensor_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="read_sensor"):
            sim.read_sensor("imu_0")

    def test_enable_dual_solver_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="enable_dual_solver"):
            sim.enable_dual_solver()

    def test_reset_envs_not_implemented(self, sim: NewtonSimulation) -> None:
        with pytest.raises(NotImplementedError, match="reset_envs"):
            sim.reset_envs([0, 1, 2])
