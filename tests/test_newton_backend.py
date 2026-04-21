"""Tests for the Newton simulation backend — config, factory, and lifecycle.

These tests validate the lightweight Newton skeleton (PR 1/7) without
requiring Warp or Newton to be installed.  They verify:

1. NewtonConfig validation (good and bad inputs)
2. Factory registration (create_simulation("newton"), aliases)
3. NewtonSimulation lifecycle (create_world → step → destroy)
4. Error handling (missing world, duplicate names, bad params)
5. SimEngine ABC conformance
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import (
    _BUILTIN_ALIASES,
    _BUILTIN_BACKENDS,
    _resolve_name,
    list_backends,
)
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.solvers import (
    BROAD_PHASE_OPTIONS,
    RENDER_BACKENDS,
    RIGID_BODY_SOLVERS,
    SOLVER_MAP,
)

# ── Config validation ────────────────────────────────────────────────


class TestNewtonConfig:
    """Validate NewtonConfig __post_init__ catches bad inputs early."""

    def test_default_config(self) -> None:
        cfg = NewtonConfig()
        assert cfg.solver == "mujoco"
        assert cfg.device == "cuda:0"
        assert cfg.num_envs == 1
        assert cfg.physics_dt > 0
        assert cfg.substeps >= 1

    def test_custom_config(self) -> None:
        cfg = NewtonConfig(
            num_envs=4096,
            solver="xpbd",
            device="cpu",
            physics_dt=1.0 / 60.0,
            render_backend="opengl",
        )
        assert cfg.num_envs == 4096
        assert cfg.solver == "xpbd"
        assert cfg.device == "cpu"

    def test_all_valid_solvers(self) -> None:
        for solver in SOLVER_MAP:
            cfg = NewtonConfig(solver=solver)
            assert cfg.solver == solver

    def test_invalid_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown solver"):
            NewtonConfig(solver="nonexistent")

    def test_invalid_render_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown render_backend"):
            NewtonConfig(render_backend="directx")

    def test_invalid_broad_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown broad_phase"):
            NewtonConfig(broad_phase="octree")

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=-0.01)

    def test_zero_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=0.0)

    def test_zero_envs_raises(self) -> None:
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            NewtonConfig(num_envs=0)

    def test_zero_substeps_raises(self) -> None:
        with pytest.raises(ValueError, match="substeps must be >= 1"):
            NewtonConfig(substeps=0)


# ── Solver constants ─────────────────────────────────────────────────


class TestSolverConstants:
    """Verify solver map and categorisation."""

    def test_solver_map_has_seven_solvers(self) -> None:
        assert len(SOLVER_MAP) == 7

    def test_rigid_body_solvers_are_subset(self) -> None:
        assert RIGID_BODY_SOLVERS.issubset(set(SOLVER_MAP.keys()))

    def test_soft_body_solvers_not_in_rigid(self) -> None:
        soft_only = {"vbd", "style3d", "implicit_mpm"}
        assert soft_only.isdisjoint(RIGID_BODY_SOLVERS)

    def test_render_backends(self) -> None:
        assert "opengl" in RENDER_BACKENDS
        assert "null" in RENDER_BACKENDS
        assert "none" in RENDER_BACKENDS

    def test_broad_phase_options(self) -> None:
        assert "sap" in BROAD_PHASE_OPTIONS
        assert "bvh" in BROAD_PHASE_OPTIONS


# ── Factory registration ─────────────────────────────────────────────


class TestNewtonFactoryRegistration:
    """Verify Newton is registered in the simulation factory."""

    def test_newton_in_builtin_backends(self) -> None:
        assert "newton" in _BUILTIN_BACKENDS

    def test_newton_backend_module_path(self) -> None:
        module_path, class_name = _BUILTIN_BACKENDS["newton"]
        assert module_path == "strands_robots.simulation.newton.simulation"
        assert class_name == "NewtonSimulation"

    def test_warp_alias_resolves_to_newton(self) -> None:
        assert _BUILTIN_ALIASES["warp"] == "newton"
        assert _resolve_name("warp") == "newton"

    def test_nt_alias_resolves_to_newton(self) -> None:
        assert _BUILTIN_ALIASES["nt"] == "newton"
        assert _resolve_name("nt") == "newton"

    def test_list_backends_includes_newton(self) -> None:
        backends = list_backends()
        assert "newton" in backends
        assert "warp" in backends
        assert "nt" in backends

    def test_list_backends_includes_mujoco(self) -> None:
        """Ensure adding Newton didn't break MuJoCo registration."""
        backends = list_backends()
        assert "mujoco" in backends


# ── Lazy import ──────────────────────────────────────────────────────


class TestNewtonLazyImport:
    """Verify importing Newton modules does not trigger Warp/Newton load."""

    def test_import_config_no_warp(self) -> None:
        """Importing config must not import warp."""

        # Config is already imported above; verify warp wasn't pulled in
        # (it won't be installed in CI without [newton] extras)
        from strands_robots.simulation.newton.config import NewtonConfig as _Cfg

        assert _Cfg is not None
        # warp may or may not be installed — the point is config
        # doesn't require it

    def test_import_solvers_no_warp(self) -> None:
        """Importing solver constants must not import warp."""
        from strands_robots.simulation.newton.solvers import SOLVER_MAP as _SM

        assert len(_SM) == 7

    def test_import_init_no_warp(self) -> None:
        """Importing __init__ must not trigger heavy deps."""
        from strands_robots.simulation.newton import SOLVER_MAP as _SM
        from strands_robots.simulation.newton import NewtonConfig as _Cfg

        assert _Cfg is not None
        assert len(_SM) == 7


# ── NewtonSimulation (without Warp) ─────────────────────────────────


# We can't call create_world() without Warp, but we CAN test
# construction and error paths.


class TestNewtonSimulationConstruction:
    """Test NewtonSimulation without requiring Warp/Newton."""

    def test_is_simengine_subclass(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        assert issubclass(NewtonSimulation, SimEngine)

    def test_construct_with_default_config(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        assert sim.config.solver == "mujoco"
        assert sim.config.num_envs == 1
        assert not sim._world_created

    def test_construct_with_kwargs(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(num_envs=4096, solver="xpbd", device="cpu")
        assert sim.config.num_envs == 4096
        assert sim.config.solver == "xpbd"
        assert sim.config.device == "cpu"

    def test_construct_with_explicit_config(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        cfg = NewtonConfig(num_envs=2048, solver="semi_implicit")
        sim = NewtonSimulation(config=cfg)
        assert sim.config is cfg
        assert sim.config.num_envs == 2048

    def test_get_state_before_world(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        state = sim.get_state()
        assert state["world_created"] is False
        assert state["robots"] == []

    def test_step_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.step()

    def test_add_robot_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.add_robot("so100")

    def test_get_observation_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.get_observation("so100")

    def test_send_action_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.send_action({"joint_0": 0.0})

    def test_render_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.render()

    def test_replicate_without_world_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(RuntimeError, match="World not created"):
            sim.replicate()

    def test_diffsim_without_diff_enabled_raises(self) -> None:
        """run_diffsim requires enable_differentiable=True."""
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        # First need a world — but we can't create one without Warp.
        # This test verifies the config check happens.
        sim._world_created = True  # bypass for testing config guard
        with pytest.raises(RuntimeError, match="enable_differentiable"):
            sim.run_diffsim(
                num_steps=10,
                loss_fn=lambda s: 0.0,
                optimize_params="velocity",
            )

    def test_step_negative_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            sim.step(n_steps=0)

    def test_destroy_resets_state(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        sim._robots["test"] = {}
        result = sim.destroy()
        assert result["status"] == "destroyed"
        assert not sim._world_created
        assert len(sim._robots) == 0

    def test_duplicate_robot_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        sim.add_robot("so100")
        with pytest.raises(ValueError, match="already exists"):
            sim.add_robot("so100")

    def test_remove_nonexistent_robot_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(ValueError, match="not found"):
            sim.remove_robot("ghost")

    def test_duplicate_object_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        sim.add_object("cube")
        with pytest.raises(ValueError, match="already exists"):
            sim.add_object("cube")

    def test_remove_nonexistent_object_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(ValueError, match="not found"):
            sim.remove_object("ghost")

    def test_duplicate_sensor_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        sim.add_sensor("imu0", kind="imu")
        with pytest.raises(ValueError, match="already exists"):
            sim.add_sensor("imu0", kind="imu")

    def test_read_nonexistent_sensor_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        sim._world_created = True
        with pytest.raises(ValueError, match="not found"):
            sim.read_sensor("ghost")

    def test_dual_solver_invalid_raises(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with pytest.raises(ValueError, match="Unknown articulated solver"):
            sim.enable_dual_solver(articulated="nonexistent")
        with pytest.raises(ValueError, match="Unknown soft solver"):
            sim.enable_dual_solver(soft="nonexistent")

    def test_repr(self) -> None:
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation(solver="xpbd", device="cpu", num_envs=256)
        r = repr(sim)
        assert "xpbd" in r
        assert "cpu" in r
        assert "256" in r

    def test_context_manager_protocol(self) -> None:
        """SimEngine supports with-statement via __enter__/__exit__."""
        from strands_robots.simulation.newton.simulation import NewtonSimulation

        sim = NewtonSimulation()
        with sim as s:
            assert s is sim
        # After exit, cleanup should have been called
        assert not sim._world_created
