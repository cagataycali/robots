"""Tests for Newton backend stub — config, factory registration, lazy imports.

These tests require NO GPU and NO warp/newton installation.
They verify the class hierarchy, factory resolution, and configuration
validation that runs before any physics engine is touched.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import (
    _BUILTIN_ALIASES,
    _BUILTIN_BACKENDS,
    _resolve_name,
    create_simulation,
    list_backends,
)
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.simulation import NewtonSimulation
from strands_robots.simulation.newton.solvers import (
    BROAD_PHASE_OPTIONS,
    RENDER_BACKENDS,
    SOLVER_MAP,
)

# ── Factory registration ──────────────────────────────────────────────


class TestFactoryRegistration:
    """Verify Newton is registered in the simulation factory."""

    def test_newton_in_builtin_backends(self) -> None:
        assert "newton" in _BUILTIN_BACKENDS
        module_path, class_name = _BUILTIN_BACKENDS["newton"]
        assert module_path == "strands_robots.simulation.newton.simulation"
        assert class_name == "NewtonSimulation"

    def test_warp_alias_resolves_to_newton(self) -> None:
        assert "warp" in _BUILTIN_ALIASES
        assert _BUILTIN_ALIASES["warp"] == "newton"
        assert _resolve_name("warp") == "newton"

    def test_list_backends_includes_newton(self) -> None:
        backends = list_backends()
        assert "newton" in backends
        assert "warp" in backends

    def test_create_simulation_newton(self) -> None:
        sim = create_simulation("newton")
        assert isinstance(sim, NewtonSimulation)
        assert isinstance(sim, SimEngine)

    def test_create_simulation_warp_alias(self) -> None:
        sim = create_simulation("warp")
        assert isinstance(sim, NewtonSimulation)

    def test_create_simulation_with_kwargs(self) -> None:
        sim = create_simulation("newton", num_envs=4096, solver="xpbd")
        assert isinstance(sim, NewtonSimulation)
        assert sim._config.num_envs == 4096
        assert sim._config.solver == "xpbd"


# ── NewtonSimulation class ────────────────────────────────────────────


class TestNewtonSimulation:
    """Verify the stub class hierarchy and behaviour."""

    def test_is_simengine_subclass(self) -> None:
        assert issubclass(NewtonSimulation, SimEngine)

    def test_default_construction(self) -> None:
        sim = NewtonSimulation()
        assert sim._config.solver == "mujoco"
        assert sim._config.device == "cuda:0"
        assert sim._config.num_envs == 1

    def test_construction_with_config(self) -> None:
        cfg = NewtonConfig(solver="xpbd", num_envs=64, device="cpu")
        sim = NewtonSimulation(config=cfg)
        assert sim._config.solver == "xpbd"
        assert sim._config.num_envs == 64

    def test_construction_with_kwargs(self) -> None:
        sim = NewtonSimulation(num_envs=256, solver="semi_implicit")
        assert sim._config.num_envs == 256
        assert sim._config.solver == "semi_implicit"

    def test_repr(self) -> None:
        sim = NewtonSimulation()
        r = repr(sim)
        assert "NewtonSimulation" in r
        assert "mujoco" in r

    def test_context_manager(self) -> None:
        with NewtonSimulation() as sim:
            assert isinstance(sim, NewtonSimulation)

    def test_cleanup_does_not_raise(self) -> None:
        sim = NewtonSimulation()
        sim.cleanup()  # Should be a no-op, not raise

    @pytest.mark.parametrize(
        "method,args",
        [
            ("create_world", ()),
            ("destroy", ()),
            ("reset", ()),
            ("step", ()),
            ("get_state", ()),
            ("add_robot", ("test_robot",)),
            ("remove_robot", ("test_robot",)),
            ("add_object", ("test_obj",)),
            ("remove_object", ("test_obj",)),
            ("get_observation", ()),
            ("send_action", ({"joint_0": 0.5},)),
            ("render", ()),
        ],
    )
    def test_abstract_methods_raise_not_implemented(self, method: str, args: tuple) -> None:
        sim = NewtonSimulation()
        with pytest.raises(NotImplementedError, match="Newton"):
            getattr(sim, method)(*args)


# ── NewtonConfig validation ───────────────────────────────────────────


class TestNewtonConfig:
    """Verify config validates inputs at construction time."""

    def test_default_config(self) -> None:
        cfg = NewtonConfig()
        assert cfg.solver == "mujoco"
        assert cfg.device == "cuda:0"
        assert cfg.num_envs == 1
        assert cfg.physics_dt == 0.005
        assert cfg.substeps == 1
        assert cfg.render_backend == "null"

    def test_all_solvers_accepted(self) -> None:
        for solver in SOLVER_MAP:
            cfg = NewtonConfig(solver=solver)
            assert cfg.solver == solver

    def test_invalid_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown solver"):
            NewtonConfig(solver="nonexistent")

    def test_invalid_render_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown render_backend"):
            NewtonConfig(render_backend="vulkan")

    def test_invalid_broad_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown broad_phase"):
            NewtonConfig(broad_phase="octree")

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=-0.001)

    def test_zero_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=0.0)

    def test_zero_envs_raises(self) -> None:
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            NewtonConfig(num_envs=0)


# ── Solver map constants ──────────────────────────────────────────────


class TestSolverConstants:
    """Verify solver map and constant sets are well-formed."""

    def test_solver_map_has_seven_entries(self) -> None:
        assert len(SOLVER_MAP) == 7

    def test_expected_solvers_present(self) -> None:
        expected = {"mujoco", "featherstone", "semi_implicit", "xpbd", "vbd", "style3d", "implicit_mpm"}
        assert set(SOLVER_MAP.keys()) == expected

    def test_render_backends_includes_null(self) -> None:
        assert "null" in RENDER_BACKENDS
        assert "none" in RENDER_BACKENDS
        assert "opengl" in RENDER_BACKENDS

    def test_broad_phase_includes_sap(self) -> None:
        assert "sap" in BROAD_PHASE_OPTIONS


# ── Lazy import guard ─────────────────────────────────────────────────


class TestLazyImports:
    """Verify importing the newton package does NOT trigger warp/newton loads."""

    def test_import_init_does_not_load_warp(self) -> None:
        import sys

        # If warp were eagerly imported, it would be in sys.modules
        # after importing the newton package.  Since we don't have
        # warp installed in CI, an eager import would raise ImportError.
        import strands_robots.simulation.newton  # noqa: F401

        # Should succeed without warp being present
        assert "strands_robots.simulation.newton" in sys.modules

    def test_import_config_is_lightweight(self) -> None:
        import strands_robots.simulation.newton.config  # noqa: F401
        import strands_robots.simulation.newton.solvers  # noqa: F401
        # These must succeed with zero heavy deps
