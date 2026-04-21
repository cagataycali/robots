"""Tests for NewtonConfig validation.

Tests config construction, validation, defaults, and edge cases.
No GPU or heavy dependencies required — config is a pure dataclass.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.newton.config import (
    BROAD_PHASE_OPTIONS,
    RENDER_BACKENDS,
    SOLVER_MAP,
    NewtonConfig,
)


class TestNewtonConfigDefaults:
    """Verify default config values match documented expectations."""

    def test_default_values(self) -> None:
        config = NewtonConfig()
        assert config.num_envs == 1
        assert config.device == "cuda:0"
        assert config.solver == "mujoco"
        assert config.physics_dt == 0.005
        assert config.substeps == 1
        assert config.render_backend == "none"
        assert config.enable_cuda_graph is False
        assert config.enable_differentiable is False
        assert config.broad_phase == "sap"

    def test_soft_contact_defaults(self) -> None:
        config = NewtonConfig()
        assert config.soft_contact_margin == 0.5
        assert config.soft_contact_ke == 10000.0
        assert config.soft_contact_kd == 10.0
        assert config.soft_contact_mu == 0.5
        assert config.soft_contact_restitution == 0.0


class TestNewtonConfigValidSolvers:
    """All 7 solvers should be accepted."""

    @pytest.mark.parametrize("solver", list(SOLVER_MAP.keys()))
    def test_valid_solver(self, solver: str) -> None:
        config = NewtonConfig(solver=solver)
        assert config.solver == solver


class TestNewtonConfigValidRenderBackends:
    """All render backends should be accepted."""

    @pytest.mark.parametrize("backend", sorted(RENDER_BACKENDS))
    def test_valid_render_backend(self, backend: str) -> None:
        config = NewtonConfig(render_backend=backend)
        assert config.render_backend == backend


class TestNewtonConfigValidBroadPhase:
    """All broad-phase options should be accepted."""

    @pytest.mark.parametrize("bp", sorted(BROAD_PHASE_OPTIONS))
    def test_valid_broad_phase(self, bp: str) -> None:
        config = NewtonConfig(broad_phase=bp)
        assert config.broad_phase == bp


class TestNewtonConfigInvalid:
    """Invalid config values must raise immediately — fail-fast."""

    def test_invalid_solver(self) -> None:
        with pytest.raises(ValueError, match="Unknown solver"):
            NewtonConfig(solver="nonexistent")

    def test_invalid_render_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown render_backend"):
            NewtonConfig(render_backend="metal")

    def test_invalid_broad_phase(self) -> None:
        with pytest.raises(ValueError, match="Unknown broad_phase"):
            NewtonConfig(broad_phase="octree")

    def test_zero_physics_dt(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=0.0)

    def test_negative_physics_dt(self) -> None:
        with pytest.raises(ValueError, match="physics_dt must be positive"):
            NewtonConfig(physics_dt=-0.001)

    def test_zero_num_envs(self) -> None:
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            NewtonConfig(num_envs=0)

    def test_negative_num_envs(self) -> None:
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            NewtonConfig(num_envs=-1)


class TestNewtonConfigCustom:
    """Non-default config combinations."""

    def test_gpu_training_config(self) -> None:
        """4096-env GPU training scenario."""
        config = NewtonConfig(
            num_envs=4096,
            device="cuda:0",
            solver="mujoco",
            physics_dt=1.0 / 60.0,
            substeps=4,
            enable_cuda_graph=True,
        )
        assert config.num_envs == 4096
        assert config.enable_cuda_graph is True
        assert config.substeps == 4

    def test_diffsim_config(self) -> None:
        """Differentiable simulation scenario."""
        config = NewtonConfig(
            solver="semi_implicit",
            enable_differentiable=True,
            enable_cuda_graph=False,  # CUDA graphs + grad tracking conflict
        )
        assert config.enable_differentiable is True
        assert config.enable_cuda_graph is False

    def test_cpu_fallback(self) -> None:
        config = NewtonConfig(device="cpu")
        assert config.device == "cpu"


class TestSolverMap:
    """SOLVER_MAP contains expected entries."""

    def test_seven_solvers(self) -> None:
        assert len(SOLVER_MAP) == 7

    def test_mujoco_solver_class_name(self) -> None:
        assert SOLVER_MAP["mujoco"] == "SolverMuJoCo"

    def test_all_solver_class_names_start_with_solver(self) -> None:
        for name, cls_name in SOLVER_MAP.items():
            assert cls_name.startswith("Solver"), f"{name} → {cls_name}"
