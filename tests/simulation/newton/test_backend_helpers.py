"""Unit tests for the Newton backend helpers (no GPU / no Newton required)."""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.newton import backend
from strands_robots.simulation.newton.simulation import _short_joint_name

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


class TestSolverRegistry:
    def test_registry_lists_rigid_body_solvers(self):
        reg = backend.solver_registry()
        assert reg["mujoco"] == "SolverMuJoCo"
        assert reg["featherstone"] == "SolverFeatherstone"
        assert reg["xpbd"] == "SolverXPBD"

    def test_registry_keys_are_lowercase(self):
        assert all(k == k.lower() for k in backend.solver_registry())


class TestShortJointName:
    def test_strips_hierarchical_path(self):
        label = "so_arm100/worldbody/Base/Rotation_Pitch/Rotation"
        assert _short_joint_name(label) == "Rotation"

    def test_plain_name_unchanged(self):
        assert _short_joint_name("Jaw") == "Jaw"


@pytest.mark.skipif(_HAS_NEWTON, reason="newton installed; missing-dep path not exercised")
class TestMissingDependency:
    def test_resolve_solver_class_raises_without_newton(self):
        with pytest.raises(ImportError, match="sim-newton"):
            backend.resolve_solver_class("mujoco")


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton not installed")
class TestSolverResolution:
    def test_resolve_known_solver(self):
        cls = backend.resolve_solver_class("featherstone")
        assert cls.__name__ == "SolverFeatherstone"

    def test_resolve_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown Newton solver"):
            backend.resolve_solver_class("not_a_solver")
