"""Tests for Newton backend factory registration and alias resolution.

Verifies that ``create_simulation("newton")`` and aliases ("warp", "wp")
correctly resolve to ``NewtonSimulation``. No GPU required — tests
only exercise the factory + import machinery, not the simulation itself.
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
from strands_robots.simulation.newton.simulation import NewtonSimulation


class TestNewtonRegistration:
    """Newton must be in the built-in backend registry."""

    def test_newton_in_builtin_backends(self) -> None:
        assert "newton" in _BUILTIN_BACKENDS

    def test_newton_module_path(self) -> None:
        mod, cls = _BUILTIN_BACKENDS["newton"]
        assert mod == "strands_robots.simulation.newton.simulation"
        assert cls == "NewtonSimulation"

    def test_warp_alias(self) -> None:
        assert _BUILTIN_ALIASES["warp"] == "newton"

    def test_wp_alias(self) -> None:
        assert _BUILTIN_ALIASES["wp"] == "newton"


class TestNewtonInListBackends:
    """list_backends() must include Newton and its aliases."""

    def test_newton_in_list(self) -> None:
        backends = list_backends()
        assert "newton" in backends

    def test_warp_in_list(self) -> None:
        backends = list_backends()
        assert "warp" in backends

    def test_wp_in_list(self) -> None:
        backends = list_backends()
        assert "wp" in backends


class TestNewtonAliasResolution:
    """Alias resolution must map to canonical 'newton'."""

    @pytest.mark.parametrize("alias", ["newton", "warp", "wp"])
    def test_resolve_to_newton(self, alias: str) -> None:
        assert _resolve_name(alias) == "newton"


class TestCreateNewtonSimulation:
    """create_simulation("newton") must return a NewtonSimulation instance."""

    def test_create_newton(self) -> None:
        sim = create_simulation("newton")
        assert isinstance(sim, NewtonSimulation)
        assert isinstance(sim, SimEngine)

    def test_create_warp_alias(self) -> None:
        sim = create_simulation("warp")
        assert isinstance(sim, NewtonSimulation)

    def test_create_wp_alias(self) -> None:
        sim = create_simulation("wp")
        assert isinstance(sim, NewtonSimulation)

    def test_create_with_kwargs(self) -> None:
        """Factory kwargs should flow through to NewtonConfig."""
        sim = create_simulation("newton", num_envs=64, solver="xpbd")
        assert isinstance(sim, NewtonSimulation)
        assert sim._config.num_envs == 64
        assert sim._config.solver == "xpbd"

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown simulation backend"):
            create_simulation("nonexistent_engine")
