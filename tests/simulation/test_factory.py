"""Tests for strands_robots.simulation.factory - backend registration + creation."""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation import factory
from strands_robots.simulation.factory import (
    DEFAULT_BACKEND,
    create_simulation,
    list_backends,
    register_backend,
)


@pytest.fixture(autouse=True)
def _clear_runtime():
    """Each test starts with a clean runtime registry."""
    factory._runtime_registry.clear()
    factory._runtime_aliases.clear()
    yield
    factory._runtime_registry.clear()
    factory._runtime_aliases.clear()


class _FakeSim:
    """Plain class stand-in for a simulation backend.

    Not a real ``SimEngine`` subclass - the factory only calls the loader
    callable and the returned class's ``__init__``; it does not enforce the
    ABC contract. Using a plain class here keeps the test focused on the
    factory's own logic (registration, lookup, aliasing).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestListBackends:
    def test_includes_builtin_mujoco(self):
        assert "mujoco" in list_backends()

    def test_includes_builtin_aliases(self):
        backends = list_backends()
        assert "mj" in backends
        assert "mjc" in backends
        assert "mjx" in backends

    def test_is_sorted_and_deduped(self):
        backends = list_backends()
        assert backends == sorted(set(backends))

    def test_includes_runtime_backends(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fk"])
        backends = list_backends()
        assert "fake_sim" in backends
        assert "fk" in backends


class TestRegisterBackend:
    def test_register_and_create(self):
        register_backend("fake_sim", lambda: _FakeSim)
        sim = create_simulation("fake_sim")
        assert isinstance(sim, _FakeSim)

    def test_register_with_aliases(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fs", "fake"])
        assert isinstance(create_simulation("fs"), _FakeSim)
        assert isinstance(create_simulation("fake"), _FakeSim)

    def test_duplicate_name_rejected(self):
        register_backend("fake_sim", lambda: _FakeSim)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("fake_sim", lambda: _FakeSim)

    def test_duplicate_conflicts_with_builtin(self):
        with pytest.raises(ValueError, match="already registered"):
            register_backend("mujoco", lambda: _FakeSim)

    def test_duplicate_conflicts_with_builtin_alias(self):
        with pytest.raises(ValueError, match="conflicts with built-in alias"):
            register_backend("mj", lambda: _FakeSim)

    def test_runtime_alias_conflict(self):
        register_backend("alpha", lambda: _FakeSim, aliases=["shared"])
        with pytest.raises(ValueError, match="already registered"):
            register_backend("beta", lambda: _FakeSim, aliases=["shared"])

    def test_alias_conflicts_with_builtin(self):
        with pytest.raises(ValueError, match="conflicts with existing backend"):
            register_backend("beta", lambda: _FakeSim, aliases=["mujoco"])

    def test_force_overrides_duplicate(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fk"])

        class _OtherSim(_FakeSim):
            pass

        register_backend("fake_sim", lambda: _OtherSim, aliases=["fk"], force=True)
        sim = create_simulation("fake_sim")
        assert type(sim).__name__ == "_OtherSim"


@pytest.mark.skipif(
    not importlib.util.find_spec("mujoco"),
    reason="mujoco not installed",
)
class TestCreateSimulation:
    def test_default_is_mujoco(self):
        sim = create_simulation()
        assert type(sim).__name__ == "Simulation"
        sim.cleanup()

    def test_by_alias(self):
        sim = create_simulation("mj")
        assert type(sim).__name__ == "Simulation"
        sim.cleanup()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown simulation backend"):
            create_simulation("nonexistent_backend_xyz")

    def test_unknown_backend_error_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            create_simulation("nonexistent_backend_xyz")
        msg = str(exc_info.value)
        assert "mujoco" in msg  # should list available backends

    def test_kwargs_forwarded_to_backend(self):
        register_backend("fake_sim", lambda: _FakeSim)
        sim = create_simulation("fake_sim", tool_name="custom", timestep=0.005)
        assert sim.kwargs == {"tool_name": "custom", "timestep": 0.005}

    def test_runtime_alias_priority_over_builtin(self):
        """Runtime aliases can shadow built-in aliases when ``force=True``."""
        register_backend("fake_sim", lambda: _FakeSim, aliases=["mj"], force=True)
        sim = create_simulation("mj")
        assert isinstance(sim, _FakeSim)


class TestDefaultBackendConstant:
    def test_default_is_documented(self):
        assert DEFAULT_BACKEND == "mujoco"
