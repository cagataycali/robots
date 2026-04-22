"""Tests for strands_robots.simulation.factory.

Regression tests for the built-in-backend-missing case and runtime
registration contracts.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import base as _base
from strands_robots.simulation import factory as _factory


@pytest.fixture(autouse=True)
def _clean_runtime_registry():
    """Snapshot + restore runtime registry so tests don't leak state."""
    saved_reg = dict(_factory._runtime_registry)
    saved_al = dict(_factory._runtime_aliases)
    yield
    _factory._runtime_registry.clear()
    _factory._runtime_registry.update(saved_reg)
    _factory._runtime_aliases.clear()
    _factory._runtime_aliases.update(saved_al)


def test_default_backend_missing_raises_import_error_with_guidance() -> None:
    """When the built-in ``mujoco`` backend module is not installed, we must
    raise :class:`ImportError` with an actionable message — **not** a cryptic
    ``ModuleNotFoundError`` from deep inside importlib.
    """
    # Remove any cached module so we reliably hit the import path.
    import sys

    sys.modules.pop("strands_robots.simulation.mujoco", None)
    sys.modules.pop("strands_robots.simulation.mujoco.simulation", None)

    with pytest.raises(ImportError) as exc:
        _factory.create_simulation()

    msg = str(exc.value)
    assert "mujoco" in msg.lower()
    assert "register_backend" in msg or "install" in msg.lower()


def test_register_backend_loader_must_be_callable() -> None:
    """``register_backend`` requires a *loader* (zero-arg callable returning a
    class), not the class itself — passing the class directly currently works
    only because ``FakeBackend()`` happens to construct an instance.  This
    test pins the contract so future refactors can't regress into accepting
    both and silently doing the wrong thing.
    """

    class FakeBackend(_base.SimEngine):
        def create_world(self, **kw):  # type: ignore[override]
            return {}

        def destroy(self):  # type: ignore[override]
            return {}

        def reset(self):  # type: ignore[override]
            return {}

        def step(self, n_steps: int = 1):  # type: ignore[override]
            return {}

        def get_state(self):  # type: ignore[override]
            return {}

        def add_robot(self, name, **kw):  # type: ignore[override]
            return {}

        def remove_robot(self, name):  # type: ignore[override]
            return {}

        def add_object(self, name, **kw):  # type: ignore[override]
            return {}

        def remove_object(self, name):  # type: ignore[override]
            return {}

        def get_observation(self, robot_name=None, camera_name=None):  # type: ignore[override]
            return {}

        def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[override]
            return None

        def render(self, camera_name="default", width=None, height=None):  # type: ignore[override]
            return {}

    # Correct usage — loader returns the class
    _factory.register_backend("fake_sim", lambda: FakeBackend)
    sim = _factory.create_simulation("fake_sim")
    assert isinstance(sim, FakeBackend)


def test_register_backend_rejects_duplicate_without_force() -> None:
    _factory.register_backend("dup_sim", lambda: _FakeMinimal)
    with pytest.raises(ValueError):
        _factory.register_backend("dup_sim", lambda: _FakeMinimal)


class _FakeMinimal(_base.SimEngine):
    """Minimal concrete backend used across assertion fixtures."""

    def create_world(self, **kw):  # type: ignore[override]
        return {}

    def destroy(self):  # type: ignore[override]
        return {}

    def reset(self):  # type: ignore[override]
        return {}

    def step(self, n_steps: int = 1):  # type: ignore[override]
        return {}

    def get_state(self):  # type: ignore[override]
        return {}

    def add_robot(self, name, **kw):  # type: ignore[override]
        return {}

    def remove_robot(self, name):  # type: ignore[override]
        return {}

    def add_object(self, name, **kw):  # type: ignore[override]
        return {}

    def remove_object(self, name):  # type: ignore[override]
        return {}

    def get_observation(self, robot_name=None, camera_name=None):  # type: ignore[override]
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[override]
        return None

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[override]
        return {}
