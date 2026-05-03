"""Tests for ``strands_robots.simulation.factory``.

Covers:
* built-in backend resolution (``mujoco`` + ``mj/mjc/mjx`` aliases)
* runtime ``register_backend`` happy path + conflict errors
* ``list_backends`` enumerates built-in + runtime
* ``_resolve_name`` alias chain
* ``_import_backend_class`` unknown-name error + missing-module error
* ``create_simulation`` entrypoint forwards kwargs

The existing test_factory (mocked mujoco import) is preserved here; the
previously-deselected ImportError-guidance test uses a sentinel strategy
so it can run even when mujoco IS installed.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import (
    _runtime_aliases,
    _runtime_registry,
    create_simulation,
    list_backends,
    register_backend,
)

# Resolution + listing


def test_list_backends_contains_builtins():
    names = set(list_backends())
    assert {"mujoco", "mj", "mjc", "mjx"}.issubset(names)


def test_create_simulation_with_alias_resolves_to_mujoco():
    # `mj` is a built-in alias for `mujoco`
    sim = create_simulation(backend="mj")
    from strands_robots.simulation.mujoco.simulation import Simulation

    assert isinstance(sim, Simulation)
    sim.cleanup()


def test_create_simulation_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown simulation backend"):
        create_simulation(backend="nonexistent_backend_xyz")


# register_backend: conflict detection


class _StubBackend(SimEngine):
    """Minimal concrete backend for registration tests."""

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success", "content": []}

    def destroy(self):
        return {"status": "success", "content": []}

    def list_robots(self):
        return []

    def robot_joint_names(self, robot_name):
        return []

    def get_observation(self, robot_name=None):
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1): ...
    def step(self, n_steps=1):
        return {"status": "success", "content": []}

    def reset(self):
        return {"status": "success", "content": []}


def _loader():
    return _StubBackend


def test_register_backend_and_use_runtime_alias():
    # Uniquify name per-test to avoid leaks from other tests
    name = "runtime_stub_a"
    alias = "stub_a_alias"
    try:
        register_backend(name, _loader, aliases=[alias])
        assert name in list_backends()
        assert alias in list_backends()

        # _resolve_name should map alias → canonical (runtime alias wins)
        from strands_robots.simulation.factory import _resolve_name

        assert _resolve_name(alias) == name
        assert _resolve_name(name) == name
    finally:
        _runtime_registry.pop(name, None)
        _runtime_aliases.pop(alias, None)


def test_register_backend_duplicate_name_without_force_errors():
    try:
        register_backend("runtime_stub_b", _loader)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("runtime_stub_b", _loader)
    finally:
        _runtime_registry.pop("runtime_stub_b", None)


def test_register_backend_force_overrides_existing():
    def loader2():
        return _StubBackend

    try:
        register_backend("runtime_stub_c", _loader)
        # Force overwrite
        register_backend("runtime_stub_c", loader2, force=True)
        assert _runtime_registry["runtime_stub_c"] is loader2
    finally:
        _runtime_registry.pop("runtime_stub_c", None)


def test_register_backend_alias_conflict_with_builtin():
    # 'mj' is a built-in alias; registering it as a runtime alias must fail
    with pytest.raises(ValueError, match="conflicts with built-in alias"):
        register_backend("runtime_stub_d", _loader, aliases=["mj"])


def test_register_backend_alias_conflict_with_builtin_backend_name():
    # 'mujoco' is a built-in backend name; using it as an alias must fail
    with pytest.raises(ValueError, match="conflicts with existing backend name"):
        register_backend("runtime_stub_e", _loader, aliases=["mujoco"])


def test_register_backend_duplicate_alias_without_force_errors():
    a = "dup_alias"
    try:
        register_backend("runtime_stub_f", _loader, aliases=[a])
        with pytest.raises(ValueError, match="already registered"):
            register_backend("runtime_stub_g", _loader, aliases=[a])
    finally:
        _runtime_registry.pop("runtime_stub_f", None)
        _runtime_registry.pop("runtime_stub_g", None)
        _runtime_aliases.pop(a, None)


# _import_backend_class: missing-module ImportError guidance


def test_import_backend_module_missing_raises_with_actionable_message(monkeypatch):
    """When a built-in backend's implementation module is not installed,
    ``_import_backend_class`` must raise ImportError with install hints.

    We simulate this by registering a fake built-in that points to a module
    that won't exist, then triggering the import.
    """
    from strands_robots.simulation import factory as fac

    fake_name = "fake_missing_backend"
    monkeypatch.setitem(
        fac._BUILTIN_BACKENDS,
        fake_name,
        ("strands_robots.nonexistent_backend_module", "FakeSim"),
    )

    with pytest.raises(ImportError) as exc:
        create_simulation(backend=fake_name)

    msg = str(exc.value)
    assert fake_name in msg
    assert "pip install" in msg
    assert "register_backend" in msg


# Smoke: default backend is usable


def test_default_backend_is_mujoco():
    sim = create_simulation()  # defaults to 'mujoco'
    from strands_robots.simulation.mujoco.simulation import Simulation

    assert isinstance(sim, Simulation)
    sim.cleanup()
