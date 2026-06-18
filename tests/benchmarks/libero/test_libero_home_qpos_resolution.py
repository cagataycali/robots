"""Resolution of the LIBERO-canonical Panda home pose + gripper init qpos.

When the LIBERO OSC controller is constructed (#176), the arm and gripper
must be written to upstream's canonical *open home* pose BEFORE the
``controller_factory`` runs, or the policy sees an out-of-distribution
``observation.state`` from the first frame. Two module-level resolvers find
that pose from whichever package layout is installed:

* :func:`_resolve_libero_arm_home_qpos` walks
  ``libero.libero.envs.robots.mounted_panda`` (stock libero) ->
  ``libero.envs.robots.mounted_panda`` (LIBERO-PRO) -> stock robosuite
  ``Panda`` -> ``None``.
* :func:`_resolve_panda_gripper_init_qpos` reads
  ``robosuite.models.grippers.panda_gripper.PandaGripper.init_qpos`` ->
  ``None``.

Neither ``libero`` nor ``robosuite`` is a hard dependency, so the real
environment exercises only the terminal ``None`` branch. These tests inject
fake leaf modules into ``sys.modules`` to drive every layer of the
resolution + fallback chain (and its process-level cache) without the heavy
packages, which is the behavior the #176 swap-and-restore depends on.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from strands_robots.benchmarks.libero import adapter as libero_adapter

# Canonical numbers, mirrored from the resolver docstrings so the test pins
# the *values* upstream writes, not just the resolution mechanics.
_MOUNTED_PANDA_HOME = np.array([0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, np.pi / 4])
_ROBOSUITE_PANDA_HOME = np.array([0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2, np.pi / 4])
_PANDA_GRIPPER_OPEN = np.array([0.020833, -0.020833])

_LIBERO_STOCK_PATH = "libero.libero.envs.robots.mounted_panda"
_LIBERO_PRO_PATH = "libero.envs.robots.mounted_panda"
_ROBOSUITE_ARM_PATH = "robosuite.models.robots.manipulators.panda_robot"
_ROBOSUITE_GRIPPER_PATH = "robosuite.models.grippers.panda_gripper"

_ALL_INJECTED = (
    _LIBERO_STOCK_PATH,
    _LIBERO_PRO_PATH,
    _ROBOSUITE_ARM_PATH,
    _ROBOSUITE_GRIPPER_PATH,
)


@pytest.fixture(autouse=True)
def _reset_resolver_caches():
    """Clear both process-level resolver caches and any injected fake modules.

    The resolvers memoize their first answer in module globals
    (``_CACHED_*``); without resetting them a value resolved in one test
    would leak into the next. Also strips any fake leaf modules a test
    injected so a later real import is not shadowed.
    """
    saved = {name: sys.modules.get(name) for name in _ALL_INJECTED}
    libero_adapter._CACHED_LIBERO_HOME_QPOS = None
    libero_adapter._CACHED_LIBERO_HOME_QPOS_RESOLVED = False
    libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS = None
    libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED = False
    yield
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
    libero_adapter._CACHED_LIBERO_HOME_QPOS = None
    libero_adapter._CACHED_LIBERO_HOME_QPOS_RESOLVED = False
    libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS = None
    libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED = False


def _install_fake_module(path: str, attr: str, init_qpos_factory) -> None:
    """Register a fake leaf module exposing ``attr`` whose instances carry
    an ``init_qpos`` property backed by ``init_qpos_factory()``."""
    module = types.ModuleType(path)

    class _Robot:
        @property
        def init_qpos(self):
            return init_qpos_factory()

    _Robot.__name__ = attr
    setattr(module, attr, _Robot)
    sys.modules[path] = module


# _resolve_libero_arm_home_qpos


def test_arm_home_resolves_from_stock_libero_layout():
    _install_fake_module(_LIBERO_STOCK_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is not None
    np.testing.assert_allclose(out, _MOUNTED_PANDA_HOME)


def test_arm_home_falls_through_to_libero_pro_layout():
    # Stock layout absent -> the LIBERO-PRO import path is tried next.
    _install_fake_module(_LIBERO_PRO_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is not None
    np.testing.assert_allclose(out, _MOUNTED_PANDA_HOME)


def test_arm_home_falls_back_to_robosuite_panda_when_libero_absent():
    # No libero layout at all -> stock robosuite Panda is the last source.
    _install_fake_module(_ROBOSUITE_ARM_PATH, "Panda", lambda: _ROBOSUITE_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is not None
    np.testing.assert_allclose(out, _ROBOSUITE_PANDA_HOME)


def test_arm_home_returns_none_when_nothing_importable():
    # The real harness env: neither libero nor robosuite installed.
    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is None


def test_arm_home_skips_layout_missing_mounted_panda_symbol():
    # A module that imports but lacks MountedPanda must be skipped, not crash.
    empty = types.ModuleType(_LIBERO_STOCK_PATH)
    sys.modules[_LIBERO_STOCK_PATH] = empty
    _install_fake_module(_LIBERO_PRO_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    np.testing.assert_allclose(out, _MOUNTED_PANDA_HOME)


def test_arm_home_ignores_shape_mismatch_from_libero():
    # A 6-DoF arm request must not accept the 7-vector Panda pose.
    _install_fake_module(_LIBERO_STOCK_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(6)

    assert out is None


def test_arm_home_swallows_init_qpos_construction_error():
    # MJCF asset load failures inside MountedPanda() must soft-fall-through.
    def _boom():
        raise RuntimeError("MJCF asset missing")

    _install_fake_module(_LIBERO_STOCK_PATH, "MountedPanda", _boom)

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is None


def test_arm_home_caches_first_resolution():
    _install_fake_module(_LIBERO_STOCK_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())
    first = libero_adapter._resolve_libero_arm_home_qpos(7)

    # Drop the fake module: a re-resolve would now fail, but the cache holds.
    sys.modules.pop(_LIBERO_STOCK_PATH, None)
    second = libero_adapter._resolve_libero_arm_home_qpos(7)

    np.testing.assert_allclose(first, second)
    assert libero_adapter._CACHED_LIBERO_HOME_QPOS_RESOLVED is True


def test_arm_home_cached_none_short_circuits():
    # First call (nothing installed) caches None; second returns it directly.
    assert libero_adapter._resolve_libero_arm_home_qpos(7) is None
    assert libero_adapter._CACHED_LIBERO_HOME_QPOS_RESOLVED is True
    assert libero_adapter._resolve_libero_arm_home_qpos(7) is None


def test_arm_home_cached_value_rejected_for_different_dof():
    # Cache holds a 7-vector; a later 6-DoF request must not reuse it.
    _install_fake_module(_LIBERO_STOCK_PATH, "MountedPanda", lambda: _MOUNTED_PANDA_HOME.copy())
    assert libero_adapter._resolve_libero_arm_home_qpos(7) is not None

    assert libero_adapter._resolve_libero_arm_home_qpos(6) is None


# _resolve_panda_gripper_init_qpos


def test_gripper_init_resolves_from_robosuite():
    _install_fake_module(_ROBOSUITE_GRIPPER_PATH, "PandaGripper", lambda: _PANDA_GRIPPER_OPEN.copy())

    out = libero_adapter._resolve_panda_gripper_init_qpos(2)

    assert out is not None
    np.testing.assert_allclose(out, _PANDA_GRIPPER_OPEN)


def test_gripper_init_returns_none_when_robosuite_absent():
    out = libero_adapter._resolve_panda_gripper_init_qpos(2)

    assert out is None


def test_gripper_init_ignores_finger_count_mismatch():
    _install_fake_module(_ROBOSUITE_GRIPPER_PATH, "PandaGripper", lambda: _PANDA_GRIPPER_OPEN.copy())

    out = libero_adapter._resolve_panda_gripper_init_qpos(3)

    assert out is None


def test_gripper_init_swallows_construction_error():
    def _boom():
        raise RuntimeError("gripper MJCF missing")

    _install_fake_module(_ROBOSUITE_GRIPPER_PATH, "PandaGripper", _boom)

    out = libero_adapter._resolve_panda_gripper_init_qpos(2)

    assert out is None


def test_gripper_init_caches_first_resolution():
    _install_fake_module(_ROBOSUITE_GRIPPER_PATH, "PandaGripper", lambda: _PANDA_GRIPPER_OPEN.copy())
    first = libero_adapter._resolve_panda_gripper_init_qpos(2)

    sys.modules.pop(_ROBOSUITE_GRIPPER_PATH, None)
    second = libero_adapter._resolve_panda_gripper_init_qpos(2)

    np.testing.assert_allclose(first, second)
    assert libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED is True


def test_gripper_init_cached_none_short_circuits():
    assert libero_adapter._resolve_panda_gripper_init_qpos(2) is None
    assert libero_adapter._CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED is True
    assert libero_adapter._resolve_panda_gripper_init_qpos(2) is None


def test_gripper_init_cached_value_rejected_for_different_finger_count():
    _install_fake_module(_ROBOSUITE_GRIPPER_PATH, "PandaGripper", lambda: _PANDA_GRIPPER_OPEN.copy())
    assert libero_adapter._resolve_panda_gripper_init_qpos(2) is not None

    assert libero_adapter._resolve_panda_gripper_init_qpos(3) is None


def test_arm_home_robosuite_fallback_shape_mismatch_yields_none():
    # robosuite Panda is the last source; a wrong-DoF init_qpos is rejected.
    _install_fake_module(_ROBOSUITE_ARM_PATH, "Panda", lambda: _ROBOSUITE_PANDA_HOME.copy())

    out = libero_adapter._resolve_libero_arm_home_qpos(6)

    assert out is None


def test_arm_home_robosuite_fallback_swallows_construction_error():
    def _boom():
        raise RuntimeError("robosuite Panda MJCF missing")

    _install_fake_module(_ROBOSUITE_ARM_PATH, "Panda", _boom)

    out = libero_adapter._resolve_libero_arm_home_qpos(7)

    assert out is None
