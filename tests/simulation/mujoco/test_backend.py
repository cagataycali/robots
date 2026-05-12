"""Unit tests for mujoco/backend.py - GL backend auto-configuration."""

from __future__ import annotations

import importlib.util
import os
import sys
from unittest.mock import patch

import pytest

from strands_robots.simulation.mujoco import backend as backend_mod


@pytest.fixture
def restore_env(monkeypatch):
    """Isolate MUJOCO_GL / DISPLAY / WAYLAND_DISPLAY per test."""
    for var in ("MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


class TestIsHeadless:
    """``_is_headless`` only returns True on Linux with no display server."""

    def test_non_linux_is_not_headless(self, restore_env):
        with patch.object(sys, "platform", "darwin"):
            assert backend_mod._is_headless() is False

    def test_linux_with_display_not_headless(self, restore_env):
        restore_env.setenv("DISPLAY", ":0")
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is False

    def test_linux_with_wayland_not_headless(self, restore_env):
        restore_env.setenv("WAYLAND_DISPLAY", "wayland-0")
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is False

    def test_linux_no_display_is_headless(self, restore_env):
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is True


class TestConfigureGLBackend:
    """``_configure_gl_backend`` respects MUJOCO_GL and probes EGL then OSMesa."""

    def test_respects_user_mujoco_gl(self, restore_env):
        restore_env.setenv("MUJOCO_GL", "glfw")
        backend_mod._configure_gl_backend()
        # Value unchanged.
        assert os.environ["MUJOCO_GL"] == "glfw"

    def test_noop_on_non_headless(self, restore_env):
        with patch.object(sys, "platform", "darwin"):
            backend_mod._configure_gl_backend()
        # Nothing was set.
        assert "MUJOCO_GL" not in os.environ

    def test_headless_picks_egl_when_available(self, restore_env):
        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = [None]
            try:
                backend_mod._configure_gl_backend()
                assert os.environ.get("MUJOCO_GL") == "egl"
                load.assert_called_once()
            finally:
                # explicit teardown - monkeypatch.delenv only covers vars it had seen at yield time
                os.environ.pop("MUJOCO_GL", None)

    def test_headless_falls_back_to_osmesa(self, restore_env):
        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = [OSError("no libEGL"), None]
            try:
                backend_mod._configure_gl_backend()
                assert os.environ.get("MUJOCO_GL") == "osmesa"
                assert load.call_count == 2
            finally:
                os.environ.pop("MUJOCO_GL", None)

    def test_headless_without_any_gl_warns(self, restore_env, caplog):
        import logging

        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = OSError("no GL")
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.backend"):
                backend_mod._configure_gl_backend()
            # MUJOCO_GL stays unset.
            assert "MUJOCO_GL" not in os.environ
            # Warning text lists both libraries.
            assert any("EGL" in rec.message and "OSMesa" in rec.message for rec in caplog.records)


class TestCanRender:
    """``_can_render`` caches the probe result and short-circuits on headless+no-GL."""

    def _clear_cache(self):
        backend_mod._rendering_available = None

    def test_returns_cached_value(self):
        self._clear_cache()
        backend_mod._rendering_available = True
        assert backend_mod._can_render() is True

        backend_mod._rendering_available = False
        assert backend_mod._can_render() is False
        self._clear_cache()

    def test_headless_without_mujoco_gl_short_circuits(self, restore_env):
        """Probe must NOT run when headless+no-GL - otherwise GLFW SIGABRTs."""
        self._clear_cache()
        with patch.object(sys, "platform", "linux"):
            # No DISPLAY, no MUJOCO_GL.
            assert backend_mod._can_render() is False
        # Cached result remembers the negative.
        assert backend_mod._rendering_available is False
        self._clear_cache()


@pytest.mark.skipif(
    not importlib.util.find_spec("mujoco"),
    reason="mujoco not installed",
)
class TestEnsureMujoco:
    """``_ensure_mujoco`` returns a module-like object with MjModel/MjData."""

    def test_returns_module(self):
        mj = backend_mod._ensure_mujoco()
        # Smoke: these attributes must exist on the real module.
        assert hasattr(mj, "MjModel")
        assert hasattr(mj, "MjData")
        assert hasattr(mj, "mj_step")

    def test_is_cached(self):
        first = backend_mod._ensure_mujoco()
        second = backend_mod._ensure_mujoco()
        assert first is second
