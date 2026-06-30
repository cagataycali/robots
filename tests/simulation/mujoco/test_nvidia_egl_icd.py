"""Library auto-registers the NVIDIA EGL vendor ICD so MuJoCo renders on GPU.

When ``MUJOCO_GL=egl`` on an NVIDIA host whose glvnd vendor directory is missing
``10_nvidia.json`` (common in CUDA base images without the ``graphics``
capability), libglvnd silently routes offscreen rendering to Mesa ``llvmpipe``
(CPU) ~100x slower. ``_ensure_nvidia_egl_vendor_icd`` stages a user-writable
NVIDIA vendor ICD and points glvnd at it via ``__EGL_VENDOR_LIBRARY_FILENAMES``
(no root needed), while never overriding an explicit user vendor config and
staying a no-op on non-NVIDIA / non-Linux hosts. These tests mock host
detection, so they need no GPU, EGL, or mujoco and run anywhere.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

import strands_robots.simulation.mujoco.backend as backend

_FILENAMES = "__EGL_VENDOR_LIBRARY_FILENAMES"
_DIRS = "__EGL_VENDOR_LIBRARY_DIRS"


@pytest.fixture
def isolate(monkeypatch, tmp_path):
    """Linux host, clean glvnd env, base dir redirected to tmp_path."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(_FILENAMES, raising=False)
    monkeypatch.delenv(_DIRS, raising=False)
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    return monkeypatch


def test_stages_nvidia_icd_when_missing_on_nvidia_host(isolate, tmp_path):
    isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: False)
    isolate.setattr(backend, "_nvidia_egl_library_present", lambda: True)

    backend._ensure_nvidia_egl_vendor_icd()

    icd = tmp_path / "egl_vendor.d" / "10_nvidia.json"
    assert icd.is_file()
    # Payload is a valid glvnd vendor ICD pointing at the NVIDIA EGL library.
    payload = json.loads(icd.read_text())
    assert payload["ICD"]["library_path"] == "libEGL_nvidia.so.0"
    # glvnd is steered at our staged ICD, NVIDIA first.
    filenames = backend.os.environ[_FILENAMES].split(":")
    assert filenames[0] == str(icd)


def test_noop_when_nvidia_icd_already_registered(isolate):
    isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: True)
    isolate.setattr(backend, "_nvidia_egl_library_present", lambda: True)

    backend._ensure_nvidia_egl_vendor_icd()

    assert _FILENAMES not in backend.os.environ


def test_noop_when_not_an_nvidia_host(isolate, tmp_path):
    isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: False)
    isolate.setattr(backend, "_nvidia_egl_library_present", lambda: False)

    backend._ensure_nvidia_egl_vendor_icd()

    assert _FILENAMES not in backend.os.environ
    assert not (tmp_path / "egl_vendor.d" / "10_nvidia.json").exists()


def test_respects_explicit_user_vendor_override(isolate):
    isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: False)
    isolate.setattr(backend, "_nvidia_egl_library_present", lambda: True)
    isolate.setenv(_FILENAMES, "/custom/10_vendor.json")

    backend._ensure_nvidia_egl_vendor_icd()

    # Untouched - an explicit override is always respected.
    assert backend.os.environ[_FILENAMES] == "/custom/10_vendor.json"


def test_noop_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(_FILENAMES, raising=False)
    # Detection should never even be consulted off Linux.
    monkeypatch.setattr(
        backend,
        "_nvidia_egl_library_present",
        lambda: pytest.fail("detection ran off Linux"),
    )
    backend._ensure_nvidia_egl_vendor_icd()
    assert _FILENAMES not in backend.os.environ


class TestNvidiaHostDetection:
    """``_nvidia_egl_library_present`` / ``_nvidia_egl_icd_registered`` scan dirs."""

    def test_library_present_detects_versioned_so(self, monkeypatch, tmp_path):
        (tmp_path / "libEGL_nvidia.so.580.126.09").write_text("")
        monkeypatch.setattr(backend, "_NVIDIA_EGL_LIB_DIRS", (str(tmp_path),))
        assert backend._nvidia_egl_library_present() is True

    def test_library_absent_when_no_nvidia_so(self, monkeypatch, tmp_path):
        (tmp_path / "libEGL_mesa.so.0").write_text("")
        monkeypatch.setattr(backend, "_NVIDIA_EGL_LIB_DIRS", (str(tmp_path),))
        assert backend._nvidia_egl_library_present() is False

    def test_icd_registered_when_vendor_json_references_nvidia(self, monkeypatch, tmp_path):
        (tmp_path / "10_nvidia.json").write_text(backend._NVIDIA_EGL_ICD_JSON)
        monkeypatch.setattr(backend, "_GLVND_EGL_VENDOR_DIRS", (str(tmp_path),))
        assert backend._nvidia_egl_icd_registered() is True

    def test_icd_not_registered_with_only_mesa(self, monkeypatch, tmp_path):
        (tmp_path / "50_mesa.json").write_text(
            '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_mesa.so.0"}}'
        )
        monkeypatch.setattr(backend, "_GLVND_EGL_VENDOR_DIRS", (str(tmp_path),))
        assert backend._nvidia_egl_icd_registered() is False


class TestConfigureGLBackendWiring:
    """``_configure_gl_backend`` registers the ICD only on an EGL backend."""

    def test_user_egl_triggers_icd_registration(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "egl")
        called: list[bool] = []
        monkeypatch.setattr(backend, "_ensure_nvidia_egl_vendor_icd", lambda: called.append(True))
        backend._configure_gl_backend()
        assert called == [True]

    def test_user_non_egl_skips_icd_registration(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "osmesa")
        monkeypatch.setattr(
            backend,
            "_ensure_nvidia_egl_vendor_icd",
            lambda: pytest.fail("ICD registration ran for a non-EGL backend"),
        )
        backend._configure_gl_backend()

    def test_auto_egl_triggers_icd_registration(self, monkeypatch):
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        called: list[bool] = []
        monkeypatch.setattr(backend, "_ensure_nvidia_egl_vendor_icd", lambda: called.append(True))
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(backend.ctypes.cdll, "LoadLibrary", return_value=None),
        ):
            try:
                backend._configure_gl_backend()
                assert backend.os.environ.get("MUJOCO_GL") == "egl"
                assert called == [True]
            finally:
                backend.os.environ.pop("MUJOCO_GL", None)
