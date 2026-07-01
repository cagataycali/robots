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
from pathlib import Path
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


class TestNvidiaIcdScanResilience:
    """Host-detection scans are best-effort: an unreadable glvnd dir or vendor
    JSON must be skipped, never abort the scan, so a later real match is still
    found. Staging failures leave glvnd's default routing untouched rather than
    crashing MuJoCo import.
    """

    def test_library_scan_skips_unreadable_dir_and_finds_later_match(self, monkeypatch, tmp_path):
        """A permission error globbing one lib dir does not hide an NVIDIA library
        installed in a subsequent dir."""
        denied = tmp_path / "denied"
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "libEGL_nvidia.so.580.1").write_text("")
        real_glob = Path.glob

        def flaky_glob(self, pattern, *args, **kwargs):
            if str(self) == str(denied):
                raise PermissionError("EACCES")
            return real_glob(self, pattern, *args, **kwargs)

        monkeypatch.setattr(Path, "glob", flaky_glob)
        monkeypatch.setattr(backend, "_NVIDIA_EGL_LIB_DIRS", (str(denied), str(lib_dir)))

        assert backend._nvidia_egl_library_present() is True

    def test_icd_scan_skips_unreadable_dir_and_unreadable_json(self, monkeypatch, tmp_path):
        """An unreadable vendor dir and an unreadable JSON entry are both skipped;
        a readable NVIDIA vendor JSON still registers as present."""
        denied_dir = tmp_path / "denied_dir"
        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        # sorted() visits 00_broken.json (unreadable) before 10_nvidia.json.
        (vendor_dir / "00_broken.json").write_text("unreadable")
        (vendor_dir / "10_nvidia.json").write_text(backend._NVIDIA_EGL_ICD_JSON)
        real_glob = Path.glob
        real_read = Path.read_text

        def flaky_glob(self, pattern, *args, **kwargs):
            if str(self) == str(denied_dir):
                raise PermissionError("EACCES")
            return real_glob(self, pattern, *args, **kwargs)

        def flaky_read(self, *args, **kwargs):
            if self.name == "00_broken.json":
                raise PermissionError("EACCES")
            return real_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "glob", flaky_glob)
        monkeypatch.setattr(Path, "read_text", flaky_read)
        monkeypatch.setattr(backend, "_GLVND_EGL_VENDOR_DIRS", (str(denied_dir), str(vendor_dir)))

        assert backend._nvidia_egl_icd_registered() is True

    def test_staging_write_failure_is_nonfatal(self, isolate):
        """When the user base dir is not writable, staging fails soft: glvnd's
        default routing is left in place (no __EGL_VENDOR_LIBRARY_FILENAMES)."""
        isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: False)
        isolate.setattr(backend, "_nvidia_egl_library_present", lambda: True)

        def boom(self, *args, **kwargs):
            raise OSError("read-only file system")

        isolate.setattr(Path, "write_text", boom)

        backend._ensure_nvidia_egl_vendor_icd()

        assert _FILENAMES not in backend.os.environ

    def test_staged_filenames_list_nvidia_first_then_system_fallbacks(self, isolate, tmp_path):
        """The staged vendor list puts the NVIDIA ICD first (wins glvnd resolution)
        and appends readable system vendor ICDs as fallback, skipping any vendor
        dir that cannot be listed."""
        isolate.setattr(backend, "_nvidia_egl_icd_registered", lambda: False)
        isolate.setattr(backend, "_nvidia_egl_library_present", lambda: True)
        system_vendor = tmp_path / "system_glvnd"
        system_vendor.mkdir()
        (system_vendor / "50_mesa.json").write_text("{}")
        denied_vendor = tmp_path / "denied_glvnd"
        real_glob = Path.glob

        def flaky_glob(self, pattern, *args, **kwargs):
            if str(self) == str(denied_vendor):
                raise PermissionError("EACCES")
            return real_glob(self, pattern, *args, **kwargs)

        isolate.setattr(Path, "glob", flaky_glob)
        isolate.setattr(backend, "_GLVND_EGL_VENDOR_DIRS", (str(denied_vendor), str(system_vendor)))

        backend._ensure_nvidia_egl_vendor_icd()

        filenames = backend.os.environ[_FILENAMES].split(":")
        staged = tmp_path / "egl_vendor.d" / "10_nvidia.json"
        assert filenames[0] == str(staged)
        assert filenames[-1] == str(system_vendor / "50_mesa.json")
