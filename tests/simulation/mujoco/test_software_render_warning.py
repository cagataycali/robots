"""Render probe surfaces a CPU software-rasterizer fallback as a warning.

MuJoCo's EGL backend silently routes to Mesa ``llvmpipe`` when no GPU EGL
vendor ICD is registered, dropping offscreen-render throughput ~100x with no
signal. ``_can_render`` reports ``GL_RENDERER`` from its subprocess probe;
``_warn_if_software_rendering`` must turn a software rasterizer into a one-time
warning while staying silent on a real GPU. These tests mock the probe
subprocess so they need no GL context and run anywhere.
"""

import logging
import subprocess

import strands_robots.simulation.mujoco.backend as backend


def _reset_caches() -> None:
    backend._rendering_available = None
    backend._software_render_warned.clear()


def _fake_probe(stdout: bytes):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0] if args else [], 0, stdout=stdout, stderr=b"")

    return _run


def test_warns_once_on_software_rasterizer(monkeypatch, caplog):
    _reset_caches()
    monkeypatch.setenv("MUJOCO_GL", "egl")  # skip the headless short-circuit
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        _fake_probe(b"__GL_RENDERER__=llvmpipe (LLVM 20.1.2, 256 bits)\n"),
    )
    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True
    warnings = [r for r in caplog.records if "software rasterizer" in r.message.lower()]
    assert len(warnings) == 1
    assert "llvmpipe" in warnings[0].getMessage()

    # Second probe in the same process must not re-warn (one-shot guard).
    caplog.clear()
    backend._rendering_available = None  # force re-probe, keep _software_render_warned
    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True
    assert not [r for r in caplog.records if "software rasterizer" in r.message.lower()]


def test_no_warn_on_gpu_renderer(monkeypatch, caplog):
    _reset_caches()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        _fake_probe(b"__GL_RENDERER__=NVIDIA L40S/PCIe/SSE2\n"),
    )
    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True
    assert not [r for r in caplog.records if "software rasterizer" in r.message.lower()]


def test_no_warn_when_marker_absent(monkeypatch, caplog):
    # PyOpenGL unavailable in the probe -> no marker -> best-effort no-op.
    _reset_caches()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(backend.subprocess, "run", _fake_probe(b""))
    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True
    assert not [r for r in caplog.records if "software rasterizer" in r.message.lower()]
