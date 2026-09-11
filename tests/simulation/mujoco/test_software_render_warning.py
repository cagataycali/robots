"""Render probe surfaces a CPU software-rasterizer fallback as a warning.

MuJoCo's EGL backend routes to Mesa ``llvmpipe`` whenever NVIDIA's EGL vendor
does not answer, dropping offscreen-render throughput ~100x with no signal.
``_can_render`` reports ``GL_RENDERER`` from its subprocess probe;
``_warn_if_software_rendering`` must turn a software rasterizer into a one-time
warning while staying silent on a real GPU.

glvnd reports every cause of that fallback identically, so the warning has to
narrow the cause from the host: a missing vendor ICD, no NVIDIA EGL library at
all, or a driver that is installed and registered but unreachable. Each wants a
different remedy, and prescribing the ICD for all three sends a host that
already registered one to re-apply a fix it has. These tests mock the probe
subprocess so they need no GL context and run anywhere.
"""

import logging
import subprocess

import pytest

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


# (library present, ICD registered) -> the substring the remedy must carry, and
# a substring it must NOT, for each state the two host facts can be in.
_REMEDY_STATES = (
    pytest.param(
        False,
        False,
        "No NVIDIA EGL library",
        "write /usr/share/glvnd",
        id="no-nvidia-library-mesa-is-correct",
    ),
    pytest.param(
        True,
        False,
        "no NVIDIA EGL vendor ICD is registered",
        "already registered",
        id="library-present-icd-missing",
    ),
    pytest.param(
        True,
        True,
        "already registered and libEGL_nvidia is installed",
        "write /usr/share/glvnd",
        id="icd-registered-driver-unreachable",
    ),
)


@pytest.mark.parametrize(("library", "icd", "expected", "forbidden"), _REMEDY_STATES)
def test_remedy_names_the_cause_that_is_still_standing(monkeypatch, caplog, library, icd, expected, forbidden):
    """The warning prescribes the fix for the state the host is actually in.

    ``forbidden`` is the anti-dead-end half: a host that already registered the
    ICD must not be told to write it, and a host with no NVIDIA EGL library must
    not be told to register an ICD for a library it does not have.
    """
    _reset_caches()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(backend, "_nvidia_egl_library_present", lambda: library)
    monkeypatch.setattr(backend, "_nvidia_egl_icd_registered", lambda: icd)
    monkeypatch.setattr(backend.subprocess, "run", _fake_probe(b"__GL_RENDERER__=llvmpipe\n"))

    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True

    warnings = [r for r in caplog.records if "software rasterizer" in r.message.lower()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # The symptom is reported in every state - only the remedy is narrowed.
    assert "llvmpipe" in message
    assert expected in message
    assert forbidden not in message


def test_each_host_state_gets_its_own_warning(monkeypatch, caplog):
    """Three distinct host states must not collapse onto one prescription.

    Graded on the warning the operator reads rather than on the helper behind
    it, so a build that narrows nothing reports one message three times instead
    of failing to import a name.
    """
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(backend.subprocess, "run", _fake_probe(b"__GL_RENDERER__=llvmpipe\n"))
    messages = []
    for library, icd, *_ in (state.values for state in _REMEDY_STATES):
        _reset_caches()
        monkeypatch.setattr(backend, "_nvidia_egl_library_present", lambda _v=library: _v)
        monkeypatch.setattr(backend, "_nvidia_egl_icd_registered", lambda _v=icd: _v)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=backend.logger.name):
            assert backend._can_render() is True
        messages += [r.getMessage() for r in caplog.records if "software rasterizer" in r.message.lower()]
    assert len(messages) == len(_REMEDY_STATES)
    assert len(set(messages)) == len(_REMEDY_STATES), messages


def test_remedy_is_not_consulted_on_a_gpu_renderer(monkeypatch, caplog):
    """A real GPU stays silent, so no host probing happens on the happy path."""
    _reset_caches()
    monkeypatch.setenv("MUJOCO_GL", "egl")

    def _boom() -> bool:
        raise AssertionError("host EGL state probed while rendering on a GPU")

    monkeypatch.setattr(backend, "_nvidia_egl_library_present", _boom)
    monkeypatch.setattr(backend, "_nvidia_egl_icd_registered", _boom)
    monkeypatch.setattr(backend.subprocess, "run", _fake_probe(b"__GL_RENDERER__=NVIDIA L40S/PCIe/SSE2\n"))
    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        assert backend._can_render() is True
    assert not [r for r in caplog.records if "software rasterizer" in r.message.lower()]
