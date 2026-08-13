"""Contract tests for the shared MuJoCo GL-availability probe.

These pin the behaviour that the render-test gating relies on: the probe reports
a boolean, honours the ``ROBOT_TEST_MUJOCO=0`` force-skip escape hatch, exposes
a reusable ``requires_gl`` skip marker, and - the safety half - constructs the
probe renderer at most once per process however often the cache on the public
entry point is cleared. On a host whose first attempt failed, a second
construction aborts the interpreter uncatchably and takes the rest of the
session with it, so a cleared cache must not be able to reach one.

They run without a GL context: the force-skip, marker-shape and
build-at-most-once assertions never construct a renderer, and the one that
exercises a *failing* probe supplies its own failure rather than needing a
headless host.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from tests.simulation.mujoco import _gl_probe
from tests.simulation.mujoco._gl_probe import gl_available, requires_gl


def _refuse_to_construct(*args: object, **kwargs: object) -> None:
    """Stand in for ``mujoco.Renderer`` and fail if anything tries to build one."""
    raise AssertionError("the probe renderer was constructed a second time")


def test_gl_available_returns_bool() -> None:
    """The probe result is a plain bool the skipif condition can consume."""
    assert isinstance(gl_available(), bool)


def test_robot_test_mujoco_zero_forces_no_gl(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_TEST_MUJOCO=0 forces a negative result without probing hardware."""
    monkeypatch.setenv("ROBOT_TEST_MUJOCO", "0")
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is False
    finally:
        # Do not leak the forced-negative result into other tests.
        _gl_probe.gl_available.cache_clear()


def test_requires_gl_is_a_skip_marker() -> None:
    """requires_gl is a usable skipif MarkDecorator (applies cleanly to tests)."""
    assert isinstance(requires_gl, pytest.MarkDecorator)
    assert requires_gl.name == "skipif"


def test_the_hardware_answer_is_latched_at_import_time() -> None:
    """Importing the module already ran the one probe this process allows.

    Non-vacuity for the tests below: "no second renderer was constructed" only
    means something if a first construction has already happened.
    """
    assert _gl_probe._HARDWARE_PROBE_RESULT is not None


def test_gl_available_reports_the_latched_hardware_answer() -> None:
    """The cached entry point answers from the latch rather than re-probing."""
    assert gl_available() is _gl_probe._HARDWARE_PROBE_RESULT


def test_a_cleared_cache_cannot_reprobe_the_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """cache_clear() re-reads the environment but never re-runs the construction.

    On a host whose first probe failed, a second renderer construction aborts
    the interpreter uncatchably, so the cleared cache must not be able to reach
    one. A renderer that refuses to be built proves nothing tries: this holds on
    a GL host and on a headless one, so the pin lives where CI can see it.
    """
    mj = pytest.importorskip("mujoco")
    monkeypatch.setattr(mj, "Renderer", _refuse_to_construct)
    latched = _gl_probe._HARDWARE_PROBE_RESULT
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is latched
    finally:
        _gl_probe.gl_available.cache_clear()


def test_a_first_probe_failure_is_latched_and_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graceful first failure is remembered; the retry that would abort never runs.

    This is the headless host's own path, driven on a host that does have GL by
    resetting the latch and making the construction fail.
    """
    mj = pytest.importorskip("mujoco")
    attempts: list[str] = []

    def _failing(*args: object, **kwargs: object) -> None:
        attempts.append("constructed")
        raise RuntimeError("X11: The DISPLAY environment variable is missing")

    monkeypatch.setattr(mj, "Renderer", _failing)
    monkeypatch.setattr(_gl_probe, "_HARDWARE_PROBE_RESULT", None)

    assert _gl_probe._probe_gl_once() is False
    assert _gl_probe._probe_gl_once() is False
    assert attempts == ["constructed"], "the failed probe was retried"


def test_the_force_skip_leaves_the_hardware_latch_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_TEST_MUJOCO=0 forces the skip without consuming or poisoning the latch.

    The force-skip short-circuits ahead of the probe, so the real hardware
    answer survives it and comes back unprobed once the variable is gone.
    """
    mj = pytest.importorskip("mujoco")
    monkeypatch.setattr(mj, "Renderer", _refuse_to_construct)
    latched = _gl_probe._HARDWARE_PROBE_RESULT

    monkeypatch.setenv("ROBOT_TEST_MUJOCO", "0")
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is False
        assert _gl_probe._HARDWARE_PROBE_RESULT is latched
    finally:
        _gl_probe.gl_available.cache_clear()

    monkeypatch.delenv("ROBOT_TEST_MUJOCO", raising=False)
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is latched
    finally:
        _gl_probe.gl_available.cache_clear()


def test_the_force_skip_avoids_the_probe_construction_entirely() -> None:
    """``ROBOT_TEST_MUJOCO=0`` answers without building a renderer at all.

    The latch makes this unobservable inside a process that has already probed:
    by the time any test runs, the import-time probe has answered and a second
    call constructs nothing either way. So it runs in a child interpreter where
    the force-skip is set before the module is first imported, which is the only
    place the ordering between the force-skip and the probe is visible.
    """
    pytest.importorskip("mujoco")
    root = pathlib.Path(inspect.getfile(_gl_probe)).resolve().parents[3]
    assert (root / "pyproject.toml").is_file(), root
    code = textwrap.dedent(
        """
        import mujoco

        built = []
        real = mujoco.Renderer

        def counting(*args, **kwargs):
            built.append(1)
            return real(*args, **kwargs)

        mujoco.Renderer = counting

        from tests.simulation.mujoco._gl_probe import gl_available

        print(f"answer={gl_available()} built={len(built)}")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "ROBOT_TEST_MUJOCO": "0", "PYTHONPATH": str(root)},
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "answer=False built=0" in proc.stdout, proc.stdout
