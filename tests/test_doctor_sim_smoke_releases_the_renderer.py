"""``strands-robots doctor`` releases the sim its smoke test opened.

``check_sim_smoke`` drives ``Robot('so100')`` through ``step`` and
``get_observation``, and that observation renders the robot's cameras, so it
opens a MuJoCo GL context. Left to the finalizer, that context is freed during
interpreter teardown - after EGL has already been de-initialised - and MuJoCo's
own ``Renderer.__del__`` writes an ``Exception ignored in`` traceback to stderr.

The command that wrote it still exits 0 and still prints "All checks passed", so
the reader of a first-run diagnostic is handed a wall of EGL tracebacks beside
the verdict that says their setup is sound. The renderer is therefore released
where the check opened it.

The scope here is that release. Whether the surrounding command reports its
findings well, and what a *failing* check should say, are the sibling suite's
questions and are unchanged by these cells.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from strands_robots.doctor import check_sim_smoke

# The marker MuJoCo writes when a GL context is freed after EGL is gone. Stated
# here rather than imported so these cells grade the observable text a reader
# meets rather than restating a constant the code under test could rename.
_FINALIZER_MARKER = "Exception ignored in"

_DOCTOR_SUITE = Path(__file__).with_name("test_doctor.py")


class _Recorder:
    """A ``Robot`` stand-in that records whether it was released.

    Carries exactly the four members ``check_sim_smoke`` uses, so a cell that
    passes here is not passing because the double happened to be permissive.
    """

    def __init__(self, obs_keys: int = 3, raise_on_observe: bool = False) -> None:
        self.cleanup_calls = 0
        self._obs_keys = obs_keys
        self._raise_on_observe = raise_on_observe

    def step(self) -> None:
        return None

    def get_observation(self, _name: str) -> dict[str, int]:
        if self._raise_on_observe:
            raise RuntimeError("observation exploded")
        return {f"joint{i}": i for i in range(self._obs_keys)}

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def _install(monkeypatch: pytest.MonkeyPatch, robot: object) -> None:
    """Point the check's ``Robot`` lookup at ``robot``.

    ``check_sim_smoke`` imports ``Robot`` from the package inside its own body,
    so the attribute on the package is the seam every cell here drives.
    """
    import strands_robots

    monkeypatch.setattr(strands_robots, "Robot", lambda *_a, **_k: robot)


def _finalbody_release_calls() -> list[str]:
    """Every release call the check makes from a ``finally`` block.

    Read off the source rather than observed, because a release that happens on
    the happy path only - or that a later edit moves out of the ``finally`` - is
    invisible to a cell that drives the passing case.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(check_sim_smoke)))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for statement in node.finalbody:
            for inner in ast.walk(statement):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                    found.append(inner.func.attr)
    return found


def _robot_doubles_in_the_doctor_suite() -> dict[str, set[str]]:
    """Classes in the sibling suite that stand in for a ``Robot``.

    Keyed by class name, valued by the members it defines. A class defining both
    ``step`` and ``get_observation`` is standing in for what ``check_sim_smoke``
    drives, whatever it is called.
    """
    tree = ast.parse(_DOCTOR_SUITE.read_text(encoding="utf-8"))
    doubles: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        members = {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if {"step", "get_observation"} <= members:
            doubles[node.name] = members
    return doubles


class TestTheSimSmokeReleasesItsRenderer:
    """The check releases the sim on every path out of it."""

    def test_a_real_sim_smoke_writes_no_traceback_to_stderr(self) -> None:
        """The observable defect: a passing check that still prints a traceback.

        Driven in a subprocess because the finalizer that wrote it ran during
        interpreter teardown, which an in-process cell cannot observe.
        """
        pytest.importorskip("mujoco")
        script = "from strands_robots.doctor import check_sim_smoke\nprint('VERDICT', check_sim_smoke().strip())\n"
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "STRANDS_MESH": "false"},
        )
        if "PASS" not in done.stdout:
            pytest.skip(f"sim smoke did not pass here, so no GL context was opened: {done.stdout.strip()}")
        assert _FINALIZER_MARKER not in done.stderr, (
            f"the check passed and still wrote {len(done.stderr)} bytes to stderr:\n{done.stderr}"
        )

    def test_the_sim_is_released_on_the_passing_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        robot = _Recorder(obs_keys=3)
        _install(monkeypatch, robot)
        result = check_sim_smoke()
        assert "  PASS  " in result
        assert robot.cleanup_calls == 1

    def test_the_sim_is_released_when_the_observation_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused verdict still owes the release: the context was opened."""
        robot = _Recorder(obs_keys=0)
        _install(monkeypatch, robot)
        result = check_sim_smoke()
        assert "  FAIL  " in result
        assert robot.cleanup_calls == 1

    def test_the_sim_is_released_when_the_observation_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The path that opened the context and then failed still releases it."""
        robot = _Recorder(raise_on_observe=True)
        _install(monkeypatch, robot)
        result = check_sim_smoke()
        assert "  FAIL  " in result
        assert "observation exploded" in result
        assert robot.cleanup_calls == 1


class TestTheReleaseIsStructurallyPinned:
    """The release is reached from a ``finally``, not from the happy path."""

    def test_the_check_releases_the_sim_in_a_finally(self) -> None:
        assert "cleanup" in _finalbody_release_calls(), (
            "check_sim_smoke must release the sim from a finally block, so a "
            "refused or raising observation still frees the GL context it opened"
        )


class TestPremises:
    """Facts these cells rest on, asserted rather than assumed."""

    def test_cleanup_is_the_one_release_verb_every_robot_return_carries(self) -> None:
        """Why ``cleanup`` and not ``with``.

        ``Robot()`` resolves to a simulation or to the hardware wrapper. Only one
        of those implements the context-manager protocol, so ``with`` would be a
        release verb that works for one of the two things the factory returns.
        """
        from strands_robots.hardware_robot import Robot as HardwareRobot
        from strands_robots.simulation.base import SimEngine

        assert callable(getattr(HardwareRobot, "cleanup", None))
        assert callable(getattr(SimEngine, "cleanup", None))
        assert not hasattr(HardwareRobot, "__enter__")
        assert hasattr(SimEngine, "__enter__")

    def test_every_robot_double_in_the_doctor_suite_carries_the_release_verb(self) -> None:
        """A permissive double would let these cells pass for the wrong reason.

        ``check_sim_smoke`` now releases what it opened, so a stand-in without
        ``cleanup`` turns that release into an ``AttributeError`` - and a cell
        asserting ``FAIL`` would still be green while grading nothing.
        """
        doubles = _robot_doubles_in_the_doctor_suite()
        assert doubles, f"found no Robot stand-in in {_DOCTOR_SUITE.name}; the scan has gone blind"
        missing = sorted(name for name, members in doubles.items() if "cleanup" not in members)
        assert not missing, f"Robot doubles without the release verb the real one always has: {missing}"


class TestWhatIsUnchanged:
    """The verdict this check reports is untouched by the release."""

    def test_the_passing_verdict_still_names_the_observation_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        robot = _Recorder(obs_keys=13)
        _install(monkeypatch, robot)
        result = check_sim_smoke()
        assert "  PASS  " in result
        assert "13 obs keys" in result


class TestTheReleaseFailureIsNotSwallowed:
    """A release that fails is named, not hidden behind a passing observation."""

    def test_a_release_failure_is_reported_rather_than_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sim that cannot be released is a real defect on this machine.

        The verdict covers the whole lifecycle the check drives, so the failure
        is named rather than hidden behind a passing observation.
        """

        class _UnreleasableRobot(_Recorder):
            def cleanup(self) -> None:
                raise RuntimeError("gl context stuck")

        _install(monkeypatch, _UnreleasableRobot(obs_keys=3))
        result = check_sim_smoke()
        assert "  FAIL  " in result
        assert "gl context stuck" in result
