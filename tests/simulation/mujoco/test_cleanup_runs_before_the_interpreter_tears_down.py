"""A simulation the caller never released is still torn down at process exit.

``MuJoCoSimEngine.cleanup`` is reachable two ways: a caller calls it (directly
or through the context manager), or :meth:`SimEngine.__del__` calls it as a
safety net. The safety net cannot work during interpreter shutdown. CPython
sets a module's globals to ``None`` before destroying the objects that module
built, so the teardown path - which reads module globals in ``cleanup`` itself
and in the four modules it delegates into - has no names left to call. The first
one reached raises ``AttributeError``, ``__del__`` reports it as a warning, and
nothing is released: the ROS 2 bridge node stays up, teleoperated devices stay
connected, the sim stays advertised to the fleet as a live peer, and policy
workers are never joined before the world they are stepping is freed.

The fix is an :mod:`atexit` hook, which runs while the import system is still
intact. These cells pin both halves: the mechanism that makes the safety net
insufficient (so the hook is not mistaken for belt-and-braces), and that the
hook releases an engine no caller released.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from strands_robots.simulation.mujoco import simulation as sim_module
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

# Reproduces the two phases CPython runs in this order at exit: every registered
# ``atexit`` hook first, then module clearing (which is what sets a module's
# globals to ``None``), and only then the objects those modules built - whose
# finalizers therefore run against nulled globals.
#
# The hook is looked up rather than imported so this script measures the DEFECT
# on a tree that has no hook (cleanup raising against nulled globals) instead of
# failing at import and measuring nothing.
_CHILD = textwrap.dedent(
    '''
    import os, sys
    os.environ["MUJOCO_GL"] = "egl"
    MARK, ROOT = sys.argv[1], sys.argv[2]

    # Import the same tree the test was collected from, not whatever else is on
    # the default path: an editable install elsewhere would answer instead and
    # the child would measure a different tree than the one under test.
    sys.path.insert(0, ROOT)
    import strands_robots
    assert strands_robots.__file__.startswith(ROOT), strands_robots.__file__

    from strands_robots import create_simulation
    from strands_robots.simulation.mujoco import simulation as m

    real = type(create_simulation(backend="mujoco")).cleanup

    def record(self, *a, _real=real, _open=open, _mark=MARK, _type=type, **k):
        """Spy that captures every name it needs, so it stays usable once the
        module globals are gone and cannot confound the measurement."""
        try:
            _real(self, *a, **k)
        except BaseException as exc:
            with _open(_mark, "a") as fh:
                fh.write("RAISED %s: %s\\n" % (_type(exc).__name__, exc))
        else:
            with _open(_mark, "a") as fh:
                fh.write("COMPLETED\\n")

    m.MuJoCoSimEngine.cleanup = record

    sim = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True)
    sim.add_robot("so101")

    # Phase 1: the interpreter runs whatever atexit hooks exist.
    hook = getattr(m, "_release_live_engines", None)
    if hook is not None:
        hook()

    # Phase 2: the interpreter clears module globals.
    for name in list(m.__dict__):
        if not name.startswith("__"):
            m.__dict__[name] = None

    # Phase 3: the objects those modules built are destroyed, so __del__ - the
    # safety net - runs here, against the nulled globals.
    del sim
    '''
)


# Records what the module hands :func:`atexit.register` at import time. Wrapping
# the real function keeps the registration itself intact, and answers the
# wiring question without depending on any shutdown ordering.
_REGISTRATION_CHILD = textwrap.dedent(
    """
    import atexit, sys
    MARK, ROOT = sys.argv[1], sys.argv[2]
    sys.path.insert(0, ROOT)

    registered = []
    real = atexit.register

    def spy(func, *a, **k):
        registered.append(getattr(func, "__name__", repr(func)))
        return real(func, *a, **k)

    atexit.register = spy
    try:
        import strands_robots.simulation.mujoco.simulation  # noqa: F401
    finally:
        atexit.register = real
    with open(MARK, "w") as fh:
        fh.write("\\n".join(registered))
    """
)


def _run_child(tmp_path: Path, source: str = _CHILD) -> str:
    """Run a child script in a fresh interpreter and return what it recorded."""
    mark = tmp_path / "outcome"
    script = tmp_path / "shutdown_order.py"
    script.write_text(source)
    root = Path(sim_module.__file__).parents[3]
    subprocess.run(
        [sys.executable, str(script), str(mark), str(root)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return mark.read_text() if mark.exists() else "NEVER INVOKED"


class TestTheSafetyNetCannotRunDuringShutdown:
    """Why ``__del__`` alone leaves the engine un-released."""

    def test_cleanup_raises_once_its_module_globals_are_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The premise, reproduced deterministically: nulling one of the module
        globals ``cleanup`` reads is enough to abort the whole teardown.

        This holds before and after the fix - it is the mechanism, not the
        regression. It is what makes the exit hook load-bearing rather than
        belt-and-braces, so an engine's own teardown staying reachable is not a
        substitute for running it earlier.
        """
        engine = MuJoCoSimEngine()
        engine.create_world(ground_plane=False)
        # Exactly what CPython's module clearing does to this name.
        monkeypatch.setattr(sim_module, "contextlib", None)
        with pytest.raises(AttributeError):
            engine.cleanup()
        # And the world it should have released is still held.
        assert engine._world is not None

    def test_the_teardown_path_reads_module_globals(self) -> None:
        """``cleanup`` is not finalization-safe on its own, and cannot cheaply
        be made so: it reads module globals directly and delegates into modules
        that read theirs."""
        own = set(MuJoCoSimEngine.cleanup.__code__.co_names) & set(vars(sim_module))
        assert own, "expected cleanup() to read at least one module global"
        assert "contextlib" in own


class TestAnEngineTheCallerNeverReleasedIsTornDownAtExit:
    """The regression: the exit hook runs the teardown while names still exist."""

    def test_the_exit_hook_releases_it(self, tmp_path: Path) -> None:
        outcome = _run_child(tmp_path)
        assert "COMPLETED" in outcome, f"cleanup did not complete at exit: {outcome!r}"
        assert "RAISED" not in outcome, f"cleanup raised during shutdown: {outcome!r}"

    def test_the_hook_is_registered_with_atexit_at_import(self, tmp_path: Path) -> None:
        """The teardown has to be wired to the phase that still has names.
        Calling the hook is not enough on its own - nothing would call it."""
        registered = _run_child(tmp_path, _REGISTRATION_CHILD).split("\n")
        assert "_release_live_engines" in registered, registered

    def test_a_constructed_engine_is_registered_for_release(self) -> None:
        engine = MuJoCoSimEngine()
        try:
            assert engine in sim_module._LIVE_ENGINES
        finally:
            engine.cleanup()

    def test_registration_does_not_keep_it_alive(self) -> None:
        """The registry is weak: holding an engine for exit release must not
        pin its MuJoCo model, renderers and worker threads past the caller's
        last reference."""
        import gc
        import weakref

        engine = MuJoCoSimEngine()
        engine.cleanup()
        ref = weakref.ref(engine)
        del engine
        gc.collect()
        assert ref() is None


class TestTheHookAndTheCallerDoNotDuplicateWork:
    def test_an_engine_released_at_exit_is_not_torn_down_twice(self) -> None:
        """After the hook has released an engine, ``__del__`` reaching
        ``cleanup`` during module destruction must return rather than raise on a
        nulled global and be reported as a cleanup failure."""
        engine = MuJoCoSimEngine()
        engine.create_world(ground_plane=False)
        engine._released_at_exit = True
        engine.cleanup()
        # Returned early, so the world it holds is untouched.
        assert engine._world is not None
        engine._released_at_exit = False
        engine.cleanup()
        assert engine._world is None

    def test_an_ordinary_caller_is_unaffected(self) -> None:
        """The latch is set only by the exit hook, so a caller releasing an
        engine - twice, or side by side with a rebuilt world - still gets the
        full teardown."""
        engine = MuJoCoSimEngine()
        engine.create_world(ground_plane=False)
        engine.cleanup()
        assert engine._world is None
        engine.cleanup()
        assert engine._world is None
