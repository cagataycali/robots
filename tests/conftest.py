"""Shared test fixtures and configuration.

Installs a numpy-backed torch stand-in when real torch is unavailable, so the
parts of the suite that need only a thin tensor surface run without the ~2GB
dependency. That stand-in is a subset rather than a replacement: a test reaching
outside it is skipped with the attribute and the remedy named, not failed.

Also disables the Zenoh mesh by default during the test suite so the
``Robot()`` / ``Simulation()`` factory does not spin up real Zenoh
sessions and background heartbeat threads when ``eclipse-zenoh`` is
installed in the test environment.  Mesh-specific tests opt back in
explicitly via ``monkeypatch.delenv`` or by patching ``init_mesh`` -- a per-test
opt-in, which is the only kind that cannot leak into the rest of the run.
"""

import os

# Disable mesh BEFORE any strands_robots import below pulls in robot.py.
#
# This is FORCED, not setdefault. setdefault meant that an ambient
# ``STRANDS_MESH=true`` -- exactly what a shell or an automation that reproduced a
# running dashboard's environment (``ps eww``) exports -- silently disarmed the
# suite's only protection and let a test run JOIN THE LIVE FLEET: publishing
# presence, and in the estop drills broadcasting a real emergency stop
# (BUGS.md Q30, Q32). A safety default that any inherited variable can switch off
# is not a default. Opting in is now a deliberate act with a name that cannot be
# inherited by accident, and it announces itself.
if os.environ.get("STRANDS_TEST_ALLOW_LIVE_MESH", "").strip().lower() in ("1", "true", "yes"):
    os.environ.pop("STRANDS_MESH", None)
    print(
        "WARNING: STRANDS_TEST_ALLOW_LIVE_MESH is set - the mesh kill switch is OFF for this "
        "run and tests may reach a real fleet.",
        flush=True,
    )
else:
    os.environ["STRANDS_MESH"] = "false"

# Disable the Device Connect dispatch path in robot_mesh by default so unit
# tests exercise the built-in mesh deterministically, without opening real
# Device Connect (Zenoh) connections. The GUIDE E2E demo runs outside pytest
# and leaves this unset, so Device Connect remains the primary path at runtime.
os.environ.setdefault("STRANDS_ROBOT_MESH_DC", "off")

from tests.mocks.torch_mock import install_torch_mock

# Must run before any test imports policy modules
install_torch_mock()


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001, ARG001
    """Q32: refuse to leave a mesh session open, and say so.

    A suite that finishes while still joined to the mesh becomes a live
    ``gateway-*`` peer on cagatay's fleet screen and keeps a rail open to real
    hardware - three such ghosts were found holding the hub, one of them three
    days old. Whatever the exit status, the session is closed here and the
    reason is printed, so the leak is visible in the run that caused it instead
    of hours later in ``lsof``.
    """
    import threading

    session_open = False
    try:
        from strands_robots.mesh import session as _mesh_session

        session_open = _mesh_session.current_session() is not None
    except Exception:  # zenoh absent, or the module never imported - nothing to leak
        _mesh_session = None  # type: ignore[assignment]

    threads = [t.name for t in threading.enumerate() if t.is_alive() and not t.daemon]

    from tests.session_leak import leak_report

    lines = leak_report(session_open=session_open, threads=threads)
    # `print` here is swallowed: global capture is still installed during
    # sessionfinish, so the first version of this guard was invisible in exactly
    # the run it was meant to warn. Write through the terminal reporter (the
    # plugin that prints the summary the user does see), and fall back to
    # suspending capture only if that plugin is absent.
    reporter = getattr(session.config, "pluginmanager", None)
    reporter = reporter.get_plugin("terminalreporter") if reporter else None
    if reporter is not None:
        for line in lines:
            reporter.write_line(line)
    elif lines:
        capman = session.config.pluginmanager.get_plugin("capturemanager") if session.config else None
        if capman is not None:
            capman.suspend_global_capture(in_=True)
        try:
            for line in lines:
                print(line)
        finally:
            if capman is not None:
                capman.resume_global_capture()

    if session_open and _mesh_session is not None:
        # release_session() is refcounted and a leaked session has lost count of
        # its owners, so close the object itself - this hook runs after the last
        # test, when nothing legitimate can still be publishing.
        closed: object = "no session handle"
        try:
            live = _mesh_session.current_session()
            if live is not None:
                live.close()
                closed = "closed"
        except Exception as exc:  # a close that fails must not be reported as success
            closed = f"close FAILED: {type(exc).__name__}: {exc}"
        # Closing the handle does not clear the module's global, so
        # current_session() would keep handing out a DEAD session to whatever
        # imported it next — a failure that looks like a transport fault rather
        # than a leak. Reset the global too, and say so.
        try:
            _mesh_session._SESSION = None  # noqa: SLF001 - test-only hygiene
            _mesh_session._SESSION_REFS = 0  # noqa: SLF001
        except Exception:
            pass
        if reporter is not None:
            reporter.write_line(f"  -> leaked mesh session: {closed}, module global reset")

