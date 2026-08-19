"""Shared test fixtures and configuration.

Installs a numpy-backed torch stand-in when real torch is unavailable, so the
parts of the suite that need only a thin tensor surface run without the ~2GB
dependency. That stand-in is a subset rather than a replacement: a test reaching
outside it is skipped with the attribute and the remedy named, not failed.

Also disables the Zenoh mesh by default during the test suite so the
``Robot()`` / ``Simulation()`` factory does not spin up real Zenoh
sessions and background heartbeat threads when ``eclipse-zenoh`` is
installed in the test environment.  Mesh-specific tests opt back in
explicitly via ``monkeypatch.delenv`` or by patching ``init_mesh``.
"""

import os

# Disable mesh BEFORE any strands_robots import below pulls in robot.py.
# Use setdefault so tests that explicitly enable the mesh (e.g. integ tests)
# can override via the environment without conftest stomping on them.
os.environ.setdefault("STRANDS_MESH", "false")

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

    for line in leak_report(session_open=session_open, threads=threads):
        print(line)

    if session_open and _mesh_session is not None:
        # release_session() is refcounted and a leaked session has lost count of
        # its owners, so close the object itself - this hook runs after the last
        # test, when nothing legitimate can still be publishing.
        try:
            live = _mesh_session.current_session()
            if live is not None:
                live.close()
        except Exception:
            pass

