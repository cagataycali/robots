"""A second dashboard on a bound port must be refused before it joins the mesh.

The dashboard is a mesh peer, and :func:`strands_robots.dashboard.cli.main`
constructs its :class:`~strands_robots.dashboard.mesh_bridge.MeshBridge` - which
opens the zenoh session - *before* handing the app to uvicorn. So a duplicate
launch on an already-bound port did the damage first and reported it second: the
extra peer was live, publishing state and answering discovery as a second hub,
by the time the bind error printed. Piled-up dashboards partitioning the fleet
this way is the BUGS.md bug #9 family, and the launch that caused it exited on a
traceback about an address, naming neither the instance already serving nor the
fact that one existed.

What is pinned here:

* the guard's verdict comes from a **bind probe**, on the wildcard and the
  loopback address both, because a listener on either conflicts with a bind on
  the other - probing one address alone reports a free port that uvicorn then
  cannot take;
* only ``EADDRINUSE`` means occupied, so a free port is ``None`` and stays
  ``None`` even on an address this host does not own;
* owner discovery is **decoration, not the decision**: the refusal must survive
  a lookup that cannot name a pid (unprivileged ``psutil`` on macOS, no
  ``lsof``), which is why the occupied case asserts a non-``None`` description
  rather than a pid;
* the probe leaves the port exactly as it found it - a guard that held the
  socket it tested would deny the port to the server it is protecting.

Every socket here is bound on an ephemeral port the kernel hands out (port
``0``). Nothing touches the real dashboard's port, and nothing here starts a
mesh peer, a server, or any hardware.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from strands_robots.dashboard import cli


@pytest.fixture()
def taken_port() -> Iterator[int]:
    """An ephemeral port with a live listener on it for the duration of a test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _free_port() -> int:
    """A port number that was free at the moment it was asked for."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestOccupiedPort:
    """A port with a listener on it is reported, whoever owns it."""

    def test_reports_a_description(self, taken_port: int) -> None:
        """The verdict is a non-empty description, not a bare boolean."""
        owner = cli._port_in_use(taken_port)
        assert owner is not None
        assert owner.strip()

    def test_reported_for_a_loopback_listener_probed_via_the_wildcard(self, taken_port: int) -> None:
        """A ``127.0.0.1`` listener conflicts with a ``0.0.0.0`` bind, so it counts.

        This is the asymmetry that makes a one-address probe wrong: the caller
        asked about ``0.0.0.0`` and the listener is on loopback.
        """
        assert cli._port_in_use(taken_port, "0.0.0.0") is not None

    def test_reported_when_the_owner_cannot_be_named(
        self, taken_port: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every lookup blinded, the port is still refused - generically."""
        monkeypatch.setattr(cli, "_listening_pid", lambda port: None)
        assert cli._port_in_use(taken_port) == "an unidentified process"

    def test_names_the_pid_and_command_when_discoverable(
        self, taken_port: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A discovered owner reaches the message as ``pid N (command)``."""
        monkeypatch.setattr(cli, "_listening_pid", lambda port: 4242)
        monkeypatch.setattr(cli, "_process_command", lambda pid: "python -m strands_robots dashboard")
        assert cli._port_in_use(taken_port) == "pid 4242 (python -m strands_robots dashboard)"

    def test_pid_alone_when_the_command_is_unreadable(
        self, taken_port: int, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pid without a command line still names the instance to look at."""
        monkeypatch.setattr(cli, "_listening_pid", lambda port: 4242)
        monkeypatch.setattr(cli, "_process_command", lambda pid: None)
        assert cli._port_in_use(taken_port) == "pid 4242"


class TestFreePort:
    """A port nobody holds is ``None``, and is still free afterwards."""

    def test_free_port_is_none(self) -> None:
        """No listener, no refusal."""
        assert cli._port_in_use(_free_port()) is None

    def test_probe_does_not_keep_the_port(self) -> None:
        """The probe closes what it opened, so the server can bind after it."""
        port = _free_port()
        assert cli._port_in_use(port) is None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))  # raises if the probe held it


class TestOwnerLookupIsBestEffort:
    """The helpers behind the description degrade instead of raising."""

    def test_listening_pid_of_a_free_port_is_none(self) -> None:
        """Nothing is listening, so no pid is invented."""
        assert cli._listening_pid(_free_port()) is None

    def test_process_command_of_this_process_is_reported(self) -> None:
        """The ``ps`` fallback path works without psutil installed."""
        import os

        command = cli._process_command(os.getpid())
        assert command is not None
        assert len(command) <= cli._COMMAND_CHARS

    def test_process_command_of_a_dead_pid_is_none(self) -> None:
        """An owner that exited between lookup and describe is not an error."""
        assert cli._process_command(2**22) is None


class TestLeftoverConnectionIsNotAPileup:
    """CLOSE_WAIT residue from a dead instance must not block a restart.

    uvicorn's listener gets ``SO_REUSEADDR`` from asyncio on POSIX, so a
    leftover ``CLOSE_WAIT``/``TIME_WAIT`` socket (no listener) does not stop
    the server from binding. A probe WITHOUT the option is stricter than the
    server it fronts: it reported "in use" for a port uvicorn would have taken,
    and the guard refused a perfectly good restart. Only a live LISTENer is a
    pileup.
    """

    def test_close_wait_leftover_reports_free(self) -> None:
        # Manufacture the exact post-restart shape: a listener accepts one
        # connection, the client side closes, the listener closes - leaving
        # only the accepted socket, now in CLOSE_WAIT, holding the port.
        lst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lst.bind(("127.0.0.1", 0))
        lst.listen(1)
        port = lst.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        accepted, _ = lst.accept()
        try:
            client.close()  # -> accepted transitions to CLOSE_WAIT
            lst.close()  # no listener remains, only the CLOSE_WAIT socket
            assert cli._port_in_use(port, "127.0.0.1") is None, (
                "a CLOSE_WAIT leftover is not a pileup - uvicorn binds "
                "past it, so the guard must too"
            )
        finally:
            accepted.close()
            client.close()

    def test_probe_still_catches_a_real_listener(self, taken_port: int) -> None:
        # SO_REUSEADDR must not blind the probe to the case it exists for.
        assert cli._port_in_use(taken_port, "127.0.0.1") is not None
