"""Server-side surface tests for :class:`PolicyServer`.

The end-to-end client/server round-trip lives in
``test_remote_policy_roundtrip.py``. This module covers the server's own
construction, dispatch, lifecycle, and CLI contract in isolation:

- building the wrapped policy from a ``policy_provider`` name,
- rejecting an unknown message type loudly,
- guarding against a double :meth:`PolicyServer.start`,
- the context-manager (start on enter, stop on exit) and the blocking
  :meth:`PolicyServer.serve` foreground entry point,
- the ``python -m strands_robots.inference.server`` CLI argument handling.

These assert observable behavior (bound port, raised errors, cleared state),
never private wiring.
"""

import threading
import time

import pytest

from strands_robots.inference import PolicyServer, protocol
from strands_robots.inference import server as server_mod


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_provider_name_builds_wrapped_policy():
    """``policy_provider`` is resolved via ``create_policy`` at construction."""
    server = PolicyServer(policy_provider="mock")

    assert server.policy.provider_name == "mock"
    # The handshake metadata is derived from the freshly built policy.
    metadata = server._metadata()
    assert metadata["provider_name"] == "mock"
    assert set(metadata) >= {
        "provider_name",
        "requires_images",
        "actions_per_step",
        "supports_rtc",
        "execution_horizon",
    }


def test_dispatch_rejects_unknown_message_type():
    """An unrecognized message type raises ``ValueError`` (not silent drop)."""
    server = PolicyServer(policy_provider="mock")

    with pytest.raises(ValueError, match="unknown message type"):
        server._dispatch({"type": "definitely-not-a-real-type"})


def test_reset_dispatch_re_advertises_metadata():
    """A reset reply carries refreshed metadata so the client stays in sync."""
    server = PolicyServer(policy_provider="mock")

    reply = server._dispatch({"type": protocol.MSG_RESET, "seed": 7})

    assert reply["type"] == protocol.MSG_OK
    assert reply["metadata"]["provider_name"] == "mock"


def test_double_start_raises():
    """Starting an already-running server is a loud error, not a no-op."""
    server = PolicyServer(policy_provider="mock", port=0).start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            server.start()
    finally:
        server.stop()


def test_stop_is_idempotent():
    """``stop`` may be called on a never-started or already-stopped server."""
    server = PolicyServer(policy_provider="mock", port=0)
    server.stop()  # never started: no-op
    server.start()
    server.stop()
    server.stop()  # already stopped: no-op
    assert server._server is None


def test_context_manager_starts_and_stops():
    """Entering binds a port; exiting tears the server down."""
    with PolicyServer(policy_provider="mock", port=0) as server:
        assert server.port > 0
        assert server._server is not None
    assert server._server is None


def test_serve_foreground_binds_and_shuts_down():
    """The blocking ``serve`` entry point binds a port and stops on shutdown."""
    server = PolicyServer(policy_provider="mock", port=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: server._server is not None), "serve() never bound"
        assert server.port > 0
    finally:
        # serve() owns the socket in its own `with` block; shutting it down
        # unblocks serve_forever and lets the thread exit cleanly.
        if server._server is not None:
            server._server.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_main_rejects_out_of_range_port():
    """The CLI validates the port range before touching the network."""
    with pytest.raises(SystemExit) as exc:
        server_mod.main(["--provider", "mock", "--port", "0"])
    assert exc.value.code == 2


def test_main_requires_provider():
    """``--provider`` is mandatory."""
    with pytest.raises(SystemExit):
        server_mod.main([])


def test_main_serves_constructed_provider(monkeypatch):
    """The happy CLI path constructs the server and blocks in ``serve``."""
    served: dict[str, object] = {}

    def fake_serve(self: PolicyServer) -> None:
        served["provider"] = self.policy.provider_name
        served["host"] = self.host
        served["port"] = self.port

    monkeypatch.setattr(PolicyServer, "serve", fake_serve)

    server_mod.main(["--provider", "mock", "--host", "127.0.0.1", "--port", "9123"])

    assert served == {"provider": "mock", "host": "127.0.0.1", "port": 9123}


def test_serve_publishes_port_before_exposing_server(monkeypatch):
    """``serve()`` must bind the port before exposing ``._server``.

    ``._server is not None`` is the server's readiness flag: callers (and the
    context manager) treat a non-None ``._server`` as "bound, ``.port`` is
    valid". If the handle were published before the OS port is read back, a
    background observer could see ``._server`` set while ``.port`` is still the
    pre-bind placeholder (0). This pins the publish ordering deterministically
    by blocking inside ``getsockname`` and asserting the handle is not yet
    visible at that instant.
    """
    import websockets.sync.server as ws_server

    getsockname_entered = threading.Event()
    port_release = threading.Event()
    shutdown_event = threading.Event()

    class _FakeSocket:
        def getsockname(self):
            # serve() has reached the port readback; hold here so the test can
            # probe the window before the real port is assigned.
            getsockname_entered.set()
            port_release.wait(2.0)
            return ("127.0.0.1", 54321)

    class _FakeServer:
        def __init__(self):
            self.socket = _FakeSocket()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def serve_forever(self):
            shutdown_event.wait(3.0)

        def shutdown(self):
            shutdown_event.set()

    monkeypatch.setattr(ws_server, "serve", lambda *a, **k: _FakeServer())

    server = PolicyServer(policy_provider="mock", port=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        # Wait until serve() is blocked inside getsockname (port not yet read).
        assert getsockname_entered.wait(2.0), "serve() never reached port readback"
        # The handle must NOT be visible while the port is still unbound.
        assert server._server is None, "._server exposed before port was bound"
        assert server.port == 0

        # Release the port readback; now the server becomes visible WITH a port.
        port_release.set()
        assert _wait_until(lambda: server._server is not None)
        assert server.port == 54321
    finally:
        port_release.set()
        shutdown_event.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
