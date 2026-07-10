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


def _capture_port_when_bound(server_cls):
    """Return a ``server_cls`` subclass that snapshots ``.port`` at the instant
    ``_server`` first becomes non-None (i.e. the moment the instance is
    published as "bound")."""
    captured: dict[str, int] = {}

    class _Capturing(server_cls):
        def __setattr__(self, name, value):
            if name == "_server" and value is not None and "port" not in captured:
                captured["port"] = self.port
            super().__setattr__(name, value)

    return _Capturing, captured


def test_serve_publishes_port_before_marking_bound():
    """``serve()`` sets ``.port`` before publishing ``_server``.

    A background caller uses ``_server is not None`` as the "server is bound"
    signal and then reads ``.port``. If ``serve()`` published ``_server`` first
    and the port second, that caller could observe the constructor default
    ``0`` in the window between the two writes. Capture the port at the exact
    instant ``_server`` becomes non-None and assert it is already the real
    bound port. Regression for a publish-ordering race.
    """
    capturing_cls, captured = _capture_port_when_bound(PolicyServer)
    server = capturing_cls(policy_provider="mock", port=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: server._server is not None), "serve() never bound"
        assert captured.get("port", 0) > 0, "port was still 0 when _server was published as bound"
    finally:
        if server._server is not None:
            server._server.shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_start_publishes_port_before_marking_bound():
    """``start()`` sets ``.port`` before publishing ``_server`` too, so the
    same non-None-``_server``-implies-real-port invariant holds for the
    background-thread entry point."""
    capturing_cls, captured = _capture_port_when_bound(PolicyServer)
    server = capturing_cls(policy_provider="mock", port=0).start()
    try:
        assert captured.get("port", 0) > 0, "port was still 0 when _server was published as bound"
        assert server.port == captured["port"]
    finally:
        server.stop()
