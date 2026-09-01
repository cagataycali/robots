"""A :class:`PolicyServer` told to stop must stop serving the clients it has.

``websockets.sync.server.Server.shutdown()`` closes the listening socket and
nothing else. The sync server keeps no record of the connections it accepted -
its own source says so at the point it spawns them::

    # Since there isn't a mechanism for tracking connections and waiting
    # for them to terminate, ...
    thread = threading.Thread(target=self.handler, args=(sock, addr))

so each connection is served on a thread that outlives the server object. A
teardown that shuts only the listener down therefore returns while a client that
was already connected goes on streaming observations in and receiving action
chunks back: the wrapped policy is still being invoked, and on a robot it is
still driving the arm, after the caller was told the server stopped. Measured on
websockets 16.1.1 before the fix, through both teardown doors - ``stop()``
returned in 0.1ms and the same live connection was served again straight after,
as did the connection held open across a returning ``serve()``.

Nothing caught it because the lifecycle tests grade the server's own *state*:
``test_stop_is_idempotent`` and ``test_context_manager_starts_and_stops`` in
``test_policy_server_lifecycle.py`` assert ``_server is None`` - the handle was
cleared, which is true of a server still serving. That is the same gap
``test_policy_server_shutdown_does_not_kill_the_serving_thread.py`` records for
the other half of this teardown (a dead accept loop also leaves ``_server``
cleared), so the tests here assert on what a *client* observes instead.

The drain is bounded, not unbounded: a handler inside an inference call notices
the close only when that call returns, so
:data:`~strands_robots.inference.server.CONNECTION_DRAIN_S` caps the wait and the
outcome is logged. ``stop()`` returns ``None``, so that log is the only record
there can be of a connection still being served.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest

from strands_robots.inference import PolicyServer
from strands_robots.inference import server as server_mod
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.mock import MockPolicy

_LOGGER_NAME = "strands_robots.inference.server"

#: Text the teardown warning must carry. Asserted rather than the whole line so
#: a reworded diagnostic does not fail, but the fact does.
_STILL_SERVED = "still being served"


def _wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _connected_client(port: int) -> RemotePolicy:
    """A connected client that has exchanged one action request already."""
    client = RemotePolicy(host="127.0.0.1", port=port)
    client.set_robot_state_keys(["joint_0"])
    assert client.get_actions_sync({"joint_0": 0.0}, ""), "the server did not serve the client before teardown"
    return client


def _served_after_teardown(client: RemotePolicy) -> bool:
    """Whether ``client``'s existing connection is still answered with actions."""
    try:
        return bool(client.get_actions_sync({"joint_0": 0.0}, ""))
    except Exception:  # noqa: BLE001 - any refusal is the pass condition
        return False


class _BlockingPolicy(MockPolicy):
    """A policy whose inference call blocks until the test releases it.

    Models the handler this drain exists for: one inside a long inference call -
    a big VLA step, a stalled remote GPU - which cannot notice a closed
    connection until that call returns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Signal that inference has started, then wait for the test."""
        self.entered.set()
        self.release.wait(timeout=30.0)
        return super().get_actions_sync(observation_dict, instruction, **kwargs)


def test_a_connection_open_at_stop_is_no_longer_served() -> None:
    """``stop()`` ends the connections it accepted, not just the listener."""
    server = PolicyServer(policy_provider="mock", port=0).start()
    client = _connected_client(server.port)
    try:
        server.stop()

        assert not _served_after_teardown(client), (
            "a client connected when stop() was called was still served afterwards: "
            "the policy is still being invoked on a server reported stopped"
        )
    finally:
        client.close()
        server.stop()


def test_a_connection_open_when_serve_returns_is_no_longer_served() -> None:
    """The foreground ``serve()`` door carries the same obligation as ``stop()``."""
    server = PolicyServer(policy_provider="mock", port=0)
    serving = threading.Thread(target=server.serve, daemon=True)
    serving.start()
    try:
        assert _wait_until(lambda: server._server is not None), "serve() never bound"
        client = _connected_client(server.port)
        # serve() owns the socket in its own `with` block, so this is how it is
        # asked to return - the same call stop() makes.
        assert server._server is not None
        server._server.shutdown()
        serving.join(timeout=5.0)
        assert not serving.is_alive(), "serve() did not return"

        assert not _served_after_teardown(client), (
            "a client connected when serve() returned was still served afterwards"
        )
    finally:
        client.close()


def test_a_handler_still_inside_inference_is_named_in_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A connection the drain could not see off is reported, not waited out.

    Also pins that the drain does not queue behind the inference lock: the
    handler holds it for the whole 30s call, and the teardown must still return
    within the drain window rather than after the call.
    """
    monkeypatch.setattr(server_mod, "CONNECTION_DRAIN_S", 0.2)
    policy = _BlockingPolicy()
    server = PolicyServer(policy=policy, port=0).start()
    client = RemotePolicy(host="127.0.0.1", port=server.port)
    client.set_robot_state_keys(["joint_0"])
    asking = threading.Thread(target=_served_after_teardown, args=(client,), daemon=True)
    asking.start()
    try:
        assert policy.entered.wait(timeout=5.0), "the handler never reached inference"

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started

        assert _STILL_SERVED in caplog.text, (
            f"a handler still inside inference at teardown was not reported: {caplog.text!r}"
        )
        assert elapsed < 5.0, f"the teardown queued behind the inference call ({elapsed:.1f}s)"
    finally:
        policy.release.set()
        asking.join(timeout=10.0)
        client.close()


def test_a_teardown_with_nothing_connected_is_silent_and_prompt(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing to close is not a problem to report, and not a window to wait out."""
    server = PolicyServer(policy_provider="mock", port=0).start()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started

    assert _STILL_SERVED not in caplog.text, f"a teardown with no client reported one: {caplog.text!r}"
    assert elapsed < server_mod.CONNECTION_DRAIN_S, "the teardown waited out the drain window with nothing to drain"


def test_a_client_that_left_on_its_own_is_not_reported(caplog: pytest.LogCaptureFixture) -> None:
    """A connection closed by the client is gone, not a connection still served."""
    server = PolicyServer(policy_provider="mock", port=0).start()
    client = _connected_client(server.port)
    client.close()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        server.stop()

    assert _STILL_SERVED not in caplog.text, f"a client that had left was reported as served: {caplog.text!r}"
