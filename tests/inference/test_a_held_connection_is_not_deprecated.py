"""A service policy connects to its server without a ``DeprecationWarning``.

Two clients hold ONE ``websockets`` connection across every request they make:
``RemotePolicy`` (``strands_robots.inference.client``) and the Cosmos 3
transport (``strands_robots.policies.cosmos3.client._RawWebsocketTransport``).
Neither can be a ``with connect(...)`` block, because the connection has to
outlive the call that opened it - it is the socket the whole rollout is served
on.

websockets 17.1 deprecates exactly that shape when it is spelled as a bare
``connect(...)``: the connection it returns warns on its first read::

    DeprecationWarning: connect() must be used as a context manager:
    with connect(...) as websocket: ...; alternatively, use
    websocket = connect(..., legacy=True) to connect directly

and its documentation announces that ``connect()`` changes behaviour once the
deprecation period ends. Measured on the unchanged sources with the 17.1 wheel:
``RemotePolicy.get_actions_sync`` against a ``PolicyServer(mock)`` and
``_RawWebsocketTransport.infer`` against a ``websockets.sync.server`` echo both
raised the warning from inside the handshake read, so a rollout run with
``-W error::DeprecationWarning`` - the setting that turns every warning into a
finding - fell at the first connect with an error that named websockets rather
than the robot.

``legacy=True`` is the supported spelling of "return the connection directly",
and it is what both clients pass now. The flag arrives in 17.1 (17.0 has no
such parameter), which is what moved the packaging floor; the floor itself is
owned by ``tests/test_websockets_floor_ships_the_imported_api.py``.

Both scenarios run against a REAL local server rather than a fake connection,
because the warning is raised by the real ``ClientConnection`` on its first
``recv`` and a stand-in would not raise it.
"""

from __future__ import annotations

import threading
import warnings

import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

from strands_robots.inference import PolicyServer, RemotePolicy  # noqa: E402
from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp  # noqa: E402
from strands_robots.policies.cosmos3.client import _RawWebsocketTransport  # noqa: E402

OBSERVATION = {"shoulder_pan.pos": 0.1, "elbow_flex.pos": 0.2}


def _no_deprecation_warning(records: list[warnings.WarningMessage], where: str) -> None:
    deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
    assert not deprecations, f"{where} raised a DeprecationWarning on a held connection: " + "; ".join(
        f"{r.filename}:{r.lineno}: {r.message}" for r in deprecations
    )


def test_remote_policy_holds_its_connection_without_a_deprecation_warning() -> None:
    server = PolicyServer(policy_provider="mock", port=0).start()
    try:
        policy = RemotePolicy(host="127.0.0.1", port=server.port, connect_timeout=5.0, request_timeout=5.0)
        try:
            policy.set_robot_state_keys(list(OBSERVATION))
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                first = policy.get_actions_sync(OBSERVATION, "hold still")
                second = policy.get_actions_sync(OBSERVATION, "hold still")
            _no_deprecation_warning(records, "RemotePolicy")
            # The same connection served both requests: this is the held shape
            # the flag exists for, not a connect-per-call that a ``with`` block
            # could have expressed.
            assert first and second
        finally:
            policy.close()
    finally:
        server.stop()


def _serve_cosmos3_echo():
    """A minimal RoboLab-shaped server: metadata handshake, then one action per observation."""
    from websockets.sync.server import serve

    packer = mnp.Packer()

    def handler(connection) -> None:
        connection.send(packer.pack({}))  # metadata handshake
        for frame in connection:
            observation = mnp.unpackb(frame)
            connection.send(packer.pack({"action": [[0.0] * len(observation)]}))

    server = serve(handler, "127.0.0.1", 0, max_size=None, compression=None)
    thread = threading.Thread(target=server.serve_forever, name="cosmos3-echo", daemon=True)
    thread.start()
    return server, server.socket.getsockname()[1]


def test_cosmos3_transport_holds_its_connection_without_a_deprecation_warning() -> None:
    server, port = _serve_cosmos3_echo()
    try:
        transport = _RawWebsocketTransport("127.0.0.1", port, read_timeout=5.0)
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                first = transport.infer(OBSERVATION)
                second = transport.infer(OBSERVATION)
            _no_deprecation_warning(records, "_RawWebsocketTransport")
            assert "action" in first and "action" in second
        finally:
            transport.close()
    finally:
        server.shutdown()
