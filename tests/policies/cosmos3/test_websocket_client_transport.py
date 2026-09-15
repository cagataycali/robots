"""Wire-transport contract for the Cosmos 3 RoboLab WebSocket client.

The Cosmos 3 policy server is reached over a msgpack + NumPy WebSocket protocol
spoken by ``strands_robots.policies.cosmos3.client``. Prior coverage exercised
only the connection-*refused* error path; the happy-path round trip - the
metadata handshake on connect, packing an observation, receiving and unpacking
an action chunk, and the string-payload server-error contract - was untested.
A regression in any of those silently corrupts every action a remote Cosmos 3
policy returns, so these tests pin the on-the-wire behavior end to end using a
fake WebSocket connection (no GPU, no live server).
"""

import numpy as np
import pytest

pytest.importorskip("websockets", reason="websockets needed for the raw transport")

from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp  # noqa: E402
from strands_robots.policies.cosmos3.client import (  # noqa: E402
    Cosmos3WebsocketClient,
    _RawWebsocketTransport,
)


class _FakeWebsocket:
    """In-memory stand-in for a ``websockets`` sync connection.

    Hands back queued, msgpack-packed payloads on ``recv()`` (the first is the
    server metadata handshake) and records everything ``send()`` writes so a
    test can decode and assert the observation actually put on the wire.
    """

    def __init__(self, recv_payloads):
        self._recv_queue = list(recv_payloads)
        self.sent = []
        self.closed = False

    def recv(self, timeout=None):
        """Hand back the next queued payload.

        ``timeout`` is accepted and ignored: the client states a deadline on
        every read, and a double that refused the keyword would be pinning its
        own signature rather than the wire.
        """
        return self._recv_queue.pop(0)

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, fake_ws, recorder=None):
    """Route ``websockets.sync.client.connect`` to return ``fake_ws``.

    When ``recorder`` is given, the connect kwargs are captured so a test can
    assert headers / URI passed by the transport.
    """
    import websockets.sync.client as wsc

    def fake_connect(uri, **kwargs):
        if recorder is not None:
            recorder["uri"] = uri
            recorder["kwargs"] = kwargs
        return fake_ws

    monkeypatch.setattr(wsc, "connect", fake_connect)


def test_raw_transport_handshake_then_infer_round_trips(monkeypatch):
    """First recv is consumed as the metadata handshake; infer packs the obs
    and unpacks the action chunk via the vendored NumPy codec."""
    action = np.arange(8, dtype=np.float32).reshape(4, 2)
    fake = _FakeWebsocket([mnp.packb({"meta": "hello"}), mnp.packb({"action": action})])
    recorder = {}
    _patch_connect(monkeypatch, fake, recorder)

    transport = _RawWebsocketTransport("example.test", 9999, api_key=None)
    obs = {"prompt": "pick the cube", "state": np.zeros(7, dtype=np.float32)}
    out = transport.infer(obs)

    # The action chunk survives the round trip with dtype/shape/values intact.
    assert isinstance(out["action"], np.ndarray)
    assert out["action"].dtype == np.float32
    assert np.array_equal(out["action"], action)
    # The observation actually placed on the wire decodes back to the input.
    assert len(fake.sent) == 1
    decoded = mnp.unpackb(fake.sent[0])
    assert decoded["prompt"] == "pick the cube"
    assert np.array_equal(decoded["state"], obs["state"])
    # The handshake payload was consumed, so the queue holds nothing extra.
    assert recorder["uri"] == "ws://example.test:9999"


def test_raw_transport_forwards_api_key_header(monkeypatch):
    """An api_key is forwarded as an Authorization: Api-Key header; without one
    no auth header is sent."""
    recorder = {}
    fake = _FakeWebsocket([mnp.packb({})])
    _patch_connect(monkeypatch, fake, recorder)
    _RawWebsocketTransport("h", 1, api_key="secret-token")._ensure()
    assert recorder["kwargs"]["additional_headers"] == {"Authorization": "Api-Key secret-token"}

    recorder2 = {}
    fake2 = _FakeWebsocket([mnp.packb({})])
    _patch_connect(monkeypatch, fake2, recorder2)
    _RawWebsocketTransport("h", 1, api_key=None)._ensure()
    assert recorder2["kwargs"]["additional_headers"] is None


def test_raw_transport_connects_once_and_caches(monkeypatch):
    """The connection is established lazily and reused across calls."""
    connects = {"n": 0}
    fake = _FakeWebsocket([mnp.packb({}), mnp.packb({"action": np.zeros((1, 1))})])

    import websockets.sync.client as wsc

    def counting_connect(uri, **kwargs):
        connects["n"] += 1
        return fake

    monkeypatch.setattr(wsc, "connect", counting_connect)

    transport = _RawWebsocketTransport("h", 1)
    assert transport.get_server_metadata() == {}
    transport.infer({"prompt": "x"})
    assert connects["n"] == 1, "transport must connect exactly once and cache the socket"


def test_raw_transport_string_payload_is_server_error(monkeypatch):
    """A str frame from the server signals an inference error and is raised as
    RuntimeError carrying the server message (not silently unpacked)."""
    fake = _FakeWebsocket([mnp.packb({}), "traceback: boom on server"])
    _patch_connect(monkeypatch, fake)
    transport = _RawWebsocketTransport("h", 1)
    with pytest.raises(RuntimeError, match="boom on server"):
        transport.infer({"prompt": "x"})


def test_raw_transport_reset_is_noop(monkeypatch):
    """The client-side raw transport is stateless: reset does nothing and never
    requires a connection."""
    transport = _RawWebsocketTransport("h", 1)
    assert transport.reset() is None


def test_client_infer_round_trips_through_real_transport(monkeypatch):
    """Cosmos3WebsocketClient lazily builds the raw transport and returns the
    server's unpacked action chunk."""
    action = np.ones((6, 3), dtype=np.float32)
    fake = _FakeWebsocket([mnp.packb({}), mnp.packb({"action": action, "server_timing": {"infer_ms": 2.0}})])
    _patch_connect(monkeypatch, fake)

    client = Cosmos3WebsocketClient(host="h", port=1)
    out = client.infer({"prompt": "stack the blocks"})
    assert np.array_equal(out["action"], action)
    assert out["server_timing"]["infer_ms"] == 2.0


def test_client_caches_transport_across_calls(monkeypatch):
    """_ensure_client builds the transport once and returns the cached instance
    on later calls (get_server_metadata + infer share one connection)."""
    fake = _FakeWebsocket([mnp.packb({}), mnp.packb({"action": np.zeros((1, 1))})])
    connects = {"n": 0}

    import websockets.sync.client as wsc

    def counting_connect(uri, **kwargs):
        connects["n"] += 1
        return fake

    monkeypatch.setattr(wsc, "connect", counting_connect)

    client = Cosmos3WebsocketClient(host="h", port=1)
    assert client.get_server_metadata() == {}
    first = client._ensure_client()
    client.infer({"prompt": "x"})
    second = client._ensure_client()
    assert first is second
    assert connects["n"] == 1


def test_client_reset_forwards_to_transport(monkeypatch):
    """reset() forwards to the transport's reset when present."""
    calls = {"n": 0}

    class _ResettableClient:
        def reset(self):
            calls["n"] += 1

    client = Cosmos3WebsocketClient(host="h", port=1)
    client._client = _ResettableClient()
    client.reset()
    assert calls["n"] == 1


def test_client_reset_swallows_transport_failure(monkeypatch):
    """reset() is a best-effort hint: a transport whose reset raises must not
    propagate (mirrors Gr00tPolicy.reset)."""

    class _AngryClient:
        def reset(self):
            raise RuntimeError("server hung up")

    client = Cosmos3WebsocketClient(host="h", port=1)
    client._client = _AngryClient()
    # Must not raise.
    assert client.reset() is None


def test_client_reset_handles_transport_without_reset(monkeypatch):
    """A transport object that exposes no callable reset is tolerated."""

    class _NoResetClient:
        reset = None

    client = Cosmos3WebsocketClient(host="h", port=1)
    client._client = _NoResetClient()
    assert client.reset() is None


def test_client_ensure_client_wraps_construction_failure(monkeypatch):
    """If building the raw transport raises a connection/OS error, it surfaces
    as a ConnectionError carrying the actionable server-start hint."""

    def boom(*args, **kwargs):
        raise OSError("socket setup failed")

    monkeypatch.setattr("strands_robots.policies.cosmos3.client._RawWebsocketTransport", boom)
    client = Cosmos3WebsocketClient(host="myhost", port=4321)
    with pytest.raises(ConnectionError) as ei:
        client._ensure_client()
    msg = str(ei.value)
    assert "ws://myhost:4321" in msg
    assert "action_policy_server_robolab" in msg


def test_client_get_server_metadata_wraps_connection_error(monkeypatch):
    """get_server_metadata raises ConnectionError with the hint when the
    transport cannot reach the server."""

    class _DownTransport:
        def __init__(self, *args, **kwargs):
            pass

        def get_server_metadata(self):
            raise ConnectionRefusedError("refused")

    monkeypatch.setattr("strands_robots.policies.cosmos3.client._RawWebsocketTransport", _DownTransport)
    client = Cosmos3WebsocketClient(host="h", port=1)
    with pytest.raises(ConnectionError, match="healthz"):
        client.get_server_metadata()


def test_client_transport_deprecation_warning_is_logged(monkeypatch, caplog):
    """A legacy transport selector is coerced to 'raw' and logs a deprecation
    warning naming the removed openpi-client dependency."""
    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.policies.cosmos3.client"):
        client = Cosmos3WebsocketClient(host="h", port=1, transport="openpi")
    assert client.transport == "raw"
    assert any("deprecated" in r.getMessage() for r in caplog.records)


def test_client_infer_wraps_connection_error(monkeypatch):
    """infer raises ConnectionError with the actionable hint when the transport
    loses the connection mid-call."""

    class _FlakyTransport:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, observation):
            raise OSError("connection dropped")

    monkeypatch.setattr("strands_robots.policies.cosmos3.client._RawWebsocketTransport", _FlakyTransport)
    client = Cosmos3WebsocketClient(host="h", port=1)
    with pytest.raises(ConnectionError, match="action_policy_server_robolab"):
        client.infer({"prompt": "x"})


# A frame the vendored msgpack codec cannot read, and the codec's own verdict on
# it. Every other malformation on this wire is a ConnectionError naming the
# server and the endpoint (refused connect, silent server); these used to escape
# as the codec's bare complaint, naming neither, and not saying which read the
# peer had answered.
_UNREADABLE_FRAMES = [
    (b"<html><body>502 Bad Gateway</body></html>", "ExtraData"),
    (b'{"action": [[0.0, 1.0]]}', "ExtraData"),
    (b"", "ValueError"),
]
_DOORS = ("metadata handshake", "reply")
#: (frame, codec error, read) - a text frame is unreadable at the handshake only,
#: because the reply door has its own server-error-string contract (pinned by
#: test_raw_transport_string_payload_is_server_error) that runs before the codec.
# ``str | bytes`` because that is what a websocket delivers: a text frame arrives
# as ``str``, a binary one as ``bytes``, and both reach the same codec.
_UNREADABLE_CASES: list[tuple[str | bytes, str, str]] = [
    (frame, codec, door) for frame, codec in _UNREADABLE_FRAMES for door in _DOORS
]
_UNREADABLE_CASES.append(("traceback: server exploded", "TypeError", "metadata handshake"))


def _dial(monkeypatch, frame, door):
    """Answer one read of the wire with *frame*, and return the call that made it."""
    payloads = [frame] if door == "metadata handshake" else [mnp.packb({"meta": "hello"}), frame]
    fake = _FakeWebsocket(payloads)
    _patch_connect(monkeypatch, fake)
    client = Cosmos3WebsocketClient(host="cosmos.test", port=8000)
    if door == "metadata handshake":
        return client, fake, client.get_server_metadata
    return client, fake, lambda: client.infer({"prompt": "pick the cube"})


@pytest.mark.parametrize(
    ("frame", "codec", "door"),
    _UNREADABLE_CASES,
    ids=[f"{codec}-at-{door.split()[-1]}" for _, codec, door in _UNREADABLE_CASES],
)
def test_an_unreadable_frame_names_the_peer_the_read_and_the_frame(monkeypatch, frame, codec, door):
    """A frame that is not msgpack reports the server, the endpoint, which read it
    answered and its opening bytes - not the codec's bare complaint."""
    _client, _fake, call = _dial(monkeypatch, frame, door)

    with pytest.raises(ConnectionError) as ei:
        call()

    msg = str(ei.value)
    assert "Cosmos 3 policy server" in msg, "the report must name the service that answered"
    assert "ws://cosmos.test:8000" in msg, "the report must name the endpoint dialled"
    assert f"a {door} frame" in msg, "the report must name which read the peer answered"
    assert codec in msg, "the codec's verdict is the evidence and is kept"
    assert repr(frame[:60]) in msg, "the opening bytes are what identify the peer that answered"
    assert type(ei.value.__cause__).__name__ == codec, "the codec failure stays the cause"
    # The start-the-server hint is written for an *absent* server, and it reaches
    # the caller through ``except OSError`` - which a ConnectionError satisfies.
    # It must not replace the report of a server that just answered.
    assert "Could not reach" not in msg
    assert "healthz" not in msg


def test_one_frame_at_both_doors_differs_only_in_the_read(monkeypatch):
    """The same unreadable frame at either read yields the same report bar the
    read it names, so a caller mid-rollout can tell the two apart."""
    frame = b"<html><body>502 Bad Gateway</body></html>"
    reports = {}
    for door in _DOORS:
        _client, _fake, call = _dial(monkeypatch, frame, door)
        with pytest.raises(ConnectionError) as ei:
            call()
        reports[door] = str(ei.value)

    assert reports["metadata handshake"].replace("metadata handshake", "reply") == reports["reply"]


def test_an_unreadable_handshake_discards_the_connection(monkeypatch):
    """A handshake that could not be read leaves no cached connection: the frame
    behind it is unread, so a later infer would have taken it for an action chunk."""
    client, fake, call = _dial(monkeypatch, b"<html>502</html>", "metadata handshake")

    with pytest.raises(ConnectionError):
        call()

    assert fake.closed is True, "the connection being abandoned must be closed"
    assert client._client._ws is None, "a connection whose handshake failed must not be cached"
