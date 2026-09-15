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


class TestAnUnreadableFrameNamesThePeer:
    """A frame the vendored codec cannot read reports the peer, not the codec.

    Every other malformation this client can meet is a ``ConnectionError``
    naming the endpoint - a server that is absent, one that accepted the
    connection and went quiet, an unusable read budget. A frame the codec cannot
    read was the one that was not: the codec's own error escaped, naming neither
    the URI nor the bytes. It is the ordinary report for an ordinary mistake,
    because this package serves policies over a WebSocket in two wire formats.
    """

    #: (frame, codec error type) - one row per failure the vendored packer can
    #: raise on an inbound frame, so a new decode path is graded by adding a row.
    UNREADABLE = [
        pytest.param(b"<html>502 Bad Gateway</html>", "ExtraData", id="html-error-page"),
        pytest.param(mnp.packb({"server": "cosmos3"})[:4], "ValueError", id="truncated-msgpack"),
        pytest.param(mnp.packb({"a": 1}) + b"\x00\x00", "ExtraData", id="trailing-bytes"),
        pytest.param(
            mnp.packb(
                {"action": {b"__ndarray__": True, b"data": b"\x00\x00\x00\x00", b"dtype": "<f4", b"shape": (7, 7)}}
            ),
            "ValueError",
            id="array-shorter-than-its-shape",
        ),
        pytest.param(
            mnp.packb(
                {"action": {b"__ndarray__": True, b"data": b"\x00\x00\x00\x00", b"dtype": "nope", b"shape": (1,)}}
            ),
            "TypeError",
            id="array-dtype-that-is-not-one",
        ),
        pytest.param(
            mnp.packb({"action": {b"__ndarray__": True, b"data": b"\x00\x00\x00\x00", b"dtype": "<f4"}}),
            "KeyError",
            id="array-header-missing-shape",
        ),
    ]

    @pytest.mark.parametrize(("frame", "codec_error"), UNREADABLE)
    def test_the_handshake_read_names_the_peer(self, monkeypatch, frame, codec_error):
        """The connect handshake is the first read, so it is the first door."""
        _patch_connect(monkeypatch, _FakeWebsocket([frame]))

        with pytest.raises(ConnectionError) as excinfo:
            Cosmos3WebsocketClient(host="peer.test", port=4242).get_server_metadata()

        report = str(excinfo.value)
        assert "ws://peer.test:4242" in report, "names no endpoint"
        assert "metadata handshake" in report, "names no read"
        assert codec_error in report, "does not say what the codec could not do"
        assert repr(frame)[:24] in report, "shows none of the frame"

    @pytest.mark.parametrize(("frame", "codec_error"), UNREADABLE)
    def test_the_reply_read_names_the_peer(self, monkeypatch, frame, codec_error):
        """A server can handshake correctly and answer ``infer`` unreadably."""
        _patch_connect(monkeypatch, _FakeWebsocket([mnp.packb({"meta": "ok"}), frame]))

        with pytest.raises(ConnectionError) as excinfo:
            Cosmos3WebsocketClient(host="peer.test", port=4242).infer({"prompt": "pick the cube"})

        report = str(excinfo.value)
        assert "ws://peer.test:4242" in report, "names no endpoint"
        assert "action chunk" in report, "names no read"
        assert codec_error in report, "does not say what the codec could not do"

    def test_a_json_handshake_names_the_other_policy_server(self, monkeypatch):
        """The mistake this report exists for: dialling the JSON server.

        ``strands_robots.inference.server`` is the other WebSocket policy server
        in this package and it speaks JSON text frames. Reaching it with this
        client is a port mixed up between two servers on one host, and the
        codec answered ``a bytes-like object is required, not 'str'``.
        """
        _patch_connect(monkeypatch, _FakeWebsocket(['{"type": "ready", "protocol_version": 1}']))

        with pytest.raises(ConnectionError) as excinfo:
            Cosmos3WebsocketClient(host="peer.test", port=4242).get_server_metadata()

        report = str(excinfo.value)
        assert "ws://peer.test:4242" in report
        assert "strands_robots.inference.server" in report, "names no other server to check for"
        assert '{"type": "ready"' in report, "shows none of the frame the peer sent"

    def test_a_text_reply_stays_the_server_error_contract(self, monkeypatch):
        """The two doors differ, and only on purpose.

        The Cosmos 3 server marshals a dispatch failure back as a *text* frame,
        so a string reply is that server's own error - not an unreadable one.
        The handshake has no such contract, which is why the same bytes are a
        peer report there and a server traceback here.
        """
        _patch_connect(monkeypatch, _FakeWebsocket([mnp.packb({}), "Traceback: boom in forward()"]))

        with pytest.raises(RuntimeError, match="boom in forward"):
            Cosmos3WebsocketClient(host="peer.test", port=4242).infer({"prompt": "x"})

    def test_the_two_reads_are_named_apart(self, monkeypatch):
        """One report per read, so a caller mid-rollout knows which one failed.

        The silent-server report cannot see which of the two reads expired,
        because the first ``infer`` performs the handshake too. The decode runs
        at each read, so it always can - and a report naming neither read would
        satisfy every other assertion here.
        """
        blob = b"<html>nope</html>"
        _patch_connect(monkeypatch, _FakeWebsocket([blob]))
        with pytest.raises(ConnectionError) as at_handshake:
            Cosmos3WebsocketClient(host="peer.test", port=4242).get_server_metadata()

        _patch_connect(monkeypatch, _FakeWebsocket([mnp.packb({}), blob]))
        with pytest.raises(ConnectionError) as at_reply:
            Cosmos3WebsocketClient(host="peer.test", port=4242).infer({"prompt": "x"})

        assert str(at_handshake.value).replace("metadata handshake", "action chunk") == str(at_reply.value)

    def test_the_codec_failure_is_kept_as_the_cause(self, monkeypatch):
        """The frame is opaque, so the codec's own error is the only detail."""
        _patch_connect(monkeypatch, _FakeWebsocket(["not msgpack at all"]))
        with pytest.raises(ConnectionError) as excinfo:
            Cosmos3WebsocketClient(host="peer.test", port=4242).get_server_metadata()

        assert isinstance(excinfo.value.__cause__.__cause__, TypeError)

    def test_an_unreadable_handshake_is_not_cached_as_a_live_connection(self, monkeypatch):
        """A handshake that did not decode leaves nothing to serve the next call.

        The metadata frame is consumed by the handshake, so a connection cached
        behind a failed decode would hand the next ``infer`` a frame this client
        has already refused to read.
        """
        fake = _FakeWebsocket([b"<html>nope</html>"])
        _patch_connect(monkeypatch, fake)
        transport = _RawWebsocketTransport("peer.test", 4242)

        with pytest.raises(Exception, match="unreadable metadata handshake"):
            transport.get_server_metadata()

        assert transport._ws is None, "a connection whose handshake did not decode was cached"
        assert fake.closed, "the discarded connection was left open"

    def test_an_absent_server_still_earns_the_start_it_hint(self, monkeypatch):
        """The new clause sits ahead of ``except OSError`` without shadowing it.

        A ``ConnectionError`` is an ``OSError``, so a peer report raised as one
        from the transport - or re-raised ahead of that clause to protect it -
        would take over the absent-server case as well, which is the one report
        the start-the-server hint exists to replace.
        """
        import websockets.sync.client as wsc

        monkeypatch.setattr(wsc, "connect", lambda uri, **kw: (_ for _ in ()).throw(ConnectionRefusedError(111)))
        client = Cosmos3WebsocketClient(host="peer.test", port=4242)

        for call in (client.get_server_metadata, lambda: client.infer({"prompt": "x"})):
            with pytest.raises(ConnectionError) as excinfo:
                call()
            assert "Start it first" in str(excinfo.value)
            assert "unreadable" not in str(excinfo.value)
