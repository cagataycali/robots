"""Wire-transport contract for the VERA policy WebSocket client.

The VERA policy server is reached over a msgpack + NumPy WebSocket protocol
spoken by ``strands_robots.policies.vera.client.VeraWebsocketClient``. Prior
coverage exercised the policy only through an in-memory fake client, so the
real client - the lazy connect + metadata handshake, packing a request,
unpacking an action chunk, the string-payload server-error sentinel, and the
best-effort reset/close semantics - was untested end to end. A regression in
any of those silently corrupts every action a remote VERA policy returns (or
turns a clean server error into a hang), so these tests pin the on-the-wire
behavior using a fake WebSocket connection (no GPU, no live server, no vera pkg).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("websockets", reason="websockets needed for the raw transport")

from strands_robots.policies.vera import _msgpack_numpy as mnp  # noqa: E402
from strands_robots.policies.vera.client import VeraWebsocketClient  # noqa: E402


class _FakeWebsocket:
    """In-memory stand-in for a ``websockets`` sync connection.

    Hands back queued, msgpack-packed payloads on ``recv()`` (the first is the
    server metadata handshake) and records everything ``send()`` writes so a
    test can decode and assert what the client actually put on the wire.
    """

    def __init__(self, recv_payloads):
        self._recv_queue = list(recv_payloads)
        self.sent = []
        self.closed = False

    def recv(self):
        return self._recv_queue.pop(0)

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, fake_ws, recorder=None):
    """Route ``websockets.sync.client.connect`` to return ``fake_ws``.

    When ``recorder`` is given, the connect URI/kwargs are captured so a test
    can assert what the client passed.
    """
    import websockets.sync.client as wsc

    def fake_connect(uri, **kwargs):
        if recorder is not None:
            recorder["uri"] = uri
            recorder["kwargs"] = kwargs
        return fake_ws

    monkeypatch.setattr(wsc, "connect", fake_connect)


def test_connect_consumes_metadata_handshake(monkeypatch):
    """The first recv on connect is consumed as the VeraServerConfig blob and
    surfaced through ``get_server_metadata`` (a copy, not the live dict)."""
    meta = {"action_dim": 7, "action_horizon": 16, "view_keys": ["cam"]}
    fake = _FakeWebsocket([mnp.packb(meta)])
    recorder = {}
    _patch_connect(monkeypatch, fake, recorder)

    client = VeraWebsocketClient(host="example.test", port=9999)
    got = client.get_server_metadata()

    assert got == meta
    # A defensive copy: mutating the returned dict must not corrupt client state.
    got["action_dim"] = -1
    assert client.get_server_metadata()["action_dim"] == 7
    # Generous open timeout and uncapped frame size are passed through.
    assert recorder["uri"] == "ws://example.test:9999"
    assert recorder["kwargs"]["max_size"] is None
    assert recorder["kwargs"]["open_timeout"] == 600


def test_connect_is_lazy_and_cached(monkeypatch):
    """No socket opens at construction; the connection is established once and
    reused across calls (the handshake recv happens exactly once)."""
    connects = {"n": 0}
    meta = {"action_dim": 2}
    fake = _FakeWebsocket([mnp.packb(meta), mnp.packb({"action": np.zeros((1, 2), np.float32)})])

    import websockets.sync.client as wsc

    def counting_connect(uri, **kwargs):
        connects["n"] += 1
        return fake

    monkeypatch.setattr(wsc, "connect", counting_connect)

    client = VeraWebsocketClient()
    assert connects["n"] == 0  # construction does not connect
    client.get_server_metadata()
    client.infer({"context_rgb": np.zeros((1, 4, 4, 3), np.uint8)})
    assert connects["n"] == 1  # single connection reused


def test_connection_refused_raises_actionable_hint(monkeypatch):
    """A refused/unreachable server is converted to a ConnectionError whose
    message tells the operator exactly how to launch the server."""
    import websockets.sync.client as wsc

    def refuse(uri, **kwargs):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(wsc, "connect", refuse)

    client = VeraWebsocketClient(port=8820)
    with pytest.raises(ConnectionError) as ei:
        client.infer({"context_rgb": np.zeros((1, 4, 4, 3), np.uint8)})
    msg = str(ei.value)
    assert "ws://127.0.0.1:8820" in msg
    assert "start_vera_server" in msg


def test_infer_round_trips_action_chunk(monkeypatch):
    """infer packs the observation (with endpoint tag) and unpacks the action
    chunk via the vendored NumPy codec with dtype/shape/values intact."""
    action = np.arange(32, dtype=np.float32).reshape(16, 2)
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), mnp.packb({"action": action})])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    ctx = np.zeros((1, 8, 8, 3), np.uint8)
    out = client.infer({"context_rgb": ctx, "view_keys": ["cam"], "session_id": "s1"})

    assert isinstance(out["action"], np.ndarray)
    assert out["action"].dtype == np.float32
    assert np.array_equal(out["action"], action)
    # The wire message carries the endpoint tag and the original observation.
    decoded = mnp.unpackb(fake.sent[0])
    assert decoded["endpoint"] == "infer"
    assert decoded["session_id"] == "s1"
    assert np.array_equal(decoded["context_rgb"], ctx)


def test_infer_string_payload_is_error_sentinel(monkeypatch):
    """A string response is the server's error sentinel and must raise rather
    than be mistaken for a (corrupt) action payload."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), "CUDA out of memory"])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        client.infer({"context_rgb": np.zeros((1, 4, 4, 3), np.uint8)})


def test_reset_round_trips_and_tags_endpoint(monkeypatch):
    """reset forwards its info dict tagged with the reset endpoint and accepts
    the server's 'reset successful' acknowledgement without raising."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), "reset successful"])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    client.reset({"seed": 7})

    decoded = mnp.unpackb(fake.sent[0])
    assert decoded["endpoint"] == "reset"
    assert decoded["seed"] == 7


def test_reset_is_best_effort_when_server_absent(monkeypatch):
    """reset must never propagate a connection failure - it is best effort, so
    a down server is swallowed (no socket was ever opened)."""
    import websockets.sync.client as wsc

    def refuse(uri, **kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(wsc, "connect", refuse)

    client = VeraWebsocketClient()
    client.reset()  # must not raise


def test_reset_error_sentinel_raises(monkeypatch):
    """Any non-'reset successful' string is a server error and must raise."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), "reset failed: bad state"])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    with pytest.raises(RuntimeError, match="reset failed"):
        client.reset()


def test_configure_round_trips_applied_knobs(monkeypatch):
    """configure tags the configure endpoint and unpacks the server's applied
    knob dict via the NumPy codec."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), mnp.packb({"applied": {"motion_plan_scale": 1.5}})])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    out = client.configure({"motion_plan_scale": 1.5})

    assert out["applied"]["motion_plan_scale"] == 1.5
    decoded = mnp.unpackb(fake.sent[0])
    assert decoded["endpoint"] == "configure"
    assert decoded["motion_plan_scale"] == 1.5


def test_configure_string_payload_raises(monkeypatch):
    """A string configure response is the error sentinel and must raise."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2}), "unknown knob"])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    with pytest.raises(RuntimeError, match="unknown knob"):
        client.configure({"bogus": 1})


def test_close_is_idempotent_and_releases_socket(monkeypatch):
    """close shuts the live socket and clears the handle so a second close is a
    no-op (and the next call would reconnect)."""
    fake = _FakeWebsocket([mnp.packb({"action_dim": 2})])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    client.get_server_metadata()  # forces a connection
    client.close()
    assert fake.closed is True
    client.close()  # second close is a harmless no-op


def test_close_swallows_socket_close_error(monkeypatch):
    """A raising ``ws.close()`` is swallowed (close is best effort) and the
    handle is still cleared."""

    class _BadClose(_FakeWebsocket):
        def close(self):
            raise OSError("already gone")

    fake = _BadClose([mnp.packb({"action_dim": 2})])
    _patch_connect(monkeypatch, fake)

    client = VeraWebsocketClient()
    client.get_server_metadata()
    client.close()  # must not raise
    assert client._ws is None


# --------------------------------------------------------------------------- #
# Vendored msgpack+NumPy codec - scalar + error paths
# --------------------------------------------------------------------------- #
class TestMsgpackNumpyCodec:
    """The wire codec underpins every request/response above; the ndarray happy
    path is covered in test_vera_unit, so these pin the numpy-scalar branch and
    the non-wire-safe dtype guard that a raw ndarray test never reaches."""

    def test_numpy_scalar_round_trips_with_dtype(self):
        for scalar in (np.float32(3.5), np.int64(-7), np.uint8(255), np.bool_(True)):
            out = mnp.unpackb(mnp.packb({"v": scalar}))["v"]
            assert out == scalar
            # The numpy dtype is preserved, not coerced to a Python builtin.
            assert isinstance(out, np.generic)
            assert out.dtype == scalar.dtype

    def test_object_dtype_ndarray_is_rejected(self):
        bad = np.array([{"not": "wire-safe"}], dtype=object)
        with pytest.raises(ValueError, match="cannot serialize ndarray of dtype"):
            mnp.packb({"x": bad})

    def test_plain_payload_passes_through_untouched(self):
        payload = {"prompt": "stack", "ids": [1, 2, 3], "ok": True}
        assert mnp.unpackb(mnp.packb(payload)) == payload

    def test_unsupported_type_is_not_silently_swallowed(self):
        # A type the codec does not special-case falls through _encode unchanged
        # and msgpack rejects it - corruption is impossible, the pack just fails.
        with pytest.raises(TypeError):
            mnp.packb({"s": {1, 2, 3}})
