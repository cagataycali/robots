"""Client-side fail-fast contract tests for the remote-policy handshake + reply.

The end-to-end round-trip tests in ``test_remote_policy_roundtrip.py`` drive a
real, well-behaved :class:`~strands_robots.inference.server.PolicyServer`, so the
client's defensive branches for a *mis*behaving peer are never exercised there.
These tests stub the WebSocket transport with a fake connection that returns
crafted frames, pinning the "loud failure" seams a real server can never
produce:

* a handshake frame whose ``type`` is not ``ready``;
* a ``ready`` handshake advertising a mismatched ``protocol_version``;
* an ``actions`` reply whose ``actions`` payload is not a list;
* a frame at either read that the *codec* cannot read at all - what a peer
  speaking another wire format on that URI actually sends.

Per the module contract (loud error propagation, never a silent zero action),
each seam must raise rather than proceed with a degraded connection or chunk,
and it must raise the documented ``ConnectionError``/``RuntimeError`` naming the
peer rather than a codec error naming neither the URI nor the exchange.
"""

import json
from collections.abc import Sequence
from typing import Any

import pytest

from strands_robots.inference import PolicyServer, RemotePolicy, protocol
from strands_robots.policies import MockPolicy


class _FakeConnection:
    """Minimal stand-in for a ``websockets.sync`` connection.

    ``recv`` pops from a pre-seeded queue of already-serialized frames; ``send``
    records outbound frames so a test can assert the client got as far as
    issuing a request before the reply tripped a contract.
    """

    def __init__(self, frames: Sequence[str | bytes]) -> None:
        self._frames: list[str | bytes] = list(frames)
        self.sent: list[str] = []
        self.closed = False

    def recv(self, timeout: float | None = None) -> str | bytes:  # noqa: ARG002 - parity with real API
        if not self._frames:
            raise AssertionError("client called recv() more times than the test seeded frames")
        return self._frames.pop(0)

    def send(self, text: str) -> None:
        self.sent.append(text)

    def close(self) -> None:
        self.closed = True


def _patch_connect(monkeypatch: pytest.MonkeyPatch, frames: Sequence[str | bytes]) -> _FakeConnection:
    """Route ``RemotePolicy._connect`` onto a fake connection yielding ``frames``."""
    fake = _FakeConnection(frames)
    # _connect() does ``from websockets.sync.client import connect`` so the patch
    # target is the attribute on that module, resolved at call time.
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: fake)
    return fake


def _ready_frame(**overrides: Any) -> str:
    """A well-formed ``ready`` handshake frame, with optional field overrides."""
    frame = {
        "type": protocol.MSG_READY,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "metadata": {"provider_name": "recording", "requires_images": False},
    }
    frame.update(overrides)
    return protocol.dumps(frame)


def test_handshake_wrong_type_raises_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first frame that is not a ``ready`` handshake is a fatal ConnectionError."""
    _patch_connect(monkeypatch, [protocol.dumps({"type": protocol.MSG_OK})])
    client = RemotePolicy(endpoint="ws://127.0.0.1:65535")

    with pytest.raises(ConnectionError, match=f"expected a '{protocol.MSG_READY}' handshake"):
        # Any attribute access forces the lazy connect + handshake.
        _ = client.requires_images


def test_handshake_version_mismatch_raises_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ready`` frame advertising a different protocol_version is rejected loudly."""
    bad_version = protocol.PROTOCOL_VERSION + 1
    _patch_connect(monkeypatch, [_ready_frame(protocol_version=bad_version)])
    client = RemotePolicy(endpoint="ws://127.0.0.1:65535")

    with pytest.raises(ConnectionError, match="protocol version mismatch"):
        _ = client.requires_images


def test_handshake_missing_version_raises_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ready`` frame with no protocol_version (``None``) is a mismatch, not a pass."""
    frame = protocol.dumps({"type": protocol.MSG_READY, "metadata": {}})
    _patch_connect(monkeypatch, [frame])
    client = RemotePolicy(endpoint="ws://127.0.0.1:65535")

    with pytest.raises(ConnectionError, match="protocol version mismatch"):
        _ = client.requires_images


def test_non_list_action_chunk_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``actions`` reply whose payload is not a list is a loud RuntimeError.

    A malformed chunk must never be substituted with a silent zero action, so
    the client raises rather than returning the non-list value downstream.
    """
    frames = [
        _ready_frame(),  # handshake succeeds
        # get_actions reply: correct type, but ``actions`` is a dict, not a list.
        protocol.dumps({"type": protocol.MSG_ACTIONS, "actions": {"j0": 0.5}}),
    ]
    _patch_connect(monkeypatch, frames)
    client = RemotePolicy(endpoint="ws://127.0.0.1:65535")

    with pytest.raises(RuntimeError, match="non-list action chunk"):
        client.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, "pick the cube")


#: What a peer that is not a ``PolicyServer`` answers with, and the codec failure
#: each frame produces. All three are ``ValueError``\s - a ``UnicodeDecodeError``
#: for bytes that are not UTF-8, a ``JSONDecodeError`` for text that is not JSON,
#: a plain one for JSON that is not an object - so the client absorbs them in a
#: single clause and the table proves the three arrive as one report.
unreadable_frame = pytest.mark.parametrize(
    ("frame", "cause"),
    [
        pytest.param(b"\x82\xa4type\xa5ready", UnicodeDecodeError, id="msgpack_bytes"),
        pytest.param("<html><body>502 Bad Gateway</body></html>", json.JSONDecodeError, id="an_http_error_page"),
        pytest.param(json.dumps([1, 2, 3]), ValueError, id="json_that_is_not_an_object"),
    ],
)

ENDPOINT = "ws://127.0.0.1:65535"


class TestAFrameTheCodecCannotReadNamesThePeerLikeEveryOtherMalformation:
    """A peer answering in another wire format is refused, not decoded.

    Dialling the wrong port is an ordinary mistake - this package also serves
    policies over a WebSocket in msgpack - and every *other* malformation on
    this connection is already a ``ConnectionError`` naming the URI. A frame the
    codec could not read was the exception: it escaped as
    ``'utf-8' codec can't decode byte 0x82 in position 0``, out of methods
    documented to raise ``ConnectionError``, naming neither the endpoint nor
    which read it answered.
    """

    def test_the_server_end_of_this_protocol_already_refuses_and_keeps_serving(self) -> None:
        """The rule is the protocol's, not one client's.

        ``PolicyServer`` marshals a frame it cannot parse back as an ``error``
        message and goes on serving the same connection, so the client half
        answering with a bare codec error was the odd end of one conversation.
        """
        server = PolicyServer(policy=MockPolicy(), host="127.0.0.1", port=0).start()
        client = RemotePolicy(host="127.0.0.1", port=server.port, request_timeout=5.0)
        try:
            _ = client.requires_images  # real handshake over a real socket
            assert client._ws is not None
            client._ws.send(b"\x82\xa4type\xa5ready")
            answer = protocol.loads(client._ws.recv(timeout=5.0))
            assert answer["type"] == protocol.MSG_ERROR
            assert "UnicodeDecodeError" in answer["error"]
            # Still serving: the unreadable frame cost the frame, not the stream.
            assert client.get_actions_sync({"joints": {}}, "pick the cube") is not None
        finally:
            client.close()
            server.stop()

    @unreadable_frame
    def test_an_unreadable_handshake_names_the_uri_and_the_read(
        self, monkeypatch: pytest.MonkeyPatch, frame: str | bytes, cause: type[Exception]
    ) -> None:
        fake = _patch_connect(monkeypatch, [frame])
        client = RemotePolicy(endpoint=ENDPOINT)

        with pytest.raises(ConnectionError) as refused:
            _ = client.requires_images

        report = str(refused.value)
        assert ENDPOINT in report, report
        assert "handshake" in report, report
        # The codec failure is the cause, not the report: nothing is swallowed.
        assert isinstance(refused.value.__cause__, cause)
        # The frame's own opening bytes, so an operator can recognise the codec
        # (msgpack, an HTML error page) they actually reached.
        assert repr(frame[:60]) in report, report
        # A refused handshake is not left cached, so a retry opens a fresh one.
        assert client._ws is None
        assert fake.closed

    @unreadable_frame
    def test_an_unreadable_reply_names_the_exchange_rather_than_the_codec(
        self, monkeypatch: pytest.MonkeyPatch, frame: str | bytes, cause: type[Exception]
    ) -> None:
        fake = _patch_connect(monkeypatch, [_ready_frame(), frame])
        client = RemotePolicy(endpoint=ENDPOINT)

        with pytest.raises(ConnectionError) as refused:
            client.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, "pick the cube")

        report = str(refused.value)
        assert ENDPOINT in report, report
        assert "reply" in report, report
        assert isinstance(refused.value.__cause__, cause)
        assert repr(frame[:60]) in report, report
        # Same rule as a reply that never arrived: an exchange that did not
        # finish leaves the connection in an unknown position, so it is dropped.
        assert client._ws is None
        assert fake.closed

    def test_the_two_reads_are_told_apart_by_the_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One frame at both doors gives two reports, differing in the read.

        A caller with a handshake and an in-flight request cannot tell which of
        them the peer answered unreadably unless the report says so.
        """
        garbage = "not json at all"
        _patch_connect(monkeypatch, [garbage])
        with pytest.raises(ConnectionError) as at_handshake:
            _ = RemotePolicy(endpoint=ENDPOINT).requires_images

        _patch_connect(monkeypatch, [_ready_frame(), garbage])
        with pytest.raises(ConnectionError) as at_reply:
            RemotePolicy(endpoint=ENDPOINT).get_actions_sync({"observation.state": [0.0]}, "")

        first, second = str(at_handshake.value), str(at_reply.value)
        assert first != second
        assert first.replace("handshake", "reply") == second
