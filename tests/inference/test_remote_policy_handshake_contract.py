"""Client-side fail-fast contract tests for the remote-policy handshake + reply.

The end-to-end round-trip tests in ``test_remote_policy_roundtrip.py`` drive a
real, well-behaved :class:`~strands_robots.inference.server.PolicyServer`, so the
client's defensive branches for a *mis*behaving peer are never exercised there.
These tests stub the WebSocket transport with a fake connection that returns
crafted frames, pinning the three "loud failure" seams a real server can never
produce:

* a handshake frame whose ``type`` is not ``ready``;
* a ``ready`` handshake advertising a mismatched ``protocol_version``;
* an ``actions`` reply whose ``actions`` payload is not a list.

Per the module contract (loud error propagation, never a silent zero action),
each seam must raise rather than proceed with a degraded connection or chunk.
"""

from typing import Any

import pytest

from strands_robots.inference import RemotePolicy, protocol


class _FakeConnection:
    """Minimal stand-in for a ``websockets.sync`` connection.

    ``recv`` pops from a pre-seeded queue of already-serialized frames; ``send``
    records outbound frames so a test can assert the client got as far as
    issuing a request before the reply tripped a contract.
    """

    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []
        self.closed = False

    def recv(self, timeout: float | None = None) -> str:  # noqa: ARG002 - parity with real API
        if not self._frames:
            raise AssertionError("client called recv() more times than the test seeded frames")
        return self._frames.pop(0)

    def send(self, text: str) -> None:
        self.sent.append(text)

    def close(self) -> None:
        self.closed = True


def _patch_connect(monkeypatch: pytest.MonkeyPatch, frames: list[str]) -> _FakeConnection:
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
