# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A peer that is listening but does not speak WebSocket is reported as a wrong peer.

:class:`~strands_robots.inference.client.RemotePolicy` already gives two
reports for a connect that fails: a refused dial is "could not reach a
PolicyServer ... Start one first", and a frame the codec cannot read after the
upgrade is the wrong-peer report from ``_parse``. Between them sat the peer
that accepts the TCP connection and answers the WebSocket upgrade with
something else - an HTTP server, a ZMQ sidecar, a socket that closes - which
``websockets`` raises as an ``InvalidHandshake`` (or a ``ConnectionClosed``
when the peer hangs up mid-upgrade). Neither is an ``OSError``, so both
escaped ``_connect`` verbatim: ``InvalidMessage: did not receive a valid HTTP
response``, naming neither the endpoint nor what to check, from a method
documented to raise ``ConnectionError``.

Pinned as the wrong-peer report, and pinned apart from the absent server: this
peer is listening, so telling the operator to start one names the only thing
that is not wrong.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

import pytest
from websockets.exceptions import InvalidHandshake, InvalidMessage

from strands_robots.inference import RemotePolicy

URI = "ws://127.0.0.1:8765"


def test_a_peer_that_fails_the_websocket_upgrade_is_a_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handshake failure ``websockets`` raises surfaces as the documented ConnectionError."""

    def answer_in_another_wire_format(*_args: Any, **_kwargs: Any) -> None:
        raise InvalidMessage("did not receive a valid HTTP response")

    monkeypatch.setattr("websockets.sync.client.connect", answer_in_another_wire_format)
    policy = RemotePolicy(endpoint=URI, connect_timeout=0.25)

    with pytest.raises(ConnectionError) as caught:
        _ = policy.requires_images

    message = str(caught.value)
    assert URI in message, f"the report does not name the endpoint: {message!r}"
    assert "did not complete the WebSocket handshake" in message, message
    assert "Start one first" not in message, f"a listening peer is told to start a server: {message!r}"
    assert isinstance(caught.value.__cause__, InvalidHandshake)
    assert policy._ws is None, "a failed upgrade left a connection cached"


def test_a_loopback_socket_that_closes_on_accept_is_reported_the_same_way() -> None:
    """The real thing on 127.0.0.1: a listener that is not a WebSocket server."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def accept_and_close() -> None:
        conn, _ = listener.accept()
        conn.close()

    thread = threading.Thread(target=accept_and_close, daemon=True)
    thread.start()
    try:
        policy = RemotePolicy(endpoint=f"ws://127.0.0.1:{port}", connect_timeout=2.0)
        with pytest.raises(ConnectionError) as caught:
            _ = policy.requires_images
    finally:
        thread.join(timeout=5.0)
        listener.close()

    message = str(caught.value)
    assert f"ws://127.0.0.1:{port}" in message, message
    assert "did not complete the WebSocket handshake" in message, message
    assert "Start one first" not in message, message
