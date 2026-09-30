"""A PolicyServer that ends an open connection is reported by endpoint, not by websockets' close text.

Stopping a server closes each connection it holds with code 1001. The client's
next exchange used to surface websockets' own ``received 1001 (going away);
then sent 1001 (going away)``, which is what ``run_policy`` put in front of the
caller for a server stopped mid-rollout. That names no endpoint and tells the
caller nothing about what happened or whether to retry. The report now names
the URI, the reply that did not arrive and the close code, and the connection
is discarded so the next call dials again.
"""

from __future__ import annotations

import pytest

from strands_robots.inference import PolicyServer
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies._ws_wire import closed_connection_error
from strands_robots.policies.mock import MockPolicy


def _served_client(port: int) -> RemotePolicy:
    client = RemotePolicy(host="127.0.0.1", port=port, request_timeout=5.0)
    client.set_robot_state_keys(["joint_0"])
    assert client.get_actions_sync({"joint_0": 0.0}, "")
    return client


def test_a_server_stopped_between_requests_is_named_with_its_close_code():
    server = PolicyServer(policy=MockPolicy(), port=0).start()
    client = _served_client(server.port)
    server.stop()

    with pytest.raises(ConnectionError) as info:
        client.get_actions_sync({"joint_0": 0.0}, "")

    msg = str(info.value)
    assert f"ws://127.0.0.1:{server.port}" in msg
    assert "closed the connection before sending its reply" in msg
    assert "close code 1001" in msg
    assert "going away" not in msg  # websockets' own wording is not the report
    assert client._ws is None  # discarded, so the next call re-dials
    client.close()


def test_the_next_call_reaches_a_server_restarted_on_the_same_port():
    server = PolicyServer(policy=MockPolicy(), port=0).start()
    port = server.port
    client = _served_client(port)
    server.stop()
    with pytest.raises(ConnectionError):
        client.get_actions_sync({"joint_0": 0.0}, "")

    restarted = PolicyServer(policy=MockPolicy(), port=port).start()
    try:
        assert client.get_actions_sync({"joint_0": 0.0}, "")
    finally:
        client.close()
        restarted.stop()


class _Closed(Exception):
    def __init__(self, rcvd):
        self.rcvd = rcvd


class _Frame:
    def __init__(self, code, reason):
        self.code, self.reason = code, reason


@pytest.mark.parametrize(
    ("rcvd", "expected"),
    [
        (_Frame(1011, "internal error"), "close code 1011 'internal error'"),
        (_Frame(1001, ""), "close code 1001)"),
        (None, "no close frame, the socket dropped"),
    ],
)
def test_the_report_quotes_what_the_peer_sent(rcvd, expected):
    msg = closed_connection_error(server="PolicyServer", uri="ws://h:1", what="reply", exc=_Closed(rcvd))
    assert msg.startswith("PolicyServer at ws://h:1 closed the connection before sending its reply (")
    assert expected in msg
