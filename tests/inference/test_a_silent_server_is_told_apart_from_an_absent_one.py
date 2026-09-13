# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A listening server that went quiet is reported as quiet, not as absent.

:class:`~strands_robots.inference.client.RemotePolicy` bounds both of its wire
reads - the ``ready`` handshake with ``connect_timeout``, every reply with
``request_timeout`` - so a peer that accepted the connection and then said
nothing cannot hold the calling thread. Bounding a read only decides *that* the
caller gets control back; it does not decide *what* it is told, and the two
reads were the only wire failures this client did not report. Each sat in a
``try``/``finally`` with no ``except``, so the expiry travelled out as a bare
``TimeoutError`` - which ``websockets`` raises with no arguments at all, so
``str(exc)`` is the empty string. An operator saw a traceback naming neither
the endpoint, nor which of the two reads expired, nor the budget that expired,
nor the parameter that carries it.

The connect side of the same method has carried an actionable report all along
("could not reach a PolicyServer ... Start one first"), and that report is the
wrong one here: this server is listening, so starting one names the only thing
that is not wrong. The remedy for a checkpoint still loading onto the GPU is to
read the server's log or raise the budget, and the budget is the number the
operator cannot guess. The sibling client on this wire,
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient`, already
draws that distinction through the shared report in MODULE
strands_robots.policies._ws_wire, whose ``budget_param`` exists precisely so
each client can name its own knob.

So both reads are pinned here, and the absent server is pinned beside them as
the control: it must keep the start-the-server hint. That pairing is the point
rather than two separate facts, because ``TimeoutError`` is a subclass of
``OSError`` - a clause order that let the absent-server handler see an expiry
first would collapse the two cases back into one report, silently.

No network access: the connection is a fake whose ``recv`` plays a script, the
same stand-in ``test_remote_policy_handshake_contract.py`` uses, so nothing here
waits out a real budget.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.inference import RemotePolicy, protocol

#: Budgets the client is built with. Distinct values so a report that quoted the
#: wrong one names the wrong number rather than coincidentally matching.
CONNECT_TIMEOUT = 0.25
REQUEST_TIMEOUT = 0.5

URI = "ws://127.0.0.1:8765"


class _ScriptedConnection:
    """A ``websockets.sync`` connection whose ``recv`` plays a fixed script.

    Each script entry is either a frame to return or an exception instance to
    raise, so "the server went quiet" is expressed as the ``TimeoutError``
    ``recv(timeout=...)`` really raises when its budget expires.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.sent: list[str] = []
        self.closed = False

    def recv(self, timeout: float | None = None) -> str:  # noqa: ARG002 - real API parity
        if not self._script:
            raise AssertionError("client called recv() more times than the test scripted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def send(self, text: str) -> None:
        self.sent.append(text)

    def close(self) -> None:
        self.closed = True


def _ready_frame() -> str:
    """A well-formed ``ready`` handshake, so the reply read is the one that expires."""
    return protocol.dumps(
        {
            "type": protocol.MSG_READY,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "metadata": {"provider_name": "recording", "requires_images": False},
        }
    )


def _client(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> tuple[RemotePolicy, _ScriptedConnection]:
    """A ``RemotePolicy`` whose next connect yields a connection playing ``script``."""
    fake = _ScriptedConnection(script)
    # ``_connect`` does ``from websockets.sync.client import connect``, so the
    # patch target is the attribute on that module, resolved at call time.
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: fake)
    policy = RemotePolicy(
        endpoint=URI,
        connect_timeout=CONNECT_TIMEOUT,
        request_timeout=REQUEST_TIMEOUT,
    )
    return policy, fake


def _read_the_handshake(policy: RemotePolicy) -> None:
    """Reach the handshake read through a public attribute that connects lazily."""
    _ = policy.requires_images


def _read_a_reply(policy: RemotePolicy) -> None:
    """Reach the reply read: the handshake succeeds, then a request is issued."""
    policy.get_actions_sync({"state": [0.0]}, "")


#: ``(label, script, reach the read, the budget it must quote, the knob it must name)``.
#: One row per bounded read, because each has its own budget and its own knob,
#: and a report that quoted the other would name a number the operator cannot act on.
SILENT_READS = [
    ("handshake", [TimeoutError()], _read_the_handshake, CONNECT_TIMEOUT, "connect_timeout"),
    ("reply", [_ready_frame(), TimeoutError()], _read_a_reply, REQUEST_TIMEOUT, "request_timeout"),
]


@pytest.mark.parametrize(
    ("label", "script", "reach", "budget", "knob"),
    SILENT_READS,
    ids=[row[0] for row in SILENT_READS],
)
def test_a_read_that_expired_names_the_endpoint_budget_and_knob(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    script: list[Any],
    reach: Any,
    budget: float,
    knob: str,
) -> None:
    """A server that accepted the connection and went quiet is reported as such."""
    policy, fake = _client(monkeypatch, script)

    with pytest.raises(ConnectionError) as caught:
        reach(policy)

    message = str(caught.value)
    assert URI in message, f"the {label} report does not name the endpoint: {message!r}"
    assert f"{knob}={budget:g}s" in message, f"the {label} report does not quote {knob}: {message!r}"
    assert "accepted the connection" in message, f"the {label} report does not say the server is listening: {message!r}"
    # The absent-server remedy is the one thing that is not wrong here.
    assert "Start one first" not in message, (
        f"the {label} report tells the operator to start a listening server: {message!r}"
    )
    # A bare TimeoutError is what this replaced, and it carries no message at all.
    assert isinstance(caught.value.__cause__, TimeoutError)
    # The reply that never arrived is still queued, so the connection must go.
    assert fake.closed, f"the {label} timeout left the connection cached"


def test_an_absent_server_still_says_to_start_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: a refused connect keeps the start-the-server hint.

    ``TimeoutError`` is an ``OSError``, so this is the report a clause order
    that saw an expiry first would hand back for a *listening* server too.
    """
    assert issubclass(TimeoutError, OSError), "the two reports are only distinct while this holds"

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr("websockets.sync.client.connect", refuse)
    policy = RemotePolicy(endpoint=URI, connect_timeout=CONNECT_TIMEOUT, request_timeout=REQUEST_TIMEOUT)

    with pytest.raises(ConnectionError) as caught:
        _read_the_handshake(policy)

    message = str(caught.value)
    assert "Start one first" in message, f"an absent server lost its actionable hint: {message!r}"
    assert "accepted the connection" not in message, f"an absent server is reported as listening: {message!r}"
    assert "connect_timeout=" not in message, f"an absent server quotes a budget that did not expire: {message!r}"
