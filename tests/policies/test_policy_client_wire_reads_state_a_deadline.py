# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A policy client's wire read is bounded, and a missed read is not a wrong one.

The two WebSocket policy clients in this package -
:class:`~strands_robots.policies.vera.client.VeraWebsocketClient` and
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` - read
their server's metadata handshake and every action chunk with
``websockets.sync``'s ``recv()``, which has no deadline of its own. ``VERA``
passed ``open_timeout=600`` to ``connect``, and that covers the TCP connect plus
the HTTP upgrade only: a server whose listener accepted the connection and then
went quiet - a checkpoint still loading onto the GPU, a wedged forward pass -
held the calling thread with no way back to the caller.

Two documented contracts fail on that. Each client converts ``OSError`` around
its connect into an actionable ``ConnectionError`` ("could not reach the server -
start it first"), and that report is unreachable for a listening server, because
no ``TimeoutError`` (an ``OSError``) is ever raised. And ``VeraServerRunner``
judges readiness with a TCP port probe, so ``start()`` returns as soon as the
listener is up - which is exactly the state that hangs the first read.

Bounding the read alone would trade the hang for a *wrong* answer, so both
halves are pinned here. A reply that was not read is still produced and still
queued on the socket, so the next request reads the previous request's chunk -
well-formed, and computed for an observation the robot has already moved past.
``RemotePolicy`` states that rule for the same wire (see
``tests/inference/test_a_failed_exchange_does_not_leave_the_connection_cached.py``);
these two clients now state it too, and the handshake gets it as well - assigned
before the metadata frame was read, a failed handshake left a live connection
cached behind the refusal it had just raised, and ``get_server_metadata``
answered ``{}`` for a server it had never spoken to.

No network access and no GPU: every server here is a loopback listener, and the
one read that has to miss its reply is parked by the server rather than raced.
"""

from __future__ import annotations

import ast
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("websockets", reason="the raw websocket transports need websockets")

from websockets.sync.server import serve  # noqa: E402

from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient  # noqa: E402
from strands_robots.policies.vera import _msgpack_numpy as mnp  # noqa: E402
from strands_robots.policies.vera.client import VeraWebsocketClient  # noqa: E402

#: Budget handed to the client for a read that must miss its reply. Waited out in
#: full on every run, so it is the one value here worth keeping small.
READ_TIMEOUT_S = 0.3

#: How long the server withholds the one parked reply. Well above
#: ``READ_TIMEOUT_S`` so the miss is deterministic rather than a race, and the
#: reply provably still exists when the client's budget expires.
PARK_S = 2.0

#: How long to wait for a worker thread that must finish. Generous - a bounded
#: read returns as soon as its budget fires, so this is never waited out on a
#: passing run - but bounded, so an unbounded read renders as a failure instead
#: of hanging the suite.
JOIN_S = 8.0


def _serve(handler: Any) -> int:
    """Run *handler* on a loopback WebSocket listener; return its port."""
    server = serve(handler, "127.0.0.1", 0)
    port = int(server.socket.getsockname()[1])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


@pytest.fixture
def silent_server() -> int:
    """A server that accepts the connection and then sends nothing at all.

    The listening-but-silent state: the port probe that ``VeraServerRunner``
    calls readiness answers yes, and the metadata frame never comes.
    """
    return _serve(lambda conn: time.sleep(JOIN_S * 4))


def _call_on_a_thread(call: Any) -> tuple[threading.Thread, list[BaseException]]:
    """Run *call* on a daemon thread, capturing whatever it raises."""
    raised: list[BaseException] = []

    def attempt() -> None:
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - the outcome is the measurement
            raised.append(exc)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(JOIN_S)
    return thread, raised


#: The two clients, each with the call that performs its metadata handshake and
#: the wording its own "the server is absent" hint uses - which is the report a
#: silent server must NOT receive.
CLIENTS = [
    pytest.param(VeraWebsocketClient, "get_server_metadata", "Could not reach the VERA policy server", id="vera"),
    pytest.param(Cosmos3WebsocketClient, "get_server_metadata", "Start it first", id="cosmos3"),
]


@pytest.mark.parametrize(("client_cls", "call_name", "absent_hint"), CLIENTS)
class TestASilentServerIsReportedRatherThanWaitedOut:
    """The read that a listening-but-silent server never answers has a deadline."""

    def test_the_read_returns_inside_the_stated_budget(
        self, client_cls: Any, call_name: str, absent_hint: str, silent_server: int
    ) -> None:
        client = client_cls(host="127.0.0.1", port=silent_server, read_timeout=READ_TIMEOUT_S)
        thread, raised = _call_on_a_thread(getattr(client, call_name))
        assert not thread.is_alive(), (
            f"{client_cls.__name__} is still blocked {JOIN_S}s into a read the server will never "
            f"answer: recv() has no deadline of its own, and open_timeout covers the connect only"
        )
        assert raised and isinstance(raised[0], ConnectionError), f"expected a ConnectionError, got {raised}"

    def test_the_report_names_the_silent_server_and_the_budget_that_expired(
        self, client_cls: Any, call_name: str, absent_hint: str, silent_server: int
    ) -> None:
        client = client_cls(host="127.0.0.1", port=silent_server, read_timeout=READ_TIMEOUT_S)
        _thread, raised = _call_on_a_thread(getattr(client, call_name))
        message = str(raised[0])
        assert f"ws://127.0.0.1:{silent_server}" in message, message
        assert "accepted the connection" in message, message
        assert f"read_timeout={READ_TIMEOUT_S:g}s" in message, message

    def test_the_absent_server_hint_is_not_the_report_a_listening_server_gets(
        self, client_cls: Any, call_name: str, absent_hint: str, silent_server: int
    ) -> None:
        """Telling an operator to start a running server names the one thing that is right."""
        client = client_cls(host="127.0.0.1", port=silent_server, read_timeout=READ_TIMEOUT_S)
        _thread, raised = _call_on_a_thread(getattr(client, call_name))
        assert absent_hint not in str(raised[0]), (
            f"a server that accepted the connection is running, so the 'start it first' hint misreports it: {raised[0]}"
        )


class TestAFailedExchangeDoesNotLeaveTheConnectionCached:
    """A connection whose exchange did not complete is discarded, not reused."""

    @staticmethod
    def _parking_server() -> int:
        """Serve tagged chunks, withholding the first reply past the read budget.

        The marker each request carries comes back in its own reply, which is
        what makes a stale answer visible at all: a desynchronised stream returns
        a well-formed ``[H, D]`` chunk, so only its content distinguishes it.
        ``served`` is shared across connections so a *fresh* connection is
        answered promptly - the discard is then measurable rather than masked by
        a server that parks every first request.
        """
        packer = mnp.Packer()
        served = {"n": 0}

        def handler(conn: Any) -> None:
            try:
                conn.send(packer.pack({"action_dim": 2, "action_horizon": 1}))
                while True:
                    request = mnp.unpackb(conn.recv())
                    served["n"] += 1
                    if served["n"] == 1:
                        time.sleep(PARK_S)  # the reply is produced late, not never
                    conn.send(packer.pack({"action": np.zeros((1, 2), np.float32), "marker": request["marker"]}))
            except Exception:  # noqa: BLE001 - a discarded connection ends the handler
                return

        return _serve(handler)

    def test_the_next_request_gets_its_own_chunk_not_the_previous_one(self) -> None:
        client = VeraWebsocketClient(host="127.0.0.1", port=self._parking_server(), read_timeout=READ_TIMEOUT_S)
        with pytest.raises(ConnectionError):
            client.infer({"marker": 1})
        time.sleep(PARK_S + 0.5)  # the parked reply has now landed on the socket
        assert client.infer({"marker": 2})["marker"] == 2, (
            "the second request was answered with the first request's chunk - a well-formed "
            "action chunk computed for an observation the robot has already moved past"
        )

    def test_a_handshake_that_did_not_complete_is_not_answered_with_empty_metadata(self) -> None:
        """A dead connection must not report a server contract nobody sent.

        The refusal's own *type* is the transport's to choose - a server that
        closes mid-handshake raises ``ConnectionClosed``, which is not what this
        rule is about. What it is about is that both attempts refuse: the second
        one answering at all means the connection was cached behind the first
        refusal, and ``{}`` then stood in for a server contract nobody sent -
        empty geometry that ``VeraPolicy._ensure_started`` would have accepted as
        the handshake, marking itself started.
        """
        port = _serve(lambda conn: conn.close())
        client = VeraWebsocketClient(host="127.0.0.1", port=port, read_timeout=READ_TIMEOUT_S)
        for attempt in (1, 2):
            try:
                metadata = client.get_server_metadata()
            except Exception:  # noqa: BLE001 - any refusal is a refusal
                continue
            pytest.fail(f"call {attempt} to a server that closed mid-handshake answered with {metadata!r}")

    def test_a_completed_exchange_keeps_its_connection(self) -> None:
        """The discard is not a reconnect-per-request: a good connection is reused."""
        connections: list[int] = []
        packer = mnp.Packer()

        def handler(conn: Any) -> None:
            connections.append(1)
            try:
                conn.send(packer.pack({"action_dim": 2}))
                while True:
                    request = mnp.unpackb(conn.recv())
                    conn.send(packer.pack({"action": np.zeros((1, 2), np.float32), "marker": request["marker"]}))
            except Exception:  # noqa: BLE001
                return

        client = VeraWebsocketClient(host="127.0.0.1", port=_serve(handler), read_timeout=JOIN_S)
        assert client.infer({"marker": 1})["marker"] == 1
        assert client.infer({"marker": 2})["marker"] == 2
        assert len(connections) == 1, f"one connection served both requests, got {len(connections)}"


@pytest.mark.parametrize(("client_cls", "call_name", "absent_hint"), CLIENTS)
class TestTheReadBudgetIsGraded:
    """The budget is refused while the caller still holds it, not mid-rollout."""

    @pytest.mark.parametrize("unusable", [0, -1.0, True, math.nan, math.inf, "600", None])
    def test_a_budget_that_bounds_nothing_is_refused(
        self, client_cls: Any, call_name: str, absent_hint: str, unusable: Any
    ) -> None:
        with pytest.raises(ValueError, match="read_timeout"):
            client_cls(host="127.0.0.1", port=8800, read_timeout=unusable)

    def test_the_default_budget_is_positive_and_finite(self, client_cls: Any, call_name: str, absent_hint: str) -> None:
        default = client_cls(host="127.0.0.1", port=8800).read_timeout
        assert math.isfinite(default) and default > 0, default


class TestEveryReadOffTheseWiresStatesADeadline:
    """No read in either client module is left on ``recv()``'s absent default."""

    #: The client modules this rule covers, relative to the package root.
    MODULES = ("policies/vera/client.py", "policies/cosmos3/client.py")

    def test_no_recv_call_omits_its_timeout(self) -> None:
        package = Path(__file__).resolve().parents[2] / "strands_robots"
        found = 0
        unbounded: list[str] = []
        for relative in self.MODULES:
            path = package / relative
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr != "recv":
                    continue
                found += 1
                if not any(keyword.arg == "timeout" for keyword in node.keywords):
                    unbounded.append(f"{relative}:{node.lineno} {ast.unparse(node)}")
        assert found >= 4, f"the scan found only {found} recv() calls; the client modules have moved"
        assert not unbounded, (
            "websockets' recv() has no default deadline, so each of these blocks indefinitely on a "
            "server that accepted the connection and then went quiet:\n" + "\n".join(unbounded)
        )
