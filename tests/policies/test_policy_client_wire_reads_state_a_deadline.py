# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A policy client's wire read is bounded, and a missed read is not a wrong one.

The WebSocket policy client in this package -
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` - reads
its server's metadata handshake and every action chunk with
``websockets.sync``'s ``recv()``, which has no deadline of its own. A
``connect``-level ``open_timeout`` covers the TCP connect plus the HTTP upgrade
only: a server whose listener accepted the connection and then went quiet - a
checkpoint still loading onto the GPU, a wedged forward pass - held the calling
thread with no way back to the caller.

A documented contract fails on that. The client converts ``OSError`` around its
connect into an actionable ``ConnectionError`` ("could not reach the server -
start it first"), and that report is unreachable for a listening server, because
no ``TimeoutError`` (an ``OSError``) is ever raised. A server runner that judges
readiness with a TCP port probe returns as soon as the listener is up - which is
exactly the state that hangs the first read.

Bounding the read alone would trade the hang for a *wrong* answer, so both
halves are pinned here. A reply that was not read is still produced and still
queued on the socket, so the next request reads the previous request's chunk -
well-formed, and computed for an observation the robot has already moved past.
``RemotePolicy`` states that rule for the same wire (see
``tests/inference/test_a_failed_exchange_does_not_leave_the_connection_cached.py``);
this client now states it too, and the handshake gets it as well - assigned
before the metadata frame was read, a failed handshake left a live connection
cached behind the refusal it had just raised, so the next request would have read
that unconsumed metadata blob as its action chunk.

No network access and no GPU: every server here is a loopback listener, and the
one read that has to miss its reply is parked by the server rather than raced.

Every listener and every client a test opens is closed when that test ends, and
the ``loopback`` fixture refuses a thread that outlives its test. A connection
left open here is not idle: ``websockets.sync`` runs a keepalive thread per
connection that draws ``random.getrandbits(32)`` for each ping it sends, on the
*process-global* ``random`` - so a client this module kept alive went on
perturbing the stream of every later test on the same xdist worker, and
``tests/policies/test_rng_parity.py`` read two ``reset(seed=4242)`` windows as
two different streams (#4023).
"""

from __future__ import annotations

import ast
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("websockets", reason="the raw websocket transports need websockets")

from websockets.sync.server import serve  # noqa: E402

from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp  # noqa: E402
from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient  # noqa: E402

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


class _Loopback:
    """The listeners and clients one test opens, closed at that test's boundary.

    ``released`` is what a handler that has to park waits on, so a server that
    would otherwise sit in ``time.sleep`` for longer than the test is let go the
    moment the test is over rather than when its sleep happens to end.
    """

    def __init__(self) -> None:
        self.released = threading.Event()
        self._servers: list[Any] = []
        self._threads: list[threading.Thread] = []
        self._clients: list[Any] = []

    def serve(self, handler: Any) -> int:
        """Run *handler* on a loopback WebSocket listener; return its port."""
        server = serve(handler, "127.0.0.1", 0)
        port = int(server.socket.getsockname()[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._servers.append(server)
        self._threads.append(thread)
        return port

    def client(self, client_cls: Any, **kwargs: Any) -> Any:
        """Construct a loopback client that is closed with the test."""
        client = client_cls(host="127.0.0.1", **kwargs)
        self._clients.append(client)
        return client

    def close(self) -> None:
        """Release the parked handlers, close every client, then every server.

        Clients first: a server's handler is in ``recv()`` on the connection its
        client holds, and returns once that side closes, so the server's own
        ``shutdown()`` - which waits for its handler threads - has nothing left
        to wait for.
        """
        self.released.set()
        for client in self._clients:
            client.close()
        for server in self._servers:
            server.shutdown()
        for thread in self._threads:
            thread.join(JOIN_S)


def _threads_started_since(before: set[int | None]) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.ident not in before and thread.is_alive()]


@pytest.fixture
def loopback() -> Iterator[_Loopback]:
    """Every listener and client the test opens, closed when the test ends.

    The teardown also refuses a thread that outlives the test. A connection this
    module leaves open is a ``websockets.sync`` keepalive thread drawing from the
    process-global ``random`` every ``ping_interval`` for the rest of the worker's
    life (#4023), and nothing else here can see that: the leaked threads are the
    library's, so no handle this module holds names them. The census below is
    the one read that does. Bounded by ``JOIN_S`` because a connection's threads
    end a moment after its socket closes rather than in the same instant.
    """
    before = {thread.ident for thread in threading.enumerate()}
    connections = _Loopback()
    yield connections
    connections.close()
    deadline = time.monotonic() + JOIN_S
    while _threads_started_since(before) and time.monotonic() < deadline:
        time.sleep(0.05)
    leftover = _threads_started_since(before)
    assert not leftover, "threads this test started are still running after its teardown:\n" + "\n".join(
        f"  {thread.name}: target={getattr(thread, '_target', None)!r}" for thread in leftover
    )


@pytest.fixture
def silent_server(loopback: _Loopback) -> int:
    """A server that accepts the connection and then sends nothing at all.

    The listening-but-silent state: a port probe that calls readiness answers
    yes, and the metadata frame never comes.
    """
    return loopback.serve(lambda conn: loopback.released.wait(JOIN_S * 4))


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
    pytest.param(Cosmos3WebsocketClient, "get_server_metadata", "Start it first", id="cosmos3"),
]


@pytest.mark.parametrize(("client_cls", "call_name", "absent_hint"), CLIENTS)
class TestASilentServerIsReportedRatherThanWaitedOut:
    """The read that a listening-but-silent server never answers has a deadline."""

    def test_the_read_returns_inside_the_stated_budget(
        self, client_cls: Any, call_name: str, absent_hint: str, loopback: _Loopback, silent_server: int
    ) -> None:
        client = loopback.client(client_cls, port=silent_server, read_timeout=READ_TIMEOUT_S)
        thread, raised = _call_on_a_thread(getattr(client, call_name))
        assert not thread.is_alive(), (
            f"{client_cls.__name__} is still blocked {JOIN_S}s into a read the server will never "
            f"answer: recv() has no deadline of its own, and open_timeout covers the connect only"
        )
        assert raised and isinstance(raised[0], ConnectionError), f"expected a ConnectionError, got {raised}"

    def test_the_report_names_the_silent_server_and_the_budget_that_expired(
        self, client_cls: Any, call_name: str, absent_hint: str, loopback: _Loopback, silent_server: int
    ) -> None:
        client = loopback.client(client_cls, port=silent_server, read_timeout=READ_TIMEOUT_S)
        _thread, raised = _call_on_a_thread(getattr(client, call_name))
        message = str(raised[0])
        assert f"ws://127.0.0.1:{silent_server}" in message, message
        assert "accepted the connection" in message, message
        assert f"read_timeout={READ_TIMEOUT_S:g}s" in message, message

    def test_the_absent_server_hint_is_not_the_report_a_listening_server_gets(
        self, client_cls: Any, call_name: str, absent_hint: str, loopback: _Loopback, silent_server: int
    ) -> None:
        """Telling an operator to start a running server names the one thing that is right."""
        client = loopback.client(client_cls, port=silent_server, read_timeout=READ_TIMEOUT_S)
        _thread, raised = _call_on_a_thread(getattr(client, call_name))
        assert absent_hint not in str(raised[0]), (
            f"a server that accepted the connection is running, so the 'start it first' hint misreports it: {raised[0]}"
        )


class TestAFailedExchangeDoesNotLeaveTheConnectionCached:
    """A connection whose exchange did not complete is discarded, not reused."""

    @staticmethod
    def _parking_server(loopback: _Loopback) -> int:
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

        return loopback.serve(handler)

    def test_the_next_request_gets_its_own_chunk_not_the_previous_one(self, loopback: _Loopback) -> None:
        client = loopback.client(
            Cosmos3WebsocketClient, port=self._parking_server(loopback), read_timeout=READ_TIMEOUT_S
        )
        with pytest.raises(ConnectionError):
            client.infer({"marker": 1})
        time.sleep(PARK_S + 0.5)  # the parked reply has now landed on the socket
        assert client.infer({"marker": 2})["marker"] == 2, (
            "the second request was answered with the first request's chunk - a well-formed "
            "action chunk computed for an observation the robot has already moved past"
        )

    def test_a_handshake_that_did_not_complete_is_not_answered_with_empty_metadata(self, loopback: _Loopback) -> None:
        """A dead connection must not report a server contract nobody sent.

                The refusal's own *type* is the transport's to choose - a server that
                closes mid-handshake raises ``ConnectionClosed``, which is not what this
                rule is about. What it is about is that both attempts refuse: the second
                one answering at all means the connection was cached behind the first
                refusal, and ``{}`` then stood in for a server contract nobody sent -
        an unconsumed metadata frame that the next request would have read as
                its action chunk.
        """
        port = loopback.serve(lambda conn: conn.close())
        client = loopback.client(Cosmos3WebsocketClient, port=port, read_timeout=READ_TIMEOUT_S)
        for attempt in (1, 2):
            try:
                metadata = client.get_server_metadata()
            except Exception:  # noqa: BLE001 - any refusal is a refusal
                continue
            pytest.fail(f"call {attempt} to a server that closed mid-handshake answered with {metadata!r}")

    def test_a_completed_exchange_keeps_its_connection(self, loopback: _Loopback) -> None:
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

        client = loopback.client(Cosmos3WebsocketClient, port=loopback.serve(handler), read_timeout=JOIN_S)
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
    """No read in the client module is left on ``recv()``'s absent default."""

    #: The client modules this rule covers, relative to the package root.
    MODULES = ("policies/cosmos3/client.py",)

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
        assert found >= 2, f"the scan found only {found} recv() calls; the client module has moved"
        assert not unbounded, (
            "websockets' recv() has no default deadline, so each of these blocks indefinitely on a "
            "server that accepted the connection and then went quiet:\n" + "\n".join(unbounded)
        )
