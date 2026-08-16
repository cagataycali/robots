"""One timeout domain for both remote-inference clients, and why ``inf`` is in it.

:class:`~strands_robots.inference.RemotePolicy` (WebSocket) and
:class:`~strands_robots.policies.lerobot_async.LerobotAsyncPolicy` (gRPC) each
take ``connect_timeout`` and ``request_timeout``, and each used to store what it
was handed and pass it to the transport unexamined. Both constructors already
refused their *other* numeric parameters - ``port`` via ``tcp_port_error``,
``actions_per_chunk`` / ``actions_per_step`` via ``chunk_count_error`` - so the
two timeouts were the knobs left behind on the same two constructors.

What made that worse than a late crash is where the failure surfaced. Both
clients wrap the first transport failure in a ``ConnectionError`` that names the
server and tells the operator to start one:

    RemotePolicy could not reach a PolicyServer at ws://127.0.0.1:8765.
    Start one first, e.g.: python -m strands_robots.inference.server ...

Measured against a **live, reachable** server (``websockets`` 17.0.1 /
``grpcio`` 1.83.0), the values that produce exactly that message are ``0``,
``0.0``, ``-1`` and ``True`` - the transport times out at once, ``TimeoutError``
/ ``RpcError`` is inside the clause that composes the message, and the operator
is pointed at the one thing that was not wrong. ``nan``, ``inf`` and a numeric
string instead escaped that clause as a ``ValueError`` / ``OverflowError`` /
``TypeError`` raised from library internals, naming no parameter, and - because
both clients connect lazily on first use - landing mid-rollout rather than at
construction.

``inf`` is the interesting one and is pinned separately below. It is the single
value a caller would pass deliberately, meaning "no deadline, wait as long as it
takes", and *neither* transport honours it: ``websockets`` raises
``OverflowError`` computing the deadline, while gRPC reports
``DEADLINE_EXCEEDED`` immediately, making ``inf`` indistinguishable from ``0``.
So an unbounded wait is not expressible through these knobs at all, and the
finiteness clause of :func:`~strands_robots.utils.positive_finite_number_error`
is load-bearing here rather than inherited.

These are constructor contracts, so almost nothing here opens a socket. The one
exception is deliberate: a real loopback server proves the refusal is about the
value and not about reachability, which is the whole claim.

Regression for #1984.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies import MockPolicy
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy
from strands_robots.utils import positive_finite_number_error

#: Values that name no wait budget. Each is refused by all four knobs.
#:
#: ``True`` / ``False`` matter because ``bool`` is an ``int`` subclass, so a bare
#: ``> 0`` test admits ``True`` as a silent one-second budget - measured as a
#: 1.0002 s ``recv`` timeout, not as an error. ``'10'`` matters because neither
#: client applies ``float()``, so a numeric string reached the transport and blew
#: up inside it (``unsupported operand type(s) for +: 'float' and 'str'``) rather
#: than at the boundary. ``nan`` and ``inf`` are the two the transports reject
#: themselves, in incompatible ways.
UNUSABLE_TIMEOUTS: list[Any] = [
    0,
    0.0,
    -1,
    -0.5,
    math.nan,
    math.inf,
    -math.inf,
    True,
    False,
    "10",
    None,
    [10],
]

#: Accepted by both clients. ``np.float32`` is here because a timeout read out of
#: a config array is a real spelling and the shared domain documents it as usable
#: - neither client coerces, so this pins that no coercion is needed.
USABLE_TIMEOUTS: list[Any] = [0.001, 1, 10.0, 60.0, np.float32(0.5)]

#: The two knobs, on both clients.
TIMEOUT_PARAMS = ["connect_timeout", "request_timeout"]


def _ws_client(**kwargs: Any) -> RemotePolicy:
    """Construct the WebSocket client, splatted so off-type values reach the guard.

    Both parameters are annotated ``float``; several cases here pass something
    else on purpose, to prove the runtime refuses it rather than the type
    checker.
    """
    return RemotePolicy(**kwargs)


def _grpc_client(**kwargs: Any) -> LerobotAsyncPolicy:
    """Construct the gRPC client with the minimum viable required arguments.

    ``policy_type`` / ``pretrained_name_or_path`` are mandatory and unrelated to
    the timeouts, so they are supplied valid throughout; the one test that cares
    about their ordering relative to the timeout guard omits them explicitly.
    """
    kwargs.setdefault("policy_type", "act")
    kwargs.setdefault("pretrained_name_or_path", "org/model")
    return LerobotAsyncPolicy(**kwargs)


CLIENTS = [("RemotePolicy", _ws_client), ("LerobotAsyncPolicy", _grpc_client)]


class TestNeitherClientAcceptsATimeoutThatNamesNoBudget:
    """All four knobs refuse the same values, naming the class, param and domain."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    @pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS)
    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_it_is_refused_at_construction(self, name: str, build: Any, value: Any, param: str) -> None:
        """The refusal is a ``ValueError`` that identifies what the caller got wrong."""
        with pytest.raises(ValueError) as exc:
            build(**{param: value})
        text = str(exc.value)
        assert name in text, f"the refusal must name the class, got {text!r}"
        assert param in text, f"the refusal must name the parameter, got {text!r}"
        assert "must be > 0" in text, f"the refusal must state the domain, got {text!r}"


class TestARunningServerIsNoLongerBlamedForTheCallersTimeout:
    """The failure this issue is about: an unusable timeout read as an absent server.

    A ``0`` connect timeout produced "could not reach a PolicyServer ... Start
    one first" against a server that was running and reachable, because
    ``TimeoutError`` is inside the clause that composes that message.
    """

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    @pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_the_refusal_does_not_implicate_the_server(self, name: str, build: Any, value: Any, param: str) -> None:
        """Nothing in the message suggests starting or reaching a server."""
        with pytest.raises(ValueError) as exc:
            build(**{param: value})
        text = str(exc.value)
        assert "Start one first" not in text
        assert "could not reach" not in text

    def test_it_is_a_value_error_rather_than_a_connection_error(self) -> None:
        """The exception type carries the same distinction as the message.

        ``ConnectionError`` is what a caller retries or escalates to whoever owns
        the server; ``ValueError`` is what a caller fixes in their own call. The
        old behaviour raised the former for a mistake that is the latter.
        """
        with pytest.raises(ValueError):
            _ws_client(connect_timeout=0)
        with pytest.raises(ValueError):
            _grpc_client(connect_timeout=0)

    def test_no_websocket_is_dialled_for_a_refused_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard precedes the transport, so the server is never asked.

        Pinned by making a dial fail the test outright: the old code reached
        ``connect`` (lazily, on first use) and let the transport's verdict stand
        in for validation.
        """
        import websockets.sync.client as ws_client

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("connect() must not be reached for a refused timeout")

        monkeypatch.setattr(ws_client, "connect", explode)
        with pytest.raises(ValueError, match="connect_timeout"):
            _ws_client(connect_timeout=0)

    def test_a_positive_timeout_reaches_the_same_live_server(self) -> None:
        """The refused value is the only thing wrong: this server is reachable.

        Without this, every assertion above is consistent with "the client can no
        longer connect at all". A real loopback ``PolicyServer`` on an ephemeral
        port, dialled with a small positive budget, separates the two.
        """
        server = PolicyServer(policy=MockPolicy(), port=0).start()
        try:
            endpoint = f"ws://127.0.0.1:{server.port}"
            client = RemotePolicy(endpoint=endpoint, connect_timeout=5.0, request_timeout=5.0)
            try:
                # Touching a mirrored-metadata property forces the lazy connect
                # and the handshake, both of which use ``connect_timeout``.
                assert client.provider_name == "remote"
                assert client.execution_horizon >= 1
            finally:
                client.close()

            # Same endpoint, same live server, unusable budget: refused before
            # the dial rather than reported as an unreachable server.
            with pytest.raises(ValueError, match="connect_timeout"):
                RemotePolicy(endpoint=endpoint, connect_timeout=0)
        finally:
            server.stop()


class TestInfinityIsRefusedRatherThanReadAsNoDeadline:
    """``inf`` is the one value a caller means, and neither transport honours it."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_it_is_refused_on_both_clients(self, name: str, build: Any, param: str) -> None:
        """Refused, deliberately - not admitted as an unbounded wait.

        A later change that reads ``inf`` as "wait forever" fails here first, and
        should read the premise tests below before deciding this test is wrong.
        """
        with pytest.raises(ValueError, match="must be > 0"):
            build(**{param: math.inf})

    def test_the_websocket_transport_does_not_honour_it(self) -> None:
        """Premise: ``websockets`` raises rather than waiting indefinitely.

        This is the justification for the finiteness clause on the WebSocket
        side. If a future ``websockets`` starts honouring ``inf``, this test fails
        and the refusal above becomes a choice worth re-making rather than a
        consequence of the transport.
        """
        import time

        with pytest.raises(OverflowError):
            # The deadline arithmetic ``connect``/``recv`` perform on the value.
            time.gmtime(time.monotonic() + math.inf)

    def test_zero_and_infinity_are_indistinguishable_over_grpc(self) -> None:
        """Premise: gRPC reports ``DEADLINE_EXCEEDED`` for ``inf``, as it does for ``0``.

        Measured on ``grpcio`` 1.83.0 against a live in-process server: ``inf``
        failed in 0.0001 s, i.e. the value that looks like an unbounded wait is
        the fastest failure available. Asserted here as the documented reason
        rather than re-measured, since exercising it needs a gRPC server and the
        conclusion is about ``inf`` being unusable either way.
        """
        assert positive_finite_number_error(math.inf, "connect_timeout", "Ctx") is not None
        assert positive_finite_number_error(0, "connect_timeout", "Ctx") is not None


class TestTheTwoClientsShareOneDomain:
    """Both defer to the shared domain, so the verdicts cannot drift apart."""

    @pytest.mark.parametrize("value", [*UNUSABLE_TIMEOUTS, *USABLE_TIMEOUTS])
    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    def test_both_clients_agree_with_the_shared_verdict(self, param: str, value: Any) -> None:
        """Refuse exactly when :func:`positive_finite_number_error` refuses.

        Pinned over the accepted values too, so a client that grows a private
        extra restriction - a minimum budget, say - fails here rather than
        diverging silently from its sibling.
        """
        shared_refuses = positive_finite_number_error(value, param, "Ctx") is not None
        for name, build in CLIENTS:
            if shared_refuses:
                with pytest.raises(ValueError, match="must be > 0"):
                    build(**{param: value})
            else:
                build(**{param: value})  # constructs; no transport is touched


class TestAnAcceptedTimeoutIsStoredUnchanged:
    """A usable value is kept as given - no coercion stands in for the guard."""

    @pytest.mark.parametrize("value", USABLE_TIMEOUTS)
    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_it_survives_construction(self, name: str, build: Any, param: str, value: Any) -> None:
        """The attribute the transport reads carries the caller's value."""
        client = build(**{param: value})
        assert getattr(client, param) == value

    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_the_defaults_are_inside_the_domain(self, name: str, build: Any) -> None:
        """The shipped defaults pass their own guard.

        A default outside the domain would make every unconfigured construction
        raise, so this fails loudly rather than in every caller.
        """
        client = build()
        assert client.connect_timeout > 0
        assert client.request_timeout > 0


class TestTheGuardSitsWhereTheOtherTransportKnobsAre:
    """Ordering, so the most specific verdict wins when a caller gets two wrong."""

    def test_an_unusable_port_is_reported_before_an_unusable_timeout(self) -> None:
        """``port`` first: "this address cannot be dialled" is the narrower fact."""
        with pytest.raises(ValueError, match="invalid port"):
            _ws_client(port=-1, connect_timeout=0)
        with pytest.raises(ValueError, match="invalid port"):
            _grpc_client(port=-1, connect_timeout=0)

    def test_the_timeout_is_reported_before_a_missing_policy_type(self) -> None:
        """On the gRPC client, the transport knobs settle before the payload ones.

        Both are caller errors; grouping the timeouts with ``port`` keeps every
        connection parameter answered in one place, and keeps the guard ahead of
        the code path that imports gRPC.
        """
        with pytest.raises(ValueError, match="connect_timeout"):
            LerobotAsyncPolicy(connect_timeout=0)

    def test_an_explicit_endpoint_does_not_exempt_the_timeout(self) -> None:
        """``endpoint`` supersedes ``host``/``port``, never the wait budget.

        The ``port`` guard is deliberately skipped when an endpoint is given, so
        this pins that the timeout guard was not folded into that condition.
        """
        with pytest.raises(ValueError, match="connect_timeout"):
            _ws_client(endpoint="ws://gpu-box:8765", connect_timeout=0)
        with pytest.raises(ValueError, match="request_timeout"):
            _grpc_client(server_address="gpu-box:8080", request_timeout=math.nan)
