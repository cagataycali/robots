# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The connection a dashboard-auth decision is made about, as it really arrives.

:mod:`strands_robots.dashboard.auth` decides on three facts about a connection
and nothing else: the headers it carries, the peer address of its socket, and the
scheme it was reached over. Two of those are not headers -- the scheme comes off
the ASGI scope through ``request.url``, the peer off ``request.client`` -- so a
stand-in answering a ``headers`` dict alone leaves the scheme reading a default,
and an expectation derived from it looks right in a test while being
caller-controlled in production. A dict also cannot hold one header twice, which
is what a two-hop proxy chain sends.

So the stand-in is the real :class:`~starlette.requests.Request` /
:class:`~starlette.websockets.WebSocket`, built from the scope a server hands
them. Only that construction is shared; every cell still spells the facts it is
about.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from starlette.requests import Request
from starlette.websockets import WebSocket

#: The peer address of a browser on the machine itself.
LOOPBACK = "127.0.0.1"

#: TEST-NET-3 (RFC 5737): a peer that is never this machine and never routable.
STRANGER = "203.0.113.9"

#: The authority a dashboard is reached at by default.
HOST = "localhost:8090"


def _scope(
    kind: str,
    scheme: str,
    host: str | None,
    peer: str | None,
    path: str,
    headers: dict[str, str],
    repeats: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """The ASGI scope of a connection carrying ``headers`` over ``scheme``.

    ``host`` supplies the ``Host`` header unless one is already spelled, in any
    case, so a cell about the wire spelling sends exactly the header it means.
    Names are passed through as given rather than lower-cased: what a decision
    may depend on is the value, never the server's normalising.
    """
    sent = [*headers.items(), *repeats]
    wire = [(name.replace("_", "-").encode(), value.encode()) for name, value in sent]
    if host is not None and not any(name.replace("_", "-").lower() == "host" for name, _ in sent):
        wire.insert(0, (b"host", host.encode()))
    scope: dict[str, Any] = {
        "type": kind,
        "scheme": scheme,
        "path": path,
        "query_string": b"",
        "headers": wire,
        "server": ("localhost", 8090),
        "client": None if peer is None else (peer, 51234),
    }
    if kind == "http":
        scope["method"] = "POST"
    return scope


def connection(
    scheme: str = "http",
    *,
    host: str | None = HOST,
    peer: str | None = LOOPBACK,
    path: str = "/auth/register/begin",
    **headers: str,
) -> Request:
    """A request that really arrived over ``scheme`` from ``peer`` carrying ``headers``.

    ``host=None`` sends no ``Host`` header, which is the request a proxy that
    drops it makes; ``peer=None`` is a connection with no peer address, which is
    what a scope carries for a unix socket or a broken transport. A name that
    cannot be spelled as a keyword is passed as ``**{"cf-ray": "..."}``; a header
    that arrives more than once needs :func:`repeating`, which a mapping of
    headers cannot express.
    """
    return Request(_scope("http", scheme, host, peer, path, headers))


def repeating(
    name: str,
    *values: str,
    scheme: str = "http",
    host: str | None = HOST,
    peer: str | None = LOOPBACK,
) -> Request:
    """A request carrying ``name`` once per value, in the order they arrived.

    The shape two proxies append rather than merge, and the one a mapping of
    headers cannot hold: a second value under one key is the first overwritten.
    """
    return Request(_scope("http", scheme, host, peer, "/auth/register/begin", {}, [(name, v) for v in values]))


async def _never_receives() -> Any:
    raise AssertionError("deciding this must not read from the socket")


async def _never_sends(message: Any) -> None:
    raise AssertionError("deciding this must not write to the socket")


def websocket(
    scheme: str = "ws",
    *,
    host: str | None = HOST,
    peer: str | None = LOOPBACK,
    path: str = "/ws",
    **headers: str,
) -> WebSocket:
    """A websocket that really arrived over ``scheme``, whose channels refuse.

    A decision about the connection reads its scope, and must not depend on
    talking to the peer.
    """
    return WebSocket(_scope("websocket", scheme, host, peer, path, headers), receive=_never_receives, send=_never_sends)
