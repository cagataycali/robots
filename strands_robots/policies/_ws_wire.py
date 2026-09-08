# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One report for a policy server that answered the connect and nothing else.

The two WebSocket policy clients in this package -
:class:`~strands_robots.policies.vera.client.VeraWebsocketClient` and
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` - each
carry an actionable "could not reach the server" hint, raised from
``except OSError`` around the connect. That report is written for a server that
is *absent*, and it is the wrong report for a server that is *present and
silent*: one whose listener accepted the connection while the checkpoint is
still loading onto the GPU, or whose forward pass is wedged.

The two cases are told apart by which side the wait is on, so they get separate
words. Telling an operator to start a server that is already running points them
at the one thing that is not wrong, and the remedy for a slow load ("wait, or
raise the budget") is not the remedy for an absent process.

``recv`` states the budget it waited out, because that budget is the knob: a WAN
world model can take minutes to answer, so a read that expired is as likely to
be a budget set too low as a server in trouble, and the operator cannot tell
which without the number.

Which case a caller is in is decided by ``except TimeoutError`` *before*
``except OSError``, at the entry point that made the call - never by re-raising
``ConnectionError`` to protect an already-specific report, because
``ConnectionRefusedError`` is a ``ConnectionError`` too and that clause would
hand a bare ``[Errno 111] Connection refused`` straight to the caller, which is
the one report the start-the-server hint exists to replace.
"""

from typing import Any


def silent_server_error(*, server: str, uri: str, what: str, timeout: float, budget_param: str) -> str:
    """Return the report for a peer that accepted the connection and went quiet.

    Args:
        server: Human name of the service being dialled (e.g. ``"VERA policy
            server"``), used to name what is silent rather than what is absent.
        uri: The WebSocket URI the client dialled, so the message names the
            endpoint actually in use rather than the one the caller meant.
        what: The reply that did not arrive (e.g. ``"metadata handshake"``,
            ``"'infer' reply"``).
        timeout: The budget, in seconds, that expired waiting for it.
        budget_param: Name of the constructor parameter that carries *timeout*,
            so the remedy names the knob a caller can actually reach.

    Returns:
        A message stating that the connection was accepted, what did not arrive,
        the budget that expired, and both remedies (read the server's log, or
        raise the budget).
    """
    return (
        f"{server} at {uri} accepted the connection but sent no {what} within "
        f"{budget_param}={timeout:g}s. The server is listening, so it is still loading "
        f"(a large checkpoint takes minutes) or it is wedged: read its log to tell those "
        f"apart, and raise {budget_param} if the model is simply slower than the budget."
    )


def close_quietly(ws: Any) -> None:
    """Close a websocket connection, ignoring any failure.

    Shared by the discard paths and each client's ``close``. A connection being
    discarded is already being abandoned over an error, and letting the close
    raise would replace the report the caller needs with the failure of the
    cleanup.
    """
    try:
        ws.close()
    except Exception:  # noqa: BLE001 - a discarded connection is already lost
        pass
