# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One report for a policy server that answered the connect and nothing else.

The WebSocket policy client in this package -
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` -
carries an actionable "could not reach the server" hint, raised from
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

import re
from importlib import metadata
from typing import Any


def silent_server_error(*, server: str, uri: str, what: str, timeout: float, budget_param: str) -> str:
    """Return the report for a peer that accepted the connection and went quiet.

    Args:
        server: Human name of the service being dialled (e.g. ``"Cosmos 3
            policy server"``), used to name what is silent rather than what is absent.
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


def require_held_connect(owner: str, extra: str) -> None:
    """Refuse a ``websockets`` that cannot hand back a connection held across calls.

    Both WebSocket policy clients hold one connection for a whole rollout, which
    ``websockets.sync.client.connect(..., legacy=True)`` is the supported way to
    obtain from 17.1 on. ``websockets`` is not a base dependency, so a base
    install can resolve an older release through another package, and there the
    flag is forwarded to ``socket.create_connection`` and comes back as
    ``TypeError: ... unexpected keyword argument 'legacy'`` on the first rollout
    step, naming neither the package nor the fix. Asked here, while the client
    is being built, the answer is the missing-dependency report every optional
    provider gives. The installed release is read rather than ``connect``'s
    signature: before 17.1 ``connect`` takes ``**kwargs`` too, which is how the
    flag reached the socket.

    Args:
        owner: The client class, quoted in the report.
        extra: The ``strands-robots`` extra that floors ``websockets`` for it.

    Raises:
        ImportError: When ``websockets`` is not installed, or is older than
            17.1. The message names the installed release and the extra.
    """
    remedy = f"pip install 'strands-robots[{extra}]'"
    try:
        installed = metadata.version("websockets")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"{owner} needs the websockets package, which is not installed: {remedy}", name="websockets"
        ) from exc
    release = tuple(int(part) for part in re.findall(r"\d+", installed)[:2])
    if release < (17, 1):
        raise ImportError(
            f"{owner} needs websockets>=17.1 to hold its connection (connect(legacy=True)); "
            f"websockets {installed} is installed: {remedy}",
            name="websockets",
        )
