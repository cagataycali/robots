"""Self-contained WebSocket client for the VERA policy server.

Speaks VERA's websocket wire protocol (``vera/server/protocol/``) directly using
only ``websockets`` + ``msgpack`` + a vendored NumPy packer - **no dependency on
the ``vera`` package** at client time. This mirrors the ``cosmos3`` client: the
heavy model stack lives in the server subprocess; the client is import-light and
composes with any numpy version.

Wire contract (verified against ``vera.server.protocol.websocket_policy_client``
and ``vera.controller.run_mimicgen_eval.RemotePolicy``):

* On connect the server sends one msgpack metadata blob - the
  ``VeraServerConfig`` (``view_keys``, ``context_frames``, ``action_space``,
  ``action_dim``, ``action_horizon``, ``control_dt``, ``gripper_dim_index`` …).
* ``infer`` request = ``{"context_rgb": (T,H,W,3) uint8, "view_keys": [...],
  "view_widths": [...], "session_id": str, "prompt"?: str, "endpoint": "infer"}``.
* ``infer`` response = ``{"action": np.ndarray[H, D], ...}``.
* A *string* response is the error sentinel - raise.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from strands_robots.policies._ws_wire import close_quietly, silent_server_error
from strands_robots.utils import positive_finite_number_error

logger = logging.getLogger(__name__)

# WAN forward passes are slow; keep the connection generous on open.
_OPEN_TIMEOUT_SECS = 600

#: Seconds any single read off the live connection may wait, by default. Held at
#: the open budget because both wait on the same thing - a WAN world model - and
#: ``open_timeout`` covers only the TCP connect plus the HTTP upgrade, so it
#: says nothing about a read. ``websockets`` gives ``recv()`` no deadline of its
#: own, so without this a server that accepted the connection and then went
#: quiet left every read blocked with no way back to the caller.
_READ_TIMEOUT_SECS = 600.0

#: What the reports call the service, so "absent" and "silent" name one server.
_SERVER_NAME = "VERA policy server"


class VeraWebsocketClient:
    """Lazy-connecting WebSocket client for ``vera.server.start_vera_server``.

    The connection (and the server metadata handshake) is established on the
    first :meth:`infer` / :meth:`get_server_metadata` call, so constructing a
    policy never requires the server to already be up.

    Args:
        host: Server hostname or IP.
        port: Server websocket port.
        read_timeout: Seconds any single read off the connection may wait - the
            metadata handshake and every endpoint reply. Only a positive finite
            number names a budget: ``0``, a negative and ``True`` expire against
            a server that is answering normally, and ``inf`` cannot express "no
            deadline" here at all, because ``websockets`` raises
            ``OverflowError`` computing a deadline from it. Refused in the
            constructor, while the caller still holds the value, since the
            connection is opened lazily and an unusable budget would otherwise
            surface mid-rollout naming no parameter.

    Raises:
        ValueError: If *read_timeout* is not a positive finite number.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8800, *, read_timeout: float = _READ_TIMEOUT_SECS) -> None:
        if err := positive_finite_number_error(read_timeout, "read_timeout", type(self).__name__):
            raise ValueError(err)
        self.host = host
        self.port = port
        self.read_timeout = float(read_timeout)
        self.uri = f"ws://{host}:{port}"
        self._ws: Any = None
        self._server_metadata: dict[str, Any] | None = None
        from . import _msgpack_numpy as _mnp  # vendored, numpy-agnostic

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    # -- connection ---------------------------------------------------------

    def _server_hint(self) -> str:
        """Actionable hint for bringing the VERA policy server up."""
        return (
            f"Could not reach the VERA policy server at {self.uri}. "
            "Start it first (holds the GPU):\n"
            "  python -m vera.server.start_vera_server "
            f"--embodiment <pusht|mimicgen|allegro|droid> --port {self.port}\n"
            "Checkpoints download (~42 GB full / ~4 GB Wave-1):\n"
            "  hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts\n"
            "Then set VERA_CKPT_ROOT to that directory."
        )

    def _silent_server_error(self, what: str) -> str:
        """Report for a server that accepted the connection and then went quiet."""
        return silent_server_error(
            server=_SERVER_NAME,
            uri=self.uri,
            what=what,
            timeout=self.read_timeout,
            budget_param="read_timeout",
        )

    def _ensure(self) -> Any:
        if self._ws is not None:
            return self._ws
        try:
            import websockets.sync.client as _wsc

            # ``Any`` for the same reason ``self._ws`` is declared ``Any``: the
            # connection object is handed straight to the vendored packer, whose
            # frames this module treats as opaque.
            ws: Any = _wsc.connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=_OPEN_TIMEOUT_SECS,
            )
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e
        # ``self._ws`` is published only once the handshake has been consumed.
        # Assigned before the read, a failed handshake left a live connection
        # cached behind the refusal it had just raised, with the metadata frame
        # still unread: the short-circuit above then handed that connection back,
        # ``get_server_metadata`` answered ``{}`` for a server it never spoke to,
        # and the next ``infer`` would have read the metadata blob as its action
        # chunk. Same rule as ``RemotePolicy._connect`` (MODULE
        # strands_robots.inference.client) - a connection whose exchange did not
        # complete is in an unknown position, so it is discarded, and the next
        # call opens a fresh one.
        established = False
        try:
            # Server sends its VeraServerConfig as the first message.
            self._server_metadata = self._mnp.unpackb(ws.recv(timeout=self.read_timeout))
            established = True
        except TimeoutError as e:
            raise ConnectionError(self._silent_server_error("metadata handshake")) from e
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e
        finally:
            if not established:
                close_quietly(ws)
        self._ws = ws
        logger.info("VeraWebsocketClient connected to %s", self.uri)
        return ws

    def _exchange(self, message: Mapping[str, Any], endpoint: str) -> Any:
        """Send one request and read its reply, or discard the connection.

        Args:
            message: Wire dict to pack and send; *endpoint* is added to it.
            endpoint: The server endpoint being called, named in the report if
                the reply does not arrive.

        Returns:
            The raw reply frame - ``bytes`` to unpack, or a ``str`` error
            sentinel for the caller to raise.

        Raises:
            ConnectionError: If the server accepted the connection but did not
                answer within ``read_timeout``.

        A reply that was not read is still produced and still queued on the
        socket, so a connection whose exchange did not complete is in an unknown
        position: the next request would read the previous request's answer, and
        a stale ``[H, D]`` chunk is well-formed - it drives the arm from an
        observation it has already moved past, with nothing to distinguish it.
        Discarding the connection is what keeps a missed read from becoming a
        wrong one, and is the rule ``RemotePolicy._request`` states for the same
        wire (MODULE strands_robots.inference.client).
        """
        ws = self._ensure()
        exchanged = False
        try:
            ws.send(self._packer.pack({**message, "endpoint": endpoint}))
            resp = ws.recv(timeout=self.read_timeout)
            exchanged = True
            return resp
        except TimeoutError as e:
            raise ConnectionError(self._silent_server_error(f"{endpoint!r} reply")) from e
        finally:
            if not exchanged:
                self.close()

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the ``VeraServerConfig`` dict the server sends on connect."""
        self._ensure()
        return dict(self._server_metadata or {})

    # -- endpoints ----------------------------------------------------------

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send one inference request; return the server response dict.

        Args:
            observation: Wire dict - must carry ``context_rgb`` (the rolling
                context window), ``view_keys``, ``view_widths`` and
                ``session_id``; ``prompt`` is optional (only when the server
                was launched with text conditioning).

        Returns:
            Response dict containing at least ``"action"`` - an ``[H, D]``
            NumPy array - the chunk the controller plays before refilling.
        """
        resp = self._exchange(observation, "infer")
        if isinstance(resp, str):
            raise RuntimeError(f"VERA inference server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        """Clear the server's per-episode state (history + KV caches)."""
        try:
            resp = self._exchange(reset_info or {}, "reset")
        except ConnectionError:
            # Reset is best-effort - never a correctness requirement (mirrors
            # Cosmos3WebsocketClient.reset / Gr00tPolicy.reset). Covers the
            # server that is absent and the one that accepted the connection
            # without answering: both report ConnectionError, and neither is a
            # reason to fail the episode.
            return
        if isinstance(resp, str) and resp != "reset successful":
            raise RuntimeError(f"VERA reset server error:\n{resp}")

    def configure(self, params: dict[str, Any]) -> dict[str, Any]:
        """Live-tune runtime knobs (motion_plan_scale, sample_steps, guidance)
        without a model rebuild. Returns the server's ``{"applied": {...}}``."""
        resp = self._exchange(params, "configure")
        if isinstance(resp, str):
            raise RuntimeError(f"VERA configure server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def close(self) -> None:
        """Close the underlying websocket if open (best-effort and idempotent)."""
        if self._ws is not None:
            close_quietly(self._ws)
            self._ws = None
