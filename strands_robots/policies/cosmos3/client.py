"""WebSocket client for the Cosmos 3 RoboLab policy server.

Cosmos Framework ships a ready-made policy server
(``cosmos_framework.scripts.action_policy_server_robolab``) that serves
``nvidia/Cosmos3-Nano-Policy-DROID`` over a msgpack + NumPy WebSocket
protocol. This module ships a self-contained client - **no
``openpi-client`` dependency** - using only ``websockets`` + ``msgpack``
plus a vendored NumPy packer (``_msgpack_numpy``).

Why no ``openpi-client``? It pins ``numpy<2.0``, which is mutually
exclusive with ``lerobot`` (``numpy>=2.0``). Speaking the wire protocol
directly lets a Cosmos 3 rollout run alongside LeRobot dataset recording
in the same venv.

Wire contract (verified against the server source):

* request  = observation dict, keys in the ``observation/...`` namespace
             plus a top-level ``prompt`` string.
* response = ``{"action": np.ndarray[T, D], "video"?: np.ndarray, ...}``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from strands_robots.policies._ws_wire import close_quietly, silent_server_error
from strands_robots.utils import positive_finite_number_error

logger = logging.getLogger(__name__)

#: Seconds any single read off the live connection may wait, by default. A
#: Cosmos 3 forward pass on a video world model is slow, so this is generous -
#: but it is a deadline, and ``websockets`` gives ``recv()`` none of its own:
#: without it a server that accepted the connection and then went quiet (a
#: checkpoint still loading onto the GPU, a wedged forward pass) left every read
#: blocked with no way back to the caller. The connect side is already bounded
#: by ``websockets``' own ``open_timeout`` default.
_READ_TIMEOUT_SECS = 600.0

#: What the reports call the service, so "absent" and "silent" name one server.
_SERVER_NAME = "Cosmos 3 policy server"


class _UnreadableFrame(ConnectionError):
    """A frame off the wire the vendored msgpack+NumPy codec cannot read.

    A ``ConnectionError`` because that is what every other malformation on this
    wire is reported as, and a *distinct* type because
    :class:`Cosmos3WebsocketClient` translates ``OSError`` into the
    start-the-server hint - and a ``ConnectionError`` is an ``OSError``, so that
    clause would replace an already-complete report with the one remedy that is
    not wrong here. Named at the entry points so the specific report survives.
    """


class _RawWebsocketTransport:
    """msgpack + NumPy wire client using ``websockets`` + a vendored packer.

    This is the *only* transport. It speaks the exact same wire protocol the
    Cosmos 3 / OpenPI ``WebsocketPolicyServer`` expects (connect → recv
    msgpack metadata → send packed obs → recv packed action) and has zero
    NumPy-version constraints, so it composes cleanly with ``lerobot``
    (``numpy>=2``).

    Requires only ``websockets`` and ``msgpack``.
    """

    def __init__(self, host: str, port: int, api_key: str | None = None, read_timeout: float = _READ_TIMEOUT_SECS):
        self.uri = f"ws://{host}:{port}"
        self.api_key = api_key
        self.read_timeout = float(read_timeout)
        self._ws: Any = None
        from . import _msgpack_numpy as _mnp  # vendored, numpy-agnostic

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    def _decode(self, frame: Any, what: str) -> Any:
        """Unpack one inbound frame, or report a peer that does not speak this wire.

        Every other way this exchange can fail already names the server and the
        endpoint: a connection that was refused carries the start-the-server
        hint, a read that expired carries the silent-server report and the
        budget it waited out. A frame the *codec* cannot read was the one that
        did not - ``msgpack`` raises ``ExtraData`` (a ``ValueError``) for
        anything that is not a msgpack document, and a ``TypeError`` for a text
        frame - so a proxy's 502 page, a JSON server on the same port or a
        checkpoint served by something else surfaced as ``unpack(b) received
        extra data.``, naming neither the server, the endpoint, nor which read
        it answered.

        Both servers this package ships already answer this way at the other end
        of their own wire: ``PolicyServer`` marshals an unreadable frame back as
        an ``error`` message (MODULE strands_robots.inference.server) and the
        MoveIt 2 node replies ``malformed_request`` (MODULE
        strands_robots.policies.moveit2.server.zmq_node). This is the client
        half of the same rule.

        Args:
            frame: The raw WebSocket frame as received, text or binary.
            what: Which read it answered (``"metadata handshake"`` / ``"reply"``,
                the same words the silent-server reports use), so the report
                names the exchange that failed rather than the codec.

        Returns:
            The decoded msgpack payload.

        Raises:
            _UnreadableFrame: If the frame is not a msgpack document. The codec
                failure is kept as the cause and the frame's opening bytes are
                quoted, since those are what identify the peer that answered.
        """
        try:
            return self._mnp.unpackb(frame)
        except (TypeError, ValueError) as exc:
            raise _UnreadableFrame(
                f"{_SERVER_NAME} at {self.uri} sent a {what} frame this client cannot read as a "
                f"msgpack message ({type(exc).__name__}: {exc}); the frame begins {frame[:60]!r}. "
                "A peer that answers here in another wire format is not a Cosmos 3 RoboLab policy "
                "server: check the port serves python -m "
                "cosmos_framework.scripts.action_policy_server_robolab and not another WebSocket "
                "policy server."
            ) from exc

    def _ensure(self) -> Any:
        if self._ws is not None:
            return self._ws
        import websockets.sync.client as _wsc

        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        # ``Any`` for the same reason ``self._ws`` is declared ``Any``: the frames
        # go straight to the vendored packer, which treats them as opaque.
        ws: Any = _wsc.connect(self.uri, compression=None, max_size=None, additional_headers=headers)
        # ``self._ws`` is published only once the handshake has been consumed.
        # Assigned before the read, a failed handshake left a live connection
        # cached behind the error it had just raised, with the metadata frame
        # still unread - so the next ``infer`` would have read that blob as its
        # action chunk. Same rule as ``RemotePolicy._connect`` (MODULE
        # strands_robots.inference.client).
        established = False
        try:
            self._decode(ws.recv(timeout=self.read_timeout), "metadata handshake")
            established = True
        finally:
            if not established:
                close_quietly(ws)
        self._ws = ws
        return ws

    def get_server_metadata(self) -> dict[str, Any]:
        self._ensure()
        return {}

    def _exchange(self, message: Mapping[str, Any]) -> Any:
        """Send one request and read its reply, or discard the connection.

        A reply that was not read is still produced and still queued on the
        socket, so a connection whose exchange did not complete is in an unknown
        position: the next request would read the previous request's answer, and
        a stale ``[T, D]`` chunk is well-formed - it drives the robot from an
        observation it has already moved past, with nothing to distinguish it.
        Discarding it is the rule ``RemotePolicy._request`` states for the same
        wire (MODULE strands_robots.inference.client).
        """
        ws = self._ensure()
        exchanged = False
        try:
            ws.send(self._packer.pack(dict(message)))
            resp = ws.recv(timeout=self.read_timeout)
            exchanged = True
            return resp
        finally:
            if not exchanged:
                self.close()

    def close(self) -> None:
        """Drop the connection if one is open (best-effort and idempotent)."""
        if self._ws is not None:
            close_quietly(self._ws)
            self._ws = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        resp = self._exchange(observation)
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        return self._decode(resp, "reply")

    def reset(self) -> None:
        pass


class Cosmos3WebsocketClient:
    """Self-contained WebSocket client for the Cosmos 3 policy server.

    Args:
        host: Server hostname or IP.
        port: Server WebSocket port.
        api_key: Optional bearer token forwarded to the server, when set.
        read_timeout: Seconds any single read off the connection may wait - the
            metadata handshake and every action chunk. Only a positive finite
            number names a budget: ``0``, a negative and ``True`` expire against
            a server that is answering normally, and ``inf`` cannot express "no
            deadline" here at all, because ``websockets`` raises
            ``OverflowError`` computing a deadline from it. Refused in the
            constructor, while the caller still holds the value, since the
            connection is opened lazily and an unusable budget would otherwise
            surface mid-rollout naming no parameter.
        transport: Accepted for backwards compatibility. The only supported
            transport is the vendored raw msgpack+websockets packer; any
            value is treated as ``"raw"`` and a deprecation warning is
            logged for the legacy ``"openpi"`` / ``"auto"`` selectors.

    The connection is established lazily on the first :meth:`infer` (or
    :meth:`get_server_metadata`) call so constructing a policy does not
    require the server to already be up - matching ``Gr00tInferenceClient``.

    Raises:
        ValueError: If *read_timeout* is not a positive finite number.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
        transport: str = "raw",
        read_timeout: float = _READ_TIMEOUT_SECS,
    ):
        if err := positive_finite_number_error(read_timeout, "read_timeout", type(self).__name__):
            raise ValueError(err)
        self.host = host
        self.port = port
        self.api_key = api_key
        self.read_timeout = float(read_timeout)
        if transport not in (None, "", "raw"):
            logger.warning(
                "Cosmos3WebsocketClient(transport=%r) is deprecated - the "
                "openpi-client dependency has been removed and the only "
                "supported transport is the vendored raw msgpack+websockets "
                "packer. Treating as transport='raw'.",
                transport,
            )
        self.transport = "raw"
        self._client: Any = None

    def _silent_server_error(self, what: str) -> str:
        """Report for a server that accepted the connection and then went quiet."""
        return silent_server_error(
            server=_SERVER_NAME,
            uri=f"ws://{self.host}:{self.port}",
            what=what,
            timeout=self.read_timeout,
            budget_param="read_timeout",
        )

    def _server_hint(self) -> str:
        """Actionable hint for starting the Cosmos 3 RoboLab policy server."""
        return (
            f"Could not reach the Cosmos 3 policy server at ws://{self.host}:{self.port}. "
            "Start it first (holds the GPU) from a Cosmos Framework checkout:\n"
            "  uv sync --all-extras --group=cu130-train --group=policy-server\n"
            "  python -m cosmos_framework.scripts.action_policy_server_robolab \\\n"
            "    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port "
            f"{self.port}\n"
            f"Then confirm it is up:  curl http://{self.host}:{self.port}/healthz"
        )

    def _ensure_client(self) -> Any:
        """Connect on first use (lazy)."""
        if self._client is not None:
            return self._client
        try:
            self._client = _RawWebsocketTransport(self.host, self.port, self.api_key, self.read_timeout)
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e
        logger.info(
            "Cosmos3WebsocketClient ready for ws://%s:%s (transport=raw)",
            self.host,
            self.port,
        )
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the metadata dict the server sends on connect.

        Returns:
            The server metadata dict.

        Raises:
            ConnectionError: If the server cannot be reached, if it accepted the
                connection and sent no handshake within ``read_timeout``, or if
                the handshake it did send is not a msgpack document
                (:meth:`_RawWebsocketTransport._decode`). Three separate
                reports: the second and third servers are listening, so telling
                the operator to start one names the only thing that is not wrong.
        """
        client = self._ensure_client()
        try:
            return client.get_server_metadata()
        except _UnreadableFrame:
            # Re-raised unchanged: the seam that held the frame already named the
            # server, the endpoint, the read and the frame itself, and the
            # ``except OSError`` clause below - a ``ConnectionError`` is an
            # ``OSError`` - would replace that with "start the server", the one
            # remedy that is not wrong for a peer which just answered.
            raise
        except TimeoutError as e:
            # Before ``OSError`` (a ``TimeoutError`` is one), because the two
            # cases have separate remedies: this server is listening and did not
            # answer, so telling the operator to start it names the one thing
            # that is not wrong.
            raise ConnectionError(self._silent_server_error("metadata handshake")) from e
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send an observation dict and return the server response.

        Args:
            observation: Observation dict in the Cosmos 3 / OpenPI-compatible
                wire schema. Must contain ``prompt`` and at least one image
                plus the required state keys for the served action space.

        Returns:
            Response dict containing at least ``"action"`` (an ``[T, D]``
            NumPy array) and optionally ``"video"`` / ``"server_timing"``.

        Raises:
            ConnectionError: If the server cannot be reached, if it accepted the
                connection and answered nothing within ``read_timeout``, or if
                the frame it did send is not a msgpack document
                (:meth:`_RawWebsocketTransport._decode`).
            RuntimeError: If the server marshals an inference failure back as a
                text frame, which carries the server-side traceback.
        """
        client = self._ensure_client()
        try:
            return client.infer(observation)
        except _UnreadableFrame:
            # Re-raised unchanged: the seam that held the frame already named the
            # server, the endpoint, the read and the frame itself, and the
            # ``except OSError`` clause below - a ``ConnectionError`` is an
            # ``OSError`` - would replace that with "start the server", the one
            # remedy that is not wrong for a peer which just answered.
            raise
        except TimeoutError as e:
            # Before ``OSError`` (a ``TimeoutError`` is one), because the two
            # cases have separate remedies: this server is listening and did not
            # answer, so telling the operator to start it names the one thing
            # that is not wrong. "reply" rather than "action chunk" because the
            # first call also performs the handshake, and this clause cannot see
            # which of the two reads expired.
            raise ConnectionError(self._silent_server_error("reply")) from e
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e

    def reset(self) -> None:
        """Best-effort per-episode reset hint to the server.

        The raw transport is stateless on the client side - reset is a
        soft hint, never a correctness requirement (mirrors
        ``Gr00tPolicy.reset``). Any failure is swallowed.
        """
        try:
            client = self._ensure_client()
            reset_fn = getattr(client, "reset", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("Cosmos3WebsocketClient.reset best-effort failed: %s", e)
