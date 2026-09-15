"""MoveIt2 ZMQ client - msgpack-encoded REQ/REP transport.

Mirrors the shape of :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`
so users familiar with the GR00T service-mode pattern can use the same mental
model. The only wire types are JSON-equivalent values plus 1-D / 2-D float
arrays (joint state and trajectory rows), so we keep msgpack handling
deliberately minimal - no custom ``__class__`` markers, no numpy probing on
the hot path.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.utils import coerce_zmq_timeout_ms, require_optional

logger = logging.getLogger(__name__)

_SERVER_NAME = "MoveIt2 sidecar"


def _load_zmq() -> Any:
    """Load ZMQ dependency."""
    return require_optional(
        "zmq",
        pip_install="pyzmq",
        extra="moveit2",
        purpose="MoveIt2 service inference",
    )


def _load_msgpack() -> Any:
    """Load msgpack dependency."""
    return require_optional(
        "msgpack",
        extra="moveit2",
        purpose="MoveIt2 service inference",
    )


class MsgSerializer:
    """(De)serialization helpers for ZMQ communication with the MoveIt2 sidecar.

    The wire format only contains JSON-shaped values (numbers, strings,
    bools, lists, and dicts); we therefore use plain msgpack with the
    default packer / unpacker and no custom hooks.
    """

    @staticmethod
    def to_bytes(data: dict[str, Any]) -> bytes:
        """Pack a JSON-shaped request dict to msgpack bytes (``use_bin_type`` keeps str and bytes distinct on the wire)."""
        msgpack = _load_msgpack()
        # use_bin_type=True keeps str / bytes distinct on the wire
        # (matches msgpack >=1.0 default but explicit is better than
        # implicit for cross-language sidecars).
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """Unpack msgpack bytes into the value they encode, decoding msgpack ``str`` back to Python ``str`` (``raw=False``).

        Returns:
            Whatever value *data* encodes. ``unpackb`` decodes any valid msgpack
            value, not just a map - the single byte ``0x2a`` is the integer 42,
            and a string, list, nil or bool decode just as cleanly - so this is
            deliberately not annotated ``dict``. A caller that needs a map grades
            for one; see
            :meth:`MoveIt2InferenceClient._decode_reply`.

        Raises:
            ValueError: If *data* is not exactly one msgpack object.
                ``msgpack``'s ``ExtraData``, ``FormatError`` and ``StackError``
                are all ``ValueError``, so trailing bytes, a truncated frame and
                a frame that is not msgpack at all arrive here.
            TypeError: If *data* is not bytes-like.
        """
        msgpack = _load_msgpack()
        # raw=False decodes msgpack ``str`` types back to Python ``str``
        # (msgpack >=1.0 default); strict_map_key=False allows
        # numeric / bytes keys but we never emit those.
        return msgpack.unpackb(data, raw=False)


def _unreadable_reply(*, uri: str, endpoint: str, problem: str, frame: bytes) -> str:
    """Return the report for a sidecar reply this client cannot read.

    The reference sidecar refuses the mirror image of this in the request
    direction, and refuses both of its shapes in one class, because "either way
    the peer did not send a request" (see
    MODULE strands_robots.policies.moveit2.server.zmq_node). The reply direction
    is graded the same way and for the same reason: bytes that are not msgpack
    and a value that decodes but is not a map are both "the peer did not send a
    reply", and neither is a fact about the codec.

    Args:
        uri: The ``tcp://host:port`` this client dialled, so the report names the
            endpoint actually in use rather than the one the caller meant.
        endpoint: The request the reply answered (``"ping"`` / ``"plan"`` /
            ``"reset"``), so a caller with several round-trips behind it knows
            which one came back unreadable.
        problem: What is wrong with the frame, in the sidecar's own vocabulary.
        frame: The raw reply, quoted from the front so an operator can recognise
            a wire format - an HTTP error page, JSON, a bare msgpack scalar.

    Returns:
        A message naming the peer, the endpoint, the problem, the frame's opening
        bytes, and the remedy.
    """
    return (
        f"{_SERVER_NAME} at {uri} answered {endpoint!r} with an unreadable reply: "
        f"{problem}; it begins {frame[:60]!r}. A peer that answers here in another wire "
        f"format is not a MoveIt2 sidecar: check the port serves "
        f"strands_robots.policies.moveit2.server.zmq_node (msgpack REQ/REP) and not "
        f"another policy server."
    )


class MoveIt2InferenceClient:
    """ZMQ REQ client for the MoveIt2 sidecar.

    Args:
        host: Server hostname or IP. Default ``"127.0.0.1"`` - bind to
            loopback by default; users opt into network exposure.
        port: Server port.
        timeout_ms: Socket send/recv timeout in milliseconds, applied as
            ``RCVTIMEO`` and ``SNDTIMEO`` on the REQ socket. Only a positive
            whole number up to
            :data:`~strands_robots.utils.MAX_ZMQ_TIMEOUT_MS` names a budget;
            an integral ``float`` or NumPy integer is accepted and stored as an
            ``int``, since ``setsockopt`` takes only the latter. ``0`` is ZMQ's
            "return immediately" spelling and ``-1`` its "block forever" one,
            and both are refused - see
            :func:`~strands_robots.utils.coerce_zmq_timeout_ms`.
        api_token: Optional token included in every request for
            authentication. When unset, no auth is sent. Sent in
            plaintext over TCP - use a TLS tunnel or SSH port-forward
            for non-localhost deployments (same caveat as
            :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`).

    Raises:
        ValueError: If ``timeout_ms`` does not name a usable wait budget - see
            :func:`~strands_robots.utils.coerce_zmq_timeout_ms`.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ) -> None:
        # A timeout that names no wait budget is refused here, while the caller
        # still holds the value, because ZMQ's reaction to one is
        # indistinguishable from an absent sidecar: ``0`` and ``False`` are its
        # "return immediately" spelling, so every request raises ``zmq.Again``
        # against a server that is running and reachable, and ``ping`` below
        # reports that as ``False`` with the reason at ``logger.debug`` only.
        # ``True`` is a silent 1 ms budget. The remaining values never reach a
        # verdict at all - ``setsockopt`` raises ``ZMQError``, ``TypeError`` or
        # ``OverflowError`` from inside ``pyzmq``, naming no parameter - and that
        # includes ``15000.0`` and ``np.int64(15000)``, which name usable
        # budgets the sibling transports accept, hence the coercion rather than
        # a bare refusal.
        coerced_timeout, timeout_reason = coerce_zmq_timeout_ms(type(self).__name__, "timeout_ms", timeout_ms)
        if coerced_timeout is None:
            raise ValueError(timeout_reason)
        self._zmq = _load_zmq()
        self.context = self._zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = coerced_timeout
        self.api_token = api_token

        if api_token and host not in ("localhost", "127.0.0.1", "::1"):
            logger.warning(
                "API token will be sent in plaintext over TCP to %s:%s. "
                "ZMQ does not encrypt traffic by default. Consider using a "
                "TLS tunnel or SSH port-forward for non-localhost deployments.",
                host,
                port,
            )

        self._init_socket()
        logger.debug(
            "MoveIt2InferenceClient initialised: %s:%s (timeout=%dms)",
            host,
            port,
            timeout_ms,
        )

    def _init_socket(self) -> None:
        """Create and connect the ZMQ REQ socket."""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        # LINGER=0 so socket.close() / context.term() never block waiting to
        # flush undelivered requests to a dead sidecar. Without it the default
        # linger is infinite, so a queued request to an unreachable server
        # hangs teardown (and interpreter shutdown / GC of __del__) forever.
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def reconnect(self) -> None:
        """Close and re-create the socket connection."""
        logger.info("Reconnecting to %s:%s", self.host, self.port)
        try:
            self.socket.close()
        except Exception:  # noqa: BLE001 - socket close failures don't matter on reconnect
            pass
        self._init_socket()

    def ping(self) -> bool:
        """Check server connectivity. Returns True if the server responds."""
        try:
            self.call_endpoint("ping")
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            logger.debug("Ping failed: %s", exc)
            return False

    def _decode_reply(self, message: bytes, endpoint: str) -> dict[str, Any]:
        """Decode one sidecar reply into a map, or refuse it naming the peer.

        Both refusals replace a report that names the codec, or no report at all,
        with one that names the peer and the endpoint. Bytes that are not msgpack
        raised ``ExtraData: unpack(b) received extra data.`` from inside
        ``msgpack``, which names neither the host, the port nor the request it
        answered. A value that decodes but is not a map was worse than that: it
        was returned as the declared ``dict``, because ``"error" in reply`` is a
        membership test that a list and a string answer ``False`` without
        raising, so the caller was handed a non-map typed as a map and
        ``response.get(...)`` failed one frame later in
        MODULE strands_robots.policies.moveit2.policy with an ``AttributeError``
        naming ``str``. A string that happens to contain ``"error"`` took the
        server-error branch instead and raised ``TypeError: string indices must
        be integers`` - a third report for one wire fault, none of them naming
        the sidecar.

        ``ConnectionError`` needs no private subclass here, unlike the WebSocket
        clients in this package: nothing between this seam and the caller catches
        ``OSError``, and ``zmq.Again`` is not one, so no broad clause can clobber
        the report on its way out. :meth:`ping` still absorbs it, which is that
        method's contract - any failure means "not reachable".

        Args:
            message: The raw reply frame, treated as opaque.
            endpoint: The request it answered, named in the report.

        Returns:
            The decoded reply map.

        Raises:
            ConnectionError: If *message* is not exactly one msgpack object, or
                decodes to a value that is not a map. The codec failure is kept
                as the cause of the former.
        """
        uri = f"tcp://{self.host}:{self.port}"
        try:
            reply = MsgSerializer.from_bytes(message)
        except (TypeError, ValueError) as exc:
            raise ConnectionError(
                _unreadable_reply(
                    uri=uri,
                    endpoint=endpoint,
                    problem=f"not msgpack ({type(exc).__name__}: {exc})",
                    frame=message,
                )
            ) from exc
        if not isinstance(reply, dict):
            raise ConnectionError(
                _unreadable_reply(
                    uri=uri,
                    endpoint=endpoint,
                    problem=f"expected a msgpack map, got {type(reply).__name__}",
                    frame=message,
                )
            )
        return reply

    def call_endpoint(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request to the server and return the parsed response.

        Args:
            endpoint: Server endpoint name (``"ping"``, ``"plan"``, ``"reset"``).
            data: Optional request payload.

        Returns:
            Parsed response dict from the server.

        Raises:
            ConnectionError: If the reply is not a msgpack map - see
                :meth:`_decode_reply`.
            RuntimeError: If the server returns an ``error`` field.
        """
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = self._decode_reply(message, endpoint)
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def plan(
        self,
        joint_state: list[float] | None,
        planning_group: str,
        target_pose: list[float] | None = None,
        target_joints: dict[str, float] | None = None,
        world_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a plan from the sidecar.

        Args:
            joint_state: Current joint configuration (radians / metres).
                ``None`` lets the server use its own latest state estimate.
            planning_group: MoveIt2 planning-group name (e.g. ``"arm"``).
            target_pose: Cartesian goal ``[x, y, z, qw, qx, qy, qz]`` in
                the planning group's base frame. Mutually exclusive with
                ``target_joints`` (server enforces).
            target_joints: Joint-space goal keyed by joint name.
            world_update: Per-call world refresh for collision-aware
                planning (depth, mesh, ...). Server-defined schema.

        Returns:
            ``{"trajectory": [[t, q0, q1, ...], ...], "success": bool, "status": str}``.
        """
        payload: dict[str, Any] = {
            "joint_state": joint_state,
            "planning_group": planning_group,
        }
        if target_pose is not None:
            payload["target_pose"] = target_pose
        if target_joints is not None:
            payload["target_joints"] = target_joints
        if world_update is not None:
            payload["world_update"] = world_update
        return self.call_endpoint("plan", payload)

    def _teardown(self) -> None:
        """Best-effort ZMQ socket + context teardown.

        Extracted from ``__del__`` so the destructor stays one line - CodeQL
        flags non-trivial logic in ``__del__`` because exceptions raised
        during interpreter shutdown are swallowed silently and can mask
        resource leaks.

        The socket is created with ``LINGER=0`` (see ``_init_socket``) so
        ``close()`` discards any undelivered request immediately instead of
        blocking to flush it to a dead sidecar; ``term()`` then returns once
        the (now-closed) socket is gone. Without the zero linger, a request
        queued to an unreachable server would hang teardown - and the GC that
        drives ``__del__`` / interpreter shutdown - indefinitely.
        """
        try:
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        self._teardown()


__all__ = [
    "MoveIt2InferenceClient",
    "MsgSerializer",
]
