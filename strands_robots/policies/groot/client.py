"""GR00T inference client - ZMQ client for inference-service communication.

Handles serialization of numpy arrays and ModalityConfig objects over ZMQ
using msgpack with custom encode/decode hooks.
"""

import io
import json
import logging
import socket as _socket
from typing import Any

import numpy as np

from strands_robots.utils import coerce_zmq_timeout_ms, require_optional

from .data_config import ModalityConfig

logger = logging.getLogger(__name__)

_SERVER_NAME = "GR00T policy server"

# How long :func:`_listener_present` gives a bare TCP connect to the server's
# port. It only has to tell "nothing listening" from "listening but silent",
# so a second is plenty on a LAN and short next to any inference budget.
_PROBE_TIMEOUT_S = 1.0


def _listener_present(host: str, port: int) -> bool | None:
    """Whether anything accepts a TCP connection on ``host:port`` right now.

    Called after a request timed out, to say which side the wait was on: an
    absent process (connection refused - ZMQ's ``connect`` is lazy and never
    reports it, so a typo in the port looks exactly like a slow model) or a
    listener that accepted the frame and never answered (checkpoint still
    loading, or a wedged forward pass). ``None`` when the probe itself could
    not decide - a name that does not resolve, a firewall that drops instead
    of refusing - so the report falls back to naming both possibilities.
    """
    try:
        with _socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_S):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return None


def unreachable_server_error(*, uri: str, endpoint: str, timeout_ms: int, listener: bool | None) -> str:
    """The report for a request that timed out, worded for the side that failed.

    Args:
        uri: ``tcp://host:port`` the client dialled.
        endpoint: The request that got no answer, named so a caller batching
            several knows which one.
        timeout_ms: The budget that expired - it is the knob, so it is named.
        listener: :func:`_listener_present`'s verdict.

    Returns:
        One sentence each for what happened, why (as far as the probe can
        tell) and what to do; the remedy names the parameter that changes the
        budget and the command that starts a server.
    """
    head = f"{_SERVER_NAME} at {uri} did not answer '{endpoint}' within timeout_ms={timeout_ms}."
    if listener is False:
        return (
            f"{head} Nothing is listening on that port (connection refused), so no server is running there "
            "or host/port name the wrong place. Start one - gr00t_inference(action='start', "
            f"port={uri.rsplit(':', 1)[-1]}) runs the container, or `python -m gr00t.eval.run_gr00t_server "
            f"--port {uri.rsplit(':', 1)[-1]}` inside an Isaac-GR00T install - or pass the host/port it is "
            "actually on."
        )
    if listener is True:
        return (
            f"{head} The port is listening, so the server is still loading (a large checkpoint takes "
            "minutes) or it is wedged: read its log to tell those apart, and raise timeout_ms if the model "
            "is simply slower than the budget."
        )
    return (
        f"{head} Could not tell whether anything is listening there (the probe got neither a refusal nor a "
        "connection). Check host/port and that a server is running there (gr00t_inference(action='start') "
        "or `python -m gr00t.eval.run_gr00t_server`), or raise timeout_ms if it is only slow."
    )


def _load_zmq():
    """Load ZMQ dependency."""
    return require_optional("zmq", pip_install="pyzmq", extra="groot-service", purpose="GR00T service inference")


def _load_msgpack():
    """Load msgpack dependency."""
    return require_optional("msgpack", extra="groot-service", purpose="GR00T service inference")


class MsgSerializer:
    """(De)serialization helpers for ZMQ communication with GR00T services.

    Handles numpy ndarray and ModalityConfig types that cannot be directly
    serialized by msgpack.
    """

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        """Pack a request dict to msgpack bytes, encoding numpy arrays and ModalityConfig values via the custom hook."""
        msgpack = _load_msgpack()
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """Unpack msgpack bytes into the value they encode, decoding numpy arrays and ModalityConfig values.

        Returns:
            Whatever value *data* encodes. ``unpackb`` decodes any valid msgpack
            value, not just a map - the single byte ``0x2a`` is the integer 42,
            and a string, list, nil or bool decode just as cleanly - so this is
            deliberately not annotated ``dict``. A caller that needs a map or a
            list grades for one; see
            :meth:`Gr00tInferenceClient._decode_reply`.

        Raises:
            ValueError: If *data* is not exactly one msgpack object.
                ``msgpack``'s ``ExtraData``, ``FormatError`` and ``StackError``
                are all ``ValueError``, so trailing bytes, a truncated frame and
                a frame that is not msgpack at all arrive here.
            TypeError: If *data* is not bytes-like.
        """
        msgpack = _load_msgpack()
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        """Decode custom types from msgpack wire format."""
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            # N1.6 serialized `as_json` as a JSON string (Pydantic `model_dump_json`).
            # N1.7 serializes `as_json` as a plain dict (via `to_json_serializable`)
            # and adds fields (sin_cos_embedding_keys, mean_std_embedding_keys, action_configs)
            # that our minimal ModalityConfig dataclass does not track.
            # Accept both wire forms AND tolerate unknown N1.7 fields so a single
            # client can talk to either server version.
            payload = obj["as_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            # Forward-compat: drop fields our dataclass does not know about.
            _allowed = {"delta_indices", "modality_keys"}
            filtered = {k: v for k, v in payload.items() if k in _allowed}
            return ModalityConfig(**filtered)
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        """Encode custom types to msgpack wire format."""
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        return obj


def _unreadable_reply(*, uri: str, endpoint: str, problem: str, frame: bytes) -> str:
    """Return the report for a server reply this client cannot read.

    The reference client (``gr00t.policy.server_client.PolicyClient``) already
    treats one wrong-peer frame this way - a bare ``b"ERROR"`` is refused with
    "Make sure we are running the correct policy server" - and reads its
    ``error`` field only off a reply that ``isinstance(response, dict)``. The
    two doors here generalise that: bytes that are not msgpack and a value that
    decodes but is neither a map nor a list are both "the peer did not send a
    reply", and neither is a fact about the codec.

    Args:
        uri: The ``tcp://host:port`` this client dialled, so the report names the
            endpoint actually in use rather than the one the caller meant.
        endpoint: The request the reply answered (``"ping"`` / ``"get_action"``
            / ``"reset"``), so a caller with several round-trips behind it knows
            which one came back unreadable.
        problem: What is wrong with the frame, in the server's own vocabulary.
        frame: The raw reply, quoted from the front so an operator can recognise
            a wire format - an HTTP error page, JSON, a bare msgpack scalar.

    Returns:
        A message naming the peer, the endpoint, the problem, the frame's opening
        bytes, and the remedy.
    """
    return (
        f"{_SERVER_NAME} at {uri} answered {endpoint!r} with an unreadable reply: "
        f"{problem}; it begins {frame[:60]!r}. A peer that answers here in another wire "
        f"format is not a GR00T policy server: check the port serves "
        f"gr00t.policy.server_client.PolicyServer (msgpack REQ/REP over ZMQ, as started by "
        f"gr00t.eval.run_gr00t_server) and not another policy server "
        f"(strands_robots.inference.server speaks JSON over WebSocket)."
    )


class Gr00tInferenceClient:
    """ZMQ REQ client for GR00T inference services.

    Handles socket lifecycle, timeout, and optional API-token authentication.

    Args:
        host: Server hostname or IP.
        port: Server port.
        timeout_ms: Socket send/receive timeout in milliseconds, applied as
            ``RCVTIMEO`` and ``SNDTIMEO`` on the REQ socket. Only a positive
            whole number up to
            :data:`~strands_robots.utils.MAX_ZMQ_TIMEOUT_MS` names a budget;
            an integral ``float`` or NumPy integer is accepted and stored as an
            ``int``, since ``setsockopt`` takes only the latter. ``0`` is ZMQ's
            "return immediately" spelling and ``-1`` its "block forever" one,
            and both are refused - see
            :func:`~strands_robots.utils.coerce_zmq_timeout_ms`.
        api_token: Optional token included in every request for authentication.

    Raises:
        ValueError: If ``timeout_ms`` does not name a usable wait budget - see
            :func:`~strands_robots.utils.coerce_zmq_timeout_ms`.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
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
        logger.debug("Gr00tInferenceClient initialized: %s:%s (timeout=%dms)", host, port, timeout_ms)

    def _init_socket(self):
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

    def reconnect(self):
        """Close and re-create the socket connection."""
        logger.info("Reconnecting to %s:%s", self.host, self.port)
        try:
            self.socket.close()
        except Exception:
            pass
        self._init_socket()

    def ping(self) -> bool:
        """Check server connectivity.

        Returns True if the server responds, False otherwise.
        Does NOT auto-reconnect - call :meth:`reconnect` explicitly if needed.
        """
        try:
            self.call_endpoint("ping")
            return True
        except Exception as exc:
            logger.debug("Ping failed: %s", exc)
            return False

    def _decode_reply(self, message: bytes, endpoint: str) -> dict | list:
        """Decode one server reply into a map or a list, or refuse it naming the peer.

        Both refusals replace a report that names the codec, or no report at all,
        with one that names the peer and the endpoint. Bytes that are not msgpack
        raised ``ExtraData: unpack(b) received extra data.`` from inside
        ``msgpack``, which names neither the host, the port nor the request it
        answered. A value that decodes but is neither a map nor a list was worse
        than that: ``"error" in response`` is a membership test, so a string
        answers it ``False`` without raising and was returned as the declared
        ``dict``, to fail one frame later in
        MODULE strands_robots.policies.groot.policy with ``AttributeError: 'str'
        object has no attribute 'items'``; an ``int`` or ``nil`` raised
        ``TypeError: argument of type 'int' is not iterable`` from the test
        itself; and a string that happens to contain ``"error"`` took the
        server-error branch and raised ``TypeError: string indices must be
        integers`` - three reports for one wire fault, none naming the server.

        A list is admitted because the reference server sends one:
        ``gr00t.policy.server_client.PolicyServer`` packs each handler's return
        value as-is, and ``get_action`` returns ``(action, info)``, which msgpack
        carries as a 2-element list. :meth:`get_action` unpacks that shape.

        ``ConnectionError`` needs no private subclass here, unlike the WebSocket
        clients in this package: nothing between this seam and the caller catches
        ``OSError``, and ``zmq.Again`` is not one, so no broad clause can clobber
        the report on its way out. :meth:`ping` still absorbs it, which is that
        method's contract - any failure means "not reachable" - and
        ``Gr00tPolicy.reset`` still logs it and continues, which is its.

        Args:
            message: The raw reply frame, treated as opaque.
            endpoint: The request it answered, named in the report.

        Returns:
            The decoded reply, a map or a list.

        Raises:
            ConnectionError: If *message* is not exactly one msgpack object, or
                decodes to a value that is neither a map nor a list. The codec
                failure is kept as the cause of the former.
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
        if not isinstance(reply, dict | list):
            raise ConnectionError(
                _unreadable_reply(
                    uri=uri,
                    endpoint=endpoint,
                    problem=f"expected a msgpack map or list, got {type(reply).__name__}",
                    frame=message,
                )
            )
        return reply

    def call_endpoint(self, endpoint: str, data: dict | None = None) -> dict | list:
        """Send a request to the server and return the parsed response.

        Args:
            endpoint: Server endpoint name (e.g. "ping", "get_action").
            data: Optional request payload.

        Returns:
            Parsed response from the server: a dict for every endpoint the
            reference server registers except ``get_action``, whose
            ``(action, info)`` tuple arrives as a 2-element list.

        Raises:
            ConnectionError: If no reply arrived within ``timeout_ms`` - the
                report names the URI, the endpoint, the budget and, from a TCP
                probe, whether anything is listening there (see
                :func:`unreachable_server_error`); ``zmq.Again`` is kept as the
                cause. Pre-fix that ``zmq.Again`` escaped raw, so the runner
                printed ``Policy failed: Resource temporarily unavailable`` -
                no host, no port, no remedy. Or if the reply is neither a
                msgpack map nor a list - see :meth:`_decode_reply`.
            RuntimeError: If the server returns an error response.
        """
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except self._zmq.Again as exc:
            # A REQ socket that timed out is stuck in its send/recv lockstep:
            # the next ``send`` raises EFSM. Re-create it here so the caller's
            # retry (or ``ping``) is a real attempt and not a second failure
            # about the state machine.
            self.reconnect()
            raise ConnectionError(
                unreachable_server_error(
                    uri=f"tcp://{self.host}:{self.port}",
                    endpoint=endpoint,
                    timeout_ms=self.timeout_ms,
                    listener=_listener_present(self.host, self.port),
                )
            ) from exc
        response = self._decode_reply(message, endpoint)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Send observations and receive an action chunk.

        Uses the envelope used by ``gr00t.policy.server_client.PolicyServer`` in
        both N1.6 and N1.7: the request body is
        ``{"observation": <obs>, "options": None}`` so the server can spread
        it as kwargs into ``policy.get_action(observation, options)``.

        The server returns ``(action, info)`` as a 2-tuple (msgpack-ed to a
        2-element list); we return just the action dict since the info dict
        is currently empty in all upstream embodiments.

        Raises:
            ConnectionError: If the reply is a list that is not ``(action, info)``
                - a length other than 2, or a first element that is not a map.
                Pre-fix such a list was returned as the declared action dict and
                failed one frame later on ``.items()``, naming the Python type
                and not the server. The wire-level refusals are
                :meth:`_decode_reply`'s.
        """
        response = self.call_endpoint("get_action", {"observation": observations, "options": None})
        # Older / custom servers may return the bare action dict.
        if isinstance(response, dict):
            return response
        # N1.6/N1.7 servers return a (action_dict, info_dict) tuple - msgpack
        # decodes tuples as lists, so this is the shape the reference server sends.
        if len(response) == 2 and isinstance(response[0], dict):
            action, _info = response
            return action
        raise ConnectionError(
            f"{_SERVER_NAME} at tcp://{self.host}:{self.port} answered 'get_action' with a list this "
            f"client cannot read as an action chunk: expected (action, info) - a 2-element list whose "
            f"first element is a map - or a bare action map, got a list of {len(response)} whose elements "
            f"are {[type(item).__name__ for item in response]}. Check the port serves "
            f"gr00t.policy.server_client.PolicyServer and not another msgpack REQ/REP service."
        )

    def __del__(self):
        # The socket is created with LINGER=0 (see _init_socket) so close()
        # discards any undelivered request immediately rather than blocking to
        # flush it to a dead sidecar; term() then returns once the closed
        # socket is gone. Without the zero linger a request queued to an
        # unreachable server hangs the GC that drives __del__ (and interpreter
        # shutdown) indefinitely.
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()


__all__ = [
    "Gr00tInferenceClient",
    "MsgSerializer",
]
