"""Qwen-VLA inference client - ZMQ client for service-mode communication.

Mirrors :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`:
a ZMQ REQ socket with msgpack (de)serialization of numpy arrays, optional
API-token auth, timeout + reconnect. The wire envelope matches the GR00T
``server_client.PolicyServer`` convention so a single server harness can host
either model family during the upstream-checkpoint gap (PLAN section 6.1).
"""

import io
import logging
from typing import Any

import numpy as np

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


def _load_zmq():
    """Load the ZMQ dependency (optional extra ``qwen-vla-service``)."""
    return require_optional("zmq", pip_install="pyzmq", extra="qwen-vla-service", purpose="Qwen-VLA service inference")


def _load_msgpack():
    """Load the msgpack dependency (optional extra ``qwen-vla-service``)."""
    return require_optional("msgpack", extra="qwen-vla-service", purpose="Qwen-VLA service inference")


class MsgSerializer:
    """(De)serialization helpers for ZMQ communication with Qwen-VLA services.

    Handles numpy ndarray types that msgpack cannot serialize natively. The
    npy round-trip preserves dtype + shape exactly, which matters for the
    action chunk ``Y[H x K]`` and the uint8 camera frames.
    """

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        msgpack = _load_msgpack()
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        msgpack = _load_msgpack()
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        """Decode custom types from msgpack wire format."""
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        """Encode custom types to msgpack wire format."""
        if isinstance(obj, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        return obj


class QwenVlaInferenceClient:
    """ZMQ REQ client for a Qwen-VLA inference service.

    Args:
        host: Server hostname or IP.
        port: Server port.
        timeout_ms: Socket timeout in milliseconds.
        api_token: Optional token included in every request for authentication.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self._zmq = _load_zmq()
        self.context = self._zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token

        if api_token and host not in ("localhost", "127.0.0.1", "::1"):
            logger.warning(
                "API token will be sent in plaintext over TCP to %s:%s. ZMQ does not "
                "encrypt traffic by default. Use a TLS tunnel or SSH port-forward for "
                "non-localhost deployments.",
                host,
                port,
            )

        self._init_socket()
        logger.debug("QwenVlaInferenceClient initialized: %s:%s (timeout=%dms)", host, port, timeout_ms)

    def _init_socket(self):
        """Create and connect the ZMQ REQ socket."""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def reconnect(self):
        """Close and re-create the socket connection."""
        logger.info("Reconnecting to %s:%s", self.host, self.port)
        try:
            self.socket.close()
        except Exception:
            # Best-effort close: the socket may already be closed or in a bad
            # state; we are about to recreate it in _init_socket() regardless.
            pass
        self._init_socket()

    def ping(self) -> bool:
        """Check server connectivity. Does NOT auto-reconnect."""
        try:
            self.call_endpoint("ping")
            return True
        except Exception as exc:
            logger.debug("Ping failed: %s", exc)
            return False

    def call_endpoint(self, endpoint: str, data: dict | None = None) -> dict:
        """Send a request to the server and return the parsed response.

        Args:
            endpoint: Server endpoint name (e.g. ``"ping"``, ``"get_action"``,
                ``"reset"``).
            data: Optional request payload.

        Returns:
            Parsed response dict from the server.

        Raises:
            RuntimeError: If the server returns an error response.
        """
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send an observation batch and receive an action chunk.

        Uses the same envelope as the GR00T server
        (``{"observation": <obs>, "options": None}``) so one server harness
        can serve both model families. The server returns either a bare
        action dict or an ``(action, info)`` 2-tuple (msgpack-decoded to a
        list); we return just the action dict.
        """
        response = self.call_endpoint("get_action", {"observation": observation, "options": None})
        if isinstance(response, list | tuple) and len(response) == 2:
            action, _info = response
            return action
        return response

    def __del__(self):
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()


__all__ = ["QwenVlaInferenceClient", "MsgSerializer"]
