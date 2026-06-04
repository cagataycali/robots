"""OpenPI WebSocket client for the Cosmos 3 RoboLab policy server.

Cosmos Framework ships a ready-made policy server
(``cosmos_framework.scripts.action_policy_server_robolab``) that serves
``nvidia/Cosmos3-Nano-Policy-DROID`` over OpenPI's ``WebsocketPolicyServer``
(msgpack + NumPy protocol). This client is a thin wrapper over OpenPI's
``WebsocketClientPolicy`` so :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy`
can speak to it in *service mode* — exactly mirroring how
:class:`~strands_robots.policies.groot.Gr00tPolicy` uses a ZMQ client.

Wire contract (verified against the server source):

* request  = observation dict, keys in the OpenPI ``observation/...`` namespace
             plus a top-level ``prompt`` string.
* response = ``{"action": np.ndarray[T, D], "video"?: np.ndarray, ...}``.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


def _load_openpi_client() -> Any:
    """Import OpenPI's websocket client policy module (optional dependency)."""
    return require_optional(
        "openpi_client.websocket_client_policy",
        pip_install="openpi-client",
        extra="cosmos3-service",
        purpose="Cosmos 3 service-mode policy inference",
    )


class Cosmos3WebsocketClient:
    """Thin OpenPI websocket client for the Cosmos 3 policy server.

    Args:
        host: Server hostname or IP.
        port: Server WebSocket port.
        api_key: Optional bearer token forwarded to the server, when set.

    The connection is established lazily on the first :meth:`infer` (or
    :meth:`get_server_metadata`) call so constructing a policy does not
    require the server to already be up — matching ``Gr00tInferenceClient``.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, api_key: str | None = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Connect on first use (lazy)."""
        if self._client is None:
            mod = _load_openpi_client()
            self._client = mod.WebsocketClientPolicy(host=self.host, port=self.port, api_key=self.api_key)
            logger.info("Cosmos3WebsocketClient connected to ws://%s:%s", self.host, self.port)
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the metadata dict the server sends on connect."""
        client = self._ensure_client()
        return client.get_server_metadata()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send an observation dict and return the server response.

        Args:
            observation: OpenPI-protocol observation. Must contain ``prompt``
                and at least one image plus the required state keys for the
                served action space.

        Returns:
            Response dict containing at least ``"action"`` (an ``[T, D]``
            NumPy array) and optionally ``"video"`` / ``"server_timing"``.
        """
        client = self._ensure_client()
        return client.infer(observation)

    def reset(self) -> None:
        """Best-effort per-episode reset hint to the server.

        OpenPI's ``WebsocketClientPolicy`` exposes ``reset()`` on newer
        builds; older ones don't. Any failure is swallowed — reset is a soft
        hint, never a correctness requirement (mirrors ``Gr00tPolicy.reset``).
        """
        try:
            client = self._ensure_client()
            reset_fn = getattr(client, "reset", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("Cosmos3WebsocketClient.reset best-effort failed: %s", e)
