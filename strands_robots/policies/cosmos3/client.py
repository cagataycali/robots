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


class _RawWebsocketTransport:
    """openpi msgpack+NumPy wire client using only ``websockets`` + a vendored
    packer — **no ``openpi-client`` dependency**.

    ``openpi-client`` pins ``numpy<2.0``, which conflicts with ``lerobot``
    (``numpy>=2.0``). This transport speaks the exact same wire protocol
    (connect → recv msgpack metadata → send packed obs → recv packed action)
    so a Cosmos 3 rollout can run in a ``numpy>=2`` environment (e.g. a
    LeRobot dataset-recording venv) without the conflicting pin.

    Requires ``websockets`` and ``msgpack`` (both numpy-version agnostic).
    """

    def __init__(self, host: str, port: int, api_key: str | None = None):
        self.uri = f"ws://{host}:{port}"
        self.api_key = api_key
        self._ws: Any = None
        from . import _msgpack_numpy as _mnp  # vendored, numpy-agnostic

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    def _ensure(self) -> Any:
        if self._ws is None:
            import websockets.sync.client as _wsc

            headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
            self._ws = _wsc.connect(self.uri, compression=None, max_size=None, additional_headers=headers)
            self._mnp.unpackb(self._ws.recv())  # server metadata handshake
        return self._ws

    def get_server_metadata(self) -> dict[str, Any]:
        self._ensure()
        return {}

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        ws = self._ensure()
        ws.send(self._packer.pack(observation))
        resp = ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        return self._mnp.unpackb(resp)

    def reset(self) -> None:
        pass


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

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
        transport: str = "auto",
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        # transport: "openpi" (use openpi-client), "raw" (vendored packer, no
        # openpi-client / numpy<2 pin), or "auto" (prefer openpi-client, fall
        # back to raw when it is not importable — e.g. in a numpy>=2 / lerobot
        # recording env). See _RawWebsocketTransport for the conflict rationale.
        self.transport = transport
        self._client: Any = None

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
        """Connect on first use (lazy).

        Transport selection (see ``__init__``):
          * ``"raw"``    -> vendored packer, no openpi-client (numpy>=2 safe).
          * ``"openpi"`` -> require openpi-client.
          * ``"auto"``   -> openpi-client if importable, else raw.
        """
        if self._client is not None:
            return self._client

        use_raw = self.transport == "raw"
        if self.transport == "auto":
            try:
                _load_openpi_client()
                use_raw = False
            except Exception:  # noqa: BLE001 - openpi-client not installed -> raw
                use_raw = True

        try:
            if use_raw:
                self._client = _RawWebsocketTransport(self.host, self.port, self.api_key)
            else:
                mod = _load_openpi_client()
                self._client = mod.WebsocketClientPolicy(host=self.host, port=self.port, api_key=self.api_key)
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        logger.info(
            "Cosmos3WebsocketClient connected to ws://%s:%s (transport=%s)",
            self.host,
            self.port,
            "raw" if use_raw else "openpi",
        )
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the metadata dict the server sends on connect."""
        client = self._ensure_client()
        try:
            return client.get_server_metadata()
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e

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
        try:
            return client.infer(observation)
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e

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
