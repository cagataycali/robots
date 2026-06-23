"""WebSocket client for the VERA policy server.

VERA (https://github.com/sizhe-li/VERA) ships a websocket policy server
(``python -m vera.server.start_vera_server``) that serves the two-stage
video-planner + Jacobian-IDM policy over msgpack+NumPy. Wire spec:
``vera/docs/SERVER_PROTOCOL_SPEC.md``.

This module ships a **self-contained** client — only ``websockets`` +
``msgpack`` + a vendored numpy packer. No ``vera``, no ``openpi-client``, no
``flash-attn``. That keeps a strands-robots rollout install-light (the heavy
GPU stack lives on the server side) and avoids numpy-version pins that would
collide with ``lerobot``.

Wire contract (verified against vera ``WebsocketPolicyServer``):

* On connect: server sends a packed ``VeraServerConfig`` dict (view_keys,
  view_widths, action_horizon, action_dim, control_dt, action_space,
  embodiment, ...).
* ``infer`` request: an observation dict that MUST include
  ``context_rgb`` ([T,H,W,3] uint8 or float[0,1]) and SHOULD include
  ``view_keys``, ``view_widths``, ``session_id``; optional ``prompt``,
  ``q_robot``, ``eef_pos``, ``eef_quat``, ``gripper_qpos``, ``rgb``
  (defaults to the last ``context_rgb`` frame).
* ``infer`` response: ``{"action": np.ndarray[H, D], "info": dict, ...}``.
* ``reset`` request: ``{"session_id": str, "reason": str}``. Response is
  the string ``"reset successful"`` (any other string is an error).
* Error sentinel: a *string* response means the server raised — surface it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VeraWebsocketClient:
    """Self-contained WebSocket client for the VERA policy server.

    Args:
        host: Server hostname or IP.
        port: Server WebSocket port (8000 default; PushT example uses 8820,
            MimicGen uses 8800).
        api_key: Optional bearer token forwarded as ``Authorization: Api-Key …``.

    The connection is established lazily on the first ``infer``/``reset`` so
    constructing a policy does not require the server to already be up
    (matching ``Cosmos3WebsocketClient`` / ``Gr00tInferenceClient``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self._uri = f"ws://{host}:{port}"
        self._ws: Any = None
        self._server_metadata: dict[str, Any] | None = None
        from . import _msgpack_numpy as _mnp

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    # ---------------------------------------------------------------- helpers
    def _server_hint(self) -> str:
        """Actionable hint for starting the VERA policy server."""
        return (
            f"Could not reach the VERA policy server at {self._uri}. Start it first "
            "(holds the GPU):\n"
            "  # From a VERA checkout (https://github.com/sizhe-li/VERA):\n"
            "  pip install -e '.[idm,video]'\n"
            "  # PushT (small DFoT planner — loads in seconds):\n"
            f"  python -m vera.server.start_vera_server --embodiment pusht --port {self.port}\n"
            "  # MimicGen (Panda — WAN 1.3B planner, needs checkpoints):\n"
            "  python -m vera.server.start_vera_server --embodiment mimicgen "
            f"--port {self.port} \\\n"
            "      --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml\n"
        )

    def _ensure(self):
        """Connect on first use (lazy)."""
        if self._ws is not None:
            return self._ws
        try:
            import websockets.sync.client as _wsc
        except ImportError as e:
            raise ImportError(
                "VeraWebsocketClient needs the 'websockets' package. "
                "Install with: pip install websockets msgpack"
            ) from e
        try:
            headers = (
                {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
            )
            # Long open timeout matches vera's server (WAN forward passes can be slow).
            self._ws = _wsc.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=600,
                additional_headers=headers,
            )
            # Server immediately sends the packed VeraServerConfig.
            self._server_metadata = self._mnp.unpackb(self._ws.recv())
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        logger.info(
            "VeraWebsocketClient connected to %s | %s + %s | H=%s D=%s views=%s",
            self._uri,
            (self._server_metadata or {}).get("planner_model"),
            (self._server_metadata or {}).get("idm_model"),
            (self._server_metadata or {}).get("action_horizon"),
            (self._server_metadata or {}).get("action_dim"),
            (self._server_metadata or {}).get("view_keys"),
        )
        return self._ws

    # ---------------------------------------------------------------- public
    def get_server_metadata(self) -> dict[str, Any]:
        """Return the ``VeraServerConfig`` the server sent on connect."""
        self._ensure()
        return dict(self._server_metadata or {})

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send a chunk-inference request and return the server response.

        Args:
            observation: Wire dict. MUST include ``context_rgb`` ([T,H,W,3]
                uint8 or float[0,1]). SHOULD include ``view_keys``,
                ``view_widths``, ``session_id``. Optional: ``prompt``,
                ``rgb`` (defaults to last context frame), ``q_robot``,
                ``eef_pos``, ``eef_quat``, ``gripper_qpos``, per-call
                guidance overrides (``lang_guidance``, ``hist_guidance``,
                ``adaptive_gains``, ``flow_comp``), ``execute_horizon``.

        Returns:
            Response dict containing at least ``"action"`` (an ``[H, D]``
            NumPy array) and ``"info"`` (latency, absmean, ...).
        """
        ws = self._ensure()
        msg = {**observation, "endpoint": "infer"}
        try:
            ws.send(self._packer.pack(msg))
            resp = ws.recv()
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        if isinstance(resp, str):
            raise RuntimeError(f"VERA inference server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        """Per-episode reset (clears WAN context queue + IDM controller + AR cache).

        Args:
            reset_info: Optional dict with ``session_id`` and ``reason``. A new
                ``session_id`` is the contract for "this is a new episode".
        """
        ws = self._ensure()
        msg = {**(reset_info or {}), "endpoint": "reset"}
        try:
            ws.send(self._packer.pack(msg))
            resp = ws.recv()
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        if isinstance(resp, str) and resp != "reset successful":
            raise RuntimeError(f"VERA reset server error:\n{resp}")

    def configure(self, params: dict[str, Any]) -> dict[str, Any]:
        """Live-tune server runtime knobs without rebuilding the model.

        Supported keys (subset, see vera's ``VeraPolicyAdapter.infer``):
        ``motion_plan_scale``, ``sample_steps``, ``lang_guidance``,
        ``hist_guidance``, ``adaptive_gains``, ``flow_comp``,
        ``debug_dump_enabled``, ``debug_dump_dir``, ``text_conditioning``.

        Returns:
            ``{"applied": {...}}`` — the server's echo of effective values.
        """
        ws = self._ensure()
        msg = {**params, "endpoint": "configure"}
        try:
            ws.send(self._packer.pack(msg))
            resp = ws.recv()
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        if isinstance(resp, str):
            raise RuntimeError(f"VERA configure server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None
