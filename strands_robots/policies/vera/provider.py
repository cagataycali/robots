"""VERA policy provider — :class:`Policy` implementation for strands-robots.

VERA (Video-to-Embodied Robot Action, MIT/CSAIL) is a two-stage closed-loop
video-to-action policy: an embodiment-agnostic **video planner** (DFoT / WAN)
dreams future frames from the current observation, and an embodiment-specific
**Jacobian IDM** translates the dream into robot actions. Both stages live in a
websocket policy server (``vera.server.start_vera_server``); this provider is a
typed websocket client (:class:`VeraWebsocketClient`) plus an optional managed
server subprocess (:class:`VeraServerRunner`), mirroring the ``cosmos3`` service
pattern.

Observation flow
----------------
``SimEngine.get_observation`` returns a **flat** dict::

    {"<joint_name>": float, ..., "<camera_name>": np.ndarray(H, W, 3)}

VERA is *video-first*: it consumes the camera frame(s) only (proprio is read
server-side from its own sim/IDM where needed). The provider keeps a rolling
**context window** of the last ``context_frames`` camera frames (width-concat
across views, matching the server's ``view_keys`` order) and calls the server's
chunked ``infer`` when its local action queue drains — exactly the
``RemotePolicy`` contract from VERA's own eval harness.

Action flow
-----------
The server returns ``{"action": np.ndarray[H, D]}``. Actions are queued and
popped one per :meth:`get_actions` chunk request; each ``D``-vector is mapped to
robot actuator names via ``action_mapping`` (or the embodiment's default action
column names). Values are coerced to python ``float`` / ``list[float]`` per the
:class:`Policy` contract — never raw ``np.ndarray``.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import VeraWebsocketClient
from .config import VeraConfig
from .server_runner import VeraServerRunner, make_server_runner

logger = logging.getLogger(__name__)

_IMAGE_KEY_HINTS = ("image", "rgb", "cam")


def _is_image_value(value: Any) -> bool:
    """Heuristic: is this observation value a camera frame ``(H, W, 3)``?"""
    arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
    return arr.ndim == 3 and arr.shape[-1] == 3


def _to_uint8_frame(value: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous ``(H, W, 3) uint8`` array."""
    arr = np.asarray(value)
    if arr.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"camera frame must be (H, W, 3); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class VeraPolicy(Policy):
    """VERA video-to-action policy (service mode via ``vera.server``).

    Args:
        embodiment: ``"pusht"`` | ``"mimicgen"`` | ``"allegro"`` | ``"droid"``.
        server_port: Policy-server websocket port (per-embodiment default).
        vis_port: MJPEG live-viewer port; ``None`` / ``0`` disables it.
        algo_config: WAN planner ``algo_config.yaml`` (point at omni to swap).
        text_prompt: Optional text conditioning for the video planner.
        ckpt_root: Root of downloaded VERA checkpoints (``VERA_CKPT_ROOT``).
        auto_launch_server: Launch + manage the server subprocess on first use.
        n_action_steps: Deploy chunk size (actions per infer).
        dynamics_run_id: Jacobian/IDM checkpoint id (per-embodiment default).
        tracker_backend: IDM point-tracker backend override.
        motion_plan_scale: IDM motion-plan scale (applied live via ``configure``).
        host: Server hostname.
        image_keys: Explicit ordered camera keys to width-concat. When ``None``
            the server's ``view_keys`` (from the connect handshake) are used,
            matched against the observation's image keys.
        action_mapping: ``{action_column_name: robot_actuator_name}`` rename of
            the server's action columns to robot actuator names. When ``None``
            columns keep their server names (``action_0``, ``action_1``, …).
        prompt: Default instruction used when ``get_actions`` is called with an
            empty ``instruction`` and the server needs a prompt.
        client: Pre-built client (dependency injection for tests).
        server_runner: Pre-built runner (dependency injection for tests).
        config: Pre-built :class:`VeraConfig` (overrides the kwargs above).

    Notes:
        * Needs camera frames — ``requires_images`` is ``True``.
        * Latency is chunked (a diffusion video planner), not 500 Hz servo;
          one infer returns ``action_horizon`` steps.
    """

    def __init__(
        self,
        embodiment: str = "pusht",
        server_port: int | None = None,
        vis_port: int | None = None,
        algo_config: Any = None,
        text_prompt: str | None = None,
        ckpt_root: Any = None,
        auto_launch_server: bool = True,
        n_action_steps: int | None = None,
        dynamics_run_id: str | None = None,
        tracker_backend: str | None = None,
        motion_plan_scale: float | None = None,
        server_mode: str = "subprocess",
        docker_image: str | None = None,
        docker_gpus: str | None = None,
        host: str = "127.0.0.1",
        image_keys: list[str] | None = None,
        action_mapping: dict[str, str] | None = None,
        prompt: str = "",
        client: VeraWebsocketClient | None = None,
        server_runner: VeraServerRunner | None = None,
        config: VeraConfig | None = None,
    ) -> None:
        self.config = config or VeraConfig(
            embodiment=embodiment,  # type: ignore[arg-type]
            host=host,
            server_port=server_port,
            vis_port=vis_port,
            algo_config=algo_config,
            text_prompt=text_prompt,
            ckpt_root=ckpt_root,
            auto_launch_server=auto_launch_server,
            n_action_steps=n_action_steps,
            dynamics_run_id=dynamics_run_id,
            tracker_backend=tracker_backend,
            motion_plan_scale=motion_plan_scale,
            server_mode=server_mode,
            docker_image=docker_image or "strands-vera-server:latest",
            docker_gpus=docker_gpus or "all",
        )
        self.image_keys = list(image_keys) if image_keys else None
        self.action_mapping = dict(action_mapping) if action_mapping else None
        self.prompt = prompt
        self._robot_state_keys: list[str] = []

        self._client = client or VeraWebsocketClient(self.config.host, self.config.server_port)
        self._runner = server_runner
        if self._runner is None and self.config.auto_launch_server:
            self._runner = make_server_runner(self.config)

        # Episode state (mirrors VERA's RemotePolicy).
        self._server_meta: dict[str, Any] | None = None
        self._window: deque[np.ndarray] = deque()
        self._queue: deque[np.ndarray] = deque()
        self._session = str(uuid.uuid4())
        self._started = False

    # -- Policy ABC ---------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "vera"

    @property
    def requires_images(self) -> bool:
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Clear local context + queue and reset the server's episode state."""
        self._window.clear()
        self._queue.clear()
        self._session = str(uuid.uuid4())
        reset_info: dict[str, Any] = {"session_id": self._session, "reason": "eval_episode"}
        if seed is not None:
            reset_info["seed"] = int(seed)
        try:
            self._client.reset(reset_info)
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("VeraPolicy.reset best-effort failed: %s", e)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return the next VERA action chunk as ``list[dict]`` (one per step).

        Appends the current camera frame to the rolling context window and, when
        the local action queue is empty, calls the server's chunked ``infer``;
        the returned ``[H, D]`` chunk is mapped to robot actuator-name dicts.
        """
        self._ensure_started()
        meta = self._server_meta or {}

        frame = self._extract_frame(observation_dict, meta)
        ctx_max = int(meta.get("context_frames", 9))
        if self._window.maxlen != ctx_max:
            self._window = deque(self._window, maxlen=ctx_max)
        self._window.append(frame)

        if not self._queue:
            chunk = self._infer(observation_dict, instruction, meta)
            for row in chunk:
                self._queue.append(np.asarray(row, dtype=np.float32))

        if not self._queue:
            return []
        action_vec = self._queue.popleft()
        return [self._vector_to_action_dict(action_vec, meta)]

    # -- internals ----------------------------------------------------------

    def _ensure_started(self) -> None:
        """Launch the server (once) and complete the metadata handshake."""
        if self._started:
            return
        if self._runner is not None:
            self._runner.start()
        self._server_meta = self._client.get_server_metadata()
        # Apply live-tunable knobs that don't need a model rebuild.
        if self.config.motion_plan_scale is not None:
            try:
                self._client.configure({"motion_plan_scale": float(self.config.motion_plan_scale)})
            except Exception as e:  # noqa: BLE001
                logger.info("VeraPolicy live configure(motion_plan_scale) skipped: %s", e)
        self._started = True

    def _resolve_view_keys(self, observation_dict: dict[str, Any], meta: dict[str, Any]) -> list[str]:
        """Ordered camera keys to width-concat: explicit > server views > discovered."""
        if self.image_keys:
            return self.image_keys
        obs_image_keys = [k for k, v in observation_dict.items() if _is_image_value(v)]
        server_views = [str(v) for v in meta.get("view_keys", [])]
        # Match server views to observation keys when names line up; otherwise
        # fall back to the discovered image keys in dict order.
        matched = [k for k in server_views if k in observation_dict]
        if matched:
            return matched
        return obs_image_keys

    def _extract_frame(self, observation_dict: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
        """Build one width-concatenated ``(H, W, 3) uint8`` frame from all views."""
        view_keys = self._resolve_view_keys(observation_dict, meta)
        if not view_keys:
            raise ValueError(
                "VeraPolicy requires at least one camera frame in the observation "
                f"(keys: {list(observation_dict)}); none look like (H, W, 3) images."
            )
        frames = [_to_uint8_frame(observation_dict[k]) for k in view_keys]
        if len(frames) == 1:
            return frames[0]
        # Width-concat across views (server's documented layout).
        h = min(f.shape[0] for f in frames)
        frames = [f[:h] for f in frames]
        return np.ascontiguousarray(np.concatenate(frames, axis=1))

    def _infer(self, observation_dict: dict[str, Any], instruction: str, meta: dict[str, Any]) -> np.ndarray:
        """Pack the rolling context window and call the server's ``infer``."""
        view_keys = self._resolve_view_keys(observation_dict, meta)
        context_rgb = np.stack(list(self._window), axis=0)  # (T, H, W, 3) uint8
        # Per-view widths: split the concatenated width evenly across views.
        total_w = context_rgb.shape[2]
        n_views = max(1, len(view_keys))
        per_w = total_w // n_views
        view_widths = [per_w] * n_views
        req: dict[str, Any] = {
            "context_rgb": context_rgb,
            "view_keys": list(view_keys),
            "view_widths": view_widths,
            "session_id": self._session,
        }
        if meta.get("needs_prompt"):
            req["prompt"] = instruction or self.prompt or ""
        out = self._client.infer(req)
        action = np.asarray(out["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action

    def _action_column_names(self, action_dim: int, meta: dict[str, Any]) -> list[str]:
        """Names for the ``D`` action columns (mapped or server-default)."""
        names = [f"action_{i}" for i in range(action_dim)]
        if self.action_mapping:
            names = [self.action_mapping.get(n, n) for n in names]
        return names

    def _vector_to_action_dict(self, vec: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
        """Map one ``D``-vector to ``{actuator_name: float}`` (gripper binarized).

        Honours the server's ``gripper_dim_index`` + ``gripper_is_raw`` contract:
        a raw gripper float is binarized at >0.5 -> close (1.0).
        """
        vec = np.asarray(vec, dtype=np.float32).ravel()
        names = self._action_column_names(vec.shape[0], meta)
        gripper_idx = int(meta.get("gripper_dim_index", -1))
        gripper_is_raw = bool(meta.get("gripper_is_raw", True))
        out: dict[str, Any] = {}
        for i, name in enumerate(names):
            val = float(vec[i])
            if i == gripper_idx and gripper_is_raw:
                val = 1.0 if val > 0.5 else 0.0
            out[name] = val
        return out

    def close(self) -> None:
        """Close the client and stop the managed server subprocess."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._runner is not None:
            self._runner.stop()
