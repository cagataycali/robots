"""Cosmos 3 policy — NVIDIA omnimodal VLA policy via Cosmos Framework.

Implements :class:`~strands_robots.policies.base.Policy` for the Cosmos 3
**Generator action surface** (``nvidia/Cosmos3-Nano-Policy-DROID`` and friends).

The Cosmos 3 ``policy`` action mode takes ``image + instruction`` and returns an
``[T, D]`` action chunk + rollout video — a 1:1 match for the robots policy
contract. We talk to the Cosmos Framework RoboLab WebSocket policy server
(``cosmos_framework.scripts.action_policy_server_robolab``) over OpenPI's
msgpack+NumPy protocol, mirroring :class:`~strands_robots.policies.groot.Gr00tPolicy`
service mode.

Observation flow
----------------
The robots ``SimEngine.get_observation`` returns a **flat** dict::

    {"<joint_name>": float, ..., "<camera_name>": np.ndarray(H, W, 3)}

We translate that into the server's OpenPI observation::

    {
        "prompt": instruction,
        "observation/wrist_image_left":      np.ndarray(H, W, 3) uint8,
        "observation/exterior_image_1_left": ...,
        "observation/exterior_image_2_left": ...,
        "observation/joint_position":  np.ndarray(1, 7) float32,
        "observation/gripper_position": np.ndarray(1, 1) float32,
    }

via an explicit ``observation_mapping`` (robot key → server key), with a
sensible auto-mapping fallback.

Action flow
-----------
The server returns ``{"action": np.ndarray(T, D)}``. Each of the ``D`` columns
is named by the embodiment's ``action_layout`` (e.g. DROID joint_pos =
``[joint_0..joint_6, gripper]``). We emit ``list[dict]`` — one dict per
timestep — optionally remapping column names to robot actuator names via
``action_mapping``.

Example::

    from strands_robots.policies import create_policy

    policy = create_policy(
        "cosmos3",
        embodiment="droid",
        host="localhost",
        port=8000,
    )
    chunk = policy.get_actions_sync(observation, "pick up the cube")
    # chunk == [{"joint_0": .., ..., "gripper": ..}, ...]  (one per timestep)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import Cosmos3WebsocketClient
from .embodiments import Cosmos3Embodiment, get_embodiment

logger = logging.getLogger(__name__)


def _to_image_uint8(value: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous ``(H, W, 3) uint8`` array."""
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"camera frame must be (H, W, 3); got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class Cosmos3Policy(Policy):
    """NVIDIA Cosmos 3 VLA policy (service mode via Cosmos Framework).

    Args:
        embodiment: Embodiment key/alias (``"droid"``, ``"umi"``, ``"av"``,
            ``"bridge"``). Selects domain, action layout, and defaults.
        host: Policy-server hostname.
        port: Policy-server WebSocket port.
        action_space: ``"joint_pos"`` or ``"midtrain"`` — must match how the
            server was launched (DROID default = ``joint_pos``).
        observation_mapping: ``{robot_obs_key: "observation/<server_key>"}``.
            Maps robot camera + state keys onto the server's OpenPI keys.
            When ``None``, a default mapping is used (see :meth:`_default_obs_mapping`).
        action_mapping: ``{action_column_name: robot_actuator_name}``. Renames
            the embodiment's action-layout columns to robot actuator names.
            When ``None``, columns keep their layout names.
        prompt: Default instruction used when ``get_actions`` is called with an
            empty instruction.
        api_key: Optional bearer token for the server.
        client: Pre-built client (dependency injection for tests).

    Notes:
        * This policy needs camera frames **and** robot state in the
          observation — ``requires_images`` is ``True``.
        * Latency is chunked (a diffusion policy), not 500 Hz servo. One
          inference returns a chunk of ~``action_chunk_size`` steps.
    """

    def __init__(
        self,
        embodiment: str = "droid",
        host: str = "localhost",
        port: int = 8000,
        action_space: str | None = None,
        observation_mapping: dict[str, str] | None = None,
        action_mapping: dict[str, str] | None = None,
        prompt: str = "",
        api_key: str | None = None,
        client: Cosmos3WebsocketClient | None = None,
        **kwargs: Any,
    ) -> None:
        self.embodiment: Cosmos3Embodiment = get_embodiment(embodiment)
        self.host = host
        self.port = port
        self.action_space = action_space or self.embodiment.default_action_space
        if self.action_space not in self.embodiment.action_layouts:
            raise ValueError(
                f"embodiment {self.embodiment.name!r} has no action_space "
                f"{self.action_space!r}; available: {sorted(self.embodiment.action_layouts)}"
            )
        self.default_prompt = prompt
        self._obs_mapping = observation_mapping or self._default_obs_mapping()
        self._action_mapping = action_mapping or {}
        self.robot_state_keys: list[str] = []
        self._client = client or Cosmos3WebsocketClient(host=host, port=port, api_key=api_key)
        logger.info(
            "Cosmos3Policy ready [embodiment=%s domain=%s action_space=%s chunk=%d ws://%s:%d]",
            self.embodiment.name, self.embodiment.domain_name, self.action_space,
            self.embodiment.action_chunk_size, host, port,
        )

    # ── Policy interface ──────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "cosmos3"

    @property
    def requires_images(self) -> bool:
        """Cosmos 3 conditions on camera frames — always needs images."""
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered joint/state keys.

        Used (a) as the fallback gripper/joint source when no explicit
        ``observation_mapping`` names them, and (b) as default action actuator
        names when no ``action_mapping`` is supplied and the layout is generic.
        """
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Per-episode reset — forward a best-effort hint to the server."""
        self._client.reset()
        if seed is not None:
            try:
                np.random.seed(seed)
            except Exception:  # noqa: BLE001
                pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Query Cosmos 3 for an action chunk.

        Args:
            observation_dict: Flat robots observation (joint floats + camera
                ndarrays), per the ``SimEngine.get_observation`` schema.
            instruction: Natural-language task instruction.

        Returns:
            ``list[dict]`` — one action dict per predicted timestep.
        """
        prompt = instruction or self.default_prompt
        obs = self._build_server_observation(observation_dict, prompt)
        result = self._client.infer(obs)
        action = np.asarray(result["action"])
        return self._unpack_actions(action)

    # ── Observation translation ───────────────────────────────────────────

    def _default_obs_mapping(self) -> dict[str, str]:
        """Identity-ish default: assume robot obs already uses server keys.

        Falls back to mapping the embodiment's expected camera keys onto
        themselves; callers with differently-named robot cameras should pass an
        explicit ``observation_mapping``.
        """
        return {k: k for k in self.embodiment.camera_keys}

    def _build_server_observation(
        self, robot_obs: dict[str, Any], prompt: str
    ) -> dict[str, Any]:
        """Translate the flat robot observation into the server's OpenPI dict."""
        obs: dict[str, Any] = {"prompt": prompt}

        # Images: map robot camera keys → server image keys.
        for robot_key, server_key in self._obs_mapping.items():
            if robot_key in robot_obs and server_key.startswith("observation/"):
                val = robot_obs[robot_key]
                if isinstance(val, np.ndarray) or hasattr(val, "__array__"):
                    arr = np.asarray(val)
                    if arr.ndim == 3:
                        obs[server_key] = _to_image_uint8(arr)
                        continue
                # Non-image mapped values pass straight through (e.g. state).
                obs[server_key] = val

        # State for joint_pos action space: joint_position (1,7) + gripper (1,1).
        if self.action_space == "joint_pos":
            self._attach_joint_state(robot_obs, obs)

        return obs

    def _attach_joint_state(self, robot_obs: dict[str, Any], obs: dict[str, Any]) -> None:
        """Build ``observation/joint_position`` + ``observation/gripper_position``.

        Priority:
            1. Explicit keys already present in robot_obs / via obs_mapping.
            2. ``robot_state_keys`` (first 7 = joints, a 'gripper'-named key).
        """
        if "observation/joint_position" in obs and "observation/gripper_position" in obs:
            return  # already provided via mapping

        joints: list[float] = []
        gripper: float | None = None

        # Use declared state-key order when available.
        state_keys = self.robot_state_keys or [
            k for k, v in robot_obs.items() if np.isscalar(v) or np.ndim(v) == 0
        ]
        for k in state_keys:
            if k not in robot_obs:
                continue
            if "gripper" in k.lower() and gripper is None:
                gripper = float(np.asarray(robot_obs[k]).reshape(-1)[0])
            elif len(joints) < 7:
                joints.append(float(np.asarray(robot_obs[k]).reshape(-1)[0]))

        if len(joints) >= 7 and "observation/joint_position" not in obs:
            obs["observation/joint_position"] = np.asarray(joints[:7], dtype=np.float32).reshape(1, 7)
        if gripper is not None and "observation/gripper_position" not in obs:
            obs["observation/gripper_position"] = np.asarray([[gripper]], dtype=np.float32)

    # ── Action unpacking ──────────────────────────────────────────────────

    def _action_column_names(self, width: int) -> list[str]:
        """Resolve the per-column action names for the active action space."""
        layout = self.embodiment.action_layouts.get(self.action_space, [])
        names = list(layout[:width])
        # Pad / fall back if the server returns a different width than expected.
        for i in range(len(names), width):
            names.append(f"action_{i}")
        return names

    def _unpack_actions(self, action: np.ndarray) -> list[dict[str, Any]]:
        """Split an ``[T, D]`` chunk into per-timestep actuator dicts."""
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2:
            raise ValueError(f"expected action chunk [T, D]; got shape {action.shape}")

        horizon, width = action.shape
        col_names = self._action_column_names(width)
        # Apply optional rename: layout column name → robot actuator name.
        out_names = [self._action_mapping.get(name, name) for name in col_names]

        steps: list[dict[str, Any]] = []
        for t in range(horizon):
            row = action[t]
            steps.append({out_names[d]: float(row[d]) for d in range(width)})
        return steps
