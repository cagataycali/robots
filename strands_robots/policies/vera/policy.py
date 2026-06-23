"""VERA policy provider — MIT's video-to-action two-stage policy via WebSocket.

Implements :class:`~strands_robots.policies.base.Policy` for the **VERA**
(Video-to-Embodied Robot Action) two-stage policy:

1. **Video planner** (WAN 14B / DFoT) — an action-free diffusion model that
   imagines future frames given a short observation history (+ optional text).
2. **Jacobian IDM** — an embodiment-specific inverse-dynamics model that
   translates the dream into a chunk of low-level actions.

Reference: https://github.com/sizhe-li/VERA · https://vera.csail.mit.edu/

Wire transport
--------------
We talk to the VERA WebSocket policy server
(``python -m vera.server.start_vera_server``) over a **self-contained**
msgpack+NumPy client — no ``vera``, no ``openpi-client``, no ``flash-attn``
on the client side. The heavy GPU stack lives entirely on the server.

Observation flow
----------------
``SimEngine.get_observation`` returns a **flat** dict::

    {"<joint_name>": float, ..., "<camera_name>": np.ndarray(H, W, 3)}

VERA's contract (per ``VeraServerConfig`` / ``VeraPolicyAdapter``) wants a
**rolling window** of width-concatenated camera frames::

    {
        "context_rgb": np.ndarray(T, H, W*sum_widths, 3) uint8,
        "view_keys":   ["ext1", "ext2", "wrist"],
        "view_widths": [128, 128, 128],
        "session_id":  "<uuid>",
        "prompt":      "<natural-language task>",
        "q_robot":     np.ndarray(D_joint,)   # optional proprio
        ...
    }

This policy maintains the **rolling context window** + **action queue** on the
client side (the same pattern VERA's own ``RemotePolicy`` uses), so each call
to :meth:`get_actions` either pops cached actions OR refills by calling the
server's chunked ``infer``.

Action flow
-----------
Server returns ``{"action": np.ndarray(H, D), "info": {...}}``. We split into
``list[dict]`` — one dict per timestep — optionally renaming the embodiment's
default action-layout column names onto real robot actuator names
(``robot="panda"`` sugar mirrors ``cosmos3``).

Example::

    from strands_robots.policies import create_policy

    # PushT — DFoT planner, loads in seconds, smallest checkpoints
    policy = create_policy("vera", embodiment="pusht", port=8820)
    chunk  = policy.get_actions_sync(obs, "push the T to the goal")

    # MimicGen — Panda arm, WAN-1.3B planner
    policy = create_policy("vera", embodiment="mimicgen", port=8800, robot="panda")
    chunk  = policy.get_actions_sync(obs, "A robot arm stacks one block on top of another")
    # chunk == [{"joint1": .., "joint2": .., ..., "finger_joint1": ..}, ...]

In MuJoCo (the ``mimicgen`` embodiment drives a Panda)::

    from strands_robots import Simulation

    sim = Simulation(tool_name="sim", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="franka")
    # ... add cube + cameras matching VERA's view_keys ...
    sim.run_policy(
        robot_name="arm",
        policy_provider="vera",
        policy_config={"embodiment": "mimicgen", "port": 8800, "robot": "panda"},
        instruction="A robot arm stacks one block on top of another block",
        n_steps=200,
        control_frequency=15.0,
    )

See ``examples/vera_sim_rollout.py`` for a complete runnable rollout +
recording. Available embodiments: ``pusht``, ``mimicgen``, ``droid``,
``allegro`` (see ``embodiments.py``).
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import VeraWebsocketClient
from .embodiments import (
    VeraEmbodiment,
    get_embodiment,
    get_robot_action_mapping,
    list_robot_action_mappings,
)

logger = logging.getLogger(__name__)


# -------------------- helpers --------------------
def _to_uint8(rgb: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous ``(H, W, 3) uint8`` array."""
    a = np.asarray(rgb)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"camera frame must be (H, W, 3); got shape {a.shape}")
    if np.issubdtype(a.dtype, np.floating):
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _hwc_concat(frames: list[np.ndarray]) -> np.ndarray:
    """Width-concatenate a list of (H, W_i, 3) frames into (H, sum_W, 3)."""
    if not frames:
        raise ValueError("cannot concat empty frame list")
    # Auto-pad / crop heights if cameras render at slightly different sizes
    # (robust to per-camera config drift). The dominant height is the median.
    heights = [f.shape[0] for f in frames]
    H = int(np.median(heights))
    out = []
    for f in frames:
        if f.shape[0] == H:
            out.append(f)
        elif f.shape[0] > H:
            out.append(f[:H, :, :])
        else:
            pad = np.zeros((H - f.shape[0], f.shape[1], 3), dtype=f.dtype)
            out.append(np.concatenate([f, pad], axis=0))
    return np.ascontiguousarray(np.concatenate(out, axis=1))


# -------------------- policy --------------------
class VeraPolicy(Policy):
    """VERA two-stage video-to-action policy (service mode via WebSocket).

    Args:
        embodiment: Embodiment key/alias — one of ``"pusht"``, ``"mimicgen"``,
            ``"droid"``, ``"allegro"`` (or their aliases). Selects the
            client-side action layout and default port/render-size.
        host: VERA policy-server hostname.
        port: VERA policy-server WebSocket port.
        observation_mapping: ``{robot_cam_key: server_view_key}`` — maps the
            robot's camera keys onto the server's ``view_keys`` order. When
            ``None``, identity mapping is used over the embodiment's default
            view_keys.
        action_mapping: ``{action_column_name: robot_actuator_name}`` — renames
            the embodiment's action-layout columns to real actuator names.
            Validated against the active layout at construction.
        robot: Convenience — name of a known robot (``"panda"``, ``"franka"``,
            ``"pusht"``) whose built-in mapping is applied when
            ``action_mapping`` is not given. Explicit ``action_mapping`` wins.
        prompt: Default instruction used when ``get_actions`` is called with an
            empty instruction string.
        api_key: Optional bearer token forwarded to the server.
        client: Pre-built ``VeraWebsocketClient`` (DI for tests).
        context_frames: Override the rolling-context window length. Default:
            read from the server's advertised ``context_frames`` (9 for WAN).
        verbose: If True, prints per-chunk denoise progress (mirrors VERA's
            own ``RemotePolicy``).
        pretrained_name_or_path: Informational only — VERA service mode picks
            the checkpoint server-side (via ``--algo-config`` /
            ``VERA_*_CKPT_DIR``). Stored for introspection.

    Notes:
        * Latency is chunked, not 500 Hz servo. One ``infer`` returns ``H``
          steps (typical: 10); we pop one per call to :meth:`get_actions`
          until empty, then refill.
        * The server is the source of truth for ``view_keys``, ``view_widths``,
          ``action_dim``, ``action_horizon``, ``control_dt``, ``action_space``;
          the embodiment table is client-side fallback only.
        * ``requires_images`` is True — VERA needs camera frames to drive the
          video planner.
    """

    def __init__(
        self,
        embodiment: str = "pusht",
        host: str = "localhost",
        port: int | None = None,
        observation_mapping: dict[str, str] | None = None,
        action_mapping: dict[str, str] | None = None,
        robot: str | None = None,
        prompt: str = "",
        api_key: str | None = None,
        client: VeraWebsocketClient | None = None,
        context_frames: int | None = None,
        verbose: bool = False,
        pretrained_name_or_path: str | None = None,
    ) -> None:
        self.embodiment: VeraEmbodiment = get_embodiment(embodiment)
        self.host = host
        self.port = int(port) if port is not None else self.embodiment.default_port
        self.default_prompt = prompt or self.embodiment.extras.get("default_prompt", "")
        self.verbose = bool(verbose)
        self.pretrained_name_or_path = pretrained_name_or_path
        if pretrained_name_or_path is not None:
            logger.info(
                "VeraPolicy: pretrained_name_or_path=%r noted. "
                "Service mode selects the checkpoint server-side (via "
                "--algo-config / VERA_*_CKPT_DIR). Ensure the server is "
                "running the expected model.",
                pretrained_name_or_path,
            )

        # ``robot=`` sugar (cosmos3-style): apply a built-in mapping unless the
        # caller supplied an explicit ``action_mapping``. Unknown robot names
        # are rejected up-front (AGENTS.md key convention #6 — no silent
        # default on error).
        if action_mapping is None and robot is not None:
            action_mapping = get_robot_action_mapping(robot)
            if action_mapping is None:
                raise ValueError(
                    f"Unknown robot {robot!r}. Available built-in mappings: "
                    f"{list_robot_action_mappings()}. Pass an explicit "
                    f"action_mapping= or omit robot=."
                )
        self._action_mapping = dict(action_mapping or {})

        # Validate action_mapping keys against the embodiment's known layout.
        # (Server's runtime ``action_dim`` is checked again at unpack time.)
        layout = set(self.embodiment.action_layout)
        bad = [k for k in self._action_mapping if k not in layout]
        if bad:
            raise ValueError(
                f"action_mapping keys {bad} are not in the "
                f"{self.embodiment.name!r} default action layout. Valid: "
                f"{sorted(layout)}"
            )

        # ``observation_mapping`` translates robot camera keys → server
        # view_keys. Default = identity over the embodiment's view_keys
        # (caller's robot is expected to expose matching camera names, or
        # provide an explicit mapping).
        self._obs_mapping = (
            dict(observation_mapping)
            if observation_mapping is not None
            else {k: k for k in self.embodiment.view_keys}
        )

        self.robot_state_keys: list[str] = []

        # Lazy client + rolling state
        self._client = client or VeraWebsocketClient(
            host=host, port=self.port, api_key=api_key
        )
        self._context_frames_override = context_frames
        self._window: deque = deque()
        self._queue: deque = deque()
        self._session_id: str = str(uuid.uuid4())
        self._chunk_count: int = 0
        # Cached server metadata (populated on first infer / get_server_metadata)
        self._server_meta: dict[str, Any] | None = None
        self._server_view_keys: list[str] | None = None
        self._server_view_widths: list[int] | None = None
        self._server_action_horizon: int | None = None
        self._server_context_frames: int | None = None

        logger.info(
            "VeraPolicy ready [embodiment=%s ws://%s:%d action_layout=%s]",
            self.embodiment.name,
            host,
            self.port,
            self.embodiment.action_layout,
        )

    # -------------------- Policy ABC --------------------
    @property
    def provider_name(self) -> str:
        return "vera"

    @property
    def requires_images(self) -> bool:
        # VERA's video planner conditions on a rolling RGB window — always.
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered joint/state keys (used as a proprio source)."""
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Per-episode reset.

        Bumps ``session_id`` (the server's per-episode key), clears the local
        context window + action queue, and sends a ``reset`` to the server.
        Optionally reseeds local RNG (server-side RNG is configured at launch
        — same caveat as ``Cosmos3Policy.reset`` / GR00T's #187).
        """
        self._window.clear()
        self._queue.clear()
        self._session_id = str(uuid.uuid4())
        self._chunk_count = 0
        try:
            self._client.reset({"session_id": self._session_id, "reason": "episode_reset"})
        except Exception:
            logger.exception("VERA reset failed (continuing — server may reject)")

        # Best-effort: reseed local RNGs the same way cosmos3 does for #187 parity.
        try:
            from strands_robots.policies._rng import reseed_client_rngs

            reseed_client_rngs(seed)
        except Exception:
            # _rng helper is best-effort; if it's missing we just continue.
            if seed is not None:
                try:
                    np.random.seed(int(seed))
                except Exception:
                    pass

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return an action chunk for the next robot tick.

        Maintains a rolling context window across calls. When the action
        queue is empty, refills by calling the server's chunked ``infer``;
        otherwise pops one cached action chunk row and returns it (the
        controller plays one row per tick).

        We honor the well-known ``Policy.get_actions`` kwargs where they make
        sense: ``execute_horizon`` forwards to the server; unknown kwargs are
        ignored (per ABC contract).
        """
        # Make sure we have server metadata so we can size the context window
        # / view-widths the way the WAN planner expects.
        if self._server_meta is None:
            self._refresh_server_metadata()

        prompt = (instruction or self.default_prompt).strip()

        # Append the current observation's view-concat frame to the rolling
        # context window.
        ctx_frame = self._build_concat_frame(observation_dict)
        max_T = int(
            self._context_frames_override
            or self._server_context_frames
            or 9
        )
        self._window.append(ctx_frame)
        while len(self._window) > max_T:
            self._window.popleft()

        # Refill the action queue from the server when empty.
        if not self._queue:
            context_rgb = np.stack(list(self._window), axis=0)  # (T,H,W,3) uint8
            req: dict[str, Any] = {
                "context_rgb": context_rgb,
                "view_keys": list(self._server_view_keys or self.embodiment.view_keys),
                "view_widths": list(
                    self._server_view_widths or self.embodiment.view_widths
                ),
                "session_id": self._session_id,
            }
            if prompt:
                req["prompt"] = prompt

            # Attach proprio when the robot has state. The server doesn't
            # *require* these but the IDM uses them when present.
            self._attach_proprio(observation_dict, req)

            # Forward well-known kwargs the server understands.
            if "execute_horizon" in kwargs:
                req["execute_horizon"] = int(kwargs["execute_horizon"])

            if self.verbose:
                T, H, W, _ = context_rgb.shape
                print(
                    f"    [VERA] chunk {self._chunk_count + 1:>3}: "
                    f"ctx={T} {H}x{W} prompt={prompt!r} — denoising on server…",
                    flush=True,
                )

            response = self._client.infer(req)
            action = np.asarray(response["action"], dtype=np.float32)
            if action.ndim == 3:
                # Server uses (B, H, D) for batched but the protocol is single.
                action = action[0]
            if action.ndim != 2:
                raise RuntimeError(
                    f"VERA server returned malformed action with shape "
                    f"{action.shape}; expected (H, D)"
                )
            for row in action:
                self._queue.append(row)
            self._chunk_count += 1

            if self.verbose:
                info = response.get("info") or {}
                print(
                    f"    [VERA] chunk {self._chunk_count} done "
                    f"({info.get('infer_s', 0.0):.2f}s, |a|={info.get('action_absmean', 0.0):.3f})",
                    flush=True,
                )

        # Pop ONE row and return as a single-element list-of-dicts (matches
        # other strands-robots policies that drive one tick at a time —
        # callers can call repeatedly to drain the chunk).
        row = self._queue.popleft()
        return [self._unpack_row(row)]

    # -------------------- helpers --------------------
    def _refresh_server_metadata(self) -> None:
        """Fetch server-advertised view/action contract on first use."""
        try:
            meta = self._client.get_server_metadata() or {}
        except Exception:
            logger.exception("VERA: get_server_metadata failed — using embodiment defaults")
            meta = {}
        self._server_meta = meta
        self._server_view_keys = list(meta.get("view_keys") or self.embodiment.view_keys)
        self._server_view_widths = list(
            meta.get("view_widths") or self.embodiment.view_widths
        )
        self._server_action_horizon = int(meta.get("action_horizon") or 0) or None
        self._server_context_frames = int(meta.get("context_frames") or 0) or None
        if meta:
            logger.info(
                "VERA server contract: views=%s widths=%s H=%s D=%s ctx=%s control_dt=%s",
                self._server_view_keys,
                self._server_view_widths,
                self._server_action_horizon,
                meta.get("action_dim"),
                self._server_context_frames,
                meta.get("control_dt"),
            )

    def _build_concat_frame(self, obs: dict[str, Any]) -> np.ndarray:
        """Build the per-step width-concatenated camera frame the server expects."""
        view_keys = list(self._server_view_keys or self.embodiment.view_keys)
        # Map robot-side camera keys onto server view_keys via obs_mapping.
        # If the robot already uses the server view_keys, the identity mapping
        # configured in __init__ keeps the obvious path working.
        frames: list[np.ndarray] = []
        missing: list[str] = []
        for vk in view_keys:
            # Find a robot key that maps to this view_key (prefer explicit
            # mapping entries; fall back to identity).
            robot_key = next(
                (rk for rk, sk in self._obs_mapping.items() if sk == vk),
                vk,
            )
            if robot_key in obs:
                frames.append(_to_uint8(obs[robot_key]))
            else:
                missing.append(f"{vk!r} (looked up via {robot_key!r})")
        if not frames:
            raise ValueError(
                f"VeraPolicy requires at least one camera frame, but none of "
                f"the expected views {view_keys} were found in the "
                f"observation. Available keys: {sorted(obs)}. "
                f"Configure observation_mapping= to map robot camera keys "
                f"onto these view_keys."
            )
        if missing and len(frames) < len(view_keys):
            logger.warning(
                "VeraPolicy: missing camera views %s — proceeding with %d of %d views. "
                "This may degrade the video planner's quality.",
                missing,
                len(frames),
                len(view_keys),
            )
        return _hwc_concat(frames)

    def _attach_proprio(self, obs: dict[str, Any], req: dict[str, Any]) -> None:
        """Best-effort: attach proprio (q_robot / eef_pos / gripper) when present.

        Names follow VERA's ``PolicyObservation`` contract; the IDM uses these
        when available but does not require them (the wire schema is permissive).
        """
        # Direct keys passed through if the user named them this way.
        for key in ("q_robot", "eef_pos", "eef_quat", "gripper_qpos"):
            if key in obs:
                req[key] = np.asarray(obs[key])

        # Fall back to robot_state_keys: gather the first 7 joint-like scalars
        # as q_robot, and any 'gripper'/'finger' key as gripper_qpos. Mirrors
        # the cosmos3 ``_attach_joint_state`` heuristic.
        if "q_robot" not in req and self.robot_state_keys:
            joints: list[float] = []
            gripper: float | None = None
            for k in self.robot_state_keys:
                if k not in obs:
                    continue
                v = np.asarray(obs[k])
                if v.ndim == 0 or v.size == 1:
                    fval = float(v.reshape(-1)[0])
                    low = k.lower()
                    if "gripper" in low or "finger" in low:
                        if gripper is None:
                            gripper = fval
                    elif len(joints) < 16:
                        joints.append(fval)
            if joints:
                req["q_robot"] = np.asarray(joints, dtype=np.float32)
            if gripper is not None and "gripper_qpos" not in req:
                req["gripper_qpos"] = np.asarray([gripper], dtype=np.float32)

    def _unpack_row(self, row: np.ndarray) -> dict[str, Any]:
        """Split one ``(D,)`` action row into an actuator dict.

        Uses the server's advertised ``action_dim`` when available; falls back
        to the embodiment's static layout otherwise. Unknown columns are
        emitted as ``action_<i>`` to surface drift rather than silently dropping.
        """
        width = int(row.shape[0])
        layout = list(self.embodiment.action_layout)
        col_names = list(layout[:width])
        for i in range(len(col_names), width):
            col_names.append(f"action_{i}")
        out_names = [self._action_mapping.get(name, name) for name in col_names]
        return {out_names[d]: float(row[d]) for d in range(width)}

    # -------------------- introspection --------------------
    def get_server_metadata(self) -> dict[str, Any]:
        """Expose the ``VeraServerConfig`` for the live server (refresh + return)."""
        self._refresh_server_metadata()
        return dict(self._server_meta or {})
