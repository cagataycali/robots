"""WBCLatentPolicy - a token-emitting VLA plus one SonicDecoder, run once per tick.

The blog's recipe splits control in two processes: a VLA (pi0.5) predicts a
chunk of 64-D SONIC motion tokens and two gripper commands at 50 fps, and the
robot's control loop consumes one token per 50 Hz tick, decoding it with the
SONIC decoder against the last ten frames of proprioception. The two halves
have different clocks (the VLA re-plans at 2.5 Hz, the decoder every tick),
and the decoder is a closed loop that needs the state measured AFTER the
previous target was applied. So this is one wrapper policy with
``execution_horizon = 1``: the runner calls :meth:`get_actions` every control
tick with a fresh observation; the wrapper re-queries the inner VLA only when
its token cache is used up or ``replan_every`` ticks have passed, and decodes
one token per call.

Why not :class:`~strands_robots.policies.composite.CompositePolicy`: a
composite merges two children's action dicts per tick and has no per-tick
state; the decoder must run after the chunk exists and see fresh proprio
between chunk items. One policy instance per process holds one inner VLA and
one decoder, as the rest of the family does.

The action dict this policy returns names the 29 G1 joints (hardware order,
absolute radians) plus ``left_gripper`` / ``right_gripper`` (the VLA's own
units; the Menagerie G1 has no grippers, so the sim reports them and the real
driver maps them). The MuJoCo backend detects a ``WBCLatentPolicy`` in the
policy tree and installs
:class:`~strands_robots.policies.wbc_latent.sim_control.WBCLatentTorqueController`,
which tracks those targets with SONIC's armature-derived PD gains.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.utils import refusal_repr, sequence_length

from .constants import NUM_JOINTS, SONIC_JOINT_NAMES, TOKEN_DIM
from .decoder import DecoderSession, SonicDecoder, sonic_variant_error

logger = logging.getLogger(__name__)

#: The embodiment lerobot_local needs for the blog's checkpoints: 31 state keys
#: (29 joints + 2 grippers) and the 66 action names, so ``align_action_values``
#: keeps every token instead of truncating to the state width.
INNER_EMBODIMENT = "unitree_g1_sonic"

#: Action keys the inner VLA must emit, in the order of the dataset.
TOKEN_KEYS: tuple[str, ...] = tuple(f"motion_token_{i}" for i in range(TOKEN_DIM))
GRIPPER_KEYS: tuple[str, ...] = ("left_gripper", "right_gripper")

#: Upstream VLA client: one forward pass every 0.4 s at a 50 Hz publish rate.
DEFAULT_REPLAN_EVERY = 20

_TOKEN_RE = re.compile(r"\Amotion_token_(\d+)\Z")


def _positive_int_error(value: Any, name: str) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"WBCLatentPolicy: {name} must be a positive whole number of control ticks, got {refusal_repr(value)}"
    return None


class WBCLatentPolicy(Policy):
    """Decode a VLA's SONIC motion tokens into Unitree G1 joint targets, one tick at a time.

    Args:
        inner: The policy that predicts the tokens: any :class:`Policy` whose
            action dicts carry ``motion_token_0..63`` (and, optionally,
            ``left_gripper`` / ``right_gripper``). ``None`` builds one from
            ``inner_provider`` / ``inner_config`` through ``create_policy``.
        inner_provider: Provider name for the inner policy when ``inner`` is
            ``None``; defaults to ``"lerobot_local"``.
        inner_config: Keyword arguments for the inner provider. For
            ``lerobot_local`` the ``embodiment`` defaults to
            :data:`INNER_EMBODIMENT` so all 66 action values keep their names.
        checkpoint: Decoder weights: a local ``.onnx`` file, a directory, or a
            HuggingFace repo id (``None`` = ``nvidia/GEAR-SONIC``, downloaded as
            one file on first use).
        variant: SONIC decoder variant, ``"default"`` | ``"low_latency"`` |
            ``"sonic_v1_1"``. The blog does not name the variant that encoded
            its dataset; a token decodes correctly only through its own encoder's
            decoder, so this is exposed rather than guessed silently.
        replan_every: Control ticks between inner VLA queries (default 20 =
            2.5 Hz at 50 Hz, the upstream client's rate). The cache running dry
            also triggers a query.
        decoder: A prebuilt :class:`SonicDecoder` (tests, shared sessions).
        session: A decoder session to wrap instead of loading weights.
        warn_token_abs: Passed to :class:`SonicDecoder`.

    Raises:
        ValueError: ``inner`` is not a policy, ``variant`` is unknown, or
            ``replan_every`` is not a positive whole number.
        RuntimeError: the decoder weights or ``onnxruntime`` are unavailable.
    """

    reads_instruction: ClassVar[bool] = True
    requires_action_controller: ClassVar[str | None] = (
        "wbc_latent torque control: the SONIC decoder's joint targets are tracked with its own per-joint PD "
        "gains on all 29 Unitree G1 joints; the scene's position servos would override them"
    )

    def __init__(
        self,
        inner: Policy | None = None,
        *,
        inner_provider: str | None = None,
        inner_config: dict[str, Any] | None = None,
        checkpoint: str | None = None,
        variant: str = "default",
        replan_every: int = DEFAULT_REPLAN_EVERY,
        decoder: SonicDecoder | None = None,
        session: DecoderSession | None = None,
        warn_token_abs: float | None = 1.25,
        **ignored_kwargs: Any,
    ) -> None:
        if error := sonic_variant_error(variant):
            raise ValueError(error)
        if error := _positive_int_error(replan_every, "replan_every"):
            raise ValueError(error)
        if inner is None and (inner_provider is not None or inner_config is not None):
            inner = self._build_inner(inner_provider, inner_config)
        if inner is None:
            raise ValueError(
                f"WBCLatentPolicy needs the policy that predicts the motion tokens: pass inner=<Policy>, or "
                f"inner_provider='lerobot_local' with inner_config={{'pretrained_name_or_path': <pi05 checkpoint>}} "
                f"(the embodiment defaults to {INNER_EMBODIMENT!r})."
            )
        if not isinstance(inner, Policy):
            raise ValueError(
                f"WBCLatentPolicy: inner must be a Policy that emits motion_token_0..{TOKEN_DIM - 1}, got "
                f"{type(inner).__name__}. Build it with create_policy('lerobot_local', pretrained_name_or_path="
                f"<pi05 checkpoint>, embodiment={INNER_EMBODIMENT!r}) or pass inner_provider/inner_config."
            )
        if ignored_kwargs:
            logger.debug("WBCLatentPolicy ignoring unknown kwargs: %s", sorted(ignored_kwargs))
        self._inner = inner
        self.variant = variant
        self.replan_every = int(replan_every)
        if decoder is None:
            decoder = SonicDecoder(session, checkpoint=checkpoint, variant=variant, warn_token_abs=warn_token_abs)
        self.decoder = decoder
        self._robot_state_keys: list[str] = []
        self._obs_joint_names: list[str] = list(SONIC_JOINT_NAMES)
        self._token_cache: list[np.ndarray] = []
        self._gripper_cache: list[tuple[float, float]] = []
        self._cache_pos = 0
        self._ticks_since_plan = 0
        self._last_grippers: tuple[float, float] = (0.0, 0.0)
        self.plans = 0
        self._warned_no_velocity = False
        logger.info("WBCLatentPolicy: inner=%s decoder variant=%s", inner.provider_name, variant)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_inner(provider: str | None, config: dict[str, Any] | None) -> Policy:
        from strands_robots.policies.factory import create_policy

        provider = provider or "lerobot_local"
        cfg = dict(config or {})
        if provider == "lerobot_local":
            cfg.setdefault("embodiment", INNER_EMBODIMENT)
        return create_policy(provider, **cfg)

    # ------------------------------------------------------------------
    # Policy contract
    # ------------------------------------------------------------------

    @property
    def inner(self) -> Policy:
        """The token-emitting policy this wrapper drives."""
        return self._inner

    @property
    def children(self) -> tuple[Policy, ...]:
        """The inner VLA, so tree probes (cameras, bodies, controllers) reach it."""
        return (self._inner,)

    @property
    def provider_name(self) -> str:
        """Provider name for logs and envelopes."""
        return "wbc_latent"

    @property
    def requires_images(self) -> bool:
        """Cameras are the inner VLA's business; the decoder reads proprio only."""
        return self._inner.requires_images

    @property
    def execution_horizon(self) -> int:
        """One: the decoder runs on every control tick with fresh proprioception."""
        return 1

    def is_chunk_emitting(self) -> bool:
        """False: one decoded action per call, even though the inner VLA is chunked."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's joint names and refuse a robot that is not a 29-joint G1.

        An inner VLA without an embodiment receives the 31 keys of the dataset
        (29 joints + 2 grippers) regardless of what the sim declares, because
        that is the state vector the checkpoint was normalised on; one with an
        embodiment keeps the keys the embodiment declares.

        Raises:
            ValueError: any of the 29 SONIC joint names is missing.
        """
        keys = [str(k) for k in robot_state_keys]
        missing = [n for n in SONIC_JOINT_NAMES if n not in keys]
        if missing:
            raise ValueError(
                f"WBCLatentPolicy drives the Unitree G1 (29 joints, SONIC order); the robot's state keys lack "
                f"{missing[:4]}{'...' if len(missing) > 4 else ''}. Use Robot('unitree_g1') or a scene whose joint "
                "names match the Menagerie g1.xml."
            )
        self._robot_state_keys = keys
        # An inner policy carrying an embodiment (lerobot_local with
        # unitree_g1_sonic) already declares both sides: 31 state keys and the
        # 66 action names. lerobot_local names its ACTION values by
        # robot_state_keys when no embodiment says otherwise, so forwarding the
        # 31 state keys there would rename the 66 tokens to 31 joint names and
        # drop 35 of them. Forward only to an inner without one.
        embodiment = getattr(self._inner, "_embodiment", None)
        if embodiment is None or not getattr(embodiment, "action_keys", None):
            self._inner.set_robot_state_keys([*SONIC_JOINT_NAMES, *GRIPPER_KEYS])

    def set_control_frequency(self, hz: float) -> None:
        """Set the control rate here and on the inner policy; warn when it is not 50 Hz."""
        super().set_control_frequency(hz)
        self._inner.set_control_frequency(hz)
        if abs(float(hz) - 50.0) > 1e-6:
            logger.warning(
                "WBCLatentPolicy: control_frequency=%s Hz but the SONIC decoder and the VLA's tokens are 50 Hz "
                "(one token per 20 ms). Run with control_frequency=50.",
                hz,
            )

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        """Forward the RTC delay to the inner policy (the wrapper adds no chunk of its own)."""
        super().set_rtc_observed_delay(steps)
        self._inner.set_rtc_observed_delay(steps)

    def reset(self, seed: int | None = None) -> None:
        """New episode: clear the token cache, the decoder history and the inner policy."""
        self._inner.reset(seed)
        self.decoder.reset()
        self._token_cache = []
        self._gripper_cache = []
        self._cache_pos = 0
        self._ticks_since_plan = 0
        self._last_grippers = (0.0, 0.0)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """One control tick: (maybe) re-plan tokens, decode one, return 29 joint targets + grippers.

        ``instruction`` and ``kwargs`` go to the inner VLA untouched.
        """
        q, dq, gyro, quat = self._extract_state(observation_dict)
        if self._needs_plan():
            await self._plan(observation_dict, instruction, q, **kwargs)
        token = self._token_cache[self._cache_pos]
        grippers = self._gripper_cache[self._cache_pos]
        self._cache_pos += 1
        self._ticks_since_plan += 1
        self._last_grippers = grippers
        targets = self.decoder.step(token, q, dq, gyro, quat)
        action = self.decoder.targets_dict(targets)
        action[GRIPPER_KEYS[0]] = float(grippers[0])
        action[GRIPPER_KEYS[1]] = float(grippers[1])
        return [action]

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------

    def _needs_plan(self) -> bool:
        """Re-query the VLA when the cache is empty or used up, or ``replan_every`` ticks have passed."""
        if self._cache_pos >= len(self._token_cache):
            return True
        return self._ticks_since_plan >= self.replan_every

    async def _plan(self, observation_dict: dict[str, Any], instruction: str, q: np.ndarray, **kwargs: Any) -> None:
        inner_obs = self._inner_observation(observation_dict, q)
        chunk = await self._inner.get_actions(inner_obs, instruction, **kwargs)
        tokens, grippers = self.parse_chunk(chunk, previous_grippers=self._last_grippers)
        self._token_cache = tokens
        self._gripper_cache = grippers
        self._cache_pos = 0
        self._ticks_since_plan = 0
        self.plans += 1

    def _inner_observation(self, observation_dict: dict[str, Any], q: np.ndarray) -> dict[str, Any]:
        """The observation the checkpoint expects: 29 joints + 2 grippers by name, cameras passed through.

        Flat ``observation.state`` vectors are dropped so the inner policy
        composes its 31-D state from the named keys (the sim's flat vector has
        no gripper slots and may be ordered differently).
        """
        out: dict[str, Any] = {}
        for k, v in observation_dict.items():
            if k in ("observation.state", "observation.velocity"):
                continue
            if k in SONIC_JOINT_NAMES or k.endswith(".vel"):
                continue
            out[k] = v
        for name, value in zip(SONIC_JOINT_NAMES, q, strict=True):
            out[name] = float(value)
        out[GRIPPER_KEYS[0]] = float(self._last_grippers[0])
        out[GRIPPER_KEYS[1]] = float(self._last_grippers[1])
        return out

    @staticmethod
    def parse_chunk(
        chunk: list[dict[str, Any]], *, previous_grippers: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Split the inner policy's action dicts into tokens and gripper pairs.

        Each dict must carry ``motion_token_0..63`` as floats (or one
        ``motion_token`` list of 64). Gripper keys are optional and hold the
        previous value when absent.

        Raises:
            ValueError: the chunk is empty, or a dict lacks the 64 token values.
        """
        if not chunk:
            raise ValueError("WBCLatentPolicy: the inner policy returned an empty action chunk")
        tokens: list[np.ndarray] = []
        grippers: list[tuple[float, float]] = []
        prev = previous_grippers
        for i, action in enumerate(chunk):
            if not isinstance(action, dict):
                raise ValueError(f"WBCLatentPolicy: chunk item {i} is {type(action).__name__}, expected a dict")
            packed = action.get("motion_token")
            tok: np.ndarray
            if packed is not None and sequence_length(packed) == TOKEN_DIM:
                tok = np.asarray(packed, dtype=np.float64).reshape(TOKEN_DIM)
            else:
                present = [k for k in TOKEN_KEYS if k in action]
                if len(present) != TOKEN_DIM:
                    missing = [k for k in TOKEN_KEYS if k not in action]
                    hint = ""
                    if not any(_TOKEN_RE.match(str(k)) for k in action) and len(action) in (29, 31):
                        hint = " The keys look like joint names: the inner policy is decoding to joints itself."
                    raise ValueError(
                        f"WBCLatentPolicy: chunk item {i} carries {len(present)} of {TOKEN_DIM} motion_token_* "
                        f"values (missing {missing[:3]}...). A lerobot_local inner policy needs "
                        f"embodiment={INNER_EMBODIMENT!r} so its 66 action values keep their names.{hint}"
                    )
                tok = np.asarray([float(action[k]) for k in TOKEN_KEYS], dtype=np.float64)
            if not np.all(np.isfinite(tok)):
                raise ValueError(f"WBCLatentPolicy: chunk item {i} holds a non-finite motion token")
            left = action.get(GRIPPER_KEYS[0], prev[0])
            right = action.get(GRIPPER_KEYS[1], prev[1])
            try:
                prev = (float(left), float(right))
            except (TypeError, ValueError) as e:
                raise ValueError(f"WBCLatentPolicy: chunk item {i} gripper value is not a number: {e}") from e
            tokens.append(tok)
            grippers.append(prev)
        return tokens, grippers

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _extract_state(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(q, dq, gyro, quat_wxyz) in hardware order from the unified observation.

        Reads joints BY NAME (``<joint>`` and ``<joint>.vel``), falling back to
        the flat ``observation.state`` / ``observation.velocity`` vectors
        indexed through the robot's state keys. Base signals come from
        ``base_ang_vel`` / ``base_quat`` and default to a still, upright base.
        """
        q = self._read_joint_vector(obs, "position")
        dq = self._read_joint_vector(obs, "velocity")
        gyro = self._read_vec(obs, ("base_ang_vel", "observation.base_ang_vel"), 3)
        if gyro is None:
            gyro = np.zeros(3, dtype=np.float64)
        quat = self._read_vec(obs, ("base_quat", "observation.base_quat"), 4)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        has_vel = obs.get("observation.velocity") is not None or any(f"{n}.vel" in obs for n in SONIC_JOINT_NAMES)
        if not has_vel and not self._warned_no_velocity:
            self._warned_no_velocity = True
            logger.warning(
                "WBCLatentPolicy: observation carries no joint velocities ('<joint>.vel' or "
                "'observation.velocity'); the SONIC decoder is fed zeros for dq. Balance may degrade."
            )
        return q, dq, gyro, quat

    def _read_joint_vector(self, obs: dict[str, Any], kind: str) -> np.ndarray:
        names = self._obs_joint_names
        out = np.zeros(NUM_JOINTS, dtype=np.float64)
        by_name = 0
        for i, name in enumerate(names):
            v = obs.get(name) if kind == "position" else obs.get(f"{name}.vel")
            if v is None:
                continue
            try:
                out[i] = float(v)
                by_name += 1
            except (TypeError, ValueError):
                continue
        if by_name == NUM_JOINTS:
            return out
        flat_key = "observation.state" if kind == "position" else "observation.velocity"
        flat = obs.get(flat_key)
        if flat is not None and sequence_length(flat) is not None and self._robot_state_keys:
            arr = np.asarray(flat, dtype=np.float64).ravel()
            index = {k: i for i, k in enumerate(self._robot_state_keys)}
            for i, name in enumerate(names):
                j = index.get(name)
                if j is not None and j < arr.shape[0]:
                    out[i] = arr[j]
        return out

    @staticmethod
    def _read_vec(obs: dict[str, Any], keys: tuple[str, ...], n: int) -> np.ndarray | None:
        for k in keys:
            v = obs.get(k)
            if v is not None and sequence_length(v) == n:
                return np.asarray(v if not hasattr(v, "tolist") else v.tolist(), dtype=np.float64).ravel()
        return None


__all__ = ["DEFAULT_REPLAN_EVERY", "GRIPPER_KEYS", "INNER_EMBODIMENT", "TOKEN_KEYS", "WBCLatentPolicy"]
