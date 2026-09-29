"""SonicDecoder - one SONIC motion token + proprio history -> 29 joint targets.

This is the on-robot half of NVIDIA's SONIC stack (``nvidia/GEAR-SONIC``,
``model_decoder.onnx``): a policy network trained in Isaac Lab that reads a
64-D latent motion token and the last ten 50 Hz frames of the robot's own
state and emits one joint-position offset per Unitree G1 joint. The reference
implementation is the C++ deploy loop in NVlabs/GR00T-WholeBodyControl
(``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref``); this module reproduces its
observation assembly and action law in NumPy so the same ONNX file drives our
MuJoCo twin and, later, the real G1 driver.

The decoder is deliberately a plain object with no inheritance from
:class:`~strands_robots.policies.base.Policy`: it is not a policy (it has no
instruction, no cameras, and needs a token somebody else produced). The
:class:`~strands_robots.policies.wbc_latent.policy.WBCLatentPolicy` owns one
and feeds it the tokens a VLA predicts.

Contract (all arrays in HARDWARE order, see :mod:`.constants`):

* :meth:`step` ``(token[64], q[29], dq[29], gyro[3], quat_wxyz[4]) -> targets[29]``
  once per 50 Hz control tick; radians, rad/s, rad/s body frame, scalar-first
  quaternion of the pelvis in the world frame.
* :meth:`reset` clears the histories (a new episode, a teleport).
* ``session=`` accepts any object with ``run(None, {"obs_dict": array})``
  returning ``[array[1, 29]]``, so tests inject a stub and never touch the
  network or onnxruntime.

Weights are never bundled. A checkpoint that is not a local file or directory
is treated as a HuggingFace repo id and exactly one file
(``<variant prefix>model_decoder.onnx``) is fetched through
``huggingface_hub.hf_hub_download`` into the HF cache; the repo also holds
37 GB of motion data that a snapshot download would pull. The weights are
licensed under the NVIDIA Open Model License, whose section 3.b asks for an
attribution notice; it is logged once per process on load.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from strands_robots.policies.wbc.control import projected_gravity
from strands_robots.utils import refusal_repr, require_optional

from .constants import (
    HARDWARE_TO_ISAACLAB,
    ISAACLAB_TO_HARDWARE,
    NUM_JOINTS,
    OBS_DIM,
    OBS_LAYOUT,
    SONIC_ACTION_SCALE,
    SONIC_DEFAULT_ANGLES,
    SONIC_JOINT_NAMES,
    SONIC_KDS,
    SONIC_KPS,
    TOKEN_DIM,
    TOKEN_WARN_ABS,
)

logger = logging.getLogger(__name__)

#: The HuggingFace repo that publishes the decoder.
SONIC_REPO_ID = "nvidia/GEAR-SONIC"

#: Variant name -> path of its decoder inside the repo (config.json ``variants``).
SONIC_VARIANT_FILES: dict[str, str] = {
    "default": "model_decoder.onnx",
    "low_latency": "low_latency/model_decoder.onnx",
    "sonic_v1_1": "sonic_v1_1/model_decoder.onnx",
}

_DECODER_FILENAME = "model_decoder.onnx"
_INPUT_NAME = "obs_dict"

# NVIDIA Open Model License, section 3.b. Logged once per process when a
# session is built from the published weights.
_ATTRIBUTION = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License."
_attribution_logged = False
_attribution_lock = threading.Lock()


class DecoderSession(Protocol):
    """What :class:`SonicDecoder` needs from an ONNX session (or a stub)."""

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> Any:  # pragma: no cover - protocol
        """Run the graph on ``input_feed`` and return its outputs as a list."""
        ...


def sonic_variant_error(variant: Any) -> str | None:
    """Reason ``variant`` is not one of the published SONIC decoder variants, or ``None``."""
    if not isinstance(variant, str) or variant not in SONIC_VARIANT_FILES:
        return (
            f"SonicDecoder: variant must be one of {sorted(SONIC_VARIANT_FILES)}, got {refusal_repr(variant)}. "
            "The variants are the entries of nvidia/GEAR-SONIC config.json; a VLA's tokens decode "
            "correctly only through the variant that encoded its training data."
        )
    return None


def resolve_decoder_path(checkpoint: str | Path | None, variant: str = "default") -> Path:
    """Turn ``checkpoint`` into the local path of a decoder ONNX file.

    * an existing ``.onnx`` file is returned as is;
    * an existing directory is searched for ``<variant path>`` then
      ``model_decoder.onnx`` at its root;
    * anything else is read as a HuggingFace repo id (``None`` means
      :data:`SONIC_REPO_ID`) and the single decoder file of ``variant`` is
      downloaded into the HF cache.

    Raises:
        ValueError: ``variant`` is unknown.
        FileNotFoundError: a local directory holds no decoder file.
        RuntimeError: ``huggingface_hub`` is missing, or the download failed.
    """
    if error := sonic_variant_error(variant):
        raise ValueError(error)
    rel = SONIC_VARIANT_FILES[variant]
    if checkpoint is not None:
        p = Path(checkpoint).expanduser()
        if p.is_file():
            return p
        if p.is_dir():
            for candidate in (p / rel, p / _DECODER_FILENAME):
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                f"SonicDecoder: {p} holds neither {rel} nor {_DECODER_FILENAME}. Point checkpoint at the "
                f"directory that contains the nvidia/GEAR-SONIC decoder, at the .onnx file itself, or at "
                f"the repo id {SONIC_REPO_ID!r} to download it."
            )
    repo_id = str(checkpoint) if checkpoint is not None else SONIC_REPO_ID
    try:
        hub = require_optional(
            "huggingface_hub",
            pip_install="huggingface_hub",
            extra="wbc",
            purpose="SonicDecoder weight download",
        )
    except ImportError as e:
        raise RuntimeError(
            f"SonicDecoder: checkpoint {repo_id!r} is not a local path and huggingface_hub is not "
            f"installed to download it. Install the [wbc] extra or pass a local .onnx path.\n{e}"
        ) from e
    logger.info("SonicDecoder fetching %s from %s (cached after first use)", rel, repo_id)
    try:
        local = hub.hf_hub_download(repo_id=repo_id, filename=rel)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 - every hub failure becomes one actionable error
        raise RuntimeError(
            f"SonicDecoder: could not download {rel} from {repo_id!r}: {e}. Check network access or pass "
            "a local .onnx path."
        ) from e
    return Path(local)


def build_onnx_session(path: str | Path) -> DecoderSession:
    """Open ``path`` with onnxruntime on CPU and check the graph is a SONIC decoder.

    Raises:
        RuntimeError: ``onnxruntime`` is not installed ([wbc] extra), or the
            graph's input is not ``obs_dict [1, 994]`` / output ``[1, 29]``.
    """
    try:
        ort = require_optional(
            "onnxruntime",
            pip_install="onnxruntime",
            extra="wbc",
            purpose="SonicDecoder ONNX inference",
        )
    except ImportError as e:
        raise RuntimeError(f"SonicDecoder requires onnxruntime (the [wbc] extra) but it is not installed.\n{e}") from e
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # type: ignore[attr-defined]
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    names = [i.name for i in inputs]
    if len(inputs) != 1 or inputs[0].name != _INPUT_NAME or list(inputs[0].shape) != [1, OBS_DIM]:
        raise RuntimeError(
            f"SonicDecoder: {path} is not a SONIC decoder graph. Expected one input {_INPUT_NAME!r} of shape "
            f"[1, {OBS_DIM}], got {names} with shapes {[list(i.shape) for i in inputs]}. The GEAR-SONIC repo "
            "also ships model_encoder.onnx and planner_sonic.onnx; this provider loads the decoder only."
        )
    if len(outputs) != 1 or list(outputs[0].shape) != [1, NUM_JOINTS]:
        raise RuntimeError(
            f"SonicDecoder: {path} emits {[list(o.shape) for o in outputs]}, expected one output of shape "
            f"[1, {NUM_JOINTS}] (one offset per G1 joint)."
        )
    global _attribution_logged
    with _attribution_lock:
        if not _attribution_logged:
            _attribution_logged = True
            logger.info("SonicDecoder weights: %s (%s)", _ATTRIBUTION, SONIC_REPO_ID)
    return session


class SonicDecoder:
    """Stateful SONIC token decoder (see the module docstring for the contract).

    Args:
        session: A decoder session (``run(None, {"obs_dict": x})``). ``None``
            builds one from ``checkpoint``/``variant`` with onnxruntime.
        checkpoint: Local ``.onnx`` file, directory, or HuggingFace repo id;
            ``None`` means :data:`SONIC_REPO_ID`. Ignored when ``session`` is given.
        variant: ``"default"`` | ``"low_latency"`` | ``"sonic_v1_1"``.
        warn_token_abs: Log a warning once when a token component exceeds this
            magnitude, as the upstream VLA client does at 1.25. ``None`` disables.

    Attributes:
        joint_names: The 29 joint names in hardware order.
        kps, kds: Per-joint PD gains the decoder's targets are meant to be
            tracked with (armature-derived, hardware order).
        default_angles: The stance the offsets are added to.
        last_targets: Targets returned by the most recent :meth:`step`, or ``None``.
        last_latency_s: Wall-clock seconds of the most recent session call.
    """

    joint_names: tuple[str, ...] = SONIC_JOINT_NAMES
    kps: np.ndarray = SONIC_KPS
    kds: np.ndarray = SONIC_KDS
    default_angles: np.ndarray = SONIC_DEFAULT_ANGLES
    action_scale: np.ndarray = SONIC_ACTION_SCALE

    def __init__(
        self,
        session: DecoderSession | None = None,
        *,
        checkpoint: str | Path | None = None,
        variant: str = "default",
        warn_token_abs: float | None = TOKEN_WARN_ABS,
    ) -> None:
        if error := sonic_variant_error(variant):
            raise ValueError(error)
        self.variant = variant
        self.path: Path | None = None
        if session is None:
            self.path = resolve_decoder_path(checkpoint, variant)
            session = build_onnx_session(self.path)
        self._session = session
        self._warn_token_abs = None if warn_token_abs is None else float(warn_token_abs)
        self._warned_token = False
        self._hw_to_il = np.asarray(HARDWARE_TO_ISAACLAB, dtype=int)
        self._il_to_hw = np.asarray(ISAACLAB_TO_HARDWARE, dtype=int)
        self.last_targets: np.ndarray | None = None
        self.last_latency_s: float = 0.0
        self.ticks: int = 0
        self._rings: dict[str, deque[np.ndarray]] = {}
        self.reset()

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the proprio and action history (zero padding, as the deploy loop starts)."""
        self._rings = {
            name: deque([np.zeros(width, dtype=np.float64) for _ in range(frames)], maxlen=frames)
            for name, frames, width in OBS_LAYOUT
            if frames > 1
        }
        self.last_targets = None
        self.ticks = 0

    def _push(self, name: str, value: np.ndarray) -> None:
        self._rings[name].append(np.asarray(value, dtype=np.float64))

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def build_observation(
        self,
        token: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Log this tick's state and assemble the 994-D decoder input.

        Histories are oldest first with zero padding before ten ticks have
        been logged (``StateLogger::GetLatest`` in the deploy loop). Joint
        positions enter in IsaacLab order with the default angles subtracted;
        velocities in IsaacLab order; the last-action history holds the raw
        network output of previous ticks in network order.
        """
        token = self._check(token, TOKEN_DIM, "token")
        q = self._check(q, NUM_JOINTS, "q")
        dq = self._check(dq, NUM_JOINTS, "dq")
        gyro = self._check(gyro, 3, "gyro")
        quat_wxyz = self._check(quat_wxyz, 4, "quat_wxyz")
        if self._warn_token_abs is not None and not self._warned_token:
            peak = float(np.max(np.abs(token)))
            if peak > self._warn_token_abs:
                self._warned_token = True
                logger.warning(
                    "SonicDecoder: token component magnitude %.3f exceeds %.2f; SONIC tokens are FSQ codes in "
                    "[-1, 1], so the VLA is extrapolating (further such tokens are not reported).",
                    peak,
                    self._warn_token_abs,
                )
        body_q = q[self._hw_to_il] - self.default_angles[self._hw_to_il]
        body_dq = dq[self._hw_to_il]
        self._push("his_base_angular_velocity_10frame_step1", gyro)
        self._push("his_body_joint_positions_10frame_step1", body_q)
        self._push("his_body_joint_velocities_10frame_step1", body_dq)
        self._push("his_gravity_dir_10frame_step1", projected_gravity(quat_wxyz))
        parts: list[np.ndarray] = []
        for name, frames, _width in OBS_LAYOUT:
            if frames == 1:
                parts.append(token)
            else:
                parts.append(np.concatenate(list(self._rings[name])))
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (OBS_DIM,):  # pragma: no cover - layout is a module constant
            raise RuntimeError(f"SonicDecoder: assembled {obs.shape[0]} values, expected {OBS_DIM}")
        return obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        token: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        gyro: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Decode one tick: returns 29 absolute joint targets in hardware order (rad).

        ``target[i] = default[i] + action[isaaclab_to_hardware[i]] * scale[i]``
        (``g1_deploy_onnx_ref.cpp`` ``CreatePolicyCommand``). The raw action is
        appended to the last-action history AFTER the observation was built, so
        the network sees actions of previous ticks only, as on the robot.

        Raises:
            ValueError: an input has the wrong length or is not finite.
            RuntimeError: the session returned a shape other than ``[1, 29]`` or non-finite values.
        """
        obs = self.build_observation(token, q, dq, gyro, quat_wxyz)
        t0 = time.perf_counter()
        out = self._session.run(None, {_INPUT_NAME: obs[None, :]})
        self.last_latency_s = time.perf_counter() - t0
        action = np.asarray(out[0], dtype=np.float64).reshape(-1)
        if action.shape != (NUM_JOINTS,):
            raise RuntimeError(f"SonicDecoder: session returned {np.asarray(out[0]).shape}, expected (1, {NUM_JOINTS})")
        if not np.all(np.isfinite(action)):
            raise RuntimeError("SonicDecoder: session returned non-finite values; refusing to command the joints")
        self._push("his_last_actions_10frame_step1", action)
        targets = self.default_angles + action[self._il_to_hw] * self.action_scale
        self.last_targets = targets
        self.ticks += 1
        return targets

    def targets_dict(self, targets: np.ndarray | None = None) -> dict[str, float]:
        """The targets as ``{joint_name: float}`` (hardware order), python floats."""
        t = self.last_targets if targets is None else targets
        if t is None:
            return {}
        return {name: float(v) for name, v in zip(self.joint_names, t, strict=True)}

    @staticmethod
    def _check(value: Any, n: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (n,):
            raise ValueError(f"SonicDecoder: {name} must have {n} values, got shape {np.asarray(value).shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"SonicDecoder: {name} contains non-finite values")
        return arr


__all__ = [
    "SONIC_REPO_ID",
    "SONIC_VARIANT_FILES",
    "DecoderSession",
    "SonicDecoder",
    "build_onnx_session",
    "resolve_decoder_path",
    "sonic_variant_error",
]
