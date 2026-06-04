"""Qwen-VLA policy - unified VLA inference (LOCAL + SERVICE).

Implements :class:`~strands_robots.policies.base.Policy` for the Qwen-VLA
model (Qwen3.5-4B VLM backbone + 1.15B DiT flow-matching action expert,
arXiv:2605.30280v2).

Design mirrors :class:`~strands_robots.policies.groot.policy.Gr00tPolicy`:

* Explicit ``observation_mapping`` / ``action_mapping`` (robot sensor/actuator
  names <-> model modality keys), with an auto-infer fallback. No positional
  guessing of action semantics.
* Two inference modes: **LOCAL** (in-proc model load, requires the qwen-vla
  package) and **SERVICE** (ZMQ to a running server, no model deps needed).
* ``reset(seed=)`` forwards to the server in SERVICE mode and reseeds RNG in
  LOCAL mode (the #187 reproducibility contract).

The Qwen-VLA-specific surface (vs GR00T):

* The **embodiment prompt** (section 2.3) is the sole platform-specific input;
  built from :class:`QwenVlaDataConfig` and concatenated with the instruction
  before being handed to the model.
* Model output is a single fixed-width chunk ``Y[H x K]`` with a per-channel
  validity mask; we unpack the leading ``c`` channels per the action_mapping
  (section 2.4 zero-padding layout).
* Inference runs a few Euler steps of the flow-matching ODE (LOCAL); the
  number of steps is exposed via ``denoising_steps``.
"""

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import QwenVlaInferenceClient
from .data_config import QwenVlaDataConfig, load_data_config

logger = logging.getLogger(__name__)


def _qwen_vla_installed() -> bool:
    """Return True iff the local Qwen-VLA inference package is importable.

    The exact upstream package name is unconfirmed at integration time (PLAN
    section 6.2), so we probe a small set of candidate module names. LOCAL
    mode is gated on this; SERVICE mode never needs it.
    """
    for candidate in ("qwen_vla", "qwenvla"):
        try:
            if importlib.util.find_spec(candidate) is not None:
                return True
        except (ModuleNotFoundError, ValueError):
            continue
    return False


# Mapping dataclasses (mirror GR00T's explicit-mapping machinery)


@dataclass(frozen=True)
class ObservationMapping:
    """Maps robot sensor names -> model modality keys.

    Attributes:
        video: ``{robot_camera: model_video_key}`` (bare, no prefix).
        state: ``{robot_state: model_state_key}`` (bare, no prefix).
        language_key: Model's language key (e.g. ``"task"``).
    """

    video: dict[str, str] = field(default_factory=dict)
    state: dict[str, str] = field(default_factory=dict)
    language_key: str = "task"


@dataclass(frozen=True)
class ActionMapping:
    """Maps model action keys -> robot actuator names.

    Attributes:
        actions: ``{model_action_key: robot_actuator}`` (bare, no prefix).
            Insertion order defines the channel layout for the unified
            ``Y[H x K]`` unpack: the first key occupies the leading channels.
    """

    actions: dict[str, str] = field(default_factory=dict)


def _parse_observation_mapping(flat: dict[str, str], language_key: str = "task") -> ObservationMapping:
    """Parse ``{robot_key: "video.X" | "state.X"}`` -> ObservationMapping."""
    video: dict[str, str] = {}
    state: dict[str, str] = {}
    for robot_key, model_key in flat.items():
        if model_key.startswith("video."):
            video[robot_key] = model_key.removeprefix("video.")
        elif model_key.startswith("state."):
            state[robot_key] = model_key.removeprefix("state.")
        else:
            raise ValueError(f"Mapping value must start with 'video.' or 'state.', got '{model_key}' for '{robot_key}'")
    return ObservationMapping(video=video, state=state, language_key=language_key)


def _parse_action_mapping(flat: dict[str, str]) -> ActionMapping:
    """Parse ``{"action.X": "robot_key"}`` -> ActionMapping (order preserved)."""
    return ActionMapping(actions={k.removeprefix("action."): v for k, v in flat.items()})


def _auto_infer_observation_mapping(cfg: QwenVlaDataConfig, language_key: str) -> ObservationMapping:
    """Auto-infer obs mapping from the data config (identity on bare keys)."""
    video = {k.removeprefix("video."): k.removeprefix("video.") for k in cfg.video_keys}
    state = {k.removeprefix("state."): k.removeprefix("state.") for k in cfg.state_keys}
    return ObservationMapping(video=video, state=state, language_key=language_key)


def _auto_infer_action_mapping(cfg: QwenVlaDataConfig) -> ActionMapping:
    """Auto-infer action mapping from the data config (identity on bare keys)."""
    return ActionMapping(actions={k.removeprefix("action."): k.removeprefix("action.") for k in cfg.action_keys})


def _to_video_batch(frame: Any) -> np.ndarray:
    """Promote a single ``(H, W, C)`` frame to the model's ``(B=1, T=1, H, W, C)`` uint8."""
    arr = np.asarray(frame)
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.uint8)


def _to_state_batch(state: Any) -> np.ndarray:
    """Promote a state vector to the model's ``(B=1, T=1, D)`` float32."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr[np.newaxis]
    while arr.ndim < 3:
        arr = arr[np.newaxis, ...]
    return arr


class QwenVlaPolicy(Policy):
    """Qwen-VLA policy - LOCAL (in-proc) and SERVICE (ZMQ) inference.

    Args:
        data_config: Config name (e.g. ``"so100"``) or a :class:`QwenVlaDataConfig`.
        host: Service host (SERVICE mode).
        port: Service port (SERVICE mode).
        model_path: HF model ID or local path. When set, triggers LOCAL mode.
        device: ``"cuda"`` or ``"cpu"`` (LOCAL mode).
        denoising_steps: Number of flow-matching Euler integration steps at
            inference (LOCAL mode). The paper uses a small number (~4) for
            real-time control.
        api_token: ZMQ auth token. Falls back to ``QWEN_VLA_API_TOKEN`` env var.
        observation_mapping: ``{robot_key: "video.X" | "state.X"}`` override.
        action_mapping: ``{"action.X": "robot_key"}`` override (order defines
            the channel layout for the unified ``Y[H x K]`` unpack).
        language_key: Override the model's language key (default ``"task"``).
        instruction: Optional default instruction baked into the embodiment
            prompt when ``get_actions`` is called without one.

    Examples::

        # SERVICE mode (no model deps needed on the client)
        policy = QwenVlaPolicy(data_config="so100", host="localhost", port=5556)

        # LOCAL mode (requires the qwen-vla package + GPU)
        policy = QwenVlaPolicy(
            data_config="aloha_bimanual",
            model_path="Qwen/Qwen-VLA-Base",
            device="cuda",
            denoising_steps=4,
        )
    """

    def __init__(
        self,
        data_config: str | QwenVlaDataConfig = "so100",
        host: str = "localhost",
        port: int = 5556,
        model_path: str | None = None,
        device: str = "cuda",
        denoising_steps: int = 4,
        api_token: str | None = None,
        observation_mapping: dict[str, str] | None = None,
        action_mapping: dict[str, str] | None = None,
        language_key: str = "task",
        instruction: str | None = None,
        flatten_to_joints: bool = False,
        **kwargs,
    ):
        self.data_config = load_data_config(data_config)
        self.data_config_name = data_config if isinstance(data_config, str) else type(data_config).__name__
        self.device = device
        self.denoising_steps = denoising_steps
        self._language_key = language_key
        self._default_instruction = instruction
        # When True (or when set_robot_state_keys gives a flat per-joint
        # list), grouped action vectors are flattened into one scalar per
        # robot joint - required to drive per-joint actuators like the
        # MuJoCo sim's so100 (Rotation, Pitch, ... Jaw). See
        # set_robot_state_keys / _flatten_to_joints.
        self._flatten_to_joints = flatten_to_joints
        self._robot_state_keys: list[str] = []

        self._local_model: Any = None
        self._client: QwenVlaInferenceClient | None = None

        # Resolve mappings up-front (they do not depend on the loaded model -
        # Qwen-VLA's modality surface comes from the data config, not a
        # model-side modality_config).
        if observation_mapping is not None:
            self._obs_mapping = _parse_observation_mapping(observation_mapping, language_key)
        else:
            self._obs_mapping = _auto_infer_observation_mapping(self.data_config, language_key)

        if action_mapping is not None:
            self._action_mapping = _parse_action_mapping(action_mapping)
        else:
            self._action_mapping = _auto_infer_action_mapping(self.data_config)

        if model_path is not None:
            self._mode = "local"
            logger.info("Qwen-VLA local mode, model=%s", model_path)
            self._load_local_model(model_path, device)
        else:
            self._mode = "service"
            resolved_token = api_token or os.environ.get("QWEN_VLA_API_TOKEN")
            self._client = QwenVlaInferenceClient(host=host, port=port, api_token=resolved_token)
            logger.info("Qwen-VLA service mode, %s:%s", host, port)

        logger.info(
            "Qwen-VLA ready [mode=%s, config=%s, actions=%s]",
            self._mode,
            self.data_config_name,
            self._action_mapping.actions,
        )

    # Model loading (LOCAL)

    def _load_local_model(self, model_path: str, device: str) -> None:
        """Load the Qwen-VLA model in-process.

        Gated on :func:`_qwen_vla_installed`. The concrete load call is kept
        behind a thin import so the rest of the module imports cleanly without
        the heavy dependency (PLAN section 6.2: package name TBD on upstream
        release - we raise a clear, actionable error until then).
        """
        if not _qwen_vla_installed():
            raise ImportError(
                "Qwen-VLA local inference requires the upstream 'qwen-vla' package, "
                "which is not installed. Install it with `pip install 'strands-robots[qwen-vla]'` "
                "once the model release is public, or use SERVICE mode (host/port) against a "
                "running Qwen-VLA inference server."
            )
        # The exact loader entrypoint is finalized on upstream release. We keep
        # the import local so module import never pulls torch/transformers.
        from qwen_vla import load_policy  # type: ignore[import-not-found]

        self._local_model = load_policy(
            model_path=model_path,
            device=device,
            denoising_steps=self.denoising_steps,
        )
        logger.info("Qwen-VLA model loaded from %s on %s", model_path, device)

    # Policy interface

    @property
    def provider_name(self) -> str:
        return "qwen_vla"

    @property
    def requires_images(self) -> bool:
        """Qwen-VLA is a vision-language-action model; it always needs frames."""
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's per-joint actuator names (from the sim/hardware).

        The simulation calls this with the robot's flat joint-name list
        (e.g. ``["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll",
        "Jaw"]`` for so100). When set, :meth:`get_actions` flattens the
        model's grouped action vectors into one scalar per joint, in order,
        so the per-joint actuator interface (``data.ctrl[i] = scalar``)
        accepts them directly. Without this the grouped vectors would be
        rejected by per-joint backends - the exact failure surfaced when
        driving the MuJoCo sim. Passing a non-empty list implicitly enables
        ``flatten_to_joints``.
        """
        self._robot_state_keys = list(robot_state_keys)
        if robot_state_keys:
            self._flatten_to_joints = True
            logger.info("Qwen-VLA will flatten actions to %d robot joints: %s", len(robot_state_keys), robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Per-episode reset (the #187 reproducibility contract).

        SERVICE mode forwards a ``reset`` to the server so its flow-matching
        sampler RNG re-initializes for byte-identical re-runs. LOCAL mode
        reseeds the in-proc RNGs (Python / NumPy / torch). Best-effort: a
        server that does not expose ``reset`` is logged and tolerated.
        """
        if self._mode == "service":
            assert self._client is not None, "service mode requires a client"
            try:
                payload: dict[str, Any] = {}
                if seed is not None:
                    payload = {"options": {"seed": int(seed)}}
                self._client.call_endpoint("reset", payload if payload else None)
                logger.debug("QwenVlaPolicy.reset: forwarded to server (seed=%r)", seed)
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.info(
                    "QwenVlaPolicy.reset: server did not accept reset (seed=%r): %s; continuing",
                    seed,
                    e,
                )
            return

        # LOCAL mode reseed.
        if seed is None:
            return
        import random as _random

        _random.seed(seed)
        np.random.seed(seed)
        try:
            import torch as _torch

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        # Forward to the model if it exposes its own reset (sampler state).
        if self._local_model is not None and hasattr(self._local_model, "reset"):
            try:
                self._local_model.reset(seed=seed)
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.info("QwenVlaPolicy.reset: local model reset failed: %s", e)
        logger.debug("QwenVlaPolicy.reset: local-mode reseed applied (seed=%r)", seed)

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs) -> list[dict[str, Any]]:
        instr = instruction or self._default_instruction
        if not instr:
            raise ValueError("get_actions requires an instruction (none provided and no default set)")
        if self._mode == "local":
            actions = self._local_get_actions(observation_dict, instr)
        else:
            actions = self._service_get_actions(observation_dict, instr)
        if self._flatten_to_joints and self._robot_state_keys:
            actions = self._flatten_actions_to_joints(actions)
        return actions

    # Observation building

    def _build_observation(self, robot_obs: dict[str, Any], instruction: str) -> dict:
        """Build the model-native observation batch.

        Qwen-VLA expects video frames per camera view tag, a flat state
        vector, and the embodiment-augmented instruction (section 2.3). We
        pack video + state into ``{view_tag: ndarray}`` and ``{state_key:
        ndarray}`` plus the rendered prompt under the language key.
        """
        # Detect a per-joint sim observation: the robot obs is keyed by the
        # actual joint names (from set_robot_state_keys) + camera frames, not by
        # the data_config's grouped state/video keys. In that case we auto-bridge
        # rather than warn-and-drop - making the provider usable in the MuJoCo
        # sim out of the box.
        sim_mode = bool(self._flatten_to_joints and self._robot_state_keys)

        video_dict: dict[str, np.ndarray] = {}
        for robot_key, model_key in self._obs_mapping.video.items():
            if robot_key in robot_obs:
                # Resolve the paper's camera view token for this video key.
                view_tag = self.data_config.image_view_tags.get(f"video.{model_key}", model_key)
                video_dict[view_tag] = _to_video_batch(robot_obs[robot_key])
            elif not sim_mode:
                logger.warning("Robot camera '%s' missing in observation", robot_key)
        # Sim fallback: use whatever camera frame(s) the sim actually rendered
        # (e.g. the MuJoCo "default" cam) under the embodiment's primary view tag.
        if sim_mode and not video_dict:
            primary_tag = next(iter(self.data_config.image_view_tags.values()), "ego")
            for k, v in robot_obs.items():
                arr = np.asarray(v)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    video_dict[primary_tag] = _to_video_batch(arr)
                    break

        state_dict: dict[str, np.ndarray] = {}
        for robot_key, model_key in self._obs_mapping.state.items():
            if robot_key in robot_obs:
                state_dict[model_key] = _to_state_batch(robot_obs[robot_key])
            elif not sim_mode:
                logger.warning("Robot state '%s' missing in observation", robot_key)
        # Sim fallback: assemble a single flat state vector from the per-joint
        # scalars the sim provides, in joint order, under the first state key.
        if sim_mode and not state_dict:
            joint_vals = [float(robot_obs[j]) for j in self._robot_state_keys if j in robot_obs]
            if joint_vals:
                first_state_key = (
                    self.data_config.state_keys[0].removeprefix("state.") if self.data_config.state_keys else "state"
                )
                state_dict[first_state_key] = _to_state_batch(np.asarray(joint_vals, dtype=np.float32))

        prompt = self.data_config.embodiment_prompt(instruction)

        return {
            "video": video_dict,
            "state": state_dict,
            "language": {self._obs_mapping.language_key: [[prompt]]},
        }

    # Action unpacking

    def _unpack_actions(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Unpack a model action dict -> per-timestep robot actuator dicts.

        The model returns ``{action_key: ndarray[H, d_key]}`` (already split
        per action family by the server/loader) OR a single ``{"action":
        ndarray[H, K]}`` chunk. For the latter we split by the action_mapping's
        channel layout. Either way, the result is a list of H per-step dicts
        keyed by robot actuator name.
        """
        squeezed: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            bare = key.removeprefix("action.")
            arr = np.asarray(value)
            while arr.ndim > 2:
                arr = arr[0]
            squeezed[bare] = arr

        if not squeezed:
            return []

        # Single unified chunk: split into the mapped action families by order.
        if set(squeezed.keys()) == {"action"} and len(self._action_mapping.actions) > 1:
            squeezed = self._split_unified_chunk(squeezed["action"])

        horizon = next(iter(squeezed.values())).shape[0]
        mapped = self._action_mapping.actions

        actions: list[dict[str, Any]] = []
        for t in range(horizon):
            step: dict[str, Any] = {}
            for model_key, robot_key in mapped.items():
                if model_key in squeezed:
                    row = squeezed[model_key][t]
                    step[robot_key] = row.tolist() if hasattr(row, "tolist") else row
            for model_key in squeezed:
                if model_key not in mapped:
                    row = squeezed[model_key][t]
                    step[f"unmapped.{model_key}"] = row.tolist() if hasattr(row, "tolist") else row
            actions.append(step)
        return actions

    def _split_unified_chunk(self, chunk: np.ndarray) -> dict[str, np.ndarray]:
        """Split a unified ``Y[H, K]`` chunk into per-action-family arrays.

        Channel layout follows the action_mapping insertion order. Each
        family's width is inferred from the data config's action_keys order;
        when the model emits more channels than the sum of known widths the
        tail is treated as zero-padding and dropped (section 2.4).

        Because per-family widths are not encoded in the data config (only the
        key names are), we split evenly across the mapped families as a
        documented fallback. Deployments that need exact per-family widths
        should pass an ``action_mapping`` whose families already match the
        server's bare-key split (the common path), so this fallback only fires
        for a single-tensor server returning ``{"action": Y}``.
        """
        families = list(self._action_mapping.actions.keys())
        n = len(families)
        total = chunk.shape[1]
        width = max(1, total // n)
        out: dict[str, np.ndarray] = {}
        for i, fam in enumerate(families):
            start = i * width
            end = start + width if i < n - 1 else total
            out[fam] = chunk[:, start:end]
        return out

    def _flatten_actions_to_joints(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten grouped action vectors into one scalar per robot joint.

        Concatenates each timestep's mapped action values (in action_mapping
        order, scalars and vectors alike) and assigns them positionally to the
        robot's joint names recorded by :meth:`set_robot_state_keys`. This
        bridges the model's grouped action layout (e.g. ``single_arm`` 6-vec +
        ``gripper`` 1-vec) to a per-joint actuator interface (so100's
        ``Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw``).

        If the concatenated width does not match the joint count, the overlap
        is used and a one-time warning is logged (no silent truncation of
        intent): extra model channels are dropped, missing joints are left
        unset so the backend keeps their last command.
        """
        joints = self._robot_state_keys
        out: list[dict[str, Any]] = []
        warned = False
        for step in actions:
            flat: list[float] = []
            for model_key in self._action_mapping.actions:
                robot_key = self._action_mapping.actions[model_key]
                if robot_key in step:
                    v = step[robot_key]
                elif model_key in step:
                    v = step[model_key]
                else:
                    continue
                arr = np.atleast_1d(np.asarray(v, dtype=np.float32))
                flat.extend(arr.tolist())
            # The unified Y[H,K] is zero-padded (section 2.4): the leading
            # channels are the valid action dims, the tail is padding. A flat
            # vector WIDER than the joint count is expected - take the valid
            # leading prefix. Only warn when the model produced FEWER channels
            # than joints (genuinely under-specified).
            if len(flat) < len(joints) and not warned:
                logger.warning(
                    "Qwen-VLA flatten: model produced %d action channels but robot has %d joints; "
                    "leaving %d joints unset (backend keeps last command)",
                    len(flat),
                    len(joints),
                    len(joints) - len(flat),
                )
                warned = True
            out.append({joints[i]: float(flat[i]) for i in range(min(len(flat), len(joints)))})
        return out

    # LOCAL inference

    def _local_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        obs = self._build_observation(robot_obs, instruction)
        assert self._local_model is not None, "local model not loaded"
        raw = self._local_model.get_action(obs)
        if isinstance(raw, tuple) and len(raw) == 2:
            raw, _info = raw
        return self._unpack_actions(raw)

    # SERVICE inference

    def _service_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        assert self._client is not None, "service client not initialized"
        obs = self._build_observation(robot_obs, instruction)
        action_chunk = self._client.get_action(obs)
        return self._unpack_actions(action_chunk)


__all__ = ["QwenVlaPolicy", "ObservationMapping", "ActionMapping"]
