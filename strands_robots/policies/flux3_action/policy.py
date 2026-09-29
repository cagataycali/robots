"""FLUX 3 Action policy provider - in-process inference through ``flux_action``.

FLUX 3 Action (Black Forest Labs, released 2026-09-22) is a flow-matching VLA that
predicts a 42-step chunk of absolute joint targets from two cameras, an 8-tick
observation history and a text instruction. Its SO-101 checkpoint
``black-forest-labs/flux-3-action-so101`` speaks the lerobot SO-101 convention
(arm joints in degrees, gripper in percent); :class:`UnitAdapter` bridges that to
the robot's own units (radians in the MuJoCo ``so101`` model).

Two consumption modes, both honouring the ``Policy`` contract that a returned list
is executed one entry per control tick:

``"queued"`` (default)
    ``get_actions`` returns ONE action per call, so the runner asks again every
    tick and ``flux_action``'s own ``select_action`` sees every observation. That
    is the control loop the model was trained for: 30 Hz, history of every tick,
    replan after ``n_action_steps`` (32) of the 42 predicted steps.
``"chunk"``
    ``get_actions`` returns ``execute_steps`` actions from one stateless
    ``predict_action_chunk`` call. Open-loop within the chunk, useful for
    fixed-horizon comparisons; the history then only sees one tick per chunk.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

import numpy as np

from strands_robots.policies._state_keys import observation_joint_keys
from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, partial_construction_repr, require_optional

from .units import (
    SO101_JOINT_LABELS,
    SO101_SIM_GRIPPER_RANGE_RAD,
    SO101_SIM_JOINT_OFFSETS_DEG,
    SO101_SIM_JOINT_SIGNS,
    UnitAdapter,
)

logger = logging.getLogger(__name__)

#: ``[flux3]`` is an empty extra on purpose: the inference library is git-only
#: and NATTEN's wheel depends on the torch/CUDA build, so no pip line the
#: package could declare supplies them; the refusal names the real install.
FLUX3_SYSTEM_INSTALL_HINT = (
    "flux-action is not published on PyPI and the [flux3] extra is empty, so no pip line supplies it.\n"
    "Install the inference library and the NATTEN wheel matching your torch/CUDA build, then retry:\n"
    "  pip install 'flux-action[encoders] @ git+https://github.com/black-forest-labs/flux-action'\n"
    "  pip install natten==0.21.6 -f https://whl.natten.org\n"
    "docs/learn/policies/flux3-action.md has the prerequisites (a CUDA GPU with ~22 GB free)."
)

DEFAULT_CHECKPOINT = "black-forest-labs/flux-3-action-so101"
_MODES = ("queued", "chunk")
_CAMERA_ROLES = ("scene", "wrist")


def _forward_natten_backend_to_neighborhood_calls() -> None:
    """Make ``flux_action``'s chosen NATTEN backend reach the neighborhood kernels.

    ``flux_action.models.video_vae._attend`` passes its backend choice as
    ``attention_kwargs={"backend": ...}``. NATTEN only reads ``attention_kwargs``
    on the dense path (window covering the whole input); a genuine neighborhood
    window ignores it and auto-selects ``cutlass-fna`` whenever the compute
    capability is >= 6.0, even when the installed library carries no kernel image
    for it (Jetson Thor sm_110). This shim forwards ``attention_kwargs["backend"]``
    as the explicit ``backend=`` argument of ``na2d`` / ``na3d`` inside the
    ``video_vae`` module only, so ``F3_NATTEN_BACKEND=flex-fna`` actually applies.
    Idempotent; a no-op when ``flux_action`` is absent.
    """
    try:
        import flux_action.models.video_vae as video_vae
        from natten import functional as natten_functional
    except ImportError:
        return
    if getattr(video_vae, "_strands_natten_backend_forwarded", False):
        return

    def _forwarding(fn: Any) -> Any:
        def call(
            *args: Any, attention_kwargs: dict[str, Any] | None = None, backend: str | None = None, **kwargs: Any
        ) -> Any:
            chosen = (attention_kwargs or {}).get("backend")
            if backend is None and isinstance(chosen, str) and chosen.endswith("-fna"):
                backend = chosen
            return fn(*args, attention_kwargs=attention_kwargs, backend=backend, **kwargs)

        call.__name__ = fn.__name__
        return call

    video_vae.na2d = _forwarding(natten_functional.na2d)
    video_vae.na3d = _forwarding(natten_functional.na3d)
    video_vae._strands_natten_backend_forwarded = True


class Flux3ActionPolicy(Policy):
    """FLUX 3 Action SO-101 checkpoint driving a 6-joint arm at 30 Hz.

    Args:
        pretrained_name_or_path: Hugging Face repo id or local directory holding the
            fine-tuned checkpoint (``config.json`` + ``model.safetensors``). The base
            video VAE and text encoder are fetched by ``flux_action`` from
            ``black-forest-labs/flux-3-action-base`` at the revision the checkpoint pins.
        revision: Optional git revision for a hub repo.
        device: Torch device for inference (``"cuda"`` by default; the 3.9B
            transformer plus encoders need roughly 22 GB in bf16).
        camera_map: Observation image key for each FLUX role, e.g.
            ``{"scene": "front", "wrist": "gripper_cam"}``. Defaults to the role names
            themselves (``"scene"`` and ``"wrist"``). Both roles are required.
        joint_units: Units of the robot's joint positions, ``"rad"`` (MuJoCo) or
            ``"deg"`` (lerobot hardware with ``use_degrees``).
        joint_signs: Five joint directions (+1/-1) applied before the offsets.
        joint_offsets_deg: Five additive degree offsets applied after the unit scale
            and sign (see :class:`UnitAdapter`); default carries the MuJoCo zero into
            the checkpoint's calibration frame. Pass all zeros for lerobot hardware.
        gripper_range: ``(closed, open)`` gripper joint values in robot units mapped
            to 0..100 percent. Defaults to the MuJoCo ``so101`` gripper travel; pass
            ``(0, 100)`` for lerobot hardware.
        mode: ``"queued"`` (one action per tick through ``select_action``) or
            ``"chunk"`` (``execute_steps`` actions per stateless chunk).
        execute_steps: Actions consumed per chunk in ``"chunk"`` mode; 1..42.
        task: Default instruction when the caller passes an empty one.
        warn_outside_training_range: Log once when the converted state leaves the
            checkpoint's ``state`` q01/q99 window, a cheap unit / calibration check.
        natten_backend: NATTEN attention backend for the video VAE
            (``blackwell-fna`` | ``hopper-fna`` | ``cutlass-fna`` | ``flex-fna``).
            ``None`` honours ``$F3_NATTEN_BACKEND`` when set, otherwise probes the
            GPU: when the installed ``natten`` library carries no kernel image for
            this device's compute capability (Jetson Thor sm_110 with the
            published wheels, which cover sm_75..sm_120 but skip sm_110) it selects
            ``flex-fna``, the pure-torch fallback, instead of letting NATTEN crash
            with ``no kernel image is available`` mid-rollout.
    """

    provider_name: ClassVar[str] = "flux3_action"
    requires_images: ClassVar[bool] = True
    reads_instruction: ClassVar[bool] = True

    def __init__(
        self,
        pretrained_name_or_path: str = DEFAULT_CHECKPOINT,
        revision: str | None = None,
        device: str = "cuda",
        camera_map: dict[str, str] | None = None,
        joint_units: str = "rad",
        joint_signs: tuple[float, float, float, float, float] | list[float] = SO101_SIM_JOINT_SIGNS,
        joint_offsets_deg: tuple[float, float, float, float, float] | list[float] = SO101_SIM_JOINT_OFFSETS_DEG,
        gripper_range: tuple[float, float] | list[float] = SO101_SIM_GRIPPER_RANGE_RAD,
        mode: str = "queued",
        execute_steps: int = 32,
        task: str = "",
        warn_outside_training_range: bool = True,
        natten_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if not 1 <= int(execute_steps) <= 42:
            raise ValueError(f"execute_steps must be within 1..42 (chunk_size), got {execute_steps}")
        self.camera_map = {role: role for role in _CAMERA_ROLES}
        if camera_map:
            unknown = set(camera_map) - set(_CAMERA_ROLES)
            if unknown:
                raise ValueError(f"camera_map roles must be {_CAMERA_ROLES}, unknown: {sorted(unknown)}")
            self.camera_map.update({k: str(v) for k, v in camera_map.items()})
        self.units = UnitAdapter(
            joint_units=joint_units,
            joint_signs=tuple(float(x) for x in joint_signs),
            joint_offsets_deg=tuple(float(x) for x in joint_offsets_deg),
            gripper_range=(float(gripper_range[0]), float(gripper_range[1])),
        )
        self.pretrained_name_or_path = pretrained_name_or_path
        self.revision = revision
        self.device = device
        self.mode = mode
        self.execute_steps = int(execute_steps)
        # In "chunk" mode get_actions hands back execute_steps actions from one
        # stateless call; declaring that as the trained chunk makes
        # Policy.execution_horizon (and so resolve_chunk_length) consume the
        # whole chunk instead of 1, which would re-query the same chunk every
        # action_horizon steps and drop its tail.
        self.actions_per_step = self.execute_steps if mode == "chunk" else 1
        self.task = task
        self.warn_outside_training_range = warn_outside_training_range
        self.robot_state_keys: list[str] = []
        self._current_instruction: str | None = None
        self._warned_range = False
        # Timing telemetry: every tick's wall time and, separately, the ticks that
        # ran the transformer (a replan). Milliseconds.
        self.tick_ms: list[float] = []
        self.inference_ms: list[float] = []

        self._torch: Any = require_optional("torch", extra="lerobot", purpose="FLUX 3 Action inference (CUDA)")
        require_optional(
            "flux_action",
            system_install=FLUX3_SYSTEM_INSTALL_HINT,
            purpose="FLUX 3 Action inference",
        )
        self.natten_backend = self._select_natten_backend(natten_backend)
        _forward_natten_backend_to_neighborhood_calls()
        # The 3.9B-parameter checkpoint is loaded lazily: constructing the provider
        # (``create_policy("flux3_action")``) stays cheap, and the weights arrive on
        # :meth:`load`, the first :meth:`reset` or the first :meth:`get_actions`.
        # Same shape as ``lerobot_local``, which also constructs without loading.
        self._policy: Any = None
        self.load_s: float | None = None
        self.n_obs_steps: int | None = None
        self.chunk_size: int | None = None
        self.n_action_steps: int | None = None
        self.fps: int | None = None
        self._state_window: tuple[list[float], list[float]] | None = None

    @property
    def loaded(self) -> bool:
        """Whether the checkpoint is resident (``load`` ran)."""
        return self._policy is not None

    def load(self) -> None:
        """Load the checkpoint onto ``device``; idempotent.

        Called by :meth:`reset` and :meth:`get_actions` when needed, so callers only
        need it to pay the load cost up front (``load_s`` records the wall time).
        """
        if self._policy is not None:
            return
        from flux_action.inference.so101 import load_policy

        t0 = time.perf_counter()
        policy = load_policy(self.pretrained_name_or_path, revision=self.revision).to(self.device)
        policy.eval()
        self.load_s = time.perf_counter() - t0
        cfg = policy.config
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.chunk_size = int(cfg.chunk_size)
        self.n_action_steps = int(cfg.n_action_steps)
        self.fps = int(cfg.fps)
        self._policy = policy
        self._state_window = self._training_state_window()
        logger.info(
            "flux3_action: loaded %s on %s in %.1fs (history %d, chunk %d, execute %d, %d Hz)",
            self.pretrained_name_or_path,
            self.device,
            self.load_s,
            self.n_obs_steps,
            self.chunk_size,
            self.n_action_steps,
            self.fps,
        )

    # -- Policy contract --------------------------------------------------

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Remember the robot's joint key order; six keys map onto the SO-101 joints.

        Raises:
            ValueError: if ``robot_state_keys`` is not an ordered list of distinct
                non-blank names (:func:`~strands_robots.utils.name_list_error`);
                a bare string would otherwise bind one joint per character.
        """
        if error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys"):
            raise ValueError(error)
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Clear the observation history and the action queue; starts a new episode."""
        super().reset(seed)
        self.load()
        self._policy.reset()
        self._current_instruction = None
        if seed is not None:
            self._torch.manual_seed(seed)

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        """Refuse before loading 3.9B parameters when a required camera is missing."""
        camera_map = {role: role for role in _CAMERA_ROLES}
        camera_map.update(policy_config.get("camera_map") or {})
        missing = [f"{role}={key!r}" for role, key in camera_map.items() if key not in observation_keys]
        if missing:
            images = sorted(k for k in observation_keys if "." not in k and not k.isdigit())
            raise ValueError(
                "flux3_action needs two cameras named by camera_map; missing "
                f"{', '.join(missing)}. Observation keys: {images}. Add cameras with "
                "robot.add_camera(name=...) or pass camera_map={'scene': ..., 'wrist': ...}."
            )

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return the next action(s) as joint targets in the robot's units.

        Args:
            observation_dict: Flat robots observation: six joint floats plus the two
                camera images as ``(H, W, 3)`` uint8 arrays.
            instruction: Task text; falls back to ``task`` when empty. A changed
                instruction resets the history (a new task is a new episode).
            **kwargs: Ignored (contract: providers ignore unknown keys).

        Returns:
            ``[{joint: target}]`` - one dict in ``"queued"`` mode, ``execute_steps``
            dicts in ``"chunk"`` mode.
        """
        self.load()
        text = instruction or self.task
        if self._current_instruction is not None and text != self._current_instruction:
            self._policy.reset()
        self._current_instruction = text
        keys, state = self._joint_state(observation_dict)
        batch = self._batch(observation_dict, state, text)
        t0 = time.perf_counter()
        if self.mode == "queued":
            queue_empty = len(getattr(self._policy, "_action_queue", ())) == 0
            with self._torch.inference_mode():
                out = self._policy.select_action(batch)
            rows = out.detach().float().cpu().numpy().reshape(1, -1)
        else:
            queue_empty = True
            with self._torch.inference_mode():
                chunk = self._policy.predict_action_chunk(batch)
            rows = chunk.detach().float().cpu().numpy()[0, : self.execute_steps]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.tick_ms.append(elapsed_ms)
        if queue_empty:
            self.inference_ms.append(elapsed_ms)
        return [dict(zip(keys, self.units.model_to_robot(row.tolist()), strict=True)) for row in rows]

    # -- helpers ----------------------------------------------------------

    _NATTEN_BACKENDS = ("blackwell-fna", "hopper-fna", "cutlass-fna", "flex-fna")

    def _select_natten_backend(self, requested: str | None) -> str | None:
        """Pick the NATTEN backend once, before the model loads, and export it.

        ``flux_action`` reads ``$F3_NATTEN_BACKEND`` at first attention call; setting
        it here keeps the choice explicit and visible in the process environment.
        """
        import os

        if requested is not None:
            if requested not in self._NATTEN_BACKENDS:
                raise ValueError(f"natten_backend must be one of {self._NATTEN_BACKENDS}, got {requested!r}")
            os.environ["F3_NATTEN_BACKEND"] = requested
            return requested
        if os.environ.get("F3_NATTEN_BACKEND"):
            return os.environ["F3_NATTEN_BACKEND"]
        torch = self._torch
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(torch.device(self.device))
        if (major, minor) in self._natten_kernel_arches():
            return None  # let flux_action probe the compiled CUTLASS kernels
        os.environ["F3_NATTEN_BACKEND"] = "flex-fna"
        logger.info(
            "flux3_action: natten has no kernel image for sm_%d%d (%s); using the flex-fna torch fallback",
            major,
            minor,
            torch.cuda.get_device_name(torch.device(self.device)),
        )
        return "flex-fna"

    @staticmethod
    def _natten_kernel_arches() -> set[tuple[int, int]]:
        """Compute capabilities baked into the installed ``libnatten`` (empty when unknown)."""
        import glob
        import re
        import shutil
        import subprocess

        try:
            import natten
        except ImportError:
            return set()
        cuobjdump = shutil.which("cuobjdump")
        libs = glob.glob(f"{natten.__path__[0]}/libnatten*.so")
        if not cuobjdump or not libs:
            return set()
        try:
            out = subprocess.run(
                [cuobjdump, "--list-elf", libs[0]], capture_output=True, text=True, errors="replace", timeout=30
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return set()
        return {divmod(int(n), 10) for n in re.findall(r"\bsm_(\d+)\b", out)}

    def _joint_state(self, observation: dict[str, Any]) -> tuple[list[str], list[float]]:
        missing = [k for k in self.robot_state_keys if k not in observation]
        if missing:
            raise KeyError(f"flux3_action: robot_state_keys name joints absent from the observation: {missing}")
        keys = observation_joint_keys(observation, self.robot_state_keys)
        if len(keys) != 6:
            raise ValueError(
                f"flux3_action drives exactly 6 joints [{', '.join(SO101_JOINT_LABELS)}]; "
                f"observation carries {len(keys)}: {keys}"
            )
        state = self.units.robot_to_model([float(observation[k]) for k in keys])
        if self.warn_outside_training_range and not self._warned_range and self._state_window is not None:
            lo, hi = self._state_window
            outside = [
                f"{name}={v:.1f} not in [{a:.1f}, {b:.1f}]"
                for name, v, a, b in zip(SO101_JOINT_LABELS, state, lo, hi, strict=True)
                if v < a or v > b
            ]
            if outside:
                self._warned_range = True
                logger.warning(
                    "flux3_action: converted state is outside the checkpoint's training q01..q99 window "
                    "(%s). Check joint_units / joint_offsets_deg / gripper_range; the model saw real "
                    "SO-101 calibrations, not this robot's zero.",
                    "; ".join(outside),
                )
        return keys, state

    def _batch(self, observation: dict[str, Any], state: list[float], text: str) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {
            "state": torch.tensor([state], dtype=torch.float32, device=self.device),
            "task": [text],
        }
        for role, key in self.camera_map.items():
            image = observation.get(key)
            if image is None:
                raise ValueError(
                    f"flux3_action camera {role!r} expects observation key {key!r}; "
                    f"available image keys: {[k for k, v in observation.items() if isinstance(v, np.ndarray) and v.ndim == 3]}"
                )
            batch[f"images.{role}"] = self._to_chw_uint8(np.asarray(image))
        return batch

    def _to_chw_uint8(self, image: np.ndarray) -> Any:
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"flux3_action expects (H, W, 3) uint8 images, got shape {image.shape}")
        rgb = image[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, non_blocking=True)

    def _training_state_window(self) -> tuple[list[float], list[float]] | None:
        """The checkpoint's ``state`` q01/q99 quantiles (``config.state_normalization``)."""
        norm = getattr(self._policy.config, "state_normalization", None)
        if not isinstance(norm, dict) or "q01" not in norm or "q99" not in norm:
            return None
        lo, hi = [float(x) for x in norm["q01"]], [float(x) for x in norm["q99"]]
        if len(lo) != 6 or len(hi) != 6:
            return None
        return lo, hi

    def __repr__(self) -> str:
        try:
            return (
                f"Flux3ActionPolicy(pretrained_name_or_path={self.pretrained_name_or_path!r}, mode={self.mode!r}, "
                f"device={self.device!r}, joint_units={self.units.joint_units!r})"
            )
        except AttributeError:
            return partial_construction_repr(self)
