"""HolosomaPolicy: Amazon FAR's Holosoma locomotion controller for the Unitree G1.

Ports the deployment loop of ``holosoma_inference`` (``holosoma_inference/policies/base.py`` +
``policies/locomotion.py``, commit ``bccd4d7``) to the Strands
:class:`~strands_robots.policies.base.Policy` contract:

* one ONNX file (``fastsac_g1_29dof.onnx`` or ``ppo_g1_29dof.onnx``), input
  ``actor_obs [1, 100]``, output ``action [1, 29]``, with the PD gains, joint
  names and command ranges in the ONNX metadata;
* a 100-wide observation whose terms are laid out alphabetically
  (:mod:`strands_robots.policies.holosoma.observation`);
* ``target = default_angles + 0.25 * clip(action, -100, 100)`` for all 29 joints;
* a two-foot gait clock at 50 Hz with a 1.0 s period.

The provider shares the G1 joint table, the quaternion helpers and the
MuJoCo PD-to-torque shim with the GR00T-WBC provider
(:mod:`strands_robots.policies.wbc`): the shim reads ``config.num_actions``,
``config.n_obs_joints``, ``config.height_cmd``, :attr:`default_angles` and
:meth:`compute_torques`, all of which this class provides.

Weights are not bundled. They are Apache-2.0 artifacts of
https://github.com/amazon-far/holosoma (``src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/``),
mirrored on the Hub at :data:`HOLOSOMA_HF_REPO`, and fetched on first use
through ``huggingface_hub`` (the ``[holosoma]`` extra) into the Hub cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from strands_robots.locomotion_envelope import target_velocity_component_error
from strands_robots.policies._log_safety import sanitize_log_value
from strands_robots.policies.base import Policy
from strands_robots.policies.holosoma.config import HOLOSOMA_OBS_DIM, HolosomaConfig
from strands_robots.policies.holosoma.observation import GaitPhase, build_actor_obs
from strands_robots.policies.wbc.control import compute_targets, pd_control, projected_gravity
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS
from strands_robots.utils import boolean_flag_error, require_optional, sequence_length

logger = logging.getLogger(__name__)

#: Hub mirror of the released G1 locomotion checkpoints (what lerobot's own
#: ``HolosomaLocomotionController`` downloads). Source of truth and licence:
#: https://github.com/amazon-far/holosoma (Apache-2.0).
HOLOSOMA_HF_REPO = "nepyope/holosoma_locomotion"

#: The mirror commit every default fetch is pinned to. The mirror is a personal
#: Hub account, not the vendor's org, so a moving reference would let whoever
#: holds that account swap the network that commands a humanoid; a commit pin
#: plus :data:`HOLOSOMA_SHA256` closes that door. Pass ``revision=`` to move it.
HOLOSOMA_HF_REVISION = "a5eaedd54270ef2d457bd395af0e15574f281e57"

#: File name per algorithm, as shipped upstream and on the mirror.
HOLOSOMA_FILES: dict[str, str] = {
    "fastsac": "fastsac_g1_29dof.onnx",
    "ppo": "ppo_g1_29dof.onnx",
}

#: sha256 of each released file, read from the authoritative Apache-2.0 tree
#: (github.com/amazon-far/holosoma@d18d6cc5, src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/)
#: and identical on the mirror at :data:`HOLOSOMA_HF_REVISION`. A fetched file
#: that hashes differently is refused before a session is built.
HOLOSOMA_SHA256: dict[str, str] = {
    "fastsac_g1_29dof.onnx": "8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717",
    "ppo_g1_29dof.onnx": "c9d310f479c2da1e86f468de24f64e102a2a275c14eed0fa341473612d942294",
}

#: The 29 joints in ``dof_names`` order. Identical to the GR00T-WBC table, so
#: the two families share one name-resolved joint map.
HOLOSOMA_G1_JOINTS: tuple[str, ...] = WBC_G1_ALL_JOINTS

_ARM_JOINTS: frozenset[str] = frozenset(HOLOSOMA_G1_JOINTS[15:])


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_holosoma_checkpoint(
    checkpoint: str | Path | None, algorithm: str, *, revision: str | None = HOLOSOMA_HF_REVISION
) -> Path:
    """Return a local ``.onnx`` file for ``checkpoint``.

    * ``None`` -> :data:`HOLOSOMA_FILES`\\ ``[algorithm]`` fetched from
      :data:`HOLOSOMA_HF_REPO` at ``revision`` (default
      :data:`HOLOSOMA_HF_REVISION`) and checked against :data:`HOLOSOMA_SHA256`.
    * An existing ``.onnx`` file -> itself (a local file is the caller's claim;
      it is not hashed).
    * An existing directory -> ``<dir>/<HOLOSOMA_FILES[algorithm]>``.
    * A bare file name with no directory part -> fetched from the mirror, and
      hashed when :data:`HOLOSOMA_SHA256` knows the name.
    * A path with directories that does not exist -> refused; the caller made
      a claim about their filesystem and a download would hide a typo.

    Raises:
        FileNotFoundError: Naming the path or the mirror and the file asked for.
        RuntimeError: A fetched file whose sha256 is not the released one.
        ImportError: When a download is needed and ``huggingface_hub`` is missing
            (the remedy names the ``[holosoma]`` extra).
    """
    if checkpoint is None:
        name = HOLOSOMA_FILES[algorithm]
    else:
        path = Path(checkpoint)
        if path.is_dir():
            candidate = path / HOLOSOMA_FILES[algorithm]
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Holosoma checkpoint directory {path} has no {HOLOSOMA_FILES[algorithm]!r}. Expected the "
                    f"upstream layout models/loco/g1_29dof/ or a copy of the file; found {sorted(p.name for p in path.iterdir())}."
                )
            return candidate
        if path.exists():
            return path
        if path.parent != Path("."):
            raise FileNotFoundError(
                f"Holosoma checkpoint not found: {path}. Pass a downloaded .onnx file, the directory holding it, or a "
                f"bare name such as {HOLOSOMA_FILES[algorithm]!r} to fetch it from {HOLOSOMA_HF_REPO}."
            )
        name = path.name
    hub = require_optional(
        "huggingface_hub",
        extra="holosoma",
        purpose=f"fetching {name} from {HOLOSOMA_HF_REPO} (or pass the path of a local file)",
    )
    try:
        downloaded = hub.hf_hub_download(HOLOSOMA_HF_REPO, name, revision=revision)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - one door for every Hub-side failure
        raise FileNotFoundError(
            f"Holosoma checkpoint {name!r} could not be fetched from {HOLOSOMA_HF_REPO}: {exc}. The same file is in "
            "the Apache-2.0 tree at github.com/amazon-far/holosoma under "
            "src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/; pass its local path."
        ) from exc
    fetched = Path(downloaded)
    expected = HOLOSOMA_SHA256.get(name)
    if expected is not None:
        actual = _sha256_of(fetched)
        if actual != expected:
            raise RuntimeError(
                f"Holosoma checkpoint {name!r} fetched from {HOLOSOMA_HF_REPO} at revision {revision} hashes to "
                f"sha256 {actual}, not the released {expected} (github.com/amazon-far/holosoma). The file is not "
                "used. Pass the path of a file you trust, or a revision of the mirror that carries the released bytes."
            )
    logger.info("Holosoma checkpoint %s fetched from %s at %s", name, HOLOSOMA_HF_REPO, revision)
    return fetched


def read_onnx_metadata(path: str | Path) -> dict[str, Any]:
    """Return the ONNX ``metadata_props`` of ``path`` with JSON values decoded.

    Upstream writes ``dof_names``, ``kp``, ``kd`` and ``command_ranges`` as
    JSON strings (``holosoma_inference/policies/base.py:396-404``). Reads through ``onnxruntime``'s model
    metadata so the ``onnx`` package is not required.
    """
    ort = require_optional("onnxruntime", extra="holosoma", purpose="reading the checkpoint's gains and joint names")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # type: ignore[attr-defined]
    raw = session.get_modelmeta().custom_metadata_map
    out: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            out[key] = json.loads(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


class HolosomaPolicy(Policy):
    """ONNX whole-body locomotion policy for the Unitree G1 (Amazon FAR Holosoma).

    Args:
        checkpoint: A local ``.onnx`` file, a directory holding the upstream
            file name, a bare Hub file name, or ``None`` to fetch the file for
            ``algorithm`` from :data:`HOLOSOMA_HF_REPO`.
        algorithm: ``"fastsac"`` (default) or ``"ppo"``; selects the file when
            ``checkpoint`` does not name one.
        revision: The mirror commit a Hub fetch is pinned to; default
            :data:`HOLOSOMA_HF_REVISION`. Every fetched file is also checked
            against :data:`HOLOSOMA_SHA256`.
        config: A :class:`HolosomaConfig` or a dict of its fields. Gains
            in the config override the checkpoint's metadata (upstream order).
        target_velocity: Constructor-time default ``[vx, vy, omega]``
            (m/s, m/s, rad/s) for paths that forward constructor kwargs only
            (the mesh ``tell()``); the per-call keyword overrides it.
        driven_joints: ``"all"`` (default, upstream: the network's 29 targets
            are all commanded) or ``"legs_waist"`` (emit the first 15 only,
            the lerobot convention that leaves the arms to a teleoperator or a
            :class:`~strands_robots.policies.composite.CompositePolicy`).
        arm_observation: ``"live"`` (default, upstream: the network sees the
            measured arm state) or ``"default"`` (lerobot: the arm slots read
            ``default_angles`` and zero velocity so arm teleop does not
            disturb the gait).
        allow_missing_models: Test seam. ``True`` skips the eager load so a
            stub can be assigned to :attr:`session`; a strict boolean.
        **kwargs: Ignored (the #300 contract for registry-resolved providers).

    Raises:
        RuntimeError: ``onnxruntime`` missing, checkpoint missing, or the
            checkpoint's input width is not 100 (a whole-body-tracking file).
        ValueError: A flag outside its domain, or a config outside its domain.
    """

    requires_images = False
    reads_instruction: bool = False
    instruction_free_actions: str | None = "the gait commanded by target_velocity"
    #: Selects the MuJoCo PD-to-torque shim shared with the GR00T-WBC provider.
    pd_torque_shim: ClassVar[bool] = True
    requires_action_controller: ClassVar[str | None] = (
        "it emits joint-position targets the scene's position servos override, and the torque shim that "
        "corrects them (WBCTorqueController applies the checkpoint's per-joint PD law to a compiled MjModel) is "
        "installed by the MuJoCo engine only. Without the shim the robot falls within a fraction of a second while "
        'the rollout reports success. Run this policy on the MuJoCo backend (backend="mujoco"), or pass '
        "wbc_install_torque_control=False to drive a torque-actuated scene directly."
    )

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        algorithm: str = "fastsac",
        config: HolosomaConfig | dict[str, Any] | None = None,
        target_velocity: list[float] | None = None,
        revision: str | None = HOLOSOMA_HF_REVISION,
        driven_joints: str = "all",
        arm_observation: str = "live",
        allow_missing_models: bool = False,
        **kwargs: Any,
    ) -> None:
        if error := boolean_flag_error(allow_missing_models, "allow_missing_models", "HolosomaPolicy"):
            raise ValueError(error)
        if driven_joints not in ("all", "legs_waist"):
            raise ValueError(f"HolosomaPolicy: driven_joints must be 'all' or 'legs_waist', got {driven_joints!r}")
        if arm_observation not in ("live", "default"):
            raise ValueError(f"HolosomaPolicy: arm_observation must be 'live' or 'default', got {arm_observation!r}")
        if isinstance(config, dict):
            config = HolosomaConfig(**{**config, "algorithm": config.get("algorithm", algorithm)})
        elif config is None:
            config = HolosomaConfig(algorithm=algorithm)
        elif not isinstance(config, HolosomaConfig):
            raise ValueError(f"HolosomaPolicy: config must be a HolosomaConfig or a dict, got {type(config).__name__}")
        self._config: HolosomaConfig = config
        self._revision = revision
        self._driven_joints = driven_joints
        self._arm_observation = arm_observation
        self._default_command = self._validate_velocity(target_velocity) if target_velocity is not None else None
        self._robot_state_keys: list[str] = []
        self._joint_names: list[str] = list(HOLOSOMA_G1_JOINTS)
        self._warned_no_velocity = False
        self._warned_clipped = False
        self._last_action = np.zeros(self._config.num_actions, dtype=np.float64)
        self._phase = GaitPhase(self._config)
        self.session: Any = None
        self.checkpoint_path: Path | None = None
        self._input_name = "actor_obs"
        if not allow_missing_models:
            self._load_session(checkpoint)
        self._default_angles = np.asarray(self._config.default_angles, dtype=np.float64)
        self._kps = np.asarray(self._config.kps, dtype=np.float64) if self._config.kps else None
        self._kds = np.asarray(self._config.kds, dtype=np.float64) if self._config.kds else None
        if kwargs:
            logger.debug("HolosomaPolicy ignoring unknown constructor kwargs: %s", sorted(kwargs))
        logger.info(
            "HolosomaPolicy ready [algorithm=%s driven=%s arm_obs=%s gains=%s default_cmd=%s]",
            self._config.algorithm,
            driven_joints,
            arm_observation,
            "config" if self._kps is not None else "unloaded",
            self._default_command.tolist() if self._default_command is not None else None,
        )

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"holosoma"``)."""
        return "holosoma"

    @property
    def config(self) -> HolosomaConfig:
        """Resolved configuration (gains filled from the checkpoint once loaded)."""
        return self._config

    @property
    def default_angles(self) -> NDArray[np.float64]:
        """Nominal stance for the 29 joints (what the torque shim holds before the first action)."""
        return self._default_angles

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Require every G1 joint name in ``robot_state_keys`` (the non-G1 refusal)."""
        keys = list(robot_state_keys)
        key_set = set(keys)
        missing = [name for name in HOLOSOMA_G1_JOINTS if name not in key_set]
        if missing:
            raise ValueError(
                "HolosomaPolicy: the robot's joint list is missing expected G1 joints: "
                f"{missing}.\n  expected (dof_names order): {list(HOLOSOMA_G1_JOINTS)}\n  robot provided: {keys}\n"
                "The released Holosoma checkpoints drive the 29-DOF Unitree G1 only; load unitree_g1."
            )
        self._robot_state_keys = keys

    def reset(self, seed: int | None = None) -> None:
        """Zero the previous action and restart the gait clock (``seed`` unused: the ONNX is deterministic)."""
        self._last_action = np.zeros(self._config.num_actions, dtype=np.float64)
        self._phase.reset()
        logger.debug("HolosomaPolicy.reset (seed=%s)", sanitize_log_value(repr(seed)))

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """One control tick: advance the clock, build the observation, run the network, emit targets."""
        if self.session is None:
            raise RuntimeError(
                "HolosomaPolicy has no ONNX session. Construct without allow_missing_models=True so the checkpoint "
                "loads, or assign a session to policy.session before the first get_actions."
            )
        lin_vel, ang_vel = self._resolve_command(kwargs)
        qj, dqj, base_ang_vel, quat = self._extract_state(observation_dict)
        phase = self._phase.step(lin_vel, ang_vel)
        obs = build_actor_obs(
            self._config,
            last_action=self._last_action,
            base_ang_vel=base_ang_vel,
            command_lin_vel=lin_vel,
            command_ang_vel=ang_vel,
            phase=phase,
            qj=qj,
            dqj=dqj,
            proj_gravity=projected_gravity(quat),
        )
        raw = self.session.run(None, {self._input_name: obs.reshape(1, -1)})[0]
        action = np.asarray(raw, dtype=np.float64).ravel()
        if action.shape[0] != self._config.num_actions:
            raise RuntimeError(
                f"HolosomaPolicy: the network returned {action.shape[0]} actions, expected {self._config.num_actions}."
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("HolosomaPolicy: the network returned a non-finite action; refusing to command it.")
        clip = self._config.action_clip
        action = np.asarray(np.clip(action, -clip, clip), dtype=np.float64).ravel()
        self._last_action = action.copy()
        target_q = compute_targets(self._default_angles, action, self._config.action_scale)
        names = self._joint_names if self._driven_joints == "all" else self._joint_names[:15]
        return [{name: float(v) for name, v in zip(names, target_q[: len(names)], strict=True)}]

    # ------------------------------------------------------------------
    # Public helpers (read by the MuJoCo torque shim)
    # ------------------------------------------------------------------

    def compute_torques(self, target_pos: np.ndarray, current_pos: np.ndarray, current_vel: np.ndarray) -> np.ndarray:
        """``tau = kp * (target - q) - kd * dq`` with the checkpoint's per-joint gains.

        Raises:
            RuntimeError: When no gains are known (no checkpoint loaded and no
                ``kps``/``kds`` in the config): upstream refuses the same way
                (``_resolve_control_gains``: "No KP/KD values found").
        """
        if self._kps is None or self._kds is None:
            raise RuntimeError(
                "HolosomaPolicy.compute_torques: no PD gains. They come from the checkpoint's ONNX metadata (kp/kd) "
                "or from HolosomaConfig(kps=..., kds=...); neither is present."
            )
        target_pos = np.asarray(target_pos, dtype=np.float64)
        current_pos = np.asarray(current_pos, dtype=np.float64)
        current_vel = np.asarray(current_vel, dtype=np.float64)
        n = target_pos.shape[0]
        zeros = np.zeros_like(target_pos)
        return pd_control(target_pos, current_pos, self._kps[:n], zeros, current_vel, self._kds[:n])

    @property
    def kps(self) -> NDArray[np.float64] | None:
        """Per-joint proportional gains, or ``None`` before a checkpoint is loaded."""
        return self._kps

    @property
    def kds(self) -> NDArray[np.float64] | None:
        """Per-joint derivative gains, or ``None`` before a checkpoint is loaded."""
        return self._kds

    def apply_metadata(self, metadata: dict[str, Any]) -> None:
        """Take gains, joint names and command ranges from a checkpoint's metadata.

        Config gains win over metadata gains (upstream order). A ``dof_names``
        list that is not the G1 table is refused: the network's output order
        would not be the joint order we emit.
        """
        names = metadata.get("dof_names")
        if names is not None and list(names) != list(HOLOSOMA_G1_JOINTS):
            raise RuntimeError(
                "HolosomaPolicy: the checkpoint's dof_names are not the Unitree G1 29-DOF table this provider emits. "
                f"Checkpoint: {list(names)[:5]}... ({len(names)} names)."
            )
        if not self._config.kps and "kp" in metadata and "kd" in metadata:
            kp = np.asarray(metadata["kp"], dtype=np.float64).ravel()
            kd = np.asarray(metadata["kd"], dtype=np.float64).ravel()
            if kp.shape[0] != self._config.num_actions or kd.shape[0] != self._config.num_actions:
                raise RuntimeError(
                    f"HolosomaPolicy: checkpoint gains have {kp.shape[0]}/{kd.shape[0]} entries, expected {self._config.num_actions}."
                )
            self._config = self._config.with_gains(tuple(kp.tolist()), tuple(kd.tolist()))
        ranges = metadata.get("command_ranges")
        if isinstance(ranges, dict):
            merged = dict(self._config.command_ranges)
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
                if key in ranges and sequence_length(ranges[key]) == 2:
                    merged[key] = (float(ranges[key][0]), float(ranges[key][1]))
            self._config = HolosomaConfig(**{**self._config.__dict__, "command_ranges": merged})
        self._kps = np.asarray(self._config.kps, dtype=np.float64) if self._config.kps else None
        self._kds = np.asarray(self._config.kds, dtype=np.float64) if self._config.kds else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_session(self, checkpoint: str | Path | None) -> None:
        try:
            ort = require_optional("onnxruntime", extra="holosoma", purpose="running the Holosoma ONNX controller")
        except ImportError as exc:
            raise RuntimeError(
                f"HolosomaPolicy requires onnxruntime (the [holosoma] extra) but it is not installed.\n{exc}"
            ) from exc
        try:
            path = resolve_holosoma_checkpoint(checkpoint, self._config.algorithm, revision=self._revision)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        options = ort.SessionOptions()  # type: ignore[attr-defined]
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])  # type: ignore[attr-defined]
        inputs = session.get_inputs()
        width = inputs[0].shape[-1] if inputs and inputs[0].shape else None
        if len(inputs) != 1 or width != HOLOSOMA_OBS_DIM:
            raise RuntimeError(
                f"HolosomaPolicy: {path.name} declares input {[(i.name, i.shape) for i in inputs]}; the G1 locomotion "
                f"contract is one input [1, {HOLOSOMA_OBS_DIM}]. The whole-body-tracking files (*_dancing.onnx) have a "
                "different observation and are not locomotion controllers."
            )
        self._input_name = inputs[0].name
        raw_meta = session.get_modelmeta().custom_metadata_map
        metadata: dict[str, Any] = {}
        for key, value in raw_meta.items():
            try:
                metadata[key] = json.loads(value)
            except (TypeError, ValueError):
                metadata[key] = value
        self.session = session
        self.checkpoint_path = path
        self.apply_metadata(metadata)
        if self._kps is None:
            raise RuntimeError(
                f"HolosomaPolicy: {path.name} carries no kp/kd metadata and the config names no gains; upstream refuses "
                "this too. Pass HolosomaConfig(kps=..., kds=...) or a checkpoint exported with gains."
            )

    def _resolve_command(self, kwargs: dict[str, Any]) -> tuple[NDArray[np.float64], float]:
        tv = kwargs.get("target_velocity")
        if tv is None:
            cmd = self._default_command if self._default_command is not None else np.zeros(3, dtype=np.float64)
        else:
            cmd = self._validate_velocity(tv)
        ranges = self._config.command_ranges
        bounds = [ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_yaw"]]
        clipped = np.array([min(max(float(cmd[i]), lo), hi) for i, (lo, hi) in enumerate(bounds)], dtype=np.float64)
        if not self._warned_clipped and np.any(clipped != cmd[:3]):
            self._warned_clipped = True
            logger.warning(
                "HolosomaPolicy: target_velocity %s clipped to the checkpoint's command ranges %s",
                sanitize_log_value(repr(cmd[:3].tolist())),
                bounds,
            )
        return clipped[:2], float(clipped[2])

    @staticmethod
    def _validate_velocity(tv: Any) -> NDArray[np.float64]:
        try:
            arr = np.asarray(tv, dtype=np.float64).ravel()
        except (TypeError, ValueError) as e:
            raise ValueError(f"target_velocity must be a numeric sequence, got {tv!r}") from e
        if arr.shape[0] < 3:
            raise ValueError(f"target_velocity must have at least 3 elements [vx, vy, omega], got {arr.shape[0]}")
        for i, v in enumerate(arr[:3]):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"target_velocity[{i}]={v!r} must be finite")
            if error := target_velocity_component_error(i, float(v), "HolosomaPolicy"):
                raise ValueError(error)
        return arr[:3]

    def _extract_state(
        self, obs: dict[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Read (qj, dqj, base_ang_vel, base_quat) from either observation shape.

        Sim schema: ``obs[name]`` / ``obs[name + ".vel"]`` per joint plus
        ``base_quat`` (wxyz) and ``base_ang_vel`` (body frame). Hardware
        schema (:mod:`strands_robots.drivers.g1` snapshot): ``obs["joints"][name]["q"|"dq"]``
        and ``obs["imu"]["quaternion"|"gyroscope"]``. Flat
        ``observation.state`` / ``observation.velocity`` vectors are indexed by
        the robot's key list when one was set, else positionally.
        """
        names = self._joint_names
        n = len(names)
        qj: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        dqj: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        joints = obs.get("joints")
        has_vel = False
        if isinstance(joints, dict):
            for i, name in enumerate(names):
                rec = joints.get(name)
                if isinstance(rec, dict):
                    qj[i] = self._num(rec.get("q"), 0.0)
                    if rec.get("dq") is not None:
                        has_vel = True
                        dqj[i] = self._num(rec.get("dq"), 0.0)
        else:
            qj = self._read_joint_vector(obs, "position", names)
            dqj = self._read_joint_vector(obs, "velocity", names)
            has_vel = obs.get("observation.velocity") is not None or any(f"{k}.vel" in obs for k in names)
        imu_raw = obs.get("imu")
        imu: dict[str, Any] = imu_raw if isinstance(imu_raw, dict) else {}
        base_ang_vel = self._read_vec(obs, ("base_ang_vel", "observation.base_ang_vel"), 3)
        if base_ang_vel is None:
            base_ang_vel = self._read_vec(imu, ("gyroscope",), 3)
        if base_ang_vel is None:
            base_ang_vel = np.zeros(3, dtype=np.float64)
        elif float(np.linalg.norm(base_ang_vel)) > 0.0:
            has_vel = True
        quat = self._read_vec(obs, ("base_quat", "observation.base_quat"), 4)
        if quat is None:
            quat = self._read_vec(imu, ("quaternion",), 4)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self._arm_observation == "default":
            for i, name in enumerate(names):
                if name in _ARM_JOINTS:
                    qj[i] = self._default_angles[i]
                    dqj[i] = 0.0
        if not has_vel and not self._warned_no_velocity:
            self._warned_no_velocity = True
            logger.warning(
                "HolosomaPolicy: observation exposes no joint velocities or base angular velocity; feeding zeros. "
                "Supply per-joint '<name>.vel' keys, 'observation.velocity' and/or 'base_ang_vel' to close the loop."
            )
        return qj, dqj, base_ang_vel, quat

    @staticmethod
    def _num(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_joint_vector(self, obs: dict[str, Any], kind: str, names: list[str]) -> NDArray[np.float64]:
        m = len(names)
        flat_key = "observation.state" if kind == "position" else "observation.velocity"
        flat = obs.get(flat_key)
        if flat is not None and sequence_length(flat) is not None:
            arr = np.asarray(flat if not hasattr(flat, "tolist") else flat.tolist(), dtype=np.float64).ravel()
            if self._robot_state_keys:
                index_of: dict[str, int] = {}
                for i, name in enumerate(self._robot_state_keys):
                    index_of.setdefault(name, i)
                out = np.zeros(m, dtype=np.float64)
                for i, name in enumerate(names):
                    j = index_of.get(name)
                    if j is not None and j < arr.shape[0]:
                        out[i] = arr[j]
                return out
            out = np.zeros(m, dtype=np.float64)
            k = min(m, arr.shape[0])
            out[:k] = arr[:k]
            return out
        out = np.zeros(m, dtype=np.float64)
        for i, name in enumerate(names):
            v = obs.get(name) if kind == "position" else obs.get(f"{name}.vel")
            if v is not None:
                out[i] = self._num(v, 0.0)
        return out

    @staticmethod
    def _read_vec(obs: dict[str, Any], keys: tuple[str, ...], n: int) -> NDArray[np.float64] | None:
        for k in keys:
            v = obs.get(k)
            if v is not None and sequence_length(v) == n:
                return np.asarray(v if not hasattr(v, "tolist") else v.tolist(), dtype=np.float64).ravel()
        return None


__all__ = [
    "HOLOSOMA_FILES",
    "HOLOSOMA_G1_JOINTS",
    "HOLOSOMA_HF_REPO",
    "HOLOSOMA_HF_REVISION",
    "HOLOSOMA_SHA256",
    "HolosomaPolicy",
    "read_onnx_metadata",
    "resolve_holosoma_checkpoint",
]
