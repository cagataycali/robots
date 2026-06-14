"""WBCPolicy - GR00T Whole-Body-Control decoupled locomotion policy.

Runs ONNX models (Balance + Walk) for lower-body control of humanoid robots.
Maintains a 6-step observation history (86-dim per step, 516-dim flattened
input to ONNX). Auto-switches between Balance and Walk based on velocity
command magnitude.

Observation schema (single_obs_dim=86, x6 history = 516):
    [ 0: 7]  command = [vx*2, vy*2, wz*0.5, height, roll, pitch, yaw]
    [ 7:10]  base_angular_velocity * 0.5
    [10:13]  gravity_orientation = quat_rotate_inverse(base_quat, [0,0,-1])
    [13:42]  (joint_positions - default_angles) * 1.0  (29 joints)
    [42:71]  joint_velocities * 0.05                   (29 joints)
    [71:86]  previous_action                           (15-dim)

ONNX output: [B, 15] float32, scaled by 0.25 and added to default joint angles.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)

# 15 joints controlled by WBC (12 leg + 3 waist), in ONNX output order.
WBC_JOINT_NAMES: list[str] = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
]

# Default standing joint angles for the 15 controlled joints (radians).
# These are the nominal positions used by the GR00T-WBC ONNX models.
_DEFAULT_ANGLES_15: np.ndarray = np.array(
    [
        -0.1,  # left_hip_pitch
        0.0,  # left_hip_roll
        0.0,  # left_hip_yaw
        0.3,  # left_knee
        -0.2,  # left_ankle_pitch
        0.0,  # left_ankle_roll
        -0.1,  # right_hip_pitch
        0.0,  # right_hip_roll
        0.0,  # right_hip_yaw
        0.3,  # right_knee
        -0.2,  # right_ankle_pitch
        0.0,  # right_ankle_roll
        0.0,  # waist_yaw
        0.0,  # waist_roll
        0.0,  # waist_pitch
    ],
    dtype=np.float32,
)

# Observation dimensions
_SINGLE_OBS_DIM = 86
_HISTORY_LEN = 6
_OBS_INPUT_DIM = _SINGLE_OBS_DIM * _HISTORY_LEN  # 516
_ACTION_DIM = 15
_N_JOINTS_FULL = 29  # Full G1 joint count

# Scaling factors (from verified config)
_CMD_SCALE = np.array([2.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_ANG_VEL_SCALE = 0.5
_DOF_POS_SCALE = 1.0
_DOF_VEL_SCALE = 0.05
_ACTION_SCALE = 0.25

# Velocity threshold for Balance vs Walk auto-switch
_WALK_THRESHOLD = 0.05


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion (wxyz convention from MuJoCo).

    MuJoCo uses [w, x, y, z] quaternion format.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Inverse rotation: conjugate of unit quaternion
    # q_conj = [w, -x, -y, -z]
    # Rotate vec by q_conj: q_conj * [0, vec] * q
    t = 2.0 * np.cross(np.array([-x, -y, -z]), vec)
    return vec + w * t + np.cross(np.array([-x, -y, -z]), t)


class WBCPolicy(Policy):
    """GR00T Whole-Body-Control decoupled locomotion policy.

    Lightweight ONNX-only policy for humanoid lower-body control.
    No PyTorch dependency required - uses onnxruntime only.

    Args:
        checkpoint: HuggingFace repo ID or local directory containing ONNX files.
            Default: "nvidia/GR00T-WholeBodyControl"
        variant: Which ONNX to use - "balance", "walk", or "auto" (default).
            "auto" switches based on velocity command magnitude.
        balance_onnx: Override path to balance ONNX file.
        walk_onnx: Override path to walk ONNX file.
        device: ONNX execution provider - "cpu" (default) or "cuda".
        default_angles: Override default standing angles (15-dim).
        action_scale: Scale factor for ONNX output (default 0.25).
        allow_missing_models: If True, allow construction without ONNX models
            (returns default-pose actions). Default False - raises RuntimeError
            when models cannot be loaded, preventing silent safety hazards on
            bipedal humanoids.
    """

    def __init__(
        self,
        checkpoint: str = "nvidia/GR00T-WholeBodyControl",
        variant: str = "auto",
        balance_onnx: str | None = None,
        walk_onnx: str | None = None,
        device: str = "cpu",
        default_angles: list[float] | np.ndarray | None = None,
        action_scale: float = _ACTION_SCALE,
        allow_missing_models: bool = False,
        **kwargs: Any,
    ) -> None:
        self._checkpoint = checkpoint
        self._variant = variant
        self._device = device
        self._action_scale = action_scale
        self._allow_missing_models = allow_missing_models
        self._robot_state_keys: list[str] = []

        # Default angles for the 15 controlled joints
        if default_angles is not None:
            self._default_angles = np.asarray(default_angles, dtype=np.float32)
        else:
            self._default_angles = _DEFAULT_ANGLES_15.copy()

        # Padded defaults for full 29-joint observation encoding
        self._default_angles_padded = np.zeros(_N_JOINTS_FULL, dtype=np.float32)
        n = min(len(self._default_angles), _N_JOINTS_FULL)
        self._default_angles_padded[:n] = self._default_angles[:n]

        # Observation history buffer (6 frames of 86-dim obs)
        self._obs_history: deque[np.ndarray] = deque(maxlen=_HISTORY_LEN)
        self._prev_action = np.zeros(_ACTION_DIM, dtype=np.float32)

        # ONNX sessions (lazy-loaded)
        self._balance_session: Any = None
        self._walk_session: Any = None
        self._balance_onnx_path = balance_onnx
        self._walk_onnx_path = walk_onnx

        # Load ONNX models
        self._load_models()

        logger.info(
            "WBCPolicy initialized: checkpoint=%s, variant=%s, device=%s",
            checkpoint,
            variant,
            device,
        )

    def _load_models(self) -> None:
        """Load ONNX models from checkpoint path or HuggingFace."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime is required for WBCPolicy. Install with: pip install onnxruntime") from e

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._device == "cuda" else ["CPUExecutionProvider"]
        )

        # Resolve ONNX file paths
        balance_path = self._resolve_onnx_path("balance")
        walk_path = self._resolve_onnx_path("walk")

        if balance_path and balance_path.exists():
            self._balance_session = ort.InferenceSession(str(balance_path), providers=providers)
            logger.info("Loaded balance ONNX: %s", balance_path)

        if walk_path and walk_path.exists():
            self._walk_session = ort.InferenceSession(str(walk_path), providers=providers)
            logger.info("Loaded walk ONNX: %s", walk_path)

        if self._balance_session is None and self._walk_session is None:
            if self._allow_missing_models:
                logger.warning(
                    "No ONNX models loaded from checkpoint '%s'. "
                    "allow_missing_models=True: policy will return default-pose actions. "
                    "Download with: huggingface-cli download %s",
                    self._checkpoint,
                    self._checkpoint,
                )
            else:
                raise RuntimeError(
                    f"WBCPolicy failed to load any ONNX models from checkpoint "
                    f"'{self._checkpoint}'. Both Balance and Walk models are unavailable. "
                    f"Refusing to construct a policy that would silently command "
                    f"default-pose targets on a bipedal humanoid. "
                    f"Download models with: huggingface-cli download {self._checkpoint} "
                    f"-- or pass allow_missing_models=True for offline testing."
                )

    def _resolve_onnx_path(self, variant: str) -> Path | None:
        """Resolve path to an ONNX file for the given variant."""
        # Direct override paths
        if variant == "balance" and self._balance_onnx_path:
            return Path(self._balance_onnx_path)
        if variant == "walk" and self._walk_onnx_path:
            return Path(self._walk_onnx_path)

        # Standard filenames in the checkpoint directory
        filename = f"GR00T-WholeBodyControl-{'Balance' if variant == 'balance' else 'Walk'}.onnx"

        # Check local directory
        checkpoint_dir = Path(self._checkpoint)
        if checkpoint_dir.is_dir():
            candidate = checkpoint_dir / filename
            if candidate.exists():
                return candidate
            # Also check nested paths
            for sub in ["decoupled_wbc", "models", "."]:
                candidate = checkpoint_dir / sub / filename
                if candidate.exists():
                    return candidate

        # Check HuggingFace cache
        cache_dir = Path.home() / ".cache" / "strands_robots" / "wbc"
        candidate = cache_dir / filename
        if candidate.exists():
            return candidate

        # Try huggingface_hub download
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self._checkpoint,
                filename=filename,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir),
            )
            return Path(path)
        except (OSError, ImportError, ValueError, ConnectionError) as e:
            logger.debug("HuggingFace download failed for %s/%s: %s", self._checkpoint, filename, e)

        return None

    @property
    def provider_name(self) -> str:
        return "wbc"

    @property
    def requires_images(self) -> bool:
        """WBC is state-only - no camera frames needed."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure output joint names."""
        self._robot_state_keys = robot_state_keys

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state (observation history + previous action)."""
        self._obs_history.clear()
        self._prev_action = np.zeros(_ACTION_DIM, dtype=np.float32)

    def _compute_single_obs(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        base_quat: np.ndarray,
        base_angular_velocity: np.ndarray,
        target_velocity: np.ndarray,
        target_height: float,
        target_rpy: np.ndarray,
    ) -> np.ndarray:
        """Compute a single 86-dim observation vector.

        Args:
            joint_positions: 29-dim joint positions (qpos[7:36] for G1).
            joint_velocities: 29-dim joint velocities (qvel[6:35] for G1).
            base_quat: 4-dim quaternion [w, x, y, z] (MuJoCo convention).
            base_angular_velocity: 3-dim angular velocity of the base.
            target_velocity: [vx, vy, wz] locomotion command.
            target_height: Desired pelvis height (metres).
            target_rpy: [roll, pitch, yaw] torso orientation target.
        """
        obs = np.zeros(_SINGLE_OBS_DIM, dtype=np.float32)

        # [0:7] Command (scaled)
        command = np.zeros(7, dtype=np.float32)
        command[0] = target_velocity[0]  # vx
        command[1] = target_velocity[1]  # vy
        command[2] = target_velocity[2]  # wz
        command[3] = target_height
        command[4:7] = target_rpy
        obs[0:7] = command * _CMD_SCALE

        # [7:10] Base angular velocity (scaled)
        obs[7:10] = base_angular_velocity * _ANG_VEL_SCALE

        # [10:13] Gravity orientation in body frame
        obs[10:13] = _quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

        # [13:42] Joint position error (29 joints)
        n_joints = min(len(joint_positions), _N_JOINTS_FULL)
        obs[13 : 13 + n_joints] = (joint_positions[:n_joints] - self._default_angles_padded[:n_joints]) * _DOF_POS_SCALE

        # [42:71] Joint velocities (29 joints, scaled)
        n_vel = min(len(joint_velocities), _N_JOINTS_FULL)
        obs[42 : 42 + n_vel] = joint_velocities[:n_vel] * _DOF_VEL_SCALE

        # [71:86] Previous action
        obs[71:86] = self._prev_action

        return obs

    def _select_session(self, target_velocity: np.ndarray) -> Any:
        """Select Balance or Walk ONNX session based on velocity magnitude."""
        if self._variant == "balance":
            return self._balance_session
        if self._variant == "walk":
            return self._walk_session

        # Auto mode: switch based on velocity magnitude
        vel_magnitude = np.linalg.norm(target_velocity)
        if vel_magnitude > _WALK_THRESHOLD:
            return self._walk_session or self._balance_session
        return self._balance_session or self._walk_session

    def _run_inference(self, obs_input: np.ndarray, session: Any) -> np.ndarray:
        """Run ONNX inference and return raw 15-dim action."""
        if session is None:
            return np.zeros(_ACTION_DIM, dtype=np.float32)

        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: obs_input.reshape(1, -1)})
        return result[0].flatten().astype(np.float32)

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Compute WBC action from observation state.

        Expected observation_dict keys:
            - "observation.state": joint positions (29-dim) OR full state vector
            - "observation.velocity": joint velocities (29-dim), optional
            - "observation.base_quat": base quaternion [w,x,y,z], optional
            - "observation.base_angular_velocity": 3-dim, optional

        Kwargs (well-known WBC keys):
            - target_velocity: [vx, vy, wz] in m/s, m/s, rad/s
            - target_height: desired pelvis height in metres (default 0.74)
            - target_rpy: [roll, pitch, yaw] target (default [0,0,0])
        """
        # Extract locomotion commands from kwargs
        target_velocity = np.asarray(kwargs.get("target_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        target_height = float(kwargs.get("target_height", 0.74))
        target_rpy = np.asarray(kwargs.get("target_rpy", [0.0, 0.0, 0.0]), dtype=np.float32)

        # Extract state from observation
        state = observation_dict.get("observation.state", np.zeros(_N_JOINTS_FULL))
        state = np.asarray(state, dtype=np.float32)

        # Parse joint positions and velocities
        if "observation.velocity" in observation_dict:
            joint_positions = state[:_N_JOINTS_FULL] if len(state) >= _N_JOINTS_FULL else state
            joint_velocities = np.asarray(observation_dict["observation.velocity"], dtype=np.float32)
        elif len(state) >= 2 * _N_JOINTS_FULL:
            # State contains both positions and velocities concatenated
            joint_positions = state[:_N_JOINTS_FULL]
            joint_velocities = state[_N_JOINTS_FULL : 2 * _N_JOINTS_FULL]
        else:
            joint_positions = (
                state[:_N_JOINTS_FULL]
                if len(state) >= _N_JOINTS_FULL
                else np.pad(state, (0, _N_JOINTS_FULL - len(state)))
            )
            joint_velocities = np.zeros(_N_JOINTS_FULL, dtype=np.float32)

        # Base quaternion and angular velocity
        base_quat = np.asarray(
            observation_dict.get("observation.base_quat", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        base_angular_velocity = np.asarray(
            observation_dict.get("observation.base_angular_velocity", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )

        # Compute single observation
        single_obs = self._compute_single_obs(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_quat=base_quat,
            base_angular_velocity=base_angular_velocity,
            target_velocity=target_velocity,
            target_height=target_height,
            target_rpy=target_rpy,
        )

        # Append to history
        self._obs_history.append(single_obs)

        # Pad history if not full yet (repeat first observation)
        while len(self._obs_history) < _HISTORY_LEN:
            self._obs_history.appendleft(self._obs_history[0].copy())

        # Flatten history into 516-dim input
        obs_input = np.concatenate(list(self._obs_history), axis=0).astype(np.float32)
        assert obs_input.shape[0] == _OBS_INPUT_DIM, (
            f"Observation history flattened to {obs_input.shape[0]}-dim, expected {_OBS_INPUT_DIM}-dim (6 x 86)"
        )

        # Select and run ONNX session
        session = self._select_session(target_velocity)
        raw_action = self._run_inference(obs_input, session)

        # Scale action and add to default angles
        action = raw_action * self._action_scale + self._default_angles
        self._prev_action = raw_action.copy()

        # Map to joint name dict
        keys = self._robot_state_keys if self._robot_state_keys else WBC_JOINT_NAMES
        action_dict: dict[str, Any] = {}
        for i, key in enumerate(keys[:_ACTION_DIM]):
            action_dict[key] = float(action[i])

        return [action_dict]

    # ------------------------------------------------------------------
    # PD torque control helpers (for torque-actuated MuJoCo models)
    # ------------------------------------------------------------------

    # Default PD gains from the verified GR00T-WBC config (g1_gear_wbc.yaml).
    # These are appropriate for the G1 humanoid with gear-ratio actuators.
    _DEFAULT_KPS = np.array(
        [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250],
        dtype=np.float32,
    )
    _DEFAULT_KDS = np.array(
        [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0],
        dtype=np.float32,
    )

    def compute_torques(
        self,
        target_positions: np.ndarray,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute PD torques from target positions.

        The WBC ONNX models output target joint positions. When driving
        torque-actuated MuJoCo models (like `g1_gear_wbc.xml`), these must
        be converted to torques via a PD controller:

            tau = kp * (target - current_pos) - kd * current_vel

        Args:
            target_positions: 15-dim target joint positions from get_actions.
            current_positions: 15-dim current joint positions from sim.
            current_velocities: 15-dim current joint velocities from sim.
            kps: 15-dim proportional gains (default: verified G1 gains).
            kds: 15-dim derivative gains (default: verified G1 gains).

        Returns:
            15-dim torque commands for the MuJoCo actuators.
        """
        if kps is None:
            kps = self._DEFAULT_KPS
        if kds is None:
            kds = self._DEFAULT_KDS
        return kps * (target_positions - current_positions) - kds * current_velocities
