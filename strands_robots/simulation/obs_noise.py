"""Additive sensor noise on observations, shared by every backend.

``set_obs_noise`` stores one config and one seeded RNG; the three appliers read
them. MuJoCo, Isaac and Newton all mix :class:`ObservationNoiseMixin` in, so an
identical call validates, reports and perturbs identically on each - which is
the promise the per-backend copies made in their docstrings and only mostly
kept.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.base import (
    finite_non_negative_error,
    randomization_seed_error,
    unknown_kwargs_error,
)

#: The keywords ``set_obs_noise`` honors. The method declares ``**kwargs`` to
#: match the ``**kwargs``-typed ``SimEngine.set_obs_noise``; anything outside
#: this tuple is a caller mistake and is refused by name, not dropped.
OBS_NOISE_PARAMS: tuple[str, ...] = (
    "joint_pos_std",
    "joint_vel_std",
    "camera_jitter_px",
    "seed",
)


class ObservationNoiseMixin:
    """``set_obs_noise`` and the passes that apply it.

    Expects the host to provide ``_lock``. The state it owns - ``_obs_noise``
    (``None`` when no noise is configured) and ``_obs_noise_rng`` - is read with
    ``getattr`` so an engine that never configured noise needs no seeding.
    """

    if TYPE_CHECKING:
        _lock: threading.RLock
        _obs_noise: dict[str, float] | None
        _obs_noise_rng: np.random.Generator | None

    def set_obs_noise(
        self,
        joint_pos_std: float = 0.0,
        joint_vel_std: float = 0.0,
        camera_jitter_px: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure additive Gaussian sensor noise on observations.

        Models real-encoder / real-camera measurement noise so a policy is not
        trained or evaluated on noise-free sensing. Once set, the noise is
        applied on every observation read and rendered camera frame the backend
        routes through this mixin, until reconfigured. All-zero stds clear it,
        after which every read is byte-for-byte what it was unconfigured.

        Args:
            joint_pos_std: Std (radians) of Gaussian noise on joint positions.
            joint_vel_std: Std (rad/s) of Gaussian noise on joint velocities -
                the ``<joint>.vel`` observation entries and the ``velocity``
                field of ``get_robot_state``.
            camera_jitter_px: Max integer pixel shift applied to rendered
                frames (uniform in ``[-px, px]`` per axis).
            seed: Optional seed for a reproducible noise stream; a non-negative
                integer, or None for fresh entropy. Validated here rather than
                where the stream is first drawn, so an unusable seed is reported
                by the call that supplied it.
            **kwargs: Declared only to match the ``SimEngine`` signature;
                any keyword arriving here is refused naming the valid ones.

        Returns:
            A success envelope whose ``json`` block echoes the stored config and
            seed, or an error envelope for an unknown keyword or a negative or
            non-finite value.
        """
        if err := unknown_kwargs_error("set_obs_noise", kwargs, OBS_NOISE_PARAMS):
            return err
        for param, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            if msg := finite_non_negative_error(value, param, "set_obs_noise"):
                return {"status": "error", "content": [{"text": msg}]}
        if msg := randomization_seed_error(seed, "set_obs_noise"):
            return {"status": "error", "content": [{"text": msg}]}
        cfg = {
            "joint_pos_std": float(joint_pos_std),
            "joint_vel_std": float(joint_vel_std),
            "camera_jitter_px": float(camera_jitter_px),
        }
        with self._lock:
            if any(v > 0 for v in cfg.values()):
                self._obs_noise = cfg
                self._obs_noise_rng = np.random.default_rng(seed)
                text = (
                    f"Sensor noise: joint_pos_std={cfg['joint_pos_std']}, "
                    f"joint_vel_std={cfg['joint_vel_std']}, camera_jitter_px={cfg['camera_jitter_px']}"
                )
            else:
                self._obs_noise = None
                self._obs_noise_rng = None
                text = "Sensor noise cleared."
        return {"status": "success", "content": [{"text": text}, {"json": {**cfg, "seed": seed}}]}

    def _noise_config(self) -> tuple[dict[str, float], np.random.Generator | None]:
        return getattr(self, "_obs_noise", None) or {}, getattr(self, "_obs_noise_rng", None)

    def _apply_obs_noise(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return ``obs`` with the configured noise applied, suffix-keyed.

        ``joint_pos_std`` goes to the plain float entries, ``joint_vel_std`` to
        the ``<joint>.vel`` floats, ``camera_jitter_px`` to ndarray frames.
        List values (the floating-base ``base_*`` signals) are left untouched -
        a quaternion would need renormalization, out of scope for additive
        scalar noise. Returns the input itself when nothing is configured.
        """
        cfg, rng = self._noise_config()
        if rng is None or not cfg or not obs:
            return obs
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        out: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                out[key] = self._maybe_jitter_frame(value)
            elif isinstance(value, float):
                std = vel_std if key.endswith(".vel") else pos_std
                out[key] = value + (float(rng.normal(0.0, std)) if std > 0 else 0.0)
            else:
                out[key] = value
        return out

    def _apply_state_noise(self, state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Return ``get_robot_state`` entries (``{joint: {"position", "velocity"}}``) with noise.

        Position noise uses ``joint_pos_std``, velocity noise ``joint_vel_std``.
        Returns the input itself when neither std is positive.
        """
        cfg, rng = self._noise_config()
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        if rng is None or (pos_std <= 0 and vel_std <= 0) or not state:
            return state
        out: dict[str, dict[str, float]] = {}
        for jname, vals in state.items():
            pos = vals["position"] + (float(rng.normal(0.0, pos_std)) if pos_std > 0 else 0.0)
            vel = vals["velocity"] + (float(rng.normal(0.0, vel_std)) if vel_std > 0 else 0.0)
            out[jname] = {"position": pos, "velocity": vel}
        return out

    def _maybe_jitter_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` rolled by a random integer pixel offset per axis.

        Returns the input itself when jitter is off or under one pixel.
        """
        cfg, rng = self._noise_config()
        max_shift = int(cfg.get("camera_jitter_px", 0.0))
        if max_shift < 1 or rng is None or frame.ndim < 2:
            return frame
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        return np.roll(frame, shift=(dy, dx), axis=(0, 1))
