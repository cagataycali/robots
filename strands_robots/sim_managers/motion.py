"""Reference-motion support for whole-body-tracking (WBT) terms.

Whole-body tracking trains a policy to imitate a reference motion: at each
control step the robot is rewarded for matching a per-joint target pose (and
velocity) sampled from a *motion clip*. This module provides the
backend-agnostic plumbing shared by the WBT command / observation / reward /
termination terms:

- :class:`MotionClip` - an immutable reference trajectory (joint positions, and
  optionally velocities) sampled at a fixed ``fps``, with phase-based
  interpolation. It carries no simulator state, so the same clip drives a
  MuJoCo, Isaac, or Newton rollout.
- The :data:`MOTION_TARGET_POS` / :data:`MOTION_TARGET_VEL` / :data:`MOTION_PHASE`
  keys, written into :attr:`EnvState.extras` by the ``motion_clip`` command term
  and read by the WBT observation / reward / termination terms. Centralising the
  keys keeps the producer and consumers in agreement (one source of truth) and
  lets a consumer raise a clear, actionable error when no command term populated
  them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, FloatArray

# Keys the ``motion_clip`` command term writes into ``EnvState.extras`` and the
# WBT observation / reward / termination terms read back.
MOTION_TARGET_POS = "motion_target_pos"
MOTION_TARGET_VEL = "motion_target_vel"
MOTION_PHASE = "motion_phase"


def read_motion_target(state: EnvState, key: str) -> Any:
    """Read a motion-tracking quantity from ``state.extras``.

    Args:
        state: The environment state, expected to have been populated by a
            ``motion_clip`` command term earlier in the same control step.
        key: One of :data:`MOTION_TARGET_POS`, :data:`MOTION_TARGET_VEL`,
            :data:`MOTION_PHASE`.

    Returns:
        The stored value (a float for the phase, a 1-D array for the targets).

    Raises:
        KeyError: If the key is absent, with a message pointing at the missing
            ``motion_clip`` command term rather than degrading silently.
    """
    try:
        return state.extras[key]
    except KeyError:
        raise KeyError(
            f"motion target {key!r} not found in EnvState.extras; available: "
            f"{sorted(state.extras)}. A 'motion_clip' command term must run "
            "(CommandManager.compute) before WBT observation/reward/termination "
            "terms can read its targets."
        ) from None


@dataclass
class MotionClip:
    """A reference joint trajectory sampled at a fixed frame rate.

    The clip is interpolated by *phase* - the fractional position along the
    trajectory in ``[0, 1)``. :meth:`sample` maps an elapsed time to a phase and
    linearly interpolates the bracketing frames, so a clip can be replayed at any
    control rate independent of its native ``fps``.

    Args:
        frames_pos: Per-frame joint positions, shape ``(num_frames, num_joints)``.
        frames_vel: Per-frame joint velocities, same shape as ``frames_pos``.
        fps: Native sampling rate of the clip in frames per second.
    """

    frames_pos: FloatArray
    frames_vel: FloatArray
    fps: float

    @classmethod
    def from_arrays(
        cls,
        frames_pos: Any,
        frames_vel: Any | None = None,
        fps: float = 30.0,
    ) -> MotionClip:
        """Build and validate a :class:`MotionClip` from array-likes.

        Args:
            frames_pos: ``(num_frames, num_joints)`` joint positions.
            frames_vel: Optional matching joint velocities; defaults to zeros.
            fps: Native frame rate (must be positive).

        Returns:
            The validated clip.

        Raises:
            ValueError: If ``frames_pos`` is not a non-empty 2-D array, if
                ``frames_vel`` does not match its shape, or if ``fps <= 0``.
        """
        pos = np.asarray(frames_pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] == 0:
            raise ValueError(
                f"frames_pos must be a non-empty 2-D (num_frames, num_joints) array, got shape {pos.shape}"
            )
        if frames_vel is None:
            vel = np.zeros_like(pos)
        else:
            vel = np.asarray(frames_vel, dtype=np.float64)
            if vel.shape != pos.shape:
                raise ValueError(f"frames_vel shape {vel.shape} must match frames_pos shape {pos.shape}")
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        return cls(frames_pos=pos, frames_vel=vel, fps=float(fps))

    @property
    def num_frames(self) -> int:
        """Number of frames in the clip."""
        return int(self.frames_pos.shape[0])

    @property
    def num_joints(self) -> int:
        """Number of joints each frame describes."""
        return int(self.frames_pos.shape[1])

    @property
    def duration(self) -> float:
        """Clip duration in seconds (``num_frames / fps``)."""
        return self.num_frames / self.fps

    def sample(self, t: float, *, loop: bool = True) -> tuple[FloatArray, FloatArray, float]:
        """Sample the interpolated target pose, velocity, and phase at time ``t``.

        Args:
            t: Elapsed time in seconds since the clip started.
            loop: When ``True`` the clip wraps (cyclic interpolation across the
                last->first frame); when ``False`` ``t`` is clamped to the final
                frame.

        Returns:
            ``(target_pos, target_vel, phase)`` where the targets are 1-D arrays
            of length ``num_joints`` and ``phase`` is in ``[0, 1)``.
        """
        n = self.num_frames
        if n == 1:
            return self.frames_pos[0].copy(), self.frames_vel[0].copy(), 0.0

        if loop:
            phase = (t / self.duration) % 1.0
            frame = phase * n
            lo = int(np.floor(frame)) % n
            hi = (lo + 1) % n
            frac = frame - np.floor(frame)
        else:
            phase = float(np.clip(t / self.duration, 0.0, 1.0))
            frame = min(phase * n, n - 1.0)
            lo = int(np.floor(frame))
            hi = min(lo + 1, n - 1)
            frac = frame - lo

        pos = (1.0 - frac) * self.frames_pos[lo] + frac * self.frames_pos[hi]
        vel = (1.0 - frac) * self.frames_vel[lo] + frac * self.frames_vel[hi]
        return pos, vel, float(phase)
