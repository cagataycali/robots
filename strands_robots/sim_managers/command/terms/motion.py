"""Reference-motion command term for whole-body tracking (WBT)."""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, FloatArray, Term, register_term
from strands_robots.sim_managers.motion import (
    MOTION_PHASE,
    MOTION_TARGET_POS,
    MOTION_TARGET_VEL,
    MotionClip,
)


@register_term("command", "motion_clip")
class MotionClipCommand(Term):
    """Replay a reference motion clip and publish its per-step targets.

    On each control step the :class:`~strands_robots.sim_managers.CommandManager`
    advances this term's clock by ``state.dt`` (via :meth:`update`) and then
    calls it. The term samples the clip at the current elapsed time and writes
    the interpolated target pose, target velocity, and phase into
    ``state.extras`` under the :mod:`~strands_robots.sim_managers.motion` keys, so
    the WBT observation / reward / termination terms can read them. The returned
    value (also published as the command vector under this term's label) is the
    target joint pose.

    Args:
        frames_pos: Reference joint positions, shape ``(num_frames, num_joints)``.
        frames_vel: Optional matching joint velocities; defaults to zeros.
        fps: Native frame rate of the clip in frames per second.
        loop: When ``True`` the clip repeats; when ``False`` it holds the final
            frame after one pass.
    """

    def __init__(
        self,
        frames_pos: Any,
        frames_vel: Any | None = None,
        fps: float = 30.0,
        loop: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(frames_pos=frames_pos, frames_vel=frames_vel, fps=fps, loop=loop, **params)
        self.clip = MotionClip.from_arrays(frames_pos, frames_vel, fps=fps)
        self.loop = bool(loop)
        self._t = 0.0

    def reset(self, state: EnvState | None = None, *, rng: np.random.Generator | None = None) -> None:
        """Rewind the clip to its start."""
        self._t = 0.0

    def update(self, dt: float) -> None:
        """Advance the clip clock by ``dt`` seconds."""
        self._t += dt

    def __call__(self, state: EnvState) -> FloatArray:
        target_pos, target_vel, phase = self.clip.sample(self._t, loop=self.loop)
        state.extras[MOTION_TARGET_POS] = target_pos
        state.extras[MOTION_TARGET_VEL] = target_vel
        state.extras[MOTION_PHASE] = phase
        return target_pos
