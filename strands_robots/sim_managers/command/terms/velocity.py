"""Velocity command terms for locomotion."""

from __future__ import annotations

from typing import Any

import numpy as np

from strands_robots.sim_managers.base import EnvState, FloatArray, Term, register_term


@register_term("command", "uniform_velocity")
class UniformVelocityCommand(Term):
    """Sample a planar base-velocity command ``[vx, vy, wz]`` from uniform ranges.

    The command is resampled every ``resampling_time`` seconds and on
    :meth:`reset`. Reading the term (``term(state)``) returns the current
    command without advancing the timer; the :class:`CommandManager` advances
    the timer via :meth:`update` once per control step.

    Args:
        lin_vel_x: ``(low, high)`` range for forward velocity (m/s).
        lin_vel_y: ``(low, high)`` range for lateral velocity (m/s).
        ang_vel_z: ``(low, high)`` range for yaw rate (rad/s).
        resampling_time: Seconds between resamples.
    """

    def __init__(
        self,
        lin_vel_x: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y: tuple[float, float] = (-1.0, 1.0),
        ang_vel_z: tuple[float, float] = (-1.0, 1.0),
        resampling_time: float = 10.0,
        **params: Any,
    ) -> None:
        super().__init__(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            resampling_time=resampling_time,
            **params,
        )
        self.lin_vel_x = (float(lin_vel_x[0]), float(lin_vel_x[1]))
        self.lin_vel_y = (float(lin_vel_y[0]), float(lin_vel_y[1]))
        self.ang_vel_z = (float(ang_vel_z[0]), float(ang_vel_z[1]))
        self.resampling_time = float(resampling_time)
        self._rng = np.random.default_rng()
        self._command: FloatArray = np.zeros(3)
        self._time_since_resample = 0.0
        self._resample()

    def _resample(self) -> None:
        self._command = np.array(
            [
                self._rng.uniform(*self.lin_vel_x),
                self._rng.uniform(*self.lin_vel_y),
                self._rng.uniform(*self.ang_vel_z),
            ]
        )
        self._time_since_resample = 0.0

    def reset(self, state: EnvState | None = None, *, rng: np.random.Generator | None = None) -> None:
        """Reset the timer and draw a fresh command, optionally with a seeded rng."""
        if rng is not None:
            self._rng = rng
        self._resample()

    def update(self, dt: float) -> None:
        """Advance the resample timer by ``dt`` seconds, resampling if due."""
        self._time_since_resample += dt
        if self._time_since_resample >= self.resampling_time:
            self._resample()

    def __call__(self, state: EnvState) -> FloatArray:
        return self._command
