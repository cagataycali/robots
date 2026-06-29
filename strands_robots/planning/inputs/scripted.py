"""A deterministic, headless input source that replays a timed command schedule.

:class:`ScriptedInput` is the programmatic counterpart to keyboard/gamepad/agent
input: it emits a fixed sequence of :class:`~strands_robots.planning.base.PlannerUpdate`
at wall-clock offsets from :meth:`start`. It needs no TTY, device, or model, so
it powers reproducible demos, the headless rollout artifact, and tests, and is
the obvious building block for a mesh peer that streams a precomputed plan.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from strands_robots.planning.base import PlannerUpdate
from strands_robots.planning.inputs.base import InputSource


class ScriptedInput(InputSource):
    """Replay a time-stamped sequence of updates relative to :meth:`start`.

    Args:
        schedule: Sequence of ``(t_seconds, update)`` pairs. ``t_seconds`` is the
            offset from :meth:`start` at which the update becomes due. The
            schedule is sorted by time on construction; one due update is
            returned per :meth:`poll` so the planner thread folds them in order.
        clock: Monotonic clock callable (injectable for tests).
        loop: When ``True``, restart the schedule from ``t=0`` after the last
            update (the cycle length is the largest ``t_seconds``).

    Raises:
        ValueError: If any ``t_seconds`` is negative.
    """

    def __init__(
        self,
        schedule: list[tuple[float, PlannerUpdate]],
        *,
        clock: Callable[[], float] = time.monotonic,
        loop: bool = False,
    ) -> None:
        for t, _ in schedule:
            if t < 0:
                raise ValueError(f"schedule offsets must be >= 0, got {t}")
        self._schedule = sorted(schedule, key=lambda item: item[0])
        self._clock = clock
        self._loop = loop
        self._period = self._schedule[-1][0] if self._schedule else 0.0
        self._start_t: float | None = None
        self._idx = 0

    def start(self) -> None:
        self._start_t = self._clock()
        self._idx = 0

    def poll(self) -> PlannerUpdate | None:
        if self._start_t is None or not self._schedule:
            return None
        elapsed = self._clock() - self._start_t
        if self._idx >= len(self._schedule):
            if not self._loop:
                return None
            self._start_t = self._clock()
            self._idx = 0
            return None
        due_t, update = self._schedule[self._idx]
        if elapsed >= due_t:
            self._idx += 1
            return update
        return None

    def stop(self) -> None:
        self._start_t = None

    def reset(self) -> None:
        self._start_t = None
        self._idx = 0
