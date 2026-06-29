"""Real-time kinematic locomotion planner driven by a pluggable input source.

:class:`KinematicPlanner` is the concrete top-of-stack controller for humanoid
locomotion. It holds the current :class:`~strands_robots.planning.base.PlannerCommand`
behind a lock and runs the attached
:class:`~strands_robots.planning.inputs.base.InputSource` on a background thread,
so reading intent never blocks the control loop that calls :meth:`poll`. Velocity
and height are clamped to configured limits and rapid style toggles are debounced.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from strands_robots.planning.base import (
    DEFAULT_HEIGHT,
    Planner,
    PlannerCommand,
    PlannerUpdate,
)
from strands_robots.planning.inputs.base import InputSource

logger = logging.getLogger(__name__)


class KinematicPlanner(Planner):
    """Hold a live locomotion command, updated off-thread by an input source.

    The planner is the single source of truth for the current intent. An
    :class:`~strands_robots.planning.inputs.base.InputSource` (keyboard, gamepad,
    agent, or a scripted sequence) produces :class:`PlannerUpdate` deltas on a
    background thread; the planner folds each into its current command, clamping
    velocity to ``+/- max_speed`` (yaw to ``+/- max_omega``) and height into
    ``height_range``, and debouncing style switches by ``style_debounce_s``. The
    control loop calls :meth:`poll` once per tick to read the latest command -
    a lock-guarded snapshot, never a blocking read.

    Args:
        input_source: Source of intent updates. ``None`` makes a static planner
            that only ever emits ``initial`` (useful for tests / fixed gaits).
        initial: Starting command. Defaults to a zero-velocity ``run`` command.
        max_speed: Absolute clamp (m/s) for ``vx`` and ``vy``.
        max_omega: Absolute clamp (rad/s) for yaw rate.
        height_range: ``(min, max)`` clamp (m) for the base height.
        style_debounce_s: Minimum seconds between accepted style changes; faster
            toggles are ignored so a key/button bounce does not thrash the gait.
        poll_interval_s: Sleep between input polls on the background thread when
            no update is pending.
        clock: Monotonic clock callable (injectable for tests).

    Raises:
        ValueError: If ``max_speed``/``max_omega`` are not positive, the height
            range is degenerate, or ``style_debounce_s`` is negative.
    """

    def __init__(
        self,
        input_source: InputSource | None = None,
        *,
        initial: PlannerCommand | None = None,
        max_speed: float = 1.0,
        max_omega: float = 2.0,
        height_range: tuple[float, float] = (0.4, 0.8),
        style_debounce_s: float = 0.2,
        poll_interval_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {max_speed}")
        if max_omega <= 0:
            raise ValueError(f"max_omega must be positive, got {max_omega}")
        if not height_range[0] < height_range[1]:
            raise ValueError(f"height_range must be (min < max), got {height_range!r}")
        if style_debounce_s < 0:
            raise ValueError(f"style_debounce_s must be >= 0, got {style_debounce_s}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {poll_interval_s}")

        self._input = input_source
        self._max_speed = float(max_speed)
        self._max_omega = float(max_omega)
        self._height_min, self._height_max = float(height_range[0]), float(height_range[1])
        self._style_debounce_s = float(style_debounce_s)
        self._poll_interval_s = float(poll_interval_s)
        self._clock = clock

        self._lock = threading.Lock()
        self._initial = self._clamp_command(initial) if initial is not None else PlannerCommand()
        self._command = self._initial
        self._stop_requested = False
        self._last_style_change = float("-inf")
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    # -- clamping ---------------------------------------------------------
    def _clamp_vel(self, vel: tuple[float, float, float]) -> tuple[float, float, float]:
        def c(x: float, lim: float) -> float:
            return max(-lim, min(lim, float(x)))

        return (c(vel[0], self._max_speed), c(vel[1], self._max_speed), c(vel[2], self._max_omega))

    def _clamp_height(self, h: float) -> float:
        return max(self._height_min, min(self._height_max, float(h)))

    def _clamp_command(self, cmd: PlannerCommand) -> PlannerCommand:
        return PlannerCommand(
            root_vel=self._clamp_vel(cmd.root_vel),
            height=self._clamp_height(cmd.height),
            style=cmd.style,
        )

    # -- intent application ----------------------------------------------
    def apply_update(self, update: PlannerUpdate) -> None:
        """Fold ``update`` into the current command (thread-safe, clamped).

        Input sources call this from their own thread; the control loop only
        ever calls :meth:`poll`. Style changes within ``style_debounce_s`` of the
        previous accepted change are ignored.
        """
        if update.is_empty():
            return
        with self._lock:
            cmd = self._command
            vel = cmd.root_vel
            height = cmd.height
            style = cmd.style
            if update.stop:
                self._stop_requested = True
                vel = (0.0, 0.0, 0.0)
            if update.root_vel is not None:
                vel = self._clamp_vel(update.root_vel)
            if update.height is not None:
                height = self._clamp_height(update.height)
            if update.style is not None and update.style != style:
                now = self._clock()
                if now - self._last_style_change >= self._style_debounce_s:
                    style = update.style
                    self._last_style_change = now
                else:
                    logger.debug("style change to %r debounced", update.style)
            self._command = PlannerCommand(root_vel=vel, height=height, style=style)

    def poll(self) -> PlannerCommand:
        """Return the current command (lock-guarded snapshot; non-blocking)."""
        with self._lock:
            return self._command

    @property
    def stop_requested(self) -> bool:
        """``True`` once an input source has requested a halt (e.g. ESC).

        Callers driving a long rollout MAY poll this to break their loop early;
        the planner itself only zeroes velocity on a stop request.
        """
        with self._lock:
            return self._stop_requested

    def reset(self) -> None:
        """Reset to the initial command and clear the stop request."""
        with self._lock:
            self._command = self._initial
            self._stop_requested = False
            self._last_style_change = float("-inf")
        if self._input is not None:
            self._input.reset()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Start the input source and its background polling thread (idempotent)."""
        if self._input is None or self._running.is_set():
            return
        self._input.start()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="kinematic-planner-input", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                update = self._input.poll() if self._input is not None else None
            except Exception:  # noqa: BLE001 - a broken input must not kill the loop
                logger.exception("input source poll failed")
                update = None
            if update is not None:
                self.apply_update(update)
            else:
                time.sleep(self._poll_interval_s)

    def stop(self) -> None:
        """Stop the polling thread and the input source (idempotent)."""
        self._running.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        if self._input is not None:
            self._input.stop()

    @property
    def provider_name(self) -> str:
        return "kinematic"


__all__ = ["KinematicPlanner", "DEFAULT_HEIGHT"]
