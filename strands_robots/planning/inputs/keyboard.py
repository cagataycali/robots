"""Non-blocking keyboard input source for the kinematic planner.

Reads single keystrokes from ``stdin`` on a dedicated reader thread (terminal in
cbreak mode) and turns them into :class:`~strands_robots.planning.base.PlannerUpdate`
deltas. The reader thread owns the only blocking ``read``; :meth:`poll` merely
drains a queue, so the planner/control loop is never blocked waiting for a key.

Key bindings:

* ``w`` / ``s`` - forward / back  (vx +/- ``speed_step``)
* ``a`` / ``d`` - strafe left / right (vy +/- ``speed_step``)
* ``q`` / ``e`` - turn left / right (omega +/- ``omega_step``)
* ``r`` / ``f`` - taller / shorter (height +/- ``height_step``)
* ``space``     - halt (zero velocity)
* ``1``..``8``  - select movement style (see :data:`~strands_robots.planning.base.STYLES`)
* ``ESC``       - request stop

When ``stdin`` is not an interactive TTY (CI, a pipe), the source degrades to a
no-op: it logs once and :meth:`poll` always returns ``None``.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any

from strands_robots.planning.base import STYLES, PlannerUpdate
from strands_robots.planning.inputs.base import InputSource

logger = logging.getLogger(__name__)


class KeyboardInput(InputSource):
    """Steer the planner from the terminal keyboard (non-blocking).

    Args:
        speed_step: Velocity increment (m/s) per ``w``/``a``/``s``/``d`` press.
        omega_step: Yaw increment (rad/s) per ``q``/``e`` press.
        height_step: Height increment (m) per ``r``/``f`` press.
    """

    def __init__(self, *, speed_step: float = 0.1, omega_step: float = 0.2, height_step: float = 0.05) -> None:
        self._speed_step = float(speed_step)
        self._omega_step = float(omega_step)
        self._height_step = float(height_step)
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._restore: list[Any] | None = None
        # Running accumulators - keyboard presses are integrated into an
        # absolute command (no key-release events in a terminal).
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._height: float | None = None

    # -- key -> intent ----------------------------------------------------
    def _apply_char(self, ch: str) -> PlannerUpdate | None:
        """Translate one character into an update, mutating the accumulators."""
        if ch == "\x1b":  # ESC
            self._vx = self._vy = self._omega = 0.0
            return PlannerUpdate(root_vel=(0.0, 0.0, 0.0), stop=True)
        if ch in "wsad qe":
            if ch == "w":
                self._vx += self._speed_step
            elif ch == "s":
                self._vx -= self._speed_step
            elif ch == "a":
                self._vy += self._speed_step
            elif ch == "d":
                self._vy -= self._speed_step
            elif ch == "q":
                self._omega += self._omega_step
            elif ch == "e":
                self._omega -= self._omega_step
            elif ch == " ":
                self._vx = self._vy = self._omega = 0.0
            return PlannerUpdate(root_vel=(self._vx, self._vy, self._omega))
        if ch in "rf":
            base = self._height if self._height is not None else 0.74
            self._height = base + (self._height_step if ch == "r" else -self._height_step)
            return PlannerUpdate(height=self._height)
        if ch.isdigit():
            idx = int(ch) - 1
            if 0 <= idx < len(STYLES):
                return PlannerUpdate(style=STYLES[idx])
        return None

    def poll(self) -> PlannerUpdate | None:
        merged: PlannerUpdate | None = None
        while True:
            try:
                ch = self._queue.get_nowait()
            except queue.Empty:
                break
            update = self._apply_char(ch)
            if update is None:
                continue
            if merged is None:
                merged = update
            else:
                # Latest key wins per dimension; stop is sticky.
                if update.root_vel is not None:
                    merged.root_vel = update.root_vel
                if update.height is not None:
                    merged.height = update.height
                if update.style is not None:
                    merged.style = update.style
                merged.stop = merged.stop or update.stop
        return merged

    # -- reader thread ----------------------------------------------------
    def _reader(self) -> None:
        try:
            import select
        except ImportError:  # pragma: no cover - select is stdlib on supported platforms
            return
        while self._running.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (OSError, ValueError):  # stdin closed
                return
            if ready:
                ch = sys.stdin.read(1)
                if ch:
                    self._queue.put(ch)

    def start(self) -> None:
        if self._running.is_set():
            return
        if not sys.stdin.isatty():
            logger.warning("KeyboardInput: stdin is not a TTY; keyboard steering disabled (poll() -> None).")
            return
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._restore = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (ImportError, OSError) as e:  # non-unix or no controlling terminal
            logger.warning("KeyboardInput: cannot set cbreak mode (%s); keyboard steering disabled.", e)
            self._restore = None
            return
        self._running.set()
        self._thread = threading.Thread(target=self._reader, name="keyboard-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        self._thread = None
        if self._restore is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._restore)
            except (ImportError, OSError):
                pass  # Best-effort terminal restore: termios may be unavailable (non-Unix) or the FD closed.
            self._restore = None

    def reset(self) -> None:
        self._vx = self._vy = self._omega = 0.0
        self._height = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
