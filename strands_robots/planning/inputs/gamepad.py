"""Gamepad input source (pygame) for the kinematic planner.

Maps an analog gamepad onto locomotion intent. Requires the optional ``pygame``
dependency (``pip install "strands-robots[planning]"``); the import is gated
through :func:`~strands_robots.utils.require_optional` so the rest of the
planning package works without it.

Axis / button mapping (standard layout):

* left stick   -> ``(vx, vy)``
* right stick X -> ``omega`` (yaw)
* right stick Y -> height trim
* face + bumper buttons -> the eight movement styles (in
  :data:`~strands_robots.planning.base.STYLES` order)
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.planning.base import STYLES, PlannerUpdate
from strands_robots.planning.inputs.base import InputSource
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


class GamepadInput(InputSource):
    """Steer the planner from a pygame-readable gamepad.

    Args:
        joystick_index: Index of the joystick to open (default first).
        max_speed: Velocity (m/s) at full left-stick deflection.
        max_omega: Yaw rate (rad/s) at full right-stick-X deflection.
        height_center: Base height (m) at neutral right-stick-Y.
        height_span: Height swing (m) at full right-stick-Y deflection.
        deadzone: Stick magnitude below which axes read as zero.

    Raises:
        ImportError: If ``pygame`` is not installed (via ``require_optional``).
    """

    def __init__(
        self,
        *,
        joystick_index: int = 0,
        max_speed: float = 1.0,
        max_omega: float = 2.0,
        height_center: float = 0.74,
        height_span: float = 0.2,
        deadzone: float = 0.1,
    ) -> None:
        self._pygame: Any = require_optional("pygame", extra="planning", purpose="gamepad planner input")
        self._index = int(joystick_index)
        self._max_speed = float(max_speed)
        self._max_omega = float(max_omega)
        self._height_center = float(height_center)
        self._height_span = float(height_span)
        self._deadzone = float(deadzone)
        self._joystick: Any = None

    def _dz(self, value: float) -> float:
        return 0.0 if abs(value) < self._deadzone else value

    def start(self) -> None:
        if self._joystick is not None:
            return
        self._pygame.init()
        self._pygame.joystick.init()
        if self._pygame.joystick.get_count() <= self._index:
            raise RuntimeError(f"no gamepad at index {self._index} (found {self._pygame.joystick.get_count()})")
        joystick = self._pygame.joystick.Joystick(self._index)
        joystick.init()
        self._joystick = joystick

    def poll(self) -> PlannerUpdate | None:
        if self._joystick is None:
            return None
        self._pygame.event.pump()
        js = self._joystick
        vx = -self._dz(js.get_axis(1)) * self._max_speed  # stick up = +forward
        vy = -self._dz(js.get_axis(0)) * self._max_speed  # stick left = +left
        omega = -self._dz(js.get_axis(3)) * self._max_omega
        height = self._height_center - self._dz(js.get_axis(4)) * self._height_span
        style: str | None = None
        for btn in range(min(js.get_numbuttons(), len(STYLES))):
            if js.get_button(btn):
                style = STYLES[btn]
                break
        return PlannerUpdate(root_vel=(vx, vy, omega), height=height, style=style)

    def stop(self) -> None:
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:  # noqa: BLE001 - device teardown is best-effort
                logger.debug("gamepad quit failed", exc_info=True)
            self._joystick = None

    def reset(self) -> None:
        pass
