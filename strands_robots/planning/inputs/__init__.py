"""Input sources that turn a control surface into locomotion intent.

Each source is a non-blocking :class:`~strands_robots.planning.inputs.base.InputSource`
the :class:`~strands_robots.planning.kinematic.KinematicPlanner` polls off-thread:

* :class:`KeyboardInput` - WASD/QE/RF + style keys from a terminal.
* :class:`GamepadInput` - analog sticks + buttons (optional ``pygame``).
* :class:`AgentInput` - a :class:`strands.Agent` emitting intent via a tool.
* :class:`ScriptedInput` - a deterministic, headless timed command sequence.

``GamepadInput`` is imported lazily so the package works without ``pygame``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands_robots.planning.inputs.agent import AgentInput
from strands_robots.planning.inputs.base import InputSource
from strands_robots.planning.inputs.keyboard import KeyboardInput
from strands_robots.planning.inputs.scripted import ScriptedInput

if TYPE_CHECKING:
    from strands_robots.planning.inputs.gamepad import GamepadInput

__all__ = ["InputSource", "KeyboardInput", "AgentInput", "ScriptedInput", "GamepadInput"]


def __getattr__(name: str) -> Any:
    # Lazy-load GamepadInput so importing the package never imports pygame.
    if name == "GamepadInput":
        from strands_robots.planning.inputs.gamepad import GamepadInput

        return GamepadInput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
