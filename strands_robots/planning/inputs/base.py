"""The :class:`InputSource` ABC: a non-blocking producer of intent updates.

An input source converts some human/agent control surface (keyboard, gamepad,
LLM agent, or a scripted timeline) into :class:`~strands_robots.planning.base.PlannerUpdate`
deltas. The :class:`~strands_robots.planning.kinematic.KinematicPlanner` runs the
source on a background thread and folds each update into the live command, so
:meth:`poll` MUST return immediately (drain a queue, read a non-blocking event) -
it must never block the planner thread waiting for the next keypress.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from strands_robots.planning.base import PlannerUpdate


class InputSource(ABC):
    """Non-blocking source of :class:`PlannerUpdate` deltas.

    Lifecycle: :meth:`start` (acquire the control surface), repeated
    :meth:`poll` (return the next pending update or ``None``), :meth:`stop`
    (release it). :meth:`reset` returns the source to its initial state.
    """

    @abstractmethod
    def poll(self) -> PlannerUpdate | None:
        """Return the next pending intent update, or ``None`` if none. Non-blocking."""

    def start(self) -> None:
        """Acquire the control surface (open device, spawn reader thread). Idempotent."""

    def stop(self) -> None:
        """Release the control surface. Idempotent."""

    def reset(self) -> None:
        """Return the source to its initial state."""
