"""LLM-agent input source: turn natural language into locomotion intent.

:class:`AgentInput` lets a :class:`strands.Agent` steer the planner. It exposes a
``set_locomotion_intent`` tool the agent calls to emit
:class:`~strands_robots.planning.base.PlannerUpdate` deltas, runs the agent on a
background thread against a goal string, and surfaces the emitted updates through
:meth:`poll`. So a prompt like *"walk forward, then switch to stealth and slow
down"* becomes a timed stream of planner commands that drive the locomotion
policy - the natural-language top of the control stack.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

from strands_robots.planning.base import STYLES, PlannerUpdate
from strands_robots.planning.inputs.base import InputSource

if TYPE_CHECKING:
    from strands import Agent

logger = logging.getLogger(__name__)


class AgentInput(InputSource):
    """Drive the planner from a :class:`strands.Agent` and a goal string.

    On :meth:`start` the ``set_locomotion_intent`` tool is registered on the
    agent and ``agent(goal)`` runs on a background thread; each tool call the
    agent makes enqueues a :class:`PlannerUpdate` that :meth:`poll` returns.

    Args:
        agent: A constructed :class:`strands.Agent`. The
            ``set_locomotion_intent`` tool is registered on it at :meth:`start`.
        goal: Natural-language locomotion goal handed to the agent.
        max_queue: Maximum buffered updates (back-pressure guard); oldest are
            dropped when the agent emits faster than the loop consumes.
    """

    def __init__(self, agent: Agent, goal: str, *, max_queue: int = 256) -> None:
        self._agent = agent
        self._goal = goal
        self._queue: queue.Queue[PlannerUpdate] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._registered = False

    def _enqueue(self, update: PlannerUpdate) -> None:
        try:
            self._queue.put_nowait(update)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
                self._queue.put_nowait(update)
            except queue.Empty:  # pragma: no cover - race only
                pass

    def build_tool(self) -> Any:
        """Build the ``set_locomotion_intent`` agent tool bound to this source.

        Exposed publicly so callers can pass it to ``Agent(tools=[...])``
        themselves instead of letting :meth:`start` register it.
        """
        from strands import tool

        styles = ", ".join(STYLES)

        @tool(name="set_locomotion_intent")
        def set_locomotion_intent(
            vx: float = 0.0,
            vy: float = 0.0,
            omega: float = 0.0,
            height: float | None = None,
            style: str | None = None,
            stop: bool = False,
        ) -> str:
            """Set the robot's locomotion intent for the next control ticks.

            Args:
                vx: Forward velocity (m/s, + is forward).
                vy: Lateral velocity (m/s, + is left).
                omega: Yaw rate (rad/s, + turns left).
                height: Target base height (m); omit to keep current.
                style: Movement style - one of run, happy, stealth, injured,
                    kneeling, hand_crawling, elbow_crawling, boxing.
                stop: Set true to halt locomotion.

            Returns:
                A short acknowledgement string.
            """
            if style is not None and style not in STYLES:
                return f"error: unknown style {style!r}; expected one of {styles}"
            self._enqueue(
                PlannerUpdate(
                    root_vel=(float(vx), float(vy), float(omega)),
                    height=None if height is None else float(height),
                    style=style,
                    stop=bool(stop),
                )
            )
            return f"intent set: vel=({vx},{vy},{omega}) height={height} style={style} stop={stop}"

        return set_locomotion_intent

    def poll(self) -> PlannerUpdate | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _run_agent(self) -> None:
        try:
            self._agent(self._goal)
        except Exception:  # noqa: BLE001 - agent errors must not crash the planner thread
            logger.exception("AgentInput: agent run failed")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._registered:
            try:
                self._agent.tool_registry.register_tool(self.build_tool())
                self._registered = True
            except Exception:  # noqa: BLE001 - fall back to caller-registered tool
                logger.warning(
                    "AgentInput: could not auto-register set_locomotion_intent; "
                    "pass AgentInput.build_tool() to Agent(tools=[...]) instead.",
                    exc_info=True,
                )
        self._thread = threading.Thread(target=self._run_agent, name="agent-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # The agent run is a daemon thread; we cannot forcibly kill it, but the
        # planner stops reading once stop() returns. Join briefly if it is done.
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.1)

    def reset(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
