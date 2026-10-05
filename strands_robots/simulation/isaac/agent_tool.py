"""The Isaac simulation as a Strands ``AgentTool``.

``Robot("so101", backend="isaac")`` returns an :class:`IsaacSimulation`, and
the README, the ``Robot`` docstring and example 05 present that object as a
tool an agent drives: ``Agent(tools=[sim])``. It was not one. ``IsaacSimulation``
subclassed ``SimEngine`` only - no ``tool_spec``, no ``stream`` - so Strands'
tool registry logged "unrecognized tool specification", dropped it, and the
agent answered with no tool at all. Every agent-driven Isaac run in our tests
went through a hand-written adapter instead.

This mixin gives the engine the tool surface the MuJoCo backend has:

* ``tool_spec`` publishes the shared action schema, with the ``action`` enum
  narrowed to the actions this backend implements - a model is never offered an
  action that would answer "unknown" (30 of MuJoCo's 77 are not implemented
  here, among them ``save_state``, ``attach_bodies`` and ``render_depth``);
* ``stream`` / ``__call__`` route an action to the engine method of that name,
  forwarding only the arguments the method takes and saying which it dropped;
* every call runs on the thread that owns Isaac's ``SimulationApp``. Kit only
  advances on that thread, and Strands runs tools on a worker thread, so a call
  from the agent is marshalled onto the main-thread pump. With no pump, the call
  is refused with the recipe rather than blocking forever -
  :meth:`IsaacAgentToolMixin.run_agent` is that recipe in one line.
"""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import json
import logging
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from strands.types.tools import ToolSpec, ToolUse

from strands_robots.simulation.base import scene_contents_sentence

logger = logging.getLogger(__name__)

#: The shared agent-facing schema. Published by the MuJoCo backend; this backend
#: publishes the subset of its ``action`` enum it implements.
_TOOL_SPEC_PATH = Path(__file__).resolve().parent.parent / "mujoco" / "tool_spec.json"

#: Argument names the shared schema spells differently from this backend's methods.
_ARGUMENT_ALIASES: dict[str, str] = {
    "checkpoint_name": "name",
    "torque_vec": "torque",
    "camera_names": "cameras",
    "joint_positions": "positions",
}

#: Engine methods an agent must not reach through the tool even though they are
#: public: they own the process or the main-thread loop the tool itself runs on.
_NOT_AGENT_ACTIONS = frozenset({"run_pump_forever", "pump", "run_on_main", "run_agent", "cleanup"})


@functools.lru_cache(maxsize=1)
def _shared_schema() -> dict[str, Any]:
    return json.loads(_TOOL_SPEC_PATH.read_text(encoding="utf-8"))


def published_actions(engine_cls: type) -> list[str]:
    """The shared schema's actions that ``engine_cls`` implements, in the schema's order."""
    enum = _shared_schema().get("properties", {}).get("action", {}).get("enum", [])
    return [
        a
        for a in enum
        if isinstance(a, str)
        and not a.startswith("_")
        and a not in _NOT_AGENT_ACTIONS
        and callable(getattr(engine_cls, a, None))
    ]


class IsaacAgentToolMixin:
    """``AgentTool`` surface for :class:`~strands_robots.simulation.isaac.simulation.IsaacSimulation`."""

    _tool_name_str: str = "isaac"

    # -- AgentTool interface ---------------------------------------------------

    @property
    def tool_name(self) -> str:
        """The name the agent calls this tool by (``Robot(..., tool_name=...)``; default ``"isaac"``)."""
        return self._tool_name_str

    @tool_name.setter
    def tool_name(self, value: str) -> None:
        self._tool_name_str = str(value)

    @property
    def tool_type(self) -> str:
        """The Strands tool category (always ``"simulation"``)."""
        return "simulation"

    @property
    def tool_spec(self) -> ToolSpec:
        """Name, description and input schema; the ``action`` enum lists only what this backend does."""
        schema = json.loads(json.dumps(_shared_schema()))
        actions = published_actions(type(self))
        schema["properties"]["action"]["enum"] = actions
        registry = getattr(self, "_robots", {}) or {}
        robots = "; ".join(
            f"'{name}' (joints {list(getattr(r, 'joint_names', []) or [])})" for name, r in sorted(registry.items())
        )
        world = (
            f"The world is ALREADY CREATED and holds robot(s) {robots}; do not call create_world. "
            f"{scene_contents_sentence(getattr(self, '_objects', {}), getattr(self, '_cameras', {}))}"
            if getattr(self, "_world_created", False) and registry
            else "Call create_world first, then add_robot. "
        )
        spec: ToolSpec = {
            "name": self.tool_name,
            "description": (
                "NVIDIA Isaac Sim simulation (stateful session, RTX rendering, PhysX). "
                f"{world}"
                "Adding or removing a dynamic object or a robot requires reset() before the next step. "
                f"Actions ({len(actions)}): {', '.join(actions)}. "
                "Call destroy() at session end to release resources."
            ),
            "inputSchema": {"json": schema},
        }
        return spec

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Run one agent tool call and yield its single ``ToolResultEvent``."""
        from strands.types._events import ToolResultEvent

        tool_use_id = tool_use.get("toolUseId", "")
        try:
            data = dict(tool_use.get("input", {}) or {})
            result = self._agent_call(str(data.get("action", "") or ""), data)
        except Exception as exc:  # noqa: BLE001 - tool boundary: an error result, never a raise into the agent
            logger.exception("isaac tool call failed")
            result = {"status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
        yield ToolResultEvent(dict(toolUseId=tool_use_id, **result))  # type: ignore[typeddict-item]

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        """``sim(action="render", camera_name="front")``: the agent path, from Python."""
        return self._agent_call(action, {"action": action, **kwargs})

    # -- dispatch ----------------------------------------------------------------

    def _agent_call(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        action = action.strip() if isinstance(action, str) else ""
        actions = published_actions(type(self))
        if not action:
            return _error(f"'action' is required; one of: {', '.join(actions)}")
        if action not in actions:
            if action in _shared_schema()["properties"]["action"]["enum"]:
                return _error(
                    f"Action '{action}' is not implemented on the Isaac backend. Available here: {', '.join(actions)}"
                )
            return _error(f"Unknown action '{action}'. Available: {', '.join(actions)}")
        method = getattr(self, action)
        kw = {k: v for k, v in data.items() if k != "action" and v is not None}
        for alias, name in _ARGUMENT_ALIASES.items():
            if alias in kw and name not in kw:
                kw[name] = kw.pop(alias)
        params = inspect.signature(method).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        dropped = sorted(k for k in kw if k not in params and not takes_any)
        kw = {k: v for k, v in kw.items() if k in params or takes_any}

        result = self._on_kit_thread(action, lambda: method(**kw))
        if not isinstance(result, dict) or "status" not in result:
            result = {"status": "success", "content": [{"text": _as_text(result)}]}
        if dropped:
            result = dict(result)
            result["content"] = [
                *result.get("content", []),
                {"text": f"(ignored parameters Isaac {action}() does not take: {dropped})"},
            ]
        return result

    def _on_kit_thread(self, action: str, fn: Any) -> Any:
        """Run ``fn`` on the ``SimulationApp`` thread: inline there, via the pump elsewhere, else refuse."""
        on_main = getattr(self, "_on_main_thread", None)
        if on_main is None or on_main():
            return fn()
        if getattr(self, "_pump_running", False):
            return self.run_on_main(fn)  # type: ignore[attr-defined]
        return _error(
            f"{action}: this tool call arrived on a worker thread (Strands runs tools there), and Isaac Sim "
            "only advances on the thread that created SimulationApp, where no pump is running - so the call "
            "would block forever. Drive the agent with sim.run_agent(agent, prompt), which runs the agent on a "
            "worker thread while this thread pumps, or run sim.run_pump_forever() on the main thread yourself."
        )

    def run_agent(self, agent: Any, prompt: Any, **kwargs: Any) -> Any:
        """Run ``agent(prompt)`` with this simulation's main-thread pump serving its tool calls.

        Call it on the thread that created the simulation. The agent runs on a
        worker thread, as Strands would run it anyway, while this thread runs
        :meth:`run_pump_forever` until the agent returns; its result (or its
        exception) is returned (or raised) here.

        Args:
            agent: A Strands ``Agent`` (anything callable with the prompt).
            prompt: The prompt.
            **kwargs: Forwarded to ``agent(prompt, **kwargs)``.
        """
        if getattr(self, "_pump_running", False):
            return agent(prompt, **kwargs)
        done = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="isaac-agent") as pool:
            # Marked before the worker starts, so its first tool call is
            # marshalled rather than refused in the instant before the pump
            # loop begins.
            self._pump_running = True
            future = pool.submit(agent, prompt, **kwargs)
            future.add_done_callback(lambda _f: done.set())
            self.run_pump_forever(stop_event=done)  # type: ignore[attr-defined]
            return future.result()


def _error(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _as_text(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:4000]
    except (TypeError, ValueError):
        return str(value)[:4000]
