"""Fleet agent for the dashboard - one Strands Agent driving the whole mesh.

The agent's toolset is the ``robot_mesh`` tool (peers/tell/send/broadcast/
stop/emergency_stop/...) so a single natural-language instruction can target
one robot or the whole fleet straight from the chat bar:

    > everyone pick up your cube      -> robot_mesh broadcast
    > tell so101-arm-1 to wave        -> robot_mesh tell target=so101-arm-1

Chat turns stream over a queue (token / reasoning / tool events) drained by
the /ws/chat websocket handler.

HITL note: robot_mesh actuation actions are gated behind Strands interrupts
by default and FAIL CLOSED outside an approving host. For the dashboard the
human typed the instruction themselves, so we default the gate to "none"
(overridable with STRANDS_MESH_HITL_ACTIONS).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are the Strands Robots fleet operator agent, embedded in the fleet dashboard.

You coordinate robots on a Zenoh mesh via the fleet tool:
- fleet(action="peers") - list every robot/sim currently online
- fleet(action="task", target="<peer_id>", instruction="...", policy_provider="mock", duration=10) - run a task on one robot
- fleet(action="status", target="<peer_id>") - task status
- fleet(action="stop", target="<peer_id>") - stop one robot
- fleet(action="stop_all") - stop everything (use immediately when asked to stop)

Rules:
- Check peers first when unsure what's online. Peer ids containing "sim" or "__" are simulations; others are real hardware.
- Real hardware moves real motors. Be precise, prefer short durations, never guess a target.
- To task "everyone", call fleet(action="task", ...) once per online peer.
- Report what each robot answered. Be brief.
"""

_agent_lock = threading.Lock()
_agent: Any = None
_turn_lock = threading.Lock()   # one agent turn at a time (chat + voice share the agent)

# Conversation history survives dashboard restarts.
HISTORY_FILE = Path(os.getenv(
    "DASHBOARD_HISTORY_FILE",
    os.path.join(tempfile.gettempdir(), "strands_dashboard", "chat_history.json"),
))


HISTORY_LIMIT = 200


def _is_plain_user_message(msg: Any) -> bool:
    """A user message carrying only text - i.e. a safe conversation boundary.

    A user turn that carries a ``toolResult`` block is the *second half* of an
    assistant ``toolUse``; starting a restored conversation there leaves the
    result orphaned and Bedrock rejects the whole request.
    """
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(c, dict) and "toolResult" not in c and "toolUse" not in c for c in content)


def _trim_history(messages: list) -> list:
    """Trim to the limit on a user-message boundary.

    A blind ``messages[-200:]`` can slice between an assistant ``toolUse`` and
    its ``toolResult``. Because the file is reloaded on every restart, that one
    bad slice then fails *every* subsequent turn until it is deleted by hand.
    Cutting only at a plain user message keeps every tool pair
    intact.
    """
    if len(messages) <= HISTORY_LIMIT:
        return list(messages)
    tail = messages[-HISTORY_LIMIT:]
    for i, msg in enumerate(tail):
        if _is_plain_user_message(msg):
            return tail[i:]
    # No clean boundary in the window: keep the last user turn onwards, or
    # drop the history rather than persist a conversation we know is broken.
    for i in range(len(messages) - 1, -1, -1):
        if _is_plain_user_message(messages[i]):
            return messages[i:]
    return []


def _json_safe(messages: list) -> list:
    """Round-trip messages through JSON, dropping anything that won't.

    ``default=str`` used to stringify non-serialisable content blocks, so they
    came back as garbage strings instead of valid message content.
    """
    safe = []
    for msg in messages:
        try:
            safe.append(json.loads(json.dumps(msg)))
        except (TypeError, ValueError):
            logger.debug("dropping non-serialisable message from history")
    return safe


def _load_history() -> list:
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning("could not load chat history: %s", e)
    return []


def _save_history(messages: list) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(_trim_history(_json_safe(list(messages)))))
    except Exception as e:
        logger.debug("could not save chat history: %s", e)


def clear_history() -> bool:
    """Drop the persisted conversation. The self-heal escape hatch."""
    try:
        HISTORY_FILE.unlink(missing_ok=True)
        return True
    except OSError as e:
        logger.warning("could not clear chat history: %s", e)
        return False

# Set by server.py at startup - the dashboard's mesh gateway. robot_mesh
# cannot be used here: it requires an in-process Robot()/Simulation() as its
# Zenoh gateway, and the dashboard is a robot-less peer.
_bridge: Any = None


def set_bridge(bridge: Any) -> None:
    global _bridge
    _bridge = bridge


def _make_fleet_tool() -> Any:
    from strands import tool

    @tool
    def fleet(
        action: str,
        target: str = "",
        instruction: str = "",
        policy_provider: str = "mock",
        duration: float = 15.0,
        robot_name: str = "",
    ) -> dict[str, Any]:
        """Coordinate robots on the mesh (dashboard gateway).

        Args:
            action: One of:
                - "peers": list every robot/sim online (id, type, cameras, task)
                - "task": run a natural-language instruction on one robot
                  (requires target + instruction; uses policy_provider + duration)
                - "stop": stop the running task on one robot (requires target)
                - "stop_all": stop every robot on the mesh
                - "status": task status of one robot (requires target)
            target: Peer id of the robot (from "peers"), e.g. "so101-arm-1".
            instruction: Natural-language task, e.g. "pick up the red cube".
            policy_provider: Policy backend (mock, lerobot_local, lerobot_async, groot, cosmos3).
            duration: Seconds to run the policy for.
            robot_name: Optional single robot inside a multi-robot sim world.
        """
        if _bridge is None:
            return {"status": "error", "content": [{"text": "mesh bridge offline"}]}
        import json as _json
        import time as _time

        if action == "peers":
            snap = _bridge.snapshot()
            lines = []
            for pid, p in sorted(snap["peers"].items()):
                if p.get("stale"):
                    continue
                pres = p.get("presence") or {}
                st = p.get("state") or {}
                task = (st.get("task") or {})
                cams = list((p.get("cameras") or {}).keys())
                joints = len((st.get("joints") or {}))
                lines.append(
                    f"- {pid}: type={pres.get('robot_type','?')} hw_connected={pres.get('connected')} "
                    f"cameras={cams} joints={joints} task={task.get('status','idle')} "
                    f"instruction={task.get('instruction','')!r}"
                )
            text = "Online peers:\n" + "\n".join(lines) if lines else "No live peers on the mesh."
            return {"status": "success", "content": [{"text": text}]}

        if action == "task":
            if not target or not instruction:
                return {"status": "error", "content": [{"text": "task requires target and instruction"}]}
            cmd = {
                "action": "execute", "instruction": instruction,
                "policy_provider": policy_provider, "duration": float(duration),
            }
            if robot_name:
                cmd["robot_name"] = robot_name
            # Child sim peers ("<parent>__<robot>") route to the parent
            # Simulation peer - the shared routing choke point.
            from strands_robots.dashboard.mesh_bridge import route_task_target

            target, cmd = route_task_target(target, cmd)
            res = _bridge.send_cmd(target, cmd, timeout=float(duration) + 30.0, source="agent")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:1500]}]}

        if action == "stop":
            if not target:
                return {"status": "error", "content": [{"text": "stop requires target"}]}
            res = _bridge.send_cmd(target, {"action": "stop"}, timeout=10.0, source="agent")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        if action == "stop_all":
            results = {}
            for pid, p in _bridge.snapshot()["peers"].items():
                if p.get("stale"):
                    continue
                results[pid] = _bridge.send_cmd(pid, {"action": "stop"}, timeout=5.0, source="agent")
            return {"status": "success", "content": [{"text": _json.dumps(results)[:1500]}]}

        if action == "status":
            if not target:
                return {"status": "error", "content": [{"text": "status requires target"}]}
            res = _bridge.send_cmd(target, {"action": "status"}, timeout=10.0, source="agent")
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        return {"status": "error", "content": [{"text": f"unknown action {action!r}. Valid: peers, task, stop, stop_all, status"}]}

    return fleet


def _build_agent() -> Any:
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    from strands import Agent

    from strands_robots.dashboard import settings

    cfg = settings.load()["agent"]
    kwargs: dict[str, Any] = {
        "tools": [_make_fleet_tool()],
        "system_prompt": cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        "callback_handler": None,
    }
    if cfg.get("model_id"):
        kwargs["model"] = cfg["model_id"]
    agent = Agent(**kwargs)

    # Sampling knobs go through the resolved model rather than the Agent
    # constructor, so they work whichever provider model_id selected.
    overrides = {k: cfg[k] for k in ("temperature", "max_tokens") if cfg.get(k) is not None}
    if overrides:
        try:
            agent.model.update_config(**overrides)
        except Exception as e:  # noqa: BLE001 - not every provider takes both
            logger.warning("could not apply %s to model: %s", overrides, e)

    # Restore conversation history from disk so the fleet agent survives
    # dashboard restarts instead of waking up amnesiac.
    history = _trim_history(_load_history())
    if history:
        try:
            agent.messages = history
            logger.info("restored %d chat messages", len(history))
        except Exception as e:
            logger.warning("history restore failed: %s", e)
    return agent


def get_agent() -> Any:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = _build_agent()
        return _agent


def reset_agent(clear_history_too: bool = False, *, clear_history: bool | None = None) -> None:
    """Drop the agent so the next turn rebuilds it from current settings.

    Takes ``_turn_lock`` first: resetting mid-turn would swap the agent out
    from under the worker thread that is streaming into a websocket.
    """
    global _agent
    if clear_history is not None:
        clear_history_too = clear_history
    with _turn_lock:
        with _agent_lock:
            _agent = None
        if clear_history_too:
            globals()["clear_history"]()


def agent_status() -> dict[str, Any]:
    """Readiness for the dock badge - never builds the agent as a side effect."""
    from strands_robots.dashboard import settings

    cfg = settings.load()["agent"]
    with _agent_lock:
        agent = _agent
    tools: list[str] = []
    messages = 0
    if agent is not None:
        with contextlib.suppress(Exception):
            registry = getattr(agent, "tool_registry", None)
            names = getattr(registry, "registry", None) if registry else None
            tools = sorted(names) if isinstance(names, dict) else []
        with contextlib.suppress(Exception):
            messages = len(agent.messages)
    else:
        tools = ["fleet"]
        messages = len(_load_history())
    return {
        "ready": True,
        "built": agent is not None,
        "busy": _turn_lock.locked(),
        "model_id": cfg.get("model_id") or "default",
        "tools": tools,
        "messages": messages,
        "bridge_online": _bridge is not None,
        "history_file": str(HISTORY_FILE),
    }


class TurnCancelled(Exception):
    """The client that asked for this turn went away - abandon it."""


class WSStreamHandler:
    """Strands callback handler -> thread-safe queue of UI events."""

    def __init__(self, q: "queue.Queue[dict]", cancel: "threading.Event | None" = None) -> None:
        self.q = q
        self.cancel = cancel
        self._tool_ids: set[str] = set()

    def __call__(self, **kwargs: Any) -> None:
        if self.cancel is not None and self.cancel.is_set():
            # Cooperative cancellation: raising out of the stream callback ends
            # the agent call so the turn lock is not held by a ghost turn.
            raise TurnCancelled("client disconnected mid-turn")
        data = kwargs.get("data")
        current_tool_use = kwargs.get("current_tool_use") or {}
        reasoning_text = kwargs.get("reasoningText")
        message = kwargs.get("message") or {}

        if reasoning_text:
            self.q.put({"type": "reasoning", "data": reasoning_text})
        if data:
            self.q.put({"type": "token", "data": data})
        if current_tool_use and current_tool_use.get("name"):
            tid = current_tool_use.get("toolUseId", "")
            if tid and tid not in self._tool_ids:
                self._tool_ids.add(tid)
                self.q.put({
                    "type": "tool", "status": "start",
                    "name": current_tool_use.get("name"), "id": tid,
                    "input_preview": str(current_tool_use.get("input", ""))[:200],
                })
        if isinstance(message, dict) and message.get("role") == "user":
            for c in message.get("content", []):
                if isinstance(c, dict) and "toolResult" in c:
                    tr = c["toolResult"]
                    preview = ""
                    for blk in tr.get("content", []):
                        if isinstance(blk, dict) and blk.get("text"):
                            preview = blk["text"][:300]
                            break
                    self.q.put({
                        "type": "tool", "status": tr.get("status", "done"),
                        "id": tr.get("toolUseId", ""), "result_preview": preview,
                    })


def _is_history_poisoned(exc: Exception) -> bool:
    """Does this failure look like a rejected restored conversation?

    A split toolUse/toolResult pair comes back as a provider validation error
    on the *whole* conversation, which then repeats on every turn because the
    same file is reloaded each time.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("validationexception", "toolresult", "tooluse", "invalid conversation", "messages")
    )


def run_turn_blocking(
    prompt: str,
    q: "queue.Queue[dict]",
    cancel: "threading.Event | None" = None,
) -> None:
    """Run one agent turn in a worker thread, streaming events into q.

    ``cancel`` is set by the websocket handler when its client goes away. The
    lock is acquired/released explicitly in try/finally so no exit path -
    including a cancelled or crashed turn - can leave it held by a ghost turn.
    """
    acquired = False
    try:
        if cancel is not None and cancel.is_set():
            return
        _turn_lock.acquire()
        acquired = True
        try:
            if cancel is not None and cancel.is_set():
                raise TurnCancelled("client disconnected before the turn started")
            agent = get_agent()
            agent.callback_handler = WSStreamHandler(q, cancel)
            try:
                result = agent(prompt)
            except TurnCancelled:
                raise
            except Exception as exc:
                # Self-heal instead of failing every future turn: drop the
                # restored history, rebuild, and try once more from a clean
                # conversation.
                if not (agent.messages and _is_history_poisoned(exc)):
                    raise
                logger.warning("turn failed on restored history (%s) - clearing and retrying", exc)
                clear_history()
                global _agent
                with _agent_lock:
                    _agent = None
                agent = get_agent()
                agent.callback_handler = WSStreamHandler(q, cancel)
                agent.messages = []
                q.put({"type": "notice", "text": "previous conversation was unusable and has been cleared"})
                result = agent(prompt)
            try:
                _save_history(list(agent.messages))
            except Exception:
                pass
        finally:
            _turn_lock.release()
            acquired = False
        text = ""
        try:
            msg = getattr(result, "message", None)
            if isinstance(msg, dict):
                text = "\n".join(c.get("text", "") for c in msg.get("content", []) if isinstance(c, dict) and "text" in c)
            else:
                text = str(result)
        except Exception:
            text = str(result)
        q.put({"type": "done", "text": text})
    except TurnCancelled:
        logger.info("agent turn abandoned: client disconnected mid-turn")
    except Exception as e:
        logger.exception("agent turn failed")
        q.put({"type": "error", "error": str(e)})
    finally:
        if acquired:
            _turn_lock.release()
        q.put({"type": "__END__"})
