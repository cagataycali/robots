"""Fleet agent for the dashboard - one Strands Agent driving the whole mesh."""

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

You also have robot_mesh - the SDK's native mesh tool - for everything beyond tasks:
- robot_mesh(action="peers"/"status") - discovery
- robot_mesh(action="tell", target=..., instruction=...) - natural-language ask, waits for the reply
- robot_mesh(action="send", target=..., command='{"action": ...}') / broadcast - raw mesh commands
- robot_mesh(action="subscribe"/"watch"/"inbox") - telemetry
- robot_mesh(action="emergency_stop") - fleet-wide stop
Its physical actions pause for the operator's yes the same way fleet task does; a declined
confirm means nothing was sent - accept the answer, never retry it.

Simulations are fully yours to build and inspect - no confirmation needed, because the verb
is refused by real hardware before anything runs:
- robot_mesh(action="sim_call", target="<sim peer>", function="<sim action>", command='{...}')
  reaches a simulation's own tool surface: add_object, move_object, list_objects, add_camera,
  list_cameras, register_urdf, load_scene, raycast, get_state, set_gravity, apply_force,
  save_state/load_state and ~60 more (function="get_features" lists them).
- Examples: add a red cube -> function="add_object",
  command='{"name": "red_cube", "shape": "box", "position": [0.3, 0, 0.05], "color": [1, 0, 0, 1]}'.
  Overhead camera that shows up in the dashboard's camera tiles -> function="add_camera",
  command='{"name": "<robot>/overhead", "position": [0, 0, 1.0], "target": [0, 0, 0]}'
  (prefix the name with the robot's name - the camera stream publishes per-robot).
  Import a robot model -> function="register_urdf",
  command='{"data_config": "<label>", "urdf_path": "/path/to/model.urdf"}' then add_robot;
  a plain mesh prop -> function="add_object" with "mesh_path".
- In a multi-robot world, put "robot_name" in the command JSON to pick the robot.
- Policy rollouts stay on fleet task / tell (sim_call refuses run_policy by design).

Every online robot is ALSO attached to you as its own native tool, named for its peer id
with dashes as underscores: so101_real_689(action="status"), so101_leader(action="execute",
instruction=..., policy_port=...), and a sim twin's tool carries the full simulation surface
directly (twin_tool(action="add_object", name="red_cube", shape="box", ...)) - no sim_call
envelope needed. The tool list follows the mesh: robots joining or leaving refresh it on the
next turn. Real-arm execute/start pause for the operator's yes exactly like fleet task;
status/stop and every sim action run immediately. When asked what tools or robots you have,
name these per-robot tools.
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
    """A user message carrying only text - i.e. a safe conversation boundary."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(c, dict) and "toolResult" not in c and "toolUse" not in c for c in content)

def _trim_history(messages: list) -> list:
    """Trim to the limit on a user-message boundary. A blind ``messages[-200:]`` can slice between an
    assistant ``toolUse`` and its ``toolResult``.
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
    """Round-trip messages through JSON, dropping anything that won't."""
    safe = []
    for msg in messages:
        try:
            safe.append(json.loads(json.dumps(msg)))
        except (TypeError, ValueError):
            logger.debug("dropping non-serialisable message from history")
    return safe

def _tool_ids(msg: Any, key: str) -> set:
    """Ids of ``toolUse``/``toolResult`` blocks carried by one message."""
    out = set()
    content = msg.get("content") if isinstance(msg, dict) else None
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and isinstance(block.get(key), dict):
            tid = block[key].get("toolUseId")
            if tid is not None:
                out.add(tid)
    return out

def _drop_orphan_tool_blocks(messages: list) -> list:
    """Remove toolUse blocks with no toolResult and vice versa."""
    used: set = set()
    resulted: set = set()
    for msg in messages:
        used |= _tool_ids(msg, "toolUse")
        resulted |= _tool_ids(msg, "toolResult")
    paired = used & resulted
    kept = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        blocks = []
        for block in content:
            if isinstance(block, dict):
                for key in ("toolUse", "toolResult"):
                    inner = block.get(key)
                    if isinstance(inner, dict) and inner.get("toolUseId") not in paired:
                        break
                else:
                    blocks.append(block)
        if blocks:
            kept.append({**msg, "content": blocks})
    return kept

def sanitize_history(messages: Any) -> list:
    """Make a persisted conversation structurally safe to restore."""
    if not isinstance(messages, list):
        return []
    out = _drop_orphan_tool_blocks(_trim_history(_json_safe(messages)))
    while out and not _is_plain_user_message(out[0]):
        out.pop(0)
    return out if _history_problem(out) is None else []

def _history_problem(messages: list) -> str | None:
    """Why this conversation would be rejected, or None if it is valid."""
    if not messages:
        return None
    if not _is_plain_user_message(messages[0]):
        return "first message is not a plain user turn"
    used: set = set()
    resulted: set = set()
    for msg in messages:
        used |= _tool_ids(msg, "toolUse")
        resulted |= _tool_ids(msg, "toolResult")
    if used ^ resulted:
        return f"orphaned tool blocks: {sorted(used ^ resulted)}"
    return None

def _backup_history_file() -> Path | None:
    """Keep the poisoned file aside so it is inspectable, not just gone."""
    try:
        if HISTORY_FILE.exists():
            backup = HISTORY_FILE.with_suffix(".poisoned.json")
            HISTORY_FILE.replace(backup)
            return backup
    except OSError as e:
        logger.warning("could not back up chat history: %s", e)
    return None

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
        HISTORY_FILE.write_text(json.dumps(sanitize_history(list(messages))))
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

# Set by server.py at startup - the dashboard's mesh gateway. robot_mesh cannot be used here:
# it requires an in-process Robot()/Simulation() as its Zenoh gateway, and the dashboard is a
# robot-less peer.
_bridge: Any = None

_refusal_listeners: list[Any] = []

def add_refusal_listener(cb: Any) -> Any:
    """Register ``cb(text)`` for continuable refusals raised inside the fleet tool. Returns a remover."""
    _refusal_listeners.append(cb)

    def _remove() -> None:
        try:
            _refusal_listeners.remove(cb)
        except ValueError:
            pass

    return _remove

def _notify_refusal(text: str) -> None:
    for cb in list(_refusal_listeners):
        try:
            cb(text)
        except Exception:  # noqa: BLE001 - a listener is a notification, never a gate
            logger.debug("refusal listener failed", exc_info=True)

def set_bridge(bridge: Any) -> None:
    global _bridge
    _bridge = bridge

def _peers_snapshot() -> dict:
    """Current fleet peers for the motion gate - read at call time, never cached."""
    if _bridge is None:
        return {}
    try:
        return (_bridge.snapshot() or {}).get("peers") or {}
    except Exception:  # noqa: BLE001 - unreadable snapshot means UNKNOWN, i.e. metal
        return {}

# --- pending HITL interrupt (one at a time: one agent, one turn lock) ---------
_pending_lock = threading.Lock()
_pending_interrupt: dict[str, Any] | None = None

def pending_interrupt() -> dict[str, Any] | None:
    """The unanswered motion confirm, or None. Surfaced in agent_status for reloads."""
    with _pending_lock:
        return dict(_pending_interrupt) if _pending_interrupt else None

def _set_pending(value: dict[str, Any] | None) -> None:
    global _pending_interrupt
    with _pending_lock:
        _pending_interrupt = value

def _abandon_pending() -> None:
    """A new prompt arrived over an unanswered confirm: drop the parked turn.

    History was never saved mid-interrupt, so rebuilding the agent restores the
    conversation from disk - the pre-turn boundary. Nothing wedges.
    """
    _set_pending(None)
    global _agent
    with _agent_lock:
        _agent = None

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
        """Coordinate robots on the mesh (dashboard gateway)."""
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
            from strands_robots.dashboard.agent_hitl import consume_grant
            from strands_robots.dashboard.agent_motion import agent_motion_allowed

            snap_peers = {}
            try:
                snap_peers = _bridge.snapshot().get("peers") or {}
            except Exception:  # noqa: BLE001 - an unreadable snapshot means UNKNOWN, i.e. metal
                snap_peers = {}
            # A human yes on the interrupt confirm deposits a one-shot grant;
            # consuming it satisfies the backstop without weakening it.
            approved = consume_grant(
                "fleet", {"action": "task", "target": target, "instruction": instruction},
            )
            if not approved:
                verdict = agent_motion_allowed(
                    "task", peer=snap_peers.get(target), target=target,
                )
                if not verdict["allowed"]:
                    _notify_refusal(verdict["reason"])
                    return {"status": "error", "content": [{"text": verdict["reason"]}]}
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

def _robot_mesh_tool() -> Any:
    """The SDK's own mesh tool, if importable - the agent-native surface.

    It carries its OWN SDK-native HITL interrupt on physical actions
    (tell/send/stop/broadcast/emergency_stop/rpc), so it is deliberately NOT
    in MOTION_ACTIONS - adding it there would double-gate every command. In
    this robot-less process it reaches the fleet via its gateway mesh (#10).
    """
    try:
        from strands_robots.tools.robot_mesh import robot_mesh

        return robot_mesh
    except Exception:  # noqa: BLE001 - the agent must build even if the tool cannot import
        logger.warning("robot_mesh tool unavailable - agent runs with fleet only", exc_info=True)
        return None

def _peer_proxy_tools() -> list:
    """One native tool per tool-worthy fleet peer, routed through the bridge.

    Robots are CHILD PROCESSES holding the serial buses / sim state, so
    Agent(tools=[Robot(...)]) in-process would collide on the bus; the proxy
    is indistinguishable to the agent and rides the validated mesh rails
    (peer_tools.py). No bridge yet = no proxies - the agent still builds.
    """
    if _bridge is None:
        return []
    try:
        from strands_robots.dashboard.peer_tools import build_peer_tools

        return build_peer_tools(_peers_snapshot(), send_cmd=_bridge.send_cmd)
    except Exception:  # noqa: BLE001 - the agent must build even if proxies cannot
        logger.warning("peer proxy tools unavailable - agent runs without them", exc_info=True)
        return []


_agent_fleet_sig: Any = None


def _build_agent() -> Any:
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    from strands import Agent

    from strands_robots.dashboard import settings
    from strands_robots.dashboard.agent_hitl import MotionInterruptHook

    cfg = settings.load()["agent"]
    proxies = _peer_proxy_tools()
    proxy_motion: dict[str, Any] = {}
    proxy_targets: dict[str, str] = {}
    if proxies:
        from strands_robots.dashboard.peer_tools import motion_actions_for

        proxy_motion = dict(motion_actions_for(proxies))
        proxy_targets = {t.tool_name: t.peer_id for t in proxies}
    global _agent_fleet_sig
    try:
        from strands_robots.dashboard.peer_tools import fleet_signature

        _agent_fleet_sig = fleet_signature(_peers_snapshot())
    except Exception:  # noqa: BLE001 - a signature failure must not block the build
        _agent_fleet_sig = None
    kwargs: dict[str, Any] = {
        "tools": [t for t in (_make_fleet_tool(), _robot_mesh_tool()) if t is not None] + list(proxies),
        "system_prompt": cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        "callback_handler": None,
        "hooks": [MotionInterruptHook(_peers_snapshot, proxy_motion, proxy_targets)],
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
    raw = _load_history()
    history = sanitize_history(raw)
    if raw and not history:
        # Poison in, nothing out: never restore a conversation a provider would
        # reject - start clean and keep the evidence on disk.
        backup = _backup_history_file()
        logger.warning(
            "chat history was unusable and was NOT restored - starting empty; kept a copy at %s",
            backup or "(no file)",
        )
    elif history:
        try:
            agent.messages = history
            logger.info("restored %d chat messages", len(history))
        except Exception as e:
            logger.warning("history restore failed: %s", e)
    return agent

def get_agent() -> Any:
    global _agent
    with _agent_lock:
        if _agent is not None and pending_interrupt() is None:
            # The tool list follows the mesh: a robot joining or leaving since
            # the build rebuilds the agent (history is restored from disk, so
            # this costs one build, not the conversation). Never mid-interrupt:
            # a parked turn must resume against the agent that raised it.
            try:
                from strands_robots.dashboard.peer_tools import fleet_signature

                if fleet_signature(_peers_snapshot()) != _agent_fleet_sig:
                    logger.info("fleet changed since agent build - refreshing its tools")
                    _agent = None
            except Exception:  # noqa: BLE001 - an unreadable fleet never blocks a turn
                pass
        if _agent is None:
            _agent = _build_agent()
        return _agent

def reset_agent(clear_history_too: bool = False, *, clear_history: bool | None = None) -> None:
    """Drop the agent so the next turn rebuilds it from current settings."""
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
        # Not built yet: report what the build WOULD produce (pure rules on the
        # live fleet), not a hardcoded guess - the old ['fleet'] badge lied.
        tools = ["fleet", "robot_mesh"]
        with contextlib.suppress(Exception):
            from strands_robots.dashboard.peer_tools import expected_tool_names

            tools += expected_tool_names(_peers_snapshot())
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
                    full = ""
                    for blk in tr.get("content", []):
                        if isinstance(blk, dict) and blk.get("text"):
                            full = blk["text"]
                            break
                    event = {
                        "type": "tool", "status": tr.get("status", "done"),
                        "id": tr.get("toolUseId", ""), "result_preview": full[:300],
                    }
                    # A continuable refusal must be clickable WHERE IT HAPPENED.
                    if tr.get("status") == "error":
                        from strands_robots.dashboard.consent import classify_refusal

                        need = classify_refusal(full)
                        if need is not None:
                            event["needs_consent"] = need.as_dict()
                    self.q.put(event)

def _is_history_poisoned(exc: Exception) -> bool:
    """Does this failure look like a rejected restored conversation?"""
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
    """Run one agent turn in a worker thread, streaming events into q. ``cancel`` is set by the
    websocket handler when its client goes away.
    """
    if pending_interrupt() is not None:
        # A fresh prompt over an unanswered confirm: the human moved on.
        _abandon_pending()
        q.put({"type": "notice", "text": "the pending motion confirm was dismissed - nothing was sent"})
    _run_agent_input(prompt, q, cancel)

def resume_interrupt_blocking(
    interrupt_id: str,
    response: Any,
    q: "queue.Queue[dict]",
    cancel: "threading.Event | None" = None,
) -> None:
    """Answer a pending motion confirm and continue the SAME turn."""
    pending = pending_interrupt()
    if pending is None or pending.get("id") != interrupt_id:
        q.put({
            "type": "error",
            "error": "no pending confirm with that id - it may have been dismissed or already answered",
        })
        q.put({"type": "__END__"})
        return
    payload = [{"interruptResponse": {"interruptId": interrupt_id, "response": response}}]
    _run_agent_input(payload, q, cancel)

def _run_agent_input(
    agent_input: Any,
    q: "queue.Queue[dict]",
    cancel: "threading.Event | None" = None,
) -> None:
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
                result = agent(agent_input)
            except TurnCancelled:
                raise
            except Exception as exc:
                # Self-heal instead of failing every future turn: drop the
                # restored history, rebuild, and try once more from a clean
                # conversation. Resumes are not retried - the parked turn is
                # what the retry would destroy.
                if not isinstance(agent_input, str) or not (agent.messages and _is_history_poisoned(exc)):
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
                result = agent(agent_input)
            if getattr(result, "stop_reason", None) == "interrupt" and getattr(result, "interrupts", None):
                # The motion gate paused the turn. Park it: history is NOT
                # saved, so a reload restores the pre-turn boundary.
                it = result.interrupts[0]
                event = {"type": "interrupt", "id": it.id, "name": it.name, "reason": it.reason}
                _set_pending({"id": it.id, "name": it.name, "reason": it.reason})
                q.put(event)
                return
            _set_pending(None)
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
