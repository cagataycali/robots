"""Fleet agent for the dashboard - one Strands Agent driving the whole mesh.

The agent's toolset is the ``robot_mesh`` tool (peers/tell/send/broadcast/
stop/emergency_stop/...) so a single natural-language instruction can target
one robot or the whole fleet, exactly the GOAL.md bottom bar:

    > everyone pick up your cube      -> robot_mesh broadcast
    > tell so101-arm-1 to wave        -> robot_mesh tell target=so101-arm-1

Chat turns stream over a queue (token / reasoning / tool events) drained by
the /ws/chat websocket handler - the scout dashboard pattern.

HITL note: robot_mesh actuation actions are gated behind Strands interrupts
by default and FAIL CLOSED outside an approving host. For the dashboard the
human typed the instruction themselves, so we default the gate to "none"
(overridable with STRANDS_MESH_HITL_ACTIONS).
"""

from __future__ import annotations

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
        HISTORY_FILE.write_text(json.dumps(messages[-200:], default=str))
    except Exception as e:
        logger.debug("could not save chat history: %s", e)

# Set by server.py at startup - the dashboard's mesh gateway. robot_mesh
# cannot be used here: it requires an in-process Robot()/Simulation() as its
# Zenoh gateway (BUGS.md #10), and the dashboard is a robot-less peer.
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
            # Child sim peers ("<parent>__<robot>") cannot execute themselves
            # (upstream BUGS.md #11) - route to the parent Simulation peer with
            # robot_name so _dispatch_sim_policy handles it.
            if "__" in target:
                parent, _, robot_name = target.partition("__")
                cmd["robot_name"] = robot_name
                target = parent
            res = _bridge.send_cmd(target, cmd, timeout=float(duration) + 30.0)
            return {"status": "success", "content": [{"text": _json.dumps(res)[:1500]}]}

        if action == "stop":
            if not target:
                return {"status": "error", "content": [{"text": "stop requires target"}]}
            res = _bridge.send_cmd(target, {"action": "stop"}, timeout=10.0)
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        if action == "stop_all":
            results = {}
            for pid, p in _bridge.snapshot()["peers"].items():
                if p.get("stale"):
                    continue
                results[pid] = _bridge.send_cmd(pid, {"action": "stop"}, timeout=5.0)
            return {"status": "success", "content": [{"text": _json.dumps(results)[:1500]}]}

        if action == "status":
            if not target:
                return {"status": "error", "content": [{"text": "status requires target"}]}
            res = _bridge.send_cmd(target, {"action": "status"}, timeout=10.0)
            return {"status": "success", "content": [{"text": _json.dumps(res)[:800]}]}

        return {"status": "error", "content": [{"text": f"unknown action {action!r}. Valid: peers, task, stop, stop_all, status"}]}

    return fleet


def _build_agent() -> Any:
    os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

    from strands import Agent

    agent = Agent(
        tools=[_make_fleet_tool()],
        system_prompt=os.getenv("DASHBOARD_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        callback_handler=None,
    )
    # Restore conversation history from disk so the fleet agent survives
    # dashboard restarts (BUGS.md #12 - "amnesiac agent").
    history = _load_history()
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


def reset_agent() -> None:
    global _agent
    with _agent_lock:
        _agent = None


class WSStreamHandler:
    """Strands callback handler -> thread-safe queue of UI events."""

    def __init__(self, q: "queue.Queue[dict]") -> None:
        self.q = q
        self._tool_ids: set[str] = set()

    def __call__(self, **kwargs: Any) -> None:
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


def run_turn_blocking(prompt: str, q: "queue.Queue[dict]") -> None:
    """Run one agent turn in a worker thread, streaming events into q."""
    try:
        with _turn_lock:
            agent = get_agent()
            agent.callback_handler = WSStreamHandler(q)
            result = agent(prompt)
            try:
                _save_history(list(agent.messages))
            except Exception:
                pass
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
    except Exception as e:
        logger.exception("agent turn failed")
        q.put({"type": "error", "error": str(e)})
    finally:
        q.put({"type": "__END__"})
