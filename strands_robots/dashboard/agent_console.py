"""The dashboard's agent: a Strands Agent whose hands are the simulator sessions.

One :class:`Console` is one operator conversation. Its tools go through the
same :class:`~strands_robots.dashboard.routes_sim.Safety` object the HTTP
routes use, so the e-stop refuses the agent exactly as it refuses a button,
and every accepted command is the same proof that the lockout is clear.

Anything that moves a robot - here ``sim_set_joints`` - raises the SDK
interrupt the real-hardware hook uses (:mod:`strands_robots.dashboard.agent_hitl`),
so the browser shows a consent card and the same turn resumes on a yes. The
operator can grant one call or the rest of the conversation; the grant lives
in this object and dies with the socket. Stopping is never gated.

The agent also sees the FLEET when the server hands it its mesh bridge: a
``fleet`` tool lists every robot on the mesh with its state, ``spawn_robot``
starts a registry robot in simulation as a mesh peer (so its card appears on
the dashboard at once), ``despawn_robot`` stops one, and every tool-worthy peer
is a native tool of its own (:mod:`strands_robots.dashboard.peer_tools`) whose
motion verbs on a real arm go through :class:`~strands_robots.dashboard.agent_hitl.MotionInterruptHook`.
The tool list follows the mesh: when the fleet signature changes between
turns, the agent is rebuilt with the new tools and its conversation carried over.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)

INTERRUPT_NAME = "sim_motion"
MODEL_ENV = "STRANDS_MODEL_ID"
MAX_PROMPT_CHARS = 8_000

SYSTEM_PROMPT = """You are the strands-robots dashboard agent. You operate robots for an operator who is
watching the same screen. Be brief.

Two kinds of robot exist here. In-process simulation sessions (sim_sessions, sim_start, sim_state,
sim_set_joints, sim_reset, sim_stop) render in the Sim tab. Fleet robots are peers on the zenoh mesh
and appear as cards on the dashboard: `fleet` lists them with their state, `spawn_robot` creates a new
simulated robot as a mesh peer (its card appears within seconds), `despawn_robot` removes one, and
each peer is also a tool named after it (dashes become underscores) whose actions are what that peer
accepts: status, state, set_joints (target_joints, radians), reset, step, stop, execute/start for
policy rollouts, and on a simulation peer every published action of the simulation tool as well
(add_object, list_objects, move_object, add_camera, render, get_robot_state, move_to, set_gripper and
the rest of its enum), called with the action's own parameters as fields. spawn_robot returns the new
peer's tool names and they are callable in the same turn. When the operator says "create a robot",
use spawn_robot. When they name a robot, use that robot's own tool. To put something into a robot's
world, use that robot's tool with the simulation action (a cube is add_object with name, shape,
size, color, position). Joint positions are radians unless the peer's state says otherwise; joints
are addressed by name or by 1-based index as strings. A move you request may be put to the operator
first; if they decline, say so and stop. Never work around a refusal or an e-stop."""

#: agent tool name -> does it move the robot (and so asks the operator first)?
MOTION_TOOLS: frozenset[str] = frozenset({"sim_set_joints"})


def model_id() -> str:
    """``STRANDS_MODEL_ID`` if the operator set one, else whatever model the SDK defaults to.

    The console holds no model name of its own: a second copy diverges from the
    installed SDK's default the moment either moves, and the operator meets that
    divergence as a ValidationException on their first turn.
    """
    from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID

    return os.environ.get(MODEL_ENV) or DEFAULT_BEDROCK_MODEL_ID


def default_model() -> Any:
    """A Bedrock model on :func:`model_id` (``STRANDS_MODEL_ID``, the variable the CLI honours)."""
    from strands.models import BedrockModel

    return BedrockModel(
        model_id=model_id(),
        region_name=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1",
    )


@dataclass
class Grants:
    """What the operator has already said yes to, for this conversation only."""

    sessions: set[str] = field(default_factory=set)

    def covers(self, session_id: str) -> bool:
        """Has the operator allowed motion on this session for the rest of the conversation?"""
        return session_id in self.sessions

    def extend(self, session_id: str) -> None:
        """Remember a 'for this conversation' yes."""
        self.sessions.add(session_id)


def response_approves(response: Any) -> tuple[bool, bool]:
    """(approved, for the rest of the conversation). Anything but an explicit yes is a no."""
    if isinstance(response, bool):
        return response, False
    if isinstance(response, Mapping):
        approve = response.get("approve")
        if isinstance(approve, bool):
            return approve, bool(response.get("always", False)) and approve
        return False, False
    if isinstance(response, str):
        return response.strip().lower() in {"yes", "y", "approve", "approved", "ok"}, False
    return False, False


class MotionGate(HookProvider):
    """Interrupt before any motion tool call the operator has not already granted."""

    def __init__(self, grants: Grants) -> None:
        self._grants = grants

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe the gate to every tool call the agent is about to make."""
        registry.add_callback(BeforeToolCallEvent, self._gate)

    def _gate(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = str(tool_use.get("name") or "")
        if name not in MOTION_TOOLS:
            return
        tool_input = dict(tool_use.get("input") or {})
        session_id = str(tool_input.get("session_id") or "")
        if self._grants.covers(session_id):
            return
        reason = {
            "tool": name,
            "session_id": session_id,
            "positions": tool_input.get("positions"),
            "detail": _detail(tool_input),
        }
        response = event.interrupt(INTERRUPT_NAME, reason=reason)
        approved, always = response_approves(response)
        from strands_robots._hitl_audit import log_operator_response

        log_operator_response("dashboard_agent_console", name, session_id, approved=approved, response=response)
        if approved:
            if always:
                self._grants.extend(session_id)
            return
        event.cancel_tool = "The operator declined this motion. Do not retry it; tell them and wait."


def _detail(tool_input: Mapping[str, Any]) -> str:
    positions = tool_input.get("positions")
    if isinstance(positions, Mapping):
        return ", ".join(f"{k} → {float(v):.3f} rad" for k, v in positions.items() if isinstance(v, (int, float)))
    return json.dumps(positions)


def build_tools(safety: Any) -> list[Any]:
    """The agent's hands: every call goes through ``safety`` like a button press would."""

    def _session(session_id: str) -> Any:
        session = safety.store.get(session_id)
        if session is None:
            raise ValueError(f"no session {session_id}")
        return session

    def _snapshot(session: Any) -> dict[str, Any]:
        snap = session.snapshot.as_dict()
        return {k: v for k, v in snap.items() if k != "model_path"}

    def _gate(action: str) -> None:
        # Safety.gate raises HTTPException(423); the agent should read a sentence.
        from fastapi import HTTPException

        try:
            safety.gate(action)
        except HTTPException as exc:
            raise PermissionError(str(exc.detail))

    def _accepted(drop: str | None = None) -> None:
        """Fold the proof this command was accepted, or report the e-stop that beat it.

        Args:
            drop: a session to forget when the e-stop landed. A session that was
                admitted and then refused must not be left in the store: it holds
                one of ``MAX_SESSIONS`` slots and thaws into a running robot on
                resume - a robot the caller was told was refused.

        Raises:
            PermissionError: the lockout latched while this command was in flight.
        """
        from fastapi import HTTPException

        try:
            safety.accepted()
        except HTTPException as exc:
            if drop is not None:
                safety.store.remove(drop)
            raise PermissionError(str(exc.detail))

    @tool
    def robots() -> list[dict[str, Any]]:
        """Robots that can be simulated: name, dof, and whether a session already runs one."""
        from strands_robots.dashboard.fleet import registry_robots

        running = {s.robot for s in safety.store.all()}
        return [{**r, "running": r["name"] in running} for r in registry_robots("sim")]

    @tool
    def sim_sessions() -> list[dict[str, Any]]:
        """Every running simulation: id, robot, state, joint names, current joint positions."""
        return [_snapshot(s) for s in safety.store.all()]

    @tool
    def sim_start(robot: str) -> dict[str, Any]:
        """Start a simulation of a registry robot and return its session (id, joints).

        A start that does not finish is forgotten rather than handed back: the
        state published before the first frame is ``running``, so a session that
        never rendered would be reported as a robot the operator can watch while
        it streams nothing and holds one of the store's slots.
        """
        _gate("create")
        from strands_robots.dashboard import routes_sim
        from strands_robots.registry.robots import get_robot, resolve_name

        entry = get_robot(robot)
        if entry is None or not entry.get("asset"):
            raise ValueError(f"{robot!r} is not a robot with a simulation asset")
        session = safety.store.create(resolve_name(robot))
        timeout = routes_sim.READY_TIMEOUT
        if not session.wait_ready(timeout):
            safety.store.remove(session.id)
            raise RuntimeError(f"{robot} did not render a first frame within {timeout:.0f}s")
        if session.snapshot.state == "error":
            safety.store.remove(session.id)
            raise RuntimeError(f"could not start {robot}: {session.snapshot.error}")
        _accepted(drop=session.id)
        return _snapshot(session)

    @tool
    def sim_state(session_id: str) -> dict[str, Any]:
        """The session's latest snapshot: state, sim time, joint names and positions (radians)."""
        return _snapshot(_session(session_id))

    @tool
    def sim_set_joints(session_id: str, positions: dict[str, float]) -> dict[str, Any]:
        """Move joints of a simulated robot to target positions in radians.

        Args:
            session_id: which simulation (from sim_sessions).
            positions: joint name or 1-based index (as a string) -> radians, e.g. {"2": 1.2}.
        """
        _gate("set_joints")
        if not positions or any(not isinstance(v, (int, float)) for v in positions.values()):
            raise ValueError("positions must map joint -> number")
        result = _session(session_id).command("set_joints", positions=dict(positions))
        if result.get("status") == "error":
            raise ValueError(str(result.get("content")))
        _accepted()
        return dict(result)

    @tool
    def sim_reset(session_id: str) -> dict[str, Any]:
        """Return a simulated robot to its home pose."""
        _gate("reset")
        result = _session(session_id).command("reset")
        _accepted()
        return dict(result)

    @tool
    def sim_stop(session_id: str) -> dict[str, Any]:
        """Stop a simulation and forget it. Never refused."""
        return {"ok": safety.store.remove(session_id)}

    @tool
    def emergency_stop() -> dict[str, Any]:
        """Freeze every simulation and latch the dashboard's lockout. Never refused."""
        return dict(safety.estop(by="agent"))

    return [robots, sim_sessions, sim_start, sim_state, sim_set_joints, sim_reset, sim_stop, emergency_stop]


#: How long ``spawn_robot`` waits for the new peer's presence on the mesh before
#: reporting it as started-but-not-yet-seen. A MuJoCo so101 announces in 2-5 s
#: on a laptop; the settle window the Devices panel uses is the same order.
SPAWN_PRESENCE_TIMEOUT_S = 20.0
SPAWN_POLL_S = 0.25

#: Names the fixed fleet tools take, so ``expected_tool_names`` and the badge agree.
FLEET_TOOL_NAMES: tuple[str, ...] = ("fleet", "spawn_robot", "despawn_robot")


def peer_summary(peer_id: str, peer: Mapping[str, Any]) -> dict[str, Any]:
    """One fleet row for the agent: what the peer is, whether it is fresh, and its joints."""
    from strands_robots.dashboard.peer_tools import classify_peer, sanitize_tool_name

    presence = peer.get("presence") or {}
    state = peer.get("state") or {}
    joints = state.get("joints")
    row: dict[str, Any] = {
        "peer_id": peer_id,
        "tool": sanitize_tool_name(peer_id),
        "kind": classify_peer(peer_id, peer),
        "robot_type": presence.get("robot_type"),
        "hostname": presence.get("hostname"),
        "stale": bool(peer.get("stale")),
        "origin": peer.get("origin"),
    }
    if isinstance(joints, Mapping):
        row["joints"] = {str(k): v for k, v in joints.items()}
    if state.get("task") is not None:
        row["task"] = state.get("task")
    if state.get("status") is not None:
        row["status"] = state.get("status")
    if peer.get("cameras"):
        row["cameras"] = sorted(peer["cameras"])
    return row


def build_fleet_tools(
    bridge: Any, devices: Any | None, on_spawned: Callable[[list[str]], list[str]] | None = None
) -> list[Any]:
    """The fleet tools: list the mesh, create a sim robot on it, remove one. Bridge-less = none.

    ``on_spawned(peer_ids)`` is called by ``spawn_robot`` once the new peer is on the
    mesh, with the ids seen; it returns the tool names now callable. The console passes
    its :meth:`Console.adopt`, so a spawned peer's tool is in the live agent's registry
    before ``spawn_robot`` returns and the same turn can use it. Without it the names are
    predicted and usable on the next turn.
    """
    if bridge is None:
        return []

    def _peers() -> dict[str, Any]:
        snap = bridge.snapshot()
        peers = snap.get("peers") if isinstance(snap, Mapping) else None
        return dict(peers) if isinstance(peers, Mapping) else {}

    @tool
    def fleet() -> dict[str, Any]:
        """Every robot on the mesh right now: peer id, the tool that drives it, kind (sim/real/host),
        freshness, joint positions and any running task. Stale peers are listed, marked stale."""
        from strands_robots.dashboard.peer_tools import KIND_SKIP

        peers = _peers()
        rows = [peer_summary(pid, p) for pid, p in peers.items()]
        rows = [r for r in rows if r["kind"] != KIND_SKIP]
        managed = []
        if devices is not None:
            try:
                managed = list(devices.managed_children())
            except Exception:  # noqa: BLE001 - the roster is a courtesy, the mesh is the truth
                managed = []
        return {"robots": rows, "count": len(rows), "managed_by_this_dashboard": managed}

    @tool
    def spawn_robot(robot: str, peer_id: str | None = None) -> dict[str, Any]:
        """Create a simulated robot as a mesh peer, so it appears on the dashboard fleet at once.

        Args:
            robot: a registry name with a simulation asset, e.g. so101, franka_panda, unitree_go2.
            peer_id: optional mesh name for it (letters, digits, - _ .); default <robot>-sim-<n>.

        Returns the peer id, the child peer that publishes its joints (<peer>__<robot>) and the
        tool names the agent can use for them, callable in this same turn (add a cube with the
        child's tool: action=add_object). Simulation only: a real robot needs a serial port and is
        started from the Devices panel.
        """
        if devices is None:
            raise RuntimeError("this dashboard has no device manager, so it cannot start robot processes")
        result = devices.spawn(str(robot).strip(), "sim", peer_id=peer_id or None)
        if not isinstance(result, dict):
            return {"result": result}
        if result.get("error"):
            raise ValueError(str(result["error"]))
        pid = str(result.get("peer_id") or peer_id or "")
        deadline = time.monotonic() + SPAWN_PRESENCE_TIMEOUT_S
        seen: list[str] = []
        while time.monotonic() < deadline:
            peers = _peers()
            seen = sorted(p for p in peers if p == pid or p.startswith(f"{pid}__"))
            if any("__" in p for p in seen):
                break
            time.sleep(SPAWN_POLL_S)
        from strands_robots.dashboard.peer_tools import sanitize_tool_name

        adopted: list[str] = []
        if seen and on_spawned is not None:
            try:
                adopted = list(on_spawned(seen))
            except Exception:  # noqa: BLE001 - the spawn succeeded; the tools arrive next turn instead
                logger.warning("spawn_robot: could not register the new peer's tools mid-turn", exc_info=True)
                adopted = []
        out = {
            **result,
            "peer_id": pid,
            "on_mesh": seen,
            "tools": adopted or [sanitize_tool_name(p) for p in seen],
            "note": (
                "the robot's card is on the dashboard now; its joints publish on the child peer; "
                + ("its tools are callable now, in this turn" if adopted else "its tools are callable next turn")
                if seen
                else f"the process started but no presence arrived within {SPAWN_PRESENCE_TIMEOUT_S:g}s; "
                "call fleet again in a moment"
            ),
        }
        return out

    @tool
    def despawn_robot(peer_id: str) -> dict[str, Any]:
        """Stop a robot this dashboard started and remove it from the mesh. Never refused."""
        if devices is None:
            raise RuntimeError("this dashboard has no device manager")
        result = devices.despawn(str(peer_id).strip())
        return dict(result) if isinstance(result, dict) else {"result": result}

    return [fleet, spawn_robot, despawn_robot]


def expected_tool_names(bridge: Any | None) -> list[str]:
    """The tool names a console over this bridge would carry, without building an agent."""
    from strands_robots.dashboard.peer_tools import expected_tool_names as proxy_names

    names = [
        "robots",
        "sim_sessions",
        "sim_start",
        "sim_state",
        "sim_set_joints",
        "sim_reset",
        "sim_stop",
        "emergency_stop",
    ]
    if bridge is None:
        return names
    names.extend(FLEET_TOOL_NAMES)
    try:
        snap = bridge.snapshot()
        peers = snap.get("peers") if isinstance(snap, Mapping) else {}
        names.extend(proxy_names(peers or {}))
    except Exception:  # noqa: BLE001 - a badge must not fail on a bridge hiccup
        logger.debug("expected_tool_names: bridge snapshot unreadable", exc_info=True)
    return names


class Console:
    """One operator conversation.

    ``bridge`` (the server's :class:`~strands_robots.dashboard.mesh_bridge.MeshBridge`) and
    ``devices`` (its :class:`~strands_robots.dashboard.device_manager.DeviceManager`) are optional:
    without them the console is the sim-only agent it always was, which is also what the tests
    that install their own factory get.
    """

    def __init__(
        self, safety: Any, model: Any | None = None, bridge: Any | None = None, devices: Any | None = None
    ) -> None:
        self.grants = Grants()
        self._safety = safety
        self._model = model if model is not None else default_model()
        self._bridge = bridge
        self._devices = devices
        self._signature: frozenset[tuple[str, str]] = frozenset()
        self._hook: Any | None = None
        self.agent = self._build(messages=None)

    def _peers(self) -> dict[str, Any]:
        if self._bridge is None:
            return {}
        try:
            snap = self._bridge.snapshot()
        except Exception:  # noqa: BLE001 - an unreadable mesh means no proxies, not no agent
            logger.debug("console: bridge snapshot unreadable", exc_info=True)
            return {}
        peers = snap.get("peers") if isinstance(snap, Mapping) else None
        return dict(peers) if isinstance(peers, Mapping) else {}

    def _build(self, messages: list[Any] | None) -> Any:
        """An Agent over the fleet as it is now; ``messages`` carries a conversation across a rebuild."""
        from strands_robots.dashboard.agent_hitl import MotionInterruptHook
        from strands_robots.dashboard.peer_tools import build_peer_tools, fleet_signature, motion_actions_for

        tools: list[Any] = build_tools(self._safety)
        hooks: list[Any] = [MotionGate(self.grants)]
        self._hook = None
        if self._bridge is not None:
            peers = self._peers()
            self._signature = fleet_signature(peers)
            bridge = self._bridge
            proxies = build_peer_tools(peers, bridge.send_cmd, peer_state=lambda pid: bridge.peers.get(pid))
            tools.extend(build_fleet_tools(bridge, self._devices, on_spawned=self.adopt))
            tools.extend(proxies)
            self._hook = MotionInterruptHook(
                peers_snapshot=lambda: bridge.peers,
                proxy_motion=motion_actions_for(proxies, peers),
                proxy_targets={t.tool_name: t.peer_id for t in proxies},
            )
            hooks.append(self._hook)
        return Agent(
            model=self._model,
            messages=messages,
            tools=tools,
            hooks=hooks,
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )

    def adopt(self, peer_ids: list[str]) -> list[str]:
        """Register the proxies for *peer_ids* into the LIVE agent, mid-turn. Returns their tool names.

        The SDK reads the registry's tool specs before every model call, so a
        tool registered while a turn runs is offered on that turn's next call:
        ``spawn_robot`` hands the model the new robot instead of "next turn".
        The fleet signature grows with them so the next turn does not rebuild
        for a change already applied; the motion hook learns the new proxies the
        way ``_build`` taught it the first ones (a spawned peer is a sim, so it
        adds no motion row, and a real peer would).
        """
        if self._bridge is None:
            return []
        from strands_robots.dashboard.peer_tools import build_peer_tools, fleet_signature, motion_actions_for

        bridge = self._bridge
        peers = {pid: p for pid, p in self._peers().items() if pid in set(peer_ids)}
        if not peers:
            return []
        registry: Any = getattr(self.agent, "tool_registry", None)
        held = set(self.tool_names())
        proxies = build_peer_tools(peers, bridge.send_cmd, peer_state=lambda pid: bridge.peers.get(pid))
        names: list[str] = []
        for proxy in proxies:
            if proxy.tool_name in held or registry is None:
                names.append(proxy.tool_name)
                continue
            registry.register_tool(proxy)
            names.append(proxy.tool_name)
        if self._hook is not None:
            self._hook.adopt(motion_actions_for(proxies, peers), {t.tool_name: t.peer_id for t in proxies})
        self._signature = self._signature | fleet_signature(peers)
        logger.info("console: adopted %d peer tool(s) mid-turn: %s", len(names), ", ".join(names))
        return names

    def tool_names(self) -> list[str]:
        """The tools the agent holds right now."""
        registry: Any = getattr(self.agent, "tool_registry", None)
        if registry is None:
            return []
        try:
            return sorted(str(spec["name"]) for spec in registry.get_all_tool_specs())
        except Exception:  # noqa: BLE001 - a name list is a courtesy
            return []

    def refresh(self) -> bool:
        """Rebuild the agent if the fleet changed since it was built. Returns True when it did.

        The conversation survives: the new agent starts from the old one's messages. A
        pending interrupt is never rebuilt across (the resume must land on the agent that raised it).
        """
        if self._bridge is None:
            return False
        from strands_robots.dashboard.peer_tools import fleet_signature

        signature = fleet_signature(self._peers())
        if signature == self._signature:
            return False
        logger.info(
            "console: fleet changed (%d -> %d tool-worthy peers); rebuilding tools",
            len(self._signature),
            len(signature),
        )
        self.agent = self._build(messages=list(getattr(self.agent, "messages", []) or []))
        return True

    async def run(self, prompt: Any) -> AsyncIterator[dict[str, Any]]:
        """Stream one turn as flat JSON events: text, tool_use, tool_result, interrupt, done, error."""
        try:
            if not _is_resume(prompt) and self.refresh():
                yield {"type": "tools", "names": self.tool_names()}
            async for event in self.agent.stream_async(prompt):
                for out in _translate(event):
                    yield out
                if "result" in event:
                    result = event["result"]
                    interrupts = list(getattr(result, "interrupts", None) or [])
                    if interrupts:
                        for it in interrupts:
                            yield {"type": "interrupt", "id": it.id, "name": it.name, "reason": it.reason}
                    else:
                        yield {"type": "done", "stop_reason": str(getattr(result, "stop_reason", ""))}
        except Exception as exc:  # noqa: BLE001 - the operator reads it as one line
            logger.warning("agent console turn failed: %s", exc)
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def resume(interrupt_id: str, approve: bool, always: bool = False) -> list[dict[str, Any]]:
        """The prompt that answers an interrupt - the SDK's interruptResponse block."""
        return [
            {
                "interruptResponse": {
                    "interruptId": interrupt_id,
                    "response": {"approve": bool(approve), "always": bool(always)},
                }
            }
        ]


def _is_resume(prompt: Any) -> bool:
    """Is this prompt the answer to an interrupt (so the agent that raised it must stay)?"""
    return isinstance(prompt, list) and any(isinstance(b, Mapping) and "interruptResponse" in b for b in prompt)


def _translate(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SDK stream events -> the flat events the browser renders. Text deltas and whole messages only."""
    if "data" in event and isinstance(event["data"], str):
        return [{"type": "text", "text": event["data"]}]
    message = event.get("message")
    out: list[dict[str, Any]] = []
    if isinstance(message, Mapping):
        for block in message.get("content") or []:
            if "toolUse" in block:
                tu = block["toolUse"]
                out.append({"type": "tool_use", "name": tu.get("name"), "input": tu.get("input")})
            elif "toolResult" in block:
                tr = block["toolResult"]
                texts = [str(c.get("text", "")) for c in tr.get("content", []) if isinstance(c, Mapping)]
                out.append(
                    {"type": "tool_result", "status": tr.get("status"), "text": "\n".join(t for t in texts if t)[:2000]}
                )
    return out
