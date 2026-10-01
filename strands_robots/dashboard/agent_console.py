"""The dashboard's agent: a Strands Agent whose hands are the robots on the mesh.

One :class:`Console` is one operator conversation. The agent holds no robot of
its own: every robot it can see or move is a peer on the zenoh mesh, the same
peers whose cards the dashboard shows. A ``fleet`` tool lists them with their
state, ``spawn_robot`` starts a registry robot in simulation as a mesh peer (so
its card appears on the dashboard at once), ``despawn_robot`` stops one, and
every tool-worthy peer is a native tool of its own
(:mod:`strands_robots.dashboard.peer_tools`) whose motion verbs on a real arm go
through :class:`~strands_robots.dashboard.agent_hitl.MotionInterruptHook`: the
browser shows a consent card and the same turn resumes on a yes. The tool list
follows the mesh: when the fleet signature changes between turns, the agent is
rebuilt with the new tools and its conversation carried over.

``emergency_stop`` is the one tool that is not a peer: it latches the dashboard's
own lockout through the same :class:`~strands_robots.dashboard.routes_sim.Safety`
object the HTTP routes use, so the e-stop refuses the agent exactly as it refuses
a button. Stopping is never gated.

The in-process simulation tools this console once carried (``sim_start``,
``sim_set_joints`` and their siblings) are gone: a robot the agent drives is a
mesh peer or it is not driven from here, so one gate, one tool shape and one
card serve every robot alike.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from strands import Agent, tool

logger = logging.getLogger(__name__)

MODEL_ENV = "STRANDS_MODEL_ID"
MAX_PROMPT_CHARS = 8_000

SYSTEM_PROMPT = """You are the strands-robots dashboard agent. You operate robots for an operator who is
watching the same screen. Be brief.

Every robot here is a peer on the zenoh mesh and appears as a card on the dashboard: `fleet` lists
them with their state, `spawn_robot` creates a new simulated robot as a mesh peer (its card appears
within seconds), `despawn_robot` removes one, and each peer is also a tool named after it (dashes
become underscores) whose actions are what that peer accepts: status, state, set_joints
(target_joints, radians), reset, step, stop, execute/start for policy rollouts, and on a simulation
peer every published action of the simulation tool as well (add_object, list_objects, move_object,
add_camera, render, get_robot_state, move_to, set_gripper and the rest of its enum), called with the
action's own parameters as fields. spawn_robot returns the new peer's tool names and they are
callable in the same turn. When the operator says "create a robot", use spawn_robot. When they name
a robot, use that robot's own tool. To put something into a robot's world, use that robot's tool
with the simulation action (a cube is add_object with name, shape, size, color, position). Joint
positions are radians unless the peer's state says otherwise; joints are addressed by name or by
1-based index as strings. A move you request may be put to the operator first; if they decline, say
so and stop. Never work around a refusal or an e-stop."""

#: How the agent chooses a policy for a peer. Appended to the prompt; the facts it
#: points at are on every ``fleet`` row (:mod:`strands_robots.dashboard.peer_policies`).
POLICY_GUIDANCE = """Each fleet row carries `policies`: the robot the peer is and `can_run`, the providers that apply to
it with the kwargs the wire accepts (types, bounds) and what each needs. For a motion request pick a
provider from that list: a Unitree G1 walks with `wbc` (model_path = the checkpoint directory on the
robot host, walk true, target_velocity [vx, vy, wz] within the bounds shown); an arm runs
`lerobot_local` with pretrained_name_or_path and policy_type. `mock` is a sine test that proves the
plumbing and never performs a task: use it only when the operator asks for it. When a provider needs
a checkpoint or a server and none was given, ask for it instead of guessing or falling back to
mock. Send only the kwargs listed for that provider; the robot host refuses anything else."""

SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + POLICY_GUIDANCE


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


#: How long one peer gets to confirm a stop before the fleet stop moves on; the route uses the same.
STOP_TIMEOUT_S = 5.0


def fleet_stop(safety: Any, bridge: Any | None, by: str = "agent") -> dict[str, Any]:
    """Stop everything this dashboard can reach, both rails, and say what confirmed.

    The local ``safety`` lockout latches first (it refuses every relayed command
    until an operator resumes). With a bridge, every live peer is then asked to
    ``stop`` and the signed fleet e-stop engages the lockout on every listening
    peer, the same two rails ``POST /api/mesh/safety/estop`` fires. ``all_stopped``
    is True only when every live peer confirmed; anything else keeps shouting.
    """
    local = safety.estop(by=by)
    out: dict[str, Any] = {"lockout": local["lockout"], "frozen": local.get("frozen", [])}
    if bridge is None:
        return {
            **out,
            "fleet": None,
            "all_stopped": False,
            "note": "no mesh bridge: nothing beyond this process was stopped",
        }
    from strands_robots.dashboard.mesh_bridge import stop_outcome

    peers = list(bridge.live_peers())
    stale = sorted(set(bridge.peers) - set(peers))

    def _stop(peer: str) -> dict[str, Any]:
        try:
            result = bridge.send_cmd(peer, {"action": "stop"}, timeout=STOP_TIMEOUT_S, source="estop")
        except Exception as exc:  # noqa: BLE001 - a peer that cannot be asked is reported, not raised past the others
            result = {"error": str(exc)}
        return result if isinstance(result, dict) else {"error": str(result)}

    with ThreadPoolExecutor(max_workers=max(1, len(peers))) as pool:
        answers = list(pool.map(_stop, peers))
    per_peer = {peer: {**stop_outcome(answer), "result": answer} for peer, answer in zip(peers, answers, strict=True)}
    counts = {"stopped": 0, "not_stopped": 0, "no_answer": 0}
    for info in per_peer.values():
        counts[info["state"]] = counts.get(info["state"], 0) + 1
    all_stopped = bool(peers) and counts["stopped"] == len(peers)
    bridge.record_activity(
        "estop",
        "stop_all",
        target="fleet",
        detail=f"{counts['stopped']}/{len(peers)} confirmed stopped",
        ok=all_stopped,
    )
    signed = bridge.signed_estop()
    return {
        **out,
        "targeted": peers,
        "stale_skipped": stale,
        "counts": counts,
        "all_stopped": all_stopped,
        "stopped": per_peer,
        "signed_rail": {k: v for k, v in signed.items() if k != "responses"},
        "lockout_engaged": bool(signed.get("lockout_engaged")),
        "peers_not_stopped": list(signed.get("peers_not_stopped", [])),
    }


def build_tools(safety: Any, bridge: Any | None = None) -> list[Any]:
    """The tools that are not peers: only the e-stop, which stops the fleet like the red button does."""

    @tool
    def emergency_stop() -> dict[str, Any]:
        """Stop every robot on the mesh and latch the lockout: per-peer stop, then the signed fleet e-stop. Never refused."""
        return fleet_stop(safety, bridge, by="agent")

    return [emergency_stop]


#: How long ``spawn_robot`` waits for the new peer's presence on the mesh before
#: reporting it as started-but-not-yet-seen. A MuJoCo so101 announces in 2-5 s
#: on a laptop; the settle window the Devices panel uses is the same order.
SPAWN_PRESENCE_TIMEOUT_S = 20.0
SPAWN_POLL_S = 0.25

#: Names the fixed fleet tools take, so ``expected_tool_names`` and the badge agree.
FLEET_TOOL_NAMES: tuple[str, ...] = ("fleet", "spawn_robot", "despawn_robot")


def peer_summary(peer_id: str, peer: Mapping[str, Any], managed: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """One fleet row for the agent: what the peer is, whether it is fresh, its joints, and the policies it can run."""
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
    from strands_robots.dashboard.peer_policies import peer_policies

    row["policies"] = peer_policies(peer_id, peer, managed)
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
        managed = []
        if devices is not None:
            try:
                managed = list(devices.managed_children())
            except Exception:  # noqa: BLE001 - the roster is a courtesy, the mesh is the truth
                managed = []
        rows = [peer_summary(pid, p, managed) for pid, p in peers.items()]
        rows = [r for r in rows if r["kind"] != KIND_SKIP]
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


def asks_first(bridge: Any | None) -> list[str]:
    """The tools that put a motion to the operator before it runs, read from the mesh.

    The same table the interrupt hook is built from
    (:func:`~strands_robots.dashboard.peer_tools.motion_actions_for`): every real-arm
    proxy, plus a sim proxy whose peer the gate itself calls metal (a wire sim claim
    this dashboard did not launch), so the badge and the gate cannot disagree. Hosts
    offer no motion verbs; stopping is never gated.
    """
    if bridge is None:
        return []
    from strands_robots.dashboard.peer_tools import build_peer_tools, motion_actions_for

    try:
        snap = bridge.snapshot()
    except Exception:  # noqa: BLE001 - a badge must not fail on a bridge hiccup
        logger.debug("asks_first: bridge snapshot unreadable", exc_info=True)
        return []
    peers = snap.get("peers") if isinstance(snap, Mapping) else None
    peers = dict(peers) if isinstance(peers, Mapping) else {}
    proxies = build_peer_tools(peers, lambda *_a, **_k: {"error": "badge only"})
    return sorted(motion_actions_for(proxies, peers))


def expected_tool_names(bridge: Any | None) -> list[str]:
    """The tool names a console over this bridge would carry, without building an agent."""
    from strands_robots.dashboard.peer_tools import expected_tool_names as proxy_names

    names = ["emergency_stop"]
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
    without them the agent holds only ``emergency_stop`` (no mesh, no robots), which is also what
    the tests that install their own factory get.
    """

    def __init__(
        self, safety: Any, model: Any | None = None, bridge: Any | None = None, devices: Any | None = None
    ) -> None:
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

        tools: list[Any] = build_tools(self._safety, self._bridge)
        hooks: list[Any] = []
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
