"""Per-peer proxy AgentTools: every fleet robot becomes a NATIVE tool on the dashboard agent.

Users of strands_robots write ``Agent(tools=[Robot('so101')])`` and the robot IS a tool.
The dashboard cannot do that literally: its robots are CHILD PROCESSES holding the
serial buses / sim state, and a second in-process ``Robot('so101')`` would collide on
the bus. So "native" here means a PROXY that is indistinguishable to the agent: for
each fleet peer we build an AgentTool named for it whose tool_spec mirrors what that
peer really is - ``hardware_robot.Robot``'s execute/start/status/stop spec for a real
arm, the wire's sim command family (state/set_joints/reset/step/stop plus
execute/start rollouts) for a sim - and whose invocation routes over the mesh
command rail that already exists (``mesh.security.validate_command``), via the
dashboard bridge's ``send_cmd``.

Gating stays ONE layer: these proxies do NOT gate themselves. ``MotionInterruptHook``
(agent_hitl) gates them by tool name + action, peer-aware through ``peer_is_physical``
 -  which is why :func:`map_invocation` guarantees a ``target`` field is always present
in the reason the hook derives (the proxy binds it). sim actions ask nothing (the peer
is provably a sim), stop/status are never gated (not in MOTION_ACTIONS).

Everything above the wire is a PURE rule in this module, tested without a mesh.
"""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, cast

from strands_robots.dashboard.agent_motion import hardware_evidence, peer_is_physical

# ── classification ──────────────────────────────────────────────────────────

#: Robot kinds a proxy can represent. ``skip`` = build no tool for this peer.
KIND_REAL = "real"
KIND_SIM = "sim"
KIND_HOST = "host"  # a robot process with no joints announced (yet): status/stop only
KIND_SKIP = "skip"

_SIM_TYPES = ("sim", "simulation", "mujoco")

#: Peer types that coordinate rather than move: no tool at all. Read off
#: ``robot_type`` because that is the field the WIRE carries -
#: ``strands_robots.mesh.core`` builds presence as ``{"robot_id", "robot_type": peer_type,
#: "hostname", "timestamp", ...}`` (``strands_robots.mesh.core``, the presence builder) and
#: ``robot_mesh._gateway_mesh()`` joins with ``peer_type="gateway"``, as does
#: ``mesh_bridge``'s safety peer. Nothing publishes a ``kind`` field: reading
#: ``presence["kind"]`` skipped nothing at all, so throwaway ``gateway-*``
#: sessions became AgentTools described to the model as "Robot peer" and
#: ``fleet_signature`` churned the agent on every probe's birth and death
#: (measured once: 4 of 8 live tools were coordinator debris).
_SKIP_TYPES = ("gateway", "dashboard")

#: Belt #1: a coordinator's peer id. ``_gateway_mesh`` names itself
#: ``gateway-<host>-<hex>``; the dashboard's own peer/safety session likewise.
_SKIP_ID_PREFIXES = ("gateway-", "dashboard-")


def _is_coordinator(peer_id: str, peer: Mapping[str, Any], presence: Mapping[str, Any]) -> bool:
    """Is this peer a mesh coordinator (gateway/dashboard) rather than a robot?

    Three independent reads, so a presence payload that drops or renames its
    type field still cannot mint a motion tool for something with no hardware:
    the published ``robot_type``, the peer id's own prefix, and the topic
    advertisement (a robot-less ``Mesh`` announces ``topics == ["health"]``
    only - ``strands_robots.mesh.core`` appends "health" unconditionally and every other topic
    needs a hardware attribute).
    """
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    kind = str(presence.get("kind") or peer.get("kind") or "").strip().lower()
    if robot_type in _SKIP_TYPES or kind in _SKIP_TYPES:
        return True
    if robot_type or kind:
        return False  # it named itself something else: believe it
    pid = peer_id.strip().lower()
    if pid.startswith(_SKIP_ID_PREFIXES):
        return True
    topics = presence.get("topics")
    if isinstance(topics, (list, tuple)) and [str(t).strip().lower() for t in topics] == ["health"]:
        state = peer.get("state") or {}
        if not (state.get("joints") or presence.get("joints") or peer.get("cameras")):
            return True
    return False


def classify_peer(peer_id: str, peer: Mapping[str, Any] | None) -> str:
    """What kind of tool should represent this peer?

    Reads presence in the order ``agent_motion.peer_is_physical`` reads it:
    hardware evidence first (``agent_motion.hardware_evidence``), so a record
    that names hardware is a real arm whatever its ``robot_type`` says; then the
    sim claims. Only the default posture differs: the GATE fails closed (unknown
    = metal), a TOOL FACTORY fails quiet (unknown/gateway/dashboard = no tool at
    all) - a tool for a peer we cannot describe would advertise a spec we
    invented.
    """
    if not peer_id or peer is None:
        return KIND_SKIP
    presence = peer.get("presence") or {}
    kind = str(presence.get("kind") or peer.get("kind") or "").strip().lower()
    if _is_coordinator(peer_id, peer, presence):
        return KIND_SKIP
    if hardware_evidence(presence) is not None:
        return KIND_REAL
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    if robot_type in _SIM_TYPES or presence.get("sim") is True or presence.get("mode") == "sim":
        return KIND_SIM
    # A child peer of a sim world (``<parent>__<robot>``) is itself a sim
    # robot even when its own presence is sparse: core delegates its commands
    # to the parent Simulation.
    if "__" in peer_id and str(peer.get("parent") or presence.get("parent") or "").strip():
        return KIND_SIM
    state = peer.get("state") or {}
    joints = state.get("joints") or presence.get("joints") or {}
    n_joints = len(joints) if isinstance(joints, Mapping) else int(joints or 0)
    if n_joints > 0 or peer.get("role"):
        return KIND_REAL
    if kind == "robot" or presence:
        return KIND_HOST
    return KIND_SKIP


# ── naming ───────────────────────────────────────────────────────────────────

_NAME_OK = re.compile(r"[^A-Za-z0-9_]")


def sanitize_tool_name(peer_id: str, taken: frozenset[str] | set[str] = frozenset()) -> str:
    """Peer id -> identifier-safe, unique tool name.

    Peer ids carry dashes (``so101-real-689``); tool names must be
    identifier-safe (``so101_real_689``). Collisions (two peers sanitizing to
    one name) get a numeric suffix - deterministic in iteration order.
    """
    name = _NAME_OK.sub("_", peer_id.strip()) or "peer"
    if name[0].isdigit():
        name = f"p_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    base, n = name, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    return name


# ── tool specs ───────────────────────────────────────────────────────────────

#: What a sim peer accepts on the wire (``strands_robots.mesh.security.ALLOWED_ACTIONS``
#: as ``mesh.core._dispatch`` serves them for a Simulation or its child SimRobot).
SIM_ACTIONS: tuple[str, ...] = ("status", "state", "set_joints", "reset", "step", "stop", "execute", "start")

#: The policy keys both proxy schemas offer on execute/start (forwarded as ``_POLICY_FIELDS``).
_POLICY_PROPERTIES: dict[str, Any] = {
    "model_path": {
        "type": "string",
        "description": "execute/start: a checkpoint directory on the robot host (wbc, wbc_gait, rl)",
    },
    "walk": {
        "type": "boolean",
        "description": "execute/start with wbc: true loads the walk policy too (default), false balances only",
    },
    "target_velocity": {
        "type": "array",
        "items": {"type": "number"},
        "description": (
            "execute/start with wbc or wbc_gait: [vx, vy, wz] in m/s, m/s, rad/s; refused outside the "
            "locomotion envelope (2 m/s, 2 rad/s by default)"
        ),
    },
    "pretrained_name_or_path": {
        "type": "string",
        "description": "execute/start with lerobot_local: the Hub checkpoint, e.g. lerobot/smolvla_base",
    },
    "policy_type": {"type": "string", "description": "execute/start with lerobot_local: act, smolvla, pi0, ..."},
}

_SIM_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "status | state | set_joints | reset | step | stop | execute | start",
            "enum": list(SIM_ACTIONS),
            "default": "state",
        },
        "target_joints": {
            "type": "object",
            "description": "set_joints: joint name (or 1-based index as a string) -> radians",
            "additionalProperties": {"type": "number"},
        },
        "hold": {
            "type": "boolean",
            "description": "set_joints: re-seed the servos so the pose survives the next step (default true)",
        },
        "steps": {"type": "integer", "description": "step: how many physics steps (default 1)"},
        "instruction": {"type": "string", "description": "execute/start: natural language task"},
        "policy_provider": {
            "type": "string",
            "description": (
                "execute/start: which policy backend. Pick from this peer's `policies.can_run` in the fleet "
                "listing, e.g. wbc, lerobot_local. A Unitree G1 walks with wbc; an arm runs lerobot_local "
                "with a checkpoint; mock is a sine test, not a task (default mock)"
            ),
        },
        **_POLICY_PROPERTIES,
        "duration": {"type": "number", "description": "execute/start: seconds (positive, finite)"},
        "robot_name": {"type": "string", "description": "a Simulation holding several robots: which one"},
    },
    "required": ["action"],
}


def peer_tool_spec(peer_id: str, kind: str, tool_name: str) -> dict[str, Any] | None:
    """The ToolSpec a proxy presents for this peer - mirrors what the peer IS."""
    if kind == KIND_SIM:
        return {
            "name": tool_name,
            "description": (
                f"Simulation peer '{peer_id}' as a native tool (routed over the mesh to the MuJoCo "
                f"process that owns it). Actions: status, state (joint names and positions), "
                f"set_joints (write target_joints in radians, held by the servos), reset, step, stop, "
                f"and execute/start (a policy rollout: instruction + policy_provider, e.g. mock or "
                f"lerobot_local with pretrained_name_or_path). Never real hardware, so nothing here "
                f"asks the operator first."
            ),
            "inputSchema": {"json": _SIM_INPUT_SCHEMA},
        }
    if kind == KIND_REAL:
        return {
            "name": tool_name,
            "description": (
                f"Real robot peer '{peer_id}' as a native tool (routed over the mesh; the "
                f"robot process holds the hardware). Actions: execute (blocking policy "
                f"rollout), start (async), status, stop. execute/start move REAL metal and "
                f"raise a human confirmation; status/stop are never gated."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "execute (blocking), start (async), status, stop",
                            "enum": ["execute", "start", "status", "stop"],
                            "default": "status",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute/start)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": (
                                "Which policy backend the peer runs: one of cosmos3, curobo, flux3_action, kimodo, "
                                "lerobot_local, microduck, mock, moveit2, protomotions, remote, rl, wbc, wbc_gait. "
                                "lerobot_local (default) runs a local checkpoint on the peer and needs "
                                "pretrained_name_or_path; moveit2 (needs policy_port) and remote dial a server. "
                                "An unknown name is refused by the peer listing its registry."
                            ),
                            "default": "lerobot_local",
                        },
                        **_POLICY_PROPERTIES,
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds (positive, finite)",
                        },
                    },
                    "required": ["action"],
                }
            },
        }
    if kind == KIND_HOST:
        return {
            "name": tool_name,
            "description": (
                f"Robot peer '{peer_id}' (no joints announced yet) as a native tool. "
                f"Only status and stop are offered until it says what it is."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "stop"],
                            "default": "status",
                        }
                    },
                    "required": ["action"],
                }
            },
        }
    return None


# ── invocation -> mesh command (pure) ────────────────────────────────────────

#: The policy keys both rails forward on execute/start: the constructor keys the
#: wire carries (``pretrained_name_or_path``, ``policy_type``, ``model_path``,
#: ``walk``) and the per-call goal ``target_velocity``. Each has a validator in
#: ``mesh/security.validate_command`` on the robot host; nothing else crosses.
_POLICY_FIELDS: tuple[str, ...] = ("pretrained_name_or_path", "policy_type", "model_path", "walk", "target_velocity")

#: Fields the real-robot rail forwards. Everything else is refused by
#: mesh/security.validate_command anyway; dropping them here makes the
#: refusal happen with a better sentence and no wire round trip.
_REAL_FIELDS: dict[str, tuple[str, ...]] = {
    "execute": ("instruction", "policy_port", "policy_host", "policy_provider", "duration", *_POLICY_FIELDS),
    "start": ("instruction", "policy_port", "policy_host", "policy_provider", "duration", *_POLICY_FIELDS),
    "status": (),
    "stop": (),
}


#: Fields the sim rail forwards per action; everything else is dropped before the wire.
_SIM_FIELDS: dict[str, tuple[str, ...]] = {
    "status": (),
    "state": ("robot_name",),
    "set_joints": ("target_joints", "hold", "robot_name"),
    "reset": ("robot_name",),
    "step": ("steps",),
    "stop": (),
    "execute": ("instruction", "policy_provider", "duration", "robot_name", *_POLICY_FIELDS),
    "start": ("instruction", "policy_provider", "duration", "robot_name", *_POLICY_FIELDS),
}


def map_invocation(
    peer_id: str, kind: str, tool_input: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Proxy tool input -> the validated mesh command to send this peer.

    Returns ``(command, error)`` - exactly one is non-None. The command is a
    dict for ``bridge.send_cmd(peer_id, command)``; its shape is what
    ``mesh/security.validate_command`` accepts (execute/start/status/stop for
    robots, the sim command family for sims).
    """
    tool_input = dict(tool_input or {})
    action = str(tool_input.pop("action", "") or "").strip()
    if not action:
        return None, "input needs an 'action'"

    if kind == KIND_SIM:
        if action not in SIM_ACTIONS:
            return None, f"unknown action {action!r} for this sim. Valid: {', '.join(SIM_ACTIONS)}"
        cmd: dict[str, Any] = {"action": action}
        for field in _SIM_FIELDS.get(action, ()):
            if tool_input.get(field) is not None:
                cmd[field] = tool_input[field]
        if action == "set_joints" and not isinstance(cmd.get("target_joints"), Mapping):
            return None, "set_joints needs target_joints: {joint name -> radians}"
        if action in ("execute", "start"):
            cmd.setdefault("policy_provider", "mock")
            if not str(cmd.get("instruction") or "").strip():
                return None, f"{action} needs an instruction"
        return cmd, None

    if kind in (KIND_REAL, KIND_HOST):
        allowed = _REAL_FIELDS if kind == KIND_REAL else {"status": (), "stop": ()}
        if action not in allowed:
            return None, f"unknown action {action!r} for this robot. Valid: {', '.join(sorted(allowed))}"
        cmd = {"action": action}
        for field in allowed[action]:
            if tool_input.get(field) is not None:
                cmd[field] = tool_input[field]
        return cmd, None

    return None, f"peer kind {kind!r} carries no tool"


# ── the AgentTool proxy ──────────────────────────────────────────────────────


def _agent_tool_base() -> type:
    from strands.types.tools import AgentTool  # local import: keep this module importable without strands

    return cast("type", AgentTool)


#: Verbs that are NEVER refused by the staleness gate. House law: a stale
#: presence read makes stopping MORE urgent, not less (one mesh-ingest blackout marked both REAL
#: arms stale 1430s while they were streaming - a refusal there lands on a
#: possibly MOVING arm). stop-class commands and status reads are attempted and
#: their real outcome reported, with a staleness NOTE attached; only actions
#: that START motion (execute/start) stay refused on stale presence.
NEVER_GATED: frozenset[str] = frozenset({"stop", "emergency_stop", "stop_all", "status"})


def stale_note(peer_id: str, peer: Mapping[str, Any] | None) -> str | None:
    """The staleness caveat attached to a stop/status that was SENT anyway.

    Returns None when presence is fresh. Never a refusal - the command has
    already been (or will be) delivered; this only tells the model the ack may
    not arrive and why, so a timeout reads as a presence gap, not a robot fault.
    """
    if not peer or not peer.get("stale"):
        return None
    age = peer.get("last_seen_age")
    if age is None:
        age = peer.get("age")
    when = f" (no presence for {float(age):.0f}s)" if isinstance(age, (int, float)) else ""
    return (
        f"NOTE: peer '{peer_id}' was STALE on the mesh when this command was sent{when}. "
        "The command was delivered anyway - stop-class and status commands are never refused "
        "on staleness - but the acknowledgement may be missing or delayed. Treat a timeout as "
        "a presence gap, not a robot fault."
    )


def stale_refusal(peer_id: str, peer: Mapping[str, Any] | None) -> str | None:
    """Why a command to this peer must not be sent, or None to proceed.

    DECISION, and it is deliberately NOT the ``fleet`` tool's one. ``fleet`` filtered stale peers
    out of its listing (agent_bridge.py:300, :361), so the honest move here looked like "drop the
    proxy". It is the wrong one: staleness is a fact about PRESENCE DELIVERY, not about the robot
    (a 26-minute mesh-ingest blackout marked both real arms stale for 1430s while they were
    connected and streaming the whole time, and it self-healed with no restart). Dropping tools on
    that would delete the agent's entire arm surface mid-blackout and rebuild it minutes later.

    So the proxy STAYS and the invocation refuses - for MOTION-STARTING actions only (
    stop-class verbs and status reads in ``NEVER_GATED`` are always attempted, with a
    ``stale_note`` attached, because refusing a stop on a stale-but-possibly-moving arm inverts
    the safety direction). A stale peer answers nothing,
    so ``send_cmd`` can only burn its 30s timeout and hand the model a bare timeout, which reads as
    a robot fault. The refusal names presence as the suspect instead.

    Checked at INVOCATION time, never baked into the tool list - a build-time flag would be a claim
    about a moment that has passed (which is also why ``fleet_signature`` still ignores stale).
    """
    if not peer or not peer.get("stale"):
        return None
    age = peer.get("last_seen_age")
    if age is None:
        age = peer.get("age")
    when = f" (no presence for {float(age):.0f}s)" if isinstance(age, (int, float)) else ""
    return (
        f"peer '{peer_id}' is STALE on the mesh{when}: the dashboard has not heard from it, so a "
        "command would only wait out its timeout. This is a PRESENCE fact, not a robot fault - the "
        "device may be fine while mesh delivery is stalled. Check the fleet screen (or /api/health's "
        "forwarded counter) and retry once it reports fresh; do not report the robot as broken."
    )


def build_peer_tools(
    peers: Mapping[str, Mapping[str, Any]],
    send_cmd: Callable[..., dict[str, Any]],
    peer_state: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> list[Any]:
    """One proxy AgentTool per tool-worthy fleet peer, names collision-free.

    ``send_cmd(peer_id, command, timeout=..., source="agent")`` is the
    dashboard bridge's sender - injected so the factory stays pure and the
    proxies stay testable with a fake.

    ``peer_state(peer_id)`` is the LIVE presence reader: the proxy asks it
    on every invocation so a peer that went stale after the tool was built
    refuses with a presence sentence instead of a 30s timeout. Omit it and the
    proxies keep their ungated behaviour - no gate, every call goes to the wire.
    """
    AgentTool = _agent_tool_base()

    class PeerProxyTool(AgentTool):  # type: ignore[misc,valid-type]
        """A fleet peer, presented to the agent as the robot itself."""

        def __init__(self, peer_id: str, kind: str, spec: dict[str, Any]) -> None:
            super().__init__()
            self._peer_id = peer_id
            self._kind = kind
            self._spec = spec

        @property
        def tool_name(self) -> str:
            return cast(str, self._spec["name"])

        @property
        def tool_spec(self) -> dict[str, Any]:
            return self._spec

        @property
        def tool_type(self) -> str:
            return "robot"

        @property
        def peer_id(self) -> str:
            """The fleet peer this proxy is bound to - the motion gate's target."""
            return self._peer_id

        @property
        def peer_kind(self) -> str:
            return self._kind

        async def stream(
            self, tool_use: Mapping[str, Any], invocation_state: dict[str, Any], **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            from strands.types._events import ToolResultEvent

            tool_use_id = tool_use.get("toolUseId", "")
            cmd, err = map_invocation(self._peer_id, self._kind, tool_use.get("input") or {})
            if err is not None:
                yield ToolResultEvent({"toolUseId": tool_use_id, "status": "error", "content": [{"text": err}]})
                return
            staleness_note: str | None = None
            if peer_state is not None:
                try:
                    live = peer_state(self._peer_id)
                except Exception:  # noqa: BLE001 - an unreadable snapshot must not block a command
                    live = None
                # stop-class verbs and status reads are NEVER refused on
                # staleness - the requested action decides.
                requested = str((tool_use.get("input") or {}).get("action") or "").strip()
                if requested in NEVER_GATED:
                    staleness_note = stale_note(self._peer_id, live)
                else:
                    refusal = stale_refusal(self._peer_id, live)
                    if refusal is not None:
                        yield ToolResultEvent(
                            {"toolUseId": tool_use_id, "status": "error", "content": [{"text": refusal}]}
                        )
                        return
            try:
                res = send_cmd(self._peer_id, cmd, timeout=30.0, source="agent")
            except Exception as exc:  # noqa: BLE001 - the wire's failure IS the result
                fail = f"mesh send to '{self._peer_id}' failed: {exc}"
                if staleness_note:
                    fail = f"{fail}\n{staleness_note}"
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": fail}],
                    }
                )
                return
            res = res if isinstance(res, dict) else {"result": res}
            status = str(res.get("status") or ("error" if res.get("error") else "success"))
            content = res.get("content")
            if not isinstance(content, list):
                content = [{"text": json.dumps(res, default=str)[:8000]}]
            if staleness_note:
                content = list(content) + [{"text": staleness_note}]
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use_id,
                    "status": "success" if status == "success" else "error",
                    "content": content,
                }
            )

    tools: list[Any] = []
    taken: set[str] = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind == KIND_SKIP:
            continue
        name = sanitize_tool_name(peer_id, taken)
        spec = peer_tool_spec(peer_id, kind, name)
        if spec is None:
            continue
        taken.add(name)
        tools.append(PeerProxyTool(peer_id, kind, spec))
    return tools


def expected_tool_names(peers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The proxy tool names this fleet would produce - pure, no strands import.

    Lets agent_status answer honestly BEFORE the agent is lazily built
    (the badge used to hardcode ['fleet'], which lied until the first turn).
    """
    names: list[str] = []
    taken: set[str] = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind == KIND_SKIP:
            continue
        name = sanitize_tool_name(peer_id, taken)
        taken.add(name)
        names.append(name)
    return names


def fleet_signature(peers: Mapping[str, Mapping[str, Any]]) -> frozenset[tuple[str, str]]:
    """What the proxy surface depends on: the set of (peer_id, kind).

    get_agent compares this at call time against the signature the agent was
    built with - a changed fleet (join/leave/reclassify) rebuilds the agent so
    the tool list follows the mesh. Presence details beyond kind do not
    matter to the tools, so they do not churn the agent.
    """
    out = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind != KIND_SKIP:
            out.add((peer_id, kind))
    return frozenset(out)


def motion_actions_for(tools: list[Any], peers: Mapping[str, Mapping[str, Any] | None]) -> dict[str, frozenset[str]]:
    """The MOTION_ACTIONS entries these proxies need - derived, never hand-kept.

    Every REAL-arm proxy appears with its motion verbs, and so does a sim proxy
    whose peer the gate itself calls metal (``agent_motion.peer_is_physical``:
    a wire ``robot_type: "sim"`` claim this dashboard did not launch cannot be
    checked, so it is metal until a peer it did launch says otherwise). The
    interrupt hook consults ``peer_is_physical`` only for tools in this table,
    so a proxy left out is a rollout nobody is asked about (f030). Host
    proxies offer no motion verbs, and stop/status are never gated. Deriving
    the table from the built tools means the gate and the tool surface cannot
    drift apart.

    ``execute`` and ``start`` are the two verbs a hardware peer runs; it refuses
    ``set_joints``, ``step`` and ``reset`` by name, so those need no row.
    """
    motion = frozenset({"execute", "start"})
    table: dict[str, frozenset[str]] = {}
    for t in tools:
        kind = getattr(t, "peer_kind", None)
        if kind == KIND_REAL:
            table[t.tool_name] = motion
        elif kind == KIND_SIM:
            physical, _ = peer_is_physical(peers.get(getattr(t, "peer_id", "")))
            if physical:
                table[t.tool_name] = motion
    return table
