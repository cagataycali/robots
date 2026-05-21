"""Agent-facing tool for coordinating robots on the Zenoh mesh.

Every :class:`~strands_robots.robot.Robot` and
:class:`~strands_robots.simulation.Simulation` constructed in this process is
already a mesh peer (see :mod:`strands_robots.mesh`); this tool exposes that
mesh to a Strands agent via a single ``robot_mesh`` action dispatcher.

The action vocabulary mirrors the underlying :class:`~strands_robots.mesh.Mesh`
API plus a few discovery helpers:

==================  ===================================================
``peers``           List local + remote peers
``status``          One-line summary of mesh state
``tell``            ``mesh.tell(target, instruction, ...)``
``send``            ``mesh.send(target, json.loads(command), ...)``
``broadcast``       ``mesh.broadcast(json.loads(command), ...)``
``stop``            Send ``{"action": "stop"}`` to a single peer
``emergency_stop``  Broadcast stop to every peer (audited)
``subscribe``       ``mesh.subscribe(target, name=...)`` (buffer mode)
``watch``           ``mesh.on_stream(target)``
``inbox``           Read buffered messages from a subscription
``unsubscribe``     Unsubscribe from a topic by name
==================  ===================================================

The tool always returns a Strands-compatible dict::

    {"status": "success" | "error", "content": [{"text": "..."}]}

It never raises out of the dispatcher: every error path renders a
human-readable text payload so the calling agent can recover.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _resolve_mesh(target: str) -> Any | None:
    """Return a local Mesh in this process to use as the gateway for RPC.

    The agent does not need to know its own peer_id: any local mesh in
    ``_LOCAL_ROBOTS`` is functionally equivalent for outbound calls because
    they all share the same Zenoh session.

    Important: when *target* matches a local peer_id, we deliberately pick a
    *different* local mesh as the gateway. Using the target as its own
    gateway triggers ``_on_cmd``'s self-loop drop (``sender_id == peer_id``)
    and the call silently times out. When the target IS the only local mesh,
    we still return it — the caller will get a timeout, which is the
    expected behaviour for "send to yourself".
    """
    from strands_robots.mesh import get_local_robots

    locals_ = get_local_robots()
    if not locals_:
        return None
    if target:
        # Prefer a local mesh whose peer_id is NOT the target so we don't
        # send-to-self via the target's own session.
        for pid, m in locals_.items():
            if pid != target:
                return m
    # Either no target was specified or every local mesh IS the target —
    # fall back to "any one" (matching the original behaviour for the
    # single-mesh case).
    return next(iter(locals_.values()))


@tool
def robot_mesh(
    action: str,
    target: str = "",
    instruction: str = "",
    command: str = "",
    policy_provider: str = "mock",
    policy_port: int = 0,
    duration: float = 30.0,
    timeout: float = 30.0,
    name: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Coordinate every robot, sim, and agent on the local Zenoh mesh.

    Args:
        action: One of ``peers`` / ``status`` / ``tell`` / ``send`` /
            ``broadcast`` / ``stop`` / ``emergency_stop`` / ``subscribe`` /
            ``unsubscribe`` / ``watch`` / ``inbox``.
        target: Peer id (for ``tell`` / ``send`` / ``stop`` / ``watch``) or
            Zenoh topic pattern (for ``subscribe``).
        instruction: Natural-language instruction for ``tell``.
        command: JSON-encoded command body for ``send`` / ``broadcast``.
        policy_provider: Policy provider tag forwarded with ``tell``.
        policy_port: Optional policy port forwarded with ``tell``.
        duration: Task duration (seconds) forwarded with ``tell``.
        timeout: Response timeout for RPC actions (seconds).
        name: Optional subscription name for ``subscribe`` / ``inbox``.
        limit: Max messages returned by ``inbox`` (default: 50).

    Returns:
        A Strands tool response dict with status and a single text block.

    Examples::

        robot_mesh(action="peers")
        robot_mesh(action="tell", target="so100_sim-a1b2",
                   instruction="pick up the cube")
        robot_mesh(action="send", target="peer-b",
                   command='{"action": "status"}')
        robot_mesh(action="emergency_stop")
    """
    try:
        from strands_robots.mesh import get_local_robots
        from strands_robots.mesh.session import get_peers
    except ImportError as exc:
        return _err(f"mesh module unavailable: {exc}")

    locals_ = get_local_robots()
    peers = get_peers()

    # ── action: peers ─────────────────────────────────────────────────────
    if action == "peers":
        lines = [f"[mesh] {len(locals_)} local, {len(peers)} remote"]
        if locals_:
            lines.append("")
            lines.append("Local (this process):")
            for pid, m in locals_.items():
                lines.append(f"  - {pid} ({m.peer_type})")
        if peers:
            lines.append("")
            lines.append("Discovered peers:")
            for p in peers:
                age = p.get("age", 0)
                ptype = p.get("type", "?")
                host = p.get("hostname", "?")
                lines.append(f"  - {p['peer_id']} ({ptype}) host={host} age={age}s")
                ts = p.get("task_status")
                if ts:
                    lines.append(f"      task: {ts} - {p.get('instruction', '')}")
        elif not locals_:
            lines.append("")
            lines.append("No peers. Create a Robot() or Simulation() to auto-join the mesh.")
        return _ok("\n".join(lines))

    # ── action: status ────────────────────────────────────────────────────
    if action == "status":
        return _ok(f"[mesh] local={len(locals_)} remote={len(peers)} peers={[p['peer_id'] for p in peers]}")

    # All remaining actions need an outbound mesh.
    mesh = _resolve_mesh(target)
    if mesh is None:
        return _err("no local mesh found. Construct a Robot()/Simulation() first to join the mesh, then retry.")

    # ── action: tell ──────────────────────────────────────────────────────
    if action == "tell":
        if not target or not instruction:
            return _err("tell requires both target and instruction")
        kwargs: dict[str, Any] = {
            "policy_provider": policy_provider,
            "duration": duration,
        }
        if policy_port:
            kwargs["policy_port"] = policy_port
        result = mesh.tell(target, instruction, **kwargs)
        return _ok(f"[tell -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: send ──────────────────────────────────────────────────────
    if action == "send":
        if not target:
            return _err("send requires target")
        if not command:
            return _err("send requires command (JSON string)")
        try:
            cmd = json.loads(command)
        except json.JSONDecodeError as exc:
            return _err(f"command is not valid JSON: {exc}")
        result = mesh.send(target, cmd, timeout=timeout)
        return _ok(f"[send -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: broadcast ─────────────────────────────────────────────────
    if action == "broadcast":
        if not command:
            return _err("broadcast requires command (JSON string)")
        try:
            cmd = json.loads(command)
        except json.JSONDecodeError as exc:
            return _err(f"command is not valid JSON: {exc}")
        results = mesh.broadcast(cmd, timeout=timeout)
        text = f"[broadcast] {len(results)} responses\n"
        for r in results[:10]:
            text += f"  - {json.dumps(r, default=str)[:200]}\n"
        if len(results) > 10:
            text += f"  ... and {len(results) - 10} more"
        return _ok(text.rstrip())

    # ── action: stop ──────────────────────────────────────────────────────
    if action == "stop":
        if not target:
            return _err("stop requires target")
        result = mesh.send(target, {"action": "stop"}, timeout=min(timeout, 5.0))
        return _ok(f"[stop -> {target}] {json.dumps(result, default=str)[:600]}")

    # ── action: emergency_stop ────────────────────────────────────────────
    if action == "emergency_stop":
        results = mesh.emergency_stop()
        return _ok(f"[E-STOP] broadcast complete - {len(results)} responses (audit log written)")

    # ── action: subscribe ─────────────────────────────────────────────────
    if action == "subscribe":
        if not target:
            return _err("subscribe requires target (Zenoh topic pattern)")
        sub_name = name or target
        out = mesh.subscribe(target, name=sub_name)
        if out is None:
            return _err("subscribe failed (mesh not running?)")
        return _ok(
            f"[sub] subscribed to '{target}' as '{sub_name}'. "
            f"Use action='inbox' name='{sub_name}' to read buffered messages."
        )

    # ── action: watch ─────────────────────────────────────────────────────
    if action == "watch":
        if not target:
            return _err("watch requires target (peer id)")
        out = mesh.on_stream(target)
        if out is None:
            return _err("watch failed (mesh not running?)")
        return _ok(f"[watch] watching peer '{target}'. Use action='inbox' name='{out}' to read buffered steps.")

    # ── action: inbox ─────────────────────────────────────────────────────
    if action == "inbox":
        sub_name = name or target
        if not sub_name:
            return _err("inbox requires name (or target)")
        msgs = mesh.inbox.get(sub_name, [])
        if not msgs:
            return _ok(f"[inbox '{sub_name}'] no messages")
        head = msgs[-limit:] if limit > 0 else msgs
        text = f"[inbox '{sub_name}'] {len(msgs)} total, showing last {len(head)}\n"
        for topic, data in head:
            text += f"  - {topic}: {json.dumps(data, default=str)[:200]}\n"
        return _ok(text.rstrip())

    # ── action: unsubscribe ────────────────────────────────────────────────
    if action == "unsubscribe":
        sub_name = name or target
        if not sub_name:
            return _err("unsubscribe requires name (or target)")
        mesh.unsubscribe(sub_name)
        return _ok(f"[unsub] unsubscribed from '{sub_name}'")

    return _err(
        f"unknown action: {action!r}. Valid: peers, status, tell, send, "
        "broadcast, stop, emergency_stop, subscribe, unsubscribe, watch, inbox."
    )


__all__ = ["robot_mesh"]
