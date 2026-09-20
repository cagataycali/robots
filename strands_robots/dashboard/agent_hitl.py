"""Human-in-the-loop gate for agent tool calls that would move real hardware.

The hook pauses the agent (SDK interrupt) instead of refusing, so a human
yes resumes the SAME turn and the tool executes. Stopping is never gated.

A yes is recorded in :mod:`strands_robots._motion_grants`, not here: the
surfaces that spend it sit below this package (the ``Robot`` agent tool,
``pose_tool``, ``serial_tool``), so the store and the identity it keys on belong
under all of them rather than inside the optional web extra.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

from strands_robots._motion_grants import DIRECT_SERIAL_TOOLS, deposit_grant, motion_fields, resolve_target
from strands_robots.dashboard.agent_motion import MOTION_ENV, peer_is_physical

logger = logging.getLogger(__name__)

INTERRUPT_NAME = "physical_motion"

#: tool name -> actions that can put a real robot in motion. Absent tool = never gated.
#: stop / emergency_stop / status / peers are deliberately NOT here: stopping is never gated.
MOTION_ACTIONS: dict[str, frozenset[str]] = {
    "fleet": frozenset({"task"}),
    # robot_mesh is deliberately ABSENT: it raises its own SDK-native interrupt
    # (tool_context.interrupt in ``strands_robots.tools.robot_mesh``) on every
    # physical action, so listing it here would ask the operator twice for one
    # command. So is the Robot agent tool (``strands_robots.hardware_robot``):
    # its real-mode execute/start run through the shared command gate and spend
    # a grant this hook deposited (consume_grant) rather than asking again.
    # This dict gates only the dashboard's bespoke tools.
    # The direct-serial tools live in ``strands_robots.tools.serial_tool`` and
    # ``strands_robots.tools.pose_tool``. Both now gate their own write / motion
    # actions through the shared command gate (serial_tool's four writes, and
    # pose_tool's five motions), so for an agent built without this hook neither
    # is unguarded; they stay listed here because this hook shows the operator
    # the dashboard's richer detail line and deposits a grant each tool spends
    # (consume_grant) instead of asking a second time. Reads, emergency_stop and
    # delete_pose stay out: stopping is never gated. serial "monitor" only ever
    # calls ser.read (``strands_robots.tools.serial_tool``) so it is a read too.
    "pose_tool": frozenset({"load_pose", "move_motor", "move_multiple", "incremental_move", "reset_to_home"}),
    "serial_tool": frozenset({"send", "send_read", "feetech_position", "feetech_velocity"}),
}


def _direct_serial_detail(action: str, tool_input: Mapping[str, Any]) -> str:
    """The gated call's own motion fields as one readable line -- never invented."""
    fields = motion_fields(tool_input)
    return " ".join((action, *fields)) if fields else ""


_TRUE = ("1", "true", "yes", "on")


def _granted(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(MOTION_ENV, "")).strip().lower() in _TRUE


def motion_intent(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    peers: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
    *,
    extra_actions: Mapping[str, frozenset[str]] | None = None,
    bound_targets: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Would this tool call start physical motion needing a human yes?

    Returns the structured interrupt reason, or None when the call may proceed
    (not a motion action, target is a sim, or the always-allow grant is set).
    """
    actions = MOTION_ACTIONS.get(tool_name)
    if actions is None and extra_actions is not None:
        # per-peer proxy tools: their gate rows are DERIVED per agent build
        # (peer_tools.motion_actions_for), never hand-kept here.
        actions = extra_actions.get(tool_name)
    if actions is None:
        return None
    tool_input = tool_input or {}
    action = str(tool_input.get("action") or "").strip()
    if action not in actions:
        return None
    if _granted(env):
        return None  # the always-allow fast lane: no interrupt is raised

    target = resolve_target(tool_name, tool_input, bound_targets)
    peer = (peers or {}).get(target)
    physical, why = peer_is_physical(peer)
    if not physical:
        return None

    instruction = str(tool_input.get("instruction") or tool_input.get("message") or "")
    if not instruction and tool_name in DIRECT_SERIAL_TOOLS:
        # pose/serial inputs carry the motion in named fields, not an
        # instruction string; show the operator WHAT a yes moves, verbatim.
        instruction = _direct_serial_detail(action, tool_input)
    reason: dict[str, Any] = {
        "tool": tool_name,
        "action": action,
        "target": target or "(unnamed peer)",
        "instruction": instruction,
        "why_physical": why,
    }
    if tool_input.get("duration") is not None:
        try:
            reason["duration"] = float(tool_input["duration"])
        except (TypeError, ValueError):
            # A duration the model wrote as prose ("30s") or as a structure is
            # dropped rather than raised on. This field is one extra line the
            # operator reads, not part of deciding WHETHER to ask: the gate has
            # already resolved the action as motion and the target as metal by
            # this point, so the interrupt fires with or without it and the tool
            # still cannot run without a yes. Raising instead would take an
            # unparseable optional field and abort the operator's turn with an
            # exception out of a BeforeToolCallEvent hook, so the human is never
            # asked the question this gate exists to ask them. The tuple is the
            # exact pair float() raises: TypeError for a non-numeric type,
            # ValueError for a string that does not parse.
            pass
    return reason


def response_approves(response: Any) -> bool:
    """Interpret a human interrupt response. Anything but an explicit yes is a no."""
    if isinstance(response, bool):
        return response
    if isinstance(response, Mapping):
        return response_approves(response.get("approve"))
    if isinstance(response, str):
        return response.strip().lower() in _TRUE + ("y", "approve", "approved")
    return False


def cancel_sentence(reason: Mapping[str, Any]) -> str:
    """What the model relays when the human says no."""
    return (
        f"the human declined: {reason.get('target')} was NOT sent "
        f"{reason.get('instruction') or 'that instruction'!r}. Nothing moved. "
        f"Ask them what they would like instead."
    )


class MotionInterruptHook(HookProvider):
    """Pause before any tool call that would start physical motion.

    ``peers_snapshot`` is a callable returning the current fleet peers dict, so
    the physicality verdict is read at call time, never cached.
    """

    def __init__(
        self,
        peers_snapshot: Callable[[], Mapping[str, Any]],
        proxy_motion: Mapping[str, frozenset[str]] | None = None,
        proxy_targets: Mapping[str, str] | None = None,
    ) -> None:
        self._peers_snapshot = peers_snapshot
        self._proxy_motion = dict(proxy_motion or {})
        self._proxy_targets = dict(proxy_targets or {})

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe the motion gate to every tool call the agent is about to make."""
        registry.add_callback(BeforeToolCallEvent, self._gate)

    def _gate(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = str(tool_use.get("name") or "")
        tool_input = tool_use.get("input") or {}
        try:
            peers = self._peers_snapshot() or {}
        except Exception:  # noqa: BLE001 - unreadable snapshot means UNKNOWN, i.e. metal
            peers = {}
        reason = motion_intent(
            name,
            tool_input,
            peers,
            extra_actions=self._proxy_motion,
            bound_targets=self._proxy_targets,
        )
        if reason is None:
            return
        # Raises InterruptException on first pass; returns the human response on resume.
        response = event.interrupt(INTERRUPT_NAME, reason=reason)
        approved = response_approves(response)
        # Record the operator's reply in the local audit log for both outcomes.
        # The reply itself never reaches the model (cancel_sentence returns a
        # flat sentinel); the audit row is the only place it survives.
        from strands_robots._hitl_audit import log_operator_response

        log_operator_response(
            "dashboard_agent_hitl",
            str(reason.get("action", "")),
            str(reason.get("target", "")),
            approved=approved,
            response=response,
        )
        if approved:
            deposit_grant(name, tool_input)
            return
        event.cancel_tool = cancel_sentence(reason)
