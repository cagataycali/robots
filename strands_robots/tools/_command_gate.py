"""Shared operator-approval gate for the tools that command a ROS 2 graph.

:mod:`~strands_robots.tools.use_ros`, :mod:`~strands_robots.tools.use_rtps` and
:mod:`~strands_robots.tools.use_rosbridge` reach a ROS 2 graph over three
different transports - in-process rclpy, raw RTPS, and a rosbridge WebSocket -
and every one of them can carry a command to a physical robot. A ``Twist`` on
``/cmd_vel`` moves the same base whichever of the three wrote it, so this module
is the single owner of the approval decision: it cannot differ between two
transports onto the same graph, where the same publish would otherwise be
refused by one and silently sent by the other.

That is not hypothetical. The gate shipped in :mod:`~strands_robots.tools.use_ros`
alone, so an agent declined at ``/cmd_vel`` could re-issue the identical command
through either sibling and it went out with no prompt, no allowlist check and no
audit row - the tool name was the whole difference. A blocklist is a statement
about a physical surface, not about a wire format, so it belongs beside the
mechanism that enforces it rather than in one of the three callers.

Which actions carry a command is a property of the transport and stays at each
call site: ``use_ros`` gates a topic ``publish``, a ``service_call`` and an
``action_send_goal``; ``use_rtps`` speaks no service or action protocol, so only
its ``publish`` can command; ``use_rosbridge`` gates ``publish`` and
``service_call``. Reading a surface is never gated on any transport, and neither
is ``use_rtps``'s ``advertise``, which creates a publisher without writing a
sample. The same reasoning keeps the per-action numeric-option tables beside
their dispatch in :mod:`~strands_robots.tools._numeric_options`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from strands.types.tools import ToolContext

from strands_robots.tools._hitl_audit import log_operator_response

logger = logging.getLogger(__name__)

# Safety-critical command surfaces, gated on every transport. Matching is on the
# final path segment (see :func:`match_blocklist`), so a bare entry covers every
# namespaced instance: a ``/cmd_vel`` entry blocks ``/robot1/cmd_vel`` too. The
# set is consulted from every verb that can carry a command rather than from
# publish alone - /navigate_to_pose and /follow_path are ROS 2 actions, and the
# e-stop / motor-enable surfaces are usually services, so a publish-only gate
# leaves its most dangerous entries unenforceable. /manual_drive covers the
# DeepRacer's /webserver_pkg/manual_drive, and /vehicle_state and /enable_state
# cover the /ctrl_pkg/... services that arm it; the arming pair belongs here for
# the same reason /motor_enable does - it is what makes the vehicle act on a
# command at all. An operator can pre-approve individual surfaces via
# STRANDS_ROS2_COMMAND_ALLOW (comma-separated) or bypass the gate entirely with
# BYPASS_TOOL_CONSENT=true.
COMMAND_BLOCKLIST = frozenset(
    {
        "/cmd_vel",
        "/cmd_vel_unstamped",
        "/manual_drive",
        "/joint_command",
        "/joint_trajectory",
        "/joint_trajectory_controller/joint_trajectory",
        "/emergency_stop",
        "/e_stop",
        "/motor_enable",
        "/enable_motor",
        "/disable_motor",
        "/vehicle_state",
        "/enable_state",
        "/navigate_to_pose",
        "/follow_path",
    }
)

COMMAND_ALLOW_ENV = "STRANDS_ROS2_COMMAND_ALLOW"
BYPASS_CONSENT_ENV = "BYPASS_TOOL_CONSENT"

_APPROVE_RESPONSES = frozenset({"y", "yes", "approve", "approved"})


def approve_response(response: object) -> bool:
    """Accept affirmative operator responses from the HIL interrupt."""
    return isinstance(response, str) and response.strip().lower() in _APPROVE_RESPONSES


def canonical_command_name(name: str) -> str:
    """Reduce a graph name to the form a ROS 2 client resolves it to, for comparison only.

    ``cmd_vel`` (relative) and ``/cmd_vel/`` (trailing separator) name the same
    surface as ``/cmd_vel`` once resolved, so a literal membership test on the
    caller's spelling misses both. The canonical form is compared against the
    blocklist and the allowlist ONLY - the caller's original string is what
    reaches the transport, so a normalisation mistake can never redirect a
    command to a surface other than the one the caller named.

    Args:
        name: The topic, service or action name as the caller spelled it.

    Returns:
        The name with a leading separator and without a trailing one. Case is
        preserved: ROS 2 graph names are case-sensitive, so folding it would
        block ``/CMD_VEL``, a different surface no ``/cmd_vel`` subscriber reads.
    """
    stripped = name.strip()
    if not stripped:
        return stripped
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    while len(stripped) > 1 and stripped.endswith("/"):
        stripped = stripped[:-1]
    return stripped


def match_blocklist(name: str, blocked: frozenset[str]) -> bool:
    """Report whether ``name`` matches ``blocked`` exactly or by final path segment.

    A bare entry covers every namespaced instance of that surface, so a
    ``/cmd_vel`` entry blocks ``/robot1/cmd_vel`` as well - the namespace is a
    deployment detail and the surface is what moves the robot. An entry that is
    itself namespaced (``/joint_trajectory_controller/joint_trajectory``) still
    matches only by its own final segment or in full, which is what keeps it from
    being widened by accident.

    Args:
        name: The graph name the caller aimed a command at.
        blocked: The entries to match against.

    Returns:
        True when the command is aimed at one of ``blocked``.
    """
    canonical = canonical_command_name(name)
    targets = {canonical_command_name(entry) for entry in blocked}
    if canonical in targets:
        return True
    return any("/" + canonical.rsplit("/", 1)[-1] == target for target in targets)


def command_block_message(kind: str, name: str) -> str | None:
    """Name the blocklisted surface a command verb is aimed at, or None."""
    if match_blocklist(name, COMMAND_BLOCKLIST):
        return f"{name!r} is a safety-critical command surface, blocked for {kind}."
    return None


def gate_command(kind: str, name: str, tool_context: ToolContext | None, *, tool: str) -> str | None:
    """HIL gate for a command verb aimed at a blocklisted surface.

    Called from every action of every transport that carries a command to a
    robot, because the same physical surface is reachable through all of them.
    The read-only actions are never gated.

    Args:
        kind: The tool action carrying the command, e.g. ``"publish"``.
        name: The topic, service or action name the command targets.
        tool_context: The agent tool context supplying ``interrupt()``.
        tool: The calling tool's name, e.g. ``"use_ros"``. Keys the interrupt id
            and the audit event source, so an audit of an incident says which
            transport reached the robot.

    Returns:
        A refusal message for the caller to return through its own error wrapper,
        or None to let the command proceed. Four outcomes, in order: the surface
        is not blocklisted -> proceed; STRANDS_ROS2_COMMAND_ALLOW names it,
        exactly or by base name (see :func:`match_blocklist` - a bare entry covers
        every namespaced instance) -> allow silently;
        BYPASS_TOOL_CONSENT=true -> allow with a WARNING log; otherwise prompt
        the operator, failing closed when no interrupt is reachable.
    """
    block_msg = command_block_message(kind, name)
    if block_msg is None:
        return None

    allow_raw = os.environ.get(COMMAND_ALLOW_ENV)
    if allow_raw is not None:
        allowed = frozenset(entry.strip() for entry in allow_raw.split(",") if entry.strip())
        if match_blocklist(name, allowed):
            logger.debug("%s to %s allowed via %s", kind, name, COMMAND_ALLOW_ENV)
            return None

    if os.environ.get(BYPASS_CONSENT_ENV, "").lower() == "true":
        logger.warning("BYPASS_TOOL_CONSENT: allowing %s to blocked command surface %s", kind, name)
        return None

    if tool_context is None:
        return (
            f"{block_msg} No tool_context available for operator approval. "
            f"Set {COMMAND_ALLOW_ENV} or {BYPASS_CONSENT_ENV}=true to allow in headless mode."
        )

    try:
        response: Any = tool_context.interrupt(
            f"{tool}-command-approval",
            reason={
                "action": kind,
                "target": name,
                "warning": f"{block_msg} Reply 'y' to approve, anything else to deny.",
            },
        )
    except RuntimeError as exc:
        return f"{kind} to {name!r} requires operator approval, but interrupts are not available: {exc}"

    approved = approve_response(response)
    log_operator_response(f"{tool}_tool", kind, name, approved=approved, response=response)
    if not approved:
        return f"{kind} to {name!r} was declined by the operator."

    logger.info("%s to %s approved via operator interrupt", kind, name)
    return None
