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

The decision is not ROS-shaped, though. :func:`gate_motion` is the same
allowlist -> bypass -> operator-interrupt -> audit-row path with the blocklist
factored out, for a tool whose command surface is not a graph name at all: a
serial write to a motor bus, where the target is a port and the verb is a wire
instruction. Such a caller names its own allowlist variable and its own way of
matching it; the interrupt, the
fail-closed rule and the audit row are the one copy here, so an operator's "no"
means the same thing whichever tool asked. :func:`gate_command` is now a thin
blocklist front on that path.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
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
    return gate_motion(
        tool,
        kind,
        name,
        block_msg,
        tool_context,
        allow_env=COMMAND_ALLOW_ENV,
        allow_match=lambda allowed: match_blocklist(name, allowed),
    )


def _allow_exact_or_star(target: str) -> Callable[[frozenset[str]], bool]:
    """The default allowlist matcher: the exact target, or ``*`` for every one."""

    def _match(allowed: frozenset[str]) -> bool:
        return "*" in allowed or target in allowed

    return _match


def gate_motion(
    tool: str,
    action: str,
    target: str,
    warning: str,
    tool_context: ToolContext | None,
    *,
    allow_env: str,
    allow_match: Callable[[frozenset[str]], bool] | None = None,
) -> str | None:
    """Transport-agnostic operator gate for one command that can move a robot.

    The caller has already decided the command is one that needs approval -
    that is what ``warning`` states - so this is the decision path only:
    the caller's allowlist variable, then ``BYPASS_TOOL_CONSENT``, then the
    operator, failing closed when no interrupt is reachable. It is the one
    copy of that path, shared by the ROS transports (through
    :func:`gate_command`) and :mod:`~strands_robots.tools.serial_tool`, so the
    interrupt id, the refusal wording and the audit row cannot differ between
    two tools reaching the same robot.

    Args:
        tool: The calling tool's name, e.g. ``"serial_tool"``. Keys the
            interrupt id ``<tool>-command-approval`` and the audit source
            ``<tool>_tool``, so an incident audit says which tool asked.
        action: The verb carrying the command, e.g. ``"publish"`` or
            ``"feetech_position"``. Recorded on the audit row.
        target: What the command is aimed at - a topic, a ``service.operation``
            pair, a serial port. Recorded on the audit row and matched against
            the allowlist.
        warning: One sentence saying why this command needs approval, shown
            to the operator and returned in the headless refusal.
        tool_context: The agent tool context supplying ``interrupt()``. None
            means no operator is reachable, and the command is refused.
        allow_env: The environment variable naming pre-approved targets for
            this tool, comma-separated. Named in the headless refusal so an
            operator knows what to set.
        allow_match: How the allowlist entries match ``target``. Default: the
            exact target, or a ``*`` entry for every target of this tool.

    Returns:
        A refusal message for the caller to return through its own error
        wrapper, or None to let the command proceed. In order: the allowlist
        names the target -> allow silently; ``BYPASS_TOOL_CONSENT=true`` ->
        allow with a WARNING log; no ``tool_context`` -> refuse, naming both
        variables; otherwise prompt the operator and record the reply.
    """
    match = allow_match or _allow_exact_or_star(target)
    allow_raw = os.environ.get(allow_env)
    if allow_raw is not None:
        allowed = frozenset(entry.strip() for entry in allow_raw.split(",") if entry.strip())
        if match(allowed):
            logger.debug("%s to %s allowed via %s", action, target, allow_env)
            return None

    if os.environ.get(BYPASS_CONSENT_ENV, "").lower() == "true":
        logger.warning("BYPASS_TOOL_CONSENT: allowing %s to blocked command surface %s", action, target)
        return None

    if tool_context is None:
        return (
            f"{warning} No tool_context available for operator approval. "
            f"Set {allow_env} or {BYPASS_CONSENT_ENV}=true to allow in headless mode."
        )

    try:
        response: Any = tool_context.interrupt(
            f"{tool}-command-approval",
            reason={
                "action": action,
                "target": target,
                "warning": f"{warning} Reply 'y' to approve, anything else to deny.",
            },
        )
    except RuntimeError as exc:
        return f"{action} to {target!r} requires operator approval, but interrupts are not available: {exc}"

    approved = approve_response(response)
    log_operator_response(f"{tool}_tool", action, target, approved=approved, response=response)
    if not approved:
        return f"{action} to {target!r} was declined by the operator."

    logger.info("%s to %s approved via operator interrupt", action, target)
    return None
