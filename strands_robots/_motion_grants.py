"""The one-shot grants a human's yes leaves behind, and the identity they carry.

An operator is asked before a robot moves, and a robot moves through more than
one surface: the dashboard's :class:`~strands_robots.dashboard.agent_hitl.MotionInterruptHook`
asks before the tool call, and the tool itself asks again through the shared
command gate. Asking twice for one motion is worse than asking once - the second
prompt is the same question with less context - so an answered yes is recorded
here and the surface that runs the motion spends it instead of re-asking.

That makes the grant store a contract three layers share: the dashboard deposits
(``app``'s :class:`~strands_robots.hardware_robot.Robot`, ``tools``'s
``pose_tool`` and ``serial_tool`` spend), so it belongs under all of them. It
lived in the dashboard package, which meant each spender reached *up* into the
web layer for it and had to survive that layer being absent::

    try:
        from strands_robots.dashboard import agent_hitl
    except ImportError:
        return False

Two consequences, both gone now that the store sits here. A safety decision was
read through an optional extra, so in an install without ``[dashboard]`` the
answer to "did a human already say yes?" was decided by a failed import rather
than by the store. And where the extra *is* installed, the first gated call on
the motion path imported the dashboard package - whose ``__init__`` requires
``fastapi``, ``uvicorn``, ``webauthn`` and PyJWT - to read a ``set`` in this
process: 232 modules and 337 ms, between the agent's request and the servo
write.

The identity a grant is keyed on is the whole of its safety: a grant spendable
by a call the operator was not shown is a motion nobody approved. So
:func:`grant_key` is built from the facts the gate resolved and showed them -
the tool, the action, the target, the instruction, and the call's own
motion-bearing fields - and :data:`DETAIL_FIELDS` is read once, by
:func:`motion_fields`, for both the line the operator reads and the key their
answer is filed under.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DETAIL_FIELDS",
    "DIRECT_SERIAL_TOOLS",
    "consume_grant",
    "deposit_grant",
    "grant_key",
    "motion_fields",
    "resolve_target",
]

#: Tools whose gated input names the motion in FIELDS, not an instruction string.
DIRECT_SERIAL_TOOLS: frozenset[str] = frozenset({"pose_tool", "serial_tool"})

#: The motion-bearing fields, in the order an operator reads them: which servo
#: first, then what it is being told to do.
#:
#: Every gated action's payload must appear here, because this roster is what
#: makes one call distinguishable from another -- it is read both for the line
#: the operator is shown and by :func:`grant_key`, for the identity their yes is
#: recorded against. A payload field missing from it is therefore invisible twice
#: over: the human approves a motion the gate declined to describe, and their
#: grant is deposited under a key some other call also owns.
#:
#: ``motor_id`` and ``velocity`` are the whole payload of ``serial_tool``'s
#: ``feetech_velocity``, and ``hex_data`` is the second spelling of ``send`` /
#: ``send_read`` -- the raw bytes that go on the bus. Absent, those three actions
#: rendered as an empty detail line. ``duration`` is here for the same reason on
#: the ``fleet`` surface: it is shown to the operator, and how long a robot moves
#: is part of what they said yes to, so a yes for a five-second task was
#: otherwise spendable by a ten-minute one.
DETAIL_FIELDS = (
    "pose_name",
    "motor_name",
    "motor_id",
    "positions",
    "position",
    "velocity",
    "delta",
    "steps",
    "data",
    "hex_data",
    "duration",
)

_grants_lock = threading.Lock()
_grants: set[str] = set()


def motion_fields(tool_input: Mapping[str, Any]) -> tuple[str, ...]:
    """``field=value`` for each motion-bearing field this call carries, in roster order.

    The one reading of :data:`DETAIL_FIELDS`, so the operator's line and the
    grant key cannot come to describe a call differently. An omitted field and an
    empty one are the same thing here: neither names any motion.

    Args:
        tool_input: The call as the gate saw it.

    Returns:
        One ``field=value`` string per field the call carries.
    """
    return tuple(
        f"{key}={tool_input[key]}"
        for key in DETAIL_FIELDS
        if tool_input.get(key) is not None and tool_input.get(key) != ""
    )


def resolve_target(
    tool_name: str,
    tool_input: Mapping[str, Any],
    bound_targets: Mapping[str, str] | None,
) -> str:
    """The peer or port a yes would move, read from the most trusted source.

    Precedence is by TRUST, never by presence. The model authors ``tool_input``
    and the ``peers`` action that lists a sim's name is deliberately ungated, so
    a sim peer's name is always within its reach; resolving the target from a
    field it writes lets it choose which robot the gate believes it is asking
    about. Each tool therefore has exactly one trusted source, and a field the
    model wrote is read only where the tool itself reads the same field:

    * a proxy tool IS its peer, so the per-build binding names the target and no
      input can move it. This is the guarantee the binding exists to make.
    * a direct-serial tool addresses a ``port``, one of its own declared
      parameters. Neither ``pose_tool`` nor ``serial_tool`` declares ``target``,
      and the SDK drops undeclared keys before the call, so a ``target`` on such
      an input is unconsumed by construction: reading it would let the model
      name a robot that is not the one the port moves.
    * every other gated tool (``fleet``) declares ``target`` itself, so the gate
      and the tool resolve the same peer from the same field.

    An unresolvable target is returned empty, which is never a key on the peers
    snapshot, so :func:`~strands_robots.dashboard.agent_motion.peer_is_physical`
    treats it as metal and the call is gated.

    Args:
        tool_name: The gated tool's name.
        tool_input: The call as the gate saw it.
        bound_targets: Per-build tool-to-peer bindings, when the host has any.

    Returns:
        The resolved peer or port, or ``""`` when none is resolvable.
    """
    if bound_targets is not None and tool_name in bound_targets:
        return str(bound_targets.get(tool_name) or "").strip()
    if tool_name in DIRECT_SERIAL_TOOLS:
        return str(tool_input.get("port") or "").strip()
    return str(tool_input.get("target") or "").strip()


def grant_key(tool_name: str, tool_input: Mapping[str, Any] | None) -> str:
    """The identity a human yes is recorded against: what they were shown, verbatim.

    A grant is spendable by exactly one call, so the key has to name that call.
    Reading ``tool_input["target"]`` did not: the two tools the dashboard hook is
    the ONLY human gate for do not declare a ``target`` at all -- their peer is
    the ``port``, which is why :func:`resolve_target` reads that field instead --
    and they carry the motion itself in :data:`DETAIL_FIELDS`, not in an
    ``instruction`` string. Three of the four parts were therefore constant for
    them, and every ``pose_tool`` / ``serial_tool`` call of one action hashed to
    the same ``tool|action||``. A yes for ``motor_name=shoulder_pan
    position=2048`` on ``/dev/ttyACM0`` was spendable by ``motor_name=elbow_flex
    position=4095`` on ``/dev/ttyACM1``: a different joint, on a different arm, to
    a different angle, with no human asked. The gate had already resolved the
    port and shown the operator those very fields -- the key was the one place
    that dropped them.

    So the parts are the facts the gate resolves, read the same way it reads
    them: the tool, the action as the gate matched it (stripped), the target
    :func:`resolve_target` resolved, the instruction, and the call's own motion
    fields. A per-build binding is not consulted, and does not need to be: a
    bound proxy tool IS its peer, so ``tool_name`` already names the robot.

    Args:
        tool_name: The gated tool's name.
        tool_input: The call as the gate saw it; ``None`` is an empty call.

    Returns:
        ``repr`` of the parts tuple. A tuple rather than a ``"|"`` join because
        these values are model-authored: a ``"|"`` inside one of them would
        otherwise shift a boundary and let two different calls agree.
    """
    tool_input = tool_input or {}
    return repr(
        (
            tool_name,
            str(tool_input.get("action") or "").strip(),
            resolve_target(tool_name, tool_input, None),
            str(tool_input.get("instruction") or tool_input.get("message") or ""),
            *motion_fields(tool_input),
        )
    )


def deposit_grant(tool_name: str, tool_input: Mapping[str, Any] | None) -> None:
    """Grant one pass through the gate to the next call with this exact shape.

    Args:
        tool_name: The tool the operator answered for.
        tool_input: The call they were shown.
    """
    with _grants_lock:
        _grants.add(grant_key(tool_name, tool_input))


def consume_grant(tool_name: str, tool_input: Mapping[str, Any] | None) -> bool:
    """True exactly once per deposited grant for this call's shape.

    The gated surfaces call this before asking the operator themselves, so a
    human who has already said yes to this exact motion is not asked twice. No
    grant deposited means no answer given, which is what the caller's own gate
    then goes and gets.

    Args:
        tool_name: The tool about to run.
        tool_input: The call as the tool received it - the same field names the
            operator was shown, with the unset ones omitted.

    Returns:
        True when a grant for this exact call existed and was spent.
    """
    key = grant_key(tool_name, tool_input)
    with _grants_lock:
        if key in _grants:
            _grants.discard(key)
            return True
    return False
