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

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.utils import refusal_repr, refusal_str

__all__ = [
    "DETAIL_FIELDS",
    "DIRECT_SERIAL_TOOLS",
    "GRANT_TTL_ENV",
    "calibration_identity",
    "consume_grant",
    "deposit_grant",
    "forget_grants_for_peer",
    "grant_key",
    "grant_ttl_s",
    "motion_fields",
    "pending_grants",
    "resolve_target",
]

logger = logging.getLogger(__name__)

#: How long a human's yes stays spendable, in seconds. A grant is a decision
#: about one motion at one moment; an operator who approved a move at 09:00
#: did not approve the same numbers at 17:00 for whoever drives the agent then.
#: The approved call can also fail ABOVE the gate (a calibration file that will
#: not load, a target outside the arm's travel) and leave its grant unspent,
#: which is the shape that makes a stale grant reachable at all.
GRANT_TTL_ENV = "STRANDS_DASH_MOTION_GRANT_TTL_S"
DEFAULT_GRANT_TTL_S = 900.0

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
#: otherwise spendable by a ten-minute one. ``source_peer_id`` and
#: ``device_name`` are the whole payload of the mesh ``teleop_receive`` verb,
#: which leader the robot will follow: a yes for one leader must not be
#: spendable by another.
#:
#: ``calibration`` comes first because it is the frame every number after it is
#: read in: the file decides where a degree target puts the joint, and a bus
#: given none commands the servo's full rotation instead of the arm's measured
#: travel. ``pose_tool`` handed it to the gate under a comment saying the
#: operator approves both together, while this roster left it out, so a yes for
#: ``position=30`` under one arm's file was spendable under another's, or under
#: no file at all. :func:`grant_key` binds the CONTENT of the file too, through
#: :func:`calibration_identity`, so an absent calibration is an explicit
#: identity rather than a missing field.
DETAIL_FIELDS = (
    "calibration",
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
    "source_peer_id",
    "device_name",
)


@dataclass(frozen=True)
class _Grant:
    """One deposited yes: who it is about and when it was given (monotonic seconds)."""

    tool: str
    action: str
    target: str
    deposited_at: float


_grants_lock = threading.Lock()
_grants: dict[str, _Grant] = {}

#: The characters a peer id loses on its way to a tool name (``peer_tools.sanitize_tool_name``).
_TOOL_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def grant_ttl_s() -> float:
    """The grant lifetime the operator set, or the default when they set nothing usable.

    Read at every spend, so a tightened window applies to grants already given.
    A value that is not a finite positive number falls back to the default
    rather than being used: ``nan`` would make the age comparison False for
    every grant and remove the bound instead of widening it, the same failure
    ``mesh.core._parse_positive_float_env`` documents for the mesh knobs.
    """
    raw = os.getenv(GRANT_TTL_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_GRANT_TTL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0f s", GRANT_TTL_ENV, raw, DEFAULT_GRANT_TTL_S)
        return DEFAULT_GRANT_TTL_S
    if not math.isfinite(value) or value <= 0:
        logger.warning("%s=%r is not a finite positive number; using %.0f s", GRANT_TTL_ENV, raw, DEFAULT_GRANT_TTL_S)
        return DEFAULT_GRANT_TTL_S
    return value


def _sweep_expired_locked(now: float, ttl: float) -> None:
    """Drop every grant older than *ttl*. Caller holds ``_grants_lock``."""
    for key in [k for k, g in _grants.items() if now - g.deposited_at > ttl]:
        _grants.pop(key, None)


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


def calibration_identity(value: Any) -> str:
    """What a grant records about the calibration a motion was approved under.

    ``"none"`` when the call carries no calibration: that is the servo's full
    rotation, a real and different frame of reference, so it is named rather
    than left out. A path is identified by the content of the file it names
    (``sha256:<16 hex>``), so a symlink or a relative spelling of the same file
    spends the same grant, and a path that cannot be read is its own identity
    (the tool refuses it anyway; the grant must not be spendable by the file
    that appears there later). An inline record is hashed canonically.

    Args:
        value: The ``calibration`` field as the gate saw it.

    Returns:
        A short string that is equal exactly when the frame of reference is.
    """
    if value is None or value == "":
        return "none"
    if isinstance(value, Mapping):
        canonical = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        return "sha256:" + hashlib.sha256(canonical).hexdigest()[:16]
    if isinstance(value, (str, Path)):
        try:
            data = Path(value).expanduser().read_bytes()
        except (OSError, ValueError):
            return f"unreadable:{refusal_str(str(value))}"
        return "sha256:" + hashlib.sha256(data).hexdigest()[:16]
    return f"other:{refusal_repr(value)}"


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
    :func:`resolve_target` resolved, the instruction, the identity of the
    calibration the numbers are read in (:func:`calibration_identity`, so a
    yes under one arm's file is not spendable under another's or under none),
    and the call's own motion fields. A per-build binding is not consulted, and does not need to be: a
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
            # The frame of reference, by CONTENT. The detail line shows the path
            # the model wrote; the key binds what the file says, so a symlink or
            # a relative spelling of the same file spends the same grant and a
            # different file, or none, does not.
            f"calibration={calibration_identity(tool_input.get('calibration'))}",
            *(field for field in motion_fields(tool_input) if not field.startswith("calibration=")),
        )
    )


def deposit_grant(tool_name: str, tool_input: Mapping[str, Any] | None) -> None:
    """Grant one pass through the gate to the next call with this exact shape.

    Args:
        tool_name: The tool the operator answered for.
        tool_input: The call they were shown.
    """
    tool_input = tool_input or {}
    record = _Grant(
        tool=tool_name,
        action=str(tool_input.get("action") or "").strip(),
        target=resolve_target(tool_name, tool_input, None),
        deposited_at=time.monotonic(),
    )
    with _grants_lock:
        _sweep_expired_locked(record.deposited_at, grant_ttl_s())
        _grants[grant_key(tool_name, tool_input)] = record


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
    now = time.monotonic()
    ttl = grant_ttl_s()
    with _grants_lock:
        record = _grants.pop(key, None)
        if record is None:
            return False
        if now - record.deposited_at > ttl:
            logger.info(
                "motion grant for %s %s on %s expired unspent after %.0f s (window %.0f s)",
                record.tool,
                record.action,
                record.target or "(no target)",
                now - record.deposited_at,
                ttl,
            )
            return False
        return True


def forget_grants_for_peer(peer_id: str) -> int:
    """Drop every grant about *peer_id*: the ones that name it as their target, and the ones
    given to its bound proxy tool, whose name IS the peer (``sanitize_tool_name``).

    Called when a peer leaves the fleet snapshot or the mesh is re-pointed. A yes
    about a robot that is no longer there is not a yes about the robot that
    comes back under that name.

    Returns:
        How many grants were dropped.
    """
    peer_id = str(peer_id or "").strip()
    if not peer_id:
        return 0
    tool_name = _TOOL_NAME_UNSAFE.sub("_", peer_id)
    with _grants_lock:
        doomed = [
            k
            for k, g in _grants.items()
            if g.target == peer_id or (g.tool == tool_name and g.tool not in DIRECT_SERIAL_TOOLS)
        ]
        for key in doomed:
            _grants.pop(key, None)
    return len(doomed)


def pending_grants() -> list[dict[str, Any]]:
    """The unspent grants, oldest first, without their keys: what is outstanding, for an operator screen."""
    now = time.monotonic()
    ttl = grant_ttl_s()
    with _grants_lock:
        _sweep_expired_locked(now, ttl)
        records = sorted(_grants.values(), key=lambda g: g.deposited_at)
    return [
        {
            "tool": g.tool,
            "action": g.action,
            "target": g.target,
            "age_s": round(now - g.deposited_at, 3),
            "expires_in_s": round(ttl - (now - g.deposited_at), 3),
        }
        for g in records
    ]
