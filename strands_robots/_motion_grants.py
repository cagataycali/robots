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

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.utils import refusal_repr, refusal_str

__all__ = [
    "DETAIL_FIELDS",
    "DIRECT_SERIAL_TOOLS",
    "GRANT_TTL_ENV",
    "POLICY_FIELDS",
    "UNKEYED_PARAMETERS",
    "calibration_identity",
    "consume_grant",
    "deposit_grant",
    "forget_grants_for_peer",
    "gated_view",
    "grant_key",
    "grant_ttl_s",
    "motion_fields",
    "pending_grants",
    "resolve_target",
    "unkeyed_parameters",
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
#:
#: ``robot_id`` names the pose library ``pose_tool`` reads, so ``pose_name=wave``
#: under another id is another set of joint targets. ``smooth``, ``steps`` and
#: ``step_delay`` are the speed profile: a yes for a move interpolated over a
#: second is not a yes for one full-speed write to the same targets. The policy
#: fields (:data:`POLICY_FIELDS`) name what drives a robot on ``execute`` /
#: ``start``: a yes for the ``mock`` policy is not a yes for a checkpoint.
#:
#: Every parameter of a direct-serial tool is either in this roster, one of the
#: fixed key parts :func:`grant_key` reads by name, or listed in
#: :data:`UNKEYED_PARAMETERS` with the reason it changes no motion;
#: :func:`unkeyed_parameters` is the check.
POLICY_FIELDS = (
    "policy_provider",
    "policy_host",
    "policy_port",
    "pretrained_name_or_path",
    "policy_type",
    "embodiment",
    "model_path",
    "walk",
    "target_velocity",
)

DETAIL_FIELDS = (
    "calibration",
    "robot_id",
    "pose_name",
    "motor_name",
    "motor_id",
    "positions",
    "position",
    "velocity",
    "delta",
    "smooth",
    "steps",
    "step_delay",
    "data",
    "hex_data",
    "duration",
    "source_peer_id",
    "device_name",
    *POLICY_FIELDS,
)

#: The fields :func:`grant_key` reads by name rather than through the roster.
_FIXED_KEY_FIELDS = frozenset({"action", "port", "target", "instruction", "message"})

#: Direct-serial tool parameters deliberately left out of the key, each with the
#: reason it cannot change what the arm does. Anything a tool declares that is
#: neither keyed nor here is reported by :func:`unkeyed_parameters`.
UNKEYED_PARAMETERS: dict[str, dict[str, str]] = {
    "pose_tool": {
        "description": "the label save_pose stores with a pose; no motion action reads it",
        "tool_context": "the SDK's invocation context, not a caller field",
    },
    "serial_tool": {
        "baudrate": "the line speed the same bytes go out at; it chooses no motor and no target",
        "timeout": "how long a read waits for a reply; no write depends on it",
        "read_bytes": "the size of the reply buffer send_read reads back into",
        "tool_context": "the SDK's invocation context, not a caller field",
    },
}

#: What ``pose_tool`` runs with when the caller leaves a keyed field out. The
#: dashboard hook sees the model's call, with the defaults omitted, while the
#: tool hands the gate the values it will actually use; both are read through
#: :func:`gated_view` so they name one call the same way.
_TOOL_DEFAULTS: dict[str, dict[str, Any]] = {
    "pose_tool": {"robot_id": "so101_follower", "smooth": True, "steps": 20, "step_delay": 0.05},
}

#: ``pose_tool`` actions that read ``smooth``; they interpolate when it is true.
_POSE_SMOOTH_ACTIONS = frozenset({"load_pose", "move_multiple"})
#: ``pose_tool`` actions that always interpolate, whatever ``smooth`` says.
_POSE_ALWAYS_SMOOTH_ACTIONS = frozenset({"reset_to_home"})


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


def gated_view(tool_name: str, tool_input: Mapping[str, Any] | None) -> dict[str, Any]:
    """The call as the tool will run it: its defaults filled in, the fields it will not read dropped.

    The dashboard hook deposits a grant for the model's call, which leaves out
    every parameter it did not spell; ``pose_tool`` spends it with the values
    it resolved. Keyed raw, a call that relied on ``robot_id``'s default and the
    tool's own payload would name one motion two ways. So both read the call
    through here: an omitted ``robot_id``, ``smooth``, ``steps`` or
    ``step_delay`` is the value the tool uses, and the speed profile is kept
    only on the actions that read it (``smooth`` on ``load_pose`` and
    ``move_multiple``; ``steps`` and ``step_delay`` whenever the action
    interpolates). Any other tool's call is returned as given.

    Args:
        tool_name: The gated tool's name.
        tool_input: The call as the gate or the tool saw it.

    Returns:
        A new dict; *tool_input* is not modified.
    """
    view = dict(tool_input or {})
    defaults = _TOOL_DEFAULTS.get(tool_name)
    if defaults is None:
        return view
    for key, value in defaults.items():
        if view.get(key) is None:
            view[key] = value
    action = str(view.get("action") or "").strip()
    if action not in _POSE_SMOOTH_ACTIONS:
        view.pop("smooth", None)
    # ``is not False`` rather than truthiness: a flag the tool will refuse still
    # keys the speed profile, so binding errs towards more of the call.
    interpolates = action in _POSE_ALWAYS_SMOOTH_ACTIONS or (
        action in _POSE_SMOOTH_ACTIONS and view.get("smooth") is not False
    )
    if not interpolates:
        view.pop("steps", None)
        view.pop("step_delay", None)
    return view


def unkeyed_parameters(tool_name: str, parameters: Iterable[str]) -> list[str]:
    """The parameters of *tool_name* that a grant would not bind and nobody said why.

    Args:
        tool_name: A direct-serial tool's name.
        parameters: The names its signature declares.

    Returns:
        Each name that is neither a fixed key part, on :data:`DETAIL_FIELDS`, nor
        in :data:`UNKEYED_PARAMETERS` for this tool, in the order given.
    """
    allowed = _FIXED_KEY_FIELDS | set(DETAIL_FIELDS) | set(UNKEYED_PARAMETERS.get(tool_name, {}))
    return [name for name in parameters if name not in allowed]


def calibration_identity(value: Any) -> str:
    """What a grant records about the calibration a motion was approved under.

    ``"none"`` when the call carries no calibration: that is the servo's full
    rotation, a real and different frame of reference, so it is named rather
    than left out. A path is identified by the content of the file it names
    (``sha256:<16 hex>`` of its records in canonical JSON, the same digest
    the loaded records give), so a symlink or a relative spelling of the same file
    spends the same grant, and a path that cannot be read as a JSON object is its own identity
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
        return _records_identity(value)
    if isinstance(value, (str, Path)):
        try:
            data = json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return f"unreadable:{refusal_str(str(value))}"
        if not isinstance(data, Mapping):
            return f"unreadable:{refusal_str(str(value))}"
        return _records_identity(data)
    return f"other:{refusal_repr(value)}"


def _records_identity(records: Mapping[str, Any]) -> str:
    """``sha256:<16 hex>`` of *records* in canonical JSON.

    A file and the records loaded from it hash the same, so ``pose_tool`` can
    spend a grant against the calibration it actually built its controller
    from instead of reading the file a second time.
    """
    canonical = json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()[:16]


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


#: ``calibration_records`` was not passed: the identity is read from the call's own field.
_FROM_FIELD: Any = object()


def _calibration_part(tool_input: Mapping[str, Any], calibration_records: Any) -> str:
    """The key's calibration part: from the records a spender loaded, or else from the field."""
    if calibration_records is _FROM_FIELD:
        return f"calibration={calibration_identity(tool_input.get('calibration'))}"
    if calibration_records is None:
        return "calibration=none"
    records = {
        str(name): dataclasses.asdict(record) if dataclasses.is_dataclass(record) else record  # type: ignore[arg-type]
        for name, record in calibration_records.items()
    }
    return f"calibration={_records_identity(records)}"


def grant_key(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    *,
    calibration_records: Mapping[str, Any] | None = _FROM_FIELD,
) -> str:
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
    and the call's own motion fields, read through :func:`gated_view` so an
    omitted default and the value the tool runs with are one call. A per-build binding is not consulted, and does not need to be: a
    bound proxy tool IS its peer, so ``tool_name`` already names the robot.

    Args:
        tool_name: The gated tool's name.
        tool_input: The call as the gate saw it; ``None`` is an empty call.
        calibration_records: The calibration records a spender actually
            loaded (``None`` for none). When given, the calibration part is
            their identity rather than a fresh read of the file the
            ``calibration`` field names, so the yes is matched against what the
            controller will be built from. The file and its records hash alike.

    Returns:
        ``repr`` of the parts tuple. A tuple rather than a ``"|"`` join because
        these values are model-authored: a ``"|"`` inside one of them would
        otherwise shift a boundary and let two different calls agree.
    """
    tool_input = gated_view(tool_name, tool_input)
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
            _calibration_part(tool_input, calibration_records),
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


def consume_grant(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    *,
    calibration_records: Mapping[str, Any] | None = _FROM_FIELD,
) -> bool:
    """True exactly once per deposited grant for this call's shape.

    The gated surfaces call this before asking the operator themselves, so a
    human who has already said yes to this exact motion is not asked twice. No
    grant deposited means no answer given, which is what the caller's own gate
    then goes and gets.

    Args:
        tool_name: The tool about to run.
        tool_input: The call as the tool received it - the same field names the
            operator was shown, with the unset ones omitted.
        calibration_records: The calibration the spender loaded; see :func:`grant_key`.

    Returns:
        True when a grant for this exact call existed and was spent.
    """
    key = grant_key(tool_name, tool_input, calibration_records=calibration_records)
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
