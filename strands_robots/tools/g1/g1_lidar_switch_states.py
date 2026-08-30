"""Agent-facing lookup for the wire strings the LiDAR switch topic accepts.

The Unitree G1 exposes a control channel for the head-mounted Livox
Mid-360 on the ``rt/utlidar/switch`` DDS topic: publishing an
``std_msgs::String_`` message whose ``data`` field is the literal
ASCII ``"ON"`` powers the LiDAR on, and the literal ``"OFF"`` powers
it off. The neon bundle's ``g1_lidar_switch`` verb
(``cagataycali/neon-the-g1/tools/g1_lidar.py``) writes those two
strings verbatim from a Python ``bool`` argument (``True`` selects
``"ON"``, ``False`` selects ``"OFF"``) and no third literal value is
honoured by the firmware today - a message carrying any other
``data`` string is silently dropped by the LiDAR firmware, and the
neon path has no refusal shape naming that miss because the write
succeeds at the DDS layer regardless.

This module snapshots that two-entry admission set as an agent-facing
lookup so a caller planning a future driver-side ``g1_lidar_switch``
wrapper can name the wire strings the firmware admits decidably
before the write is attempted, rather than watching the LiDAR fail
to change state after a mistyped literal reaches the wire. The verb
pair mirrors :mod:`~strands_robots.tools.g1.g1_arm_actions` and
:mod:`~strands_robots.tools.g1.g1_dangerous_publish_topics`: one
snapshot lookup naming the whole set with the wire literals
preserved byte-for-byte, one membership decision on one query.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_lidar_switch`` writes
  ``rt/utlidar/switch`` through the neon-side ``_STATE["switch_pub"]``
  publisher; that write is a live DDS reach that would compete with
  the driver's own subscribers for the same bus, and a future
  driver-side switch method belongs on the driver rather than on a
  second publisher path in this package. This module ports the
  read-only lookup half without also introducing a second wire
  writer the driver does not yet own; refs
  strands-labs/robots#358 for the SDK-facing seam that write
  belongs on. When the driver's switch method lands, its refusal
  will quote the same two literals this lookup admits so both
  sides name the same set.
* An SDK re-import. The wire literals are captured here as a
  ``tuple[str, str]`` of ASCII strings; a snapshot lookup reading
  this module pulls no ``unitree_sdk2py`` submodule (the
  module-load hygiene contract every other file in this package
  carries, refs strands-labs/robots#358) and pulls no CycloneDDS
  binding either. A firmware release that widens the admission
  set (a hypothetical ``"STANDBY"`` third state) is a driver-side
  update; when the driver's method surfaces the widened set, this
  snapshot and its tests move together.

What this module does not decide.

* Whether the LiDAR is currently on or off. That is a live-state
  read the driver's ``_on_lidar_state`` handler answers through
  :mod:`~strands_robots.tools.g1.g1_lidar_state`; this lookup only
  names which wire literals the switch topic admits as writes, not
  what the LiDAR is doing right now. A caller planning a switch
  reads the current state through the sibling verb, decides
  whether a write is needed, then admits the intended literal
  through this verb before the write.
* Whether the ``rt/utlidar/switch`` topic is currently reachable.
  The topic sits on the driver's DDS bus; a bus that has not been
  initialised or a driver that has not connected surfaces the
  refusal at the write site, not here. This lookup answers a pure
  membership question against a two-entry set and cannot report a
  wire condition.
* The IDL type of the switch message. The topic carries
  ``std_msgs::String_`` (a single UTF-8 ``data`` field); that
  type answer lives on :mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types`
  next to every other catalog topic's IDL declaration, not on
  this lookup. Colocating the type with this lookup would restate
  a fact the sibling module already carries verbatim.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the ordered ``rt/utlidar/switch`` wire literals the
#: neon bundle's ``g1_lidar_switch`` verb publishes. The tuple
#: preserves the observed truth-table order the neon source writes
#: the switch in: ``"ON"`` first because the neon verb's default
#: ``on=True`` argument selects that literal on the branch a caller
#: reaches without arguments, and ``"OFF"`` second because it is
#: the ``on=False`` branch. The wire strings are ASCII uppercase
#: exactly as the LiDAR firmware admits them; a lowercase
#: ``"on"`` or a mixed-case ``"On"`` is silently ignored by the
#: firmware today and this snapshot pins the case-sensitive form.
#:
#: The tuple type is deliberate: a caller reading
#: :data:`_LIDAR_SWITCH_STATES` should not be able to mutate the
#: snapshot in place and thereby drift the lookup out of sync with
#: the neon source. The verbs iterate the tuple in order so the
#: list payload names the same wire order a future driver-side
#: switch method would honour.
_LIDAR_SWITCH_STATES: tuple[str, ...] = ("ON", "OFF")

#: The error-table entry a future driver-side wrapper would quote
#: on a switch literal outside :data:`_LIDAR_SWITCH_STATES`. The
#: neon bundle's ``g1_lidar_switch`` does not refuse an unknown
#: literal (it writes whatever ``on`` maps to, and the firmware
#: silently drops the frame if the ``data`` field is not one of
#: the two admitted strings); a caller-side membership refusal
#: uses the ``7404`` gate-refusal shape a future driver-side
#: wrapper would quote when refusing at the same boundary. The
#: write path and this lookup share the constant so the same
#: numeric surfaces on both refusal sites, rather than the
#: lookup inventing its own code that the driver's refusal
#: would not name back.
_INVALID_SWITCH_STATE_CODE: int = 7404

#: Roles the tuple entries carry, keyed by the wire literal. The
#: role labels are descriptive rather than functional - the LiDAR
#: firmware treats each literal as an idempotent switch write and
#: does not carry state semantics beyond the two-branch on/off
#: choice - so a caller reading the descriptor sees which power
#: state each literal selects and what the observable effect is,
#: without also having to read the neon source. The role text is
#: a snapshot of the neon module's own inline comments and the
#: Livox Mid-360 manual's power-cycle wording.
_LIDAR_SWITCH_ROLES: dict[str, dict[str, str]] = {
    "ON": {
        "role": "power_on",
        "description": (
            "Powers the head-mounted Livox Mid-360 on. The neon "
            "verb's default ``on=True`` argument selects this "
            "literal. After a successful frame reaches the wire "
            "the LiDAR begins publishing ``rt/utlidar/cloud_livox_mid360`` "
            "point-cloud frames (typical warm-up is one to two "
            "seconds before the first cloud arrives) and its "
            "``rt/utlidar/lidar_state`` topic reports the active "
            "operating mode a caller reads through "
            "``g1_lidar_state``. A second ``ON`` write against an "
            "already-on LiDAR is a no-op the firmware accepts "
            "silently; the neon path does not gate the second "
            "write and neither does a future driver-side wrapper."
        ),
    },
    "OFF": {
        "role": "power_off",
        "description": (
            "Powers the head-mounted Livox Mid-360 off. The neon "
            "verb's ``on=False`` argument selects this literal. "
            "After a successful frame reaches the wire the LiDAR "
            "stops publishing ``rt/utlidar/cloud_livox_mid360`` "
            "point-cloud frames within one publish tick, and its "
            "``rt/utlidar/lidar_state`` topic reports the powered-"
            "off operating mode. A second ``OFF`` write against "
            "an already-off LiDAR is a no-op the firmware accepts "
            "silently, matching the ``ON`` idempotence."
        ),
    },
}


def _describe(state: str) -> dict[str, Any]:
    """Build the per-state descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_lidar_switch_states` so
    :func:`g1_lidar_switch_state_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor
    lands in one place. Every field is a snapshot read; no bus
    is touched and no DDS topic is written.
    """
    role = _LIDAR_SWITCH_ROLES[state]
    return {
        "wire_literal": state,
        "role": role["role"],
        "description": role["description"],
        "is_default": state == "ON",
    }


@tool
def g1_list_lidar_switch_states() -> dict[str, Any]:
    """Name the two wire literals the ``rt/utlidar/switch`` topic admits.

    Read-only. Returns the same two ASCII strings the neon
    bundle's ``g1_lidar_switch`` verb writes to
    ``rt/utlidar/switch``, in the same truth-table order the
    neon source writes them in (``"ON"`` first, ``"OFF"``
    second). No driver instance, no DDS, no SDK: the payload
    reads only module-level constants.

    The ordering carries meaning the sibling lookups do not: the
    neon verb's default ``on=True`` argument selects the first
    literal on the branch a caller reaches without arguments,
    so preserving ``"ON"`` first pins the argumentless-default
    behaviour a future driver-side wrapper would honour.

    Returns:
        A dict with ``status``; an ``ordered_states`` list
        naming the two wire literals in the neon truth-table
        order; a ``states`` list of per-state descriptors
        carrying ``wire_literal``, ``role``, ``description``,
        and ``is_default`` (the ``"ON"`` entry the neon verb
        selects with no arguments); a ``count`` integer naming
        the tuple length; and a ``refusals`` list naming the
        ``7404`` refusal code a future driver-side wrapper
        would quote on a literal outside the admitted set.
    """
    return {
        "status": "success",
        "ordered_states": list(_LIDAR_SWITCH_STATES),
        "states": [_describe(state) for state in _LIDAR_SWITCH_STATES],
        "count": len(_LIDAR_SWITCH_STATES),
        "refusals": [
            {"code": _INVALID_SWITCH_STATE_CODE, "text": ERR_CODES[_INVALID_SWITCH_STATE_CODE]},
        ],
    }


@tool
def g1_lidar_switch_state_admits(wire_literal: str | None = None) -> dict[str, Any]:
    """Decide whether a ``wire_literal`` sits inside the switch admission set.

    Read-only. Compares one string against the neon-observed
    :data:`_LIDAR_SWITCH_STATES` tuple and reports the admitted
    descriptor on match, or the ``7404`` refusal code a future
    driver-side wrapper would quote on miss. No driver instance,
    no DDS, no SDK: the decision reads only module-level
    constants and the argument itself.

    A literal inside the admitted set is *not* the same as an
    admitted write: the ``rt/utlidar/switch`` topic sits on the
    driver's DDS bus, and a bus that has not been initialised
    surfaces the refusal at the write site rather than here.
    This verb answers a pure membership question against a
    two-entry set; a caller comparing an intended write against
    both conditions (membership + bus reachable) reaches the
    driver after this verb admits the literal.

    Args:
        wire_literal: The ``rt/utlidar/switch`` ``data`` string
            to check, in ASCII uppercase (``"ON"`` or ``"OFF"``
            today). The comparison is on string identity
            against the snapshot; a literal outside the tuple
            is refused with the ``7404`` code. Non-string
            arguments (``bool``, ``int``, ``None``) are refused
            with the same code because the wire message's
            ``data`` field is a UTF-8 string and a non-string
            argument would not be identity-equal to any entry.
            Case-variant strings (``"on"``, ``"On"``) are
            refused decidably rather than uppercased silently:
            the LiDAR firmware admits the ASCII uppercase form
            only, and a lowercase literal reaching the wire is
            silently dropped rather than coerced. Empty
            strings are refused with the same code.

    Returns:
        A dict with ``status``; on admit, a ``state``
        descriptor with ``wire_literal``, ``role``,
        ``description``, and ``is_default`` (the same shape
        :func:`g1_list_lidar_switch_states` returns). On
        refuse, ``refusal_code`` and ``refusal_text`` name the
        ``7404`` code and its decoded text, plus a ``reason``
        string that names why the argument was refused
        (missing argument, non-string argument, empty string,
        or unknown wire literal).
    """
    if wire_literal is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_SWITCH_STATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SWITCH_STATE_CODE],
            "reason": (
                f"wire_literal is required; pass one of {list(_LIDAR_SWITCH_STATES)} so the lookup is decidable"
            ),
        }
    # bool subclasses int (which subclasses object but not str); refuse
    # explicitly so a caller passing True/False as a mistyped switch
    # argument sees a shape refusal rather than a "not a string"
    # message that hides the bool subclass surprise.
    if isinstance(wire_literal, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_SWITCH_STATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SWITCH_STATE_CODE],
            "reason": (
                f"wire_literal={wire_literal!r} is a bool; pass one of {list(_LIDAR_SWITCH_STATES)} as an ASCII str"
            ),
        }
    if not isinstance(wire_literal, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_SWITCH_STATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SWITCH_STATE_CODE],
            "reason": (
                f"wire_literal={wire_literal!r} is not a str; pass one of {list(_LIDAR_SWITCH_STATES)} as an ASCII str"
            ),
        }
    if wire_literal == "":
        return {
            "status": "error",
            "refusal_code": _INVALID_SWITCH_STATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SWITCH_STATE_CODE],
            "reason": (f"wire_literal is empty; pass one of {list(_LIDAR_SWITCH_STATES)} as a non-empty ASCII str"),
        }
    if wire_literal not in _LIDAR_SWITCH_STATES:
        return {
            "status": "error",
            "refusal_code": _INVALID_SWITCH_STATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SWITCH_STATE_CODE],
            "reason": (
                f"wire_literal={wire_literal!r} is not in the admitted set {list(_LIDAR_SWITCH_STATES)}; "
                "the LiDAR firmware admits ASCII uppercase 'ON' and 'OFF' only, and a case-variant "
                "or unrelated string reaching the wire is silently dropped rather than coerced"
            ),
        }
    return {
        "status": "success",
        "state": _describe(wire_literal),
    }
