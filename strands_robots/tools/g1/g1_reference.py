"""Agent-facing reads of the four constant tables the G1 gates its writes on.

Every verb here answers from a module-level table and touches nothing else: no
DDS, no driver instance, no SDK. ``import strands_robots.tools.g1.g1_reference``
pulls no ``unitree_sdk2py`` submodule, the import-hygiene contract every other
file in this package carries (refs strands-labs/robots#358), so a headless
runner with no robot resolves the same answers a bring-up host does.

One rule covers all four: **no query lists the table, a query resolves one
entry in it.** That is why there are four verbs rather than the nine this
package used to publish - a table's catalogue and its membership question are
one verb with an optional argument, not two tool-schema slots in every agent's
context.

The four tables, and the write each one decides before it is attempted:

* ``g1_joints`` - ``G1Driver.send_action`` refuses an action-dict key outside
  the driver's joint map.
* ``g1_motion_gates`` - ``G1Driver._check_motion_gates`` refuses a write while
  the live FSM id sits outside the scope's admission set.
* ``g1_arm_actions`` - ``G1ArmActionClient.ExecuteAction`` returns ``rc=7402``
  for an action id outside the SDK's own set.
* ``g1_error_codes`` - any handler's integer ``rc``, decoded to the text every
  verb in this package quotes verbatim on a refusal.

The joint map and the gate sets are read live off the driver's own constants,
so a driver-side widen moves the write path and this lookup together. The arm
action map and the error catalogue are literal snapshots of the SDK's tables,
kept out of the SDK so this module stays importable without it; an SDK release
that renumbers either one is a driver-side update both sides then read.

What none of these decide:

* Live driver state. Whether the robot's ``fsm_id`` currently sits inside a
  gate, or whether ``rt/armsdk`` is already held by another writer
  (``rc=7400`` at wire time), is a read on a running driver - ``g1_get_state``
  answers the first and only an execute answers the second.
* Which joints a given G1 build physically has. Slots ``13`` / ``14`` (waist
  roll and pitch) and ``20 / 21 / 27 / 28`` (the wrist pairs) are named by the
  driver's map on every build because ``send_action`` accepts every name in
  the map regardless of the hardware - a caller naming a joint the robot does
  not have receives a firmware refusal, not a name error. The per-build
  question, and the ankle-pitch / ankle-roll rename, are open on refs
  strands-labs/robots#2765; this lookup returns the names the driver's map
  declares today and follows it if that lands.
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _G1_NAMED_JOINTS,
    _SDK_KD,
    _SDK_KP,
)
from strands_robots.drivers.unitree._common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)

_ISSUE = "Refs strands-labs/robots#358."
_JOINT_ISSUE = "Refs strands-labs/robots#2765."

# --------------------------------------------------------------------------
# joints
# --------------------------------------------------------------------------

#: The five body groups the driver's joint map is organised around, indexed by
#: the same slot as the driver's own map. Kept here rather than in the driver
#: because the driver has no use for the grouping (its write path indexes by
#: slot); the group name is a label this lookup returns to make the map
#: legible.
_GROUP_SLOTS: dict[str, tuple[int, ...]] = {
    "left_leg": (0, 1, 2, 3, 4, 5),
    "right_leg": (6, 7, 8, 9, 10, 11),
    "waist": (12, 13, 14),
    "left_arm": (15, 16, 17, 18, 19, 20, 21),
    "right_arm": (22, 23, 24, 25, 26, 27, 28),
}

#: The driver's map reversed: slot -> name. Precomputed once at module load
#: rather than searched linearly on every lookup - the map is 29 entries, so
#: the O(1) here is not a hot-path win, it is a correctness win: a
#: lookup-by-slot that walked the dict would silently return the first name any
#: two slots collided on, and this reversal is where such a collision surfaces
#: as a ``ValueError`` at import time rather than as a wrong answer at call
#: time.
_NAMES_BY_SLOT: tuple[str, ...] = tuple(name for name, _ in sorted(_G1_JOINT_INDEX.items(), key=lambda kv: kv[1]))
if len(_NAMES_BY_SLOT) != _G1_NAMED_JOINTS:
    raise ValueError(
        f"G1 joint map has {len(_G1_JOINT_INDEX)} names filling "
        f"{_G1_NAMED_JOINTS} slots - the reversal is not one-to-one, so the "
        f"driver's map is malformed. {_JOINT_ISSUE}"
    )

#: Splits on either an underscore boundary or a lower-to-upper camel-case
#: boundary so ``left_knee``, ``LeftKnee`` and ``leftKnee`` all normalise to
#: the same key. This is the only alias layer: the driver's write path indexes
#: by the exact snake_case key, so a wider tolerance here would let a caller
#: reach the lookup with a spelling the driver would then refuse - which is the
#: failure mode this verb exists to prevent.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _slot_group(slot: int) -> str | None:
    """The group name for a slot, or ``None`` if the slot is out of range."""
    for group, slots in _GROUP_SLOTS.items():
        if slot in slots:
            return group
    return None


def _slot_row(slot: int) -> dict[str, Any]:
    """The record this module returns for a single joint slot."""
    return {
        "index": slot,
        "name": _NAMES_BY_SLOT[slot],
        "group": _slot_group(slot),
        "kp": _SDK_KP[slot],
        "kd": _SDK_KD[slot],
    }


def _joint_table(group: str) -> dict[str, Any]:
    """The catalogue payload for the whole map, or for one group of it."""
    slots = _GROUP_SLOTS[group] if group else tuple(range(_G1_NAMED_JOINTS))
    return {
        "status": "success",
        "count": len(slots),
        "group": group or None,
        "groups": sorted(_GROUP_SLOTS),
        "joints": [_slot_row(slot) for slot in slots],
    }


@tool
def g1_joints(query: str | int = "") -> dict[str, Any]:
    """Read the joint-name / slot / gain table ``G1Driver.send_action`` writes against.

    Read-only; every field is a driver constant and no bus is touched. Call it
    before a ``send_action`` or ``run_policy`` to confirm the name a caller
    intends to send is a key the driver will honour, rather than triggering the
    driver's unknown-key refusal at wire time.

    Args:
        query: Empty lists every slot. A group name (one of ``left_leg``,
            ``right_leg``, ``waist``, ``left_arm``, ``right_arm``) lists that
            group. A joint name resolves that one joint, accepting the
            driver's canonical snake_case verbatim and normalising a
            PascalCase or camelCase spelling to it (``LeftKnee`` and
            ``leftKnee`` both reach ``left_knee``); the alias is one-way, so a
            caller who needs the write-path spelling reads the returned
            ``name`` rather than the input. An int resolves that slot, in
            ``[0, 28]``.

    Returns:
        For a catalogue query, a dict with ``status``, a ``count`` of returned
        rows, the ``group`` filter that was applied (``None`` when none was),
        the list of ``groups``, and a ``joints`` list of records carrying
        ``index``, ``name``, ``group``, ``kp`` and ``kd``. For a query that
        resolves one joint, a dict with ``status`` and that single record's
        fields at the top level. A name or group that is not in the driver's
        map, a slot outside ``[0, 28]``, a bool slot (``True`` is an ``int``
        but a caller writing it means a typo, and a live write built on it
        would land at the left hip) and any other type all carry
        ``status="error"`` with a message naming the value and the domain.
    """
    if isinstance(query, bool):
        return {
            "status": "error",
            "message": f"query must be a joint name, a group name or an int slot; got bool {query!r}. {_JOINT_ISSUE}",
        }
    if isinstance(query, int):
        if not 0 <= query < _G1_NAMED_JOINTS:
            return {
                "status": "error",
                "message": f"index {query} is out of range [0, {_G1_NAMED_JOINTS - 1}]. {_JOINT_ISSUE}",
            }
        return {"status": "success", **_slot_row(query)}
    if not isinstance(query, str):
        return {
            "status": "error",
            "message": (
                f"query must be a joint name, a group name or an int slot; "
                f"got {type(query).__name__} {query!r}. {_JOINT_ISSUE}"
            ),
        }
    if not query or query in _GROUP_SLOTS:
        return _joint_table(query)
    key = _CAMEL_BOUNDARY.sub("_", query.strip()).lower()
    if key not in _G1_JOINT_INDEX:
        return {
            "status": "error",
            "message": (
                f"no joint or group named {query!r}. The groups are "
                f"{sorted(_GROUP_SLOTS)} and the driver's map is "
                f"{sorted(_G1_JOINT_INDEX)}. {_JOINT_ISSUE}"
            ),
        }
    return {"status": "success", **_slot_row(_G1_JOINT_INDEX[key])}


# --------------------------------------------------------------------------
# motion gates
# --------------------------------------------------------------------------

#: The two scopes the driver's gate accepts, each reaching the driver's own
#: admission set. The driver's gate takes a scope name and picks one set; this
#: map is the agent-facing side of the same choice, and is kept here because
#: the driver has no use for the mapping.
_SCOPE_SETS: dict[str, frozenset[int]] = {
    "arm": HANDSHAKE_FSMS,
    "loco": WALK_FSMS,
}

#: The scope a membership query answers for when the caller names none. The
#: arm-SDK set is the broader of the two, and is the default the driver's own
#: gate check reads.
_DEFAULT_SCOPE = "arm"

#: The error-table entry the driver's write path quotes when it refuses on an
#: FSM outside its admitted set. Named here so the returned envelope carries
#: the exact refusal string the driver would surface.
_FSM_REFUSAL_CODE: int = 7404


def _unknown_scope(scope: str) -> dict[str, Any]:
    """The refusal for a scope name the driver's gate does not accept."""
    return {
        "status": "error",
        "message": f"unknown scope {scope!r}. Valid scopes are {sorted(_SCOPE_SETS)}. {_ISSUE}",
    }


@tool
def g1_motion_gates(fsm_id: int | None = None, scope: str = "") -> dict[str, Any]:
    """Read the FSM-id sets ``G1Driver._check_motion_gates`` admits writes on.

    Read-only; every field is a driver constant and no driver instance is
    required. A caller compares the live ``fsm_id`` (from ``g1_get_state``)
    against the set the gate would test membership in, and phrases the refusal
    in its own voice instead of triggering the driver's at wire time.

    Args:
        fsm_id: Omit to list the gates. Supply an int to ask whether that FSM
            id is inside the scope's admission set; a bool is refused, because
            a caller writing ``True`` would otherwise be told the robot is not
            ready when they never asked a valid question.
        scope: Which admission set to read. ``arm`` is the set arm-SDK-shaped
            writes need, ``loco`` the narrower set locomotion needs (it
            excludes ``500``, since sitting accepts arm gestures but not
            walking). Empty lists both scopes, and on a membership query means
            the arm gate.

    Returns:
        For a listing, a dict with ``status``, a ``count`` of returned
        records, the ``scope`` filter that was applied (``None`` when none
        was), the list of ``scopes``, and a ``gates`` list of records carrying
        ``scope``, a sorted ``fsm_ids`` list and the ``refusal_code`` /
        ``refusal_text`` pair the write path would surface. For a membership
        query, a dict with ``status``, the ``scope``, the tested ``fsm_id``,
        an ``admitted`` boolean, the ``fsm_ids`` the answer was computed
        against, and - only when the gate would refuse - the same
        ``refusal_code`` / ``refusal_text`` pair. An unknown scope or a
        non-int ``fsm_id`` carries ``status="error"``.
    """
    if scope and scope not in _SCOPE_SETS:
        return _unknown_scope(scope)
    if fsm_id is None:
        scopes = (scope,) if scope else tuple(sorted(_SCOPE_SETS))
        return {
            "status": "success",
            "count": len(scopes),
            "scope": scope or None,
            "scopes": sorted(_SCOPE_SETS),
            "gates": [
                {
                    "scope": name,
                    "fsm_ids": sorted(_SCOPE_SETS[name]),
                    "refusal_code": _FSM_REFUSAL_CODE,
                    "refusal_text": ERR_CODES[_FSM_REFUSAL_CODE],
                }
                for name in scopes
            ],
        }
    if not isinstance(fsm_id, int) or isinstance(fsm_id, bool):
        return {
            "status": "error",
            "message": f"fsm_id must be an int; got {type(fsm_id).__name__} {fsm_id!r}. {_ISSUE}",
        }
    resolved = scope or _DEFAULT_SCOPE
    admitted = fsm_id in _SCOPE_SETS[resolved]
    result: dict[str, Any] = {
        "status": "success",
        "scope": resolved,
        "fsm_id": fsm_id,
        "admitted": admitted,
        "fsm_ids": sorted(_SCOPE_SETS[resolved]),
    }
    if not admitted:
        result["refusal_code"] = _FSM_REFUSAL_CODE
        result["refusal_text"] = ERR_CODES[_FSM_REFUSAL_CODE]
    return result


# --------------------------------------------------------------------------
# arm actions
# --------------------------------------------------------------------------

#: Snapshot of ``unitree_sdk2py.g1.arm.g1_arm_action_client.action_map`` as the
#: SDK ships it today: 16 pre-programmed gestures plus release-arm. Read here
#: rather than imported so this module stays loadable with no SDK; a driver
#: method fronting ``ExecuteAction`` validates the id against the SDK's own map
#: at wire time, and this snapshot is the agent-facing side of the same set.
_ARM_ACTION_MAP: dict[str, int] = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}

#: The id ``ExecuteAction`` uses to drop the arm-action hold and let the
#: driver's ``send_action`` path resume.
_ARM_RELEASE_ACTION_ID: int = 99

#: The three refusals a caller can see from ``ExecuteAction``: ``7402`` is the
#: id-not-in-set refusal this lookup is the pre-check for, ``7401`` the "arm is
#: holding, release first" refusal a second gesture without a ``99`` between
#: raises, ``7400`` the single-writer refusal two concurrent executes raise.
#: Named here so the returned envelope carries the exact strings a driver-side
#: wrapper would surface, and so a re-wording lands once.
_INVALID_ACTION_CODE: int = 7402
_HOLDING_CODE: int = 7401
_TOPIC_BUSY_CODE: int = 7400


@tool
def g1_arm_actions(query: str | int = "") -> dict[str, Any]:
    """Read the arm-action ids ``G1ArmActionClient.ExecuteAction`` admits.

    Read-only; every field is an SDK or driver constant and no channel is
    opened. A caller resolves an intended gesture against the SDK's set before
    an execute dispatches, rather than triggering the SDK's ``rc=7402`` at wire
    time. Execution itself is not here: the gestures write ``rt/armsdk``,
    which the driver does not yet front (refs strands-labs/robots#2765).

    Args:
        query: Empty lists the whole table. A name asks whether the SDK admits
            that gesture, case-sensitively, because the SDK does not
            lower-case its own lookup and a caller writing ``Two-Hand Kiss``
            gets a key miss on the wire. An int asks the same of a raw action
            id; a bool is refused, since ``True`` is an ``int`` but not an id a
            caller can have meant.

    Returns:
        For a listing, a dict with ``status``, a ``count`` of actions, an
        ``action_map`` of name to id (a fresh dict, so a caller mutating it
        cannot poison the snapshot), a sorted ``action_ids`` list, the
        ``release_action_id``, an ``arm_ready_fsm_ids`` list naming the FSM ids
        the arm-SDK gate admits on (arm-action execution is arm-SDK-shaped and
        shares that gate), and a ``refusals`` list carrying the three SDK-side
        refusal codes with their decoded text. For a membership query, a dict
        with ``status``, a ``query`` sub-dict echoing what was asked, an
        ``admitted`` boolean, and either the resolved ``action_name`` /
        ``action_id`` pair an execute would forward or the ``refusal_code`` /
        ``refusal_text`` the SDK would return. A bool or any other type
        carries ``status="error"``.
    """
    if isinstance(query, bool):
        return {
            "status": "error",
            "message": f"query must be an action name or an int id; got bool {query!r}. {_ISSUE}",
        }
    if not isinstance(query, (str, int)):
        return {
            "status": "error",
            "message": (f"query must be an action name or an int id; got {type(query).__name__} {query!r}. {_ISSUE}"),
        }
    if query == "":
        return {
            "status": "success",
            "count": len(_ARM_ACTION_MAP),
            "action_map": dict(_ARM_ACTION_MAP),
            "action_ids": sorted(_ARM_ACTION_MAP.values()),
            "release_action_id": _ARM_RELEASE_ACTION_ID,
            "arm_ready_fsm_ids": sorted(HANDSHAKE_FSMS),
            "refusals": [
                {"code": code, "text": ERR_CODES[code]}
                for code in (_INVALID_ACTION_CODE, _HOLDING_CODE, _TOPIC_BUSY_CODE)
            ],
        }
    if isinstance(query, str):
        resolved_name = query if query in _ARM_ACTION_MAP else None
        echo: dict[str, Any] = {"action": query}
    else:
        # The map is 16 entries, so the reverse scan is not worth an index.
        resolved_name = next((name for name, aid in _ARM_ACTION_MAP.items() if aid == query), None)
        echo = {"action_id": query}
    result: dict[str, Any] = {
        "status": "success",
        "query": echo,
        "admitted": resolved_name is not None,
    }
    if resolved_name is not None:
        result["action_name"] = resolved_name
        result["action_id"] = _ARM_ACTION_MAP[resolved_name]
    else:
        result["refusal_code"] = _INVALID_ACTION_CODE
        result["refusal_text"] = ERR_CODES[_INVALID_ACTION_CODE]
    return result


# --------------------------------------------------------------------------
# error codes
# --------------------------------------------------------------------------

#: What the catalogue renders for a code it does not carry. Mirrors the
#: driver-side decoder: a code outside the snapshot reads as ``unknown`` so a
#: caller can tell a name the package has from a name it does not, and never
#: has to branch on a missing key.
_UNKNOWN_CODE_TEXT: str = "unknown"


@tool
def g1_error_codes(code: int | None = None) -> dict[str, Any]:
    """Read the SDK return codes the G1's locomotion and arm handlers surface.

    Read-only; the answer is a snapshot of the driver-side catalogue every verb
    in this package quotes verbatim on a refusal, so a caller who received a
    ``refusal_code`` from any other verb resolves it here and reads the same
    sentence. No SDK call runs, and a code the SDK invented after the snapshot
    was taken reads as unknown rather than raising.

    Args:
        code: Omit to list the catalogue. Supply an int to decode one code. A
            bool is refused - a passed-through truth value is a caller mistake,
            not a decode query - while a negative rc is admitted as unknown:
            the driver-side renderer is total over every integer, and ``-1`` is
            the convention for an SDK call that raised instead of returning a
            code, so refusing it here would make this verb narrower than the
            renderer whose text it quotes.

    Returns:
        For a listing, a dict with ``status``, a ``count`` of catalogued
        codes, an ``error_codes`` list of descriptors carrying ``code`` and
        ``text`` (sorted ascending), and a bare ``codes`` list of the integers
        alone; the containers are fresh, so a caller mutating them cannot
        poison the catalogue. For a decode, a dict with ``status``, a
        ``query`` sub-dict carrying the supplied ``code``, a ``known``
        boolean, and a ``text`` that is the catalogued sentence when the code
        is known and the ``unknown`` marker when it is not. A bool or any
        other non-int carries ``status="error"`` naming the type.
    """
    if isinstance(code, bool):
        return {"status": "error", "message": f"code must be int, got bool ({code!r}). {_ISSUE}"}
    if code is None:
        codes = sorted(ERR_CODES)
        return {
            "status": "success",
            "count": len(codes),
            "error_codes": [{"code": code_, "text": ERR_CODES[code_]} for code_ in codes],
            "codes": codes,
        }
    if not isinstance(code, int):
        return {
            "status": "error",
            "message": f"code must be int, got {type(code).__name__} ({code!r}). {_ISSUE}",
        }
    known = code in ERR_CODES
    return {
        "status": "success",
        "query": {"code": code},
        "known": known,
        "text": ERR_CODES[code] if known else _UNKNOWN_CODE_TEXT,
    }
