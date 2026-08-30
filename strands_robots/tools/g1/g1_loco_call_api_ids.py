"""Agent-facing lookup for the ``LocoClient._Call`` API ids the neon bundle admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
handful of read-side motion-state queries and one write-side setter as
raw ``_Call(api_id, payload_json)`` dispatches rather than as named
Python helpers. The neon bundle's
``cagataycali/neon-the-g1/tools/_g1_common.py`` catalogues those ids
across its ``read_fsm_id`` / ``read_fsm_mode`` / ``read_balance_mode``
/ ``read_swing_height`` / ``read_stand_height`` / ``set_swing_height``
helpers, and its ``_loco_call`` / ``_loco_read_data`` wrappers pin the
one non-zero rc a wedged singleton returns
(:data:`_RPC_TIMEOUT_CODE`) as the marker for the "recreate the
LocoClient and retry once" recovery path. This module snapshots the
six admitted API ids and their neon-observed roles into a module-level
constant, and exposes two agent-facing verbs -
:func:`g1_list_loco_call_api_ids` (name the whole envelope) and
:func:`g1_loco_call_api_id_admits` (decide one query) - so a caller
can decide the refusal decidably before a future driver-side wrapper
for the ``_Call`` path is attempted, rather than triggering the SDK's
``rc=3103`` ("RPC_CLIENT_API_NOT_REG") refusal at wire time. Refs
strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``_loco_call`` helper wrapped
  ``LocoClient._Call(api_id, payload_json)`` under its
  ``_LOCO_CALL_LOCK`` because the underlying SDK future is a single
  in-flight slot per client (concurrent ``_Call`` from different
  threads returns ``rc=3104``, "RPC_CLIENT_API_TIMEOUT"); those
  writes / reads are the same locomotion RPC channel today's
  :class:`~strands_robots.drivers.g1.G1Driver` gates through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  before ``send_action`` / ``run_policy`` accept a joint payload
  (for API id ``7103``, which sets swing height) and through the
  driver's own cached motion-switcher reads (for the ``7001`` /
  ``7002`` FSM reads, which the driver now sources via
  :mod:`~strands_robots.tools.g1._motion_switcher`, refs
  strands-labs/robots#2916). A future driver method that fronts the
  read-side ids will land alongside the write-side gate; refs
  strands-labs/robots#358 for the SDK-facing gate work that write
  belongs on. This module ports the read-only enumeration half
  without also introducing a second locomotion writer path the
  driver does not yet own.
* An SDK re-import. The API-id table is captured here as a
  module-level constant snapshot of the six ids the neon bundle
  observed against the real robot; the constant lives here rather
  than being re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_loco_call_api_ids`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358. An SDK release that widens the ``_Call``
  vocabulary (a new locomotion RPC id) is a driver-side update;
  when the driver's read/write method lands, its refusal will quote
  the same ``rc=3103`` "RPC_CLIENT_API_NOT_REG" entry the
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the singleton ``LocoClient`` is currently wedged. The
  neon bundle's ``_loco_call`` treats ``rc=3104``
  (``RPC_CLIENT_API_TIMEOUT``) as a one-shot recovery signal:
  recreate the cached client, retry once, then surface the rc.
  That is a driver-instance read (whether the RPC future is
  in-flight), not a snapshot answer; a caller reaching a future
  driver-side wrapper of any of these ids would see the recovery
  path in the returned envelope, and this lookup only names the
  code the recovery path pivots on so a caller planning the call
  reads the same number the write path will surface.
* Whether the driver's live ``_fsm_id`` currently admits a write
  on ``7103``. That is a driver-instance read carried on the
  driver's ``get_status`` envelope; a caller planning a
  swing-height write compares the driver's live FSM against
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (via
  :mod:`~strands_robots.tools.g1.g1_motion_gates` or
  :mod:`~strands_robots.tools.g1.g1_swing_height_envelope`). The
  ``kind`` field this lookup returns names ``"read"`` for
  ``7001`` / ``7002`` / ``7003`` / ``7004`` / ``7005`` and
  ``"write"`` for ``7103`` so a caller sees which of the two
  gate conversations the id belongs to at a glance, and reaches
  the paired envelope verb when the answer is ``"write"``.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS

#: Snapshot of the ``LocoClient._Call`` API ids the neon bundle
#: (``cagataycali/neon-the-g1/tools/_g1_common.py``) admits as
#: locomotion-side RPC dispatches today. Each descriptor names:
#:
#: * ``role`` - the neon helper the id fronts (``read_fsm_id``,
#:   ``read_fsm_mode``, ``read_balance_mode``, ``read_swing_height``,
#:   ``read_stand_height``, ``set_swing_height``);
#: * ``kind`` - ``"read"`` (the ``_loco_read_data`` path, which
#:   parses ``{"data": X}`` off the response) or ``"write"`` (the
#:   ``set_swing_height`` path, which sends a ``{"data": <float>}``
#:   payload and reads back the rc);
#: * ``payload`` - a short prose description of the payload shape
#:   the neon bundle's wrappers construct (``"{}"`` for reads;
#:   ``'{"data": <float>}'`` for the ``7103`` write).
#:
#: The label table lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``_Call``-side of the conversation; a caller
#: that only needs the write gate reaches :data:`WALK_FSMS` directly.
#: Colocating the id table with the enumeration verb mirrors
#: ``_LOCO_TASK_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_loco_task_ids` and
#: ``_BALANCE_MODE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_balance_modes`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_LOCO_CALL_API_MAP: dict[int, dict[str, str]] = {
    7001: {
        "role": "read_fsm_id",
        "kind": "read",
        "payload": "{}",
        "description": (
            "Read the current FSM id (the motion-switcher state the "
            "controller reports). The neon bundle's ``read_fsm_id`` "
            "helper fronts this; the driver now sources FSM state via "
            ":mod:`~strands_robots.tools.g1._motion_switcher` (refs "
            "strands-labs/robots#2916) rather than through this raw "
            "call."
        ),
    },
    7002: {
        "role": "read_fsm_mode",
        "kind": "read",
        "payload": "{}",
        "description": (
            "Read the current FSM mode (the controller-side "
            "sub-mode number a single FSM id can carry). The neon "
            "bundle's ``read_fsm_mode`` helper fronts this."
        ),
    },
    7003: {
        "role": "read_balance_mode",
        "kind": "read",
        "payload": "{}",
        "description": (
            "Read the current balance mode "
            "(0=Static / 3=Dynamic, per "
            ":mod:`~strands_robots.tools.g1.g1_balance_modes`). The "
            "neon bundle's ``read_balance_mode`` helper fronts this."
        ),
    },
    7004: {
        "role": "read_swing_height",
        "kind": "read",
        "payload": "{}",
        "description": (
            "Read the current swing height (meters). Paired write "
            "path is ``7103``. The neon bundle's "
            "``read_swing_height`` helper fronts this."
        ),
    },
    7005: {
        "role": "read_stand_height",
        "kind": "read",
        "payload": "{}",
        "description": (
            "Read the current stand height (meters). The neon "
            "bundle's ``read_stand_height`` helper fronts this; the "
            "write path is ``LocoClient.SetStandHeight``, not a raw "
            "``_Call`` id."
        ),
    },
    7103: {
        "role": "set_swing_height",
        "kind": "write",
        "payload": '{"data": <float>}',
        "description": (
            "Set the swing height (meters). Reached only through "
            "the raw ``_Call``; the SDK does not expose a named "
            "``SetSwingHeight`` method. The neon bundle's "
            "``set_swing_height`` helper fronts this under its "
            "single-writer lock; the envelope the neon wrapper "
            "clamps is described by "
            ":mod:`~strands_robots.tools.g1.g1_swing_height_envelope`."
        ),
    },
}

#: The subset of :data:`_LOCO_CALL_API_MAP` that fronts a
#: locomotion-shaped *write* to the SDK. Only ``7103`` (swing height)
#: today; called out separately so a caller filtering for writes
#: (which the driver's ``_check_motion_gates`` refuses outside
#: :data:`WALK_FSMS`) does not have to walk the ``kind`` field on
#: every descriptor.
_LOCO_CALL_WRITE_API_IDS: frozenset[int] = frozenset({7103})

#: The error-table entry the SDK's ``_Call`` returns for an API id
#: outside :data:`_LOCO_CALL_API_MAP` (a mis-typed number, or an id
#: from a firmware release the SDK on this host does not know about).
#: Named here so the returned envelope carries the exact refusal
#: string a future driver-side wrapper would surface, and so a
#: re-wording of it lands in one place instead of drifting between
#: the SDK-side log and this lookup.
_INVALID_API_CODE: int = 3103

#: The error-table entry the SDK returns when the LocoClient's RPC
#: future is already in flight ("RPC_CLIENT_API_TIMEOUT"). The neon
#: bundle's ``_loco_call`` uses this specifically as a recovery
#: signal (recreate the cached client, retry once, then surface the
#: rc); named here because :func:`g1_list_loco_call_api_ids` surfaces
#: it alongside the ``3103`` on the returned refusal list so a caller
#: sees both the shape refusal (bad id) and the concurrency refusal
#: (client wedged) at once.
_RPC_TIMEOUT_CODE: int = 3104

#: The error-table entry the driver's own ``_check_motion_gates``
#: quotes when it refuses a locomotion-shaped write on an FSM outside
#: :data:`WALK_FSMS`. Named here because
#: :func:`g1_loco_call_api_id_admits` surfaces it on the write-side
#: descriptor - the ``7103`` swing-height setter would face the gate
#: refusal on top of the SDK's own bad-id refusal.
_GATE_REFUSAL_CODE: int = 7404


def _describe(api_id: int) -> dict[str, Any]:
    """Build the per-id descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_loco_call_api_ids`
    so :func:`g1_loco_call_api_id_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    entry = _LOCO_CALL_API_MAP[api_id]
    return {
        "api_id": api_id,
        "role": entry["role"],
        "kind": entry["kind"],
        "payload": entry["payload"],
        "description": entry["description"],
        "admits_loco_writes": api_id in _LOCO_CALL_WRITE_API_IDS,
    }


@tool
def g1_list_loco_call_api_ids() -> dict[str, Any]:
    """Return the ``LocoClient._Call`` API ids the neon bundle admits.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for any of these ids is called, so a caller can compare an
    intended API id against the set the neon bundle observed against
    the real robot, and decide alongside that whether the id is a
    read (safe from the driver's motion-gate refusal) or a write
    (gated by :data:`WALK_FSMS`).

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        admitted API ids; a ``loco_call_api_ids`` list of
        descriptors (one per admitted id, sorted ascending) carrying
        ``api_id``, ``role`` (the neon helper name), ``kind``
        (``"read"`` or ``"write"``), ``payload`` (the payload shape
        the neon bundle constructs), ``description`` (the
        neon-observed purpose), and ``admits_loco_writes``
        (``True`` on ``7103``, ``False`` elsewhere; surfaced for
        shape parity with
        :mod:`~strands_robots.tools.g1.g1_fsm_targets` and
        :mod:`~strands_robots.tools.g1.g1_loco_task_ids`); an
        ``api_ids`` list of just the ids in sorted order; a
        ``write_api_ids`` list mirroring
        :data:`_LOCO_CALL_WRITE_API_IDS` (the subset the driver's
        gate refuses outside :data:`WALK_FSMS`); a
        ``loco_ready_fsm_ids`` list mirroring
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (the
        gate every write API id shares); and a ``refusals`` list
        carrying the three refusal codes (``3103`` invalid API id,
        ``3104`` RPC future in flight, ``7404`` gate-refused write)
        and their decoded text that a future call verb would
        surface. Every field is a snapshot of an SDK or neon
        constant; no dynamic decode runs here.
    """
    api_ids = sorted(_LOCO_CALL_API_MAP)
    return {
        "status": "success",
        "count": len(_LOCO_CALL_API_MAP),
        "loco_call_api_ids": [_describe(api_id) for api_id in api_ids],
        "api_ids": api_ids,
        "write_api_ids": sorted(_LOCO_CALL_WRITE_API_IDS),
        "loco_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _INVALID_API_CODE, "text": ERR_CODES[_INVALID_API_CODE]},
            {"code": _RPC_TIMEOUT_CODE, "text": ERR_CODES[_RPC_TIMEOUT_CODE]},
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_loco_call_api_id_admits(api_id: int | None = None) -> dict[str, Any]:
    """Decide whether ``api_id`` is inside the neon-observed dispatch set.

    Read-only. Compares one argument against the neon-observed
    :data:`_LOCO_CALL_API_MAP` and reports the admitted descriptor on
    match, or the ``3103`` refusal code a future driver-side wrapper
    would quote on miss. No driver instance, no DDS, no SDK: the
    decision reads only module-level constants and the argument
    itself.

    An id inside the admitted set is *not* the same as an admitted
    call: the driver's gate also refuses a *write* id (``7103``) on
    any FSM outside :data:`WALK_FSMS`, and any id is refused with
    ``rc=3104`` while the singleton ``LocoClient``'s RPC future is
    in flight. Neither of those is a snapshot answer; both are
    live-driver reads a caller reaches after this verb admits the
    id. The returned payload's ``kind`` field names ``"read"`` or
    ``"write"`` so a caller comparing an intended call against both
    conditions (membership + FSM gate for writes) sees which side of
    the gate the id lands on.

    Args:
        api_id: The API id to check. Must be an ``int``; ``bool``
            is refused with the ``3103`` code because ``int(True)``
            is ``1`` and a passed-through boolean is a caller
            mistake, not a valid dispatch query. A missing argument
            (``None``) is refused decidably rather than treated as
            a default.

    Returns:
        A dict with ``status``; on admit, an ``api`` descriptor
        with ``api_id``, ``role``, ``kind``, ``payload``,
        ``description``, and ``admits_loco_writes`` (the same shape
        :func:`g1_list_loco_call_api_ids` returns). On refuse,
        ``refusal_code`` and ``refusal_text`` name the ``3103``
        code and its decoded text, plus a ``reason`` string that
        names why the argument was refused (missing argument, bool
        argument, non-int argument, or unknown API id).
    """
    if api_id is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id is required; pass one of {sorted(_LOCO_CALL_API_MAP)} "
                "so the lookup is decidable. Refs strands-labs/robots#358."
            ),
        }
    if isinstance(api_id, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is a bool; pass one of "
                f"{sorted(_LOCO_CALL_API_MAP)} as an int. "
                "Refs strands-labs/robots#358."
            ),
        }
    if not isinstance(api_id, int):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is not an int; pass one of "
                f"{sorted(_LOCO_CALL_API_MAP)} as an int. "
                "Refs strands-labs/robots#358."
            ),
        }
    if api_id not in _LOCO_CALL_API_MAP:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is not in the admitted set "
                f"{sorted(_LOCO_CALL_API_MAP)}. "
                "Refs strands-labs/robots#358."
            ),
        }
    return {
        "status": "success",
        "api": _describe(api_id),
    }
