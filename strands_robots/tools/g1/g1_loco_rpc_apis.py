"""Agent-facing lookup for the raw RPC api ids ``LocoClient._Call`` reads on.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
small set of high-level helpers (``SetFsmId``, ``BalanceStand``,
``StopMove`` ...) as thin wrappers around a lower-level ``_Call(api_id,
payload)`` transport. The neon bundle's ``_g1_common.py``
(``cagataycali/neon-the-g1/tools/_g1_common.py::_loco_call``) reaches
directly through ``_Call`` to read fields the SDK's Python surface
does not expose (``read_fsm_id`` -> ``7001``, ``read_fsm_mode`` ->
``7002``, ``read_balance_mode`` -> ``7003``, ``read_swing_height`` ->
``7004``, ``read_stand_height`` -> ``7005``) and to write one that the
SDK gates behind a distinct api id (``set_swing_height`` -> ``7103``).
The SDK does not ship a canonical api-id to operation-name mapping, so
this module snapshots the six api ids the neon bundle observed against
the real robot into a module-level dict and exposes two agent-facing
verbs so a caller planning a raw ``_Call`` (or reading a refusal that
quotes an api id) can decide the operation decidably before a future
driver-side wrapper is attempted, rather than pinning the mapping
inside the write path where the refusal is invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``_loco_call`` helper wrapped
  ``LocoClient._Call(api_id, payload)`` under a single-writer lock
  (the same ``_LOCO_CALL_LOCK`` the neon file names) and stashed the
  most recent rc per api id in ``_LAST_LOCO_RC`` for the read verbs
  above to surface an actionable error; those calls reach the same
  ``rt/lowcmd``-adjacent locomotion path today's
  :class:`~strands_robots.drivers.g1.G1Driver` gates through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  before ``send_action`` / ``run_policy`` accept a joint payload. A
  future driver method that fronts ``_Call`` for the read api ids
  will land the read verb; the write api id (``7103``
  ``SetSwingHeight``) is a locomotion-shaped write and belongs on
  the same gate the ``BalanceStand`` refusal quotes (see
  :mod:`~strands_robots.tools.g1.g1_balance_modes` and
  :mod:`~strands_robots.tools.g1.g1_swing_height_envelope`). This
  module ports the read-only lookup half without also introducing a
  second locomotion caller path the driver does not yet own.
* An SDK re-import. The api-id table is captured here as a
  module-level constant snapshot of the six api ids the neon bundle
  observed against the real robot; the constant lives here rather
  than being re-imported from the SDK so ``import
  strands_robots.tools.g1.g1_loco_rpc_apis`` pulls no
  ``unitree_sdk2py`` submodule (the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358). An SDK release that widens or narrows
  the raw ``_Call`` api set is a driver-side update; when the
  driver's read/write verbs land, their refusals will name the same
  ``3102`` / ``3104`` transport codes and the same ``7302`` /
  ``7404`` gate-refusal codes the
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entries here
  return, so both sides quote the same text.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` currently admits a write on
  the ``7103`` api. ``SetSwingHeight`` is a locomotion-shaped write
  that reaches the controller through the same gate
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` narrows on;
  the caller planning that write compares the driver's live fsm
  (from ``G1Driver.get_status``) against ``walk_ready_fsm_ids`` this
  verb surfaces to decide whether the gate is currently open. The
  read api ids (``7001`` .. ``7005``) do not touch the write gate;
  they read state fields off the locomotion service directly.
* Whether the LocoClient's cached RPC future is currently wedged.
  The neon bundle observed that a long-running LocoClient singleton
  can desynchronise its per-call response future and return
  ``rc=3104`` (``RPC_CLIENT_API_TIMEOUT``) forever until the client
  is dropped and rebuilt (the neon comment on ``_recreate_loco_client``
  spells the fix out); this lookup names the ``3104`` code as the
  transport refusal a caller can see, but does not decide whether a
  live singleton is wedged. That is a live driver-instance query
  answered by :mod:`~strands_robots.tools.g1.g1_state` when the
  future read wrapper lands.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    WALK_FSMS,
)

#: Snapshot of the raw ``_Call`` api ids the Unitree G1 locomotion SDK
#: reads on. Each key is the integer id the neon bundle's
#: ``_loco_call(api_id, payload)`` helper passes through to
#: ``LocoClient._Call``; each value is the operation label the neon
#: bundle observed against the real robot (the read-side helpers
#: ``read_fsm_id`` .. ``read_stand_height`` and the write-side
#: ``set_swing_height`` each pin one row here). The SDK does not ship
#: a canonical id-to-name mapping; these labels are the ones the neon
#: bundle observed against the real robot and the ones a future
#: driver-side wrapper's refusal string would quote.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``_Call``-side of the conversation; a caller
#: that only needs the gate set reaches
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` directly.
#: Colocating the map with the verb mirrors ``_BALANCE_MODE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_balance_modes` and
#: ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_LOCO_RPC_APIS: dict[int, str] = {
    7001: "SetFsmId / GetFsmId",
    7002: "SetFsmMode / GetFsmMode",
    7003: "SetBalanceMode / GetBalanceMode",
    7004: "SetSwingHeight / GetSwingHeight",
    7005: "SetStandHeight / GetStandHeight",
    7103: "SetSwingHeight (write)",
}

#: The subset of :data:`_LOCO_RPC_APIS` that writes to the locomotion
#: service. Only one api id today: ``7103`` (``SetSwingHeight``); the
#: neon bundle's ``read_swing_height`` uses ``7004`` for the paired
#: read. A future SDK release that adds a second write api id would
#: widen this set; the sibling test asserts every id here is also in
#: :data:`_LOCO_RPC_APIS` so a widen surfaces as a shape change here.
_WRITE_APIS: frozenset[int] = frozenset({7103})

#: The subset of :data:`_LOCO_RPC_APIS` that reads state off the
#: locomotion service rather than writing to it. Read-side api ids do
#: not touch the driver's ``_check_motion_gates`` (they do not shape
#: an ``rt/lowcmd`` frame); the write-side api id ``7103``
#: (``SetSwingHeight``) does. A caller planning a raw ``_Call``
#: compares the intended api id against this set to decide whether
#: the driver's motion gate is on its path.
#:
#: Derived from :data:`_LOCO_RPC_APIS` minus :data:`_WRITE_APIS` so a
#: widen to :data:`_LOCO_RPC_APIS` cannot leave this set stale: a new
#: api id lands in exactly one of the two partitions by construction,
#: and the sibling test asserts the partition is disjoint and covers
#: the admitted set.
_READ_APIS: frozenset[int] = frozenset(_LOCO_RPC_APIS) - _WRITE_APIS

#: The transport-level rc codes ``LocoClient._Call`` surfaces when the
#: RPC channel itself fails, independent of whether the api id is
#: valid. Named here so the returned envelope carries the exact codes
#: the driver's read wrapper would surface, and so a re-wording lands
#: in one place instead of drifting between the driver's log and this
#: lookup. Both codes come from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`.
_TRANSPORT_REFUSAL_CODES: tuple[int, ...] = (3102, 3104)

#: The error-table entry the SDK's own handler quotes when an api id
#: outside the admitted set reaches ``_Call``. The SDK does not ship
#: a dedicated "invalid loco api id" code; the same ``7404``
#: gate-refusal shape a locomotion-write refusal uses is quoted here
#: because a call to an unknown api id lands at the same handler
#: boundary. Named separately from the transport refusal codes so a
#: future SDK release that adds a distinct code lands here without
#: also renaming the transport entries.
_INVALID_API_CODE: int = 7404


def _describe(api_id: int) -> dict[str, Any]:
    """Build the per-api descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_loco_rpc_apis` so
    :func:`g1_loco_rpc_api_admits`'s admitted-path payload names the
    same fields, and so a widen to the descriptor lands in one place.
    Every field is a snapshot read; no bus is touched.
    """
    return {
        "api_id": api_id,
        "operation": _LOCO_RPC_APIS[api_id],
        "kind": "write" if api_id in _WRITE_APIS else "read",
        "touches_motion_gate": api_id in _WRITE_APIS,
    }


@tool
def g1_list_loco_rpc_apis() -> dict[str, Any]:
    """Return the raw ``_Call`` api ids ``LocoClient`` admits today.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant snapshot of the api ids the neon bundle
    observed against the real robot. Useful before a future
    driver-side wrapper for ``LocoClient._Call`` is called, so a
    caller can compare an intended api id against the set the neon
    bundle's ``_loco_call`` helper documented as valid, and can also
    compare a write-side api id against ``walk_ready_fsm_ids`` to
    decide whether the driver's motion gate would admit the write.

    Returns:
        A dict with ``status``; an ``apis`` list of per-api
        descriptors sorted by ``api_id`` ascending, each carrying
        ``api_id``, ``operation`` (the neon-observed label),
        ``kind`` (``"read"`` for the five read api ids, ``"write"``
        for ``7103``), and ``touches_motion_gate`` (``True`` when
        the api id shapes a locomotion write the driver's
        ``_check_motion_gates`` refuses on ``_fsm_id`` outside
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS``); a
        ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the
        set the driver's motion gate admits locomotion-shaped
        writes on; a ``transport_refusals`` list carrying the
        ``3102`` (RPC send fail) and ``3104`` (RPC timeout) codes
        and their decoded text (the transport-level refusals a
        caller can see independent of the api id); and a ``refusals``
        list carrying the ``7404`` invalid-api-id code and its
        decoded text (the boundary refusal a future driver-side
        wrapper would surface on an api id outside the admitted set).
        Every field is a snapshot of an observed api id or a driver
        constant; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "apis": [_describe(api_id) for api_id in sorted(_LOCO_RPC_APIS)],
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "transport_refusals": [{"code": code, "text": ERR_CODES[code]} for code in _TRANSPORT_REFUSAL_CODES],
        "refusals": [
            {"code": _INVALID_API_CODE, "text": ERR_CODES[_INVALID_API_CODE]},
        ],
    }


@tool
def g1_loco_rpc_api_admits(api_id: int | None = None) -> dict[str, Any]:
    """Decide whether an ``api_id`` sits inside the admitted set.

    Read-only. Compares one integer argument against the neon-observed
    :data:`_LOCO_RPC_APIS` and reports the admitted descriptor on
    match, or the ``7404`` invalid-api-id code on miss. No driver
    instance, no DDS, no SDK: the decision reads only module-level
    constants and the argument itself.

    An api id inside the admitted set is *not* the same as an
    admitted call: the driver's motion gate (``_check_motion_gates``)
    also refuses locomotion-shaped writes on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    payload names ``walk_ready_fsm_ids`` and ``touches_motion_gate``
    so a caller comparing an intended call against both conditions
    has the FSM set on hand.

    Args:
        api_id: The raw ``_Call`` api id to check
            (``7001`` .. ``7005`` for the read side, or ``7103`` for
            the write side today). Bool values (``True`` / ``False``)
            are refused with the ``7404`` code because Python's
            ``bool`` is a subclass of ``int`` and a caller passing
            ``True`` would otherwise look up ``1`` (unknown),
            returning a confusing refusal. A non-integer non-bool
            argument is refused with the same code for the same
            reason. A missing argument is refused with the same
            code so the lookup is decidable in every case.

    Returns:
        A dict with ``status``; on admit, an ``api`` descriptor with
        ``api_id``, ``operation``, ``kind``, and
        ``touches_motion_gate`` (the same shape
        :func:`g1_list_loco_rpc_apis` returns), plus
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        refuse, ``refusal_code`` and ``refusal_text`` name the
        ``7404`` code and its decoded text, plus a ``reason`` string
        that names why the argument was refused (unknown api id,
        missing argument, or non-integer argument).
    """
    if api_id is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (f"api_id not supplied; pass an int in the admitted set {sorted(_LOCO_RPC_APIS)}"),
        }
    if isinstance(api_id, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (f"api_id={api_id!r} is a bool; pass an int in the admitted set {sorted(_LOCO_RPC_APIS)}"),
        }
    if not isinstance(api_id, int):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (f"api_id={api_id!r} is not an int; pass an int in the admitted set {sorted(_LOCO_RPC_APIS)}"),
        }
    if api_id not in _LOCO_RPC_APIS:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (f"api_id={api_id} is not in the admitted set {sorted(_LOCO_RPC_APIS)}"),
        }
    return {
        "status": "success",
        "api": _describe(api_id),
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
