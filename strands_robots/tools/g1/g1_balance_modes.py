"""Agent-facing lookup for the balance modes ``LocoClient.BalanceStand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
balance-mode selector via ``BalanceStand(int)`` that internally calls
``SetBalanceMode`` and admits a small set of pre-programmed modes: ``0``
(static balance, the default) and ``3`` (dynamic balance, from the neon
bundle's field notes against the real robot). The SDK does not ship a
canonical mode-id to name mapping, so this module snapshots the two
mode ids the neon bundle's ``g1_balance_stand`` verb
(``cagataycali/neon-the-g1/tools/g1_posture.py``) documented as
walkable, and exposes them as an agent-facing lookup so a caller can
decide the mode decidably before a future driver-side wrapper for
``BalanceStand`` is attempted, rather than pinning it inside the write
path where the refusal is invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_balance_stand`` verb
  wrapped ``LocoClient.BalanceStand(mode)`` under a single-writer
  lock; that write is the ``rt/lowcmd``-adjacent locomotion topic
  today's :class:`~strands_robots.drivers.g1.G1Driver` gates through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  before ``send_action`` / ``run_policy`` accept a joint payload. A
  future driver method that fronts ``BalanceStand`` will land the
  transition verb; refs strands-labs/robots#358 for the SDK-facing
  gate work that write belongs on. This module ports the read-only
  lookup half without also introducing a second locomotion writer
  path the driver does not yet own.
* An SDK re-import. The mode table is captured here as a module-level
  constant snapshot of the two mode ids the neon bundle observed as
  walkable; the constant lives here rather than being re-imported
  from the SDK so ``import strands_robots.tools.g1.g1_balance_modes``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. An SDK release that widens or narrows the
  set is a driver-side update; when the driver's transition method
  lands, its refusal will name the ``rc=7404`` gate-refusal code the
  same :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entry
  this lookup returns.

What this module does not decide.

* Whether the driver's live ``_fsm_id`` currently admits a
  ``BalanceStand`` write. ``BalanceStand`` is a locomotion-shaped
  write that reaches the controller through the same gate
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` narrows on;
  a caller planning a mode change compares the driver's live fsm
  (from ``G1Driver.get_status``) against the ``walk_ready_fsm_ids``
  this verb surfaces to decide whether the write gate is currently
  open. The two membership tests together - mode inside this
  lookup, fsm inside the walk-ready set - are the two conditions a
  future driver-side wrapper would refuse on.
* Which balance modes the controller *will* be stable in from the
  current pose. The neon bundle's field notes flag mode 3 (dynamic)
  as the higher-headroom option; mode 0 (static) is the default
  ``BalanceStand`` used by the neon verb. This module surfaces the
  set the SDK admits, not a pose-dependent stability answer.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    WALK_FSMS,
)

#: Snapshot of the balance-mode ids the Unitree G1 locomotion SDK's
#: ``BalanceStand`` admits today. The neon bundle's ``g1_balance_stand``
#: verb (``cagataycali/neon-the-g1/tools/g1_posture.py``) documented two
#: modes against the real robot: ``0`` (static balance, the SDK default)
#: and ``3`` (dynamic balance, the higher-headroom option observed in
#: the neon bundle's memory.md field notes). The SDK does not ship a
#: canonical mode-id to name mapping; these labels are the ones the
#: neon bundle observed against the real robot and the ones a future
#: driver-side wrapper's refusal string would quote.
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``BalanceStand``-side of the conversation; a
#: caller that only needs the gate set reaches
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` directly.
#: Colocating the map with the verb mirrors ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions` and ``_FSM_NAME_MAP``
#: in :mod:`~strands_robots.tools.g1.g1_fsm_targets`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_BALANCE_MODE_MAP: dict[int, str] = {
    0: "Static",
    3: "Dynamic",
}

#: The error-table entry the driver's own ``_check_motion_gates`` quotes
#: when it refuses a locomotion-shaped write on an FSM outside
#: :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. Named here so
#: the returned envelope carries the exact refusal string a future
#: driver-side ``BalanceStand`` wrapper would surface, and so a
#: re-wording of it lands in one place instead of drifting between the
#: driver's log and this lookup. The write path and this lookup share
#: the constant.
_GATE_REFUSAL_CODE: int = 7404

#: The error-table entry a future driver-side wrapper would quote on a
#: mode id outside :data:`_BALANCE_MODE_MAP`. The SDK's own handler for
#: ``SetBalanceMode`` does not ship a distinct rc for an unknown mode
#: (the mode is a raw integer the controller silently accepts and
#: ignores when outside its programmed set); the neon bundle refused
#: unknown modes at the verb boundary, so this lookup uses the same
#: ``7404`` gate-refusal shape a future driver-side wrapper would
#: quote when refusing at the same boundary. Named separately from
#: :data:`_GATE_REFUSAL_CODE` so a future SDK release that adds a
#: dedicated "invalid balance mode" code lands here without also
#: renaming the gate-refusal constant.
_INVALID_MODE_CODE: int = 7404


def _describe(mode_id: int) -> dict[str, Any]:
    """Build the per-mode descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_balance_modes` so
    :func:`g1_balance_mode_admits`'s admitted-path payload names the
    same fields, and so a widen to the descriptor lands in one place.
    Every field is a snapshot read; no bus is touched.
    """
    return {
        "mode_id": mode_id,
        "name": _BALANCE_MODE_MAP[mode_id],
        "admits_loco_writes": True,
    }


@tool
def g1_list_balance_modes() -> dict[str, Any]:
    """Return the balance-mode ids ``LocoClient.BalanceStand`` admits.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``LocoClient.BalanceStand`` is called, so a caller can compare
    an intended mode against the set the neon bundle's
    ``g1_balance_stand`` verb documented as walkable, and can also
    compare the driver's live ``fsm_id`` (from
    ``G1Driver.get_status``) against ``walk_ready_fsm_ids`` to decide
    whether the locomotion write gate is currently open.

    Returns:
        A dict with ``status``; a ``modes`` list of per-mode
        descriptors sorted by ``mode_id`` ascending, each carrying
        ``mode_id``, ``name`` (the neon-observed label), and
        ``admits_loco_writes`` (always ``True``, because every
        admitted balance mode is a locomotion-shaped write by
        definition; the flag is surfaced so the descriptor shape
        matches :mod:`~strands_robots.tools.g1.g1_fsm_targets` and
        :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim);
        a ``walk_ready_fsm_ids`` list quoting
        :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, the
        set the driver's motion gate admits locomotion-shaped
        writes on; and a ``refusals`` list carrying the ``7404``
        gate-refusal code and its decoded text, the one a future
        write verb would surface. Every field is a snapshot of an
        observed mode label or a driver constant; no dynamic decode
        runs here.
    """
    return {
        "status": "success",
        "modes": [_describe(mode_id) for mode_id in sorted(_BALANCE_MODE_MAP)],
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
        "refusals": [
            {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
        ],
    }


@tool
def g1_balance_mode_admits(
    mode_id: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Decide whether a ``mode_id`` or ``name`` sits inside the admitted set.

    Read-only. Compares one argument (either ``mode_id`` or ``name``,
    not both) against the neon-observed :data:`_BALANCE_MODE_MAP` and
    reports the admitted descriptor on match, or the ``7404``
    gate-refusal code the driver would quote on miss. No driver
    instance, no DDS, no SDK: the decision reads only module-level
    constants and the arguments themselves.

    A mode inside the admitted set is *not* the same as an admitted
    write: the driver's motion gate (``_check_motion_gates``) also
    refuses on ``_fsm_id`` outside
    :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`, which this
    verb does not read (that is a live driver-instance query answered
    by :mod:`~strands_robots.tools.g1.g1_state` and
    :mod:`~strands_robots.tools.g1.g1_motion_gates`). The returned
    payload names ``walk_ready_fsm_ids`` so a caller comparing an
    intended write against both conditions has the FSM set on hand.

    Args:
        mode_id: The mode id to check (``0`` for Static or ``3`` for
            Dynamic today). Mutually exclusive with ``name``. Bool
            values (``True``/``False``) are refused with the
            ``7404`` code because Python's ``bool`` is a subclass of
            ``int`` and a caller passing ``True`` would otherwise
            look up ``1`` (unknown), returning a confusing refusal.
            A non-integer non-bool argument is refused with the same
            code for the same reason.
        name: The mode label to check (``"Static"`` or ``"Dynamic"``
            today). Mutually exclusive with ``mode_id``. The label
            comparison is case-sensitive against the snapshot in
            :data:`_BALANCE_MODE_MAP`; a mis-cased or unknown label
            is refused with the ``7404`` code.

    Returns:
        A dict with ``status``; on admit, a ``mode`` descriptor with
        ``mode_id``, ``name``, and ``admits_loco_writes`` (the same
        shape :func:`g1_list_balance_modes` returns), plus
        ``walk_ready_fsm_ids`` for the follow-on gate decision. On
        refuse, ``refusal_code`` and ``refusal_text`` name the
        ``7404`` code and its decoded text, plus a ``reason`` string
        that names why the argument was refused (unknown mode,
        unknown label, ambiguous input, or non-integer argument).
    """
    if mode_id is not None and name is not None:
        return {
            "status": "error",
            "refusal_code": _INVALID_MODE_CODE,
            "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
            "reason": ("both mode_id and name supplied; pass exactly one so the lookup is decidable"),
        }
    if mode_id is None and name is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_MODE_CODE,
            "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
            "reason": ("neither mode_id nor name supplied; pass exactly one so the lookup is decidable"),
        }

    if mode_id is not None:
        if isinstance(mode_id, bool):
            return {
                "status": "error",
                "refusal_code": _INVALID_MODE_CODE,
                "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
                "reason": (f"mode_id={mode_id!r} is a bool; pass an int (0 for Static or 3 for Dynamic)"),
            }
        if not isinstance(mode_id, int):
            return {
                "status": "error",
                "refusal_code": _INVALID_MODE_CODE,
                "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
                "reason": (f"mode_id={mode_id!r} is not an int; pass an int (0 for Static or 3 for Dynamic)"),
            }
        if mode_id not in _BALANCE_MODE_MAP:
            return {
                "status": "error",
                "refusal_code": _INVALID_MODE_CODE,
                "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
                "reason": (f"mode_id={mode_id} is not in the admitted set {sorted(_BALANCE_MODE_MAP)}"),
            }
        return {
            "status": "success",
            "mode": _describe(mode_id),
            "walk_ready_fsm_ids": sorted(WALK_FSMS),
        }

    # name is not None here
    reverse_lookup = {label: mode_id for mode_id, label in _BALANCE_MODE_MAP.items()}
    if name not in reverse_lookup:
        return {
            "status": "error",
            "refusal_code": _INVALID_MODE_CODE,
            "refusal_text": ERR_CODES[_INVALID_MODE_CODE],
            "reason": (f"name={name!r} is not in the admitted set {sorted(reverse_lookup)}"),
        }
    return {
        "status": "success",
        "mode": _describe(reverse_lookup[name]),
        "walk_ready_fsm_ids": sorted(WALK_FSMS),
    }
