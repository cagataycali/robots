"""Agent-facing lookup for the ``mode_machine`` ids the G1 driver treats as arm-ready.

The Unitree G1 firmware publishes a ``mode_machine`` byte on every
``rt/lowstate`` frame that names the hardware-layout id the low-level
control loop is currently running. The neon bundle observed against
the real robot that ``mode_machine`` in ``{5, 6}`` is the second
source of truth the driver's arm-write path uses when the
:class:`~unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`
``GetFsmId`` RPC is wedged (returns ``rc=3104``) but the robot is
physically arm-ready. The
:class:`~strands_robots.drivers.g1.G1Driver` caches every
``LowState_.mode_machine`` on :attr:`~strands_robots.drivers.g1.G1Driver._mode_machine`;
its :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refuses the write with ``"mode_machine unknown - lowstate has not
delivered yet"`` when the cache is ``None`` before any FSM read is
attempted.

This module surfaces the arm-ready ``mode_machine`` id set to an
agent so a caller reading the driver's
:meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope can
decide the arm-ready refusal decidably before dispatching a
:meth:`~strands_robots.drivers.g1.G1Driver.send_action` that would
otherwise reach the driver's own refusal path.

Two things this module is deliberately *not*:

* An execution path. The verbs read a module-level constant snapshot
  of the ``{5, 6}`` set the neon bundle's ``ARM_READY_MODE_MACHINES``
  ships (refs strands-labs/robots#358 for the SDK-facing gate work
  the write itself belongs on) and do not touch the DDS bus. A
  caller planning a ``send_action`` compares the driver's live
  ``mode_machine`` against this set before dispatch; the actual
  write still runs through
  :meth:`~strands_robots.drivers.g1.G1Driver.send_action` and is
  gated by the driver's ``_check_motion_gates`` which is wired in
  strands-labs/robots#2916. There is no second FSM / gate code path
  here.
* An SDK re-import. The set lives here as a module-level snapshot
  rather than being re-read from the SDK, so
  ``import strands_robots.tools.g1.g1_mode_machines`` pulls zero
  ``unitree_sdk2py`` submodules - the same SDK-load-hygiene rule
  every other file under :mod:`strands_robots.tools.g1` carries.
  The mapping is a driver-observed contract (the SDK does not ship
  a canonical ``mode_machine`` id table); a firmware release that
  widens or narrows the arm-ready set is a driver-side update, and
  when it lands the driver's refusal will name the same field this
  lookup returns.

What this module does not decide.

* Whether the driver's live ``_mode_machine`` is currently arm-ready.
  That is a driver-instance read carried on the driver's
  :meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope; a
  caller planning a write compares the driver's live value against
  this lookup's set. This module answers "which ``mode_machine`` ids
  admit an arm write at all", not "is the robot currently in one".
* Whether the FSM currently admits a write. That is
  :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`, answered
  by :mod:`~strands_robots.tools.g1.g1_motion_gates` and
  :mod:`~strands_robots.tools.g1.g1_fsm_targets`. The FSM gate and
  the ``mode_machine`` fallback are the two independent sources of
  truth the driver's ``_check_motion_gates`` consults; this lookup
  is the second-of-two, kept structurally separate so a caller can
  see which source the driver's refusal named.
* Whether ``mode_machine`` has been delivered at all. A driver whose
  ``_mode_machine`` is ``None`` refuses every write with
  ``"mode_machine unknown - lowstate has not delivered yet"`` before
  it reads either gate; that refusal is a driver-side liveness
  check, not a membership question this lookup answers.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the ``mode_machine`` ids the G1 firmware publishes when
#: the robot is in an arm-ready hardware layout. Observed by the neon
#: bundle against the real robot and surfaced by
#: :data:`~strands_robots.tools.g1._g1_common.ARM_READY_MODE_MACHINES`
#: in the neon-side helper (that module doesn't ship in this package;
#: the constant is the same set, snapshotted here). Named as a
#: ``frozenset`` so a caller cannot mutate the module state by
#: mistake.
#:
#: The set lives colocated with the verb rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because it is only
#: useful for the arm-write fallback branch the driver's
#: ``_check_motion_gates`` runs when the loco-SDK ``GetFsmId`` RPC is
#: unavailable; a caller that needs the FSM gate set reaches
#: :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`
#: directly. Mirrors :data:`~strands_robots.tools.g1.g1_fsm_targets._FSM_NAME_MAP`:
#: one snapshot per SDK- or driver-facing table, one verb pair per
#: snapshot.
_ARM_READY_MODE_MACHINES: frozenset[int] = frozenset({5, 6})

#: The driver-local refusal
#: :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
#: quotes when its cached ``_mode_machine`` is ``None`` (the
#: ``rt/lowstate`` subscriber has not delivered a frame yet). Named
#: here so :func:`g1_mode_machine_admits_arm`'s refusal path on a
#: ``mode_machine=None`` query surfaces the same string a caller
#: would see on a follow-up
#: :meth:`~strands_robots.drivers.g1.G1Driver.send_action`. Unlike
#: :mod:`~strands_robots.tools.g1.g1_fsm_targets`, this lookup's
#: refusal does *not* come from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` - the
#: SDK-side error table quotes ``rc=`` codes on wire refusals, and a
#: never-delivered ``mode_machine`` is a driver-local liveness fail
#: that never reaches the wire.
_UNKNOWN_MODE_MACHINE_REFUSAL: str = "mode_machine unknown - lowstate has not delivered yet"


def _describe(mode_machine: int) -> dict[str, Any]:
    """Build the per-id descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_arm_ready_mode_machines`
    so :func:`g1_mode_machine_admits_arm`'s admitted-path payload names
    the same fields, and so a widen to the descriptor lands in one
    place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "mode_machine": mode_machine,
        "admits_arm_writes": mode_machine in _ARM_READY_MODE_MACHINES,
    }


@tool
def g1_list_arm_ready_mode_machines() -> dict[str, Any]:
    """Return the ``mode_machine`` ids the driver treats as arm-ready.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a
    :meth:`~strands_robots.drivers.g1.G1Driver.send_action` dispatch,
    so a caller reading the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope
    can compare the driver's live ``mode_machine`` against the
    arm-ready set the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    tests membership in when the loco-SDK ``GetFsmId`` RPC is wedged
    (returns ``rc=3104``). The FSM gate is the driver's first
    source of truth; this ``mode_machine`` set is the second source
    the driver falls back to on an unavailable FSM read.

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        arm-ready ``mode_machine`` ids, a ``mode_machines`` list of
        descriptors (one per admitted id, sorted ascending) carrying
        ``mode_machine`` and ``admits_arm_writes`` (always ``True``
        on this list), and a bare ``mode_machine_ids`` field with the
        set as a sorted list so a caller filtering on membership
        compares against that field directly rather than walking the
        descriptors. A ``refusal`` sub-dict carries the driver-local
        refusal string ``_check_motion_gates`` quotes on a
        never-delivered ``mode_machine`` (the second-of-two refusals
        a caller may hit alongside the FSM-gate ``rc=7404`` from
        :mod:`~strands_robots.tools.g1.g1_fsm_targets`); no SDK
        ``rc=`` code is surfaced because a driver-local liveness fail
        never reaches the wire. Every field is a snapshot of the
        neon-observed contract; no dynamic decode runs here.
    """
    mode_machines = sorted(_ARM_READY_MODE_MACHINES)
    return {
        "status": "success",
        "count": len(_ARM_READY_MODE_MACHINES),
        "mode_machines": [_describe(mm) for mm in mode_machines],
        "mode_machine_ids": mode_machines,
        "refusal": {"text": _UNKNOWN_MODE_MACHINE_REFUSAL},
    }


@tool
def g1_mode_machine_admits_arm(
    mode_machine: int | None = None,
) -> dict[str, Any]:
    """Decide whether an observed ``mode_machine`` id admits an arm write.

    Read-only. Reads the module's snapshot of the driver's arm-ready
    ``mode_machine`` set and returns the same membership answer the
    driver's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    fallback branch computes when its FSM read is wedged. A caller
    reading the driver's live ``mode_machine`` from
    :meth:`~strands_robots.drivers.g1.G1Driver.get_status` resolves
    it against the arm-ready set before dispatching a
    :meth:`~strands_robots.drivers.g1.G1Driver.send_action`, rather
    than triggering the driver's local refusal at write time.

    Args:
        mode_machine: The ``mode_machine`` byte to test. Must be an
            ``int``; ``bool`` is refused (``True`` is ``int(1)`` but
            a passed-through boolean is a caller mistake, not a
            valid membership query). ``None`` names the pre-lowstate
            state the driver's cached ``_mode_machine`` sits in
            before the first ``rt/lowstate`` frame lands and surfaces
            the same driver-local refusal string
            :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
            quotes on that liveness fail.

    Returns:
        A dict with ``status`` (``"success"`` on any decidable
        answer, ``"error"`` on a ``bool`` or non-``int`` non-``None``
        query), a ``query`` sub-dict carrying the supplied
        ``mode_machine``, an ``admitted`` boolean naming whether the
        driver's fallback branch would admit an arm write on that
        ``mode_machine``, and (when ``admitted`` is ``True``) a
        ``target`` sub-dict carrying the same descriptor
        :func:`g1_list_arm_ready_mode_machines` returns for the id
        (``mode_machine``, ``admits_arm_writes``). On a not-admitted
        query the dict carries ``refusal_text`` naming the
        driver-local refusal string
        :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
        would quote; on the ``mode_machine=None`` liveness query the
        same refusal string surfaces because the driver quotes it on
        that branch before it reads either gate. Unlike
        :mod:`~strands_robots.tools.g1.g1_fsm_targets` the refusal
        does not carry an ``rc=`` code: the driver's ``mode_machine``
        refusal is a local liveness check that never reaches the wire.
    """
    if isinstance(mode_machine, bool):
        return {
            "status": "error",
            "message": (
                f"mode_machine must be int or None, got bool ({mode_machine!r}). Refs strands-labs/robots#358."
            ),
        }
    if mode_machine is not None and not isinstance(mode_machine, int):
        return {
            "status": "error",
            "message": (
                f"mode_machine must be int or None, got {type(mode_machine).__name__} "
                f"({mode_machine!r}). Refs strands-labs/robots#358."
            ),
        }

    query: dict[str, Any] = {"mode_machine": mode_machine}
    if mode_machine is None:
        # Pre-lowstate: driver refuses with the liveness message before
        # either gate is consulted. Surface the same string here so a
        # caller polling before the first frame lands sees the exact
        # refusal a follow-up send_action would carry.
        return {
            "status": "success",
            "query": query,
            "admitted": False,
            "refusal_text": _UNKNOWN_MODE_MACHINE_REFUSAL,
        }

    admitted = mode_machine in _ARM_READY_MODE_MACHINES
    result: dict[str, Any] = {
        "status": "success",
        "query": query,
        "admitted": admitted,
    }
    if admitted:
        result["target"] = _describe(mode_machine)
    else:
        # Non-arm-ready mode_machine: the driver's ``_check_motion_gates``
        # reaches this branch only when the FSM read is also wedged;
        # its refusal names the FSM gate (rc=7404) at that point rather
        # than the mode_machine value. This lookup answers the
        # membership question independently, and the returned refusal
        # is the same driver-local liveness string used for None so a
        # caller sees a single, consistent refusal channel for any
        # ``mode_machine`` outside the arm-ready set.
        result["refusal_text"] = _UNKNOWN_MODE_MACHINE_REFUSAL
    return result
