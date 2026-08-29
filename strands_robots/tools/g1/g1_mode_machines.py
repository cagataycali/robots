"""Agent-facing lookup for the ``mode_machine`` ids the neon bundle observed as arm-ready.

The Unitree G1 firmware publishes a ``mode_machine`` byte on every
``rt/lowstate`` frame that names the hardware-layout id the low-level
control loop is currently running. The neon bundle observed against
the real robot that ``mode_machine`` in ``{5, 6}`` is what the
firmware publishes when the balance controller admits an arm write,
and ships that observation as ``ARM_READY_MODE_MACHINES``.

This repository's driver does **not** consult that membership. The
:class:`~strands_robots.drivers.g1.G1Driver` caches every
``LowState_.mode_machine`` on
:attr:`~strands_robots.drivers.g1.G1Driver._mode_machine`, and its
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` reads
that cache for exactly one decision: the ``is None`` liveness refusal
``"mode_machine unknown - lowstate has not delivered yet"``. The
admission decision itself is taken on ``_fsm_id``, which arrives from
the motion-switcher API wired in strands-labs/robots#2916; when that
read is unavailable the driver refuses outright with ``"FSM id
unknown - motion-switcher source has not been wired; see issue #2765
for the wire-side decision"`` rather than falling back to
``mode_machine`` membership. A future driver-side fallback that
consults the arm-ready set would read the set this module snapshots.

So this module answers a membership question about a driver-observed
contract, not a prediction of the driver's admission decision: a
caller reading the driver's
:meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope can
resolve its live ``mode_machine`` against the neon-observed arm-ready
set, and must still expect the driver's own FSM gate to decide the
write.

Two things this module is deliberately *not*:

* An execution path. The verbs read a module-level constant snapshot
  of the ``{5, 6}`` set the neon bundle's ``ARM_READY_MODE_MACHINES``
  ships (refs strands-labs/robots#358 for the SDK-facing gate work
  the write itself belongs on) and do not touch the DDS bus. The
  actual write still runs through
  :meth:`~strands_robots.drivers.g1.G1Driver.send_action` and is
  gated by the driver's ``_check_motion_gates``, which decides on
  ``_fsm_id`` and not on this set. There is no second FSM / gate code
  path here.
* An SDK re-import. The set lives here as a module-level snapshot
  rather than being re-read from the SDK, so
  ``import strands_robots.tools.g1.g1_mode_machines`` pulls zero
  ``unitree_sdk2py`` submodules - the same SDK-load-hygiene rule
  every other file under :mod:`strands_robots.tools.g1` carries.
  The mapping is a driver-observed contract (the SDK does not ship
  a canonical ``mode_machine`` id table); a firmware release that
  widens or narrows the arm-ready set is a driver-side update.

What this module does not decide.

* Whether the driver would admit an arm write. That decision is the
  driver's FSM gate (:data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS`,
  answered by :mod:`~strands_robots.tools.g1.g1_motion_gates` and
  :mod:`~strands_robots.tools.g1.g1_fsm_targets`). An arm-ready
  ``mode_machine`` is necessary-by-observation, not sufficient: the
  driver refuses on an unwired FSM read regardless of
  ``mode_machine``.
* Whether the driver's live ``_mode_machine`` is currently arm-ready.
  That is a driver-instance read carried on the driver's
  :meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope; a
  caller planning a write compares the driver's live value against
  this lookup's set. This module answers "which ``mode_machine`` ids
  did the neon bundle observe as arm-ready", not "is the robot
  currently in one".
* Whether ``mode_machine`` has been delivered at all. A driver whose
  ``_mode_machine`` is ``None`` refuses every write with
  ``"mode_machine unknown - lowstate has not delivered yet"`` before
  it reads the FSM gate; that refusal is a driver-side liveness
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
#: This repository's driver does not test membership in this set - its
#: ``_check_motion_gates`` decides on ``_fsm_id`` - so the set is a
#: reference contract a future driver-side ``mode_machine`` fallback
#: would read, not a mirror of a live driver branch. It lives
#: colocated with the verb rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because it is only
#: useful for that arm-ready membership question; a caller that needs
#: the FSM gate set reaches
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


def _not_arm_ready_refusal(mode_machine: int) -> str:
    """Build the refusal for a delivered ``mode_machine`` outside the arm-ready set.

    Kept distinct from :data:`_UNKNOWN_MODE_MACHINE_REFUSAL`: that
    string names a liveness fail whose remedy is to wait for
    ``rt/lowstate``, which is a dead end for a caller who already
    supplied a delivered byte (waiting never makes ``0`` arm-ready).
    This refusal names the knob the caller actually got wrong - the
    queried value - and the set it has to reach, so the remedy on
    offer is the one that can work.
    """
    return f"mode_machine {mode_machine} is not arm-ready; needs one of {sorted(_ARM_READY_MODE_MACHINES)}"


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
    """Return the ``mode_machine`` ids the neon bundle observed as arm-ready.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a
    :meth:`~strands_robots.drivers.g1.G1Driver.send_action` dispatch,
    so a caller reading the driver's
    :meth:`~strands_robots.drivers.g1.G1Driver.get_status` envelope
    can compare the driver's live ``mode_machine`` against the
    arm-ready set the neon bundle observed against the real robot.

    This repository's driver does not test membership in the set: its
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    decides admission on ``_fsm_id`` and reads ``mode_machine`` only
    for the ``is None`` liveness refusal. A future driver-side
    ``mode_machine`` fallback would read this set; until it lands, an
    arm-ready ``mode_machine`` is necessary-by-observation and not
    sufficient for the driver to admit the write.

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        arm-ready ``mode_machine`` ids, a ``mode_machines`` list of
        descriptors (one per admitted id, sorted ascending) carrying
        ``mode_machine`` and ``admits_arm_writes`` (always ``True``
        on this list), and a bare ``mode_machine_ids`` field with the
        set as a sorted list so a caller filtering on membership
        compares against that field directly rather than walking the
        descriptors. A ``refusal`` sub-dict carries the driver-local
        liveness refusal string ``_check_motion_gates`` quotes on a
        never-delivered ``mode_machine``; no SDK ``rc=`` code is
        surfaced because a driver-local liveness fail never reaches
        the wire. Every field is a snapshot of the neon-observed
        contract; no dynamic decode runs here.
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
    """Decide whether an observed ``mode_machine`` id is in the arm-ready set.

    Read-only. Reads the module's snapshot of the ``mode_machine`` ids
    the neon bundle observed as arm-ready and answers membership. A
    caller reading the driver's live ``mode_machine`` from
    :meth:`~strands_robots.drivers.g1.G1Driver.get_status` resolves it
    against that set before dispatching a
    :meth:`~strands_robots.drivers.g1.G1Driver.send_action`.

    ``admitted`` is a membership answer about the neon-observed
    contract, **not** a prediction of the driver's admission decision.
    This repository's
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    decides on ``_fsm_id`` and never tests this set, so
    ``admitted=True`` does not imply the driver would take the write -
    an unwired FSM read refuses it regardless. A future driver-side
    ``mode_machine`` fallback would read the same set.

    Args:
        mode_machine: The ``mode_machine`` byte to test. Must be an
            ``int``; ``bool`` is refused (``True`` is ``int(1)`` but
            a passed-through boolean is a caller mistake, not a
            valid membership query). ``None`` names the pre-lowstate
            state the driver's cached ``_mode_machine`` sits in
            before the first ``rt/lowstate`` frame lands and surfaces
            the same driver-local liveness refusal string
            :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
            quotes on that fail.

    Returns:
        A dict with ``status`` (``"success"`` on any decidable
        answer, ``"error"`` on a ``bool`` or non-``int`` non-``None``
        query), a ``query`` sub-dict carrying the supplied
        ``mode_machine``, an ``admitted`` boolean naming whether that
        ``mode_machine`` is in the neon-observed arm-ready set, and
        (when ``admitted`` is ``True``) a ``target`` sub-dict carrying
        the same descriptor
        :func:`g1_list_arm_ready_mode_machines` returns for the id
        (``mode_machine``, ``admits_arm_writes``). A not-admitted query
        carries ``refusal_text``, on one of two distinct channels: a
        ``mode_machine=None`` query gets the driver-local liveness
        string
        :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
        quotes before it reads the FSM gate (remedy: wait for
        ``rt/lowstate``), while a delivered-but-non-arm-ready query
        gets a membership refusal naming the queried value and the
        arm-ready set (remedy: reach one of those ids). The two are
        kept separate so the remedy the text implies is always the one
        that can work. Unlike
        :mod:`~strands_robots.tools.g1.g1_fsm_targets` neither refusal
        carries an ``rc=`` code: neither the liveness fail nor a
        membership miss reaches the wire.
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
        # the FSM gate is consulted. Surface the same string here so a
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
        # Delivered but non-arm-ready. The liveness string does not
        # apply here - the caller supplied a real byte, so "wait for
        # lowstate" is a dead end - and the driver's own refusal on
        # this input names its FSM gate, not the mode_machine value.
        # Name the queried value and the set it must reach instead.
        result["refusal_text"] = _not_arm_ready_refusal(mode_machine)
    return result
