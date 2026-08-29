"""The ``mode_machine`` lookup tools name exactly what the driver treats as arm-ready.

The Unitree G1 firmware publishes a ``mode_machine`` byte on every
``rt/lowstate`` frame; the neon bundle observed against the real
robot that :data:`~strands_robots.tools.g1.g1_mode_machines._ARM_READY_MODE_MACHINES`
(``{5, 6}``) is the second source of truth the driver's arm-write
path uses when the loco-SDK ``GetFsmId`` RPC is wedged (returns
``rc=3104``) but the robot is physically arm-ready. The
:mod:`strands_robots.tools.g1.g1_mode_machines` module snapshots that
set into a module-level constant and exposes two agent-facing verbs -
:func:`g1_list_arm_ready_mode_machines` (list the whole set) and
:func:`g1_mode_machine_admits_arm` (decide one query) - so a caller
reading the driver's :meth:`~strands_robots.drivers.g1.G1Driver.get_status`
envelope can decide the arm-ready refusal decidably before
dispatching a :meth:`~strands_robots.drivers.g1.G1Driver.send_action`.
The tests here fix that contract without pulling the SDK: the module
is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
constant surfaces here as a shape change rather than as a diverging
table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The firmware's own answer at wire time. The verbs answer against
  the module-level snapshot, not against a live ``rt/lowstate``
  ``mode_machine`` byte (the whole point of the port is that the
  snapshot lets a headless host answer). The driver's
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  re-validates against its own cached ``_mode_machine`` at write
  time; testing the snapshot vs the live frame is a driver-side
  test, not a lookup-side one.
* Which FSM ids the arm-SDK write gate admits on. That is the
  first-of-two sources
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
  consults; its membership rule is
  :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` and its
  refusal is answered by
  :mod:`~strands_robots.tools.g1.g1_fsm_targets`. This file pins the
  second-of-two path (``mode_machine``) independently, so a caller
  seeing the driver's refusal knows which source named it.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_mode_machines import (
    _ARM_READY_MODE_MACHINES,
    _UNKNOWN_MODE_MACHINE_REFUSAL,
    g1_list_arm_ready_mode_machines,
    g1_mode_machine_admits_arm,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the
    ``mode_machine`` lookup verbs to it too
    (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_mode_machines")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_mode_machines imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_matches_the_neon_observed_set() -> None:
    """The snapshot names exactly the ids the neon bundle observed as arm-ready.

    The neon bundle's ``ARM_READY_MODE_MACHINES = {5, 6}`` is the
    driver-observed contract this port snapshots. A drift on either
    side surfaces here: a firmware release that widens or narrows
    the set is a driver-side update, and when the driver's
    ``_check_motion_gates`` learns a new id the constant here has to
    move too. The set is pinned by value rather than by count so a
    swap (e.g. ``{6, 7}``) surfaces here rather than passing a size
    check.
    """
    assert _ARM_READY_MODE_MACHINES == frozenset({5, 6})


def test_the_snapshot_is_a_frozenset() -> None:
    """The constant cannot be mutated by an in-process caller.

    A caller reaching for the module's snapshot to filter its own
    state must not be able to poison the module state by a
    ``_ARM_READY_MODE_MACHINES.add(...)``. This cell pins the
    frozen shape so a swap to a plain ``set`` surfaces here as a
    typing regression rather than as a mutation slipped in
    downstream.
    """
    assert isinstance(_ARM_READY_MODE_MACHINES, frozenset)


def test_the_refusal_string_names_the_driver_side_text() -> None:
    """The refusal quotes the driver's own ``_check_motion_gates`` string.

    The driver refuses every write with
    ``"mode_machine unknown - lowstate has not delivered yet"`` when
    its cached ``_mode_machine`` is ``None`` before either gate is
    consulted. The lookup surfaces the same text verbatim so a
    caller sees the exact refusal a follow-up ``send_action`` would
    carry (refs strands-labs/robots#358).
    """
    assert _UNKNOWN_MODE_MACHINE_REFUSAL == ("mode_machine unknown - lowstate has not delivered yet")


def test_g1_list_arm_ready_mode_machines_returns_the_whole_set() -> None:
    """The verb's payload names the count, the descriptors, the id list, the refusal.

    ``count`` is the size of the module's own snapshot,
    ``mode_machines`` is one descriptor per admitted id (sorted
    ascending), ``mode_machine_ids`` is the sorted id list alone
    (the field a caller filtering on membership reaches), and
    ``refusal`` carries the driver-local liveness string the
    driver's ``_check_motion_gates`` quotes on a never-delivered
    ``mode_machine``. Unlike
    :mod:`~strands_robots.tools.g1.g1_fsm_targets`, no SDK ``rc=``
    code is surfaced because the driver's ``mode_machine`` refusal
    is a local liveness check that never reaches the wire.
    """
    result = _call(g1_list_arm_ready_mode_machines)
    assert result["status"] == "success"
    assert result["count"] == len(_ARM_READY_MODE_MACHINES)
    assert result["mode_machine_ids"] == sorted(_ARM_READY_MODE_MACHINES)
    assert len(result["mode_machines"]) == len(_ARM_READY_MODE_MACHINES)
    # Every descriptor carries the same field set and reads its
    # membership flag from the module's constant (not restated in
    # the test body).
    for descriptor in result["mode_machines"]:
        mm = descriptor["mode_machine"]
        assert descriptor["admits_arm_writes"] is (mm in _ARM_READY_MODE_MACHINES)
        assert descriptor["admits_arm_writes"] is True  # every listed id is arm-ready
    assert result["refusal"]["text"] == _UNKNOWN_MODE_MACHINE_REFUSAL


def test_g1_list_arm_ready_mode_machines_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh lists and dicts; a mutation on the
    returned ``mode_machine_ids`` list or ``mode_machines``
    descriptors does not leak back into the module's constants. Same
    guarantee as the ``fsm_ids`` snapshot in
    :mod:`~strands_robots.tools.g1.g1_fsm_targets`: one snapshot per
    lookup, one shared-reference regression it would catch.
    """
    result = _call(g1_list_arm_ready_mode_machines)
    result["mode_machine_ids"].append(9999)
    result["mode_machines"][0]["synthetic"] = True
    fresh = _call(g1_list_arm_ready_mode_machines)
    assert 9999 not in fresh["mode_machine_ids"]
    assert "synthetic" not in fresh["mode_machines"][0]


def test_g1_mode_machine_admits_arm_admits_a_ready_value() -> None:
    """A ``mode_machine`` inside the arm-ready set is admitted.

    ``5`` is one of the two ids the neon bundle observed as
    arm-ready; the verb reports ``admitted=True`` and carries the
    resolved descriptor a caller planning a ``send_action`` would
    use to decide the follow-up write path. No refusal fields fire
    on the admitted path.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=5)
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"mode_machine": 5}
    assert result["target"]["mode_machine"] == 5
    assert result["target"]["admits_arm_writes"] is True
    assert "refusal_text" not in result


def test_g1_mode_machine_admits_arm_admits_the_second_ready_value() -> None:
    """``6`` is the second observed arm-ready id.

    Pinning both admitted values separately (rather than
    parameterising) makes a narrow to a single-id set surface here
    as a failing test rather than a passing one against a partial
    check.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=6)
    assert result["admitted"] is True
    assert result["target"]["mode_machine"] == 6


def test_g1_mode_machine_admits_arm_refuses_a_non_ready_value() -> None:
    """A ``mode_machine`` outside the arm-ready set is refused.

    ``0`` is not in ``{5, 6}``; the verb reports ``admitted=False``
    and carries the driver-local liveness refusal string
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    quotes on the ``mode_machine`` fallback branch. The refusal is
    the same text used for the ``None`` liveness case because both
    branches surface a single, consistent refusal channel for any
    ``mode_machine`` outside the arm-ready set.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=0)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"mode_machine": 0}
    assert result["refusal_text"] == _UNKNOWN_MODE_MACHINE_REFUSAL
    assert "target" not in result


def test_g1_mode_machine_admits_arm_refuses_a_none_liveness_query() -> None:
    """``mode_machine=None`` names the pre-lowstate driver state.

    Before the first ``rt/lowstate`` frame lands the driver's cached
    ``_mode_machine`` is ``None``; its ``_check_motion_gates``
    refuses every write on that branch with the liveness string
    before it reads either gate. The lookup surfaces the same
    refusal so a caller polling before the first frame lands sees
    the exact text a follow-up ``send_action`` would carry.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=None)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"mode_machine": None}
    assert result["refusal_text"] == _UNKNOWN_MODE_MACHINE_REFUSAL
    assert "target" not in result


def test_g1_mode_machine_admits_arm_defaults_to_the_none_query() -> None:
    """Calling the verb with no args resolves to the ``None`` liveness path.

    A caller with no ``mode_machine`` to hand still gets a decidable
    answer - the same one they would see reading the driver's live
    ``mode_machine`` before ``rt/lowstate`` has delivered. This cell
    pins the default so a swap of the signature (say, making the
    argument required) surfaces here rather than as a caller-side
    ``TypeError`` at a downstream call site.
    """
    result = _call(g1_mode_machine_admits_arm)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"mode_machine": None}
    assert result["refusal_text"] == _UNKNOWN_MODE_MACHINE_REFUSAL


def test_g1_mode_machine_admits_arm_refuses_a_bool_query() -> None:
    """``bool`` is not an ``int`` this verb should silently accept.

    Python's ``bool`` is a subclass of ``int`` so ``True == 1``, but
    a caller passing ``True`` for a ``mode_machine`` byte is a type
    mistake, not a valid query. The verb refuses so a mis-typed
    argument surfaces at the lookup rather than reaching the
    driver's own type coercion.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_mode_machine_admits_arm_refuses_a_bool_false_query() -> None:
    """``False`` is the second ``bool`` value the same rule refuses.

    Pinning both bool values separately (rather than parameterising)
    makes a shape change to the guard - say, letting ``False``
    through because it evaluates to ``0`` - surface here as a
    failing test rather than a passing one against a partial check.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine=False)
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_g1_mode_machine_admits_arm_refuses_a_non_int_query() -> None:
    """A non-int, non-bool, non-``None`` ``mode_machine`` names the type in the refusal.

    ``"5"`` looks correct to a human reader but is a string; the
    refusal names the type and the value the caller passed, so a
    caller sees which of their many parallel tool calls hit the
    wrong shape.
    """
    result = _call(g1_mode_machine_admits_arm, mode_machine="5")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
