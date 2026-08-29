"""The FSM-target lookup tools name exactly what ``SetFsmId`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) admits FSM
transition targets by integer id from a fixed table; the
:mod:`strands_robots.tools.g1.g1_fsm_targets` module snapshots that
table into a module-level constant and exposes two agent-facing verbs -
:func:`g1_list_fsm_targets` (list the whole set) and
:func:`g1_fsm_target_admits` (decide one query) - so a caller can decide
the SDK's ``rc=7302`` refusal decidably before a future transition path
is attempted. The tests here fix that contract without pulling the SDK:
the module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the constant
surfaces here as a shape change rather than as a diverging table this
file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  transition table (the whole point of the port is that the snapshot
  lets a headless host answer). A driver-side wrapper for
  ``SetFsmId`` that lands later will re-validate against the SDK's
  live table at wire time; testing the snapshot vs the live table is
  a driver-side test, not a lookup-side one.
* Which FSM ids the arm-SDK / locomotion write gates admit on. The
  verb surfaces :data:`HANDSHAKE_FSMS` / :data:`WALK_FSMS` verbatim
  because a caller planning a transition compares the target against
  the write gate too; the membership rule for both gates is already
  pinned in
  :mod:`tests.drivers.test_g1_motion_gates_reads_the_driver_contract`,
  so this file only checks that the surfaced sets match what the
  driver's constants ship.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
)
from strands_robots.tools.g1.g1_fsm_targets import (
    _DANGEROUS_FSM_IDS,
    _FSM_NAME_MAP,
    _GATE_REFUSAL_CODE,
    _INVALID_FSM_CODE,
    g1_fsm_target_admits,
    g1_list_fsm_targets,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim. This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the FSM-target lookup
    verbs to it too (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_fsm_targets")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_fsm_targets imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_sdk_shipped_set() -> None:
    """The snapshot names every FSM the SDK's ``SetFsmId`` admits today.

    The SDK's own transition table has 10 entries (the pre-programmed
    FSMs including ``0`` ZeroTorque, ``1`` Damp, ``500`` Start,
    ``501`` Walk, ``801`` BalanceExpert, ...). A drift on either side
    surfaces here: a driver-side transition wrapper (when it lands)
    will validate the same set at wire time and its refusal string
    will quote ``rc=7302`` on any id outside it. The count is pinned
    rather than listed name-by-name so a caller widening the map on
    the driver side updates one number here rather than 10
    assertions.
    """
    assert len(_FSM_NAME_MAP) == 10


def test_the_snapshot_carries_the_gate_ready_targets() -> None:
    """Every arm-SDK / loco write-gate id is a transition target too.

    The driver's :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS` name the
    FSMs where the arm-SDK and locomotion write paths accept a joint
    payload. Each of those ids must also be reachable as a transition
    target - otherwise a caller could see a live ``fsm_id`` outside
    the write gate but have no admitted ``SetFsmId`` call to reach
    a gate-open state. This cell pins the invariant so a widen of
    the write gate that outgrows the transition table surfaces here.
    """
    for fsm_id in HANDSHAKE_FSMS:
        assert fsm_id in _FSM_NAME_MAP, (
            f"arm-SDK gate id {fsm_id} is in HANDSHAKE_FSMS but not in the "
            f"transition-target snapshot. A caller reaching for that gate "
            f"has no admitted SetFsmId path to it."
        )
    for fsm_id in WALK_FSMS:
        assert fsm_id in _FSM_NAME_MAP, (
            f"locomotion gate id {fsm_id} is in WALK_FSMS but not in the "
            f"transition-target snapshot. Refs strands-labs/robots#358."
        )


def test_the_snapshot_flags_the_off_gantry_targets() -> None:
    """The two safety-flagged targets are ``0`` (ZeroTorque) and ``1`` (Damp).

    The neon bundle's ``g1_set_fsm`` docstring calls out ZeroTorque and
    Damp as gantry-only: ``0`` fully limps every joint and ``1`` leaves
    gravity doing the work with only soft limits. This cell pins the
    pair so an off-gantry filter that checks against the flag matches
    the neon-bundle warning verbatim, and so a widen of the danger set
    lands here first.
    """
    assert _DANGEROUS_FSM_IDS == frozenset({0, 1})
    for fsm_id in _DANGEROUS_FSM_IDS:
        assert fsm_id in _FSM_NAME_MAP, (
            f"dangerous id {fsm_id} is flagged but not in the transition "
            f"snapshot; the flag can only apply to admitted targets."
        )


def test_g1_list_fsm_targets_returns_the_whole_table() -> None:
    """The verb's payload names the map, the ids, and the SDK refusals.

    ``count`` is the size of the module's own snapshot, ``fsm_targets``
    is one descriptor per admitted id (sorted ascending), ``fsm_ids``
    is the sorted id list alone, ``dangerous_ids`` names the gantry-
    only pair, ``arm_ready_fsm_ids`` / ``loco_ready_fsm_ids`` mirror
    the driver's write-gate sets, and ``refusals`` names the two
    refusal codes (``7302`` invalid transition id, ``7404`` gate-
    refused write) with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_fsm_targets)
    assert result["status"] == "success"
    assert result["count"] == len(_FSM_NAME_MAP)
    assert result["fsm_ids"] == sorted(_FSM_NAME_MAP)
    assert result["dangerous_ids"] == sorted(_DANGEROUS_FSM_IDS)
    assert result["arm_ready_fsm_ids"] == sorted(HANDSHAKE_FSMS)
    assert result["loco_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert len(result["fsm_targets"]) == len(_FSM_NAME_MAP)
    # Every descriptor carries the same field set and reads its flags
    # from the module's constants (not restated in the test body).
    for descriptor in result["fsm_targets"]:
        fsm_id = descriptor["fsm_id"]
        assert descriptor["name"] == _FSM_NAME_MAP[fsm_id]
        assert descriptor["dangerous"] is (fsm_id in _DANGEROUS_FSM_IDS)
        assert descriptor["admits_arm_writes"] is (fsm_id in HANDSHAKE_FSMS)
        assert descriptor["admits_loco_writes"] is (fsm_id in WALK_FSMS)
    codes = {r["code"] for r in result["refusals"]}
    assert codes == {_INVALID_FSM_CODE, _GATE_REFUSAL_CODE}
    for refusal in result["refusals"]:
        assert refusal["text"] == ERR_CODES[refusal["code"]]


def test_g1_list_fsm_targets_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh lists and dicts; a mutation on the returned
    ``fsm_ids`` list or ``fsm_targets`` descriptors does not leak back
    into the module's constants. This cell is where a share-a-reference
    regression would surface once, not scattered across every call
    site (same guarantee as the ``action_map`` snapshot in
    :mod:`strands_robots.tools.g1.g1_arm_actions`).
    """
    result = _call(g1_list_fsm_targets)
    result["fsm_ids"].append(9999)
    result["fsm_targets"][0]["synthetic"] = True
    fresh = _call(g1_list_fsm_targets)
    assert 9999 not in fresh["fsm_ids"]
    assert "synthetic" not in fresh["fsm_targets"][0]


def test_g1_fsm_target_admits_resolves_a_valid_id() -> None:
    """An id inside the SDK's set is admitted and the descriptor lands.

    ``500`` is Start; the verb reports ``admitted=True`` and carries
    the resolved descriptor (``fsm_id``, ``name``, ``dangerous``, gate
    flags) a future transition verb would use to decide the follow-up
    write path. No refusal fields fire on the admitted path.
    """
    result = _call(g1_fsm_target_admits, fsm_id=500)
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"fsm_id": 500}
    assert result["target"]["fsm_id"] == 500
    assert result["target"]["name"] == "Start"
    assert result["target"]["dangerous"] is False
    assert result["target"]["admits_arm_writes"] is True
    assert result["target"]["admits_loco_writes"] is False
    assert "refusal_code" not in result


def test_g1_fsm_target_admits_resolves_a_valid_name() -> None:
    """A name in the snapshot resolves to the SDK's id.

    ``"Walk"`` is id ``501``; the verb reports ``admitted=True`` and
    the descriptor names ``501`` on both write gates (Walk is inside
    both HANDSHAKE_FSMS and WALK_FSMS by driver contract).
    """
    result = _call(g1_fsm_target_admits, name="Walk")
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"name": "Walk"}
    assert result["target"]["fsm_id"] == 501
    assert result["target"]["name"] == "Walk"
    assert result["target"]["admits_loco_writes"] is True


def test_g1_fsm_target_admits_flags_a_dangerous_target() -> None:
    """An admitted-but-dangerous id carries ``dangerous=True``.

    ``0`` (ZeroTorque) is a valid SDK transition target - the SDK
    admits ``SetFsmId(0)`` - but the robot collapses off-gantry. The
    verb still reports ``admitted=True`` (it is a lookup, not a
    policy) but the descriptor's ``dangerous`` flag lets a caller
    decide the off-gantry refusal decidably before dispatch.
    """
    result = _call(g1_fsm_target_admits, fsm_id=0)
    assert result["admitted"] is True
    assert result["target"]["dangerous"] is True
    assert result["target"]["name"] == "ZeroTorque"


def test_g1_fsm_target_admits_refuses_an_unknown_id() -> None:
    """An id outside the SDK's set is refused with ``rc=7302``.

    ``42`` is not a G1 FSM. The verb reports ``admitted=False`` and
    carries the SDK's own refusal code (``7302``) with the decoded
    text a future driver-side wrapper would surface, so the two
    sides quote the same error verbatim.
    """
    result = _call(g1_fsm_target_admits, fsm_id=42)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"fsm_id": 42}
    assert result["refusal_code"] == _INVALID_FSM_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_FSM_CODE]
    assert "target" not in result


def test_g1_fsm_target_admits_refuses_an_unknown_name() -> None:
    """A name not in the snapshot is refused with the same ``rc=7302``.

    ``"walk"`` (lower-case) is not in the snapshot; the snapshot ships
    ``"Walk"``. The verb mirrors the SDK's dict semantics (a caller
    writing ``arm.ExecuteAction`` against a mis-cased name would see
    the same ``rc=7402`` at wire time on the arm side).
    """
    result = _call(g1_fsm_target_admits, name="walk")
    assert result["admitted"] is False
    assert result["query"] == {"name": "walk"}
    assert result["refusal_code"] == _INVALID_FSM_CODE


def test_g1_fsm_target_admits_refuses_both_supplied() -> None:
    """Supplying both ``fsm_id`` and ``name`` is a caller mistake.

    The verb refuses with ``status="error"`` rather than picking a
    resolution: the ambiguous case is not one the lookup should
    resolve arbitrarily.
    """
    result = _call(g1_fsm_target_admits, fsm_id=500, name="Start")
    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_fsm_target_admits_refuses_neither_supplied() -> None:
    """Supplying neither ``fsm_id`` nor ``name`` is a caller mistake.

    Same ambiguous-case rule as the both-supplied refusal: the caller
    has to say what they are asking about.
    """
    result = _call(g1_fsm_target_admits)
    assert result["status"] == "error"
    assert "exactly one" in result["message"]


def test_g1_fsm_target_admits_refuses_a_bool_fsm_id() -> None:
    """``bool`` is not an ``int`` this verb should silently accept.

    Python's ``bool`` is a subclass of ``int`` so ``True == 1``, but a
    caller passing ``True`` for an FSM id is a type mistake, not a
    valid query. The verb refuses so a mis-typed argument surfaces at
    the lookup rather than reaching the SDK's own type coercion.
    """
    result = _call(g1_fsm_target_admits, fsm_id=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_g1_fsm_target_admits_refuses_a_non_int_fsm_id() -> None:
    """A non-int, non-bool ``fsm_id`` surfaces the type in the refusal.

    ``"500"`` looks correct to a human reader but is a string; the
    refusal names the type and the value the caller passed, so a
    caller sees which of their many parallel tool calls hit the
    wrong shape.
    """
    result = _call(g1_fsm_target_admits, fsm_id="500")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
