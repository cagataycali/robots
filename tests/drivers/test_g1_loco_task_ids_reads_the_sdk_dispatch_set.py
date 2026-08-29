"""The loco-task lookup tools name exactly what ``SetTaskId`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) admits
task dispatch ids by integer from a fixed table; the
:mod:`strands_robots.tools.g1.g1_loco_task_ids` module snapshots that
table into a module-level constant and exposes two agent-facing verbs -
:func:`g1_list_loco_tasks` (list the whole set) and
:func:`g1_loco_task_admits` (decide one query) - so a caller can decide
the SDK's ``rc=7303`` refusal decidably before a future dispatch path
is attempted. The tests here fix that contract without pulling the
SDK: the module is loadable on a host without ``unitree_sdk2py`` (the
same SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
constant surfaces here as a shape change rather than as a diverging
table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  dispatch table (the whole point of the port is that the snapshot
  lets a headless host answer). A driver-side wrapper for
  ``SetTaskId`` that lands later will re-validate against the SDK's
  live table at wire time; testing the snapshot vs the live table is
  a driver-side test, not a lookup-side one.
* Which FSM ids the locomotion write gate admits on. The verb
  surfaces :data:`WALK_FSMS` verbatim because a caller planning a
  dispatch compares the target against the write gate too; the
  membership rule for that gate is already pinned in
  :mod:`tests.drivers.test_g1_motion_gates_reads_the_driver_contract`,
  so this file only checks that the surfaced set matches what the
  driver's constant ships.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_loco_task_ids import (
    _GATE_REFUSAL_CODE,
    _INVALID_TASK_CODE,
    _LOCO_TASK_MAP,
    _SEQUENCED_TASK_IDS,
    g1_list_loco_tasks,
    g1_loco_task_admits,
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
    path that loads the SDK); this cell holds the loco-task lookup
    verbs to it too (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_loco_task_ids")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_loco_task_ids imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_sdk_shipped_set() -> None:
    """The snapshot names every task the SDK's ``SetTaskId`` admits today.

    The SDK's own dispatch table has 4 entries observed against the
    real robot (``0`` WaveHand-no-turn, ``1`` WaveHand-with-turn,
    ``2`` ShakeHand-reach, ``3`` ShakeHand-shake). A drift on either
    side surfaces here: a driver-side dispatch wrapper (when it
    lands) will validate the same set at wire time and its refusal
    string will quote ``rc=7303`` on any id outside it. The count is
    pinned rather than listed name-by-name so a caller widening the
    map on the driver side updates one number here rather than 4
    assertions.
    """
    assert len(_LOCO_TASK_MAP) == 4


def test_the_snapshot_flags_the_sequenced_task_ids() -> None:
    """The two sequenced task ids are ``2`` (reach) and ``3`` (shake).

    The neon bundle's ``g1_shake_hand_loco`` wrapper drove tasks ``2``
    then ``3`` as the two stages of a single shake-hand gesture;
    dispatching ``3`` alone leaves the robot mid-gesture with no
    reach-out preamble. This cell pins the pair so a one-shot-task
    filter that checks against the flag matches the neon-bundle
    behaviour verbatim, and so a widen of the sequenced set lands
    here first.
    """
    assert _SEQUENCED_TASK_IDS == frozenset({2, 3})
    for task_id in _SEQUENCED_TASK_IDS:
        assert task_id in _LOCO_TASK_MAP, (
            f"sequenced id {task_id} is flagged but not in the dispatch "
            f"snapshot; the flag can only apply to admitted tasks."
        )


def test_g1_list_loco_tasks_returns_the_whole_table() -> None:
    """The verb's payload names the map, the ids, and the SDK refusals.

    ``count`` is the size of the module's own snapshot, ``loco_tasks``
    is one descriptor per admitted id (sorted ascending), ``task_ids``
    is the sorted id list alone, ``sequenced_ids`` names the
    stage-only pair, ``loco_ready_fsm_ids`` mirrors the driver's
    write-gate set, and ``refusals`` names the two refusal codes
    (``7303`` invalid task id, ``7404`` gate-refused write) with the
    decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_loco_tasks)
    assert result["status"] == "success"
    assert result["count"] == len(_LOCO_TASK_MAP)
    assert result["task_ids"] == sorted(_LOCO_TASK_MAP)
    assert result["sequenced_ids"] == sorted(_SEQUENCED_TASK_IDS)
    assert result["loco_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert len(result["loco_tasks"]) == len(_LOCO_TASK_MAP)
    # Every descriptor carries the same field set and reads its flags
    # from the module's constants (not restated in the test body).
    for descriptor in result["loco_tasks"]:
        task_id = descriptor["task_id"]
        assert descriptor["name"] == _LOCO_TASK_MAP[task_id]
        assert descriptor["sequenced"] is (task_id in _SEQUENCED_TASK_IDS)
        # Every admitted task is a locomotion write by definition.
        assert descriptor["admits_loco_writes"] is True
    codes = {r["code"] for r in result["refusals"]}
    assert codes == {_INVALID_TASK_CODE, _GATE_REFUSAL_CODE}
    for refusal in result["refusals"]:
        assert refusal["text"] == ERR_CODES[refusal["code"]]


def test_g1_list_loco_tasks_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh lists and dicts; a mutation on the returned
    ``task_ids`` list or ``loco_tasks`` descriptors does not leak
    back into the module's constants. This cell is where a
    share-a-reference regression would surface once, not scattered
    across every call site (same guarantee as the ``action_map``
    snapshot in :mod:`strands_robots.tools.g1.g1_arm_actions`).
    """
    result = _call(g1_list_loco_tasks)
    result["task_ids"].append(9999)
    result["loco_tasks"][0]["synthetic"] = True
    fresh = _call(g1_list_loco_tasks)
    assert 9999 not in fresh["task_ids"]
    assert "synthetic" not in fresh["loco_tasks"][0]


def test_g1_loco_task_admits_resolves_a_valid_id() -> None:
    """An id inside the SDK's set is admitted and the descriptor lands.

    ``0`` is WaveHand (no turn); the verb reports ``admitted=True``
    and carries the resolved descriptor (``task_id``, ``name``,
    ``sequenced``, ``admits_loco_writes``) a future dispatch verb
    would use to decide the follow-up write path. No refusal fields
    fire on the admitted path.
    """
    result = _call(g1_loco_task_admits, task_id=0)
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"task_id": 0}
    assert result["target"]["task_id"] == 0
    assert result["target"]["name"] == "WaveHand (no turn)"
    assert result["target"]["sequenced"] is False
    assert result["target"]["admits_loco_writes"] is True
    assert "refusal_code" not in result


def test_g1_loco_task_admits_resolves_a_valid_name() -> None:
    """A name in the snapshot resolves to the SDK's id.

    ``"WaveHand (with turn)"`` is id ``1``; the verb reports
    ``admitted=True`` and the descriptor names ``1`` on the loco
    write gate (every admitted task is a locomotion write).
    """
    result = _call(g1_loco_task_admits, name="WaveHand (with turn)")
    assert result["status"] == "success"
    assert result["admitted"] is True
    assert result["query"] == {"name": "WaveHand (with turn)"}
    assert result["target"]["task_id"] == 1
    assert result["target"]["name"] == "WaveHand (with turn)"
    assert result["target"]["admits_loco_writes"] is True


def test_g1_loco_task_admits_flags_a_sequenced_task() -> None:
    """An admitted stage-of-a-sequence task carries ``sequenced=True``.

    ``2`` (ShakeHand reach) is a valid SDK dispatch target - the SDK
    admits ``SetTaskId(2)`` - but it is the first stage of a two-stage
    gesture. The verb still reports ``admitted=True`` (it is a lookup,
    not a policy) but the descriptor's ``sequenced`` flag lets a
    caller decide the one-shot refusal decidably before dispatch.
    """
    result = _call(g1_loco_task_admits, task_id=2)
    assert result["admitted"] is True
    assert result["target"]["sequenced"] is True
    assert result["target"]["name"] == "ShakeHand stage 1 (reach out)"


def test_g1_loco_task_admits_refuses_an_unknown_id() -> None:
    """An id outside the SDK's set is refused with ``rc=7303``.

    ``42`` is not a G1 loco task. The verb reports ``admitted=False``
    and carries the SDK's own refusal code (``7303``) with the
    decoded text a future driver-side wrapper would surface, so the
    two sides quote the same error verbatim.
    """
    result = _call(g1_loco_task_admits, task_id=42)
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["query"] == {"task_id": 42}
    assert result["refusal_code"] == _INVALID_TASK_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_TASK_CODE]
    assert "target" not in result


def test_g1_loco_task_admits_refuses_an_unknown_name() -> None:
    """A name not in the snapshot is refused with the same ``rc=7303``.

    ``"wavehand"`` (lower-case, no parenthetical) is not in the
    snapshot; the snapshot ships ``"WaveHand (no turn)"``. The verb
    mirrors the SDK's dict semantics (a caller writing
    ``LocoClient.SetTaskId`` against a mis-labelled name would see
    the same ``rc=7303`` at wire time).
    """
    result = _call(g1_loco_task_admits, name="wavehand")
    assert result["admitted"] is False
    assert result["query"] == {"name": "wavehand"}
    assert result["refusal_code"] == _INVALID_TASK_CODE


def test_g1_loco_task_admits_refuses_both_supplied() -> None:
    """Supplying both ``task_id`` and ``name`` is a caller mistake.

    The verb refuses with ``status="error"`` rather than picking a
    resolution: the ambiguous case is not one the lookup should
    resolve arbitrarily.
    """
    result = _call(g1_loco_task_admits, task_id=0, name="WaveHand (no turn)")
    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_loco_task_admits_refuses_neither_supplied() -> None:
    """Supplying neither ``task_id`` nor ``name`` is a caller mistake.

    Same ambiguous-case rule as the both-supplied refusal: the caller
    has to say what they are asking about.
    """
    result = _call(g1_loco_task_admits)
    assert result["status"] == "error"
    assert "exactly one" in result["message"]


def test_g1_loco_task_admits_refuses_a_bool_task_id() -> None:
    """``bool`` is not an ``int`` this verb should silently accept.

    Python's ``bool`` is a subclass of ``int`` so ``True == 1``, but a
    caller passing ``True`` for a task id is a type mistake, not a
    valid query. The verb refuses so a mis-typed argument surfaces
    at the lookup rather than reaching the SDK's own type coercion.
    """
    result = _call(g1_loco_task_admits, task_id=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_g1_loco_task_admits_refuses_a_non_int_task_id() -> None:
    """A non-int, non-bool ``task_id`` surfaces the type in the refusal.

    ``"0"`` looks correct to a human reader but is a string; the
    refusal names the type and the value the caller passed, so a
    caller sees which of their many parallel tool calls hit the
    wrong shape.
    """
    result = _call(g1_loco_task_admits, task_id="0")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
