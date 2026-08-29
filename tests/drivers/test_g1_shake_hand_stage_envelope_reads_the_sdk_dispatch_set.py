"""The shake-hand-stage envelope tools name exactly what ``ShakeHand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) admits
``ShakeHand(stage: int)`` dispatches from a fixed three-value set
(``-1`` toggle, ``0`` reach out, ``1`` shake); the
:mod:`strands_robots.tools.g1.g1_shake_hand_stage_envelope` module
snapshots that table into module-level constants and exposes two
agent-facing verbs - :func:`g1_list_shake_hand_stages` (list the whole
envelope) and :func:`g1_shake_hand_stage_admits` (decide one query) -
so a caller can decide the SDK's ``rc=7303`` refusal decidably before a
future dispatch path is attempted. The tests here fix that contract
without pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen or
narrow to the constant surfaces here as a shape change rather than as
a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  dispatch table (the whole point of the port is that the snapshot
  lets a headless host answer). A driver-side wrapper for
  ``ShakeHand`` that lands later will re-validate against the SDK's
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
from strands_robots.tools.g1.g1_shake_hand_stage_envelope import (
    _GATE_REFUSAL_CODE,
    _INVALID_STAGE_CODE,
    _SEQUENCED_STAGES,
    _SHAKE_HAND_STAGE_MAP,
    _TOGGLE_STAGE,
    g1_list_shake_hand_stages,
    g1_shake_hand_stage_admits,
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
    path that loads the SDK); this cell holds the shake-hand-stage
    envelope verbs to it too (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_shake_hand_stage_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_shake_hand_stage_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK loads only "
        "inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_sdk_shipped_set() -> None:
    """The snapshot names every stage the SDK's ``ShakeHand`` admits today.

    The SDK's own dispatch table has 3 entries observed against the
    real robot (``-1`` toggle, ``0`` reach, ``1`` shake). A drift on
    either side surfaces here: a driver-side dispatch wrapper (when
    it lands) will validate the same set at wire time and its
    refusal string will quote ``rc=7303`` on any stage outside it.
    The count is pinned rather than listed value-by-value so a
    caller widening the map on the driver side updates one number
    here rather than 3 assertions.
    """
    assert len(_SHAKE_HAND_STAGE_MAP) == 3, (
        f"expected 3 shake-hand stages in the SDK-observed snapshot, got "
        f"{len(_SHAKE_HAND_STAGE_MAP)}: {sorted(_SHAKE_HAND_STAGE_MAP)}. "
        "A firmware release that widened or narrowed the ShakeHand dispatch "
        "table would update this count; refs strands-labs/robots#358."
    )


def test_the_toggle_sentinel_is_negative_one() -> None:
    """The toggle sentinel is the SDK's own default.

    ``LocoClient.ShakeHand`` takes ``stage: int = -1`` and interprets
    that value as "advance the internal counter"; a caller that omits
    the argument entirely picks the toggle. Named here so a firmware
    release that changed the sentinel lands as a shape change on this
    constant rather than as a silent divergence in the tests.
    """
    assert _TOGGLE_STAGE == -1, (
        f"expected toggle sentinel -1 (the SDK's own default for "
        f"ShakeHand(stage=int)), got {_TOGGLE_STAGE!r}. Refs "
        "strands-labs/robots#358."
    )
    assert _TOGGLE_STAGE in _SHAKE_HAND_STAGE_MAP, (
        f"toggle sentinel {_TOGGLE_STAGE!r} must be in the admitted stage set "
        f"{sorted(_SHAKE_HAND_STAGE_MAP)}; the SDK admits it as a dispatch "
        "value. Refs strands-labs/robots#358."
    )


def test_the_sequenced_pair_is_the_two_non_toggle_stages() -> None:
    """The ordered pair is ``{0, 1}`` and excludes the toggle sentinel.

    Stage ``0`` (reach) and stage ``1`` (shake) are only useful in
    order; the neon bundle's ``g1_shake_hand_loco`` wrapper drove
    them sequentially. The toggle sentinel ``-1`` skips the ordering
    question because the SDK advances the counter itself. A caller
    planning a one-shot dispatch filters against
    :data:`_SEQUENCED_STAGES` to see whether a stage argument
    requires a prior dispatch to make sense.
    """
    assert _SEQUENCED_STAGES == frozenset({0, 1}), (
        f"expected sequenced-pair {{0, 1}} (the two ordered stages the SDK's "
        f"ShakeHand handler expects in order), got {sorted(_SEQUENCED_STAGES)}. "
        "Refs strands-labs/robots#358."
    )
    assert _TOGGLE_STAGE not in _SEQUENCED_STAGES, (
        f"toggle sentinel {_TOGGLE_STAGE!r} must not be in the sequenced-pair "
        f"set {sorted(_SEQUENCED_STAGES)}; the SDK advances the counter itself "
        "on the toggle, so a caller using -1 does not need to enforce "
        "ordering. Refs strands-labs/robots#358."
    )


def test_the_refusal_codes_come_from_the_driver_error_table() -> None:
    """Both refusal codes decode against :data:`ERR_CODES`.

    The SDK routes ``ShakeHand`` through the same task-dispatch
    handler that :mod:`~strands_robots.tools.g1.g1_loco_task_ids`
    names on ``SetTaskId``, so the invalid-stage refusal quotes the
    same ``7303`` code both verbs surface. The gate refusal
    ``7404`` is the one the driver's own ``_check_motion_gates``
    quotes on any locomotion-shaped write with the live FSM outside
    :data:`WALK_FSMS`. Both codes must be decodable text so a caller
    can quote the refusal in its own voice; a missing entry here
    means the driver's error table has drifted from the envelope's
    surfaced codes.
    """
    assert _INVALID_STAGE_CODE in ERR_CODES, (
        f"invalid-stage refusal code {_INVALID_STAGE_CODE!r} must be a key of "
        "the driver's ERR_CODES table; the envelope surfaces it as a decoded "
        "refusal text. Refs strands-labs/robots#358."
    )
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"gate-refusal code {_GATE_REFUSAL_CODE!r} must be a key of the "
        "driver's ERR_CODES table; the envelope surfaces it as a decoded "
        "refusal text. Refs strands-labs/robots#358."
    )
    assert _INVALID_STAGE_CODE == 7303, (
        f"expected invalid-stage refusal 7303 (the SDK's task-dispatch "
        f"handler's Invalid task id code), got {_INVALID_STAGE_CODE!r}. Refs "
        "strands-labs/robots#358."
    )
    assert _GATE_REFUSAL_CODE == 7404, (
        f"expected gate refusal 7404 (the driver's _check_motion_gates FSM-"
        f"outside-set refusal code), got {_GATE_REFUSAL_CODE!r}. Refs "
        "strands-labs/robots#358."
    )


def test_list_returns_every_stage_in_sorted_order() -> None:
    """The list verb returns every admitted stage sorted ascending.

    The returned envelope's ``stages`` list must be identical to
    ``sorted(_SHAKE_HAND_STAGE_MAP)`` so a caller iterating the list
    sees the SDK's own key order (``-1``, ``0``, ``1``). The
    descriptors carried on ``shake_hand_stages`` are one-per-stage
    in the same order.
    """
    payload = _call(g1_list_shake_hand_stages)
    assert payload["status"] == "success"
    assert payload["count"] == len(_SHAKE_HAND_STAGE_MAP)
    assert payload["stages"] == sorted(_SHAKE_HAND_STAGE_MAP)
    assert [d["stage"] for d in payload["shake_hand_stages"]] == sorted(_SHAKE_HAND_STAGE_MAP)


def test_list_names_the_toggle_and_sequenced_flags() -> None:
    """Every descriptor carries a ``sequenced`` and a ``toggle`` boolean.

    The two flags together let a caller decide the shape of an
    intended dispatch without walking the descriptor list: a caller
    planning a one-shot dispatch filters by ``sequenced=False`` and
    picks either the toggle or a lone-stage id (though no lone-stage
    ids are admitted today; the whole set is either sequenced or
    toggle).
    """
    payload = _call(g1_list_shake_hand_stages)
    by_stage = {d["stage"]: d for d in payload["shake_hand_stages"]}
    assert by_stage[-1]["toggle"] is True, "the -1 sentinel is the toggle"
    assert by_stage[-1]["sequenced"] is False, "the toggle skips the sequenced-pair question"
    assert by_stage[0]["toggle"] is False, "stage 0 (reach) is not the toggle"
    assert by_stage[0]["sequenced"] is True, "stage 0 (reach) is the first of the ordered pair"
    assert by_stage[1]["toggle"] is False, "stage 1 (shake) is not the toggle"
    assert by_stage[1]["sequenced"] is True, "stage 1 (shake) is the second of the ordered pair"


def test_list_surfaces_the_walk_ready_fsm_ids() -> None:
    """The returned envelope names :data:`WALK_FSMS` verbatim.

    ``ShakeHand`` dispatch is a locomotion-shaped write; the driver's
    ``_check_motion_gates`` refuses it outside :data:`WALK_FSMS`. The
    envelope surfaces the same set the driver's gate would test
    membership in so a caller comparing the driver's live ``fsm_id``
    (from ``G1Driver.get_status``) against this list sees whether
    the write gate is currently open.
    """
    payload = _call(g1_list_shake_hand_stages)
    assert payload["walk_ready_fsm_ids"] == sorted(WALK_FSMS), (
        f"walk_ready_fsm_ids {payload['walk_ready_fsm_ids']!r} must be "
        f"sorted(WALK_FSMS)={sorted(WALK_FSMS)!r}; the envelope surfaces the "
        "same set the driver's motion gate reads. Refs strands-labs/robots#358."
    )


def test_list_surfaces_both_refusal_codes_with_decoded_text() -> None:
    """The refusals list carries both codes and their :data:`ERR_CODES` text.

    A caller comparing an intended dispatch against the envelope sees
    the two refusals it would face on the same call: ``7303`` on a
    stage outside the admitted set, and ``7404`` on an FSM outside
    the write gate. Both must carry the exact text the driver's
    error table ships so a caller quotes the refusal in its own
    voice.
    """
    payload = _call(g1_list_shake_hand_stages)
    codes = {r["code"]: r["text"] for r in payload["refusals"]}
    assert codes == {
        _INVALID_STAGE_CODE: ERR_CODES[_INVALID_STAGE_CODE],
        _GATE_REFUSAL_CODE: ERR_CODES[_GATE_REFUSAL_CODE],
    }


def test_admits_returns_true_on_every_snapshot_stage() -> None:
    """Every stage in the snapshot returns ``admitted=True``.

    A membership check against the SDK's own table must answer
    ``True`` on the exact set the module ships; a widening of the
    table (a fourth stage lands) surfaces here as a failure on the
    new stage until this test's iterator is updated.
    """
    for stage in _SHAKE_HAND_STAGE_MAP:
        payload = _call(g1_shake_hand_stage_admits, stage=stage)
        assert payload["status"] == "success"
        assert payload["admitted"] is True, (
            f"stage {stage!r} is in _SHAKE_HAND_STAGE_MAP but g1_shake_hand_stage_admits "
            f"refused it: {payload!r}. The snapshot and the admits verb must agree; "
            "refs strands-labs/robots#358."
        )
        assert payload["query"] == {"stage": stage}
        target = payload["target"]
        assert target["stage"] == stage
        assert target["name"] == _SHAKE_HAND_STAGE_MAP[stage]
        assert target["sequenced"] == (stage in _SEQUENCED_STAGES)
        assert target["toggle"] == (stage == _TOGGLE_STAGE)
        assert target["admits_loco_writes"] is True


def test_admits_refuses_an_out_of_set_integer() -> None:
    """An integer outside the snapshot returns ``admitted=False``.

    The refusal payload names the SDK's own ``rc=7303`` code and its
    decoded text so a caller quoting the refusal in its own voice
    sees the exact string a future driver-side wrapper would
    surface at wire time.
    """
    # 2 is an admitted SetTaskId value (ShakeHand stage 1 via that path) but
    # NOT an admitted ShakeHand(stage=) value; the SDK's dispatchers keep
    # separate tables. Picking it here surfaces exactly that: a caller who
    # confused the two entry points sees a decidable refusal.
    payload = _call(g1_shake_hand_stage_admits, stage=2)
    assert payload["status"] == "success"
    assert payload["admitted"] is False
    assert payload["query"] == {"stage": 2}
    assert payload["refusal_code"] == _INVALID_STAGE_CODE
    assert payload["refusal_text"] == ERR_CODES[_INVALID_STAGE_CODE]


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``bool`` for ``stage`` is a shape error, not a membership answer.

    ``True`` is ``int(1)`` under Python's ``isinstance`` but a caller
    passing a boolean to a stage-id argument has made a shape error,
    not asked a decidable membership question. The verb refuses the
    query rather than resolving it to ``stage=1``.
    """
    payload = _call(g1_shake_hand_stage_admits, stage=True)  # type: ignore[arg-type]
    assert payload["status"] == "error"
    assert "bool" in payload["message"]
    assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_a_non_int_argument_as_a_shape_error() -> None:
    """A non-int for ``stage`` is a shape error, not a membership answer.

    The SDK's own dispatcher takes an ``int``; a caller passing a
    string, float or ``None`` has made a shape error the verb
    surfaces decidably rather than resolving it through Python's
    coercions.
    """
    for bad in ("0", 0.0, None, [0], (0,)):
        payload = _call(g1_shake_hand_stage_admits, stage=bad)  # type: ignore[arg-type]
        assert payload["status"] == "error", f"non-int {bad!r} must refuse"
        assert type(bad).__name__ in payload["message"]
        assert "strands-labs/robots#358" in payload["message"]
