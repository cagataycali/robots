"""The wave-hand turn-flag envelope tools name exactly what ``WaveHand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) admits
``WaveHand(turn_flag: bool)`` dispatches from a fixed two-value set
(``False`` wave in place, ``True`` wave while turning 180 degrees);
the :mod:`strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope`
module snapshots that table into module-level constants and exposes
two agent-facing verbs - :func:`g1_list_wave_hand_turn_flags` (list
the whole envelope) and :func:`g1_wave_hand_turn_flag_admits` (decide
one query) - so a caller can decide the SDK's ``rc=7303`` refusal
decidably before a future dispatch path is attempted. The tests here
fix that contract without pulling the SDK: the module is loadable on
a host without ``unitree_sdk2py`` (the same SDK-load-hygiene rule
every other file under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen
or narrow to the constant surfaces here as a shape change rather than
as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  dispatch table (the whole point of the port is that the snapshot
  lets a headless host answer). A driver-side wrapper for
  ``WaveHand`` that lands later will re-validate against the SDK's
  live table at wire time; testing the snapshot vs the live table
  is a driver-side test, not a lookup-side one.
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
from strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope import (
    _GATE_REFUSAL_CODE,
    _INVALID_TASK_CODE,
    _SDK_METHOD,
    _WAVE_HAND_TASK_ID_MAP,
    _WAVE_HAND_TURN_FLAG_MAP,
    g1_list_wave_hand_turn_flags,
    g1_wave_hand_turn_flag_admits,
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

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up. The driver enforces the same rule
    against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the
    only path that loads the SDK); this cell holds the wave-hand
    turn-flag envelope verbs to it too (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_wave_hand_turn_flag_envelope imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_sdk_shipped_set() -> None:
    """The snapshot names every turn-flag the SDK's ``WaveHand`` admits today.

    The SDK's ``WaveHand`` dispatcher takes a boolean, so the
    admitted set is exactly ``{False, True}`` - count pinned at 2.
    A firmware release that widened the surface (adding a third
    variant, e.g. as a tri-state enum) would update this count and
    the map on both sides at once.
    """
    assert len(_WAVE_HAND_TURN_FLAG_MAP) == 2, (
        f"expected 2 wave-hand turn-flag variants (the SDK's boolean surface), "
        f"got {len(_WAVE_HAND_TURN_FLAG_MAP)}: "
        f"{sorted(_WAVE_HAND_TURN_FLAG_MAP)}. A firmware release that widened "
        "or narrowed the WaveHand dispatch table would update this count; "
        "refs strands-labs/robots#358."
    )
    assert set(_WAVE_HAND_TURN_FLAG_MAP) == {False, True}, (
        f"expected the admitted set to be exactly {{False, True}}, got "
        f"{set(_WAVE_HAND_TURN_FLAG_MAP)!r}. Refs strands-labs/robots#358."
    )


def test_the_composed_task_ids_route_through_the_task_id_dispatcher() -> None:
    """Each turn-flag composes to a ``SetTaskId`` value the sibling lookup names.

    The SDK routes ``WaveHand(turn_flag=False)`` through
    ``SetTaskId(0)`` and ``WaveHand(turn_flag=True)`` through
    ``SetTaskId(1)`` - both ids are members of
    :data:`~strands_robots.tools.g1.g1_loco_task_ids._LOCO_TASK_MAP`.
    A caller planning a two-lookup composition (this envelope for
    the boolean surface, then the task-id envelope for the wire-
    level dispatch) sees the same handler on both sides. Pinned
    here so a firmware release that renumbered the ``WaveHand`` task
    ids lands as a shape change on this constant rather than as a
    silent divergence in the tests.
    """
    from strands_robots.tools.g1.g1_loco_task_ids import _LOCO_TASK_MAP

    assert _WAVE_HAND_TASK_ID_MAP == {False: 0, True: 1}, (
        f"expected WaveHand(turn_flag=False)->SetTaskId(0) and "
        f"WaveHand(turn_flag=True)->SetTaskId(1) (the SDK's own routing), got "
        f"{_WAVE_HAND_TASK_ID_MAP!r}. Refs strands-labs/robots#358."
    )
    for turn_flag, task_id in _WAVE_HAND_TASK_ID_MAP.items():
        assert task_id in _LOCO_TASK_MAP, (
            f"composed task id {task_id!r} for turn_flag={turn_flag!r} must be "
            f"a member of g1_loco_task_ids._LOCO_TASK_MAP "
            f"({sorted(_LOCO_TASK_MAP)}); the two envelopes share the SDK's "
            "task-dispatch handler. Refs strands-labs/robots#358."
        )


def test_the_sdk_method_names_the_caller_facing_entry() -> None:
    """The SDK method is ``"WaveHand"`` - the caller-facing dispatch surface.

    The neon bundle's ``g1_wave_hand_loco`` wrapper calls
    ``LocoClient.WaveHand(turn_flag=bool(turn))`` verbatim. Pinned
    here so a firmware release that renamed the SDK method (e.g. to
    ``Wave`` or ``WaveHandTask``) lands as a shape change on this
    constant rather than as a silent divergence in the tests.
    """
    assert _SDK_METHOD == "WaveHand", (
        f"expected SDK method 'WaveHand' (the caller-facing LocoClient "
        f"dispatch), got {_SDK_METHOD!r}. Refs strands-labs/robots#358."
    )


def test_the_refusal_codes_come_from_the_driver_error_table() -> None:
    """Both refusal codes decode against :data:`ERR_CODES`.

    The SDK routes ``WaveHand`` through the same task-dispatch
    handler that :mod:`~strands_robots.tools.g1.g1_loco_task_ids`
    and :mod:`~strands_robots.tools.g1.g1_shake_hand_stage_envelope`
    name on their sides, so the invalid-task refusal quotes the
    same ``7303`` code all three lookups surface. The gate refusal
    ``7404`` is the one the driver's own ``_check_motion_gates``
    quotes on any locomotion-shaped write with the live FSM outside
    :data:`WALK_FSMS`. Both codes must be decodable text so a caller
    can quote the refusal in its own voice.
    """
    assert _INVALID_TASK_CODE in ERR_CODES, (
        f"invalid-task refusal code {_INVALID_TASK_CODE!r} must be a key of "
        "the driver's ERR_CODES table; the envelope surfaces it as a decoded "
        "refusal text. Refs strands-labs/robots#358."
    )
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"gate-refusal code {_GATE_REFUSAL_CODE!r} must be a key of the "
        "driver's ERR_CODES table; the envelope surfaces it as a decoded "
        "refusal text. Refs strands-labs/robots#358."
    )
    assert _INVALID_TASK_CODE == 7303, (
        f"expected invalid-task refusal 7303 (the SDK's task-dispatch "
        f"handler's Invalid task id code), got {_INVALID_TASK_CODE!r}. Refs "
        "strands-labs/robots#358."
    )
    assert _GATE_REFUSAL_CODE == 7404, (
        f"expected gate refusal 7404 (the driver's _check_motion_gates FSM-"
        f"outside-set refusal code), got {_GATE_REFUSAL_CODE!r}. Refs "
        "strands-labs/robots#358."
    )


def test_list_returns_both_turn_flags_in_false_then_true_order() -> None:
    """The list verb returns ``False`` then ``True`` - the SDK's default first.

    The returned envelope's ``turn_flags`` list must be
    ``[False, True]`` so a caller iterating the list sees the SDK's
    own default variant first (``turn_flag=False`` is the SDK's own
    default argument for ``WaveHand``). The descriptors carried on
    ``wave_hand_turn_flags`` are one-per-variant in the same order.
    """
    payload = _call(g1_list_wave_hand_turn_flags)
    assert payload["status"] == "success"
    assert payload["count"] == len(_WAVE_HAND_TURN_FLAG_MAP)
    assert payload["turn_flags"] == [False, True]
    assert [d["turn_flag"] for d in payload["wave_hand_turn_flags"]] == [False, True]


def test_list_names_the_composed_task_id_on_each_descriptor() -> None:
    """Every descriptor carries the ``SetTaskId`` value the variant composes to.

    A caller comparing the composed dispatch against the sibling
    :mod:`~strands_robots.tools.g1.g1_loco_task_ids` envelope reads
    the id off this descriptor rather than restating the routing.
    Pinned here so a firmware release that renumbered the routing
    surfaces on the descriptor's ``composed_task_id`` field first.
    """
    payload = _call(g1_list_wave_hand_turn_flags)
    by_flag = {d["turn_flag"]: d for d in payload["wave_hand_turn_flags"]}
    assert by_flag[False]["composed_task_id"] == 0, (
        f"WaveHand(turn_flag=False) must compose to SetTaskId(0); got "
        f"{by_flag[False]['composed_task_id']!r}. Refs strands-labs/robots#358."
    )
    assert by_flag[True]["composed_task_id"] == 1, (
        f"WaveHand(turn_flag=True) must compose to SetTaskId(1); got "
        f"{by_flag[True]['composed_task_id']!r}. Refs strands-labs/robots#358."
    )
    # The list-level composed_task_ids mirrors the per-descriptor field so a
    # caller filtering by composed dispatch does not have to walk descriptors.
    assert payload["composed_task_ids"] == [0, 1]


def test_list_names_the_sdk_method_verbatim_on_each_descriptor() -> None:
    """Every descriptor carries ``sdk_method='WaveHand'`` and the top-level too.

    The list surfaces the caller-facing entry both at the list level
    (one field read for the whole envelope) and on each descriptor
    (one field read per admitted variant). A future descriptor widen
    that changed the per-variant SDK method (unlikely; both variants
    dispatch through the same method today) would land as a
    per-descriptor divergence here.
    """
    payload = _call(g1_list_wave_hand_turn_flags)
    assert payload["sdk_method"] == _SDK_METHOD
    for descriptor in payload["wave_hand_turn_flags"]:
        assert descriptor["sdk_method"] == _SDK_METHOD, (
            f"descriptor {descriptor!r} must carry sdk_method={_SDK_METHOD!r}; "
            "both variants dispatch through the same method today. Refs "
            "strands-labs/robots#358."
        )
        assert descriptor["admits_loco_writes"] is True, (
            f"descriptor {descriptor!r} must carry admits_loco_writes=True; "
            "every WaveHand dispatch is a locomotion write. Refs "
            "strands-labs/robots#358."
        )


def test_list_surfaces_the_walk_ready_fsm_ids() -> None:
    """The returned envelope names :data:`WALK_FSMS` verbatim.

    ``WaveHand`` dispatch is a locomotion-shaped write; the driver's
    ``_check_motion_gates`` refuses it outside :data:`WALK_FSMS`.
    The envelope surfaces the same set the driver's gate would test
    membership in so a caller comparing the driver's live
    ``fsm_id`` (from ``G1Driver.get_status``) against this list sees
    whether the write gate is currently open.
    """
    payload = _call(g1_list_wave_hand_turn_flags)
    assert payload["walk_ready_fsm_ids"] == sorted(WALK_FSMS), (
        f"walk_ready_fsm_ids {payload['walk_ready_fsm_ids']!r} must be "
        f"sorted(WALK_FSMS)={sorted(WALK_FSMS)!r}; the envelope surfaces the "
        "same set the driver's motion gate reads. Refs strands-labs/robots#358."
    )


def test_list_surfaces_both_refusal_codes_with_decoded_text() -> None:
    """The refusals list carries both codes and their :data:`ERR_CODES` text.

    A caller comparing an intended dispatch against the envelope
    sees the two refusals it would face on the same call: ``7303``
    on a task id outside the SDK's dispatcher's admitted set, and
    ``7404`` on an FSM outside the write gate. Both must carry the
    exact text the driver's error table ships so a caller quotes
    the refusal in its own voice.
    """
    payload = _call(g1_list_wave_hand_turn_flags)
    codes = {r["code"]: r["text"] for r in payload["refusals"]}
    assert codes == {
        _INVALID_TASK_CODE: ERR_CODES[_INVALID_TASK_CODE],
        _GATE_REFUSAL_CODE: ERR_CODES[_GATE_REFUSAL_CODE],
    }


def test_admits_returns_true_on_both_bool_values() -> None:
    """Both ``False`` and ``True`` return ``admitted=True``.

    The SDK's ``WaveHand`` dispatcher admits both variants; the
    envelope surfaces the same answer without touching the bus. A
    firmware release that narrowed the admitted set (unlikely but
    possible) would land here as an ``admitted=False`` payload on
    the removed variant, with a decoded ``rc=7303`` refusal.
    """
    for turn_flag in (False, True):
        payload = _call(g1_wave_hand_turn_flag_admits, turn_flag=turn_flag)
        assert payload["status"] == "success"
        assert payload["admitted"] is True, (
            f"turn_flag {turn_flag!r} is in _WAVE_HAND_TURN_FLAG_MAP but "
            f"g1_wave_hand_turn_flag_admits refused it: {payload!r}. The "
            "snapshot and the admits verb must agree; refs "
            "strands-labs/robots#358."
        )
        assert payload["query"] == {"turn_flag": turn_flag}
        target = payload["target"]
        assert target["turn_flag"] == turn_flag
        assert target["name"] == _WAVE_HAND_TURN_FLAG_MAP[turn_flag]
        assert target["composed_task_id"] == _WAVE_HAND_TASK_ID_MAP[turn_flag]
        assert target["sdk_method"] == _SDK_METHOD
        assert target["admits_loco_writes"] is True


def test_admits_refuses_an_int_argument_as_a_shape_error() -> None:
    """An ``int`` for ``turn_flag`` is a shape error, not a coerced bool.

    Python's ``bool()`` folds every integer to a boolean
    (``bool(1)`` is ``True``, ``bool(0)`` is ``False``), so a
    caller passing ``turn_flag=1`` reaches the SDK's dispatcher
    with an admitted task id at wire time. But at the tool surface
    a caller passing an integer has made a shape error, not asked a
    decidable membership question. The verb refuses the query
    rather than resolving it through the coercion. The neon wrapper
    itself calls ``bool(turn)`` before the SDK sees the value; this
    verb makes the boolean shape decidable at the lookup surface,
    one layer above.
    """
    for bad in (0, 1, -1, 2):
        payload = _call(g1_wave_hand_turn_flag_admits, turn_flag=bad)  # type: ignore[arg-type]
        assert payload["status"] == "error", (
            f"integer {bad!r} must refuse as a shape error, not coerce; got {payload!r}. Refs strands-labs/robots#358."
        )
        assert "int" in payload["message"]
        assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_a_non_bool_argument_as_a_shape_error() -> None:
    """A non-bool for ``turn_flag`` is a shape error, not a membership answer.

    A caller passing a string, float, ``None``, list or tuple has
    made a shape error the verb surfaces decidably rather than
    resolving it through Python's ``bool()`` coercion (which would
    fold every non-empty string to ``True`` and reach the SDK's
    dispatcher with an admitted task id).
    """
    bad_values: tuple[object, ...] = ("True", "yes", 0.0, 1.0, None, [False], (True,), {})
    for bad in bad_values:
        payload = _call(g1_wave_hand_turn_flag_admits, turn_flag=bad)  # type: ignore[arg-type]
        assert payload["status"] == "error", (
            f"non-bool {bad!r} must refuse as a shape error; got {payload!r}. Refs strands-labs/robots#358."
        )
        assert type(bad).__name__ in payload["message"]
        assert "strands-labs/robots#358" in payload["message"]
