"""The safe-posture-fsm-gate tools name exactly what the neon _assert_safe_for_damp whitelists gate on.

``cagataycali/neon-the-g1/tools/g1_safe_posture.py::_assert_safe_for_damp``
refuses a ``LocoClient.Damp()`` preamble whenever the driver's live
``_fsm_id`` sits outside a verb-specific whitelist: ``{3, 4, 706}`` for
``g1_safe_squat_to_stand``, ``{1, 702}`` for ``g1_safe_lie_to_stand``,
``{500, 501, 801}`` for ``g1_safe_stand_to_squat``. Two agent-facing
tools - :func:`g1_list_safe_posture_fsm_gates` and
:func:`g1_safe_posture_fsm_admits` - exist to surface those three
whitelists before the neon verb (or a future driver-side wrapper for it)
is called, so a caller can decide the refusal decidably rather than
firing the Damp preamble against an out-of-set FSM.

The membership rules the tools carry are cross-checked here against
:data:`~strands_robots.tools.g1.g1_fsm_targets._FSM_NAME_MAP` (the
SDK-admitted transition-target set), so a neon-side widen that named
an FSM id the SDK does not admit as a transition target would surface
here at CI rather than at wire time. What the tests do restate is the
shape of each returned record and the SDK-load-hygiene contract every
file under :mod:`strands_robots.tools.g1` carries: importing the
module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_fsm_targets import _FSM_NAME_MAP
from strands_robots.tools.g1.g1_safe_posture_fsm_gates import (
    _SAFE_POSTURE_DESCRIPTIONS,
    _SAFE_POSTURE_FSM_GATES,
    g1_list_safe_posture_fsm_gates,
    g1_safe_posture_fsm_admits,
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
    office bring-up. This test holds the safe-posture-fsm-gate module
    to that rule.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_safe_posture_fsm_gates")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        "strands_robots.tools.g1.g1_safe_posture_fsm_gates imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs "
        "strands-labs/robots#358)."
    )


def test_whitelist_verbs_are_the_three_neon_safe_posture_verb_names() -> None:
    """The whitelist covers exactly the three neon safe-posture verbs today.

    A neon-side widen that added a fourth safe-posture verb would ship
    a new whitelist row here (the port grows by one). A caller reading
    the envelope wants the row count to match the neon module's verb
    count, so this test pins the whole set rather than any single row.
    """
    expected = {
        "g1_safe_squat_to_stand",
        "g1_safe_lie_to_stand",
        "g1_safe_stand_to_squat",
    }
    assert set(_SAFE_POSTURE_FSM_GATES) == expected, (
        f"whitelist verbs {sorted(_SAFE_POSTURE_FSM_GATES)} diverged from "
        f"the neon safe-posture verb set {sorted(expected)}. Update the "
        "port or the expectation."
    )


def test_every_whitelist_id_is_admitted_by_the_sdk_transition_set() -> None:
    """Every FSM id on every whitelist is one the SDK admits as a target.

    A neon-side whitelist that named an FSM id the SDK's own
    ``SetFsmId`` handler refuses would leave a caller unable to enter
    the FSM the safe-posture verb requires. The cross-check here is
    against :data:`~strands_robots.tools.g1.g1_fsm_targets._FSM_NAME_MAP`
    (the SDK-admitted transition-target set); an id in a whitelist
    that is not in the name map is either a neon-side typo or an SDK
    contraction the port has to follow.
    """
    admitted = set(_FSM_NAME_MAP)
    for verb, fsm_ids in _SAFE_POSTURE_FSM_GATES.items():
        strays = set(fsm_ids) - admitted
        assert strays == set(), (
            f"safe-posture whitelist {verb!r} named FSM ids {sorted(strays)} "
            f"not in the SDK-admitted set {sorted(admitted)}. Refs "
            "strands-labs/robots#358."
        )


def test_squat_to_stand_whitelist_matches_the_neon_expected_fsms_verbatim() -> None:
    """The squat-to-stand row is the exact neon ``{3, 4, 706}`` set.

    The neon file cites ``expected_fsms={3, 4, 706}`` for the verb;
    the port has to reproduce that byte-for-byte or a caller reading
    a refusal would see a different admitted set than the neon gate
    would compute.
    """
    assert _SAFE_POSTURE_FSM_GATES["g1_safe_squat_to_stand"] == frozenset({3, 4, 706})


def test_lie_to_stand_whitelist_matches_the_neon_expected_fsms_verbatim() -> None:
    """The lie-to-stand row is the exact neon ``{1, 702}`` set."""
    assert _SAFE_POSTURE_FSM_GATES["g1_safe_lie_to_stand"] == frozenset({1, 702})


def test_stand_to_squat_whitelist_matches_the_neon_expected_fsms_verbatim() -> None:
    """The stand-to-squat row is the exact neon ``{500, 501, 801}`` set."""
    assert _SAFE_POSTURE_FSM_GATES["g1_safe_stand_to_squat"] == frozenset({500, 501, 801})


def test_description_table_keys_match_whitelist_keys() -> None:
    """Every whitelist row carries a description; no orphan rows either way.

    The description table is the human-readable half of the envelope
    the list verb returns; a row on one side without a row on the
    other would surface either as a ``KeyError`` at call time or as a
    silently-dropped description in the returned record.
    """
    assert set(_SAFE_POSTURE_DESCRIPTIONS) == set(_SAFE_POSTURE_FSM_GATES), (
        f"description keys {sorted(_SAFE_POSTURE_DESCRIPTIONS)} diverged "
        f"from whitelist keys {sorted(_SAFE_POSTURE_FSM_GATES)}."
    )


def test_list_returns_every_whitelist_the_module_names() -> None:
    """The list verb with no argument returns all three whitelist rows.

    The neon file has three safe-posture verbs today; the envelope
    reports three ``gates`` entries and a matching ``count``. A row
    count drift would surface here.
    """
    result = _call(g1_list_safe_posture_fsm_gates)
    assert result["status"] == "success"
    assert result["count"] == len(_SAFE_POSTURE_FSM_GATES)
    assert len(result["gates"]) == len(_SAFE_POSTURE_FSM_GATES)
    returned_verbs = {row["verb"] for row in result["gates"]}
    assert returned_verbs == set(_SAFE_POSTURE_FSM_GATES)


def test_list_returns_gates_sorted_lexicographically_by_verb() -> None:
    """The ``gates`` list is stable across calls and sorted by verb name.

    The list verb has no ordering argument, so callers rely on a
    stable order for diff-shaped comparisons. Sort by verb name
    ascending matches the sibling ``g1_list_motion_gates`` /
    ``g1_list_dds_topics`` verbs.
    """
    result = _call(g1_list_safe_posture_fsm_gates)
    verbs_in_order = [row["verb"] for row in result["gates"]]
    assert verbs_in_order == sorted(verbs_in_order)
    assert result["verbs"] == sorted(_SAFE_POSTURE_FSM_GATES)


def test_every_gate_row_carries_the_whitelist_and_description() -> None:
    """Every returned row carries ``fsm_ids``, ``fsm_count``, and ``description``.

    The row shape has to be uniform across all three verbs so a
    caller can walk the list without special-casing any single row.
    """
    result = _call(g1_list_safe_posture_fsm_gates)
    for row in result["gates"]:
        verb = row["verb"]
        assert row["fsm_ids"] == sorted(_SAFE_POSTURE_FSM_GATES[verb])
        assert row["fsm_count"] == len(_SAFE_POSTURE_FSM_GATES[verb])
        assert row["description"] == _SAFE_POSTURE_DESCRIPTIONS[verb]


def test_verb_filter_returns_that_verb_verbatim() -> None:
    """Passing a valid verb name returns exactly that one row.

    The verb-filter argument is the way a caller reads a single
    row without walking the whole list; passing a known verb narrows
    the envelope to that row and reports the same verb on the top-level
    ``verb`` field.
    """
    result = _call(g1_list_safe_posture_fsm_gates, verb="g1_safe_squat_to_stand")
    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["verb"] == "g1_safe_squat_to_stand"
    assert len(result["gates"]) == 1
    row = result["gates"][0]
    assert row["verb"] == "g1_safe_squat_to_stand"
    assert row["fsm_ids"] == [3, 4, 706]


def test_verb_filter_refuses_an_unknown_verb_by_name() -> None:
    """An off-partition verb name is refused with the valid set listed.

    A caller that typos a verb name gets both the refused input and
    the resolution path (the three valid verb names) in one refusal,
    so no follow-up call is needed to recover.
    """
    result = _call(g1_list_safe_posture_fsm_gates, verb="g1_safe_headstand")
    assert result["status"] == "error"
    assert "g1_safe_headstand" in result["message"]
    for valid_verb in _SAFE_POSTURE_FSM_GATES:
        assert valid_verb in result["message"]


def test_admits_reports_membership_the_neon_gate_would_compute() -> None:
    """The admits verb returns ``True`` for every id in the named whitelist.

    The neon ``_assert_safe_for_damp`` gate refuses the Damp preamble
    when the FSM sits outside the whitelist; the admits verb reports
    the same membership answer without firing the gate.
    """
    for verb, fsm_ids in _SAFE_POSTURE_FSM_GATES.items():
        for fsm_id in fsm_ids:
            result = _call(g1_safe_posture_fsm_admits, fsm_id=fsm_id, verb=verb)
            assert result["status"] == "success"
            assert result["admitted"] is True, (
                f"verb {verb!r} whitelist id {fsm_id} was reported as refused; expected admitted."
            )
            assert result["fsm_ids"] == sorted(fsm_ids)


def test_admits_reports_a_non_member_as_refused() -> None:
    """An FSM outside the whitelist is refused with the same set the gate would test.

    A caller with a live FSM outside the whitelist gets the ``admitted``
    boolean plus the verb's whitelist, so it can phrase its own refusal
    without a follow-up call. The FSM used here (``2`` Squat) is on
    the SDK-admitted set but not on any of the three safe-posture
    whitelists, so this is a real off-partition query rather than a
    shape typo.
    """
    result = _call(g1_safe_posture_fsm_admits, fsm_id=2, verb="g1_safe_squat_to_stand")
    assert result["status"] == "success"
    assert result["admitted"] is False
    assert result["fsm_ids"] == [3, 4, 706]
    assert result["verb"] == "g1_safe_squat_to_stand"


def test_admits_refuses_an_empty_verb_with_the_valid_set_listed() -> None:
    """Empty verb is refused because the whitelist is verb-specific.

    The three safe-posture verbs have three different admitted FSM
    sets; asking ``does FSM=500 admit?`` without naming the verb is a
    shape error - the answer depends on which verb the caller has in
    mind. The refusal names all three valid verb names and points at
    :func:`g1_list_safe_posture_fsm_gates` for the whole-envelope
    read.
    """
    result = _call(g1_safe_posture_fsm_admits, fsm_id=500, verb="")
    assert result["status"] == "error"
    assert "verb is required" in result["message"]
    for valid_verb in _SAFE_POSTURE_FSM_GATES:
        assert valid_verb in result["message"]
    assert "g1_list_safe_posture_fsm_gates" in result["message"]


def test_admits_refuses_an_unknown_verb_by_name() -> None:
    """An off-partition verb name is refused with the three valid names listed."""
    result = _call(g1_safe_posture_fsm_admits, fsm_id=500, verb="g1_safe_backflip")
    assert result["status"] == "error"
    assert "g1_safe_backflip" in result["message"]
    for valid_verb in _SAFE_POSTURE_FSM_GATES:
        assert valid_verb in result["message"]


def test_admits_refuses_bool_as_fsm_id_despite_being_int_subclass() -> None:
    """``bool`` is refused even though ``True`` is ``int(1)``.

    ``True`` is a Python-side coincidence with FSM id ``1`` (Damp);
    a caller passing a boolean where an FSM id is expected has made
    a typing mistake and should get a decidable refusal rather than
    a coincidentally-correct admits answer.
    """
    result = _call(g1_safe_posture_fsm_admits, fsm_id=True, verb="g1_safe_lie_to_stand")
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_admits_refuses_non_int_fsm_id_and_names_the_type() -> None:
    """A non-int FSM id is refused and the returned message names the offending type."""
    for bad_input in ("500", 500.0, None, [500], (500,)):
        result = _call(
            g1_safe_posture_fsm_admits,
            fsm_id=bad_input,
            verb="g1_safe_stand_to_squat",
        )
        assert result["status"] == "error", f"non-int fsm_id {bad_input!r} was not refused"
        assert type(bad_input).__name__ in result["message"]


def test_admits_refuses_non_str_verb() -> None:
    """A non-str verb is refused decidably rather than through key-lookup coercions."""
    for bad_verb in (None, 42, 3.14, ["g1_safe_squat_to_stand"]):
        result = _call(g1_safe_posture_fsm_admits, fsm_id=3, verb=bad_verb)
        assert result["status"] == "error", f"non-str verb {bad_verb!r} was not refused"
        assert type(bad_verb).__name__ in result["message"]


def test_whitelists_are_disjoint_or_overlap_deliberately() -> None:
    """The three whitelists partition the SDK-admitted set into three roles.

    ``{3, 4, 706}`` (squat-to-stand) and ``{1, 702}`` (lie-to-stand)
    are disjoint - a robot is either sitting or lying, not both. The
    ``{500, 501, 801}`` (stand-to-squat) set is also disjoint from
    the other two; this is the neon bundle's chosen invariant today.
    A future safe-posture verb that overlapped an existing whitelist
    would be a caller-visible ambiguity (``fsm_id=500`` would admit
    two verbs at once); the invariant is captured here so a neon-side
    widen that broke it would surface at CI.
    """
    squat = _SAFE_POSTURE_FSM_GATES["g1_safe_squat_to_stand"]
    lie = _SAFE_POSTURE_FSM_GATES["g1_safe_lie_to_stand"]
    stand = _SAFE_POSTURE_FSM_GATES["g1_safe_stand_to_squat"]
    assert squat & lie == frozenset()
    assert squat & stand == frozenset()
    assert lie & stand == frozenset()
