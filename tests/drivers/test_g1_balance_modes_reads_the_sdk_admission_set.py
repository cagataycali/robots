"""The balance-mode lookup tools name what ``LocoClient.BalanceStand`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``BalanceStand(int)`` which internally calls ``SetBalanceMode`` and
admits a small set of pre-programmed modes: ``0`` (static balance,
the default) and ``3`` (dynamic balance, from the neon bundle's field
notes). The :mod:`strands_robots.tools.g1.g1_balance_modes` module
snapshots the observed admitted set into a module-level dict and
exposes two agent-facing verbs -
:func:`g1_list_balance_modes` (name the whole set) and
:func:`g1_balance_mode_admits` (decide one query) - so a caller can
decide the refusal decidably before a future locomotion write path is
attempted. The tests here fix that contract without pulling the SDK:
the module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen
or narrow to the observed set surfaces here as a shape change rather
than as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The snapshot is the neon
  bundle's observed admitted set, not the SDK's own admissions (the
  SDK admits any int silently). A driver-side wrapper for
  ``BalanceStand`` that lands later will re-check the mode at wire
  time and its refusal string will quote the ``7404`` gate-refusal
  code the driver's ``_check_motion_gates`` also quotes.
* Whether the driver's live ``fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  live driver-instance read and belongs on
  :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; the verb surfaces
  the set as a snapshot so a caller comparing an intended write
  against both conditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_balance_modes import (
    _BALANCE_MODE_MAP,
    _GATE_REFUSAL_CODE,
    _INVALID_MODE_CODE,
    g1_balance_mode_admits,
    g1_list_balance_modes,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_balance_modes")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_balance_modes imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_set() -> None:
    """The mode map pins the two modes the neon bundle documented.

    ``0`` (Static, the SDK default) and ``3`` (Dynamic, from
    neon's memory.md field notes) are the two modes
    ``g1_balance_stand`` was called with against the real robot.
    A widen or narrow of that set surfaces here as a shape change
    the tests read off the module's own snapshot, not as a
    diverging copy this file would need to update.
    """
    assert set(_BALANCE_MODE_MAP) == {0, 3}, (
        f"Balance-mode snapshot drifted from the neon-observed set. "
        f"Got {sorted(_BALANCE_MODE_MAP)}, expected [0, 3]. Update the "
        f"snapshot and this test together."
    )
    assert _BALANCE_MODE_MAP[0] == "Static"
    assert _BALANCE_MODE_MAP[3] == "Dynamic"


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The verb's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side ``BalanceStand`` wrapper would surface. Pinned
    here so a re-wording of that message lands in one place, not one
    in the driver and a diverging copy in this verb.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _INVALID_MODE_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"verb quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_balance_modes_returns_the_whole_table() -> None:
    """The verb's payload names every mode, the gate set, and the refusal.

    ``modes`` carries every entry in :data:`_BALANCE_MODE_MAP` sorted
    by ``mode_id``, ``walk_ready_fsm_ids`` quotes :data:`WALK_FSMS`,
    and ``refusals`` names the ``7404`` gate-refusal code with the
    decoded text :data:`ERR_CODES` carries.
    """
    result = _call(g1_list_balance_modes)
    assert result["status"] == "success"
    modes = result["modes"]
    assert len(modes) == 2
    # Sorted by mode_id ascending
    assert [m["mode_id"] for m in modes] == [0, 3]
    for descriptor in modes:
        mode_id = descriptor["mode_id"]
        assert descriptor["name"] == _BALANCE_MODE_MAP[mode_id]
        assert descriptor["admits_loco_writes"] is True
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_list_balance_modes_returns_fresh_containers() -> None:
    """Two calls do not share the returned mode list or refusals list.

    A shared list would let a caller who mutates one call's payload
    see the mutation on the next call, which is exactly the sort of
    surprise the neon bundle's caller-owned-copy contract exists to
    prevent. Pin the invariant so a future refactor that memoises
    the payload surfaces here.
    """
    a = _call(g1_list_balance_modes)
    b = _call(g1_list_balance_modes)
    assert a is not b
    assert a["modes"] is not b["modes"]
    assert a["refusals"] is not b["refusals"]
    a["modes"].append({"mode_id": 999, "name": "Injected", "admits_loco_writes": False})
    assert len(b["modes"]) == 2, "mode list is shared between calls"


def test_g1_balance_mode_admits_resolves_a_valid_id() -> None:
    """A mode id inside the admitted set returns the full descriptor.

    The admitted-path payload names the same fields
    :func:`g1_list_balance_modes` returns, plus
    ``walk_ready_fsm_ids`` for the follow-on gate decision. Pins
    the descriptor shape so a caller consuming both verbs sees the
    same keys on both sides.
    """
    result = _call(g1_balance_mode_admits, mode_id=0)
    assert result["status"] == "success"
    assert result["mode"] == {
        "mode_id": 0,
        "name": "Static",
        "admits_loco_writes": True,
    }
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_balance_mode_admits_resolves_a_valid_name() -> None:
    """The reverse-lookup by name returns the same descriptor as the id path.

    A caller who has the neon-observed label but not the id (a
    common case when reading a config file) should see the same
    admitted-path payload the id-side query returns; this test
    pins that the two entry points agree on the descriptor.
    """
    result = _call(g1_balance_mode_admits, name="Dynamic")
    assert result["status"] == "success"
    assert result["mode"] == {
        "mode_id": 3,
        "name": "Dynamic",
        "admits_loco_writes": True,
    }


def test_g1_balance_mode_admits_refuses_an_unknown_id() -> None:
    """A mode id outside the admitted set is refused with the 7404 code.

    The neon bundle refused unknown modes at the verb boundary; a
    future driver-side wrapper would refuse them with the same
    code the driver's motion gate uses for FSM refusals. This test
    pins that shape so a caller sees one refusal shape across
    every mode-side entry point.
    """
    result = _call(g1_balance_mode_admits, mode_id=42)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_MODE_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_MODE_CODE]
    assert "42" in result["reason"]
    assert "not in the admitted set" in result["reason"]


def test_g1_balance_mode_admits_refuses_an_unknown_name() -> None:
    """A mis-cased or unknown label is refused with the 7404 code.

    The comparison is case-sensitive against the snapshot; a caller
    passing ``"static"`` (lowercase) should see a refusal rather
    than a silent fallthrough to the closest-match label. Pinned
    so a future soft-match refactor surfaces here.
    """
    result = _call(g1_balance_mode_admits, name="static")
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_MODE_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_MODE_CODE]
    assert "'static'" in result["reason"]


def test_g1_balance_mode_admits_refuses_both_supplied() -> None:
    """Passing both ``mode_id`` and ``name`` is refused as ambiguous.

    The two entry points are mutually exclusive so the verb is
    decidable on exactly one argument; a caller passing both would
    otherwise get a silent id-side answer that ignores the name
    argument, which is exactly the surprise the ambiguity refusal
    prevents.
    """
    result = _call(g1_balance_mode_admits, mode_id=0, name="Static")
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_MODE_CODE
    assert "both" in result["reason"]


def test_g1_balance_mode_admits_refuses_neither_supplied() -> None:
    """Passing neither argument is refused so the caller sees the ambiguity.

    An empty call would otherwise fall through to the name-side
    branch's dict lookup for ``None``, which raises a TypeError -
    the neon bundle's caller-owned-error contract wants that
    surfaced as a refusal envelope, not an unhandled exception.
    """
    result = _call(g1_balance_mode_admits)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_MODE_CODE
    assert "neither" in result["reason"]


def test_g1_balance_mode_admits_refuses_a_bool_mode_id() -> None:
    """A bool ``mode_id`` is refused because ``bool`` subclasses ``int``.

    A caller passing ``True`` would otherwise look up ``1``
    (unknown), returning a confusing "1 is not in the admitted set"
    refusal that hides the type mistake. Refusing at the boundary
    surfaces the mistake to the caller instead.
    """
    result = _call(g1_balance_mode_admits, mode_id=True)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_MODE_CODE
    assert "bool" in result["reason"].lower()


def test_g1_balance_mode_admits_refuses_a_non_int_mode_id() -> None:
    """A non-int non-bool ``mode_id`` is refused with a shape-mistake reason.

    A caller passing ``"0"`` (str) or ``0.0`` (float) would
    otherwise fall through to the ``in`` lookup which reads them as
    unknown ids; refusing at the boundary surfaces the type
    mistake instead.
    """
    for bad_value in ("0", 0.0, [0], {0}, None):
        # None is caught by the neither-supplied branch already; skip it
        if bad_value is None:
            continue
        result = _call(g1_balance_mode_admits, mode_id=bad_value)  # type: ignore[arg-type]
        assert result["status"] == "error", f"mode_id={bad_value!r} should be refused as not-an-int"
        assert result["refusal_code"] == _INVALID_MODE_CODE
        assert "not an int" in result["reason"], (
            f"mode_id={bad_value!r} refusal reason should name the type mistake, got: {result['reason']!r}"
        )
