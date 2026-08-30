"""The joint-variant-note lookup verbs name the per-slot caveats the neon bundle observed.

The Unitree G1 ships in three physical builds -- ``23dof``,
``29dof-with-waist-locked``, and the fully-populated ``29dof`` -- and
the driver's ``send_action`` writes against a 29-slot map that names
every joint on the widest build. Six slots (``13``, ``14``, ``20``,
``21``, ``27``, ``28``) are physically absent on at least one narrower
build; the neon bundle
(``cagataycali/neon-the-g1/tools/g1_joints.py::INVALID_NOTES``)
observed that gap and captured the caveat text one row per slot. The
:mod:`strands_robots.tools.g1.g1_joint_variant_notes` module ports the
six-row table to the ``@tool`` surface so a caller planning a rollout
can decide the wire refusal decidably before ``send_action`` is
attempted, rather than triggering it against the firmware and reading
the refusal off the next lowstate.

The tests here fix that contract without pulling the SDK: the module
is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every note text answer is read off the
module's own snapshot rather than restated in the tests, so a widen
or narrow to the observation surfaces here as a shape change rather
than as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* Whether the local G1 physically has the named DoF. The bundle-side
  ``INVALID_NOTES`` is the neon observation; deciding whether *this*
  robot is a ``23dof`` or a ``29dof`` build is a driver-side answer
  the driver does not yet expose (refs
  strands-labs/robots#2765). The tests here pin the six caveats
  the bundle observed and no more; a driver-side variant-detection
  method that lands later will surface the same slot indices this
  verb names, and this cell will be one of the two sides that
  quotes them.
* The exact wire refusal ``send_action`` returns on a slot the local
  build lacks. That is a firmware answer the driver does not model
  today (the driver's map admits every name in the 29-slot table);
  the caveat this verb surfaces is the neon-side observation of
  which slots produce that refusal, not the refusal text itself.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_joint_variant_notes import (
    _MAX_SLOT,
    _MIN_SLOT,
    _VARIANT_NOTES,
    g1_joint_variant_note,
    g1_list_joint_variant_notes,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site (same
    idiom as
    :mod:`tests.drivers.test_g1_error_codes_reads_the_sdk_return_code_catalogue`).
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up. The bundle's
    ``INVALID_NOTES`` is a pure Python literal, so nothing in this
    module has a reason to reach the SDK; this cell holds the port to
    that rule too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_joint_variant_notes")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_joint_variant_notes imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_names_the_six_neon_observed_slots() -> None:
    """The snapshot's key set is exactly the six slots the bundle observed.

    ``13``/``14`` are the waist roll/pitch, ``20``/``21`` are the
    left wrist pitch/yaw, ``27``/``28`` are the right wrist pitch/yaw.
    A widen (a firmware revision that removes another slot on a
    narrower build) or a narrow (a build that starts wiring the
    wrist yaw) lands here as a test failure so this cell and the
    bundle stay in sync.
    """
    assert set(_VARIANT_NOTES) == {13, 14, 20, 21, 27, 28}


def test_the_snapshot_bounds_frame_the_driver_map() -> None:
    """The min/max bounds frame the driver's 29-slot map.

    ``_MIN_SLOT`` is ``0`` (the first index in the driver's joint
    table), ``_MAX_SLOT`` is ``29`` (one past the last slot in the
    map). A widen to a 33-DoF G1 would land here first, so the range
    refusal on :func:`g1_joint_variant_note` continues to quote the
    right bound.
    """
    assert _MIN_SLOT == 0
    assert _MAX_SLOT == 29


def test_the_snapshot_slot_indices_lie_inside_the_driver_map() -> None:
    """Every noted slot is a valid index into the driver's 29-slot map.

    A caveat about slot ``42`` would be a fabrication about a slot the
    driver's ``send_action`` would refuse at wire time anyway; the
    bundle's observation only makes sense for slots the driver's
    frame can name. This cell holds that invariant.
    """
    for slot in _VARIANT_NOTES:
        assert _MIN_SLOT <= slot < _MAX_SLOT, (
            f"slot {slot} is outside the driver's map "
            f"[{_MIN_SLOT}, {_MAX_SLOT}); the bundle's observation cannot "
            "cover a slot send_action would refuse at wire time."
        )


def test_g1_list_joint_variant_notes_returns_the_full_snapshot() -> None:
    """The list verb's payload names the six caveats and their texts.

    ``count`` is six, ``notes`` is one descriptor per slot (sorted
    ascending), ``noted_slots`` is the sorted integer list alone.
    Every descriptor reads its ``note`` text from :data:`_VARIANT_NOTES`
    (not restated in the test body) so a re-word of one entry lands
    once.
    """
    result = _call(g1_list_joint_variant_notes)
    assert result["status"] == "success"
    assert result["count"] == len(_VARIANT_NOTES)
    assert result["noted_slots"] == sorted(_VARIANT_NOTES)
    assert len(result["notes"]) == len(_VARIANT_NOTES)
    for descriptor in result["notes"]:
        slot = descriptor["slot"]
        assert descriptor["has_note"] is True
        assert descriptor["note"] == _VARIANT_NOTES[slot], (
            f"descriptor for slot={slot} carried note "
            f"{descriptor['note']!r} but snapshot holds "
            f"{_VARIANT_NOTES[slot]!r}. The two must not diverge."
        )


def test_g1_list_joint_variant_notes_names_the_slot_counts() -> None:
    """The list verb names the covered and uncovered slot counts.

    ``covered_slot_count`` is the driver's map width (``29`` today).
    ``uncovered_slot_count`` is that width minus the six caveats
    (``23`` today) - the number of slots present on every G1 variant
    the bundle observed. A caller can compare their intended slot
    against those two counts to see whether it lies in the covered
    or uncovered region.
    """
    result = _call(g1_list_joint_variant_notes)
    assert result["covered_slot_count"] == _MAX_SLOT - _MIN_SLOT
    assert result["uncovered_slot_count"] == (_MAX_SLOT - _MIN_SLOT) - len(_VARIANT_NOTES)


def test_g1_list_joint_variant_notes_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the snapshot.

    The verb returns fresh lists and dicts; a mutation on the returned
    ``noted_slots`` list or a descriptor does not leak back into the
    module's constants. This cell is where a share-a-reference
    regression would surface once, not scattered across every call
    site (same guarantee as the ``fsm_targets`` snapshot in
    :mod:`strands_robots.tools.g1.g1_fsm_targets`).
    """
    result = _call(g1_list_joint_variant_notes)
    result["noted_slots"].append(9999)
    result["notes"][0]["synthetic"] = True
    fresh = _call(g1_list_joint_variant_notes)
    assert 9999 not in fresh["noted_slots"]
    assert "synthetic" not in fresh["notes"][0]


def test_g1_joint_variant_note_resolves_a_noted_slot() -> None:
    """A slot inside :data:`_VARIANT_NOTES` returns the caveat text.

    Slot ``13`` (waist roll) has the ``23dof/29dof with waist locked``
    caveat; the verb returns ``has_note=True`` and the exact text
    :data:`_VARIANT_NOTES` holds so a caller reading it sees the same
    sentence a caller enumerating the whole list would.
    """
    result = _call(g1_joint_variant_note, slot=13)
    assert result["status"] == "success"
    assert result["query"] == {"slot": 13}
    assert result["has_note"] is True
    assert result["note"] == _VARIANT_NOTES[13]
    assert result["slot"] == 13


def test_g1_joint_variant_note_resolves_an_unnoted_slot() -> None:
    """A slot inside the driver's map but outside the caveat table is admitted.

    Slot ``3`` (left knee) is present on every G1 variant; the verb
    returns ``has_note=False`` and an empty ``note`` string so a
    caller composing an admission decision does not have to branch on
    which of the two verbs answered - the row shape is the same for
    a noted and an unnoted slot.
    """
    result = _call(g1_joint_variant_note, slot=3)
    assert result["status"] == "success"
    assert result["query"] == {"slot": 3}
    assert result["has_note"] is False
    assert result["note"] == ""
    assert result["slot"] == 3


def test_g1_joint_variant_note_resolves_slot_zero() -> None:
    """Slot ``0`` (left hip pitch) is inside the map and unnoted.

    Pins the boundary case at the low end so a re-word or narrow of
    :data:`_MIN_SLOT` does not silently admit a negative slot as
    ``has_note=False``.
    """
    result = _call(g1_joint_variant_note, slot=0)
    assert result["status"] == "success"
    assert result["has_note"] is False


def test_g1_joint_variant_note_resolves_the_last_slot() -> None:
    """Slot ``28`` (right wrist yaw) is the last inside the map and IS noted.

    Pins the boundary case at the high end so a re-word or widen of
    :data:`_MAX_SLOT` does not silently refuse a slot the bundle
    observed a caveat for.
    """
    result = _call(g1_joint_variant_note, slot=28)
    assert result["status"] == "success"
    assert result["has_note"] is True
    assert result["note"] == _VARIANT_NOTES[28]


def test_g1_joint_variant_note_refuses_a_slot_below_min() -> None:
    """A negative ``slot`` is refused as a range violation.

    ``send_action`` would refuse the same name at wire time - the
    driver's map has no entry for a negative slot - so this verb
    refuses at admission rather than fabricating a no-caveat answer.
    The refusal names the bounds so a caller can quote them back.
    """
    result = _call(g1_joint_variant_note, slot=-1)
    assert result["status"] == "error"
    assert "out of range" in result["message"]
    assert "strands-labs/robots#358" in result["message"]
    assert "strands-labs/robots#2765" in result["message"]


def test_g1_joint_variant_note_refuses_a_slot_at_max() -> None:
    """A ``slot == _MAX_SLOT`` is refused as out of range.

    The map's upper bound is exclusive: slot ``29`` is not in the
    driver's 29-slot map, so the verb refuses it. The refusal quotes
    the exclusive-bound convention (``[0, 29)``) so a caller can see
    which end of the range they violated.
    """
    result = _call(g1_joint_variant_note, slot=_MAX_SLOT)
    assert result["status"] == "error"
    assert "out of range" in result["message"]


def test_g1_joint_variant_note_refuses_a_slot_far_beyond_max() -> None:
    """A slot well beyond the map is refused with a range violation.

    Same rule as :func:`test_g1_joint_variant_note_refuses_a_slot_at_max`;
    the cell exists so a caller who mistypes an SDK index (say ``42``
    thinking of a hex value) sees the range refusal rather than a
    silent no-caveat answer that a future variant might invert.
    """
    result = _call(g1_joint_variant_note, slot=42)
    assert result["status"] == "error"
    assert "out of range" in result["message"]


def test_g1_joint_variant_note_refuses_a_bool_slot() -> None:
    """``bool`` is not a valid slot query even though it subclasses ``int``.

    Python's ``bool`` is a subclass of ``int`` so ``True == 1``, but a
    caller passing ``True`` as a slot is a type mistake. The verb
    refuses so a mis-typed argument surfaces at admission rather than
    reaching the descriptor's ``int(True)`` coercion (same rule as
    :func:`~strands_robots.tools.g1.g1_error_codes.g1_decode_error_code`).
    """
    result = _call(g1_joint_variant_note, slot=True)
    assert result["status"] == "error"
    assert "bool" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_joint_variant_note_refuses_a_string_slot() -> None:
    """A string ``slot`` names its type in the refusal.

    ``"13"`` looks correct to a human reader but is a string; the
    refusal names the type and the value the caller passed, so a
    caller sees which of their many parallel tool calls hit the
    wrong shape (same refusal pattern as
    :func:`~strands_robots.tools.g1.g1_error_codes.g1_decode_error_code`).
    """
    result = _call(g1_joint_variant_note, slot="13")  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "str" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_the_list_and_the_admits_agree_on_every_covered_slot() -> None:
    """The two verbs answer the same shape for every slot in the driver's map.

    A caller who enumerates :func:`g1_list_joint_variant_notes` sees
    the six caveats; a caller who calls :func:`g1_joint_variant_note`
    against each of the twenty-nine slots sees six ``has_note=True``
    answers matching the list's rows, and twenty-three ``has_note=False``
    answers naming the slots the bundle observed no caveat for. This
    cell holds that invariant against a drift where one verb is
    updated without the other.
    """
    listing = _call(g1_list_joint_variant_notes)
    listed_slots = {row["slot"]: row for row in listing["notes"]}

    for slot in range(_MIN_SLOT, _MAX_SLOT):
        admits = _call(g1_joint_variant_note, slot=slot)
        assert admits["status"] == "success"
        if slot in listed_slots:
            assert admits["has_note"] is True
            assert admits["note"] == listed_slots[slot]["note"]
        else:
            assert admits["has_note"] is False
            assert admits["note"] == ""
