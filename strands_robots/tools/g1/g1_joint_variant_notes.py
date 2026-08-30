"""Agent-facing lookup for the per-slot variant caveats the neon bundle observed.

``strands_robots.drivers.g1.G1Driver`` writes ``LowCmd_`` frames against a
29-slot motor layout that names every joint on the fully-populated 29-DoF
G1 build (the driver's docstring on the joint table cites this in
``strands_robots.tools.g1.g1_joints``). Two of the physical G1 builds ship
with fewer active DoFs, and the driver's ``send_action`` admits *every*
name in the 29-slot map regardless of the physical build: a caller
pointing at a joint the local robot does not have receives a firmware
refusal at wire time, not a name-error at admission.

The neon bundle
(``cagataycali/neon-the-g1/tools/g1_joints.py::INVALID_NOTES``) captured
the six per-slot caveats that come with that trade-off, one line per
slot the caller should be aware of before pointing a policy at it:

* Slots ``13`` (:data:`waist_roll`) and ``14`` (:data:`waist_pitch`) are
  invalid on the ``23dof`` and the ``29dof-with-waist-locked`` variants
  -- both physical builds where the waist assembly ships with a
  mechanical lock rather than a controller.
* Slots ``20`` / ``21`` (:data:`left_wrist_pitch` / :data:`left_wrist_yaw`)
  and slots ``27`` / ``28`` (:data:`right_wrist_pitch` /
  :data:`right_wrist_yaw`) are invalid on the ``23dof`` variant, whose
  wrist assemblies are the two-DoF (roll only) build.

Slots not in this table (the twenty-three legs, waist yaw, shoulders,
elbows and wrist rolls) are present on every G1 variant the driver's
send-action accepts a name for. This module surfaces the six-row table
as an agent-facing snapshot so a caller planning a rollout can decide
the wire refusal decidably before ``send_action`` is attempted, rather
than triggering it against the firmware and reading the refusal off the
next lowstate. Refs strands-labs/robots#358, strands-labs/robots#2765.

Two things this module is deliberately *not*:

* An execution path. The driver's own ``send_action`` gate is where a
  write-time refusal fires, and neither this verb nor a future
  ``send_action`` change alters that admission surface: ``send_action``
  accepts every name in the driver's map because the *local* variant is
  a build-time answer the driver does not yet know (that question is
  open on ``refs strands-labs/robots#2765``). This module ports the
  read-only observation from the neon bundle without introducing a
  second admission gate that would drift from the driver's.
* An SDK re-import. The six-row table lives here as module-level
  constants captured from the neon bundle so
  ``import strands_robots.tools.g1.g1_joint_variant_notes`` pulls no
  ``unitree_sdk2py`` submodule -- the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358.
  The invariant the observation carries is that the six slot indices
  named here match the ones the neon bundle observed against the real
  robot; a firmware revision that widens or narrows the variant set is
  a bundle-side update the port here carries into the strands-labs tree
  once the neon bundle observes it.

What this module does not decide.

* Whether the *local* G1 physically has the named DoF. The driver's own
  ``send_action`` writes to whichever motor slot the caller names in
  the ``LowCmd_`` frame; the firmware answers whether that slot is
  wired to a real motor. The per-build presence question is open on
  ``refs strands-labs/robots#2765``; when the driver's build-detection
  method lands, its refusal will name the same slot indices this
  verb surfaces.
* Whether the driver's live joint-torques on those slots are zero. The
  driver's ``rt/lowstate`` cache does not yet expose the per-motor
  torque array (the driver's ``_on_lowstate`` caches only the
  IMU sub-record, see the ``g1_imu`` module's docstring); a future
  driver-side change that surfaces per-motor state would carry the
  caveat this table names into the returned envelope, not this
  static lookup.
* Which variant a specific G1 is. There is no ``variant`` field in
  ``get_status`` today; the neon bundle's ``INVALID_NOTES`` names the
  three variant labels (``23dof``, ``29dof-with-waist-locked``,
  ``29dof``) in the note text, so a caller who knows their build can
  compare a note to their build label directly. A structured variant
  read is a driver-side change; refs strands-labs/robots#2765.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The six per-slot variant caveats the neon bundle
#: (``cagataycali/neon-the-g1/tools/g1_joints.py::INVALID_NOTES``)
#: observed against the physical G1 builds. Slots not in this map are
#: present on every variant the driver's send-action names. Named as a
#: module-level constant so a future widen or narrow lands in one place
#: instead of drifting between the two verbs' payloads and the tests.
#: The strings quote the neon bundle's exact observation verbatim so
#: a caller comparing the return of this verb against the bundle's
#: source sees the same sentence on both sides.
_VARIANT_NOTES: dict[int, str] = {
    13: "invalid on 23dof/29dof with waist locked",
    14: "invalid on 23dof/29dof with waist locked",
    20: "invalid on 23dof",
    21: "invalid on 23dof",
    27: "invalid on 23dof",
    28: "invalid on 23dof",
}

#: The lower bound (inclusive) on the slot indices ``send_action``
#: names in its ``LowCmd_`` frame. Named as a module constant so
#: :func:`g1_joint_variant_note`'s out-of-range refusal quotes the same
#: bound the driver's map uses, and so a widen (a 33-DoF G1) lands in
#: one place. The 29-slot layout matches
#: ``strands_robots.tools.g1.g1_joints.G1_JOINT_NAMES``; the invariant
#: this verb quotes is byte-identity with that length.
_MIN_SLOT: int = 0

#: The upper bound (exclusive) on the slot indices. Twenty-nine slots
#: is the driver's map width; a slot at or above this bound is refused
#: with a range refusal rather than an ``unknown``-tagged answer,
#: because the driver's ``send_action`` would refuse the same name at
#: wire time (the map has no entry for it), so this verb refuses the
#: query at admission rather than fabricating a note the bundle did
#: not observe.
_MAX_SLOT: int = 29


def _describe(slot: int) -> dict[str, Any]:
    """Build the per-slot descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_joint_variant_notes`
    so :func:`g1_joint_variant_note`'s admitted-path payload names the
    same fields, and so a widen to the descriptor lands in one place.
    Every field is a snapshot read; no bus is touched.

    Slots not in :data:`_VARIANT_NOTES` render with an empty ``note``
    and ``has_note=False``, so a caller iterating over the list sees a
    uniform row shape rather than a hole for the twenty-three
    every-variant slots. The empty-string convention matches
    ``strands_robots.tools.g1.g1_joints.g1_joint_name`` which returns
    ``note: ""`` on a slot the neon bundle observed no caveat for.
    """
    note = _VARIANT_NOTES.get(slot, "")
    return {
        "slot": slot,
        "has_note": bool(note),
        "note": note,
    }


@tool
def g1_list_joint_variant_notes() -> dict[str, Any]:
    """Return the per-slot variant caveats the neon bundle observed on the physical G1s.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    snapshot of :data:`_VARIANT_NOTES` and the driver's slot-count
    bounds read at call time. Useful before ``send_action`` is called
    with a joint name on a slot the *local* G1 variant does not
    physically wire: the caveat text names the variant(s) that lack
    the slot so a caller can compare it against the build label they
    were given.

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        catalogued caveats (six today, one per slot the neon bundle
        observed absent on at least one physical variant); a
        ``notes`` list of descriptors sorted by slot, each carrying
        ``slot`` (the integer index in the driver's 29-slot map),
        ``has_note`` (``True`` on every entry in this list;
        preserved for row-shape parity with
        :func:`g1_joint_variant_note` which returns ``has_note=False``
        for slots outside this table), and ``note`` (the caveat text
        the neon bundle observed, verbatim). Also carries a
        ``noted_slots`` list of just the integer slot indices for a
        caller who only needs the set, a ``covered_slot_count``
        naming the driver's map width (``29`` today), and an
        ``uncovered_slot_count`` naming the number of slots without
        a caveat (``23`` today). Every field is a snapshot of a
        module-level constant; no dynamic decode runs here. Refs
        strands-labs/robots#358, strands-labs/robots#2765.
    """
    slots = sorted(_VARIANT_NOTES)
    return {
        "status": "success",
        "count": len(_VARIANT_NOTES),
        "notes": [_describe(slot) for slot in slots],
        "noted_slots": slots,
        "covered_slot_count": _MAX_SLOT - _MIN_SLOT,
        "uncovered_slot_count": (_MAX_SLOT - _MIN_SLOT) - len(_VARIANT_NOTES),
    }


@tool
def g1_joint_variant_note(slot: int) -> dict[str, Any]:
    """Decide whether one slot carries a variant caveat.

    Read-only. Compares ``slot`` against :data:`_VARIANT_NOTES` and
    returns the caveat text if the neon bundle observed one for that
    slot. A slot inside the driver's 29-slot map that is *not* in the
    caveat table returns ``has_note=False`` and an empty ``note`` --
    the row shape matches :func:`g1_list_joint_variant_notes` so a
    caller composing an admission decision does not have to branch on
    which verb answered.

    Args:
        slot: The integer index into the driver's 29-slot motor map.
            Must be an ``int``; ``bool`` is refused (``True`` is
            ``int(1)`` but a passed-through boolean is a caller
            mistake, not a valid slot query). An out-of-range slot
            (below :data:`_MIN_SLOT` or at/above :data:`_MAX_SLOT`)
            is refused as a range violation rather than admitted as a
            no-caveat answer: ``send_action`` would refuse the same
            name at wire time (the driver's map has no entry for the
            slot), so a note about it would be a fiction the neon
            bundle did not observe.

    Returns:
        A dict with ``status`` (``"success"`` on any admitted query,
        ``"error"`` on the type-mistake or range refusals); a
        ``query`` sub-dict carrying the supplied ``slot``; a
        ``has_note`` boolean naming whether the neon bundle observed
        a caveat for that slot; a ``note`` string carrying the caveat
        text on ``has_note=True`` and empty on ``has_note=False``;
        and a ``covered_slot_count`` naming the driver's map width so
        a caller comparing a batch of queries can name the same bound
        the list verb quotes. Refs strands-labs/robots#358,
        strands-labs/robots#2765.
    """
    if isinstance(slot, bool):
        return {
            "status": "error",
            "message": (f"slot must be int, got bool ({slot!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(slot, int):
        return {
            "status": "error",
            "message": (f"slot must be int, got {type(slot).__name__} ({slot!r}). Refs strands-labs/robots#358."),
        }
    if slot < _MIN_SLOT or slot >= _MAX_SLOT:
        return {
            "status": "error",
            "message": (
                f"slot {slot} out of range [{_MIN_SLOT}, {_MAX_SLOT}). "
                f"The driver's 29-slot motor map admits {_MIN_SLOT} through "
                f"{_MAX_SLOT - 1}; a slot at or above {_MAX_SLOT} would be "
                "refused by send_action at wire time (the driver's map has "
                "no entry for it). Refs strands-labs/robots#358, "
                "strands-labs/robots#2765."
            ),
        }

    return {
        "status": "success",
        "query": {"slot": slot},
        **_describe(slot),
        "covered_slot_count": _MAX_SLOT - _MIN_SLOT,
    }
