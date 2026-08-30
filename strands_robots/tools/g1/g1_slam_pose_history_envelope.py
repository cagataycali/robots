"""Agent-facing lookup for the pose-history capacity the neon SLAM runner caps at.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._process_frame``)
appends a per-frame pose dict (``x`` / ``y`` / ``z`` / ``theta`` /
``timestamp``) to an in-memory ``_pose_history`` list on every ICP
registration.  Before the next append the runner reads
``if len(self._pose_history) > 2000: self._pose_history =
self._pose_history[-2000:]`` -- an inclusive strict-greater ceiling
that truncates the list back to its last ``2000`` entries.  The
neon comment on that ceiling is that the pose history is a
process-local trail (``g1_slam_pose`` reads the tail entry, not
the whole list) and the ``2000`` figure is the observed ceiling
before the per-frame append becomes a memory pressure -- at 10 Hz
that is ~3.3 minutes of trail before the head starts scrolling
off, which is longer than any single relocalise-plus-map-save
session the neon runner is designed to serve.

This module surfaces that ceiling to an agent so a caller planning
a long-running SLAM session can decide the pose-history bound
decidably before a future driver-side wrapper for the runner
fires, rather than reading the neon runner's silent
truncate-to-tail behaviour off a trail that has already lost its
head.  A caller reading ``g1_slam_pose_history_admits(len)`` with
its intended session length sees the ``2000``-entry ceiling and
can either shorten the session, buffer to disk on its own timer,
or accept the trail-truncation semantics the runner ships with.

Twin of :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
(the merged strands-labs/robots#3006) and
:mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope` (the
merged strands-labs/robots#3005): those name the ``_try_relocalize``
match-quality and precondition halves against a candidate map, this
one names the ``_process_frame`` bookkeeping capacity against the
runner's own trail.  The three stay separate because the neon
runner reads them in different code paths and on different
arguments: the relocalise envelope is a *result*-side judgement on
what open3d's ICP produced (a live registration result), the
map-liveness envelope is a *precondition* on the caller's map
argument (refused before any ICP dispatch), and this envelope is a
*bookkeeping* ceiling on the runner's own append log (refused
before the next append lands).  Colocating them would hand an
agent planner a single refusal payload that mixed a
"the fit was aliased across the room" remedy, a "build a bigger
map before you try" remedy, and a "shorten the session or buffer
to disk" remedy against three distinct surfaces.

Two things this module is deliberately *not*.

* An execution path.  The neon bundle's ``_process_frame`` writes
  the trail on ``_SlamRunner._process_frame``'s thread; today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front that
  write, so no motion admission is at stake at this bookkeeping
  ceiling.  A future driver method that fronts SLAM will surface
  the same module-local :data:`_REFUSAL_TEXT` on a caller planning
  a session longer than the runner's trail; this module ports the
  read-only capacity half without also introducing a second SLAM
  path the driver does not yet own, refs strands-labs/robots#358.
* An SDK re-import.  The ceiling is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_slam_pose_history_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import -- the
  import-hygiene contract every other file in this package
  carries, refs strands-labs/robots#358.  A caller authoring a
  SLAM plan before any SLAM extra is installed on their host
  still gets the ceiling back verbatim.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its
FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The SLAM
pose-history ceiling runs on the neon runner's own thread against
an in-memory Python list -- it never touches ``rt/lowcmd``, never
talks to the locomotion controller, and reaches no SDK RPC service
that ships an rc table for a trail-capacity refusal.  Borrowing
``7404`` on an over-ceiling refusal would hand an agent planner a
motion-FSM remedy for a bookkeeping argument.  The refusal shape
this module returns names the numeric bound violation in
module-local text so a planner reads a remedy that matches the
surface, and a future driver-side SLAM session wrapper will
surface the same module-local text.  This mirrors the
same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` names
for the ``_try_relocalize`` match-quality gate, refs
strands-labs/robots#358.

What this module does not decide.

* What happens *after* the ceiling.  The neon runner does not
  refuse when the trail reaches ``2000`` entries; it truncates the
  list to its last ``2000`` on the next append and keeps going.
  This envelope names the ceiling so a caller who wants to keep
  the whole trail can plan around it (buffer to disk on their own
  timer, shorten the session, or accept the truncation), not so a
  future write verb refuses at ``2001``.  A future driver-side
  SLAM wrapper that answered ``g1_slam_pose_history_admits`` on
  the runner's live length would surface the same refusal text on
  a caller who wanted a hard-refuse-at-ceiling semantics rather
  than the neon runner's soft-truncate one.
* The trail's *time span*.  The ``2000``-entry ceiling is a count,
  not a duration; at the neon runner's observed 10 Hz that maps
  to ~3.3 minutes, but a slower or faster LiDAR frame rate would
  produce a different span.  A caller who wants a time-bounded
  trail reads the LiDAR frame-rate envelope (a future
  ``g1_lidar_frequency_envelope`` lookup pair would name it) and
  multiplies against this count, not the other way around.
* Whether the trail is ever *read*.  The neon runner exposes
  :func:`~cagataycali.neon-the-g1.tools.g1_slam.g1_slam_pose` as
  a tail-only reader; a caller who wanted the whole trail would
  have to reach into ``_SlamRunner._pose_history`` directly.  A
  future driver-side SLAM wrapper that surfaced the whole trail
  would inherit this ceiling as the maximum length it could ever
  return, and a caller planning against that would read this
  envelope to size its buffer.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_count_error

#: The inclusive upper bound on the pose-history list's length
#: that the neon SLAM runner's ``_process_frame`` truncates back
#: down to.  The neon bundle reads
#: ``if len(self._pose_history) > 2000: self._pose_history =
#: self._pose_history[-2000:]``, so a list at exactly ``2000``
#: entries is the boundary case the runner admits (the next
#: append lands at ``2001`` and triggers the truncate).  Named as
#: an inclusive upper bound so
#: :func:`g1_slam_pose_history_admits` admits a caller-supplied
#: ``2000`` and refuses a caller-supplied ``2001``, mirroring the
#: runner's ``>`` refusal exactly.
_POSE_HISTORY_MAX: int = 2000

#: The module-local refusal text every
#: ``g1_slam_pose_history_admits`` refusal quotes when the
#: caller-supplied session length sits above the
#: neon-runner-observed ceiling.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the SLAM bookkeeping path ships no distinct rc -- the
#: neon runner just truncates ``_pose_history`` on the next
#: ``_process_frame`` append, which the caller reads as "the head
#: of the trail is gone".  The motion-FSM ``7404`` entry (its
#: nearest neighbour) reads
#: ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a remedy
#: that points a planner at locomotion FSM transitions to fix a
#: bookkeeping argument.  Surfacing the module-local text keeps
#: the refusal payload's remedy on the same surface the write
#: belongs on; a future driver-side SLAM session wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "pose history capacity refused - the intended session length sits "
    "above the neon-runner-observed pose-history ceiling; the runner "
    "truncates to the tail at this point and the head of the trail is "
    "lost. Refs strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_pose_history_envelope` so
    :func:`g1_slam_pose_history_admits` names the same field on
    its admitted-path payload and so a widen to the descriptor
    lands in one place.  Every field is a snapshot read; no bus
    is touched.
    """
    return {
        "pose_history_max": _POSE_HISTORY_MAX,
    }


@tool
def g1_list_slam_pose_history_envelope() -> dict[str, Any]:
    """Return the pose-history capacity the neon SLAM runner caps at.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: the
    field is a module-level constant.  Useful before a future
    driver-side wrapper for a long-running SLAM session is called,
    so a caller can compare an intended session length against the
    ceiling the neon runner's ``_process_frame`` truncates back
    down to and can carry the module-local refusal text a
    driver-side wrapper would surface on a bounds violation.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed ceiling (``pose_history_max``);
        and a ``refusals`` list carrying a single descriptor with
        the module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on an over-ceiling session length.  Every
        field is a snapshot of an observed bound or a
        module-local text; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_slam_pose_history_admits(
    session_length: int = _POSE_HISTORY_MAX,
) -> dict[str, Any]:
    """Decide whether a candidate session length sits at or below the pose-history ceiling.

    Read-only.  Compares ``session_length`` against the ceiling
    :func:`g1_list_slam_pose_history_envelope` returns and reports
    the bound the argument violates.  No driver instance, no DDS,
    no SDK, no ``numpy`` / ``open3d`` / ``kiss_icp`` submodule
    import: the decision reads only a module-level constant and
    the argument itself.

    A length at or below the ceiling is *not* the same as an
    admitted long-running session: the neon runner also refuses
    on the match-quality dimensions (fitness, translation, trace)
    which the twin envelope
    :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
    names, and on the map-liveness precondition which the twin
    envelope
    :mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
    names.  The returned payload names only the numeric
    bookkeeping decision.

    Args:
        session_length: The intended trail length, i.e. the number
            of ``_process_frame`` calls the caller plans to run
            before the head of ``_pose_history`` is allowed to
            scroll off.  The default ``2000`` (the observed
            ceiling) admits, so a caller who does not pass an
            explicit argument lands on the admitted boundary case
            the runner itself admits.  The shared
            :func:`~strands_robots.utils.positive_count_error`
            domain refuses non-``int`` inputs (including ``bool``,
            which is an ``int`` subclass whose ``True`` would
            otherwise be a silent ``1``), values below ``1``, and
            any type coercion that could hide a floating-point
            argument.  A length of ``0`` (an empty session) is
            refused by the shared domain rather than by the
            capacity ceiling because a session that plans zero
            frames is a shape mistake rather than a
            capacity-exceeded refusal.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the session length sits at or below the ceiling; a
        ``refusals`` list of refusal descriptors, each carrying
        the dimension name, the offending value, the clamp it
        violated, the comparison ("value > bound" or the
        shared-domain "shared-domain" descriptor for a shape
        mistake), and the module-local :data:`_REFUSAL_TEXT` a
        driver-side wrapper would surface if the session were
        run through the runner without buffering; the same
        ``envelope`` sub-dict
        :func:`g1_list_slam_pose_history_envelope` returns.  On
        an admitted length the ``refusals`` list is empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check first: the shared
    # positive_count_error refuses bool, non-int, and value < 1.
    # This lands before the pose-history ceiling check so a shape
    # mistake reads decidably against the module-local ceiling.
    domain_err = positive_count_error(session_length, "session_length", "g1_slam_pose_history_admits")
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "session_length",
                "value": session_length,
                "bound_key": "pose_history_max",
                "bound": _POSE_HISTORY_MAX,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )
    else:
        # The shared domain has admitted the shape; now grade the
        # pose-history ceiling.  The runner reads
        # len(_pose_history) > 2000 and refuses (truncates) on
        # strict-greater, so exactly 2000 admits.
        if session_length > _POSE_HISTORY_MAX:
            refusals.append(
                {
                    "dimension": "session_length",
                    "value": session_length,
                    "bound_key": "pose_history_max",
                    "bound": _POSE_HISTORY_MAX,
                    "comparison": "value > bound",
                    "text": _REFUSAL_TEXT,
                }
            )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
