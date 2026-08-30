"""Agent-facing lookup for the map-chunks compaction trigger the neon SLAM runner fires at.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner._process_frame``)
appends a per-frame ``(N, 4)`` XYZI world-point block to an in-memory
``_map_chunks`` list on every ICP registration the runner's
``_accumulating`` flag admits.  Before the next append the runner reads
``if len(self._map_chunks) > 100:`` -- an inclusive strict-greater
ceiling that steals the chunk list under the runner's own lock,
stitches the stolen chunks into a single ``(N, 4)`` array, calls the
module-level ``_voxel_dedup`` helper (the 5 cm grid the merged
strands-labs/robots#3014 names), and installs the deduped result back
as the sole chunk on the map.  The neon comment on that ceiling is
that the compaction pass is amortised bookkeeping -- the chunk list
grows one entry per admitted frame at the runner's LiDAR rate, and the
compaction runs on the ``_process_frame`` thread rather than a
dedicated worker -- so the ``100`` figure is the observed batch size
before the amortised compaction cost starts pressuring the per-frame
budget.  At the neon-observed 10 Hz that maps to a compaction every
~10 seconds under continuous accumulation, which is longer than any
single ICP frame's own budget and shorter than the pose-history trail
the twin ``g1_slam_pose_history_envelope`` (strands-labs/robots#3026,
in flight) ceiling names.

This module surfaces that ceiling to an agent so a caller planning a
long-running SLAM accumulation session can decide the chunk-list bound
decidably before a future driver-side wrapper for the runner fires,
rather than reading the neon runner's silent steal-and-dedup behaviour
off a list that has already lost its per-chunk provenance.  A caller
reading ``g1_slam_map_chunks_compaction_admits(len)`` with its intended
batch depth sees the ``100``-entry ceiling and can either shorten the
accumulation window, install its own compaction timer upstream of the
runner, or accept the runner's amortised compaction semantics.

Twin of ``g1_slam_pose_history_envelope`` (strands-labs/robots#3026,
in flight) and ``g1_slam_frame_queue_envelope``
(strands-labs/robots#3027, in flight) -- all three port a distinct
in-memory ceiling the same neon ``_SlamRunner`` reads on the same
``_process_frame`` code path.  The three stay separate because the
runner reads them on different arguments and with different remedies:

* The pose-history envelope answers a *bookkeeping* ceiling on the
  runner's own append log (``_pose_history``, refused before the
  next append lands, remedy: shorten the session or buffer to disk).
* The frame-queue envelope answers a *capacity* policy on the
  producer-side channel (``_frame_q``, a single-slot
  ``queue.Queue(maxsize=1)`` where the ``_on_cloud`` producer
  swallows ``queue.Full``, remedy: accept the freshness-preferring
  drop policy or throttle upstream).
* This map-chunks envelope answers a *compaction* trigger on the
  accumulated world-point log (``_map_chunks``, refused before the
  next append lands, remedy: shorten the accumulation window or
  install a compaction timer upstream).

Colocating them would hand an agent planner a single refusal payload
that mixed a "shorten the trail" remedy, a "accept the drop policy"
remedy, and a "shorten the accumulation window" remedy against three
distinct write paths.  Colocating them would also tie a future
compaction-batch revision (a runner patch that raised the ``100``
threshold because a faster host absorbed the dedup cost) to a
pose-history or frame-queue revision the neon runner does not couple.

Also a twin of the same-surface dedup-cell-size envelope on the same
``_SlamRunner._process_frame`` compaction pass:
:mod:`~strands_robots.tools.g1.g1_slam_voxel_dedup_envelope` (the
merged strands-labs/robots#3014) names the 5 cm *cell size* the
compaction reads at each fire; this module names the 100-chunk
*trigger count* that decides when the compaction fires at all.  They
stay separate because a caller planning a runner deployment reads
them on different decisions: the dedup-cell envelope answers "how
finely will each fire collapse the map"; this envelope answers "how
often will the fires happen at the runner's LiDAR rate".  A widen to
one would not co-widen the other -- a coarser cell would not change
the batch size, and a larger batch would not change the cell size --
so keeping them on separate surfaces keeps each widen local.

Two things this module is deliberately *not*.

* An execution path.  The neon bundle's ``_process_frame`` writes
  the chunk list on ``_SlamRunner._process_frame``'s thread;
  today's :class:`~strands_robots.drivers.g1.G1Driver` does not
  front that write, so no motion admission is at stake at this
  compaction ceiling.  A future driver method that fronts SLAM
  accumulation will surface the same module-local
  :data:`_REFUSAL_TEXT` on a caller planning a batch deeper than
  the runner's compaction trigger; this module ports the
  read-only capacity half without also introducing a second SLAM
  path the driver does not yet own, refs strands-labs/robots#358.
* An SDK re-import.  The ceiling is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, or ``kiss_icp`` submodule at import -- the
  import-hygiene contract every other file in this package
  carries, refs strands-labs/robots#358.  A caller authoring a
  SLAM plan before any SLAM extra is installed on their host still
  gets the ceiling back verbatim.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its
FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The SLAM
map-chunks compaction ceiling runs on the neon runner's own thread
against an in-memory Python list -- it never touches ``rt/lowcmd``,
never talks to the locomotion controller, and reaches no SDK RPC
service that ships an rc table for a chunk-list capacity refusal.
Borrowing ``7404`` on an over-ceiling refusal would hand an agent
planner a motion-FSM remedy for a bookkeeping argument.  The refusal
shape this module returns names the numeric bound violation in
module-local text so a planner reads a remedy that matches the
surface, and a future driver-side SLAM accumulation wrapper will
surface the same module-local text.  This mirrors the same-surface
refusal rule the twin
``g1_slam_pose_history_envelope`` (strands-labs/robots#3026, in
flight) names for the ``_process_frame`` bookkeeping ceiling and the same rule
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` (the
merged strands-labs/robots#3006) names for the ``_try_relocalize``
match-quality gate, refs strands-labs/robots#358.

What this module does not decide.

* What happens *after* the ceiling.  The neon runner does not
  refuse when the chunk list crosses ``100`` entries; it steals
  the list under its own lock, stitches every chunk into one
  ``(N, 4)`` array, dedups against the 5 cm voxel grid, and
  installs the deduped result back as the sole chunk.  This
  envelope names the ceiling so a caller who wants a
  hard-refuse-at-ceiling semantics (rather than the neon runner's
  soft-compact-and-continue one) can plan around it, not so a
  future write verb refuses at ``101``.  A future driver-side SLAM
  wrapper that answered ``g1_slam_map_chunks_compaction_admits``
  on the runner's live length would surface the same refusal text
  on a caller who wanted the batch deferred.
* The dedup cell size.  The compaction pass's dedup cell edge is
  named on the twin
  :mod:`~strands_robots.tools.g1.g1_slam_voxel_dedup_envelope`
  (the merged strands-labs/robots#3014); a caller sizing the
  deduped map reads that envelope for the per-cell footprint and
  this envelope for the per-fire batch depth, then reasons about
  the two together.
* The compaction's *time span*.  The ``100``-entry ceiling is a
  count, not a duration; at the neon runner's observed 10 Hz that
  maps to ~10 seconds of continuous accumulation, but a slower or
  faster LiDAR frame rate would produce a different span.  A
  caller who wants a time-bounded compaction cadence reads the
  LiDAR frame-rate envelope (a future ``g1_lidar_frequency_envelope``
  lookup pair would name it) and divides against this count, not
  the other way around.
* Whether accumulation is *on*.  The compaction only runs while the
  runner's ``_accumulating`` flag is true; a caller who wants to
  pin the accumulation flag reads a future
  ``g1_slam_accumulation_envelope`` lookup pair, not this one.
  This envelope answers only the numeric decision the compaction
  pass would face if accumulation *were* on.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_count_error

#: The inclusive upper bound on the ``_map_chunks`` list's length
#: that the neon SLAM runner's ``_process_frame`` fires the
#: compaction pass at.  The neon bundle reads
#: ``if len(self._map_chunks) > 100:`` and steals the list under
#: its own lock on strict-greater, so a list at exactly ``100``
#: entries is the boundary case the runner admits (the next
#: append lands at ``101`` and triggers the steal-and-dedup).
#: Named as an inclusive upper bound so
#: :func:`g1_slam_map_chunks_compaction_admits` admits a
#: caller-supplied ``100`` and refuses a caller-supplied ``101``,
#: mirroring the runner's ``>`` refusal exactly.
_MAP_CHUNKS_COMPACTION_MAX: int = 100

#: The module-local refusal text every
#: ``g1_slam_map_chunks_compaction_admits`` refusal quotes when
#: the caller-supplied batch depth sits above the
#: neon-runner-observed ceiling.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the SLAM compaction path ships no distinct rc -- the
#: neon runner just steals the chunk list on the next
#: ``_process_frame`` compaction, which the caller reads as "the
#: per-chunk provenance is gone".  The motion-FSM ``7404`` entry
#: (its nearest neighbour) reads
#: ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a remedy
#: that points a planner at locomotion FSM transitions to fix a
#: bookkeeping argument.  Surfacing the module-local text keeps
#: the refusal payload's remedy on the same surface the write
#: belongs on; a future driver-side SLAM accumulation wrapper
#: will surface this same text rather than re-borrowing a motion
#: code.
_REFUSAL_TEXT: str = (
    "map chunks compaction refused - the intended batch depth sits "
    "above the neon-runner-observed compaction ceiling; the runner "
    "steals the chunk list and dedups against a 5 cm voxel grid at "
    "this point, and the per-chunk provenance is lost. Refs "
    "strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_map_chunks_compaction_envelope` so
    :func:`g1_slam_map_chunks_compaction_admits` names the same
    field on its admitted-path payload and so a widen to the
    descriptor lands in one place.  Every field is a snapshot
    read; no bus is touched.
    """
    return {
        "map_chunks_compaction_max": _MAP_CHUNKS_COMPACTION_MAX,
    }


@tool
def g1_list_slam_map_chunks_compaction_envelope() -> dict[str, Any]:
    """Return the chunk-list compaction ceiling the neon SLAM runner fires at.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` submodule import at load time: the
    field is a module-level constant.  Useful before a future
    driver-side wrapper for SLAM accumulation is called, so a
    caller can compare an intended batch depth against the value
    the neon runner's ``_process_frame`` compaction pass reads and
    can carry the module-local refusal text a driver-side wrapper
    would surface on an over-ceiling batch.

    The envelope carries one field, the neon-observed batch depth
    trigger, because the neon runner authors one compaction
    ceiling at build time.  A caller wanting a different trigger
    today must patch the neon runner's own module; a future
    driver-side accumulation wrapper that parameterised the
    trigger would land on this envelope's
    :func:`g1_slam_map_chunks_compaction_admits` capacity grader.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed value
        (``map_chunks_compaction_max``); and a ``refusals`` list
        carrying a single descriptor with the module-local
        :data:`_REFUSAL_TEXT` a future write verb would surface
        on an over-ceiling batch.  Every field is a snapshot of
        an observed bound or a module-local text; no dynamic
        decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_slam_map_chunks_compaction_admits(
    batch_depth: int = _MAP_CHUNKS_COMPACTION_MAX,
) -> dict[str, Any]:
    """Decide whether a candidate batch depth sits at or below the compaction ceiling.

    Read-only.  Compares ``batch_depth`` against the ceiling
    :func:`g1_list_slam_map_chunks_compaction_envelope` returns
    and reports the bound the argument violates.  No driver
    instance, no DDS, no SDK, no ``numpy`` / ``open3d`` /
    ``kiss_icp`` submodule import: the decision reads only a
    module-level constant and the argument itself.

    A depth at or below the ceiling is *not* the same as an
    admitted accumulation batch: the neon runner also authors
    a dedup cell size which the twin envelope
    :mod:`~strands_robots.tools.g1.g1_slam_voxel_dedup_envelope`
    names, and a pose-history ceiling which the twin envelope
    ``g1_slam_pose_history_envelope`` (strands-labs/robots#3026,
    in flight) names.  The returned payload names only the numeric
    compaction-trigger decision.

    Args:
        batch_depth: The intended chunk-list depth, i.e. the
            number of ``_process_frame`` append calls the caller
            plans to run between compaction fires.  The default
            ``100`` (the observed ceiling) admits, so a caller who
            does not pass an explicit argument lands on the
            admitted boundary case the runner itself admits.  The
            shared :func:`~strands_robots.utils.positive_count_error`
            domain refuses non-``int`` inputs (including ``bool``,
            which is an ``int`` subclass whose ``True`` would
            otherwise be a silent ``1``), values below ``1``, and
            any type coercion that could hide a floating-point
            argument.  A depth of ``0`` (an empty batch) is
            refused by the shared domain rather than by the
            compaction ceiling because a batch that plans zero
            chunks is a shape mistake rather than a
            capacity-exceeded refusal.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the batch depth sits at or below the ceiling; a
        ``refusals`` list of refusal descriptors, each carrying
        the dimension name, the offending value, the clamp it
        violated, the comparison ("value > bound" or the
        shared-domain "shared-domain" descriptor for a shape
        mistake), and the module-local :data:`_REFUSAL_TEXT` a
        driver-side wrapper would surface if the batch were run
        through the runner without deferring; the same
        ``envelope`` sub-dict
        :func:`g1_list_slam_map_chunks_compaction_envelope`
        returns.  On an admitted depth the ``refusals`` list is
        empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check first: the shared
    # positive_count_error refuses bool, non-int, and value < 1.
    # This lands before the compaction ceiling check so a shape
    # mistake reads decidably against the module-local ceiling.
    domain_err = positive_count_error(batch_depth, "batch_depth", "g1_slam_map_chunks_compaction_admits")
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "batch_depth",
                "value": batch_depth,
                "bound_key": "map_chunks_compaction_max",
                "bound": _MAP_CHUNKS_COMPACTION_MAX,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )
    else:
        # The shared domain has admitted the shape; now grade the
        # compaction ceiling.  The runner reads
        # len(_map_chunks) > 100 and fires the compaction on
        # strict-greater, so exactly 100 admits.
        if batch_depth > _MAP_CHUNKS_COMPACTION_MAX:
            refusals.append(
                {
                    "dimension": "batch_depth",
                    "value": batch_depth,
                    "bound_key": "map_chunks_compaction_max",
                    "bound": _MAP_CHUNKS_COMPACTION_MAX,
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
