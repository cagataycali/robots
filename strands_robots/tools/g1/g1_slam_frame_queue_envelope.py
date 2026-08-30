"""Agent-facing lookup for the frame-queue capacity the neon SLAM runner drops on.

The neon bundle's SLAM runner
(``cagataycali/neon-the-g1/tools/g1_slam.py::_SlamRunner``) hands each
incoming LiDAR ``PointCloud2_`` message off to its ICP worker through a
single-slot ``queue.Queue(maxsize=1)``.  The producer side reads
``self._frame_q.put_nowait(msg)`` in ``_on_cloud`` and swallows
``queue.Full`` (``pass  # drop: worker still processing previous frame``)
-- so a message that arrives while the worker is still on the previous
frame is silently discarded, and the worker only ever registers the
most-recent-when-idle frame rather than a strict FIFO of every cloud.
The neon comment on that ceiling is that ICP registration is
throughput-bound (the worker takes tens of milliseconds per frame on
Thor) while the LiDAR ships at ~10 Hz, so a strict FIFO would grow the
queue without bound; the single-slot capacity is a *policy* -- prefer
freshness over completeness -- rather than a hardware limit.

This module surfaces that capacity to an agent so a caller planning a
frame-sensitive SLAM session (a caller who assumes every cloud reaches
the ICP registration) can decide the queue-shape bound decidably
before a future driver-side wrapper for the runner fires, rather than
reading the neon runner's silent frame-drop behaviour off a pose trail
that already skipped every cloud whose predecessor took too long.  A
caller reading ``g1_slam_frame_queue_admits(queue_capacity)`` with its
intended capacity sees the single-slot ceiling and can either accept
the freshness-preferring policy the runner ships with, throttle its
own LiDAR subscription upstream of the runner, or plan the session
against the register-most-recent-when-idle semantics.

Twin of :mod:`~strands_robots.tools.g1.g1_slam_pose_history_envelope`
(the merged strands-labs/robots#3026) on the same
:class:`~cagataycali.neon-the-g1.tools.g1_slam._SlamRunner` surface --
that envelope names the ``_process_frame`` pose-trail bookkeeping
ceiling on the runner's own append log, this envelope names the
``_on_cloud`` producer-side queue capacity on the runner's own
handoff to the ICP worker.  They stay separate because the runner
reads them in different code paths, on different arguments, with two
distinct remedies: the pose-history refusal points a planner at
"shorten the session or buffer to disk on your own timer", and this
refusal points a planner at "accept the freshness-preferring drop
policy or throttle upstream of the runner".  Colocating would hand an
agent planner a single refusal payload mixing two remedies against
two surfaces the runner reads on two different threads.

Twin also of :mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
(strands-labs/robots#3005, merged),
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
(strands-labs/robots#3006, merged),
:mod:`~strands_robots.tools.g1.g1_slam_save_envelope` and
:mod:`~strands_robots.tools.g1.g1_slam_cloud_range_envelope`: each of
those names a *result*-side or *precondition* judgement on an
argument the runner takes, and this one names a *policy* capacity on
the runner's own producer-consumer channel.  A caller planning a
long-running SLAM session composes every one of them.

Two things this module is deliberately *not*.

* An execution path.  The neon bundle's ``_on_cloud`` writes the
  queue on the DDS subscriber's own callback thread; today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front that
  write, so no motion admission is at stake at this capacity
  ceiling.  A future driver method that fronts SLAM will surface
  the same module-local :data:`_REFUSAL_TEXT` on a caller planning
  a session with a queue capacity larger than the neon runner's
  single-slot policy; this module ports the read-only capacity half
  without also introducing a second SLAM path the driver does not
  yet own, refs strands-labs/robots#358.
* An SDK re-import.  The ceiling is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_slam_frame_queue_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``,
  ``open3d``, ``kiss_icp``, or ``queue`` submodule at import -- the
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
frame-queue capacity runs on the neon runner's own producer thread
against an in-memory ``queue.Queue`` -- it never touches
``rt/lowcmd``, never talks to the locomotion controller, and reaches
no SDK RPC service that ships an rc table for a queue-capacity
refusal.  Borrowing ``7404`` on a queue-capacity refusal would hand
an agent planner a motion-FSM remedy for a bookkeeping argument.
The refusal shape this module returns names the numeric bound
violation in module-local text so a planner reads a remedy that
matches the surface, and a future driver-side SLAM session wrapper
will surface the same module-local text.  This mirrors the
same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_slam_pose_history_envelope` names
for the ``_process_frame`` trail ceiling, refs
strands-labs/robots#358.

What this module does not decide.

* What the *worker* does with the queued frame.  The neon runner's
  ``_process_loop`` reads one message at a time and hands it to
  ``_process_frame``; a queue capacity larger than one would let
  older frames pile up ahead of newer ones and the worker would
  still register them one at a time, so the registration rate
  would not change but the pose trail would lag the wall clock by
  the queue depth times the per-frame processing cost.  That
  interaction is a runner-side design choice the neon bundle
  already made; this envelope names the shape of the choice, not a
  different one.
* What the *producer* drops.  The neon runner's ``_on_cloud``
  swallows ``queue.Full`` on the ``put_nowait`` call, so a caller
  reading a per-frame counter downstream of the runner cannot tell
  a dropped-because-full frame from a never-arrived-because-LiDAR-
  stalled frame.  A future runner-side drop counter is a separate
  concern with its own refusal shape; this envelope names the
  capacity, not the observability of the drops.
* The LiDAR's frame rate.  The single-slot capacity is a *count*,
  not a frequency; the neon runner ships with a policy that
  prefers freshness at any LiDAR rate.  A caller who wants to
  reason about drop rate reads the LiDAR frequency envelope (a
  future ``g1_lidar_frequency_envelope`` lookup pair would name it)
  and multiplies against the worker's per-frame cost, not the
  other way around.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_count_error

#: The inclusive upper bound on the ``_SlamRunner._frame_q`` queue
#: capacity the neon bundle constructs.  The neon runner reads
#: ``self._frame_q: queue.Queue = queue.Queue(maxsize=1)`` at
#: ``__init__`` time -- a strict-one-slot channel -- and the
#: producer-side ``_on_cloud`` swallows ``queue.Full`` on
#: ``put_nowait`` so a message arriving while the ICP worker is
#: still on the previous frame is dropped.  Named as an inclusive
#: upper bound so :func:`g1_slam_frame_queue_admits` admits a
#: caller-supplied ``1`` and refuses a caller-supplied ``2``,
#: mirroring the runner's single-slot policy exactly.  The lower
#: bound is the shared ``positive_count_error`` domain's ``>= 1``,
#: so a capacity of ``0`` refuses on the shared floor rather than
#: the module-local ceiling (a zero-slot queue would drop every
#: frame and is a shape mistake, not a capacity-exceeded refusal).
_FRAME_QUEUE_MAX: int = 1

#: The module-local refusal text every
#: ``g1_slam_frame_queue_admits`` refusal quotes when the
#: caller-supplied queue capacity sits above the
#: neon-runner-observed ceiling.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the SLAM producer-consumer path ships no distinct rc --
#: the neon runner just swallows ``queue.Full`` on the next
#: ``_on_cloud`` callback, which the caller reads as "the frame
#: is gone, and no counter incremented".  The motion-FSM ``7404``
#: entry (its nearest neighbour) reads
#: ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a remedy
#: that points a planner at locomotion FSM transitions to fix a
#: producer-consumer capacity argument.  Surfacing the module-local
#: text keeps the refusal payload's remedy on the same surface the
#: drop belongs on; a future driver-side SLAM session wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "frame queue capacity refused - the intended queue capacity sits "
    "above the neon-runner-observed single-slot ceiling; the runner "
    "drops on queue.Full at this capacity and prefers freshness over "
    "completeness. Refs strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_slam_frame_queue_envelope` so
    :func:`g1_slam_frame_queue_admits` names the same field on its
    admitted-path payload and so a widen to the descriptor lands
    in one place.  Every field is a snapshot read; no bus is
    touched.
    """
    return {
        "frame_queue_max": _FRAME_QUEUE_MAX,
    }


@tool
def g1_list_slam_frame_queue_envelope() -> dict[str, Any]:
    """Return the frame-queue capacity the neon SLAM runner ships with.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy`` /
    ``open3d`` / ``kiss_icp`` / ``queue`` submodule import at load
    time: the field is a module-level constant.  Useful before a
    future driver-side wrapper for the SLAM producer-consumer
    channel is called, so a caller can compare an intended queue
    capacity against the single-slot policy the neon runner's
    ``_frame_q`` ships with and can carry the module-local
    refusal text a driver-side wrapper would surface on a bounds
    violation.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed ceiling (``frame_queue_max``);
        and a ``refusals`` list carrying a single descriptor with
        the module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on an above-ceiling capacity argument.
        Every field is a snapshot of an observed bound or a
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
def g1_slam_frame_queue_admits(
    queue_capacity: int = _FRAME_QUEUE_MAX,
) -> dict[str, Any]:
    """Decide whether a candidate queue capacity sits at or below the frame-queue ceiling.

    Read-only.  Compares ``queue_capacity`` against the ceiling
    :func:`g1_list_slam_frame_queue_envelope` returns and reports
    the bound the argument violates.  No driver instance, no DDS,
    no SDK, no ``numpy`` / ``open3d`` / ``kiss_icp`` / ``queue``
    submodule import: the decision reads only a module-level
    constant and the argument itself.

    A capacity at or below the ceiling is *not* the same as an
    admitted SLAM session: the neon runner also refuses on the
    match-quality dimensions (fitness, translation, trace) which
    :mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope`
    names, on the map-liveness precondition which
    :mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope`
    names, and on the pose-history bookkeeping ceiling which
    :mod:`~strands_robots.tools.g1.g1_slam_pose_history_envelope`
    names.  The returned payload names only the numeric queue
    capacity decision.

    Args:
        queue_capacity: The intended queue capacity, i.e. the
            maximum number of pending LiDAR frames the caller
            plans to let the producer buffer before the drop
            policy fires.  The default ``1`` (the observed
            ceiling) admits, so a caller who does not pass an
            explicit argument lands on the admitted boundary
            case the runner itself ships with.  The shared
            :func:`~strands_robots.utils.positive_count_error`
            domain refuses non-``int`` inputs (including
            ``bool``, which is an ``int`` subclass whose ``True``
            would otherwise be a silent ``1``), values below
            ``1``, and any type coercion that could hide a
            floating-point argument.  A capacity of ``0`` (a
            zero-slot queue that would drop every frame) is
            refused by the shared domain rather than by the
            module-local ceiling because a queue that admits no
            frames is a shape mistake rather than a
            capacity-exceeded refusal.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the queue capacity sits at or below the ceiling; a
        ``refusals`` list of refusal descriptors, each carrying
        the dimension name, the offending value, the clamp it
        violated, the comparison ("value > bound" or the
        shared-domain "shared-domain" descriptor for a shape
        mistake), and the module-local :data:`_REFUSAL_TEXT` a
        driver-side wrapper would surface if the queue capacity
        were carried into a future runner-side constructor
        without honouring the single-slot policy; the same
        ``envelope`` sub-dict
        :func:`g1_list_slam_frame_queue_envelope` returns.  On an
        admitted capacity the ``refusals`` list is empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check first: the shared
    # positive_count_error refuses bool, non-int, and value < 1.
    # This lands before the frame-queue ceiling check so a shape
    # mistake reads decidably against the module-local ceiling.
    domain_err = positive_count_error(queue_capacity, "queue_capacity", "g1_slam_frame_queue_admits")
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "queue_capacity",
                "value": queue_capacity,
                "bound_key": "frame_queue_max",
                "bound": _FRAME_QUEUE_MAX,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )
    else:
        # The shared domain has admitted the shape; now grade the
        # frame-queue ceiling.  The runner constructs
        # ``queue.Queue(maxsize=1)`` and drops on queue.Full, so
        # exactly 1 admits and anything strictly greater refuses
        # against the observed policy.
        if queue_capacity > _FRAME_QUEUE_MAX:
            refusals.append(
                {
                    "dimension": "queue_capacity",
                    "value": queue_capacity,
                    "bound_key": "frame_queue_max",
                    "bound": _FRAME_QUEUE_MAX,
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
