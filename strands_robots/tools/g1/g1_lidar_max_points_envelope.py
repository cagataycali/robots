"""Agent-facing lookup for the lidar downsample envelope the neon bundle admits.

The neon bundle's lidar snapshot verb
(``cagataycali/neon-the-g1/tools/g1_lidar.py::g1_lidar_snapshot``) takes
a ``max_points`` argument that names the stride-based downsample target
for one Livox MID-360 :class:`PointCloud2_` frame.  The parser
(``_cloud_to_numpy`` in the same neon module) divides the frame's
observed point count by ``max_points`` to pick a stride, so the
argument names an *upper* bound on the returned array's length rather
than an exact count -- a caller who passes ``max_points=4000`` on a
frame that carries ~4200 points reads back the whole 4200-point array
because the stride computed as ``max(1, total_points // max_points)``
collapses to ``1``.  The neon bundle names two observed values on the
same argument tuple: the agent-facing ``g1_lidar_snapshot`` verb
defaults ``max_points=4000`` (a token-budgeted subsample the neon
verb docstring names as its default), and the SLAM feeder path
(``g1_slam.py``) calls ``_cloud_to_numpy(msg, max_points=50000)`` with
the pipeline-internal ceiling that keeps the ICP registration cost
bounded on a full Livox frame.  This module snapshots the observed
envelope into module-level constants and exposes two agent-facing
verbs so a caller can decide the refusal decidably before a future
driver-side ``g1_lidar_snapshot`` wrapper is called, rather than
pinning the range inside the write path where the refusal is invisible
to the planner.

Twin of :mod:`~strands_robots.tools.g1.g1_capture_rate_candidates`,
which surfaces the *rate* dimension on the same lidar surface
(``sample_rate_hz`` on the Livox MID-360 capture) rather than the
*downsample cap* dimension.  The two modules stay separate because
the capture rate is a driver-side subscribe cadence (Livox firmware
argument on ``rt/utlidar/cloud``) while the downsample cap is a
caller-side parser argument on the already-received frame: two
different surfaces with disjoint refusal shapes.  Colocating them
here would hand an agent planner a single refusal payload that mixed
the capture and downsample remedies and would tie a future Livox
firmware revision to a caller-side parser revision the neon bundle
does not couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``g1_lidar_snapshot`` verb
  subscribes ``rt/utlidar/cloud`` under a module-level ``_LOCK`` and
  copies the latest cached frame into a numpy array; that subscribe
  path is the same lidar-DDS path the driver's future lidar wrapper
  would front.  A future driver method that fronts ``g1_lidar_snapshot``
  will land the transition verb; refs strands-labs/robots#358 for the
  SDK-facing gate work that lidar path belongs on.  This module ports
  the read-only envelope half without also introducing a second
  lidar-DDS reader path the driver does not yet own.
* An SDK re-import.  The envelope is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_lidar_max_points_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no ``numpy``
  submodule at import time (the parser's numpy stride math is a
  driver-side compute; the envelope names only the numeric bound
  decision) -- the import-hygiene contract every other file in this
  package carries, refs strands-labs/robots#358.  A revision of the
  observed bounds is a driver-side update; when the driver's lidar
  wrapper lands, its refusal will surface the same module-local
  :data:`_REFUSAL_TEXT` this module names for a bounds violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The Livox MID-360
frame parser runs on the caller's Python thread in-process -- it never
touches ``rt/lowcmd`` and never touches an RPC service the SDK ships
an rc table for -- so the lidar-frame parser ships no distinct rc for
a bounds-violated ``max_points`` argument.  Borrowing ``7404`` on a
downsample-cap refusal would hand an agent planner a motion-FSM
remedy (``"need FSM in {500, 501, 801}"``) for a bounds violation on
a value that has nothing to do with the locomotion FSM.  The refusal
shape this module returns names the numeric bound violation in
module-local text so a planner reads a remedy that matches the
surface, and a future driver-side ``g1_lidar_snapshot`` wrapper will
surface the same module-local text -- not a re-borrowed motion code.
This mirrors the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
names for ``pywebrtc_audio.AudioProcessor(stream_delay_ms=...)``, refs
strands-labs/robots#358.

What this module does not decide.

* The live lidar state.  Whether ``_ensure_subs`` has subscribed
  ``rt/utlidar/cloud``, whether the latest cache carries a fresh
  frame, whether the Livox firmware is publishing at all: none of
  those live-instance reads run here.  A caller planning a lidar
  snapshot reads this verb's ``envelope`` to decide whether their
  ``max_points`` argument is inside the observed range and reads the
  driver's own lidar liveness signal separately (a future
  ``g1_lidar_state`` companion verb) to decide whether a frame is
  currently available.
* Whether the downsample loses a feature.  A stride-based subsample
  admits at ``max_points=4000`` may drop a feature the ICP pipeline
  needs (a wall corner sparsely sampled at the frame's periphery);
  whether the resulting 4000-point cloud is *sufficient* for a
  downstream pipeline is a semantic decision the pipeline owns, not
  a numeric bound decision.  The refusal shape this module surfaces
  names only whether the argument sits inside the observed
  ``[1, 50000]`` range -- a caller pinning a smaller value that
  passes the bound but starves the ICP fitness reads that fitness on
  the driver's own SLAM verb, not on this envelope.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The lower clamp on ``max_points`` (inclusive integer).  The parser's
#: stride formula ``max(1, total_points // max_points)`` divides by
#: ``max_points`` directly; a value of ``0`` would raise
#: ``ZeroDivisionError`` on the parser side and a negative value would
#: hand ``//`` a signed divisor that flips the sign of the stride and
#: yields an empty slice.  Named as an inclusive bound
#: (``value < bound`` refuses, ``value == bound`` admits) with a floor
#: of ``1`` because a caller asking for exactly one point from the
#: frame is a legitimate token-cheap probe (the single-point bbox is
#: still a signal on whether the lidar is publishing at all) rather
#: than the shape-error the refusal exists to catch.
_MAX_POINTS_MIN: int = 1

#: The upper clamp on ``max_points``.  The neon SLAM feeder path
#: (``cagataycali/neon-the-g1/tools/g1_slam.py``) calls the parser
#: with ``max_points=50000`` and that ceiling is the pipeline-observed
#: bound the ICP registration cost stays tractable at: above that
#: number the stride collapses to ``1`` on every Livox MID-360 frame
#: (which fires ~24000 points per 100 ms sweep), so a caller asking
#: for more points than the frame carries is a shape mistake that
#: reads the whole frame anyway.  Named as an inclusive bound like
#: the lower clamp so a caller writing ``max_points=50000`` (the SLAM
#: ceiling) is admitted rather than tripping an off-by-one.
_MAX_POINTS_MAX: int = 50000

#: The neon-bundle-tuned default the ``g1_lidar_snapshot`` verb
#: reaches for when the caller does not name ``max_points``.  The
#: neon docstring for the verb names ``4000`` as the "downsample
#: target (stride-based). Default 4000." -- a value tuned for the
#: agent-facing token budget on one snapshot payload.  Surfaced here
#: so a caller planning a lidar snapshot with the neon-tuned value
#: can name the same integer without re-measuring the token cost, and
#: so a widen or narrow to the observed default lands in one place.
_MAX_POINTS_NEON_DEFAULT: int = 4000

#: The pipeline-internal downsample value the SLAM feeder passes.
#: Named on the envelope so a caller planning a SLAM-feeder-style
#: read can reach for the same integer without re-deriving it from
#: the ICP fitness threshold, and so a driver-side SLAM revision
#: that widens the ceiling lands in one place.
_MAX_POINTS_SLAM_INTERNAL: int = 50000

#: The module-local refusal text every ``g1_max_points_admits``
#: refusal quotes when the caller's argument sits outside the
#: neon-bundle-observed envelope.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the lidar frame parser ships no distinct rc for a
#: bounds-violated downsample-cap argument and the motion-FSM
#: ``7404`` entry (its nearest neighbour) reads ``"Invalid FSM id -
#: need FSM in {500, 501, 801}"`` -- a remedy that points a planner
#: at locomotion FSM transitions to fix a downsample argument.
#: Surfacing the module-local text keeps the refusal payload's
#: remedy on the same surface the write belongs on, and a future
#: driver-side lidar-snapshot wrapper will surface this same text
#: rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = f"max_points out of envelope - need max_points in [{_MAX_POINTS_MIN}, {_MAX_POINTS_MAX}]"


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_lidar_max_points_envelope` so
    :func:`g1_max_points_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands
    in one place.  Every field is a snapshot read; no bus is
    touched.
    """
    return {
        "max_points_min": _MAX_POINTS_MIN,
        "max_points_max": _MAX_POINTS_MAX,
        "max_points_neon_default": _MAX_POINTS_NEON_DEFAULT,
        "max_points_slam_internal": _MAX_POINTS_SLAM_INTERNAL,
    }


@tool
def g1_list_lidar_max_points_envelope() -> dict[str, Any]:
    """Return the ``max_points`` envelope the neon bundle observed as usable.

    Read-only.  No driver instance, no DDS, no SDK, no ``numpy``
    submodule import at load time: every field is a module-level
    constant.  Useful before a future driver-side wrapper for
    ``g1_lidar_snapshot`` is called, so a caller can compare an
    intended ``max_points`` argument against the envelope the neon
    bundle observed as usable and can carry the module-local
    refusal text a driver-side wrapper would surface on a bounds
    violation.  Two neon-observed values are named on the envelope:
    the agent-facing default (``4000``) tuned for one snapshot's
    token budget, and the SLAM-feeder internal value (``50000``)
    that keeps the ICP registration cost bounded on a full Livox
    MID-360 frame.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon bundle observed
        (``max_points_min``, ``max_points_max``,
        ``max_points_neon_default``, ``max_points_slam_internal``);
        and a ``refusals`` list carrying a single descriptor with
        the module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on a bounds violation.  Every field is a
        snapshot of an observed bound or a module-local text; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_max_points_admits(max_points: int = 4000) -> dict[str, Any]:
    """Decide whether a ``max_points`` argument sits inside the envelope.

    Read-only.  Compares the argument against the clamps
    :func:`g1_list_lidar_max_points_envelope` returns and reports
    the refusal shape if any bound is violated.  No driver
    instance, no DDS, no SDK, no ``numpy`` submodule import: the
    decision reads only module-level constants and the argument
    itself.

    A ``max_points`` inside the envelope is *not* the same as an
    admitted read: the driver's lidar singleton may refuse on
    liveness grounds (no fresh frame in the cache, subscribe not
    yet acked by the Livox firmware, ``rt/utlidar/cloud`` bus
    silent), which this verb does not read (that is a live
    driver-instance query answered by the existing lidar state
    verb).  The returned envelope names only the numeric bound
    decision.

    Args:
        max_points: integer downsample target in ``[1, 50000]``.
            The default ``4000`` matches the neon-bundle-tuned value
            for the agent-facing snapshot verb, so a caller who
            does not pass an explicit argument lands on the
            neon-observed admitted value.  Refused below
            ``max_points_min`` (a value of ``0`` would raise
            ``ZeroDivisionError`` on the parser's stride divide and
            a negative value would flip the sign of the stride)
            and above ``max_points_max`` (the SLAM-feeder ceiling;
            larger values collapse the stride to ``1`` on every
            Livox MID-360 frame anyway so the refusal surfaces the
            ceiling rather than letting the caller pay the token
            cost for a value that has no effect).  Boolean values
            are refused explicitly at the boundary because Python's
            ``bool`` is a subclass of ``int``, so a caller passing
            ``True`` would otherwise silently look up ``1`` (a
            legitimate one-point probe) and hide the type mistake;
            naming the refusal at the boundary surfaces the
            mistake instead.  Non-integer numeric values
            (``float``, ``Decimal``) are refused with the same
            shape so a caller passing ``max_points=4000.0`` sees an
            actionable refusal rather than a silent truncation the
            parser's ``//`` operator would perform.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value is inside the clamp pair; a ``refusals`` list of
        refusal descriptors, each carrying the dimension name, the
        offending value, the clamp it violated, and the
        module-local :data:`_REFUSAL_TEXT` a driver-side wrapper
        would surface if the read were attempted while the value
        is outside the envelope; the same ``envelope`` sub-dict
        :func:`g1_list_lidar_max_points_envelope` returns.  On an
        admitted value the ``refusals`` list is empty; on a
        rejected value the single violated bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(value: Any, bound_key: str, bound: int, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "max_points",
                "value": value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "text": _REFUSAL_TEXT,
            }
        )

    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 and hide a type mistake at the boundary.
    if isinstance(max_points, bool):
        _reject(max_points, "max_points_min", _MAX_POINTS_MIN, "non-int")
    elif not isinstance(max_points, int):
        _reject(max_points, "max_points_min", _MAX_POINTS_MIN, "non-int")
    else:
        v = int(max_points)
        if v < _MAX_POINTS_MIN:
            _reject(max_points, "max_points_min", _MAX_POINTS_MIN, "value < bound")
        elif v > _MAX_POINTS_MAX:
            _reject(max_points, "max_points_max", _MAX_POINTS_MAX, "value > bound")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
