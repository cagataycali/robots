"""Agent-facing lookup for the DDS subscribe buffer envelope the neon bundle admits.

The neon bundle's DDS subscribe verb
(``cagataycali/neon-the-g1/tools/g1_dds.py::g1_dds_subscribe``) takes a
``max_buffer`` argument that names the ``collections.deque`` ``maxlen``
on the per-topic subscription handle
(``cagataycali/neon-the-g1/tools/_dds_engine.py::_SubHandle.buffer``).
The subscribe callback appends every decoded message to the deque under
its per-handle ``threading.Lock``; when the deque is full the oldest
entry drops off the left end and the new entry lands on the right end,
so ``max_buffer`` names the *window* on the topic's recent-message ring
that :func:`g1_dds_read` can read back rather than the *rate* at which
messages arrive.  The neon bundle's default is ``max_buffer=20`` (a
value tuned for the agent-facing snapshot verb's token budget on one
read); the argument is forwarded verbatim to :func:`collections.deque`
which admits ``maxlen=0`` (drops every message on arrival) at the
Python level, so a caller who passes ``max_buffer=0`` reaches a
subscription that decodes every message but returns an empty read.

This module snapshots the observed envelope into module-level
constants and exposes two agent-facing verbs so a caller can decide
the refusal decidably before a future driver-side DDS subscribe path
is called, rather than pinning the range inside the write path where
the refusal is invisible to the planner.

Twin of :mod:`~strands_robots.tools.g1.g1_dds_topic_categories` and
:mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types`, which surface
the *topic* and *IDL type* dimensions on the same subscribe surface
rather than the *buffer size* dimension.  The three modules stay
separate because the topic name is a per-request discovery decision,
the IDL type is a per-request decode decision, and the buffer size is
a per-handle state decision -- three different surfaces with disjoint
refusal shapes.  Colocating them here would hand an agent planner a
single refusal payload that mixed the three remedies and would tie a
future IDL-catalog revision to a buffer-size revision the neon bundle
does not couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``g1_dds_subscribe`` verb
  reaches for ``unitree_sdk2py.core.channel.ChannelSubscriber`` under
  the singleton ``_DDS_INIT_LOCK`` and installs the buffered callback
  on the SDK-owned reader thread; that subscribe path is the same
  DDS-catalog path the driver's future generic-subscribe wrapper
  would front.  A future driver method that fronts
  ``g1_dds_subscribe`` will land the transition verb; refs
  strands-labs/robots#358 for the SDK-facing gate work that
  DDS-subscribe path belongs on.  This module ports the read-only
  envelope half without also introducing a second DDS-subscribe path
  the driver does not yet own.
* An SDK re-import.  The envelope is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_dds_max_buffer_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no
  :mod:`collections` submodule at import time -- the import-hygiene
  contract every other file in this package carries, refs
  strands-labs/robots#358.  A revision of the observed bounds is a
  driver-side update; when the driver's DDS-subscribe method lands,
  its refusal will surface the same module-local :data:`_REFUSAL_TEXT`
  this module names for a bounds violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The DDS
subscription handle sits on the SDK-owned reader thread in the Python
process itself -- it never touches ``rt/lowcmd`` and never touches an
RPC service the SDK ships an rc table for -- so the DDS-subscribe
handle ships no distinct rc for a bounds-violated ``max_buffer``
argument.  Borrowing ``7404`` on a buffer-size refusal would hand an
agent planner a motion-FSM remedy (``"need FSM in {500, 501, 801}"``)
for a bounds violation on a value that has nothing to do with the
locomotion FSM.  The refusal shape this module returns names the
numeric bound violation in module-local text so a planner reads a
remedy that matches the surface, and a future driver-side
``g1_dds_subscribe`` wrapper will surface the same module-local text
-- not a re-borrowed motion code.  This mirrors the same-surface
refusal rule
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
names for ``pywebrtc_audio.AudioProcessor(stream_delay_ms=...)``,
refs strands-labs/robots#358.

What this module does not decide.

* Whether the topic is currently published.  Whether the Livox
  firmware is up, whether the mainboard is emitting ``rt/lowstate``,
  whether the audio bus is silent: none of those live-instance reads
  run here.  A caller planning a DDS subscribe reads this verb's
  ``envelope`` to decide whether their ``max_buffer`` argument is
  inside the observed range and reads the driver's own topic-liveness
  signal separately (a future ``g1_dds_liveness`` companion verb) to
  decide whether the topic is currently emitting.
* Whether the buffer window loses a message.  A caller reading with
  :func:`g1_dds_read` after the deque has cycled past the message
  they wanted reads only the last ``max_buffer`` decoded entries;
  whether the read *misses* a message the caller cared about is a
  timing decision the caller owns (raise ``max_buffer`` or read more
  often), not a numeric bound decision.  The refusal shape this
  module surfaces names only whether the argument sits inside the
  observed ``[1, 10000]`` range -- a caller pinning a smaller value
  that passes the bound but starves the read reads that starve on
  the driver's own subscribe verb, not on this envelope.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The lower clamp on ``max_buffer`` (inclusive integer).  The neon
#: bundle passes the argument verbatim to :func:`collections.deque`
#: which admits ``maxlen=0`` at the Python level (drops every message
#: on arrival, decodes-but-returns-empty on read), but a caller
#: passing ``max_buffer=0`` is asking for a subscription that
#: guarantees an empty :func:`g1_dds_read` on every call -- a shape
#: mistake rather than a smaller-window request, so the bound is
#: inclusive at ``1`` and refuses below it.  A negative value is
#: refused by :func:`collections.deque` at ``__init__`` time with
#: ``ValueError`` and would raise on the driver side; naming the
#: refusal at the boundary surfaces the shape mistake before the
#: subscribe path is entered.
_MAX_BUFFER_MIN: int = 1

#: The upper clamp on ``max_buffer``.  The DDS buffer stores each
#: entry as a ``(timestamp, message)`` tuple where the message is a
#: decoded IDL object; a G1 :class:`PointCloud2_` message on
#: ``rt/utlidar/cloud`` carries a byte-string of ~24000 * 16 = 384000
#: bytes per frame, so a ``max_buffer=10000`` PointCloud2 subscription
#: reserves ~3.84 GB of live memory before eviction begins.  A ceiling
#: of ``10000`` names the practical upper bound where a caller
#: reaching for the largest useful window still stays inside the
#: process RSS a mesh peer allocates; above that number the caller
#: is more likely to reach the driver's own RSS bound than the
#: subscribe path's own limit.  Named as an inclusive bound like the
#: lower clamp so a caller writing ``max_buffer=10000`` (the observed
#: ceiling) is admitted rather than tripping an off-by-one.
_MAX_BUFFER_MAX: int = 10000

#: The neon-bundle-tuned default the ``g1_dds_subscribe`` verb reaches
#: for when the caller does not name ``max_buffer``.  The neon
#: verb signature names ``max_buffer: int = 20`` -- a value tuned for
#: the agent-facing snapshot verb's token budget on one
#: :func:`g1_dds_read` payload.  Surfaced here so a caller planning a
#: DDS subscribe with the neon-tuned value can name the same integer
#: without re-measuring the token cost, and so a widen or narrow to
#: the observed default lands in one place.
_MAX_BUFFER_NEON_DEFAULT: int = 20

#: The module-local refusal text every ``g1_max_buffer_admits``
#: refusal quotes when the caller's argument sits outside the
#: neon-bundle-observed envelope.  Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the DDS-subscribe path ships no distinct rc for a
#: bounds-violated ``max_buffer`` argument and the motion-FSM
#: ``7404`` entry (its nearest neighbour) reads ``"Invalid FSM id -
#: need FSM in {500, 501, 801}"`` -- a remedy that points a planner
#: at locomotion FSM transitions to fix a buffer-size argument.
#: Surfacing the module-local text keeps the refusal payload's
#: remedy on the same surface the write belongs on, and a future
#: driver-side ``g1_dds_subscribe`` wrapper will surface this same
#: text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = f"max_buffer out of envelope - need max_buffer in [{_MAX_BUFFER_MIN}, {_MAX_BUFFER_MAX}]"


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_dds_max_buffer_envelope` so
    :func:`g1_max_buffer_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "max_buffer_min": _MAX_BUFFER_MIN,
        "max_buffer_max": _MAX_BUFFER_MAX,
        "max_buffer_neon_default": _MAX_BUFFER_NEON_DEFAULT,
    }


@tool
def g1_list_dds_max_buffer_envelope() -> dict[str, Any]:
    """Return the ``max_buffer`` envelope the neon bundle observed as usable.

    Read-only.  No driver instance, no DDS, no SDK, no
    :mod:`collections` submodule import at load time: every field is
    a module-level constant.  Useful before a future driver-side
    wrapper for ``g1_dds_subscribe`` is called, so a caller can
    compare an intended ``max_buffer`` argument against the envelope
    the neon bundle observed as usable and can carry the module-local
    refusal text a driver-side wrapper would surface on a bounds
    violation.  The neon-tuned default (``20`` for the agent-facing
    read verb's token budget) is named on the envelope so a caller
    who wants the neon-observed value can pin it without re-measuring
    the deque overhead.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon bundle observed
        (``max_buffer_min``, ``max_buffer_max``,
        ``max_buffer_neon_default``); and a ``refusals`` list
        carrying a single descriptor with the module-local
        :data:`_REFUSAL_TEXT` a future write verb would surface on a
        bounds violation.  Every field is a snapshot of an observed
        bound or a module-local text; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_max_buffer_admits(max_buffer: int = 20) -> dict[str, Any]:
    """Decide whether a ``max_buffer`` argument sits inside the envelope.

    Read-only.  Compares the argument against the clamps
    :func:`g1_list_dds_max_buffer_envelope` returns and reports the
    refusal shape if any bound is violated.  No driver instance, no
    DDS, no SDK, no :mod:`collections` submodule import: the decision
    reads only module-level constants and the argument itself.

    A ``max_buffer`` inside the envelope is *not* the same as an
    admitted subscribe: the driver's DDS singleton may refuse on
    liveness grounds (topic not published, IDL type not resolvable,
    subscription already open), which this verb does not read (that
    is a live driver-instance query answered by a future
    ``g1_dds_liveness`` verb).  The returned envelope names only the
    numeric bound decision.

    Args:
        max_buffer: integer deque window in ``[1, 10000]``.  The
            default ``20`` matches the neon-bundle-tuned value for
            the agent-facing subscribe verb, so a caller who does
            not pass an explicit argument lands on the neon-observed
            admitted value.  Refused below ``max_buffer_min`` (a
            value of ``0`` reserves a deque that decodes every
            message and drops it before :func:`g1_dds_read` can
            observe it; a negative value raises ``ValueError`` on
            :class:`collections.deque` construction) and above
            ``max_buffer_max`` (the practical RSS ceiling; a
            ``PointCloud2_`` window that large reserves ~3.84 GB
            before eviction begins, more likely to reach the process
            RSS bound than the subscribe path's own limit).  Boolean
            values are refused explicitly at the boundary because
            Python's ``bool`` is a subclass of ``int``, so a caller
            passing ``True`` would otherwise silently look up ``1``
            (a legitimate one-message window) and hide the type
            mistake; naming the refusal at the boundary surfaces the
            mistake instead.  Non-integer numeric values (``float``,
            ``Decimal``) are refused with the same shape so a caller
            passing ``max_buffer=20.0`` sees an actionable refusal
            rather than a silent ``TypeError`` the deque constructor
            would raise on the driver side.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value is inside the clamp pair; a ``refusals`` list of
        refusal descriptors, each carrying the dimension name, the
        offending value, the clamp it violated, and the
        module-local :data:`_REFUSAL_TEXT` a driver-side wrapper
        would surface if the subscribe were attempted while the
        value is outside the envelope; the same ``envelope``
        sub-dict :func:`g1_list_dds_max_buffer_envelope` returns.
        On an admitted value the ``refusals`` list is empty; on a
        rejected value the single violated bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(value: Any, bound_key: str, bound: int, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "max_buffer",
                "value": value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "text": _REFUSAL_TEXT,
            }
        )

    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 and hide a type mistake at the boundary.
    if isinstance(max_buffer, bool):
        _reject(max_buffer, "max_buffer_min", _MAX_BUFFER_MIN, "non-int")
    elif not isinstance(max_buffer, int):
        _reject(max_buffer, "max_buffer_min", _MAX_BUFFER_MIN, "non-int")
    else:
        v = int(max_buffer)
        if v < _MAX_BUFFER_MIN:
            _reject(max_buffer, "max_buffer_min", _MAX_BUFFER_MIN, "value < bound")
        elif v > _MAX_BUFFER_MAX:
            _reject(max_buffer, "max_buffer_max", _MAX_BUFFER_MAX, "value > bound")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
