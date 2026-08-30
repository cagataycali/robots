"""Agent-facing lookup for the AEC stream-delay envelope the neon bidi bundle admits.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) constructs a
WebRTC :class:`AudioProcessor` from ``pywebrtc_audio`` with a
``stream_delay_ms`` argument that names the speaker->mic loopback
delay AEC must compensate for on the G1's DDS audio path. The
processor itself places *no* upper clamp on that argument beyond
the WebRTC library's internal handling: a caller that passes
``stream_delay_ms=100000`` reaches the ``AudioProcessor``
constructor unchanged, and WebRTC's echo canceller silently
truncates every value past its own compile-time delay-buffer bound
to that bound - the caller learns nothing about the truncation and
the echo signal on the mic degrades as the near/far alignment
drifts. The neon bundle names the G1-DDS-tuned value as a
module-level default (``DEFAULT_STREAM_DELAY_MS = 120``) and its
verb docstring (``g1_speak``) surfaces the same number as the one
"for G1 DDS path"; this module snapshots the observed range into
module-level constants and exposes two agent-facing verbs so a
caller can decide the refusal decidably before a future
driver-side wrapper for the bidi audio IO is called, rather than
pinning the range inside the write path where the refusal is
invisible to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_speak(action="start")``
  verb spins up a ``G1BidiAudioIO`` under a single-thread lock;
  that construction is the same audio-processing path the driver's
  future bidi wrapper would front. A future driver method that
  fronts the bidi IO will land the write verb; refs
  strands-labs/robots#358 for the SDK-facing gate work that write
  belongs on. This module ports the read-only envelope half
  without also introducing a second audio-processing writer path
  the driver does not yet own.
* An SDK re-import. The clamp table is captured here as
  module-level constants so ``import
  strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope``
  pulls no ``unitree_sdk2py`` submodule and no ``pywebrtc_audio``
  submodule - the import-hygiene contract every other file in
  this package carries, refs strands-labs/robots#358. A revision
  of the observed bounds is a driver-side update; when the
  driver's bidi audio method lands, its refusal will surface the
  same module-local :data:`_REFUSAL_TEXT` this module names for a
  bounds violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its
FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``). The
``AudioProcessor`` runs on the mic pre-processing thread in the
Python process itself - it never touches ``rt/lowcmd`` and never
touches an RPC service the SDK ships an rc table for - so the
audio-processing SDK ships no distinct rc for a bounds-violated
``stream_delay_ms``. Borrowing ``7404`` on a stream-delay refusal
would hand an agent planner a motion-FSM remedy (``"need FSM in
{500, 501, 801}"``) for a bounds violation on a value that has
nothing to do with the locomotion FSM. The refusal shape this
module returns names the numeric bound violation in module-local
text so a planner reads a remedy that matches the surface, and a
future driver-side bidi audio wrapper will surface the same
module-local text - not a re-borrowed motion code. This mirrors
the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_audio_volume_envelope` names for
``AudioClient.SetVolume``, refs strands-labs/robots#358.

What this module does not decide.

* The live bidi state. Whether ``G1BidiAudioIO`` is currently
  constructed, whether the mic autopick has resolved a device,
  whether the chest speaker writer thread has started: none of
  those live-instance reads run here. A caller planning a bidi
  start reads this verb's ``envelope`` to decide whether their
  ``stream_delay_ms`` is inside the observed range and reads the
  driver's own liveness signal separately to decide whether the
  bidi path is currently free.
* Whether the far-buffer reference queue is currently draining.
  The ``AudioProcessor`` needs a fresh far-buffer frame each
  callback for AEC to subtract correctly; whether the DDS
  PlayStream writer thread is running and feeding the far-buffer
  is a live driver-instance query the neon bundle answers under
  ``STATS["ref_buf_qsize"]``, not a numeric envelope decision.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The lower clamp on ``stream_delay_ms`` (integer milliseconds).
#: The neon bundle passes a non-negative default (``120``) and the
#: WebRTC :class:`AudioProcessor` constructor treats a negative
#: argument as "no delay compensation" - a value below zero is a
#: shape mistake rather than a stronger compensation request, so
#: the bound is inclusive of zero and refuses below it. Named as
#: an inclusive bound (``value < bound`` refuses, ``value == bound``
#: admits) because a caller disabling the AEC delay hint with
#: ``stream_delay_ms=0`` is a legitimate command on a headphone-out
#: path where the loopback is negligible, not the shape-error the
#: refusal exists to catch.
_STREAM_DELAY_MS_MIN: int = 0

#: The upper clamp on ``stream_delay_ms``. WebRTC's
#: ``AudioProcessor`` truncates every value past ~500 ms to its
#: internal delay-buffer bound (``kEchoPathDelayEstimatorConfig``)
#: without surfacing the truncation to the caller; above the bound
#: the near/far alignment degrades silently and the echo signal
#: leaks through the mic. Named as an inclusive bound like the
#: lower clamp, so a caller writing ``stream_delay_ms=500`` (the
#: WebRTC ceiling) is admitted rather than tripping an off-by-one.
_STREAM_DELAY_MS_MAX: int = 500

#: The neon-bundle-tuned default the ``G1BidiAudioIO`` constructor
#: reaches for when the caller does not name ``stream_delay_ms``.
#: The neon docstring for ``g1_speak`` names ``120`` as the value
#: "for G1 DDS path" - a measurement the neon bundle took against
#: the G1's chest speaker roundtrip on ``rt/audio_stream`` and its
#: ``PlayStream``-return-time far-buffer feed. Surfaced here so a
#: caller planning a bidi start with the neon-tuned value can name
#: the same integer without re-measuring it, and so a widen or
#: narrow to the observed default lands in one place.
_STREAM_DELAY_MS_NEON_DEFAULT: int = 120

#: The module-local refusal text every ``g1_stream_delay_ms_admits``
#: refusal quotes when the caller's argument sits outside the
#: neon-bundle-observed envelope. Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the audio-processing pipeline ships no distinct rc for
#: a bounds-violated stream-delay argument and the motion-FSM
#: ``7404`` entry (its nearest neighbour) reads ``"Invalid FSM id -
#: need FSM in {500, 501, 801}"`` - a remedy that points a planner
#: at locomotion FSM transitions to fix an audio pre-processing
#: argument. Surfacing the module-local text keeps the refusal
#: payload's remedy on the same surface the write belongs on, and
#: a future driver-side bidi audio wrapper will surface this same
#: text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    f"stream_delay_ms out of envelope - need stream_delay_ms in [{_STREAM_DELAY_MS_MIN}, {_STREAM_DELAY_MS_MAX}]"
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bidi_audio_stream_delay_envelope` so
    :func:`g1_stream_delay_ms_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is
    touched.
    """
    return {
        "stream_delay_ms_min": _STREAM_DELAY_MS_MIN,
        "stream_delay_ms_max": _STREAM_DELAY_MS_MAX,
        "stream_delay_ms_neon_default": _STREAM_DELAY_MS_NEON_DEFAULT,
    }


@tool
def g1_list_bidi_audio_stream_delay_envelope() -> dict[str, Any]:
    """Return the ``stream_delay_ms`` envelope the neon bundle observed as usable.

    Read-only. No driver instance, no DDS, no SDK, no
    ``pywebrtc_audio`` import: every field is a module-level
    constant. Useful before a future driver-side wrapper for
    ``G1BidiAudioIO`` is called, so a caller can compare an
    intended ``stream_delay_ms`` argument against the envelope the
    neon bundle observed as usable and can carry the module-local
    refusal text a driver-side wrapper would surface on a bounds
    violation. The neon-tuned default (``120`` ms for the G1 DDS
    path) is named on the envelope so a caller who wants the
    neon-observed value can pin it without re-measuring the
    speaker->mic loopback.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon bundle observed
        (``stream_delay_ms_min``, ``stream_delay_ms_max``,
        ``stream_delay_ms_neon_default``); and a ``refusals`` list
        carrying a single descriptor with the module-local
        :data:`_REFUSAL_TEXT` a future write verb would surface on
        a bounds violation. Every field is a snapshot of an
        observed bound or a module-local text; no dynamic decode
        runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_stream_delay_ms_admits(stream_delay_ms: int = 120) -> dict[str, Any]:
    """Decide whether a ``stream_delay_ms`` argument sits inside the envelope.

    Read-only. Compares the argument against the clamps
    :func:`g1_list_bidi_audio_stream_delay_envelope` returns and
    reports the refusal shape if any bound is violated. No driver
    instance, no DDS, no SDK, no ``pywebrtc_audio`` import: the
    decision reads only module-level constants and the argument
    itself.

    A ``stream_delay_ms`` inside the envelope is *not* the same as
    an admitted write: the driver's audio singleton may refuse on
    liveness grounds (an in-flight bidi run, a not-yet-constructed
    ``AudioProcessor``, a stalled far-buffer feed), which this verb
    does not read (that is a live driver-instance query answered
    by a future bidi state verb). The returned envelope names only
    the numeric bound decision.

    Args:
        stream_delay_ms: integer milliseconds in ``[0, 500]``. The
            default ``120`` matches the neon-bundle-tuned value for
            the G1 DDS speaker->mic loopback so a caller who does
            not pass an explicit argument lands on the neon-observed
            admitted value. Refused below ``stream_delay_ms_min``
            (a negative delay hint) and above
            ``stream_delay_ms_max`` (WebRTC truncates values past
            its internal delay-buffer bound silently, so the
            refusal surfaces the ceiling rather than letting the
            AEC quality degrade off-screen). Boolean values are
            refused explicitly at the boundary because Python's
            ``bool`` is a subclass of ``int``, so a caller passing
            ``True`` would otherwise silently look up ``1`` (a
            legitimate one-millisecond hint) and hide the type
            mistake; naming the refusal at the boundary surfaces
            the mistake instead. Non-integer numeric values
            (``float``, ``Decimal``) are refused with the same
            shape so a caller passing ``stream_delay_ms=120.0``
            sees an actionable refusal rather than a silent
            truncation the ``AudioProcessor`` constructor would
            perform.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value is inside the clamp pair; a ``refusals`` list of
        refusal descriptors, each carrying the dimension name, the
        offending value, the clamp it violated, and the
        module-local :data:`_REFUSAL_TEXT` a driver-side wrapper
        would surface if the write were attempted while the value
        is outside the envelope; the same ``envelope`` sub-dict
        :func:`g1_list_bidi_audio_stream_delay_envelope` returns.
        On an admitted value the ``refusals`` list is empty; on a
        rejected value the single violated bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(value: Any, bound_key: str, bound: int, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "stream_delay_ms",
                "value": value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "text": _REFUSAL_TEXT,
            }
        )

    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 and hide a type mistake at the boundary.
    if isinstance(stream_delay_ms, bool):
        _reject(stream_delay_ms, "stream_delay_ms_min", _STREAM_DELAY_MS_MIN, "non-int")
    elif not isinstance(stream_delay_ms, int):
        _reject(stream_delay_ms, "stream_delay_ms_min", _STREAM_DELAY_MS_MIN, "non-int")
    else:
        v = int(stream_delay_ms)
        if v < _STREAM_DELAY_MS_MIN:
            _reject(stream_delay_ms, "stream_delay_ms_min", _STREAM_DELAY_MS_MIN, "value < bound")
        elif v > _STREAM_DELAY_MS_MAX:
            _reject(stream_delay_ms, "stream_delay_ms_max", _STREAM_DELAY_MS_MAX, "value > bound")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
