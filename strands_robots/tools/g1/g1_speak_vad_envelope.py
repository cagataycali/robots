"""Agent-facing lookup for the VAD envelope the neon ``g1_speak`` verb admits.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) takes two turn-detection
knobs at ``action="start"``: ``vad_threshold`` (a float in ``[0.0, 1.0]``
naming the voice-activity detector's admission floor) and
``silence_duration_ms`` (a positive integer naming the trailing-silence
window that ends a turn).  The bundle passes both to the ``BidiAgent``
factory the ``g1.build_voice_agent`` helper constructs, and the factory
forwards them to the turn-detector configuration inside
``strands.experimental.bidi`` without itself refusing an out-of-range
value.  The neon docstring for the verb names the two observed defaults
as ``vad_threshold=0.7`` ("higher = less twitchy. 0.7 stops echo
triggers") and ``silence_duration_ms=700`` ("how long of silence ends a
turn. 700 = relaxed"), and the ``[0.0, 1.0]`` range on the threshold is
the ``BidiAgent`` turn-detector's own numeric domain.  This module
snapshots the observed envelope into module-level constants and exposes
two agent-facing verbs so a caller can decide the refusal decidably
before a future driver-side ``g1_speak`` wrapper is called, rather than
pinning the two knobs inside the write path where the refusal is
invisible to the planner.

Twin of :mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`,
which surfaces the ``stream_delay_ms`` knob on the *audio-processing*
half of the same ``g1_speak`` argument tuple.  The two modules stay
separate because ``stream_delay_ms`` names the AEC delay-buffer bound
(a ``pywebrtc_audio`` argument) while the two knobs here name the
turn-detector bounds (a ``strands.experimental.bidi`` argument): two
different downstream libraries with disjoint refusal shapes.  Colocating
them here would hand an agent planner a single refusal payload that
mixed the two surfaces' remedies and would tie a future audio-processing
revision to a turn-detector revision the neon bundle does not couple.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``g1_speak(action="start")``
  spawns a background thread that runs a full ``BidiAgent`` (webrtc
  AEC, pyaudio input, DDS chest-speaker output); that thread carries
  the audio-stack imports (``pywebrtc_audio``, ``pyaudio``,
  ``strands.experimental.bidi``) which are optional dependencies the
  ``strands-robots`` package does not require.  A future driver method
  that fronts ``g1_speak`` will land the transition verb; refs
  strands-labs/robots#358 for the SDK-facing gate work that audio path
  belongs on.  This module ports the read-only envelope half without
  also introducing a second bidi-writer path the driver does not yet
  own.
* An SDK re-import.  The envelope is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_speak_vad_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no optional
  audio-stack submodule (``pywebrtc_audio``, ``pyaudio``,
  ``strands.experimental.bidi``) - the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358.
  A revision of the observed bounds is a driver-side update; when the
  driver's bidi voice method lands, its refusal will surface the same
  module-local :data:`_REFUSAL_TEXT_VAD_THRESHOLD` and
  :data:`_REFUSAL_TEXT_SILENCE_DURATION_MS` this module names for a
  bounds violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The ``BidiAgent``
turn detector runs on the mic pre-processing thread in the Python
process itself - it never touches ``rt/lowcmd`` and never touches an
RPC service the SDK ships an rc table for - so the bidi voice pipeline
ships no distinct rc for a bounds-violated ``vad_threshold`` or
``silence_duration_ms``.  Borrowing ``7404`` on a turn-detector refusal
would hand an agent planner a motion-FSM remedy (``"need FSM in {500,
501, 801}"``) for a bounds violation on a value that has nothing to do
with the locomotion FSM.  The refusal shape this module returns names
the numeric bound violation in module-local text so a planner reads a
remedy that matches the surface, and a future driver-side ``g1_speak``
wrapper will surface the same module-local text - not a re-borrowed
motion code.  This mirrors the same-surface refusal rule
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
names for ``pywebrtc_audio.AudioProcessor(stream_delay_ms=...)``, refs
strands-labs/robots#358.

What this module does not decide.

* The live bidi state.  Whether the neon bundle's ``g1_speak`` runner
  thread is currently active, whether the ``BidiAgent`` factory has
  resolved a provider, whether the chest-speaker writer has started:
  none of those live-instance reads run here.  A caller planning a
  ``g1_speak(action="start")`` reads this verb's ``envelope`` to
  decide whether their two knobs are inside the observed range and
  reads the driver's own liveness signal separately to decide whether
  the bidi path is currently free.
* Whether the caller's ``vad_threshold`` matches the caller's mic
  gain.  A ``vad_threshold=0.9`` on a mic with a low input gain silences
  the turn detector regardless of what this envelope admits, and a
  ``vad_threshold=0.1`` on a mic with a high input gain triggers on
  every stray sound.  Whether the caller's gain-and-threshold pair is
  jointly usable is a live-instance decision the neon bundle answers
  under its ``STATS["energy_mean_abs"]`` reading, not a numeric envelope
  decision.
"""

from __future__ import annotations

import math
from typing import Any

from strands import tool

#: The lower clamp on ``vad_threshold``.  The ``BidiAgent`` turn
#: detector's numeric domain is ``[0.0, 1.0]``; below the lower bound
#: the detector's admission floor is undefined and every frame reads
#: as speech (the detector's ``score >= threshold`` check degenerates
#: because the score is itself a probability in ``[0.0, 1.0]`` and no
#: threshold below zero can refuse a non-negative score).  Named as
#: an inclusive bound (``value < bound`` refuses, ``value == bound``
#: admits) because a caller naming ``vad_threshold=0.0`` is a
#: legitimate "accept every frame" command on a rig where the turn
#: boundary is decided by ``silence_duration_ms`` alone, not the
#: shape-error the refusal exists to catch.
_VAD_THRESHOLD_MIN: float = 0.0

#: The upper clamp on ``vad_threshold``.  Above ``1.0`` the detector
#: refuses every frame (the score is itself in ``[0.0, 1.0]``, so no
#: score can meet a threshold above the domain), and the neon bundle
#: docstring names the observed maximum as ``1.0`` on the ``0.0-1.0``
#: range.  Named as an inclusive bound like the lower clamp, so a
#: caller writing ``vad_threshold=1.0`` (the ceiling, "accept only a
#: certain-speech frame") is admitted rather than tripping an
#: off-by-one.
_VAD_THRESHOLD_MAX: float = 1.0

#: The neon-bundle-observed default ``vad_threshold`` value.  The
#: neon docstring names it as ``0.7`` and its inline note "higher =
#: less twitchy. 0.7 stops echo triggers" describes the field
#: observation: at ``0.7`` the detector refuses the residual echo that
#: leaks through the AEC on the G1's DDS speaker->mic loop, while
#: still admitting the operator's speech.  Surfaced here so a caller
#: planning a bidi start with the neon-tuned value can pin the same
#: number without re-measuring the loop, and so a widen or narrow to
#: the observed default lands in one place.
_VAD_THRESHOLD_NEON_DEFAULT: float = 0.7

#: The lower clamp on ``silence_duration_ms``.  A non-positive value
#: means "end the turn immediately on the first non-speech frame",
#: which the ``BidiAgent`` turn detector accepts but degenerates into
#: single-frame turns that cut off every operator pause between
#: words.  The neon bundle names positive integers only, matching the
#: shared :func:`~strands_robots.utils.positive_count_error` domain
#: every other integer-count knob in this package uses.  Named as an
#: exclusive-of-zero bound (``value <= 0`` refuses) because a
#: ``silence_duration_ms=0`` collapses the turn boundary to a single
#: frame and is not the "wait longer" command the parameter's name
#: suggests.
_SILENCE_DURATION_MS_MIN: int = 1

#: The neon-bundle-observed default ``silence_duration_ms`` value.
#: The neon docstring names it as ``700`` and its inline note "700 =
#: relaxed" describes the field observation: at ``700`` ms the turn
#: detector waits long enough for an operator's natural pause between
#: words without also holding the turn open past the operator's own
#: silence.  Surfaced here so a caller planning a bidi start with the
#: neon-tuned value can pin the same integer without re-measuring the
#: pause, and so a widen or narrow to the observed default lands in
#: one place.
_SILENCE_DURATION_MS_NEON_DEFAULT: int = 700

#: The module-local refusal text every ``g1_speak_vad_admits`` refusal
#: quotes when the caller's ``vad_threshold`` sits outside the
#: ``BidiAgent`` turn-detector's numeric domain.  Named here rather
#: than borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because the
#: bidi voice pipeline ships no distinct rc for a bounds-violated
#: threshold argument and the motion-FSM ``7404`` entry (its nearest
#: neighbour) reads ``"Invalid FSM id - need FSM in {500, 501, 801}"``
#: - a remedy that points a planner at locomotion FSM transitions to
#: fix a turn-detector argument.  Surfacing the module-local text
#: keeps the refusal payload's remedy on the same surface the write
#: belongs on, and a future driver-side ``g1_speak`` wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT_VAD_THRESHOLD: str = (
    f"vad_threshold out of envelope - need vad_threshold in [{_VAD_THRESHOLD_MIN}, {_VAD_THRESHOLD_MAX}]"
)

#: The module-local refusal text every ``g1_speak_vad_admits`` refusal
#: quotes when the caller's ``silence_duration_ms`` sits outside the
#: positive-integer domain.  Named here for the same reason as
#: :data:`_REFUSAL_TEXT_VAD_THRESHOLD` (the pipeline ships no distinct
#: rc for a bounds-violated silence argument, and the motion-FSM code
#: is on the wrong surface).  The remedy names the minimum positive
#: integer rather than a range because the ``BidiAgent`` turn detector
#: has no observed upper bound on the wait window; a caller who wants
#: to hold the turn open for a long dictation pass names any positive
#: integer and the detector honours it.
_REFUSAL_TEXT_SILENCE_DURATION_MS: str = (
    f"silence_duration_ms out of envelope - need silence_duration_ms >= {_SILENCE_DURATION_MS_MIN}"
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_speak_vad_envelope` so
    :func:`g1_speak_vad_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place.  Every field is a snapshot read; no bus is touched.
    """
    return {
        "vad_threshold_min": _VAD_THRESHOLD_MIN,
        "vad_threshold_max": _VAD_THRESHOLD_MAX,
        "vad_threshold_neon_default": _VAD_THRESHOLD_NEON_DEFAULT,
        "silence_duration_ms_min": _SILENCE_DURATION_MS_MIN,
        "silence_duration_ms_neon_default": _SILENCE_DURATION_MS_NEON_DEFAULT,
    }


@tool
def g1_list_speak_vad_envelope() -> dict[str, Any]:
    """Return the VAD envelope the neon ``g1_speak`` verb admits.

    Read-only.  No driver instance, no DDS, no SDK, no optional
    audio-stack import: every field is a module-level constant.
    Useful before a future driver-side wrapper for the neon
    ``g1_speak(action="start")`` is called, so a caller can compare
    an intended ``vad_threshold`` / ``silence_duration_ms`` pair
    against the envelope the ``BidiAgent`` turn detector admits and
    can carry the module-local refusal text a driver-side wrapper
    would surface on a bounds violation.  The neon-tuned defaults
    (``vad_threshold=0.7`` and ``silence_duration_ms=700`` for the G1
    DDS speaker->mic loop) are named on the envelope so a caller who
    wants the neon-observed values can pin them without re-measuring
    the loop.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon bundle observed
        (``vad_threshold_min``, ``vad_threshold_max``,
        ``vad_threshold_neon_default``, ``silence_duration_ms_min``,
        ``silence_duration_ms_neon_default``); and a ``refusals``
        list carrying one descriptor per dimension with the
        module-local :data:`_REFUSAL_TEXT_VAD_THRESHOLD` and
        :data:`_REFUSAL_TEXT_SILENCE_DURATION_MS` a future write verb
        would surface on a bounds violation.  Every field is a
        snapshot of an observed bound or a module-local text; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"dimension": "vad_threshold", "text": _REFUSAL_TEXT_VAD_THRESHOLD},
            {
                "dimension": "silence_duration_ms",
                "text": _REFUSAL_TEXT_SILENCE_DURATION_MS,
            },
        ],
    }


@tool
def g1_speak_vad_admits(
    vad_threshold: float = 0.7,
    silence_duration_ms: int = 700,
) -> dict[str, Any]:
    """Decide whether a VAD argument pair sits inside the envelope.

    Read-only.  Compares each argument against the clamp
    :func:`g1_list_speak_vad_envelope` returns and reports every
    refusal shape a bound violation would surface.  No driver
    instance, no DDS, no SDK, no optional audio-stack import: the
    decision reads only module-level constants and the two arguments
    themselves.

    A ``vad_threshold`` / ``silence_duration_ms`` pair inside the
    envelope is *not* the same as an admitted write: the driver's
    bidi singleton may refuse on liveness grounds (an in-flight
    session, a missing provider credential, a stalled mic autopick),
    which this verb does not read (that is a live driver-instance
    query answered by a future bidi state verb).  The returned
    envelope names only the numeric bound decision.

    Args:
        vad_threshold: float in ``[0.0, 1.0]``.  The default ``0.7``
            matches the neon-bundle-tuned value for the G1 DDS
            speaker->mic loop so a caller who does not pass an
            explicit argument lands on the neon-observed admitted
            value.  Refused below ``vad_threshold_min`` (the
            detector's score is a probability in ``[0.0, 1.0]`` and
            no threshold below zero can refuse a non-negative score)
            and above ``vad_threshold_max`` (above ``1.0`` no score
            can meet the threshold and the detector silences every
            frame).  Boolean values are refused explicitly because
            Python's ``bool`` is a subclass of ``int`` (and ``int``
            is admitted as a widening to ``float`` below), so a
            caller passing ``True`` would otherwise silently look up
            ``1.0`` (the admitted ceiling) and hide the type mistake.
            Non-finite floats (``nan``, ``inf``, ``-inf``) are
            refused because the detector's comparison against a
            non-finite threshold degenerates (``score >= nan`` is
            always ``False``, ``score >= inf`` is always ``False``);
            the refusal at the boundary surfaces the type mistake
            rather than silencing every frame.
        silence_duration_ms: positive integer.  The default ``700``
            matches the neon-bundle-tuned value for a relaxed
            operator pause so a caller who does not pass an explicit
            argument lands on the neon-observed admitted value.
            Refused at or below zero because a non-positive wait
            collapses the turn boundary to a single frame and cuts
            off every operator pause between words.  Boolean values
            are refused explicitly at the boundary because Python's
            ``bool`` is a subclass of ``int``, so a caller passing
            ``True`` would otherwise silently look up ``1`` (a
            legitimate one-millisecond wait) and hide the type
            mistake.  Non-integer numeric values (``float``,
            ``Decimal``) are refused with the same shape so a caller
            passing ``silence_duration_ms=700.0`` sees an actionable
            refusal rather than a silent truncation the turn detector
            would perform.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        both values are inside their respective clamps; a
        ``refusals`` list of refusal descriptors, each carrying the
        dimension name, the offending value, the clamp it violated,
        and the module-local refusal text a driver-side wrapper
        would surface if the write were attempted while the value is
        outside the envelope; the same ``envelope`` sub-dict
        :func:`g1_list_speak_vad_envelope` returns.  On an admitted
        pair the ``refusals`` list is empty; on a rejected pair
        every violated bound is named (both arguments are graded
        independently, so a caller with two bad arguments reads two
        refusals in one payload).
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # ---- vad_threshold ----
    # bool subclasses int; refuse first so True/False do not silently
    # widen to 1.0/0.0 and hide a type mistake at the boundary.
    if isinstance(vad_threshold, bool):
        refusals.append(
            {
                "dimension": "vad_threshold",
                "value": vad_threshold,
                "bound_key": "vad_threshold_min",
                "bound": _VAD_THRESHOLD_MIN,
                "comparison": "non-float",
                "text": _REFUSAL_TEXT_VAD_THRESHOLD,
            }
        )
    elif not isinstance(vad_threshold, (int, float)):
        # ``int`` is admitted as a widening to ``float`` (a caller
        # writing ``vad_threshold=1`` should land on the admitted
        # ceiling), but any other type is refused with the same
        # non-float remedy the neon bundle's turn detector would
        # surface.
        refusals.append(
            {
                "dimension": "vad_threshold",
                "value": vad_threshold,
                "bound_key": "vad_threshold_min",
                "bound": _VAD_THRESHOLD_MIN,
                "comparison": "non-float",
                "text": _REFUSAL_TEXT_VAD_THRESHOLD,
            }
        )
    else:
        v = float(vad_threshold)
        if not math.isfinite(v):
            refusals.append(
                {
                    "dimension": "vad_threshold",
                    "value": vad_threshold,
                    "bound_key": "vad_threshold_min",
                    "bound": _VAD_THRESHOLD_MIN,
                    "comparison": "non-finite",
                    "text": _REFUSAL_TEXT_VAD_THRESHOLD,
                }
            )
        elif v < _VAD_THRESHOLD_MIN:
            refusals.append(
                {
                    "dimension": "vad_threshold",
                    "value": vad_threshold,
                    "bound_key": "vad_threshold_min",
                    "bound": _VAD_THRESHOLD_MIN,
                    "comparison": "value < bound",
                    "text": _REFUSAL_TEXT_VAD_THRESHOLD,
                }
            )
        elif v > _VAD_THRESHOLD_MAX:
            refusals.append(
                {
                    "dimension": "vad_threshold",
                    "value": vad_threshold,
                    "bound_key": "vad_threshold_max",
                    "bound": _VAD_THRESHOLD_MAX,
                    "comparison": "value > bound",
                    "text": _REFUSAL_TEXT_VAD_THRESHOLD,
                }
            )

    # ---- silence_duration_ms ----
    if isinstance(silence_duration_ms, bool):
        refusals.append(
            {
                "dimension": "silence_duration_ms",
                "value": silence_duration_ms,
                "bound_key": "silence_duration_ms_min",
                "bound": _SILENCE_DURATION_MS_MIN,
                "comparison": "non-int",
                "text": _REFUSAL_TEXT_SILENCE_DURATION_MS,
            }
        )
    elif not isinstance(silence_duration_ms, int):
        refusals.append(
            {
                "dimension": "silence_duration_ms",
                "value": silence_duration_ms,
                "bound_key": "silence_duration_ms_min",
                "bound": _SILENCE_DURATION_MS_MIN,
                "comparison": "non-int",
                "text": _REFUSAL_TEXT_SILENCE_DURATION_MS,
            }
        )
    else:
        s = int(silence_duration_ms)
        if s < _SILENCE_DURATION_MS_MIN:
            refusals.append(
                {
                    "dimension": "silence_duration_ms",
                    "value": silence_duration_ms,
                    "bound_key": "silence_duration_ms_min",
                    "bound": _SILENCE_DURATION_MS_MIN,
                    "comparison": "value < bound",
                    "text": _REFUSAL_TEXT_SILENCE_DURATION_MS,
                }
            )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
