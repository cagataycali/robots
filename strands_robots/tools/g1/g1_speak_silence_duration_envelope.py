"""Agent-facing lookup for the silence-duration hint the neon bidi speak agent ends a turn on.

The neon bundle's ``g1_speak``
(``cagataycali/neon-the-g1/tools/g1_speak.py::g1_speak``) forwards a
``silence_duration_ms`` argument through to the strands bidi voice
agent it wraps.  The bidi runner reads the value as the *turn-end*
detector's silence-window length: after the WebRTC VAD flags a
frame as non-speech, the runner waits ``silence_duration_ms``
milliseconds of continuous silence before declaring the user's
turn complete and letting the model emit a response.  A shorter
window makes the agent feel snappy but interrupts the user
mid-thought; a longer window feels patient but stalls turn-taking.
The neon bundle's own default is authored inline as ``700`` at the
top of ``g1_speak``'s signature and named in the docstring as
"How long of silence ends a turn.  700 = relaxed".

This module snapshots the observed default into a module-level
constant and exposes two agent-facing verbs so a caller planning
a ``g1_speak`` start against a target latency budget can compare
an intended silence-duration argument decidably before a future
driver-side wrapper is called, rather than pinning the value
inside the write path where the refusal is invisible to the
planner.

Twin of :mod:`~strands_robots.tools.g1.g1_speak_vad_envelope`
(the merged ``g1_speak_vad_envelope``) and
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
(the merged stream-delay envelope).  The three modules stay
separate because they name three disjoint decisions on the same
neon speak surface:

* :mod:`~strands_robots.tools.g1.g1_speak_vad_envelope` names the
  VAD *sensitivity threshold* (``vad_threshold``, a 0.0-1.0 float
  the WebRTC VAD compares each 10 ms frame's speech probability
  against).  Its refusal shape is "the caller supplied a
  threshold outside the closed unit interval" -- a sensitivity
  argument, not a duration.
* :mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
  names the *AEC speaker-to-mic delay hint* (``stream_delay_ms``,
  a 0-500 ms window WebRTC's AudioProcessor uses to align the
  far-buffer reference against the near-mic frames).  Its
  refusal shape is a delay-buffer clamp, not a turn-end wait.
* This module names the *turn-end silence window*
  (``silence_duration_ms``, an integer millisecond count of
  continuous silence before ending a user turn).  Its refusal
  shape is a positive-count domain violation on the same speak
  argument list.

Colocating the three would hand an agent planner a single
refusal payload that mixed the sensitivity remedy against the
AEC-alignment remedy against the turn-end-latency remedy.  The
three surfaces stay separate so a refusal names the argument its
remedy belongs on.

Two things this module is deliberately *not*:

* An execution path.  The neon bundle's ``g1_speak`` runs the
  bidi agent under ``_runner_main`` on a background thread with
  the audio-processing pipeline live (pyaudio input, DDS chest
  speaker output); that thread carries the audio-stack imports
  (``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``)
  which are optional dependencies the ``strands-robots`` package
  does not require.  A future driver method that fronts
  ``g1_speak`` will surface the same module-local
  :data:`_REFUSAL_TEXT` on a below-domain-floor refusal; this
  module ports the read-only envelope half without also
  introducing a second bidi audio path the driver does not yet
  own, refs strands-labs/robots#358.
* An SDK re-import.  The value is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_speak_silence_duration_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no
  ``pywebrtc_audio`` / ``pyaudio`` / ``strands.experimental.bidi``
  submodule at import time -- the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358.  A caller authoring a speak plan
  before any audio extra is installed on their host still gets
  the default back verbatim.

Why this module does not quote a driver-side ``rc``.

The G1 driver's
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``);
its FSM rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``).  The
turn-end silence-duration decision runs on the strands bidi
runner's own VAD callback against an in-memory 10 ms frame
window -- it never touches ``rt/lowcmd``, never talks to the
locomotion controller, and reaches no SDK RPC service that
ships an rc table for a below-count refusal.  Borrowing
``7404`` on a bad-count refusal would hand an agent planner a
motion-FSM remedy for a turn-end-latency argument.  The
refusal shape this module returns names the shape violation in
module-local text so a planner reads a remedy that matches the
surface, and a future driver-side speak wrapper will surface
the same module-local text.  This mirrors the same-surface
refusal rule
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
names for the AEC stream-delay hint, refs
strands-labs/robots#358.

What this module does not decide.

* An explicit upper bound on the silence-duration argument.
  The neon bundle authors no ceiling on ``silence_duration_ms``
  -- a caller who wants a very-patient 5000 ms turn-end wait
  reads a decidable shape refusal only if the argument fails
  the shared positive-count domain, not because a numeric
  ceiling refused it.  Inventing an upper clamp here would
  hand a planner a refusal shape the write path itself does
  not produce, so this envelope names only the shared-domain
  positivity floor the neon default (``700``) implicitly
  relies on.
* Whether the caller's silence-duration argument is a *good*
  match for their conversational pacing goal.  A 200 ms
  argument makes the agent interrupt mid-thought on a
  thoughtful user; a 3000 ms argument stalls turn-taking on a
  fast-paced user.  Both are *quality* decisions a planner
  makes against their session profile; this envelope names
  only the *shape* decision on the argument value.
* The far-buffer alignment or the VAD threshold on the same
  speak call.  The two twins name those dimensions on their
  own modules because each remedy is disjoint from the
  silence-duration remedy: fixing an AEC alignment ("adjust
  stream_delay_ms") or a sensitivity floor ("raise
  vad_threshold") is a different argument on the same call.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.utils import positive_count_error

#: The neon-bundle-tuned default the ``g1_speak`` verb reaches for
#: when the caller does not name ``silence_duration_ms``.  The
#: neon docstring for ``g1_speak`` names ``700`` as the value the
#: bundle carries "for a relaxed turn-end pace" against the WebRTC
#: VAD's 10 ms frame cadence -- a measurement the neon bundle
#: took against its own bidi agent's turn-taking latency budget.
#: Surfaced here so a caller planning a bidi start with the neon
#: default reads the same integer without re-measuring it, and so
#: a widen or narrow to the observed default lands in one place.
_SILENCE_DURATION_MS_NEON_DEFAULT: int = 700

#: The module-local refusal text every
#: ``g1_speak_silence_duration_ms_admits`` refusal quotes when the
#: caller-supplied duration fails the shared positive-count
#: domain the neon runner's turn-end detector implicitly reads
#: (a millisecond count is a positive integer of continuous
#: silence samples, and a value below one would collapse the
#: turn-end wait to zero samples).  Named here rather than
#: borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the bidi speak path ships no distinct rc -- the strands bidi
#: runner just uses the argument as a wall-clock timeout and
#: never round-trips through a bus that returns one.  The
#: motion-FSM ``7404`` entry (its nearest neighbour) reads
#: ``"Invalid FSM id - need FSM in {500, 501, 801}"`` -- a remedy
#: that points a planner at locomotion FSM transitions to fix a
#: turn-end-latency argument.  Surfacing the module-local text
#: keeps the refusal payload's remedy on the same surface the
#: write belongs on; a future driver-side speak wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    "speak silence-duration gate refused - the argument sits outside "
    "the shared positive-count domain the neon speak turn-end "
    "detector admits. Refs strands-labs/robots#358."
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_speak_silence_duration_envelope` so
    :func:`g1_speak_silence_duration_ms_admits` names the same
    field on its admitted-path payload and so a widen to the
    descriptor lands in one place.  Every field is a snapshot
    read; no bus is touched.
    """
    return {
        "silence_duration_ms_neon_default": _SILENCE_DURATION_MS_NEON_DEFAULT,
    }


@tool
def g1_list_speak_silence_duration_envelope() -> dict[str, Any]:
    """Return the turn-end silence-duration the neon bidi speak agent admits.

    Read-only.  No driver instance, no DDS, no SDK, no
    ``pywebrtc_audio`` / ``pyaudio`` /
    ``strands.experimental.bidi`` submodule import at load time:
    the field is a module-level constant.  Useful before a future
    driver-side wrapper for ``g1_speak`` is called, so a caller
    can compare an intended turn-end silence window against the
    value the neon runner's bidi turn-end detector reads and can
    carry the module-local refusal text a driver-side wrapper
    would surface on a shape violation.

    The envelope carries one field, the neon-observed default in
    integer milliseconds, because the neon bundle authors one
    silence-duration value at build time and would need a source
    patch to admit a different one.  A caller planning a bidi
    session against a target turn-end latency reads this
    envelope's :func:`g1_speak_silence_duration_ms_admits` shape
    grader against their intended value.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        the neon-runner-observed default
        (``silence_duration_ms_neon_default``); and a
        ``refusals`` list carrying a single descriptor with the
        module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on a shape violation.  Every field is a
        snapshot of an observed value or a module-local text;
        no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_speak_silence_duration_ms_admits(
    silence_duration_ms: int = _SILENCE_DURATION_MS_NEON_DEFAULT,
) -> dict[str, Any]:
    """Decide whether a candidate turn-end silence-duration clears the shared positive-count domain.

    Read-only.  Grades ``silence_duration_ms`` against the shared
    :func:`~strands_robots.utils.positive_count_error` domain the
    neon runner's turn-end detector implicitly reads (a
    millisecond count is a positive integer of continuous silence
    samples).  No driver instance, no DDS, no SDK, no
    ``pywebrtc_audio`` / ``pyaudio`` /
    ``strands.experimental.bidi`` submodule import: the decision
    reads only the argument itself and the module-level default.

    A value that clears the shared domain is *not* the same as an
    admitted turn-end wait: the neon runner also carries a
    conversational-latency budget which a caller matches against
    their target session profile ("snappy" vs "relaxed" vs
    "patient"), which this envelope does not decide.  The
    returned payload names only the numeric shape decision on
    the argument itself.

    Args:
        silence_duration_ms: The candidate turn-end silence
            window in integer milliseconds.  The default ``700``
            (the observed neon runner value) admits, so a caller
            who does not pass an explicit argument lands on the
            runner's own boundary case.  The shared
            :func:`~strands_robots.utils.positive_count_error`
            domain refuses non-``int`` inputs (including
            ``bool``, which is an ``int`` subclass whose ``True``
            would otherwise be a silent ``1``), values below
            ``1``, and any type coercion that could hide a
            floating-point argument.  A value of ``0`` (an
            immediate turn-end on the first silent frame) is
            refused by the shared domain rather than by an
            inline check because the WebRTC VAD's own 10 ms
            frame cadence makes a zero-millisecond wait
            equivalent to no wait at all, which the neon runner
            was never designed to admit; a caller who probed
            ``silence_duration_ms=0`` receives the shared-domain
            refusal that names the shape mistake decidably before
            the write.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the argument clears the shared positive-count domain; a
        ``refusals`` list of refusal descriptors, each carrying
        the dimension name, the offending value, the comparison
        (``"shared-domain"`` for a shape mistake), the shared
        domain's own text, and the module-local
        :data:`_REFUSAL_TEXT` a driver-side wrapper would surface
        if the speak call were dispatched with an out-of-domain
        duration; the same ``envelope`` sub-dict
        :func:`g1_list_speak_silence_duration_envelope` returns.
        On an admitted duration the ``refusals`` list is empty.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    # Shared-domain shape check: positive_count_error refuses
    # bool, non-int, and value < 1.  A duration that clears this
    # domain is the neon runner's own implicit precondition on a
    # millisecond wait; the neon bundle authors no explicit
    # ceiling, so this envelope names only the shared-domain
    # positivity floor.
    domain_err = positive_count_error(
        silence_duration_ms,
        "silence_duration_ms",
        "g1_speak_silence_duration_ms_admits",
    )
    if domain_err is not None:
        refusals.append(
            {
                "dimension": "silence_duration_ms",
                "value": silence_duration_ms,
                "bound_key": "silence_duration_ms_neon_default",
                "bound": _SILENCE_DURATION_MS_NEON_DEFAULT,
                "comparison": "shared-domain",
                "domain_error": domain_err,
                "text": _REFUSAL_TEXT,
            }
        )

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
