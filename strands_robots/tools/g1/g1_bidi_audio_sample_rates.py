"""Agent-facing lookup for the sample rates the neon bidi audio path pins.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) pins three
distinct sample rates for the three audio surfaces it moves data
between: ``MIC_RATE = 16000`` (the laptop USB mic capture rate the
WebRTC :class:`AudioProcessor` runs its AEC on), ``G1_RATE = 16000``
(the G1 chest-speaker DDS feed rate the ``rt/audio_stream`` far-buffer
uses for the AEC reference), and ``OPENAI_RATE = 24000`` (the OpenAI
Realtime endpoint's own sample rate the ``BidiAgent`` upsamples to
before publishing and downsamples from before consuming). The rates
are not a policy the neon bundle picks freely: WebRTC's AEC contract
fixes the mic/reference side at 16 kHz (the rate the 10 ms frame size
``FRAME_SIZE = 160`` matches, ``160 / 16000 == 0.01``), and the OpenAI
Realtime endpoint fixes the endpoint side at 24 kHz. This module
snapshots the three observed rates and exposes two agent-facing verbs
so a caller can decide the refusal decidably before a future
driver-side wrapper for the bidi audio IO is attempted, rather than
pinning the rate inside the write path where the refusal is invisible
to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_speak(action="start")``
  verb spins up a ``G1BidiAudioIO`` under a single-thread lock; that
  construction is the same audio-processing path the driver's
  future bidi wrapper would front. A future driver method that
  fronts the bidi IO will land the write verb; refs
  strands-labs/robots#358 for the SDK-facing gate work that write
  belongs on. This module ports the read-only lookup half without
  also introducing a second audio-processing writer path the driver
  does not yet own.
* An SDK re-import. The rate table is captured here as a
  module-level constant snapshot of the three integers the neon
  bundle observed; the constant lives here rather than being
  re-imported from ``pywebrtc_audio`` or ``pyaudio`` so ``import
  strands_robots.tools.g1.g1_bidi_audio_sample_rates`` pulls no
  ``unitree_sdk2py`` submodule *and* pulls no optional audio-stack
  submodule - the import-hygiene contract every other file in this
  package carries, refs strands-labs/robots#358. A revision of the
  observed rate table is a driver-side update; when the driver's
  bidi audio method lands, its refusal will surface the same
  module-local :data:`ERR_CODES` ``7404`` refusal this module names
  for an off-set role.

Why this module quotes the ``7404`` gate-refusal code.

The G1 SDK ships no distinct rc for a mis-named audio-side sample
rate (the rate is a caller-side integer the WebRTC / OpenAI Realtime
factories consume; neither ships a numbered SDK rc for a bad
argument). This lookup uses the ``7404`` gate-refusal shape a future
driver-side wrapper would quote when refusing at the same boundary,
mirroring
:mod:`~strands_robots.tools.g1.g1_voice_providers`'s use of the same
code for the neighbouring provider surface. Named separately from
the neighbouring constants so a future SDK release that ships a
dedicated "invalid sample rate" code lands here without also
renaming the shared gate-refusal constant.

What this module does not decide.

* Whether the current host has the audio dependencies installed.
  The neon bundle's ``_probe_bidi`` probe (``pywebrtc_audio`` +
  ``pyaudio`` + ``strands.experimental.bidi.BidiAgent``) is a live
  runtime check answered where the write path is; a caller planning
  a bidi start compares an intended sample-rate role against the
  set this verb surfaces first, and only then reaches the runtime
  probe for the missing-dep refusal.
* Whether the endpoint upstream (OpenAI Realtime, WebRTC AEC, G1
  DDS speaker) accepts a rate revision. The rates surfaced here are
  the ones the neon bundle observed as usable against those three
  endpoints; a caller who wants to attempt a different rate is
  outside the observed envelope and the refusal points them at the
  admitted set.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the (role -> integer sample rate) mapping the neon
#: bundle's ``G1BidiAudioIO`` (``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``)
#: pins today. The three roles are the three surfaces the bundle
#: moves audio between:
#:
#: * ``mic_rate`` - the laptop USB mic capture rate the WebRTC AEC
#:   runs on. Fixed at 16 kHz because WebRTC's AEC contract admits
#:   only 16 kHz frames at the ``FRAME_SIZE = 160`` (10 ms) granularity
#:   the AudioProcessor pins.
#: * ``g1_rate`` - the G1 chest-speaker DDS feed rate the AEC reference
#:   (far-buffer) uses. Also 16 kHz because the bundle downsamples the
#:   24 kHz OpenAI output to 16 kHz before publishing it on
#:   ``rt/audio_stream`` and reuses the same rate as the AEC reference
#:   queue so the near/far alignment is a same-rate comparison.
#: * ``openai_rate`` - the OpenAI Realtime endpoint's own sample rate.
#:   Fixed at 24 kHz by the endpoint; the bundle upsamples the 16 kHz
#:   mic capture to 24 kHz before publishing and downsamples the
#:   24 kHz endpoint output to 16 kHz before feeding it to the DDS
#:   chest speaker.
#:
#: The mapping lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the three roles
#: are only meaningful for the ``g1_speak`` / bidi audio side of the
#: conversation. Colocating the map with the verb mirrors
#: ``_VOICE_PROVIDER_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_voice_providers` and
#: ``_BALANCE_MODE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_balance_modes`: one snapshot per
#: neon-observed surface.
_BIDI_AUDIO_SAMPLE_RATE_MAP: dict[str, dict[str, Any]] = {
    "mic_rate": {
        "rate_hz": 16000,
        "description": (
            "Laptop USB mic capture rate. The neon bundle's "
            "``AudioProcessor`` runs its WebRTC AEC on 10 ms frames "
            "at 16 kHz (``FRAME_SIZE = 160``); the mic capture path "
            "resamples any non-16 kHz device down to this rate before "
            "the AEC touches it. Fixed by WebRTC's own AEC contract, "
            "not by the neon bundle's policy."
        ),
    },
    "g1_rate": {
        "rate_hz": 16000,
        "description": (
            "G1 chest-speaker DDS feed rate on ``rt/audio_stream``. The "
            "neon bundle downsamples the 24 kHz OpenAI Realtime output "
            "to 16 kHz before publishing to the G1 and reuses the same "
            "rate for the WebRTC AEC's far-buffer reference queue so "
            "the near/far alignment is a same-rate comparison. "
            "Numerically equal to ``mic_rate`` but named separately "
            "because the two carry different signal directions and a "
            "future rate revision on either surface would land "
            "independently."
        ),
    },
    "openai_rate": {
        "rate_hz": 24000,
        "description": (
            "OpenAI Realtime endpoint sample rate. Fixed at 24 kHz by "
            "the endpoint; the neon bundle upsamples the 16 kHz mic "
            "capture to 24 kHz before publishing to OpenAI and "
            "downsamples the 24 kHz endpoint output to 16 kHz before "
            "feeding it to the DDS chest speaker. A caller who wants "
            "to route the bidi path through a non-OpenAI provider "
            "(Nova Sonic, Gemini) reaches the same 24 kHz rate on the "
            "endpoint side; the label preserves the neon bundle's "
            "own naming."
        ),
    },
}

#: The error-table entry a future driver-side wrapper would quote on
#: a sample-rate role outside :data:`_BIDI_AUDIO_SAMPLE_RATE_MAP`. The
#: WebRTC / OpenAI Realtime / DDS-speaker factories do not ship a
#: numbered SDK rc for a bad sample-rate argument (the rate is a raw
#: integer they consume), so this lookup uses the ``7404`` gate-refusal
#: shape a future driver-side wrapper would quote when refusing at
#: the same boundary. The write path and this lookup share the
#: constant. Named separately from the neighbouring constants so a
#: future SDK release that ships a dedicated "invalid sample rate"
#: code lands here without also renaming the shared gate-refusal
#: constant.
_INVALID_SAMPLE_RATE_CODE: int = 7404


def _describe(role: str) -> dict[str, Any]:
    """Build the per-role descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bidi_audio_sample_rates` so
    :func:`g1_bidi_audio_sample_rate_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is touched.
    """
    entry = _BIDI_AUDIO_SAMPLE_RATE_MAP[role]
    return {
        "role": role,
        "rate_hz": entry["rate_hz"],
        "description": entry["description"],
        "admits_bidi_writes": True,
    }


@tool
def g1_list_bidi_audio_sample_rates() -> dict[str, Any]:
    """Return the sample rates the neon bidi audio path pins per role.

    Read-only. No driver instance, no DDS, no SDK, no
    ``pywebrtc_audio`` import, no ``pyaudio`` import: every field is
    a module-level constant. Useful before a future driver-side
    wrapper for ``G1BidiAudioIO`` is called, so a caller can compare
    an intended (role, rate) pair against the neon-observed mapping
    and can carry the module-local refusal text a driver-side
    wrapper would surface on an off-set role.

    Returns:
        A dict with ``status``; a ``rates`` list of per-role
        descriptors sorted by ``role`` ascending, each carrying
        ``role`` (``"mic_rate"``, ``"g1_rate"``, or ``"openai_rate"``),
        ``rate_hz`` (the integer sample rate in Hertz),
        ``description`` (why the rate is fixed at that value on that
        surface), and ``admits_bidi_writes`` (always ``True``,
        because every admitted rate is a bidi-shaped write by
        definition; the flag is surfaced so the descriptor shape
        matches :mod:`~strands_robots.tools.g1.g1_voice_providers`
        and :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim);
        a ``roles`` list quoting the admitted role names in sorted
        order; and a ``refusals`` list carrying the ``7404`` refusal
        code and its decoded text, the one a future write verb would
        surface. Every field is a snapshot of an observed rate; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "rates": [_describe(role) for role in sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)],
        "roles": sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP),
        "refusals": [
            {"code": _INVALID_SAMPLE_RATE_CODE, "text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE]},
        ],
    }


@tool
def g1_bidi_audio_sample_rate_admits(role: str | None = None) -> dict[str, Any]:
    """Decide whether a sample-rate ``role`` sits inside the admitted set.

    Read-only. Compares one argument against the neon-observed
    :data:`_BIDI_AUDIO_SAMPLE_RATE_MAP` and reports the admitted
    descriptor on match, or the ``7404`` refusal code a future
    driver-side wrapper would quote on miss. No driver instance, no
    DDS, no SDK, no audio stack: the decision reads only module-level
    constants and the argument itself.

    A role inside the admitted set is *not* the same as an admitted
    write: the neon bundle's runtime ``_probe_bidi`` also refuses on
    ``pywebrtc_audio`` / ``pyaudio`` /
    ``strands.experimental.bidi.BidiAgent`` missing at import time,
    and the endpoint upstream may still refuse at wire time on
    credentials or connectivity grounds. Neither of those is a
    snapshot answer; both are live-host reads a caller reaches after
    this verb admits the role name. The returned payload names the
    ``rate_hz`` on the descriptor so a caller comparing an intended
    write against the endpoint's fixed rate has the integer on hand.

    Args:
        role: The sample-rate role to check (``"mic_rate"``,
            ``"g1_rate"``, or ``"openai_rate"`` today). The
            comparison is case-sensitive against the snapshot in
            :data:`_BIDI_AUDIO_SAMPLE_RATE_MAP`; a mis-cased or
            unknown name is refused with the ``7404`` code. Bool
            values (``True``/``False``) are refused with the same
            code because ``str(True) == "True"`` would otherwise
            silently mis-match; a non-string non-bool argument is
            refused with the same code for the same reason. An
            empty string is refused decidably rather than treated
            as a default.

    Returns:
        A dict with ``status``; on admit, a ``rate`` descriptor with
        ``role``, ``rate_hz``, ``description``, and
        ``admits_bidi_writes`` (the same shape
        :func:`g1_list_bidi_audio_sample_rates` returns). On refuse,
        ``refusal_code`` and ``refusal_text`` name the ``7404`` code
        and its decoded text, plus a ``reason`` string that names
        why the argument was refused (missing argument, bool
        argument, non-string argument, empty-string argument, or
        unknown role).
    """
    if role is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_SAMPLE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE],
            "reason": (
                f"role is required; pass one of {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)} so the lookup is decidable"
            ),
        }
    if isinstance(role, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_SAMPLE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE],
            "reason": (f"role={role!r} is a bool; pass one of {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)} as a string"),
        }
    if not isinstance(role, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_SAMPLE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE],
            "reason": (f"role={role!r} is not a string; pass one of {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)} as a string"),
        }
    if role == "":
        return {
            "status": "error",
            "refusal_code": _INVALID_SAMPLE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE],
            "reason": (
                f"role is the empty string; pass one of {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)} so the lookup is decidable"
            ),
        }
    if role not in _BIDI_AUDIO_SAMPLE_RATE_MAP:
        return {
            "status": "error",
            "refusal_code": _INVALID_SAMPLE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_SAMPLE_RATE_CODE],
            "reason": (f"role={role!r} is not in the admitted set {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"),
        }
    return {
        "status": "success",
        "rate": _describe(role),
    }
