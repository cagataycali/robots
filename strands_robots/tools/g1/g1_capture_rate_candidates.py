"""Agent-facing lookup for the PCM sample rates the neon mic-open path tries.

The neon bundle's ``g1_bidi_audio`` module
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) opens the USB mic
device through PyAudio before handing frames to the bidi agent, and it
does that by iterating a fixed list of candidate sample rates against
``PyAudio.is_format_supported`` until one is accepted; the neon
constant ``CAPTURE_RATE_CANDIDATES = [16000, 48000, 44100, 32000]``
names that list in the order it is tried, and the neon
``pick_capture_rate`` helper falls back to ``48000`` when the whole
list is refused. That list is a live-host read: the rate a device
accepts depends on the operating system's ALSA/CoreAudio backend, on
whether PulseAudio is in the path, and on the specific USB mic
plugged in (Brio ships 16k directly; DJI Mic Mini reaches only 48k).
The refusal a neon caller sees when the whole list fails is a shape
without naming the four numbers a planner would need to know to
diagnose the miss.

This module snapshots that candidate list as an agent-facing lookup
so a caller planning a ``g1_speak`` rollout can name the four rates
the future driver-side wrapper's mic-open probe would iterate,
without also running the probe. The verb pair mirrors
:mod:`~strands_robots.tools.g1.g1_bidi_audio_dependencies` and
:mod:`~strands_robots.tools.g1.g1_voice_providers`: one snapshot
lookup naming the whole set with the iteration order preserved, one
membership decision on one query. The ordering matters here in a way
it does not for the sibling lookups - the neon helper stops at the
first accepted rate, so the *first* candidate is the rate a Brio
host lands on and the *last* candidate is the rate a host that
refuses all higher options would land on; that ordering is part of
the contract this snapshot pins.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``pick_capture_rate`` runs
  ``PyAudio.is_format_supported`` against each candidate; that
  probe is a live host read reaching for the audio backend, and it
  is out of scope for this lookup. A future driver-side verb that
  fronts the probe through the driver's audio-writer path is a
  separate port; refs strands-labs/robots#358 for the SDK-facing
  seam that audio path belongs on. This module ports the read-only
  lookup half without also introducing a second mic-open path the
  driver does not yet own.
* An SDK or audio-stack re-import. The candidate rates are
  captured here as an integer tuple; a snapshot lookup reading
  this module does not pull ``pyaudio`` or ``unitree_sdk2py`` (the
  module-load hygiene contract every other file in this package
  carries, refs strands-labs/robots#358). The invariant a
  neon-side change must preserve is byte-for-byte identity between
  the integers quoted here and the integers the
  ``CAPTURE_RATE_CANDIDATES`` list holds *in the same order*: a
  reorder or a widen that does not update this snapshot leaves the
  probe the neon verb actually runs and the probe this lookup
  reports diverge silently.

What this module does not decide.

* Whether a rate is currently supported by the host's audio
  backend. The neon bundle's ``is_format_supported`` probe is the
  live-host answer; a caller comparing an intended rate against
  both conditions (membership + backend support) reaches the probe
  after this verb admits the rate.
* Which USB mic ships with which rate. The Brio-vs-DJI split is
  observed behaviour named in the neon module's own comments, and
  this snapshot preserves the order the neon list was written in
  (16k first so Brio short-circuits on the first attempt); the
  snapshot does not carry a per-mic table because the mic-to-rate
  mapping is not a static answer (a firmware update can widen or
  narrow a mic's supported set).
* The fallback rate. The neon ``pick_capture_rate`` returns
  ``48000`` when every candidate is refused; that fallback is a
  separate constant on the neon side (``return 48000``) and is
  surfaced here in the ``fallback_rate_hz`` field of the list
  payload so a caller reading this module has the number the neon
  path lands on when the whole candidate list misses.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the ordered PCM sample-rate list the neon bundle's
#: ``pick_capture_rate`` helper iterates. The tuple preserves the
#: iteration order the neon source writes the list in - ``16000``
#: first so a Brio-family USB mic (which supports 16 kHz directly)
#: lands on the first attempt without walking the rest of the list,
#: and ``48000`` second so a DJI Mic Mini class device (which
#: refuses 16 kHz but accepts 48 kHz) lands on the second attempt.
#: ``44100`` and ``32000`` are the remaining common consumer USB
#: mic rates the neon list includes as a widening safety net.
#:
#: The tuple type is deliberate: a caller reading
#: :data:`_CAPTURE_RATE_CANDIDATES` should not be able to mutate
#: the snapshot in place and thereby drift the lookup out of sync
#: with the neon source. The verbs iterate the tuple in order so
#: the list payload names the same iteration a future driver-side
#: mic-open would run.
_CAPTURE_RATE_CANDIDATES: tuple[int, ...] = (16000, 48000, 44100, 32000)

#: The fallback sample rate the neon ``pick_capture_rate`` helper
#: returns when every entry in :data:`_CAPTURE_RATE_CANDIDATES`
#: is refused by ``PyAudio.is_format_supported``. Named separately
#: from the candidate tuple because the neon source expresses it
#: as a bare ``return 48000`` rather than a member of the list;
#: surfacing it in the payload lets a planner reading this module
#: name the rate the neon path lands on when the whole candidate
#: list misses, without also having to model the fallback branch
#: as a snapshot lookup itself.
_CAPTURE_RATE_FALLBACK_HZ: int = 48000

#: The error-table entry a future driver-side wrapper would quote
#: on a rate outside :data:`_CAPTURE_RATE_CANDIDATES`. The neon
#: bundle's ``pick_capture_rate`` does not refuse an unknown rate
#: (it silently returns the fallback); a caller-side membership
#: refusal uses the ``7404`` gate-refusal shape a future
#: driver-side wrapper would quote when refusing at the same
#: boundary. The write path and this lookup share the constant.
#: Named separately from the voice-provider and audio-dependency
#: constants so a future SDK release that ships a dedicated
#: "invalid capture rate" code lands here without also renaming
#: the shared gate-refusal constant.
_INVALID_CAPTURE_RATE_CODE: int = 7404

#: Roles the tuple entries carry, keyed by rate. The role labels
#: are descriptive rather than functional - the neon path treats
#: every rate the same at the ``is_format_supported`` boundary -
#: so a caller reading the descriptor sees which USB mic family
#: the rate is known-good for without also having to read the
#: neon source. The role text is a snapshot of the neon module's
#: own inline comments about which mic each candidate targets.
_CAPTURE_RATE_ROLES: dict[int, dict[str, str]] = {
    16000: {
        "role": "brio_native",
        "description": (
            "16 kHz PCM. The Logitech Brio USB webcam mic and "
            "similar 16 kHz-native consumer devices land here on "
            "the first attempt. The neon path also uses 16 kHz "
            "as its internal AEC and pipeline rate, so a device "
            "that accepts this candidate needs no downsample step "
            "between capture and the bidi agent."
        ),
    },
    48000: {
        "role": "usb_high_rate",
        "description": (
            "48 kHz PCM. The DJI Mic Mini USB receiver and other "
            "48 kHz-only consumer USB mics land here on the "
            "second attempt. The neon path downsamples 48 kHz to "
            "16 kHz through ``audioop.ratecv`` before AEC because "
            "the pipeline stays at 16 kHz internally. This is "
            "also the fallback rate the neon helper returns when "
            "every candidate is refused."
        ),
    },
    44100: {
        "role": "cd_rate",
        "description": (
            "44.1 kHz PCM. Consumer USB audio devices that ship "
            "the CD-lineage sample rate rather than 48 kHz land "
            "here. Named in the neon candidate list as a "
            "widening safety net for older USB devices; on Jetson "
            "the ALSA backend typically routes 44.1 kHz through "
            "the same downsample path 48 kHz takes."
        ),
    },
    32000: {
        "role": "usb_low_rate",
        "description": (
            "32 kHz PCM. A less common consumer USB audio rate "
            "the neon candidate list includes for completeness. "
            "A device that refuses 16 kHz, 48 kHz and 44.1 kHz "
            "but accepts 32 kHz is unusual today; the entry is "
            "kept in the list because the neon source writes it "
            "as the last candidate and this snapshot preserves "
            "the iteration order byte-for-byte."
        ),
    },
}


def _describe(rate: int) -> dict[str, Any]:
    """Build the per-rate descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_capture_rate_candidates` so
    :func:`g1_capture_rate_candidate_admits`'s admitted-path
    payload names the same fields, and so a widen to the
    descriptor lands in one place. Every field is a snapshot
    read; no bus is touched and no audio backend is queried.
    """
    role = _CAPTURE_RATE_ROLES[rate]
    return {
        "rate_hz": rate,
        "role": role["role"],
        "description": role["description"],
        "is_pipeline_native": rate == 16000,
        "is_fallback": rate == _CAPTURE_RATE_FALLBACK_HZ,
    }


@tool
def g1_list_capture_rate_candidates() -> dict[str, Any]:
    """Name the ordered PCM sample-rate list the neon mic-open probes.

    Read-only. Returns the same four integers the neon bundle's
    ``pick_capture_rate`` helper iterates against
    ``PyAudio.is_format_supported`` before opening the USB mic,
    in the same order the neon source writes them in. No driver
    instance, no DDS, no SDK, no audio stack: the payload reads
    only module-level constants.

    The ordering carries meaning the sibling lookups do not: the
    neon helper stops at the first accepted rate, so a caller
    reading the payload knows which rate a Brio-class device
    (16 kHz first) lands on versus which rate a DJI-class
    device (16 kHz refused, 48 kHz accepted) lands on. The
    fallback rate the neon helper returns on a whole-list miss
    is surfaced separately so a caller does not have to model
    the fallback branch as a candidate itself.

    Returns:
        A dict with ``status``; an ``ordered_rates_hz`` list
        naming the four integers in the neon iteration order; a
        ``candidates`` list of per-rate descriptors carrying
        ``rate_hz``, ``role``, ``description``,
        ``is_pipeline_native`` (the 16 kHz entry the internal
        pipeline runs at without a downsample), and
        ``is_fallback`` (the 48 kHz entry the neon helper falls
        back to on a whole-list miss); a ``fallback_rate_hz``
        integer naming the rate the neon helper returns when
        every candidate is refused; a ``count`` integer naming
        the tuple length; and a ``refusals`` list naming the
        ``7404`` refusal code a future driver-side wrapper would
        quote on a rate outside the admitted set.
    """
    return {
        "status": "success",
        "ordered_rates_hz": list(_CAPTURE_RATE_CANDIDATES),
        "candidates": [_describe(rate) for rate in _CAPTURE_RATE_CANDIDATES],
        "count": len(_CAPTURE_RATE_CANDIDATES),
        "fallback_rate_hz": _CAPTURE_RATE_FALLBACK_HZ,
        "refusals": [
            {"code": _INVALID_CAPTURE_RATE_CODE, "text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE]},
        ],
    }


@tool
def g1_capture_rate_candidate_admits(rate_hz: int | None = None) -> dict[str, Any]:
    """Decide whether a ``rate_hz`` sits inside the neon iteration list.

    Read-only. Compares one integer against the neon-observed
    :data:`_CAPTURE_RATE_CANDIDATES` tuple and reports the
    admitted descriptor on match, or the ``7404`` refusal code
    a future driver-side wrapper would quote on miss. No driver
    instance, no DDS, no SDK, no audio stack: the decision reads
    only module-level constants and the argument itself.

    A rate inside the admitted set is *not* the same as an
    admitted write: the neon bundle's ``pick_capture_rate`` also
    consults ``PyAudio.is_format_supported`` at runtime, which
    depends on the host's audio backend, the specific USB mic
    plugged in, and whether PulseAudio is in the path. Those
    are live-host reads a caller reaches after this verb
    admits the rate. The returned payload names
    ``is_pipeline_native`` and ``is_fallback`` so a caller
    comparing an intended rate against downstream conditions
    (does the neon path downsample this, would this be the
    fallback) has the flags on hand.

    Args:
        rate_hz: The PCM sample rate to check, in hertz
            (``16000``, ``48000``, ``44100``, or ``32000``
            today). The comparison is on integer identity
            against the snapshot; a rate outside the tuple is
            refused with the ``7404`` code. Bool values
            (``True``/``False``) are refused with the same
            code because Python's ``bool`` is a subclass of
            ``int`` and ``True == 1`` / ``False == 0`` would
            otherwise silently mis-match; a non-integer numeric
            argument (``float``, ``Decimal``) is refused with
            the same code because the neon source writes the
            list as bare ``int`` literals and a float would
            not be identity-equal to any entry. Negative or
            zero rates are refused decidably rather than
            treated as a default.

    Returns:
        A dict with ``status``; on admit, a ``candidate``
        descriptor with ``rate_hz``, ``role``, ``description``,
        ``is_pipeline_native``, and ``is_fallback`` (the same
        shape :func:`g1_list_capture_rate_candidates` returns).
        On refuse, ``refusal_code`` and ``refusal_text`` name
        the ``7404`` code and its decoded text, plus a
        ``reason`` string that names why the argument was
        refused (missing argument, bool argument, non-integer
        argument, non-positive argument, or unknown rate).
    """
    if rate_hz is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_CAPTURE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE],
            "reason": (f"rate_hz is required; pass one of {list(_CAPTURE_RATE_CANDIDATES)} so the lookup is decidable"),
        }
    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 (neither of which is a candidate) and hide a type
    # mistake at the boundary.
    if isinstance(rate_hz, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_CAPTURE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE],
            "reason": (f"rate_hz={rate_hz!r} is a bool; pass one of {list(_CAPTURE_RATE_CANDIDATES)} as an int"),
        }
    if not isinstance(rate_hz, int):
        return {
            "status": "error",
            "refusal_code": _INVALID_CAPTURE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE],
            "reason": (f"rate_hz={rate_hz!r} is not an int; pass one of {list(_CAPTURE_RATE_CANDIDATES)} as an int"),
        }
    if rate_hz <= 0:
        return {
            "status": "error",
            "refusal_code": _INVALID_CAPTURE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE],
            "reason": (
                f"rate_hz={rate_hz!r} is not positive; pass one of {list(_CAPTURE_RATE_CANDIDATES)} as a positive int"
            ),
        }
    if rate_hz not in _CAPTURE_RATE_CANDIDATES:
        return {
            "status": "error",
            "refusal_code": _INVALID_CAPTURE_RATE_CODE,
            "refusal_text": ERR_CODES[_INVALID_CAPTURE_RATE_CODE],
            "reason": (f"rate_hz={rate_hz!r} is not in the admitted set {list(_CAPTURE_RATE_CANDIDATES)}"),
        }
    return {
        "status": "success",
        "candidate": _describe(rate_hz),
    }
