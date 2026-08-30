"""Agent-facing lookup for the voice providers ``g1_speak`` admits.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) admits a small set of
bidi-model providers for the G1 voice persona: ``openai`` /
``openai_realtime`` (the two names the bundle's own
``prov in ("openai", "openai_realtime")`` guard treats identically for
the ``OPENAI_API_KEY`` env-var check), ``nova_sonic`` (Amazon Nova
Sonic, reached through AWS credentials), and ``gemini`` (Google
Gemini Live, reached through ``GOOGLE_API_KEY``). This module
snapshots the observed provider set and exposes two agent-facing
verbs so a caller can decide the refusal decidably before a future
driver-side ``g1_speak`` wrapper is attempted, rather than pinning
the choice inside the write path where the refusal is invisible to
the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_speak`` verb spawns a
  background thread that runs a full ``BidiAgent`` (webrtc AEC,
  pyaudio input, DDS chest-speaker output); that thread carries the
  audio-stack imports (``pywebrtc_audio``, ``pyaudio``,
  ``strands.experimental.bidi``) which are optional dependencies the
  ``strands-robots`` package does not require. A future driver-side
  method that fronts ``g1_speak`` will land the transition verb;
  refs strands-labs/robots#358 for the SDK-facing gate work that
  audio path belongs on. This module ports the read-only lookup
  half without also introducing a second audio-writer path the
  driver does not yet own.
* An SDK re-import. The provider set is captured here as a
  module-level constant snapshot of the four names the neon bundle
  observed; the constant lives here rather than being re-imported
  from ``strands.experimental.bidi`` so
  ``import strands_robots.tools.g1.g1_voice_providers`` pulls no
  ``unitree_sdk2py`` submodule *and* pulls no optional audio-stack
  submodule - the import-hygiene contract every other file in this
  package carries, refs strands-labs/robots#358.

What this module does not decide.

* Whether the current host has the audio dependencies installed.
  The neon bundle's ``_probe_bidi`` probe (``pywebrtc_audio`` +
  ``pyaudio`` + ``strands.experimental.bidi.BidiAgent``) is a live
  runtime check answered where the write path is; a caller planning
  a ``g1_speak`` call compares an intended provider against the set
  this verb surfaces first, and only then reaches the runtime probe
  for the missing-dep refusal.
* Whether the current environment has the provider's credential
  set. The neon bundle refuses ``openai`` / ``openai_realtime``
  without ``OPENAI_API_KEY``; the verb payload names the env-var
  each admitted provider expects so a caller comparing an intended
  provider against both conditions (membership + credential) has
  the credential name on hand.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the voice-provider names the neon bundle's ``g1_speak``
#: verb (``cagataycali/neon-the-g1/tools/g1_speak.py``) admits today.
#: The bundle's own ``prov in ("openai", "openai_realtime")`` guard
#: treats the first two names identically for the ``OPENAI_API_KEY``
#: env-var check, so both are surfaced as admitted members with the
#: same credential-env descriptor. ``nova_sonic`` and ``gemini`` are
#: the two other providers the bundle's docstring names as the
#: ``VOICE_PROVIDER`` env-var options. The SDK does not ship a
#: canonical provider-id to name mapping (the provider is a raw
#: string the bidi factory routes on); the snapshot lives here rather
#: than in :mod:`~strands_robots.tools.g1._g1_common` because the
#: mapping is only useful for the ``g1_speak`` side of the
#: conversation. Colocating the map with the verb mirrors
#: ``_BALANCE_MODE_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_balance_modes` and
#: ``_FSM_NAME_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_fsm_targets`: one snapshot per
#: SDK/neon-facing table, one verb pair per snapshot.
_VOICE_PROVIDER_MAP: dict[str, dict[str, str]] = {
    "openai": {
        "credential_env": "OPENAI_API_KEY",
        "description": (
            "OpenAI Realtime (bidi audio-in / audio-out over the "
            "OpenAI Realtime API). The neon bundle's ``g1_speak`` "
            "guard treats ``openai`` and ``openai_realtime`` "
            "identically for the ``OPENAI_API_KEY`` env-var check; "
            "both names route to the same bidi factory."
        ),
    },
    "openai_realtime": {
        "credential_env": "OPENAI_API_KEY",
        "description": (
            "OpenAI Realtime, explicit alias. Same credential and "
            "same bidi factory as ``openai``; the neon bundle "
            "accepts the aliased name so a caller who names the "
            "protocol explicitly is not refused."
        ),
    },
    "nova_sonic": {
        "credential_env": "AWS_ACCESS_KEY_ID",
        "description": (
            "Amazon Nova Sonic (bidi audio-in / audio-out over "
            "Bedrock). Reached through standard AWS credentials; "
            "the neon bundle's docstring names this as one of the "
            "``VOICE_PROVIDER`` env-var options."
        ),
    },
    "gemini": {
        "credential_env": "GOOGLE_API_KEY",
        "description": (
            "Google Gemini Live (bidi audio-in / audio-out over the "
            "Gemini Live API). The neon bundle's docstring names "
            "this as one of the ``VOICE_PROVIDER`` env-var options; "
            "the default voice on this provider is ``Kore`` per the "
            "bundle's field notes."
        ),
    },
}

#: The error-table entry a future driver-side wrapper would quote on a
#: provider name outside :data:`_VOICE_PROVIDER_MAP`. The bidi factory
#: routes the provider string through a Python ``match`` and raises
#: ``ValueError`` on an unknown name; the neon bundle refused unknown
#: providers at the verb boundary with a caller-side shape refusal
#: (not an SDK ``rc``), so this lookup uses the ``7404`` gate-refusal
#: shape a future driver-side wrapper would quote when refusing at
#: the same boundary. The write path and this lookup share the
#: constant. Named separately from the balance-mode constant so a
#: future SDK release that ships a dedicated "invalid voice provider"
#: code lands here without also renaming the shared gate-refusal
#: constant.
_INVALID_PROVIDER_CODE: int = 7404


def _describe(name: str) -> dict[str, Any]:
    """Build the per-provider descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_voice_providers`
    so :func:`g1_voice_provider_admits`'s admitted-path payload names
    the same fields, and so a widen to the descriptor lands in one
    place. Every field is a snapshot read; no bus is touched.
    """
    entry = _VOICE_PROVIDER_MAP[name]
    return {
        "name": name,
        "credential_env": entry["credential_env"],
        "description": entry["description"],
        "admits_bidi_writes": True,
    }


@tool
def g1_list_voice_providers() -> dict[str, Any]:
    """Return the voice-provider names ``g1_speak`` admits.

    Read-only. No driver instance, no DDS, no SDK, no audio stack:
    every field is a module-level constant. Useful before a future
    driver-side wrapper for ``g1_speak`` is called, so a caller can
    compare an intended provider against the set the neon bundle
    documented as bidi-capable, and can also read the
    ``credential_env`` on the returned descriptor to decide whether
    the current environment carries the credential the provider's
    factory reaches for.

    Returns:
        A dict with ``status``; a ``providers`` list of per-provider
        descriptors sorted by ``name`` ascending, each carrying
        ``name`` (the ``VOICE_PROVIDER`` env-var value the neon
        bundle routes on), ``credential_env`` (the env-var name the
        provider's factory reaches for), ``description`` (the
        provider label the neon bundle observed), and
        ``admits_bidi_writes`` (always ``True``, because every
        admitted provider is a bidi-shaped write by definition; the
        flag is surfaced so the descriptor shape matches
        :mod:`~strands_robots.tools.g1.g1_balance_modes` and
        :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim);
        a ``names`` list quoting the admitted provider names in
        sorted order; and a ``refusals`` list carrying the ``7404``
        refusal code and its decoded text, the one a future write
        verb would surface. Every field is a snapshot of an observed
        provider label; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "providers": [_describe(name) for name in sorted(_VOICE_PROVIDER_MAP)],
        "names": sorted(_VOICE_PROVIDER_MAP),
        "refusals": [
            {"code": _INVALID_PROVIDER_CODE, "text": ERR_CODES[_INVALID_PROVIDER_CODE]},
        ],
    }


@tool
def g1_voice_provider_admits(name: str | None = None) -> dict[str, Any]:
    """Decide whether a provider ``name`` sits inside the admitted set.

    Read-only. Compares one argument against the neon-observed
    :data:`_VOICE_PROVIDER_MAP` and reports the admitted descriptor
    on match, or the ``7404`` refusal code a future driver-side
    wrapper would quote on miss. No driver instance, no DDS, no SDK,
    no audio stack: the decision reads only module-level constants
    and the argument itself.

    A provider inside the admitted set is *not* the same as an
    admitted write: the neon bundle's runtime ``_probe_bidi`` also
    refuses on ``pywebrtc_audio`` / ``pyaudio`` /
    ``strands.experimental.bidi.BidiAgent`` missing at import time,
    and on the provider's ``credential_env`` missing from the
    environment. Neither of those is a snapshot answer; both are
    live-host reads a caller reaches after this verb admits the
    provider name. The returned payload names ``credential_env`` so
    a caller comparing an intended write against both conditions
    (membership + credential) has the env-var name on hand.

    Args:
        name: The provider name to check (``"openai"``,
            ``"openai_realtime"``, ``"nova_sonic"``, or ``"gemini"``
            today). The comparison is case-sensitive against the
            snapshot in :data:`_VOICE_PROVIDER_MAP`; a mis-cased or
            unknown name is refused with the ``7404`` code. Bool
            values (``True``/``False``) are refused with the same
            code because ``str(True) == "True"`` would otherwise
            silently mis-match; a non-string non-bool argument is
            refused with the same code for the same reason. An
            empty string is refused decidably rather than treated
            as a default.

    Returns:
        A dict with ``status``; on admit, a ``provider`` descriptor
        with ``name``, ``credential_env``, ``description``, and
        ``admits_bidi_writes`` (the same shape
        :func:`g1_list_voice_providers` returns). On refuse,
        ``refusal_code`` and ``refusal_text`` name the ``7404``
        code and its decoded text, plus a ``reason`` string that
        names why the argument was refused (missing argument,
        bool argument, non-string argument, empty-string argument,
        or unknown provider).
    """
    if name is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_PROVIDER_CODE,
            "refusal_text": ERR_CODES[_INVALID_PROVIDER_CODE],
            "reason": (f"name is required; pass one of {sorted(_VOICE_PROVIDER_MAP)} so the lookup is decidable"),
        }
    if isinstance(name, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_PROVIDER_CODE,
            "refusal_text": ERR_CODES[_INVALID_PROVIDER_CODE],
            "reason": (f"name={name!r} is a bool; pass one of {sorted(_VOICE_PROVIDER_MAP)} as a string"),
        }
    if not isinstance(name, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_PROVIDER_CODE,
            "refusal_text": ERR_CODES[_INVALID_PROVIDER_CODE],
            "reason": (f"name={name!r} is not a string; pass one of {sorted(_VOICE_PROVIDER_MAP)} as a string"),
        }
    if name == "":
        return {
            "status": "error",
            "refusal_code": _INVALID_PROVIDER_CODE,
            "refusal_text": ERR_CODES[_INVALID_PROVIDER_CODE],
            "reason": (
                f"name is the empty string; pass one of {sorted(_VOICE_PROVIDER_MAP)} so the lookup is decidable"
            ),
        }
    if name not in _VOICE_PROVIDER_MAP:
        return {
            "status": "error",
            "refusal_code": _INVALID_PROVIDER_CODE,
            "refusal_text": ERR_CODES[_INVALID_PROVIDER_CODE],
            "reason": (f"name={name!r} is not in the admitted set {sorted(_VOICE_PROVIDER_MAP)}"),
        }
    return {
        "status": "success",
        "provider": _describe(name),
    }
