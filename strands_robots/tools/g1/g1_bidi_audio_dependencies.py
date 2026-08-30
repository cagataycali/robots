"""Agent-facing lookup for the imports the neon bidi audio path probes.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) runs a ``_probe_bidi``
guard before spawning its bidi-audio background thread: it ``import``s
three optional modules - ``pywebrtc_audio`` (the AEC front-end),
``pyaudio`` (the PortAudio input capture), and
``strands.experimental.bidi.BidiAgent`` (the bidi agent factory) - and
returns ``False`` if any raises ``ImportError``. That probe is *not* a
snapshot answer; it is a live host read, and a headless CI runner or
Thor before an office bring-up has none of those three modules
installed. The refusal the ``g1_speak`` verb quotes on a failed probe
is a shape-level refusal ("bidi audio deps missing") without naming
which of the three the caller has to install, so a planner asked to
diagnose the refusal has to read the neon source itself to enumerate
the dependency set.

This module snapshots that dependency set as an agent-facing lookup
so a caller planning a ``g1_speak`` rollout can name the three
modules the future driver-side wrapper's audio probe would reach
for, without also running the probe. The verb pair mirrors
:mod:`~strands_robots.tools.g1.g1_voice_providers` and
:mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot lookup
naming the whole set, one membership decision on one query.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``_probe_bidi`` runs the
  actual ``import`` statements against the host; that probe is out
  of scope for this lookup, which only names the three module names
  the probe reads. A future verb that fronts the probe through the
  driver's own audio-writer path is a separate port; refs
  strands-labs/robots#358 for the SDK-facing seam that audio path
  belongs on. This module ports the read-only lookup half without
  also introducing a second audio-probe path the driver does not
  yet own.
* An SDK or audio-stack re-import. The dependency names are
  captured here as string constants; a snapshot lookup reading this
  module does not pull ``pywebrtc_audio`` or ``pyaudio`` or
  ``strands.experimental.bidi`` (the module-load hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358). The invariant a neon-side probe change
  must preserve is byte-for-byte identity between the module names
  quoted here and the module names the ``_probe_bidi`` guard
  imports: a widen or narrow of the probe that does not update this
  snapshot leaves the two out of sync, so the probe the neon
  verb actually runs and the probe this lookup reports diverge
  silently.

What this module does not decide.

* Whether a dependency is currently *installed* on the host.
  Import availability is a host-layer answer; the neon bundle's
  ``_probe_bidi`` reaches for it. This lookup answers a static
  question: which three modules the probe would ``import``,
  independent of whether they resolve.
* Which pip package a dependency ships with. The three names are
  module-import names (``import pywebrgc_audio``), not
  distribution names; the distribution a caller installs to obtain
  the module is out of scope for this snapshot. The ``pip_hint``
  field surfaces a suggested distribution name where the module's
  name and the pip package's name are unambiguously the same, so a
  caller reading the payload has a starting install command; the
  hint is *not* a canonical resolver answer.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the module-import names the neon bundle's
#: ``_probe_bidi`` guard reads. Each entry names the module
#: (matching the neon source's ``import`` statement byte-for-byte),
#: a ``role`` describing what the module contributes to the bidi
#: audio path, and a ``pip_hint`` surfacing a suggested distribution
#: name where the module name and the distribution name coincide.
#:
#: The role labels are the neon-observed contributions:
#:
#: * ``aec_frontend`` for ``pywebrtc_audio``, the WebRTC-lineage
#:   acoustic-echo-cancellation front-end the neon path runs mic
#:   input through before handing frames to the bidi agent.
#: * ``mic_capture`` for ``pyaudio``, the PortAudio binding the
#:   neon path uses to read the USB mic device the ``autopick_mic``
#:   helper selects.
#: * ``bidi_agent`` for ``strands.experimental.bidi.BidiAgent``, the
#:   agent factory the neon runtime instantiates the ``g1_speak``
#:   bidi loop against; the module is inside ``strands`` and its
#:   ``experimental`` submodule may be gated behind an
#:   ``experimental`` install extra.
_BIDI_AUDIO_DEPENDENCIES: dict[str, dict[str, str]] = {
    "pywebrtc_audio": {
        "role": "aec_frontend",
        "description": (
            "WebRTC-lineage acoustic-echo-cancellation front-end. "
            "The neon bundle's ``_probe_bidi`` guard imports this "
            "module before instantiating the bidi agent; a caller "
            "planning a ``g1_speak`` write on a host without this "
            "module is refused by the probe with a shape-level "
            "``bidi audio deps missing`` refusal."
        ),
        "pip_hint": "pywebrtc-audio",
    },
    "pyaudio": {
        "role": "mic_capture",
        "description": (
            "PortAudio Python binding. The neon bundle reads mic "
            "frames through this module and hands them to the "
            "``autopick_mic`` helper's device-index result; the "
            "``_probe_bidi`` guard imports it before the bidi loop "
            "starts. On Jetson this ships as a system package a "
            "caller installs through ``apt`` rather than ``pip``, "
            "so the ``pip_hint`` is the PyPI distribution name a "
            "non-Jetson host would use."
        ),
        "pip_hint": "pyaudio",
    },
    "strands.experimental.bidi": {
        "role": "bidi_agent",
        "description": (
            "The strands experimental bidi agent factory. The "
            "neon bundle imports ``BidiAgent`` from this submodule "
            "and instantiates the bidi loop against it; the "
            "``experimental`` submodule may be gated behind an "
            "install extra on the ``strands`` distribution. A "
            "future driver-side ``g1_speak`` wrapper would front "
            "the same import; refs strands-labs/robots#358 for "
            "the SDK-facing seam that write path belongs on."
        ),
        "pip_hint": "strands-agents",
    },
}

#: The error-table entry a future driver-side wrapper would quote on
#: a dependency name outside :data:`_BIDI_AUDIO_DEPENDENCIES`. The
#: neon bundle's ``_probe_bidi`` guard raises ``ImportError`` on the
#: import statements themselves; a caller-side membership refusal
#: uses the ``7404`` gate-refusal shape a future driver-side wrapper
#: would quote when refusing at the same boundary. The write path
#: and this lookup share the constant. Named separately from the
#: voice-provider constant so a future SDK release that ships a
#: dedicated "invalid audio dependency" code lands here without
#: also renaming the shared gate-refusal constant.
_INVALID_DEPENDENCY_CODE: int = 7404


def _describe(name: str) -> dict[str, Any]:
    """Build the per-dependency descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bidi_audio_dependencies` so
    :func:`g1_bidi_audio_dependency_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched
    and no dependency is imported.
    """
    entry = _BIDI_AUDIO_DEPENDENCIES[name]
    return {
        "name": name,
        "role": entry["role"],
        "description": entry["description"],
        "pip_hint": entry["pip_hint"],
        "admits_bidi_writes": True,
    }


@tool
def g1_list_bidi_audio_dependencies() -> dict[str, Any]:
    """Return the module names the neon bidi audio probe reads.

    Read-only. No driver instance, no DDS, no SDK, no audio stack:
    every field is a module-level constant. Useful before a future
    driver-side wrapper for ``g1_speak`` is called, so a caller can
    compare an intended dependency name against the set the neon
    bundle's ``_probe_bidi`` guard imports, and can also read the
    ``pip_hint`` on each descriptor to name the distribution a host
    would install to satisfy the probe.

    Returns:
        A dict with ``status``; a ``dependencies`` list of
        per-dependency descriptors sorted by ``name`` ascending,
        each carrying ``name`` (the module-import name the neon
        ``_probe_bidi`` guard reads), ``role`` (a short label
        describing what the module contributes to the bidi audio
        path: ``aec_frontend``, ``mic_capture``, or
        ``bidi_agent``), ``description`` (the neon-observed
        contribution the module makes), ``pip_hint`` (a suggested
        distribution name a caller would ``pip install`` to obtain
        the module), and ``admits_bidi_writes`` (always ``True``,
        because every admitted dependency is inside the neon
        bundle's probe set by definition; the flag is surfaced so
        the descriptor shape matches
        :mod:`~strands_robots.tools.g1.g1_voice_providers` and
        :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim);
        a ``names`` list quoting the admitted module names in
        sorted order; and a ``refusals`` list naming the ``7404``
        refusal code a future driver-side wrapper would quote on
        a dependency name outside the admitted set. Every field is
        a snapshot of a neon-observed module name; no dynamic
        decode runs here.
    """
    return {
        "status": "success",
        "dependencies": [_describe(name) for name in sorted(_BIDI_AUDIO_DEPENDENCIES)],
        "names": sorted(_BIDI_AUDIO_DEPENDENCIES),
        "refusals": [
            {"code": _INVALID_DEPENDENCY_CODE, "text": ERR_CODES[_INVALID_DEPENDENCY_CODE]},
        ],
    }


@tool
def g1_bidi_audio_dependency_admits(name: str | None = None) -> dict[str, Any]:
    """Decide whether a module ``name`` sits inside the probe set.

    Read-only. Compares one argument against the neon-observed
    :data:`_BIDI_AUDIO_DEPENDENCIES` and reports the admitted
    descriptor on match, or the ``7404`` refusal code a future
    driver-side wrapper would quote on miss. No driver instance,
    no DDS, no SDK, no audio stack: the decision reads only
    module-level constants and the argument itself.

    A dependency inside the admitted set is *not* the same as an
    admitted write: the neon bundle's ``_probe_bidi`` also refuses
    when the module fails to ``import`` at runtime, and the
    ``g1_speak`` verb refuses independently on missing
    ``credential_env`` for the chosen provider (see
    :mod:`~strands_robots.tools.g1.g1_voice_providers`). Neither of
    those is a snapshot answer; both are live-host reads a caller
    reaches after this verb admits the dependency name. The
    returned payload names ``pip_hint`` so a caller comparing an
    intended write against both conditions (membership + host
    install) has the distribution name on hand.

    Args:
        name: The module-import name to check (``"pywebrtc_audio"``,
            ``"pyaudio"``, or ``"strands.experimental.bidi"``
            today). The comparison is case-sensitive against the
            snapshot in :data:`_BIDI_AUDIO_DEPENDENCIES`; a
            mis-cased or unknown name is refused with the ``7404``
            code. Bool values (``True``/``False``) are refused with
            the same code because ``str(True) == "True"`` would
            otherwise silently mis-match; a non-string non-bool
            argument is refused with the same code for the same
            reason. An empty string is refused decidably rather
            than treated as a default.

    Returns:
        A dict with ``status``; on admit, a ``dependency``
        descriptor with ``name``, ``role``, ``description``,
        ``pip_hint``, and ``admits_bidi_writes`` (the same shape
        :func:`g1_list_bidi_audio_dependencies` returns). On
        refuse, ``refusal_code`` and ``refusal_text`` name the
        ``7404`` code and its decoded text, plus a ``reason``
        string that names why the argument was refused (missing
        argument, bool argument, non-string argument, empty-string
        argument, or unknown module name).
    """
    if name is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_DEPENDENCY_CODE,
            "refusal_text": ERR_CODES[_INVALID_DEPENDENCY_CODE],
            "reason": (f"name is required; pass one of {sorted(_BIDI_AUDIO_DEPENDENCIES)} so the lookup is decidable"),
        }
    if isinstance(name, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_DEPENDENCY_CODE,
            "refusal_text": ERR_CODES[_INVALID_DEPENDENCY_CODE],
            "reason": (f"name={name!r} is a bool; pass one of {sorted(_BIDI_AUDIO_DEPENDENCIES)} as a string"),
        }
    if not isinstance(name, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_DEPENDENCY_CODE,
            "refusal_text": ERR_CODES[_INVALID_DEPENDENCY_CODE],
            "reason": (f"name={name!r} is not a string; pass one of {sorted(_BIDI_AUDIO_DEPENDENCIES)} as a string"),
        }
    if name == "":
        return {
            "status": "error",
            "refusal_code": _INVALID_DEPENDENCY_CODE,
            "refusal_text": ERR_CODES[_INVALID_DEPENDENCY_CODE],
            "reason": (
                f"name is the empty string; pass one of {sorted(_BIDI_AUDIO_DEPENDENCIES)} so the lookup is decidable"
            ),
        }
    if name not in _BIDI_AUDIO_DEPENDENCIES:
        return {
            "status": "error",
            "refusal_code": _INVALID_DEPENDENCY_CODE,
            "refusal_text": ERR_CODES[_INVALID_DEPENDENCY_CODE],
            "reason": (f"name={name!r} is not in the admitted set {sorted(_BIDI_AUDIO_DEPENDENCIES)}"),
        }
    return {
        "status": "success",
        "dependency": _describe(name),
    }
