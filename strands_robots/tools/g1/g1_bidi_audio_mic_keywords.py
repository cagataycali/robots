"""Agent-facing lookup for the mic-name keywords ``autopick_mic`` admits.

The neon bundle's ``autopick_mic`` helper
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) walks the host's
PyAudio input-device list once and selects the first device whose
name contains a caller-configured keyword substring. The bundle's
``DEFAULT_MIC_KEYWORDS = ["DJI", "Logi", "Brio", "USB", "Mic"]``
constant names the four USB-mic families the field has actually
observed on Jetson Orin office rigs plus the generic ``USB`` and
``Mic`` fallbacks the helper reaches for on a hardware family it has
not yet observed. The bundle's ``_mic_keywords`` accessor lets a
caller override the whole list through the ``VOICE_MIC_NAME``
environment variable (comma-separated, case-insensitive); this
module surfaces the *default* priority order the helper uses when
the env-var is unset, so a caller planning a bidi-audio bring-up on
a rig with unknown mic hardware can decide, before the helper runs,
whether their device name matches the neon-observed set.

The bundle's helper preserves priority: the first keyword that
matches wins, and within the same keyword the first device index
wins. That two-level ordering is a driver-side contract the write
path relies on (a caller who set ``VOICE_MIC_NAME="Brio,DJI"`` gets
the Brio, not the DJI, even if the DJI has a lower PyAudio index),
so the snapshot returns the same order the helper's default list
declares - names are keyed by their zero-based priority slot rather
than a bare set.

Two things this module is deliberately *not*:

* A live device probe. The neon bundle's ``autopick_mic`` opens a
  PyAudio session, walks ``p.get_device_count()``, and reads
  ``p.get_device_info_by_index(i)`` for each entry - a live
  hardware read the ``strands_robots`` package cannot answer
  without the optional ``pyaudio`` dependency
  (:mod:`~strands_robots.tools.g1.g1_bidi_audio_dependencies` names
  it as ``mic_capture``). This module ports the read-only keyword
  half of the helper's contract without also introducing the
  PyAudio wire read; a future driver-side ``g1_bidi_audio`` bring-up
  verb will land the live probe. Refs strands-labs/robots#358.
* An SDK re-import. The keyword list is captured here as a
  module-level constant so
  ``import strands_robots.tools.g1.g1_bidi_audio_mic_keywords``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no optional
  audio-stack submodule (``pyaudio``, ``pywebrtc_audio``,
  ``strands.experimental.bidi``) - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358. A revision of the observed keyword set
  is a driver-side update; when the driver's bidi-audio bring-up
  verb lands, it will read the same module-level constant this
  lookup returns.

What this module does not decide.

* Whether the current host has any input device at all. The neon
  bundle's helper returns ``None`` when PyAudio reports zero
  input-capable devices; that answer is a live-host read a caller
  reaches after this verb admits the keyword. This module names
  only the keyword priority order the helper walks.
* Whether the caller's ``VOICE_MIC_NAME`` override is set. The
  bundle's ``_mic_keywords`` accessor reads the env-var at call
  time; the override, when set, replaces this module's default
  list entirely rather than extending it. A caller planning to
  ship a ``VOICE_MIC_NAME`` value reads this verb to decide
  whether the default already covers their device (in which case
  the override is unnecessary), or to see the shape of the list
  they are overriding.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the mic-name keyword priority order the neon bundle's
#: ``autopick_mic`` helper (``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``)
#: walks when ``VOICE_MIC_NAME`` is unset. The bundle's own
#: ``DEFAULT_MIC_KEYWORDS`` list carries these five names in this
#: order; the first keyword whose substring is present in a PyAudio
#: input-device name wins, and the priority is stable across calls
#: because the helper walks the list from index zero. Named as a
#: tuple rather than a list to make the snapshot immutable at the
#: module boundary; a caller who wants a mutable copy calls
#: :func:`list` on the tuple.
#:
#: The five entries name three specific USB-mic families the neon
#: field notes observed on Jetson Orin office rigs (``DJI`` for the
#: DJI wireless receiver, ``Logi`` and ``Brio`` for the Logitech
#: Brio family - the ``Logi`` prefix catches both the ``Logitech``
#: and the ``Logi`` naming variants Logitech has shipped across
#: firmware revisions - and ``USB`` and ``Mic`` as generic fallbacks
#: for a device family the bundle has not yet observed). The
#: comparison is case-insensitive at the helper's boundary: the
#: bundle lowercases both the keyword and the device name before
#: the ``in`` check, so a ``"dji receiver"`` device name matches
#: ``"DJI"`` and a ``"USB Audio"`` name matches ``"USB"``. The
#: keyword strings themselves are the exact literals the neon
#: bundle carries, so a caller who compares against this list uses
#: the same casing the driver's write path sees.
_DEFAULT_MIC_KEYWORDS: tuple[str, ...] = (
    "DJI",
    "Logi",
    "Brio",
    "USB",
    "Mic",
)

#: The env-var name the neon bundle's ``_mic_keywords`` accessor
#: reads to replace :data:`_DEFAULT_MIC_KEYWORDS` at call time. When
#: set to a non-empty comma-separated string, the accessor returns
#: the caller's list verbatim and never falls back to the module
#: default; when unset or empty, the accessor returns
#: :data:`_DEFAULT_MIC_KEYWORDS`. Named here rather than only in the
#: descriptor payload so a caller planning to override the priority
#: order has a single string to grep for, and so a future
#: driver-side bring-up verb reads the same env-var name this
#: module surfaces (mirroring the ``credential_env`` field the
#: :mod:`~strands_robots.tools.g1.g1_voice_providers` snapshot
#: carries per provider).
_OVERRIDE_ENV_VAR: str = "VOICE_MIC_NAME"


def _describe(index: int, keyword: str) -> dict[str, Any]:
    """Build the per-keyword descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bidi_audio_mic_keywords` so
    :func:`g1_bidi_audio_mic_keyword_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched and
    no device is opened.
    """
    return {
        "priority": index,
        "keyword": keyword,
        "match_case_insensitive": True,
        "override_env": _OVERRIDE_ENV_VAR,
    }


@tool
def g1_list_bidi_audio_mic_keywords() -> dict[str, Any]:
    """Return the mic-name keyword priority order ``autopick_mic`` walks.

    Read-only. No driver instance, no DDS, no SDK, no audio stack,
    and no PyAudio session: every field is a module-level constant.
    Useful before a future driver-side wrapper for the neon bundle's
    ``g1_bidi_audio`` bring-up is called, so a caller can compare an
    intended mic device name against the priority order the helper's
    default list walks, and can also read the ``override_env`` on the
    returned descriptor to decide whether to ship a
    ``VOICE_MIC_NAME`` override.

    The five entries are returned in the exact priority order the
    neon bundle's ``autopick_mic`` walks: ``priority=0`` is the
    highest-precedence keyword (``"DJI"`` today) and ``priority=4``
    is the lowest (``"Mic"`` today). A caller comparing a device
    name against this list walks it from the lowest priority number
    upward and stops at the first match - the same order the
    helper's own loop uses.

    Returns:
        A dict with ``status="success"``; a ``count`` naming how
        many keywords the snapshot carries (five today, matching the
        neon bundle's ``DEFAULT_MIC_KEYWORDS``); a ``keywords`` list
        of per-keyword descriptors sorted by ``priority`` ascending,
        each carrying ``priority`` (the zero-based index the helper
        walks), ``keyword`` (the exact substring literal the helper
        lowercases and compares against a PyAudio device name),
        ``match_case_insensitive`` (always ``True``, because the
        helper lowercases both operands before the ``in`` check),
        and ``override_env`` (the ``VOICE_MIC_NAME`` env-var name a
        caller can ship to replace the default list); an
        ``override_env`` field naming the same env-var at the top
        level so a caller grepping the payload has one place to
        find it; and a ``names`` list quoting the keyword literals
        in priority order for callers who only want the strings.
        Every field is a snapshot of an observed keyword; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "count": len(_DEFAULT_MIC_KEYWORDS),
        "keywords": [_describe(i, kw) for i, kw in enumerate(_DEFAULT_MIC_KEYWORDS)],
        "names": list(_DEFAULT_MIC_KEYWORDS),
        "override_env": _OVERRIDE_ENV_VAR,
    }


@tool
def g1_bidi_audio_mic_keyword_admits(keyword: str | None = None) -> dict[str, Any]:
    """Decide whether a ``keyword`` sits inside the default priority order.

    Read-only. Compares one argument against
    :data:`_DEFAULT_MIC_KEYWORDS` and reports the admitted descriptor
    on match, or a shape-refusal on miss. No driver instance, no
    DDS, no SDK, no audio stack, no PyAudio session: the decision
    reads only module-level constants and the argument itself.

    A keyword inside the default priority order is *not* the same as
    an admitted mic bring-up: the neon bundle's ``autopick_mic``
    also refuses when the host reports zero PyAudio input devices
    matching the keyword substring - a live-device read this verb
    does not answer. A caller reaches that read after this verb
    admits the keyword, via a future driver-side bring-up verb.

    The comparison is exact-case against the snapshot; the neon
    bundle's helper lowercases both operands before its own ``in``
    check, but the *snapshot* names the keyword literals the bundle
    ships (``"DJI"``, ``"Logi"``, ``"Brio"``, ``"USB"``, ``"Mic"``),
    so a caller comparing a keyword they intend to *store* in a
    ``VOICE_MIC_NAME`` override reads the same casing the module
    carries. A mis-cased argument (``"dji"``) is refused decidably
    with a message naming the admitted casing, rather than silently
    resolved through the helper's own lowercasing.

    Args:
        keyword: The keyword substring to test. Must be one of the
            five neon-observed literals in exact casing. ``None`` is
            refused (the accessor's required-argument shape). Bool
            values are refused explicitly at the boundary because
            ``str(True) == "True"`` would otherwise never match any
            literal and hand the caller a confusing off-set refusal
            for what is really a type error. Non-string non-bool
            arguments are refused with the same shape. The empty
            string is refused decidably rather than treated as a
            default.

    Returns:
        On admission, a dict with ``status="success"`` and a
        ``keyword`` descriptor with ``priority``, ``keyword``,
        ``match_case_insensitive``, and ``override_env`` (the same
        shape :func:`g1_list_bidi_audio_mic_keywords` returns per
        entry). On refusal, ``status="error"`` and a ``message``
        naming why the argument was refused (missing argument, bool
        argument, non-string argument, empty-string argument, or
        off the admitted set) with a citation to
        ``strands-labs/robots#358`` and the sorted admitted set so
        a caller can see the correct casing without a second
        lookup.
    """
    admitted = list(_DEFAULT_MIC_KEYWORDS)
    if keyword is None:
        return {
            "status": "error",
            "message": (
                f"keyword is required; pass one of {admitted} so the lookup is decidable. Refs strands-labs/robots#358."
            ),
        }
    if isinstance(keyword, bool):
        return {
            "status": "error",
            "message": (
                f"keyword must be a str; got bool {keyword!r}. Pass one of {admitted}. Refs strands-labs/robots#358."
            ),
        }
    if not isinstance(keyword, str):
        return {
            "status": "error",
            "message": (
                f"keyword must be a str; got {type(keyword).__name__} "
                f"{keyword!r}. Pass one of {admitted}. Refs strands-labs/robots#358."
            ),
        }
    if keyword == "":
        return {
            "status": "error",
            "message": (
                f"keyword must be a non-empty str; got the empty string. Pass "
                f"one of {admitted}. Refs strands-labs/robots#358."
            ),
        }
    for i, kw in enumerate(_DEFAULT_MIC_KEYWORDS):
        if keyword == kw:
            return {
                "status": "success",
                "keyword": _describe(i, kw),
            }
    return {
        "status": "error",
        "message": (
            f"keyword {keyword!r} is not in the neon-observed default set {admitted}. Refs strands-labs/robots#358."
        ),
    }
