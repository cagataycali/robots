"""Agent-facing lookup for the action names ``g1_speak`` admits.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) admits a small set of
action names on its ``action`` keyword argument: ``start`` spawns the
background bidi thread, ``stop`` sets the thread's stop event,
``status`` reads module-level counters without touching the bus,
``say`` runs a one-shot ``TtsMaker`` synth without the bidi thread,
and ``debug`` prints a diagnostic bundle (audio-device enumeration,
pulseaudio snapshot, cross-persona logs). The bundle's own trailing
guard (``return {"status": "error", "content": [{"text": f"unknown
action {action!r} - use start/stop/status/say/debug"}]}``) rejects
every other value at the verb boundary with a caller-side shape
refusal (not an SDK ``rc``). This module snapshots that observed
action-name set and exposes two agent-facing verbs so a caller
planning a future driver-side ``g1_speak`` wrapper can decide the
refusal decidably before the write path is attempted, rather than
pinning the choice inside the write path where the refusal is
invisible to the planner.

The verb pair mirrors
:mod:`~strands_robots.tools.g1.g1_voice_providers` (the four-
provider lookup for the same neon verb): one snapshot lookup
naming the whole admitted set, one membership decision on one
query. The two modules answer disjoint halves of the same
``g1_speak`` shape - which ``action`` verb the caller is invoking
(this module) and which bidi ``provider`` the ``start`` action
routes to (:mod:`~strands_robots.tools.g1.g1_voice_providers`) -
so a caller planning a call reads both.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_speak`` verb spawns a
  background thread (``_runner_main``) that runs a full
  ``BidiAgent`` (webrtc AEC, pyaudio input, DDS chest-speaker
  output); that thread carries the audio-stack imports
  (``pywebrtc_audio``, ``pyaudio``,
  ``strands.experimental.bidi``) which are optional dependencies
  the ``strands-robots`` package does not require. A future
  driver-side method that fronts ``g1_speak`` will land the
  transition verb; refs strands-labs/robots#358 for the tool-side
  gate work that speak path belongs on. This module ports the
  read-only lookup half without also introducing a second audio-
  writer path the driver does not yet own.
* An SDK re-import. The action-name set is captured here as a
  module-level constant snapshot of the five action names the
  neon bundle's ``g1_speak`` verb admitted; the constant lives
  here rather than being re-imported from ``strands.experimental
  .bidi`` or from any ``unitree_sdk2py`` submodule so
  ``import strands_robots.tools.g1.g1_speak_actions_envelope``
  pulls no ``unitree_sdk2py`` submodule *and* pulls no optional
  audio-stack submodule - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358.

What this module does not decide.

* Whether the current host has the audio dependencies installed.
  The neon bundle's ``_probe_bidi`` probe (``pywebrtc_audio`` +
  ``pyaudio`` + ``strands.experimental.bidi.BidiAgent``) is a live
  runtime check answered where the write path is; a caller
  planning a ``g1_speak`` call compares an intended ``action``
  against the set this verb surfaces first, and only then reaches
  the runtime probe for the missing-dep refusal. The
  ``g1_bidi_audio_dependencies`` module names the same probe set.
* Whether the ``start`` action's bidi thread is already running.
  The neon bundle branches on the module-level ``_STATE["running"]``
  flag ("already running session=..." vs "spawn a new thread"); a
  caller planning a start compares intent against the admitted
  action-name set here first, and only then reads the live-state
  flag on the driver side.
* Whether the ``say`` action's ``text`` argument is non-empty.
  The bundle refuses ``say`` with an empty ``text`` at the verb
  boundary (``"say requires text=..."``); the empty-string check
  is per-action-argument shape and does not sit on this envelope.

Module-local refusal text (not the shared FSM-motion
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES` ``7404``
entry) names why a caller-side action-name refusal is not a motion-
FSM refusal: the neon bundle's own trailing guard rejects the name
without reaching the bus, the SDK ships no dedicated "unknown
speak action" ``rc``, and borrowing the motion-FSM ``7404`` text
here would mis-label the shape refusal as a hardware FSM refusal.
The refusal text quotes the observed neon bundle guard phrasing so
a caller reading ``refusal_text`` sees the same string the neon
verb surfaced today.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the action-name set the neon bundle's ``g1_speak`` verb
#: (``cagataycali/neon-the-g1/tools/g1_speak.py``) admits today. The
#: five names are the ones the bundle's own action branches key on:
#: ``start`` / ``stop`` / ``status`` / ``say`` / ``debug``, and the
#: trailing guard (``return {"status": "error", "content": [{"text":
#: f"unknown action {action!r}"}]}``) refuses every other value at
#: the verb boundary. The SDK does not ship a canonical speak-action
#: enum (the choice is a Python string the verb branches on); the
#: snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``g1_speak`` side of the conversation.
#: Colocating the map with the verb mirrors ``_VOICE_PROVIDER_MAP``
#: in :mod:`~strands_robots.tools.g1.g1_voice_providers`: one
#: snapshot per neon-facing table, one verb pair per snapshot.
_SPEAK_ACTION_MAP: dict[str, dict[str, str]] = {
    "start": {
        "role": "transition",
        "description": (
            "Spawn the background bidi voice thread. The neon "
            "bundle's ``_runner_main`` builds the ``BidiAgent`` + "
            "``G1BidiAudioIO`` pair, starts the DDS chest speaker, "
            "and runs ``agent.run(inputs=[mic, briefing], "
            "outputs=[speaker, log])`` until the stop event fires. "
            "This is the only action that opens an audio write "
            "path; the other four read module-level state or run "
            "one-shot SDK calls."
        ),
    },
    "stop": {
        "role": "transition",
        "description": (
            "Set the background thread's stop event and join it. "
            "The neon bundle's ``_STATE['stop_event']`` is checked "
            "on the async bridge coroutine every 200 ms; setting "
            "it drives the bidi ``agent.run`` task to cancel, and "
            "the runner tears down the audio IO in its ``finally`` "
            "block. Refuses silently on a not-running thread with "
            "a ``not running`` status message."
        ),
    },
    "status": {
        "role": "read",
        "description": (
            "Read the module-level ``_STATE`` snapshot plus the "
            "``STATS`` counters from ``tools/g1_bidi_audio.py`` "
            "(``frames_captured``, ``g1_frames_sent``, "
            "``ref_buf_qsize``, ``energy_mean_abs``, "
            "``energy_max_abs``) and the last error. Does not "
            "touch the bus or the audio stack; safe to call on a "
            "stopped thread."
        ),
    },
    "say": {
        "role": "one_shot",
        "description": (
            "Speak ``text`` once through the G1 chest speaker via "
            "``AudioClient.TtsMaker(text, 0)``. Bypasses the bidi "
            "thread entirely: no mic, no AEC, no async loop, one "
            "synchronous SDK call under ``ensure_dds``. The "
            "``0`` argument is the neon-observed speaker-id "
            "default; a caller who wants a non-default speaker "
            "reaches the SDK directly. Refuses on empty ``text`` "
            'with ``"say requires text=..."`` at the verb '
            "boundary."
        ),
    },
    "debug": {
        "role": "diagnostic",
        "description": (
            "Print a diagnostic bundle for the bidi audio path: "
            "``_STATE`` snapshot, ``STATS`` counters, PyAudio "
            "input-device enumeration (with the auto-picked Brio "
            "row starred), pulseaudio default+source list "
            "(``pactl info`` / ``pactl list sources short``), "
            "cross-persona ``voice_bridge`` + ``agent_log`` "
            "stats, and the last error. Reads only; does not "
            "spawn a thread or write to the bus."
        ),
    },
}

#: Module-local refusal text for a name outside :data:`_SPEAK_ACTION_MAP`.
#: The SDK ships no dedicated "unknown speak action" ``rc`` (the
#: neon bundle's trailing guard is a caller-side shape refusal
#: returning ``{"status": "error", "content": [{"text": f"unknown
#: action {action!r} - use start/stop/status/say/debug"}]}``), so a
#: refusal on this envelope is a shape refusal, not a hardware FSM
#: refusal. Borrowing the motion-FSM ``7404`` entry from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` would mis-
#: label the shape refusal as a hardware FSM refusal (a caller
#: seeing ``\"Invalid FSM id\"`` on an unknown speak action would
#: read the wrong root cause); the module-local text quotes the
#: neon bundle's own guard phrasing verbatim so a caller reading
#: ``refusal_text`` sees the same string the neon verb surfaces
#: today. Named separately from the shared ``ERR_CODES`` table for
#: the same reason
#: :mod:`~strands_robots.tools.g1.g1_speak_vad_envelope`'s
#: ``_REFUSAL_TEXT_VAD_THRESHOLD`` and
#: ``_REFUSAL_TEXT_SILENCE_DURATION_MS`` sit module-local: the
#: refusal is per-envelope shape, not per-shared-code.
_REFUSAL_TEXT_UNKNOWN_ACTION: str = (
    "unknown speak action - the neon g1_speak verb admits only "
    "start/stop/status/say/debug; refs strands-labs/robots#358"
)


def _describe(name: str) -> dict[str, Any]:
    """Build the per-action descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_speak_actions_envelope`
    so :func:`g1_speak_action_admits`'s admitted-path payload names
    the same fields, and so a widen to the descriptor lands in one
    place. Every field is a snapshot read; no bus is touched.
    """
    entry = _SPEAK_ACTION_MAP[name]
    return {
        "name": name,
        "role": entry["role"],
        "description": entry["description"],
        "admits_speak_write": True,
    }


@tool
def g1_list_speak_actions_envelope() -> dict[str, Any]:
    """Return the action names ``g1_speak`` admits.

    Read-only. No driver instance, no DDS, no SDK, no audio stack:
    every field is a module-level constant. Useful before a future
    driver-side wrapper for ``g1_speak`` is called, so a caller can
    compare an intended ``action`` argument against the set the
    neon bundle documented, and can also read the ``role`` field
    on the returned descriptor to see which action opens the audio
    write path (``role="transition"``) vs which reads state
    (``role="read"``) vs which runs one-shot SDK calls
    (``role="one_shot"``) vs which prints diagnostics
    (``role="diagnostic"``).

    Returns:
        A dict with ``status``; an ``actions`` list of per-action
        descriptors sorted by ``name`` ascending, each carrying
        ``name`` (the action string the neon bundle branches on),
        ``role`` (one of ``transition`` / ``read`` / ``one_shot``
        / ``diagnostic``), ``description`` (the action label the
        neon bundle observed), and ``admits_speak_write`` (always
        ``True``, because every admitted action is a
        ``g1_speak``-shaped call by definition; the flag is
        surfaced so the descriptor shape matches
        :mod:`~strands_robots.tools.g1.g1_voice_providers` and
        :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim);
        a ``names`` list quoting the admitted action names in
        sorted order; and a ``refusals`` list carrying the module-
        local ``refusal_text`` a future write verb would surface
        on an unknown action name. Every field is a snapshot of an
        observed action label; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "actions": [_describe(name) for name in sorted(_SPEAK_ACTION_MAP)],
        "names": sorted(_SPEAK_ACTION_MAP),
        "refusals": [
            {"text": _REFUSAL_TEXT_UNKNOWN_ACTION},
        ],
    }


@tool
def g1_speak_action_admits(name: str | None = None) -> dict[str, Any]:
    """Decide whether an action ``name`` sits inside the admitted set.

    Read-only. Compares one argument against the neon-observed
    :data:`_SPEAK_ACTION_MAP` and reports the admitted descriptor
    on match, or the module-local refusal text a future driver-
    side wrapper would surface on miss. No driver instance, no
    DDS, no SDK, no audio stack: the decision reads only module-
    level constants and the argument itself.

    An action inside the admitted set is *not* the same as an
    admitted write: the neon bundle's runtime ``_probe_bidi`` also
    refuses on ``pywebrtc_audio`` / ``pyaudio`` /
    ``strands.experimental.bidi.BidiAgent`` missing at import
    time, the ``start`` branch refuses on ``OPENAI_API_KEY``
    missing when the provider is ``openai`` / ``openai_realtime``,
    and the ``say`` branch refuses on empty ``text`` at the verb
    boundary. None of those are snapshot answers; each is a
    live-host or per-argument read a caller reaches after this
    verb admits the action name. The returned payload names
    ``role`` so a caller comparing an intended write against the
    other conditions has the routing hint on hand.

    Args:
        name: The action name to check (``"start"``, ``"stop"``,
            ``"status"``, ``"say"``, or ``"debug"`` today). The
            comparison is case-sensitive against the snapshot in
            :data:`_SPEAK_ACTION_MAP`; a mis-cased or unknown
            name is refused with the module-local
            :data:`_REFUSAL_TEXT_UNKNOWN_ACTION`. Bool values
            (``True``/``False``) are refused with the same text
            because ``str(True) == "True"`` would otherwise
            silently mis-match; a non-string non-bool argument is
            refused with the same text for the same reason. An
            empty string is refused decidably rather than treated
            as a default.

    Returns:
        A dict with ``status``; on admit, an ``action`` descriptor
        with ``name``, ``role``, ``description``, and
        ``admits_speak_write`` (the same shape
        :func:`g1_list_speak_actions_envelope` returns). On
        refuse, ``refusal_text`` names the module-local shape-
        refusal string, plus a ``reason`` string that names why
        the argument was refused (missing argument, bool argument,
        non-string argument, empty-string argument, or unknown
        action).
    """
    if name is None:
        return {
            "status": "error",
            "refusal_text": _REFUSAL_TEXT_UNKNOWN_ACTION,
            "reason": (f"name is required; pass one of {sorted(_SPEAK_ACTION_MAP)} so the lookup is decidable"),
        }
    if isinstance(name, bool):
        return {
            "status": "error",
            "refusal_text": _REFUSAL_TEXT_UNKNOWN_ACTION,
            "reason": (f"name={name!r} is a bool; pass one of {sorted(_SPEAK_ACTION_MAP)} as a string"),
        }
    if not isinstance(name, str):
        return {
            "status": "error",
            "refusal_text": _REFUSAL_TEXT_UNKNOWN_ACTION,
            "reason": (f"name={name!r} is not a string; pass one of {sorted(_SPEAK_ACTION_MAP)} as a string"),
        }
    if name == "":
        return {
            "status": "error",
            "refusal_text": _REFUSAL_TEXT_UNKNOWN_ACTION,
            "reason": (f"name is the empty string; pass one of {sorted(_SPEAK_ACTION_MAP)} so the lookup is decidable"),
        }
    if name not in _SPEAK_ACTION_MAP:
        return {
            "status": "error",
            "refusal_text": _REFUSAL_TEXT_UNKNOWN_ACTION,
            "reason": (f"name={name!r} is not in the admitted set {sorted(_SPEAK_ACTION_MAP)}"),
        }
    return {
        "status": "success",
        "action": _describe(name),
    }
