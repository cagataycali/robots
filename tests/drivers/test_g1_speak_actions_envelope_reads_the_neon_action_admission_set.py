"""The speak-actions-envelope lookup tools name what ``g1_speak`` admits.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) admits a small set
of action names on its ``action`` keyword argument: ``start`` /
``stop`` / ``status`` / ``say`` / ``debug``. The bundle's own
trailing guard (``return {"status": "error", "content": [{"text":
f"unknown action {action!r} - use start/stop/status/say/debug"}]}``)
rejects every other value at the verb boundary. The
:mod:`strands_robots.tools.g1.g1_speak_actions_envelope` module
snapshots the observed admitted set into a module-level dict and
exposes two agent-facing verbs -
:func:`g1_list_speak_actions_envelope` (name the whole set) and
:func:`g1_speak_action_admits` (decide one query) - so a caller can
decide the refusal decidably before a future audio write path is
attempted. The tests here fix that contract without pulling the SDK
or the audio stack: the module is loadable on a host without
``unitree_sdk2py`` *and* without the optional bidi audio deps
(``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``)
that the neon bundle's runtime ``_probe_bidi`` check reaches for,
so a headless CI runner and Thor before an office bring-up can
read the action set without triggering an import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime probe. The neon bundle's ``_probe_bidi`` is a live
  ``ImportError``-shaped check for ``pywebrtc_audio`` + ``pyaudio``
  + ``strands.experimental.bidi.BidiAgent``; a caller comparing an
  intended action against both conditions (membership + audio
  stack present) reaches the probe after this verb admits the
  action name. This file does not exercise the probe.
* The per-action argument shape. The bundle refuses ``say`` on
  empty ``text``, ``start`` on missing ``OPENAI_API_KEY`` when the
  provider is ``openai`` / ``openai_realtime``, and ``stop`` on a
  not-running thread with a distinct status message. None of those
  are snapshot answers; each is a live-host or per-argument read
  a caller reaches after this envelope admits the action name.
  This file does not read the environment or the driver state.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_speak_actions_envelope import (
    _REFUSAL_TEXT_UNKNOWN_ACTION,
    _SPEAK_ACTION_MAP,
    g1_list_speak_actions_envelope,
    g1_speak_action_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up. The action snapshot is
    a string table; no SDK submodule should load on the import path.
    """
    sys.modules.pop("strands_robots.tools.g1.g1_speak_actions_envelope", None)
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_speak_actions_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_speak_actions_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_audio_stack_module() -> None:
    """The tool module is loadable without the optional bidi audio deps.

    The neon bundle's ``_probe_bidi`` check reaches for
    ``pywebrtc_audio`` + ``pyaudio`` +
    ``strands.experimental.bidi.BidiAgent`` at runtime; those are
    optional dependencies the ``strands-robots`` package does not
    require. A caller who only wants to read the admitted action
    set must not be forced to install the audio stack; a module
    that pulled any of those on import would refuse on a headless
    host the ``strands-robots`` package must run on.
    """
    sys.modules.pop("strands_robots.tools.g1.g1_speak_actions_envelope", None)
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_speak_actions_envelope")
    after = set(sys.modules)
    audio_dep_prefixes = ("pywebrtc_audio", "pyaudio", "strands.experimental.bidi")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_dep_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_speak_actions_envelope imports pulled audio-stack "
        f"submodules: {leaked}. The action snapshot is a string table; the "
        "audio stack loads on the write path, not on tool import."
    )


def test_the_snapshot_covers_the_neon_observed_set() -> None:
    """The snapshot names the five actions the neon bundle's guard lists.

    The neon bundle's trailing guard (``"unknown action ... - use "
    "start/stop/status/say/debug"``) names five action strings the
    verb branches on. A widen or narrow to the observed set is a
    driver-side decision; pinning the count here surfaces a silent
    drift as a shape change rather than as a quiet re-labeling of
    an existing action.
    """
    assert len(_SPEAK_ACTION_MAP) == 5, (
        f"expected 5 admitted actions, got {len(_SPEAK_ACTION_MAP)}: {sorted(_SPEAK_ACTION_MAP)}"
    )
    assert set(_SPEAK_ACTION_MAP) == {
        "start",
        "stop",
        "status",
        "say",
        "debug",
    }, f"action snapshot drifted from the neon-observed set: {sorted(_SPEAK_ACTION_MAP)}"


def test_every_snapshot_entry_carries_a_role() -> None:
    """Every admitted action names a non-empty role.

    The ``role`` field surfaces which action opens the audio write
    path (``transition``) vs which reads state (``read``) vs which
    runs one-shot SDK calls (``one_shot``) vs which prints
    diagnostics (``diagnostic``); a caller reading the envelope for
    routing must see the field on every row.
    """
    for name, entry in _SPEAK_ACTION_MAP.items():
        assert "role" in entry, f"action {name!r} has no role; every admitted action must name its routing role"
        assert isinstance(entry["role"], str) and entry["role"], f"action {name!r} has an empty role: {entry['role']!r}"


def test_the_role_labels_sit_inside_the_documented_partition() -> None:
    """Every action's role sits inside the four-label partition.

    The docstring surfaces exactly four role labels
    (``transition`` / ``read`` / ``one_shot`` / ``diagnostic``); a
    caller reading the field for routing must not see a fifth label
    silently appear. A widen to the label set is a snapshot
    decision that lands on this test.
    """
    admitted_roles = {"transition", "read", "one_shot", "diagnostic"}
    for name, entry in _SPEAK_ACTION_MAP.items():
        assert entry["role"] in admitted_roles, (
            f"action {name!r} role={entry['role']!r} sits outside the documented partition {sorted(admitted_roles)}"
        )


def test_start_and_stop_are_transitions() -> None:
    """The two thread-lifecycle actions carry the ``transition`` role.

    The neon bundle's ``start`` branch spawns the bidi thread and
    the ``stop`` branch sets its stop event; both change the
    thread's lifecycle state. A caller planning a write compares
    intent against ``role="transition"`` to see which actions
    change bus-facing state.
    """
    assert _SPEAK_ACTION_MAP["start"]["role"] == "transition", (
        f"start role={_SPEAK_ACTION_MAP['start']['role']!r} drifted from 'transition'"
    )
    assert _SPEAK_ACTION_MAP["stop"]["role"] == "transition", (
        f"stop role={_SPEAK_ACTION_MAP['stop']['role']!r} drifted from 'transition'"
    )


def test_status_is_a_read_and_debug_is_a_diagnostic() -> None:
    """The two read-only actions carry disjoint roles.

    The neon bundle's ``status`` branch reads module-level counters
    (``_STATE`` + ``STATS``) without touching the bus; the
    ``debug`` branch adds device enumeration and cross-persona
    logs. Both are read-only but serve different callers: a
    caller who wants a shape-stable snapshot uses ``status``, one
    who wants a device dump uses ``debug``. The role labels reflect
    that split.
    """
    assert _SPEAK_ACTION_MAP["status"]["role"] == "read", (
        f"status role={_SPEAK_ACTION_MAP['status']['role']!r} drifted from 'read'"
    )
    assert _SPEAK_ACTION_MAP["debug"]["role"] == "diagnostic", (
        f"debug role={_SPEAK_ACTION_MAP['debug']['role']!r} drifted from 'diagnostic'"
    )


def test_say_is_a_one_shot_action() -> None:
    """The ``say`` action carries the ``one_shot`` role.

    The neon bundle's ``say`` branch bypasses the bidi thread
    entirely: it runs ``AudioClient.TtsMaker(text, 0)`` under
    ``ensure_dds`` and returns. A caller planning a one-shot
    TTS write reads ``role="one_shot"`` to see that the action
    does not spawn a thread.
    """
    assert _SPEAK_ACTION_MAP["say"]["role"] == "one_shot", (
        f"say role={_SPEAK_ACTION_MAP['say']['role']!r} drifted from 'one_shot'"
    )


def test_every_snapshot_entry_carries_a_description() -> None:
    """Every admitted action carries a non-empty description.

    The description is what the caller reads to disambiguate the
    aliased-looking names (``say`` runs TtsMaker but ``start``
    runs bidi; both output audio); an empty description would leave
    the caller reading a bare enum without context.
    """
    for name, entry in _SPEAK_ACTION_MAP.items():
        assert "description" in entry, (
            f"action {name!r} has no description; every admitted action must carry a caller-facing label"
        )
        assert isinstance(entry["description"], str) and entry["description"], (
            f"action {name!r} has an empty description"
        )


def test_the_refusal_text_names_the_admitted_action_set() -> None:
    """The refusal text quotes the neon bundle's guard verbatim on the set.

    The neon bundle's trailing guard reads ``"unknown action ... -
    use start/stop/status/say/debug"``; the module-local
    :data:`_REFUSAL_TEXT_UNKNOWN_ACTION` must name the same five
    action strings so a caller reading ``refusal_text`` sees a
    string that resolves against the neon verb's own error message.
    A widen to the admitted set that does not land on the refusal
    text would leave a caller's log grep out of sync with the neon
    bundle.
    """
    for name in _SPEAK_ACTION_MAP:
        assert name in _REFUSAL_TEXT_UNKNOWN_ACTION, (
            f"refusal text does not name admitted action {name!r}: {_REFUSAL_TEXT_UNKNOWN_ACTION!r}"
        )


def test_the_refusal_text_names_the_repo_issue_reference() -> None:
    """The refusal text cites the resolvable issue for the driver-side gate.

    The refusal string a future driver-side wrapper would quote
    must cite the repo-scoped issue where the audio-write gate work
    is tracked (refs strands-labs/robots#358), so a caller reading
    the string can navigate to the open decision. The rule is
    strands-labs/robots#2872: refusal strings cite resolvable
    references.
    """
    assert "#358" in _REFUSAL_TEXT_UNKNOWN_ACTION, (
        f"refusal text does not cite the tracking issue: {_REFUSAL_TEXT_UNKNOWN_ACTION!r}"
    )


def test_list_returns_every_action_in_sorted_order() -> None:
    """The list verb surfaces the admitted names in stable order.

    A caller iterating the ``actions`` list must see the same
    order across calls so a diff against the returned payload does
    not fluctuate with dict-iteration order under a hostile Python
    build; the verb sorts by name ascending.
    """
    payload = _call(g1_list_speak_actions_envelope)
    names = [descriptor["name"] for descriptor in payload["actions"]]
    assert names == sorted(_SPEAK_ACTION_MAP), (
        f"list verb returned actions in unsorted order: {names}. Expected sorted: {sorted(_SPEAK_ACTION_MAP)}"
    )


def test_list_names_every_snapshot_action_and_no_others() -> None:
    """The list verb round-trips the snapshot with no drift.

    A silent divergence between the snapshot and the list verb's
    surface would let a widen on one side land without the other;
    the test round-trips the two sets to fix that contract.
    """
    payload = _call(g1_list_speak_actions_envelope)
    listed = {descriptor["name"] for descriptor in payload["actions"]}
    assert listed == set(_SPEAK_ACTION_MAP), (
        f"list verb surface {listed} drifted from snapshot {set(_SPEAK_ACTION_MAP)}"
    )
    assert set(payload["names"]) == set(_SPEAK_ACTION_MAP), (
        f"list verb names field {set(payload['names'])} drifted from snapshot {set(_SPEAK_ACTION_MAP)}"
    )


def test_list_surfaces_every_descriptor_with_admits_flag_true() -> None:
    """Every listed descriptor names ``admits_speak_write=True``.

    The flag is surfaced so the descriptor shape matches
    :mod:`~strands_robots.tools.g1.g1_voice_providers` and
    :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim; every
    admitted action is a ``g1_speak``-shaped call by definition,
    so the flag is always ``True`` on the list side. A caller
    reading the payload for shape parity across verbs must see the
    same flag on every row.
    """
    payload = _call(g1_list_speak_actions_envelope)
    for descriptor in payload["actions"]:
        assert descriptor.get("admits_speak_write") is True, (
            f"listed descriptor for {descriptor.get('name')!r} does not carry admits_speak_write=True: {descriptor}"
        )


def test_list_surfaces_the_refusal_text() -> None:
    """The envelope carries the module-local refusal text.

    A caller planning a ``g1_speak`` call reads the ``refusals``
    list to see the string a future driver-side wrapper would
    surface on an unknown action; the string is the module-local
    shape-refusal text, not a shared FSM-motion refusal from
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`.
    """
    payload = _call(g1_list_speak_actions_envelope)
    texts = {refusal["text"] for refusal in payload["refusals"]}
    assert _REFUSAL_TEXT_UNKNOWN_ACTION in texts, (
        f"envelope refusals list does not carry the module-local refusal text: {payload['refusals']}"
    )


def test_admits_returns_success_on_every_snapshot_action() -> None:
    """The admits verb round-trips every admitted name.

    Every entry in the snapshot must be admitted by the verb; a
    divergence would let a widen on the snapshot side land without
    the verb agreeing.
    """
    for name in _SPEAK_ACTION_MAP:
        payload = _call(g1_speak_action_admits, name=name)
        assert payload["status"] == "success", f"admits verb refused a snapshot action {name!r}: {payload}"
        assert payload["action"]["name"] == name, (
            f"admits verb returned a different action than requested: asked {name!r}, got {payload['action']['name']!r}"
        )
        assert payload["action"]["role"] == _SPEAK_ACTION_MAP[name]["role"], (
            f"admits verb role drifted from snapshot for "
            f"{name!r}: verb={payload['action']['role']!r} "
            f"vs snapshot={_SPEAK_ACTION_MAP[name]['role']!r}"
        )


def test_admits_refuses_an_off_set_action_with_the_module_local_text() -> None:
    """An off-set name refuses with the module-local shape-refusal text.

    A caller passing an action name outside the snapshot must see
    the same refusal shape a future driver-side wrapper would
    surface; the reason string names the admitted set so the caller
    can correct the argument.
    """
    payload = _call(g1_speak_action_admits, name="restart")
    assert payload["status"] == "error", f"admits verb admitted an off-set action: {payload}"
    assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
        f"admits verb refusal_text drifted: {payload['refusal_text']!r} vs {_REFUSAL_TEXT_UNKNOWN_ACTION!r}"
    )
    assert "restart" in payload["reason"], (
        f"admits verb reason string does not quote the argument: {payload['reason']!r}"
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``True`` / ``False`` argument refuses decidably.

    Python's ``bool`` would silently mis-match against the string
    snapshot; the verb rejects it up front so the caller sees a
    shape error rather than a confusing "unknown action" refusal.
    """
    for value in (True, False):
        payload = _call(g1_speak_action_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted bool argument {value!r}: {payload}"
        assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
            f"admits verb refusal_text for bool {value!r} drifted: {payload['refusal_text']!r}"
        )
        assert "bool" in payload["reason"], (
            f"admits verb reason for bool {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """A non-string non-bool argument refuses decidably.

    Ints, floats, lists, dicts, tuples: none of them are action
    names; the verb rejects them up front rather than reaching the
    membership branch where ``in`` would refuse for the wrong
    reason (or worse, silently match on a coincidental repr).
    """
    for value in (0, 1, 1.5, [], {}, (), object()):
        payload = _call(g1_speak_action_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted non-str argument {value!r}: {payload}"
        assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
            f"admits verb refusal_text for {value!r} drifted: {payload['refusal_text']!r}"
        )
        assert "not a string" in payload["reason"], (
            f"admits verb reason for {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """An empty ``name`` refuses decidably, not as an off-set query.

    A caller who passes ``name=""`` almost certainly forgot to
    fill the argument; the verb rejects it with a shape reason so
    the error names the missing input rather than falsely claiming
    the empty string is an unknown action.
    """
    payload = _call(g1_speak_action_admits, name="")
    assert payload["status"] == "error", f"admits verb admitted empty string: {payload}"
    assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
        f"admits verb refusal_text for '' drifted: {payload['refusal_text']!r}"
    )
    assert "empty" in payload["reason"], (
        f"admits verb reason for '' does not name the empty-string shape error: {payload['reason']!r}"
    )


def test_admits_refuses_the_missing_argument_as_a_shape_error() -> None:
    """A ``None`` (default) ``name`` refuses with a missing-argument reason.

    A caller who invokes the verb without passing ``name`` sees
    the Python default of ``None``; the verb rejects that up front
    with a "name is required" reason so the caller sees the
    missing argument named, not a downstream membership refusal.
    """
    payload = _call(g1_speak_action_admits)
    assert payload["status"] == "error", f"admits verb admitted a missing name argument: {payload}"
    assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
        f"admits verb refusal_text for missing name drifted: {payload['refusal_text']!r}"
    )
    assert "required" in payload["reason"], (
        f"admits verb reason for missing name does not name the required argument: {payload['reason']!r}"
    )


def test_admits_is_case_sensitive_against_the_snapshot() -> None:
    """The admits verb compares case-sensitively against the snapshot.

    The neon bundle's own action branches (``if action ==
    "start":`` etc.) are case-sensitive; the verb must be too, so
    a caller passing ``"START"`` sees a refusal rather than
    silently matching the ``"start"`` branch. Coercing case here
    would leave the verb agreeing with a call the neon bundle
    itself would refuse.
    """
    for miscased in ("START", "Start", "sTaRt", "STATUS", "Debug"):
        payload = _call(g1_speak_action_admits, name=miscased)
        assert payload["status"] == "error", f"admits verb admitted a mis-cased action {miscased!r}: {payload}"
        assert payload["refusal_text"] == _REFUSAL_TEXT_UNKNOWN_ACTION, (
            f"admits verb refusal_text for mis-cased {miscased!r} drifted: {payload['refusal_text']!r}"
        )


def test_admits_and_list_agree_on_the_admitted_set() -> None:
    """The two verbs surface the same admitted set.

    A silent divergence between :func:`g1_list_speak_actions_envelope`
    and :func:`g1_speak_action_admits` would let a widen on one side
    land without the other; the test cross-checks that every name
    the list verb returns is admitted by the admits verb and vice
    versa.
    """
    payload = _call(g1_list_speak_actions_envelope)
    for name in payload["names"]:
        admit_payload = _call(g1_speak_action_admits, name=name)
        assert admit_payload["status"] == "success", (
            f"list verb returned {name!r} but admits verb refused it: {admit_payload}"
        )
    for name in _SPEAK_ACTION_MAP:
        assert name in payload["names"], f"snapshot has {name!r} but list verb did not surface it: {payload['names']}"
