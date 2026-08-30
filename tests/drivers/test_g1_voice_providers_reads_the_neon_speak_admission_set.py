"""The voice-provider lookup tools name what ``g1_speak`` admits.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) admits a small set of
bidi-model providers for the G1 voice persona: ``openai`` /
``openai_realtime`` (both routed to the OpenAI Realtime factory),
``nova_sonic`` (Amazon Nova Sonic), and ``gemini`` (Google Gemini
Live). The :mod:`strands_robots.tools.g1.g1_voice_providers` module
snapshots the observed admitted set into a module-level dict and
exposes two agent-facing verbs -
:func:`g1_list_voice_providers` (name the whole set) and
:func:`g1_voice_provider_admits` (decide one query) - so a caller can
decide the refusal decidably before a future audio write path is
attempted. The tests here fix that contract without pulling the SDK
or the audio stack: the module is loadable on a host without
``unitree_sdk2py`` *and* without the optional bidi audio deps
(``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``) that
the neon bundle's runtime ``_probe_bidi`` check reaches for, so a
headless CI runner and Thor before an office bring-up can read the
provider set without triggering an import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime probe. The neon bundle's ``_probe_bidi`` is a live
  ``ImportError``-shaped check for ``pywebrtc_audio`` + ``pyaudio``
  + ``strands.experimental.bidi.BidiAgent``; a caller comparing an
  intended provider against both conditions (membership + audio
  stack present) reaches the probe after this verb admits the
  provider name. This file does not exercise the probe.
* The credential env-var. The bundle refuses ``openai`` /
  ``openai_realtime`` without ``OPENAI_API_KEY`` at write time; the
  membership snapshot names the ``credential_env`` on every
  descriptor so a caller has the env-var name on hand, but this
  file does not read the environment.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_voice_providers import (
    _INVALID_PROVIDER_CODE,
    _VOICE_PROVIDER_MAP,
    g1_list_voice_providers,
    g1_voice_provider_admits,
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
    runner and Thor before an office bring-up. The provider snapshot
    is a string table; no SDK submodule should load on the import
    path.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_voice_providers")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_voice_providers imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_audio_stack_module() -> None:
    """The tool module is loadable without the optional bidi audio deps.

    The neon bundle's ``_probe_bidi`` check reaches for
    ``pywebrtc_audio`` + ``pyaudio`` +
    ``strands.experimental.bidi.BidiAgent`` at runtime; those are
    optional dependencies the ``strands-robots`` package does not
    require. A caller who only wants to read the admitted provider
    set must not be forced to install the audio stack; a module that
    pulled any of those on import would refuse on a headless host
    the ``strands-robots`` package must run on.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_voice_providers")
    after = set(sys.modules)
    audio_dep_prefixes = ("pywebrtc_audio", "pyaudio", "strands.experimental.bidi")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_dep_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_voice_providers imports pulled audio-stack "
        f"submodules: {leaked}. The provider snapshot is a string table; the "
        "audio stack loads on the write path, not on tool import."
    )


def test_the_snapshot_covers_the_neon_observed_set() -> None:
    """The snapshot names the four providers the neon bundle documented.

    The neon bundle's ``g1_speak`` docstring names ``openai`` /
    ``nova_sonic`` / ``gemini`` as the ``VOICE_PROVIDER`` env-var
    options, and its own guard code treats ``openai`` and
    ``openai_realtime`` as identical members. A widen or narrow to
    the observed set is a driver-side decision; pinning the count
    here surfaces a silent drift as a shape change rather than as a
    quiet re-labeling of an existing provider.
    """
    assert len(_VOICE_PROVIDER_MAP) == 4, (
        f"expected 4 admitted providers, got {len(_VOICE_PROVIDER_MAP)}: {sorted(_VOICE_PROVIDER_MAP)}"
    )
    assert set(_VOICE_PROVIDER_MAP) == {
        "openai",
        "openai_realtime",
        "nova_sonic",
        "gemini",
    }, f"provider snapshot drifted from the neon-observed set: {sorted(_VOICE_PROVIDER_MAP)}"


def test_openai_and_openai_realtime_share_a_credential_env() -> None:
    """The two OpenAI aliases route to the same credential env-var.

    The neon bundle's own guard code
    (``prov in ("openai", "openai_realtime")``) treats the two names
    identically for the ``OPENAI_API_KEY`` env-var check; a caller
    who reads the ``credential_env`` off one descriptor and compares
    it against the other must see the same env-var name, or the
    caller-side credential check would refuse on one alias and admit
    on the other.
    """
    openai_env = _VOICE_PROVIDER_MAP["openai"]["credential_env"]
    realtime_env = _VOICE_PROVIDER_MAP["openai_realtime"]["credential_env"]
    assert openai_env == realtime_env == "OPENAI_API_KEY", (
        f"the openai / openai_realtime aliases route to different "
        f"credential envs: openai={openai_env!r} vs "
        f"openai_realtime={realtime_env!r}. Both must name "
        "``OPENAI_API_KEY`` so a caller's credential check admits "
        "both aliases identically."
    )


def test_every_snapshot_entry_carries_a_credential_env() -> None:
    """Every admitted provider names a non-empty credential env-var.

    A provider descriptor without a ``credential_env`` field would
    force the caller to know the env-var out of band; the neon
    bundle's own guard code branches on the field, so every entry
    must name it.
    """
    for name, entry in _VOICE_PROVIDER_MAP.items():
        assert "credential_env" in entry, (
            f"provider {name!r} has no credential_env; every admitted "
            "provider must name the env-var its factory reaches for"
        )
        assert isinstance(entry["credential_env"], str) and entry["credential_env"], (
            f"provider {name!r} has an empty credential_env: {entry['credential_env']!r}"
        )


def test_every_snapshot_entry_carries_a_description() -> None:
    """Every admitted provider carries a non-empty description.

    The description is what the caller reads to disambiguate the
    aliased names (``openai`` vs ``openai_realtime``) and to see the
    provider's default voice name; an empty description would leave
    the caller reading a bare enum without context.
    """
    for name, entry in _VOICE_PROVIDER_MAP.items():
        assert "description" in entry, (
            f"provider {name!r} has no description; every admitted provider must carry a caller-facing label"
        )
        assert isinstance(entry["description"], str) and entry["description"], (
            f"provider {name!r} has an empty description"
        )


def test_the_refusal_code_matches_the_shared_gate_refusal() -> None:
    """The refusal code sits inside the shared error table.

    The neon bundle refused unknown providers at the verb boundary
    with a caller-side shape refusal; this lookup uses the ``7404``
    gate-refusal shape a future driver-side wrapper would quote when
    refusing at the same boundary. The code must decode against
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so a caller
    reading ``refusal_text`` sees the same string the driver's own
    ``_check_motion_gates`` would quote.
    """
    assert _INVALID_PROVIDER_CODE in ERR_CODES, (
        f"_INVALID_PROVIDER_CODE={_INVALID_PROVIDER_CODE} is not in "
        f"ERR_CODES; the refusal string would not decode. Update "
        "``_g1_common.ERR_CODES`` or point the constant at a "
        "registered code."
    )


def test_list_returns_every_provider_in_sorted_order() -> None:
    """The list verb surfaces the admitted names in stable order.

    A caller iterating the ``providers`` list must see the same
    order across calls so a diff against the returned payload does
    not fluctuate with dict-iteration order under a hostile Python
    build; the verb sorts by name ascending.
    """
    payload = _call(g1_list_voice_providers)
    names = [descriptor["name"] for descriptor in payload["providers"]]
    assert names == sorted(_VOICE_PROVIDER_MAP), (
        f"list verb returned providers in unsorted order: {names}. Expected sorted: {sorted(_VOICE_PROVIDER_MAP)}"
    )


def test_list_names_every_snapshot_provider_and_no_others() -> None:
    """The list verb round-trips the snapshot with no drift.

    A silent divergence between the snapshot and the list verb's
    surface would let a widen on one side land without the other;
    the test round-trips the two sets to fix that contract.
    """
    payload = _call(g1_list_voice_providers)
    listed = {descriptor["name"] for descriptor in payload["providers"]}
    assert listed == set(_VOICE_PROVIDER_MAP), (
        f"list verb surface {listed} drifted from snapshot {set(_VOICE_PROVIDER_MAP)}"
    )
    assert set(payload["names"]) == set(_VOICE_PROVIDER_MAP), (
        f"list verb names field {set(payload['names'])} drifted from snapshot {set(_VOICE_PROVIDER_MAP)}"
    )


def test_list_surfaces_every_descriptor_with_admits_flag_true() -> None:
    """Every listed descriptor names ``admits_bidi_writes=True``.

    The flag is surfaced so the descriptor shape matches
    :mod:`~strands_robots.tools.g1.g1_balance_modes` and
    :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim; every
    admitted provider is a bidi-shaped write by definition, so the
    flag is always ``True`` on the list side. A caller reading the
    payload for shape parity across verbs must see the same flag on
    every row.
    """
    payload = _call(g1_list_voice_providers)
    for descriptor in payload["providers"]:
        assert descriptor.get("admits_bidi_writes") is True, (
            f"listed descriptor for {descriptor.get('name')!r} does not carry admits_bidi_writes=True: {descriptor}"
        )


def test_list_surfaces_the_refusal_code_and_text() -> None:
    """The envelope carries the ``7404`` refusal code and decoded text.

    A caller planning a ``g1_speak`` call reads the ``refusals``
    list to see the string a future driver-side wrapper would surface
    on an unknown provider; the string must decode against the
    shared error table.
    """
    payload = _call(g1_list_voice_providers)
    codes = {refusal["code"]: refusal["text"] for refusal in payload["refusals"]}
    assert _INVALID_PROVIDER_CODE in codes, (
        f"envelope refusals list does not carry the {_INVALID_PROVIDER_CODE} code: {payload['refusals']}"
    )
    assert codes[_INVALID_PROVIDER_CODE] == ERR_CODES[_INVALID_PROVIDER_CODE], (
        f"envelope refusal text drifted from ERR_CODES: "
        f"{codes[_INVALID_PROVIDER_CODE]!r} vs "
        f"{ERR_CODES[_INVALID_PROVIDER_CODE]!r}"
    )


def test_admits_returns_true_on_every_snapshot_provider() -> None:
    """The admits verb round-trips every admitted name.

    Every entry in the snapshot must be admitted by the verb; a
    divergence would let a widen on the snapshot side land without
    the verb agreeing.
    """
    for name in _VOICE_PROVIDER_MAP:
        payload = _call(g1_voice_provider_admits, name=name)
        assert payload["status"] == "success", f"admits verb refused a snapshot provider {name!r}: {payload}"
        assert payload["provider"]["name"] == name, (
            f"admits verb returned a different provider than requested: "
            f"asked {name!r}, got {payload['provider']['name']!r}"
        )
        assert payload["provider"]["credential_env"] == (_VOICE_PROVIDER_MAP[name]["credential_env"]), (
            f"admits verb credential_env drifted from snapshot for "
            f"{name!r}: verb={payload['provider']['credential_env']!r} "
            f"vs snapshot={_VOICE_PROVIDER_MAP[name]['credential_env']!r}"
        )


def test_admits_refuses_an_off_set_provider_with_the_shared_code() -> None:
    """An off-set name refuses with the ``7404`` code and decoded text.

    A caller passing a provider name outside the snapshot must see
    the same refusal shape a future driver-side wrapper would
    surface; the reason string names the admitted set so the caller
    can correct the argument.
    """
    payload = _call(g1_voice_provider_admits, name="claude_realtime")
    assert payload["status"] == "error", f"admits verb admitted an off-set provider: {payload}"
    assert payload["refusal_code"] == _INVALID_PROVIDER_CODE, (
        f"admits verb refusal_code drifted: {payload['refusal_code']} vs {_INVALID_PROVIDER_CODE}"
    )
    assert payload["refusal_text"] == ERR_CODES[_INVALID_PROVIDER_CODE], (
        f"admits verb refusal_text drifted: {payload['refusal_text']!r} vs {ERR_CODES[_INVALID_PROVIDER_CODE]!r}"
    )
    assert "claude_realtime" in payload["reason"], (
        f"admits verb reason string does not quote the argument: {payload['reason']!r}"
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``True`` / ``False`` argument refuses decidably.

    Python's ``bool`` would silently mis-match against the string
    snapshot; the verb rejects it up front so the caller sees a
    shape error rather than a confusing "unknown provider" refusal.
    """
    for value in (True, False):
        payload = _call(g1_voice_provider_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted bool argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_PROVIDER_CODE, (
            f"admits verb refusal_code for bool {value!r} drifted: {payload['refusal_code']}"
        )
        assert "bool" in payload["reason"], (
            f"admits verb reason for bool {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """A non-string non-bool argument refuses decidably.

    Ints, floats, lists, dicts, tuples: none of them are provider
    names; the verb rejects them up front rather than reaching the
    membership branch where ``in`` would refuse for the wrong reason
    (or worse, silently match on a coincidental repr).
    """
    for value in (0, 1, 1.5, [], {}, (), object()):
        payload = _call(g1_voice_provider_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted non-str argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_PROVIDER_CODE, (
            f"admits verb refusal_code for {value!r} drifted: {payload['refusal_code']}"
        )
        assert "not a string" in payload["reason"], (
            f"admits verb reason for {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """An empty ``name`` refuses decidably, not as an off-set query.

    A caller who passes ``name=""`` almost certainly forgot to fill
    the argument; the verb rejects it with a shape reason so the
    error names the missing input rather than falsely claiming the
    empty string is an unknown provider.
    """
    payload = _call(g1_voice_provider_admits, name="")
    assert payload["status"] == "error", f"admits verb admitted empty string: {payload}"
    assert payload["refusal_code"] == _INVALID_PROVIDER_CODE, (
        f"admits verb refusal_code for '' drifted: {payload['refusal_code']}"
    )
    assert "empty" in payload["reason"], (
        f"admits verb reason for '' does not name the empty-string shape error: {payload['reason']!r}"
    )


def test_admits_refuses_the_missing_argument_as_a_shape_error() -> None:
    """A ``None`` (default) ``name`` refuses with a missing-argument reason.

    A caller who invokes the verb without passing ``name`` sees the
    Python default of ``None``; the verb rejects that up front with a
    "name is required" reason so the caller sees the missing argument
    named, not a downstream membership refusal.
    """
    payload = _call(g1_voice_provider_admits)
    assert payload["status"] == "error", f"admits verb admitted a missing name argument: {payload}"
    assert payload["refusal_code"] == _INVALID_PROVIDER_CODE, (
        f"admits verb refusal_code for missing name drifted: {payload['refusal_code']}"
    )
    assert "required" in payload["reason"], (
        f"admits verb reason for missing name does not name the required argument: {payload['reason']!r}"
    )
