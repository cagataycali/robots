"""The sample-rate lookup tools name what the neon bidi audio path pins.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) pins three
sample rates for the three surfaces it moves audio between:
``MIC_RATE = 16000`` (laptop USB mic capture, WebRTC AEC input),
``G1_RATE = 16000`` (G1 DDS chest-speaker feed and WebRTC AEC
reference), and ``OPENAI_RATE = 24000`` (OpenAI Realtime endpoint).
The :mod:`strands_robots.tools.g1.g1_bidi_audio_sample_rates` module
snapshots the observed mapping into a module-level dict and exposes
two agent-facing verbs -
:func:`g1_list_bidi_audio_sample_rates` (name the whole mapping) and
:func:`g1_bidi_audio_sample_rate_admits` (decide one query) - so a
caller can decide the refusal decidably before a future audio write
path is attempted. The tests here fix that contract without pulling
the SDK or the audio stack: the module is loadable on a host without
``unitree_sdk2py`` *and* without the optional bidi audio deps
(``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``) that
the neon bundle's runtime ``_probe_bidi`` check reaches for, so a
headless CI runner and Thor before an office bring-up can read the
rate mapping without triggering an import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime probe. The neon bundle's ``_probe_bidi`` is a live
  ``ImportError``-shaped check for ``pywebrtc_audio`` + ``pyaudio``
  + ``strands.experimental.bidi.BidiAgent``; a caller comparing an
  intended (role, rate) against both conditions (membership + audio
  stack present) reaches the probe after this verb admits the role
  name. This file does not exercise the probe.
* The endpoint's own rate policy. The OpenAI Realtime endpoint's
  24 kHz rate is fixed by the endpoint itself; a rate revision
  upstream is a driver-side update the snapshot re-reads. This file
  does not exercise the endpoint.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_bidi_audio_sample_rates import (
    _BIDI_AUDIO_SAMPLE_RATE_MAP,
    _INVALID_SAMPLE_RATE_CODE,
    g1_bidi_audio_sample_rate_admits,
    g1_list_bidi_audio_sample_rates,
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
    runner and Thor before an office bring-up. The rate snapshot is
    an integer table; no SDK submodule should load on the import
    path.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_sample_rates")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_sample_rates imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_audio_stack_module() -> None:
    """The tool module is loadable without the optional bidi audio deps.

    The neon bundle's ``_probe_bidi`` check reaches for
    ``pywebrtc_audio`` + ``pyaudio`` +
    ``strands.experimental.bidi.BidiAgent`` at runtime; those are
    optional dependencies the ``strands-robots`` package does not
    require. A caller who only wants to read the admitted rate
    mapping must not be forced to install the audio stack; a module
    that pulled any of those on import would refuse on a headless
    host the ``strands-robots`` package must run on.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_sample_rates")
    after = set(sys.modules)
    audio_dep_prefixes = ("pywebrtc_audio", "pyaudio", "strands.experimental.bidi")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_dep_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_sample_rates imports pulled audio-stack "
        f"submodules: {leaked}. The rate snapshot is an integer table; the "
        "audio stack loads on the write path, not on tool import."
    )


def test_the_snapshot_covers_the_neon_observed_roles() -> None:
    """The snapshot names the three roles the neon bundle pins.

    The neon bundle pins ``MIC_RATE`` (laptop USB mic capture),
    ``G1_RATE`` (G1 DDS chest-speaker feed / AEC reference), and
    ``OPENAI_RATE`` (OpenAI Realtime endpoint). A widen or narrow to
    the observed set is a driver-side decision; pinning the count
    here surfaces a silent drift as a shape change rather than as a
    quiet re-labeling of an existing role.
    """
    assert len(_BIDI_AUDIO_SAMPLE_RATE_MAP) == 3, (
        f"expected 3 admitted sample-rate roles, got "
        f"{len(_BIDI_AUDIO_SAMPLE_RATE_MAP)}: {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"
    )
    assert set(_BIDI_AUDIO_SAMPLE_RATE_MAP) == {
        "mic_rate",
        "g1_rate",
        "openai_rate",
    }, f"sample-rate snapshot drifted from the neon-observed set: {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"


def test_mic_rate_matches_the_neon_mic_rate_constant() -> None:
    """``mic_rate`` is 16 kHz, matching neon ``MIC_RATE = 16000``.

    The neon bundle names ``MIC_RATE = 16000`` for the WebRTC AEC
    input; the snapshot must quote the same integer or a caller
    would compare against a drifted rate and mis-route the resample.
    """
    assert _BIDI_AUDIO_SAMPLE_RATE_MAP["mic_rate"]["rate_hz"] == 16000, (
        f"mic_rate drifted from the neon ``MIC_RATE = 16000`` constant: "
        f"snapshot names {_BIDI_AUDIO_SAMPLE_RATE_MAP['mic_rate']['rate_hz']}"
    )


def test_g1_rate_matches_the_neon_g1_rate_constant() -> None:
    """``g1_rate`` is 16 kHz, matching neon ``G1_RATE = 16000``.

    The neon bundle names ``G1_RATE = 16000`` for the G1 DDS
    chest-speaker feed on ``rt/audio_stream`` and reuses the same
    rate for the WebRTC AEC's far-buffer reference queue; the
    snapshot must quote the same integer.
    """
    assert _BIDI_AUDIO_SAMPLE_RATE_MAP["g1_rate"]["rate_hz"] == 16000, (
        f"g1_rate drifted from the neon ``G1_RATE = 16000`` constant: "
        f"snapshot names {_BIDI_AUDIO_SAMPLE_RATE_MAP['g1_rate']['rate_hz']}"
    )


def test_openai_rate_matches_the_neon_openai_rate_constant() -> None:
    """``openai_rate`` is 24 kHz, matching neon ``OPENAI_RATE = 24000``.

    The neon bundle names ``OPENAI_RATE = 24000`` for the OpenAI
    Realtime endpoint's own sample rate; the snapshot must quote
    the same integer or the resample math on the endpoint side
    would drift.
    """
    assert _BIDI_AUDIO_SAMPLE_RATE_MAP["openai_rate"]["rate_hz"] == 24000, (
        f"openai_rate drifted from the neon ``OPENAI_RATE = 24000`` constant: "
        f"snapshot names {_BIDI_AUDIO_SAMPLE_RATE_MAP['openai_rate']['rate_hz']}"
    )


def test_mic_rate_and_g1_rate_are_numerically_equal() -> None:
    """The mic and G1 rates share the same integer.

    The neon bundle's ``G1_RATE = MIC_RATE = 16000`` equality is not
    a coincidence: the AEC near/far alignment is a same-rate
    comparison on the WebRTC side, so both surfaces must land on
    the same integer. Pinning the equality here surfaces a silent
    divergence on either side as a shape change.
    """
    assert _BIDI_AUDIO_SAMPLE_RATE_MAP["mic_rate"]["rate_hz"] == _BIDI_AUDIO_SAMPLE_RATE_MAP["g1_rate"]["rate_hz"], (
        f"mic_rate and g1_rate must share the same integer for the AEC "
        f"near/far alignment; snapshot names "
        f"mic_rate={_BIDI_AUDIO_SAMPLE_RATE_MAP['mic_rate']['rate_hz']} "
        f"g1_rate={_BIDI_AUDIO_SAMPLE_RATE_MAP['g1_rate']['rate_hz']}"
    )


def test_every_snapshot_entry_carries_a_positive_rate_hz() -> None:
    """Every admitted role names a positive integer ``rate_hz``.

    A role descriptor with a zero, negative, or non-integer
    ``rate_hz`` would fail the sample-rate contract on every
    resample library; the snapshot must carry a positive integer
    or callers reading the field would refuse for the wrong reason.
    """
    for role, entry in _BIDI_AUDIO_SAMPLE_RATE_MAP.items():
        assert "rate_hz" in entry, f"role {role!r} has no rate_hz; every admitted role must name the sample rate"
        assert isinstance(entry["rate_hz"], int) and not isinstance(entry["rate_hz"], bool), (
            f"role {role!r} has a non-int rate_hz: {entry['rate_hz']!r}"
        )
        assert entry["rate_hz"] > 0, f"role {role!r} has a non-positive rate_hz: {entry['rate_hz']}"


def test_every_snapshot_entry_carries_a_description() -> None:
    """Every admitted role carries a non-empty description.

    The description is what the caller reads to disambiguate the
    two 16 kHz roles (``mic_rate`` vs ``g1_rate``, numerically
    equal but semantically distinct); an empty description would
    leave the caller reading a bare integer without context.
    """
    for role, entry in _BIDI_AUDIO_SAMPLE_RATE_MAP.items():
        assert "description" in entry, (
            f"role {role!r} has no description; every admitted role must carry a caller-facing label"
        )
        assert isinstance(entry["description"], str) and entry["description"], f"role {role!r} has an empty description"


def test_the_refusal_code_matches_the_shared_gate_refusal() -> None:
    """The refusal code sits inside the shared error table.

    The WebRTC / OpenAI Realtime / DDS-speaker factories ship no
    numbered SDK rc for a bad sample-rate argument; this lookup
    uses the ``7404`` gate-refusal shape a future driver-side
    wrapper would quote when refusing at the same boundary. The
    code must decode against
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so a
    caller reading ``refusal_text`` sees the same string the
    driver's own ``_check_motion_gates`` would quote.
    """
    assert _INVALID_SAMPLE_RATE_CODE in ERR_CODES, (
        f"_INVALID_SAMPLE_RATE_CODE={_INVALID_SAMPLE_RATE_CODE} is not in "
        f"ERR_CODES; the refusal string would not decode. Update "
        "``_g1_common.ERR_CODES`` or point the constant at a "
        "registered code."
    )


def test_list_returns_every_role_in_sorted_order() -> None:
    """The list verb surfaces the admitted roles in stable order.

    A caller iterating the ``rates`` list must see the same order
    across calls so a diff against the returned payload does not
    fluctuate with dict-iteration order under a hostile Python
    build; the verb sorts by role ascending.
    """
    payload = _call(g1_list_bidi_audio_sample_rates)
    roles = [descriptor["role"] for descriptor in payload["rates"]]
    assert roles == sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP), (
        f"list verb returned roles in unsorted order: {roles}. Expected sorted: {sorted(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"
    )


def test_list_names_every_snapshot_role_and_no_others() -> None:
    """The list verb round-trips the snapshot with no drift.

    A silent divergence between the snapshot and the list verb's
    surface would let a widen on one side land without the other;
    the test round-trips the two sets to fix that contract.
    """
    payload = _call(g1_list_bidi_audio_sample_rates)
    listed = {descriptor["role"] for descriptor in payload["rates"]}
    assert listed == set(_BIDI_AUDIO_SAMPLE_RATE_MAP), (
        f"list verb surface {listed} drifted from snapshot {set(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"
    )
    assert set(payload["roles"]) == set(_BIDI_AUDIO_SAMPLE_RATE_MAP), (
        f"list verb roles field {set(payload['roles'])} drifted from snapshot {set(_BIDI_AUDIO_SAMPLE_RATE_MAP)}"
    )


def test_list_surfaces_every_descriptor_with_admits_flag_true() -> None:
    """Every listed descriptor names ``admits_bidi_writes=True``.

    The flag is surfaced so the descriptor shape matches
    :mod:`~strands_robots.tools.g1.g1_voice_providers` and
    :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim; every
    admitted rate is a bidi-shaped write by definition, so the
    flag is always ``True`` on the list side.
    """
    payload = _call(g1_list_bidi_audio_sample_rates)
    for descriptor in payload["rates"]:
        assert descriptor.get("admits_bidi_writes") is True, (
            f"listed descriptor for {descriptor.get('role')!r} does not carry admits_bidi_writes=True: {descriptor}"
        )


def test_list_surfaces_the_rate_hz_for_every_role() -> None:
    """The list verb quotes ``rate_hz`` on every listed descriptor.

    A caller reading the payload for the integer rate must not have
    to reach the snapshot dict directly; every listed descriptor
    must name its own ``rate_hz`` verbatim from the snapshot.
    """
    payload = _call(g1_list_bidi_audio_sample_rates)
    for descriptor in payload["rates"]:
        role = descriptor["role"]
        assert descriptor["rate_hz"] == _BIDI_AUDIO_SAMPLE_RATE_MAP[role]["rate_hz"], (
            f"listed descriptor for {role!r} drifted from snapshot: "
            f"verb={descriptor['rate_hz']} vs "
            f"snapshot={_BIDI_AUDIO_SAMPLE_RATE_MAP[role]['rate_hz']}"
        )


def test_list_surfaces_the_refusal_code_and_text() -> None:
    """The envelope carries the ``7404`` refusal code and decoded text.

    A caller planning a bidi start reads the ``refusals`` list to
    see the string a future driver-side wrapper would surface on
    an unknown role; the string must decode against the shared
    error table.
    """
    payload = _call(g1_list_bidi_audio_sample_rates)
    codes = {refusal["code"]: refusal["text"] for refusal in payload["refusals"]}
    assert _INVALID_SAMPLE_RATE_CODE in codes, (
        f"envelope refusals list does not carry the {_INVALID_SAMPLE_RATE_CODE} code: {payload['refusals']}"
    )
    assert codes[_INVALID_SAMPLE_RATE_CODE] == ERR_CODES[_INVALID_SAMPLE_RATE_CODE], (
        f"envelope refusal text drifted from ERR_CODES: "
        f"{codes[_INVALID_SAMPLE_RATE_CODE]!r} vs "
        f"{ERR_CODES[_INVALID_SAMPLE_RATE_CODE]!r}"
    )


def test_admits_returns_true_on_every_snapshot_role() -> None:
    """The admits verb round-trips every admitted role.

    Every entry in the snapshot must be admitted by the verb; a
    divergence would let a widen on the snapshot side land without
    the verb agreeing.
    """
    for role in _BIDI_AUDIO_SAMPLE_RATE_MAP:
        payload = _call(g1_bidi_audio_sample_rate_admits, role=role)
        assert payload["status"] == "success", f"admits verb refused a snapshot role {role!r}: {payload}"
        assert payload["rate"]["role"] == role, (
            f"admits verb returned a different role than requested: asked {role!r}, got {payload['rate']['role']!r}"
        )
        assert payload["rate"]["rate_hz"] == _BIDI_AUDIO_SAMPLE_RATE_MAP[role]["rate_hz"], (
            f"admits verb rate_hz drifted from snapshot for "
            f"{role!r}: verb={payload['rate']['rate_hz']} "
            f"vs snapshot={_BIDI_AUDIO_SAMPLE_RATE_MAP[role]['rate_hz']}"
        )


def test_admits_refuses_an_off_set_role_with_the_shared_code() -> None:
    """An off-set role refuses with the ``7404`` code and decoded text.

    A caller passing a role name outside the snapshot must see the
    same refusal shape a future driver-side wrapper would surface;
    the reason string names the admitted set so the caller can
    correct the argument.
    """
    payload = _call(g1_bidi_audio_sample_rate_admits, role="whisper_rate")
    assert payload["status"] == "error", f"admits verb admitted an off-set role: {payload}"
    assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
        f"admits verb refusal_code drifted: {payload['refusal_code']} vs {_INVALID_SAMPLE_RATE_CODE}"
    )
    assert payload["refusal_text"] == ERR_CODES[_INVALID_SAMPLE_RATE_CODE], (
        f"admits verb refusal_text drifted: {payload['refusal_text']!r} vs {ERR_CODES[_INVALID_SAMPLE_RATE_CODE]!r}"
    )
    assert "whisper_rate" in payload["reason"], (
        f"admits verb reason string does not quote the argument: {payload['reason']!r}"
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``True`` / ``False`` argument refuses decidably.

    Python's ``bool`` would silently mis-match against the string
    snapshot; the verb rejects it up front so the caller sees a
    shape error rather than a confusing "unknown role" refusal.
    """
    for value in (True, False):
        payload = _call(g1_bidi_audio_sample_rate_admits, role=value)
        assert payload["status"] == "error", f"admits verb admitted bool argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
            f"admits verb refusal_code for bool {value!r} drifted: {payload['refusal_code']}"
        )
        assert "bool" in payload["reason"], (
            f"admits verb reason for bool {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """A non-string non-bool argument refuses decidably.

    Ints, floats, lists, dicts, tuples: none of them are role
    names; the verb rejects them up front rather than reaching
    the membership branch where ``in`` would refuse for the wrong
    reason.
    """
    for value in (0, 1, 16000, 1.5, [], {}, (), object()):
        payload = _call(g1_bidi_audio_sample_rate_admits, role=value)
        assert payload["status"] == "error", f"admits verb admitted non-str argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
            f"admits verb refusal_code for {value!r} drifted: {payload['refusal_code']}"
        )
        assert "not a string" in payload["reason"], (
            f"admits verb reason for {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """An empty ``role`` refuses decidably, not as an off-set query.

    A caller who passes ``role=""`` almost certainly forgot to fill
    the argument; the verb rejects it with a shape reason so the
    error names the missing input rather than falsely claiming
    the empty string is an unknown role.
    """
    payload = _call(g1_bidi_audio_sample_rate_admits, role="")
    assert payload["status"] == "error", f"admits verb admitted empty string: {payload}"
    assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
        f"admits verb refusal_code for '' drifted: {payload['refusal_code']}"
    )
    assert "empty" in payload["reason"], (
        f"admits verb reason for '' does not name the empty-string shape error: {payload['reason']!r}"
    )


def test_admits_refuses_the_missing_argument_as_a_shape_error() -> None:
    """A ``None`` (default) ``role`` refuses with a missing-argument reason.

    A caller who invokes the verb without passing ``role`` sees the
    Python default of ``None``; the verb rejects that up front with
    a "role is required" reason so the caller sees the missing
    argument named, not a downstream membership refusal.
    """
    payload = _call(g1_bidi_audio_sample_rate_admits)
    assert payload["status"] == "error", f"admits verb admitted a missing role argument: {payload}"
    assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
        f"admits verb refusal_code for missing role drifted: {payload['refusal_code']}"
    )
    assert "required" in payload["reason"], (
        f"admits verb reason for missing role does not name the required argument: {payload['reason']!r}"
    )


def test_admits_is_case_sensitive_against_the_snapshot() -> None:
    """A mis-cased ``role`` refuses; the snapshot keys are case-sensitive.

    The neon bundle's constants are named in a fixed case
    (``MIC_RATE`` / ``G1_RATE`` / ``OPENAI_RATE``, snapshotted as
    lowercase); the verb must not silently normalise the argument's
    case so a caller who typed ``"MIC_RATE"`` sees a refusal
    naming the admitted set rather than a match on the
    coincidentally-uppercase constant name.
    """
    payload = _call(g1_bidi_audio_sample_rate_admits, role="MIC_RATE")
    assert payload["status"] == "error", f"admits verb admitted a mis-cased role: {payload}"
    assert payload["refusal_code"] == _INVALID_SAMPLE_RATE_CODE, (
        f"admits verb refusal_code for mis-cased role drifted: {payload['refusal_code']}"
    )
    assert "MIC_RATE" in payload["reason"], (
        f"admits verb reason does not quote the mis-cased argument: {payload['reason']!r}"
    )
