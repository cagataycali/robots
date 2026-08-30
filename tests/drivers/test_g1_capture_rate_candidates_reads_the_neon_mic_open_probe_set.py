"""The capture-rate candidate lookup tools name what the neon mic-open probes.

The neon bundle's ``g1_bidi_audio`` module
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) opens the USB mic
device through PyAudio before handing frames to the bidi agent, and
it does that by iterating a fixed list of candidate sample rates
against ``PyAudio.is_format_supported`` until one is accepted; the
neon constant ``CAPTURE_RATE_CANDIDATES = [16000, 48000, 44100,
32000]`` names that list in the order it is tried, and the neon
``pick_capture_rate`` helper falls back to ``48000`` when the whole
list is refused. The
:mod:`strands_robots.tools.g1.g1_capture_rate_candidates` module
snapshots that ordered list and the fallback value into module-level
constants and exposes two agent-facing verbs -
:func:`g1_list_capture_rate_candidates` (name the whole list) and
:func:`g1_capture_rate_candidate_admits` (decide one query) - so a
caller can name the rate set decidably before a future driver-side
wrapper for ``g1_speak`` is attempted. The tests here fix that
contract without pulling the SDK or the audio stack: the module is
loadable on a host without ``unitree_sdk2py`` *and* without
``pyaudio`` installed, so a headless CI runner and Thor before an
office bring-up can read the rate set without triggering an
import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime rate probe. The neon bundle's ``pick_capture_rate``
  runs ``PyAudio.is_format_supported`` against each candidate; that
  probe is a live host read reaching for the audio backend, and it
  is out of scope for this lookup. A caller comparing an intended
  write against both conditions (membership + backend support)
  reaches the probe after this verb admits the rate. This file
  does not exercise the probe.
* The per-mic supported-rate table. The Brio-vs-DJI split is
  observed behaviour named in the neon module's inline comments,
  and the ``role`` field surfaces the mic family each candidate is
  known-good for as a label rather than a resolver answer; this
  file does not read PyAudio or ALSA to check what a specific USB
  mic accepts.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_capture_rate_candidates import (
    _CAPTURE_RATE_CANDIDATES,
    _CAPTURE_RATE_FALLBACK_HZ,
    _CAPTURE_RATE_ROLES,
    _INVALID_CAPTURE_RATE_CODE,
    g1_capture_rate_candidate_admits,
    g1_list_capture_rate_candidates,
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
    runner and Thor before an office bring-up. The candidate snapshot
    is an integer tuple; no SDK submodule should load on the import
    path.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_capture_rate_candidates")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_capture_rate_candidates imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_audio_stack_module() -> None:
    """The tool module is loadable without the optional bidi audio deps.

    The neon bundle's ``pick_capture_rate`` runs against
    ``PyAudio.is_format_supported`` at runtime; ``pyaudio`` is an
    optional dependency the ``strands-robots`` package does not
    require. A caller who only wants to read the candidate list
    must not be forced to install the audio stack; a module that
    pulled ``pyaudio`` on import would refuse on a headless host the
    ``strands-robots`` package must run on.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_capture_rate_candidates")
    after = set(sys.modules)
    audio_dep_prefixes = ("pywebrtc_audio", "pyaudio", "strands.experimental.bidi")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_dep_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_capture_rate_candidates imports pulled "
        f"audio-stack submodules: {leaked}. The candidate snapshot is an "
        "integer tuple; the audio stack loads on the write path, not on "
        "tool import."
    )


def test_the_snapshot_is_an_immutable_tuple() -> None:
    """The snapshot is a ``tuple`` so a caller cannot mutate it in place.

    The neon source writes ``CAPTURE_RATE_CANDIDATES`` as a bare
    ``list``, but a list on the snapshot side would let a caller
    reading :data:`_CAPTURE_RATE_CANDIDATES` mutate the module state
    and drift the lookup out of sync with the neon source. The
    tuple type is a defensive contract the tests fix.
    """
    assert isinstance(_CAPTURE_RATE_CANDIDATES, tuple), (
        f"_CAPTURE_RATE_CANDIDATES is not a tuple: {type(_CAPTURE_RATE_CANDIDATES).__name__}. "
        "A mutable snapshot would let a caller drift the lookup out of sync with the neon source."
    )


def test_the_snapshot_covers_the_neon_observed_order() -> None:
    """The snapshot names the four rates in the neon iteration order.

    The neon source writes ``CAPTURE_RATE_CANDIDATES = [16000, 48000,
    44100, 32000]``; the ``pick_capture_rate`` helper iterates the
    list in that order and stops at the first accepted rate. The
    ordering is part of the contract this snapshot pins - a Brio
    lands on the first attempt (16 kHz), a DJI Mic Mini on the
    second (48 kHz). Pinning the exact tuple surfaces a silent
    reorder as a shape change rather than as a quiet re-labeling
    of one entry.
    """
    assert _CAPTURE_RATE_CANDIDATES == (16000, 48000, 44100, 32000), (
        f"capture-rate candidate order drifted from the neon source: {_CAPTURE_RATE_CANDIDATES}. "
        "The neon ``pick_capture_rate`` stops at the first accepted rate, so the ordering "
        "controls which rate a given USB mic lands on; a reorder is a semantic change."
    )


def test_the_snapshot_covers_four_candidates() -> None:
    """The snapshot names exactly four rates.

    A widen or narrow to the candidate list is a neon-side change
    that must be reflected here; pinning the count surfaces a silent
    drift as a shape change.
    """
    assert len(_CAPTURE_RATE_CANDIDATES) == 4, (
        f"expected 4 admitted candidates, got {len(_CAPTURE_RATE_CANDIDATES)}: {_CAPTURE_RATE_CANDIDATES}"
    )


def test_the_fallback_rate_is_inside_the_candidate_set() -> None:
    """The fallback rate is one of the admitted candidates.

    The neon ``pick_capture_rate`` returns ``48000`` when every
    candidate is refused; that fallback is only coherent if the
    fallback rate is itself a member of the iteration list -
    otherwise the helper could return a rate the probe never
    accepted, and a caller reading the payload would see a
    ``fallback_rate_hz`` that is not in ``ordered_rates_hz``.
    """
    assert _CAPTURE_RATE_FALLBACK_HZ in _CAPTURE_RATE_CANDIDATES, (
        f"fallback rate {_CAPTURE_RATE_FALLBACK_HZ} is not in the candidate set {_CAPTURE_RATE_CANDIDATES}"
    )


def test_every_candidate_carries_a_role() -> None:
    """Every admitted rate names a non-empty role label.

    The role is what the caller reads to classify the rate
    (``brio_native`` vs ``usb_high_rate`` vs ``cd_rate`` vs
    ``usb_low_rate``); an empty role would leave the caller reading
    a bare integer without context.
    """
    for rate in _CAPTURE_RATE_CANDIDATES:
        assert rate in _CAPTURE_RATE_ROLES, (
            f"rate {rate} has no role entry in _CAPTURE_RATE_ROLES; every "
            "admitted rate must name what USB mic family it targets"
        )
        entry = _CAPTURE_RATE_ROLES[rate]
        assert "role" in entry, f"rate {rate} has no role field; every admitted rate must name what it contributes"
        assert isinstance(entry["role"], str) and entry["role"], f"rate {rate} has an empty role: {entry['role']!r}"


def test_the_four_roles_are_distinct() -> None:
    """Each admitted rate plays a distinct role in the mic-open path.

    The neon path treats each candidate as a distinct USB mic
    family (Brio native, DJI high-rate USB, CD-rate USB, low-rate
    USB); a silent collapse of two roles onto one label would let
    a widen to a fifth rate land without the caller reading a
    distinct classification.
    """
    roles = [_CAPTURE_RATE_ROLES[rate]["role"] for rate in _CAPTURE_RATE_CANDIDATES]
    assert len(set(roles)) == len(roles), (
        f"candidate roles are not pairwise distinct: {roles}. Every admitted candidate must name a distinct mic family."
    )


def test_every_candidate_carries_a_description() -> None:
    """Every admitted rate carries a non-empty description.

    The description is what the caller reads to understand why the
    neon path reaches for the rate; an empty description would leave
    the caller reading a bare enum without context.
    """
    for rate in _CAPTURE_RATE_CANDIDATES:
        entry = _CAPTURE_RATE_ROLES[rate]
        assert "description" in entry, (
            f"rate {rate} has no description; every admitted rate must carry a caller-facing label"
        )
        assert isinstance(entry["description"], str) and entry["description"], f"rate {rate} has an empty description"


def test_the_refusal_code_matches_the_shared_gate_refusal() -> None:
    """The refusal code sits inside the shared error table.

    The neon bundle's ``pick_capture_rate`` does not refuse unknown
    rates (it silently returns the fallback); this lookup uses the
    ``7404`` gate-refusal shape a future driver-side wrapper would
    quote when refusing at the same boundary. The code must decode
    against :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so
    a caller reading ``refusal_text`` sees the same string the
    driver's own ``_check_motion_gates`` would quote.
    """
    assert _INVALID_CAPTURE_RATE_CODE in ERR_CODES, (
        f"_INVALID_CAPTURE_RATE_CODE={_INVALID_CAPTURE_RATE_CODE} is not in "
        f"ERR_CODES; the refusal string would not decode. Update "
        "``_g1_common.ERR_CODES`` or point the constant at a "
        "registered code."
    )


def test_list_returns_every_rate_in_neon_iteration_order() -> None:
    """The list verb surfaces the rates in the neon iteration order.

    The ordering carries meaning: the neon helper stops at the first
    accepted rate, so the payload's iteration order names which rate
    a given USB mic family lands on. A silent reorder in the verb
    would let a caller iterating the payload see a different rate
    on the first attempt than the neon path itself would try.
    """
    payload = _call(g1_list_capture_rate_candidates)
    assert payload["ordered_rates_hz"] == list(_CAPTURE_RATE_CANDIDATES), (
        f"list verb returned rates in a different order than the neon source: "
        f"{payload['ordered_rates_hz']} vs {list(_CAPTURE_RATE_CANDIDATES)}"
    )
    candidate_rates = [descriptor["rate_hz"] for descriptor in payload["candidates"]]
    assert candidate_rates == list(_CAPTURE_RATE_CANDIDATES), (
        f"list verb candidate descriptors returned in wrong order: {candidate_rates}"
    )


def test_list_names_every_snapshot_rate_and_no_others() -> None:
    """The list verb round-trips the snapshot with no drift.

    A silent divergence between the snapshot tuple and the verb's
    surface would let a widen on one side land without the other;
    the test round-trips the two sets to fix that contract.
    """
    payload = _call(g1_list_capture_rate_candidates)
    listed = {descriptor["rate_hz"] for descriptor in payload["candidates"]}
    assert listed == set(_CAPTURE_RATE_CANDIDATES), (
        f"list verb surface {listed} drifted from snapshot {set(_CAPTURE_RATE_CANDIDATES)}"
    )
    assert set(payload["ordered_rates_hz"]) == set(_CAPTURE_RATE_CANDIDATES), (
        f"list verb ordered_rates_hz field {set(payload['ordered_rates_hz'])} "
        f"drifted from snapshot {set(_CAPTURE_RATE_CANDIDATES)}"
    )


def test_list_surfaces_the_fallback_rate() -> None:
    """The envelope carries the fallback rate the neon helper returns.

    The ``fallback_rate_hz`` field names the rate the neon
    ``pick_capture_rate`` returns when every candidate is refused;
    a caller who does not want to walk the ordered list needs the
    fallback surfaced separately from the iteration list.
    """
    payload = _call(g1_list_capture_rate_candidates)
    assert payload["fallback_rate_hz"] == _CAPTURE_RATE_FALLBACK_HZ, (
        f"list verb fallback_rate_hz {payload['fallback_rate_hz']} drifted from snapshot {_CAPTURE_RATE_FALLBACK_HZ}"
    )


def test_list_surfaces_the_pipeline_native_and_fallback_flags() -> None:
    """The descriptors name the 16 kHz pipeline-native and the fallback rate.

    ``is_pipeline_native`` is ``True`` only for 16 kHz (the internal
    AEC and bidi-pipeline rate); ``is_fallback`` is ``True`` only
    for the fallback rate (48 kHz today). A caller iterating the
    payload for downstream branching (does the neon path
    downsample, does this rate land on the fallback) reads these
    flags rather than re-deriving them.
    """
    payload = _call(g1_list_capture_rate_candidates)
    for descriptor in payload["candidates"]:
        rate = descriptor["rate_hz"]
        assert descriptor["is_pipeline_native"] is (rate == 16000), (
            f"is_pipeline_native for {rate} Hz drifted: {descriptor['is_pipeline_native']}"
        )
        assert descriptor["is_fallback"] is (rate == _CAPTURE_RATE_FALLBACK_HZ), (
            f"is_fallback for {rate} Hz drifted: {descriptor['is_fallback']}"
        )


def test_list_surfaces_the_refusal_code_and_text() -> None:
    """The envelope carries the ``7404`` refusal code and decoded text.

    A caller planning a ``g1_speak`` call reads the ``refusals``
    list to see the string a future driver-side wrapper would surface
    on an unknown rate; the string must decode against the shared
    error table.
    """
    payload = _call(g1_list_capture_rate_candidates)
    codes = {refusal["code"]: refusal["text"] for refusal in payload["refusals"]}
    assert _INVALID_CAPTURE_RATE_CODE in codes, (
        f"envelope refusals list does not carry the {_INVALID_CAPTURE_RATE_CODE} code: {payload['refusals']}"
    )
    assert codes[_INVALID_CAPTURE_RATE_CODE] == ERR_CODES[_INVALID_CAPTURE_RATE_CODE], (
        f"envelope refusal text drifted from ERR_CODES: "
        f"{codes[_INVALID_CAPTURE_RATE_CODE]!r} vs "
        f"{ERR_CODES[_INVALID_CAPTURE_RATE_CODE]!r}"
    )


def test_list_surfaces_the_count_matching_the_tuple_length() -> None:
    """The envelope's ``count`` field names the tuple length.

    A caller reading ``count`` must not have to compute it from
    the ``ordered_rates_hz`` list; pinning it here surfaces a
    silent drift as a shape change.
    """
    payload = _call(g1_list_capture_rate_candidates)
    assert payload["count"] == len(_CAPTURE_RATE_CANDIDATES), (
        f"list verb count {payload['count']} drifted from tuple length {len(_CAPTURE_RATE_CANDIDATES)}"
    )


def test_admits_returns_true_on_every_snapshot_rate() -> None:
    """The admits verb round-trips every admitted rate.

    Every entry in the snapshot must be admitted by the verb; a
    divergence would let a widen on the snapshot side land without
    the verb agreeing.
    """
    for rate in _CAPTURE_RATE_CANDIDATES:
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=rate)
        assert payload["status"] == "success", f"admits verb refused a snapshot rate {rate}: {payload}"
        assert payload["candidate"]["rate_hz"] == rate, (
            f"admits verb returned a different rate than requested: asked {rate}, got {payload['candidate']['rate_hz']}"
        )
        assert payload["candidate"]["role"] == _CAPTURE_RATE_ROLES[rate]["role"], (
            f"admits verb role drifted from snapshot for {rate}: "
            f"verb={payload['candidate']['role']!r} vs "
            f"snapshot={_CAPTURE_RATE_ROLES[rate]['role']!r}"
        )


def test_admits_reports_the_pipeline_native_flag_for_16k_only() -> None:
    """Only the 16 kHz rate carries ``is_pipeline_native=True``.

    The internal AEC and bidi pipeline run at 16 kHz; any other
    rate is downsampled by the neon path before AEC. A caller
    reading the flag must see it True on 16 kHz and False on the
    other three.
    """
    for rate in _CAPTURE_RATE_CANDIDATES:
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=rate)
        assert payload["candidate"]["is_pipeline_native"] is (rate == 16000), (
            f"admits verb is_pipeline_native for {rate} Hz drifted: {payload['candidate']['is_pipeline_native']}"
        )


def test_admits_reports_the_fallback_flag_for_48k_only() -> None:
    """Only the 48 kHz rate carries ``is_fallback=True``.

    The neon ``pick_capture_rate`` returns 48 kHz when every
    candidate is refused; the flag names that entry on the admit
    side so a caller reading the descriptor knows which rate the
    neon helper falls back to.
    """
    for rate in _CAPTURE_RATE_CANDIDATES:
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=rate)
        assert payload["candidate"]["is_fallback"] is (rate == _CAPTURE_RATE_FALLBACK_HZ), (
            f"admits verb is_fallback for {rate} Hz drifted: {payload['candidate']['is_fallback']}"
        )


def test_admits_refuses_an_off_set_rate_with_the_shared_code() -> None:
    """An off-set rate refuses with the ``7404`` code and decoded text.

    A caller passing a rate outside the snapshot must see the same
    refusal shape a future driver-side wrapper would surface; the
    reason string names the admitted set so the caller can correct
    the argument. 22050 is the canonical off-set rate (a common
    consumer rate the neon path does not include).
    """
    payload = _call(g1_capture_rate_candidate_admits, rate_hz=22050)
    assert payload["status"] == "error", f"admits verb admitted an off-set rate: {payload}"
    assert payload["refusal_code"] == _INVALID_CAPTURE_RATE_CODE, (
        f"admits verb refusal_code drifted: {payload['refusal_code']} vs {_INVALID_CAPTURE_RATE_CODE}"
    )
    assert payload["refusal_text"] == ERR_CODES[_INVALID_CAPTURE_RATE_CODE], (
        f"admits verb refusal_text drifted: {payload['refusal_text']!r} vs {ERR_CODES[_INVALID_CAPTURE_RATE_CODE]!r}"
    )
    assert "22050" in payload["reason"], f"admits verb reason string does not quote the argument: {payload['reason']!r}"


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``True`` / ``False`` argument refuses decidably.

    Python's ``bool`` is a subclass of ``int`` and ``True == 1`` /
    ``False == 0`` would otherwise silently mis-match against the
    integer snapshot (neither 0 nor 1 is a candidate, so the
    downstream refusal would name a confusing "unknown rate"
    reason). The verb rejects the bool up front so the caller sees
    a shape error.
    """
    for value in (True, False):
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=value)
        assert payload["status"] == "error", f"admits verb admitted bool argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_CAPTURE_RATE_CODE, (
            f"admits verb refusal_code for bool {value!r} drifted: {payload['refusal_code']}"
        )
        assert "bool" in payload["reason"], (
            f"admits verb reason for bool {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_int_argument_as_a_shape_error() -> None:
    """A non-integer non-bool argument refuses decidably.

    Floats, strings, lists, dicts, tuples: none of them are the
    integer literals the neon source writes the list with; the verb
    rejects them up front rather than reaching the membership branch
    where ``in`` would refuse for the wrong reason.
    """
    for value in (16000.0, "16000", [16000], {16000}, (16000,), object()):
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=value)
        assert payload["status"] == "error", f"admits verb admitted non-int argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_CAPTURE_RATE_CODE, (
            f"admits verb refusal_code for {value!r} drifted: {payload['refusal_code']}"
        )
        assert "not an int" in payload["reason"], (
            f"admits verb reason for {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_positive_rate_as_a_shape_error() -> None:
    """A zero or negative rate refuses decidably, not as an off-set query.

    A caller passing ``rate_hz=0`` or a negative rate almost
    certainly has an upstream computation bug; the verb rejects it
    with a shape reason so the error names the sign mistake rather
    than falsely claiming the value is an unknown rate.
    """
    for value in (0, -1, -16000):
        payload = _call(g1_capture_rate_candidate_admits, rate_hz=value)
        assert payload["status"] == "error", f"admits verb admitted non-positive rate {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_CAPTURE_RATE_CODE, (
            f"admits verb refusal_code for {value!r} drifted: {payload['refusal_code']}"
        )
        assert "positive" in payload["reason"], (
            f"admits verb reason for {value!r} does not name the sign shape error: {payload['reason']!r}"
        )


def test_admits_refuses_the_missing_argument_as_a_shape_error() -> None:
    """A ``None`` (default) ``rate_hz`` refuses with a missing-argument reason.

    A caller who invokes the verb without passing ``rate_hz`` sees
    the Python default of ``None``; the verb rejects that up front
    with a "rate_hz is required" reason so the caller sees the
    missing argument named, not a downstream membership refusal.
    """
    payload = _call(g1_capture_rate_candidate_admits)
    assert payload["status"] == "error", f"admits verb admitted a missing rate_hz argument: {payload}"
    assert payload["refusal_code"] == _INVALID_CAPTURE_RATE_CODE, (
        f"admits verb refusal_code for missing rate_hz drifted: {payload['refusal_code']}"
    )
    assert "required" in payload["reason"], (
        f"admits verb reason for missing rate_hz does not name the required argument: {payload['reason']!r}"
    )
