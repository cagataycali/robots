"""The bidi-audio dependency lookup tools name what ``_probe_bidi`` reads.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) runs a ``_probe_bidi``
guard before spawning its bidi-audio thread: it ``import``s three
optional modules - ``pywebrtc_audio`` (AEC front-end), ``pyaudio``
(PortAudio mic capture), and ``strands.experimental.bidi.BidiAgent``
(the bidi agent factory) - and refuses on ``ImportError``. The
:mod:`strands_robots.tools.g1.g1_bidi_audio_dependencies` module
snapshots that dependency set into a module-level dict and exposes
two agent-facing verbs -
:func:`g1_list_bidi_audio_dependencies` (name the whole set) and
:func:`g1_bidi_audio_dependency_admits` (decide one query) - so a
caller can name the module set decidably before a future driver-side
wrapper for ``g1_speak`` is attempted. The tests here fix that
contract without pulling the SDK or the audio stack: the module is
loadable on a host without ``unitree_sdk2py`` *and* without the
optional bidi audio deps, so a headless CI runner and Thor before an
office bring-up can read the dependency set without triggering an
import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime probe. The neon bundle's ``_probe_bidi`` is a live
  ``ImportError``-shaped check for ``pywebrtc_audio`` + ``pyaudio``
  + ``strands.experimental.bidi.BidiAgent``; a caller comparing an
  intended write against both conditions (membership + audio
  stack present) reaches the probe after this verb admits the
  dependency name. This file does not exercise the probe.
* The pip package a module ships with. The ``pip_hint`` field
  surfaces a suggested distribution name where the module name and
  the distribution name coincide, but this file does not read PyPI
  or the local ``pip`` index; the hint is a starting-point label,
  not a canonical resolver answer.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_bidi_audio_dependencies import (
    _BIDI_AUDIO_DEPENDENCIES,
    _INVALID_DEPENDENCY_CODE,
    g1_bidi_audio_dependency_admits,
    g1_list_bidi_audio_dependencies,
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
    runner and Thor before an office bring-up. The dependency snapshot
    is a string table; no SDK submodule should load on the import path.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_dependencies")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_dependencies imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_audio_stack_module() -> None:
    """The tool module is loadable without the optional bidi audio deps.

    The neon bundle's ``_probe_bidi`` check reaches for
    ``pywebrtc_audio`` + ``pyaudio`` +
    ``strands.experimental.bidi.BidiAgent`` at runtime; those are
    optional dependencies the ``strands-robots`` package does not
    require. A caller who only wants to read the dependency set must
    not be forced to install the audio stack; a module that pulled
    any of those on import would refuse on a headless host the
    ``strands-robots`` package must run on.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_dependencies")
    after = set(sys.modules)
    audio_dep_prefixes = ("pywebrtc_audio", "pyaudio", "strands.experimental.bidi")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_dep_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_dependencies imports pulled "
        f"audio-stack submodules: {leaked}. The dependency snapshot is a "
        "string table; the audio stack loads on the write path, not on "
        "tool import."
    )


def test_the_snapshot_covers_the_neon_observed_set() -> None:
    """The snapshot names the three modules the neon probe imports.

    The neon bundle's ``_probe_bidi`` guard imports ``pywebrtc_audio``,
    ``pyaudio``, and ``strands.experimental.bidi.BidiAgent``; a widen
    or narrow to the observed set is a driver-side decision. Pinning
    the count here surfaces a silent drift as a shape change rather
    than as a quiet re-labeling of an existing dependency.
    """
    assert len(_BIDI_AUDIO_DEPENDENCIES) == 3, (
        f"expected 3 admitted dependencies, got {len(_BIDI_AUDIO_DEPENDENCIES)}: {sorted(_BIDI_AUDIO_DEPENDENCIES)}"
    )
    assert set(_BIDI_AUDIO_DEPENDENCIES) == {
        "pywebrtc_audio",
        "pyaudio",
        "strands.experimental.bidi",
    }, f"dependency snapshot drifted from the neon-observed set: {sorted(_BIDI_AUDIO_DEPENDENCIES)}"


def test_every_snapshot_entry_carries_a_role() -> None:
    """Every admitted dependency names a non-empty role label.

    The role is what the caller reads to classify the dependency
    (``aec_frontend`` vs ``mic_capture`` vs ``bidi_agent``); an
    empty role would leave the caller reading a bare module name
    without context.
    """
    for name, entry in _BIDI_AUDIO_DEPENDENCIES.items():
        assert "role" in entry, (
            f"dependency {name!r} has no role; every admitted dependency "
            "must name what it contributes to the bidi audio path"
        )
        assert isinstance(entry["role"], str) and entry["role"], (
            f"dependency {name!r} has an empty role: {entry['role']!r}"
        )


def test_the_three_roles_are_distinct() -> None:
    """Each admitted dependency plays a distinct role in the path.

    The neon bundle reaches for the three modules for three
    different reasons (AEC front-end, mic capture, bidi factory); a
    silent collapse of two roles onto one label would let a widen
    to a fourth dependency land without the caller reading a
    distinct role classification. The three roles must be pairwise
    distinct.
    """
    roles = [entry["role"] for entry in _BIDI_AUDIO_DEPENDENCIES.values()]
    assert len(set(roles)) == len(roles), (
        f"dependency roles are not pairwise distinct: {roles}. Every "
        "admitted dependency must name a distinct contribution to the "
        "bidi audio path."
    )


def test_every_snapshot_entry_carries_a_description() -> None:
    """Every admitted dependency carries a non-empty description.

    The description is what the caller reads to understand why the
    neon path reaches for the module; an empty description would
    leave the caller reading a bare enum without context.
    """
    for name, entry in _BIDI_AUDIO_DEPENDENCIES.items():
        assert "description" in entry, (
            f"dependency {name!r} has no description; every admitted dependency must carry a caller-facing label"
        )
        assert isinstance(entry["description"], str) and entry["description"], (
            f"dependency {name!r} has an empty description"
        )


def test_every_snapshot_entry_carries_a_pip_hint() -> None:
    """Every admitted dependency names a non-empty pip hint.

    The ``pip_hint`` is a suggested distribution name the caller
    would ``pip install`` to satisfy the probe; an empty hint would
    force the caller to know the distribution name out of band.
    """
    for name, entry in _BIDI_AUDIO_DEPENDENCIES.items():
        assert "pip_hint" in entry, (
            f"dependency {name!r} has no pip_hint; every admitted "
            "dependency must name a distribution a caller would install"
        )
        assert isinstance(entry["pip_hint"], str) and entry["pip_hint"], (
            f"dependency {name!r} has an empty pip_hint: {entry['pip_hint']!r}"
        )


def test_the_refusal_code_matches_the_shared_gate_refusal() -> None:
    """The refusal code sits inside the shared error table.

    The neon bundle refused unknown dependencies at the probe
    boundary with an ``ImportError``; this lookup uses the ``7404``
    gate-refusal shape a future driver-side wrapper would quote when
    refusing at the same boundary. The code must decode against
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so a caller
    reading ``refusal_text`` sees the same string the driver's own
    ``_check_motion_gates`` would quote.
    """
    assert _INVALID_DEPENDENCY_CODE in ERR_CODES, (
        f"_INVALID_DEPENDENCY_CODE={_INVALID_DEPENDENCY_CODE} is not in "
        f"ERR_CODES; the refusal string would not decode. Update "
        "``_g1_common.ERR_CODES`` or point the constant at a "
        "registered code."
    )


def test_list_returns_every_dependency_in_sorted_order() -> None:
    """The list verb surfaces the admitted names in stable order.

    A caller iterating the ``dependencies`` list must see the same
    order across calls so a diff against the returned payload does
    not fluctuate with dict-iteration order under a hostile Python
    build; the verb sorts by name ascending.
    """
    payload = _call(g1_list_bidi_audio_dependencies)
    names = [descriptor["name"] for descriptor in payload["dependencies"]]
    assert names == sorted(_BIDI_AUDIO_DEPENDENCIES), (
        f"list verb returned dependencies in unsorted order: {names}. "
        f"Expected sorted: {sorted(_BIDI_AUDIO_DEPENDENCIES)}"
    )


def test_list_names_every_snapshot_dependency_and_no_others() -> None:
    """The list verb round-trips the snapshot with no drift.

    A silent divergence between the snapshot and the list verb's
    surface would let a widen on one side land without the other;
    the test round-trips the two sets to fix that contract.
    """
    payload = _call(g1_list_bidi_audio_dependencies)
    listed = {descriptor["name"] for descriptor in payload["dependencies"]}
    assert listed == set(_BIDI_AUDIO_DEPENDENCIES), (
        f"list verb surface {listed} drifted from snapshot {set(_BIDI_AUDIO_DEPENDENCIES)}"
    )
    assert set(payload["names"]) == set(_BIDI_AUDIO_DEPENDENCIES), (
        f"list verb names field {set(payload['names'])} drifted from snapshot {set(_BIDI_AUDIO_DEPENDENCIES)}"
    )


def test_list_surfaces_every_descriptor_with_admits_flag_true() -> None:
    """Every listed descriptor names ``admits_bidi_writes=True``.

    The flag is surfaced so the descriptor shape matches
    :mod:`~strands_robots.tools.g1.g1_voice_providers` and
    :mod:`~strands_robots.tools.g1.g1_arm_actions` verbatim; every
    admitted dependency is inside the neon probe set by definition,
    so the flag is always ``True`` on the list side. A caller
    reading the payload for shape parity across verbs must see the
    same flag on every row.
    """
    payload = _call(g1_list_bidi_audio_dependencies)
    for descriptor in payload["dependencies"]:
        assert descriptor.get("admits_bidi_writes") is True, (
            f"listed descriptor for {descriptor.get('name')!r} does not carry admits_bidi_writes=True: {descriptor}"
        )


def test_list_surfaces_the_refusal_code_and_text() -> None:
    """The envelope carries the ``7404`` refusal code and decoded text.

    A caller planning a ``g1_speak`` call reads the ``refusals``
    list to see the string a future driver-side wrapper would surface
    on an unknown dependency; the string must decode against the
    shared error table.
    """
    payload = _call(g1_list_bidi_audio_dependencies)
    codes = {refusal["code"]: refusal["text"] for refusal in payload["refusals"]}
    assert _INVALID_DEPENDENCY_CODE in codes, (
        f"envelope refusals list does not carry the {_INVALID_DEPENDENCY_CODE} code: {payload['refusals']}"
    )
    assert codes[_INVALID_DEPENDENCY_CODE] == ERR_CODES[_INVALID_DEPENDENCY_CODE], (
        f"envelope refusal text drifted from ERR_CODES: "
        f"{codes[_INVALID_DEPENDENCY_CODE]!r} vs "
        f"{ERR_CODES[_INVALID_DEPENDENCY_CODE]!r}"
    )


def test_admits_returns_true_on_every_snapshot_dependency() -> None:
    """The admits verb round-trips every admitted name.

    Every entry in the snapshot must be admitted by the verb; a
    divergence would let a widen on the snapshot side land without
    the verb agreeing.
    """
    for name in _BIDI_AUDIO_DEPENDENCIES:
        payload = _call(g1_bidi_audio_dependency_admits, name=name)
        assert payload["status"] == "success", f"admits verb refused a snapshot dependency {name!r}: {payload}"
        assert payload["dependency"]["name"] == name, (
            f"admits verb returned a different dependency than requested: "
            f"asked {name!r}, got {payload['dependency']['name']!r}"
        )
        assert payload["dependency"]["role"] == _BIDI_AUDIO_DEPENDENCIES[name]["role"], (
            f"admits verb role drifted from snapshot for {name!r}: "
            f"verb={payload['dependency']['role']!r} vs "
            f"snapshot={_BIDI_AUDIO_DEPENDENCIES[name]['role']!r}"
        )
        assert payload["dependency"]["pip_hint"] == _BIDI_AUDIO_DEPENDENCIES[name]["pip_hint"], (
            f"admits verb pip_hint drifted from snapshot for {name!r}: "
            f"verb={payload['dependency']['pip_hint']!r} vs "
            f"snapshot={_BIDI_AUDIO_DEPENDENCIES[name]['pip_hint']!r}"
        )


def test_admits_refuses_an_off_set_dependency_with_the_shared_code() -> None:
    """An off-set name refuses with the ``7404`` code and decoded text.

    A caller passing a dependency name outside the snapshot must see
    the same refusal shape a future driver-side wrapper would
    surface; the reason string names the admitted set so the caller
    can correct the argument.
    """
    payload = _call(g1_bidi_audio_dependency_admits, name="alsa_audio")
    assert payload["status"] == "error", f"admits verb admitted an off-set dependency: {payload}"
    assert payload["refusal_code"] == _INVALID_DEPENDENCY_CODE, (
        f"admits verb refusal_code drifted: {payload['refusal_code']} vs {_INVALID_DEPENDENCY_CODE}"
    )
    assert payload["refusal_text"] == ERR_CODES[_INVALID_DEPENDENCY_CODE], (
        f"admits verb refusal_text drifted: {payload['refusal_text']!r} vs {ERR_CODES[_INVALID_DEPENDENCY_CODE]!r}"
    )
    assert "alsa_audio" in payload["reason"], (
        f"admits verb reason string does not quote the argument: {payload['reason']!r}"
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """A ``True`` / ``False`` argument refuses decidably.

    Python's ``bool`` would silently mis-match against the string
    snapshot; the verb rejects it up front so the caller sees a
    shape error rather than a confusing "unknown dependency" refusal.
    """
    for value in (True, False):
        payload = _call(g1_bidi_audio_dependency_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted bool argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_DEPENDENCY_CODE, (
            f"admits verb refusal_code for bool {value!r} drifted: {payload['refusal_code']}"
        )
        assert "bool" in payload["reason"], (
            f"admits verb reason for bool {value!r} does not name the shape error: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """A non-string non-bool argument refuses decidably.

    Ints, floats, lists, dicts, tuples: none of them are module
    names; the verb rejects them up front rather than reaching the
    membership branch where ``in`` would refuse for the wrong reason
    (or worse, silently match on a coincidental repr).
    """
    for value in (0, 1, 1.5, [], {}, (), object()):
        payload = _call(g1_bidi_audio_dependency_admits, name=value)
        assert payload["status"] == "error", f"admits verb admitted non-str argument {value!r}: {payload}"
        assert payload["refusal_code"] == _INVALID_DEPENDENCY_CODE, (
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
    empty string is an unknown module.
    """
    payload = _call(g1_bidi_audio_dependency_admits, name="")
    assert payload["status"] == "error", f"admits verb admitted empty string: {payload}"
    assert payload["refusal_code"] == _INVALID_DEPENDENCY_CODE, (
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
    payload = _call(g1_bidi_audio_dependency_admits)
    assert payload["status"] == "error", f"admits verb admitted a missing name argument: {payload}"
    assert payload["refusal_code"] == _INVALID_DEPENDENCY_CODE, (
        f"admits verb refusal_code for missing name drifted: {payload['refusal_code']}"
    )
    assert "required" in payload["reason"], (
        f"admits verb reason for missing name does not name the required argument: {payload['reason']!r}"
    )
