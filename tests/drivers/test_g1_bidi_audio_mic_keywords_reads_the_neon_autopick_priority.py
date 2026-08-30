"""The mic-keyword lookup tools name what ``autopick_mic`` walks by default.

The neon bundle's ``autopick_mic`` helper
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) walks a fixed
priority order of substring keywords - ``DJI``, ``Logi``, ``Brio``,
``USB``, ``Mic`` - when the ``VOICE_MIC_NAME`` env-var is unset, and
returns the first PyAudio input-capable device whose lowercased name
contains the lowercased keyword. The
:mod:`strands_robots.tools.g1.g1_bidi_audio_mic_keywords` module
snapshots that priority order into a module-level tuple and exposes
two agent-facing verbs -
:func:`g1_list_bidi_audio_mic_keywords` (name the whole list) and
:func:`g1_bidi_audio_mic_keyword_admits` (decide one query) - so a
caller can decide whether their device matches the neon-observed set
before a future driver-side bring-up runs the PyAudio walk. The
tests here fix that contract without pulling the SDK or the optional
audio dependencies (``pyaudio``, ``pywebrtc_audio``,
``strands.experimental.bidi``): the module is a plain string table
whose import path stays clean on a host missing every one of those
extras, so a headless CI runner and Thor before an office bring-up
can read the priority order without triggering an import-time
refusal.

Two things this file's cells deliberately do not pin:

* The live PyAudio probe. The bundle's ``autopick_mic`` opens a
  PyAudio session, walks ``get_device_count()``, and reads
  ``get_device_info_by_index(i)`` for each entry - a live hardware
  read this file cannot answer without the optional ``pyaudio``
  dependency present. A future driver-side bring-up verb will land
  the wire read; this file exercises only the read-only keyword
  membership decision that precedes it.
* The ``VOICE_MIC_NAME`` override. The bundle's ``_mic_keywords``
  accessor reads the env-var at call time; when set, the override
  replaces the whole default list. The snapshot names the env-var
  in the payload so a caller can wire the override themselves, but
  this file does not read the environment. A caller who set the
  override reaches a different admitted set than the one this
  snapshot carries; the module's contract is to name the *default*
  list.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_bidi_audio_mic_keywords import (
    _DEFAULT_MIC_KEYWORDS,
    _OVERRIDE_ENV_VAR,
    g1_bidi_audio_mic_keyword_admits,
    g1_list_bidi_audio_mic_keywords,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


# ---------------------------------------------------------------------- #
# Import hygiene                                                         #
# ---------------------------------------------------------------------- #


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up. The keyword snapshot
    is a string tuple; no SDK submodule should load on the import
    path.
    """
    before = set(sys.modules)
    importlib.import_module(
        "strands_robots.tools.g1.g1_bidi_audio_mic_keywords",
    )
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_mic_keywords imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_optional_audio_module() -> None:
    """The tool module is loadable without ``pyaudio`` / ``pywebrtc_audio``.

    The neon bundle's ``g1_bidi_audio`` runtime imports optional deps
    (``pyaudio`` for mic capture, ``pywebrtc_audio`` for the AEC
    front-end, ``strands.experimental.bidi`` for the audio agent);
    those are absent on every headless CI runner. This snapshot is a
    plain string table and must stay importable when none of them are
    installed. A module that pulled any of them at import time would
    break the same hosts this module exists to serve.
    """
    before = set(sys.modules)
    importlib.import_module(
        "strands_robots.tools.g1.g1_bidi_audio_mic_keywords",
    )
    after = set(sys.modules)
    optional = ("pyaudio", "pywebrtc_audio", "strands.experimental.bidi")
    leaked = {name for name in after - before if any(name == m or name.startswith(m + ".") for m in optional)}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_mic_keywords imports pulled "
        f"optional audio-stack submodules: {leaked}. The snapshot must load "
        "on hosts missing those extras (refs strands-labs/robots#358)."
    )


# ---------------------------------------------------------------------- #
# Snapshot fidelity vs. the neon bundle                                  #
# ---------------------------------------------------------------------- #


def test_the_default_snapshot_names_the_neon_observed_priority_order() -> None:
    """The tuple matches the bundle's ``DEFAULT_MIC_KEYWORDS`` literal.

    The neon bundle's ``DEFAULT_MIC_KEYWORDS = ["DJI", "Logi", "Brio",
    "USB", "Mic"]`` is the priority order the field observed on
    Jetson Orin office rigs: three specific USB-mic families
    (``DJI``, ``Logi``, ``Brio``) followed by two generic fallbacks
    (``USB``, ``Mic``). A driver-side widen (a new mic family the
    field observed) or narrow (a family the bundle stopped
    supporting) that does not update this snapshot leaves the two
    out of sync; this test pins the exact literal so the drift is
    audible.
    """
    assert _DEFAULT_MIC_KEYWORDS == ("DJI", "Logi", "Brio", "USB", "Mic"), (
        f"_DEFAULT_MIC_KEYWORDS drifted from the neon bundle's "
        f"DEFAULT_MIC_KEYWORDS. Snapshot: {_DEFAULT_MIC_KEYWORDS}. Expected: "
        "('DJI', 'Logi', 'Brio', 'USB', 'Mic'). "
        "See cagataycali/neon-the-g1/tools/g1_bidi_audio.py."
    )


def test_the_snapshot_names_the_neon_observed_override_env() -> None:
    """The env-var name matches the bundle's ``_mic_keywords`` accessor.

    The bundle's ``_mic_keywords`` reads ``VOICE_MIC_NAME`` and returns
    the override verbatim when set. Naming a different env-var here
    would silently make a caller's override useless: the driver-side
    write path would read one variable and the caller-facing lookup
    would advertise a different one. This test pins the exact string.
    """
    assert _OVERRIDE_ENV_VAR == "VOICE_MIC_NAME", (
        f"_OVERRIDE_ENV_VAR drifted from the neon bundle's env-var. Snapshot: "
        f"{_OVERRIDE_ENV_VAR!r}. Expected: 'VOICE_MIC_NAME'. "
        "See cagataycali/neon-the-g1/tools/g1_bidi_audio.py."
    )


def test_the_snapshot_is_a_tuple_not_a_list() -> None:
    """The default list is an immutable tuple at the module boundary.

    A caller who mutates ``_DEFAULT_MIC_KEYWORDS`` at import time
    (``_DEFAULT_MIC_KEYWORDS.append("Fake")``) would shift every
    subsequent lookup for every other caller in the same process,
    because Python module globals are shared. Naming the snapshot
    as a tuple prevents that failure mode at the type level; a
    caller who wants a mutable copy calls ``list()`` on it. This
    test surfaces a widen of the snapshot type as a review-time
    signal.
    """
    assert isinstance(_DEFAULT_MIC_KEYWORDS, tuple), (
        f"_DEFAULT_MIC_KEYWORDS must be a tuple to stay immutable at the "
        f"module boundary; got {type(_DEFAULT_MIC_KEYWORDS).__name__}."
    )


# ---------------------------------------------------------------------- #
# g1_list_bidi_audio_mic_keywords                                        #
# ---------------------------------------------------------------------- #


def test_list_returns_the_admitted_keyword_count() -> None:
    """The verb reports ``count`` matching the snapshot length.

    A caller planning a ``VOICE_MIC_NAME`` override reads ``count``
    to decide whether to iterate over the ``keywords`` list; a drift
    between the reported count and the tuple length would send the
    caller into either an off-by-one or an infinite-loop shape.
    """
    payload = _call(g1_list_bidi_audio_mic_keywords)
    assert payload["status"] == "success"
    assert payload["count"] == len(_DEFAULT_MIC_KEYWORDS)
    assert payload["count"] == 5


def test_list_returns_the_keywords_in_priority_order() -> None:
    """The ``keywords`` list preserves the neon-observed priority order.

    The bundle's helper walks its list from index zero and stops at
    the first match; a snapshot that re-sorted the tuple
    alphabetically would put ``"Brio"`` before ``"DJI"`` and change
    which device the write path selects. The priority index on each
    descriptor matches its list position so a caller iterating the
    payload sees the same order the driver's own walk uses.
    """
    payload = _call(g1_list_bidi_audio_mic_keywords)
    for i, entry in enumerate(payload["keywords"]):
        assert entry["priority"] == i, (
            f"keywords[{i}] priority={entry['priority']} disagrees with its "
            "list position; the snapshot must preserve the bundle's own "
            "walk order."
        )
        assert entry["keyword"] == _DEFAULT_MIC_KEYWORDS[i], (
            f"keywords[{i}] keyword={entry['keyword']!r} disagrees with "
            f"_DEFAULT_MIC_KEYWORDS[{i}]={_DEFAULT_MIC_KEYWORDS[i]!r}."
        )


def test_list_names_the_bundle_case_insensitivity_flag() -> None:
    """Each descriptor names ``match_case_insensitive=True``.

    The bundle's helper lowercases both operands before its ``in``
    check; a caller comparing an intended device name against the
    snapshot has to know the match is case-insensitive to avoid
    over-refusing a device whose PyAudio name is ``"dji receiver"``.
    Every entry carries the flag so a caller reading one entry (via
    :func:`g1_bidi_audio_mic_keyword_admits`) sees the same shape.
    """
    payload = _call(g1_list_bidi_audio_mic_keywords)
    for entry in payload["keywords"]:
        assert entry["match_case_insensitive"] is True


def test_list_carries_the_override_env_on_every_entry_and_top_level() -> None:
    """The ``VOICE_MIC_NAME`` env-var name appears on every entry and top-level.

    A caller planning to ship a ``VOICE_MIC_NAME`` value must have
    the exact string to grep for; carrying the name only inside the
    ``keywords`` list would force a caller who only wants the string
    to iterate every entry. Naming it at the top level too gives one
    place to look. Both fields must name the same env-var literal so
    a caller reading either path sees the same answer.
    """
    payload = _call(g1_list_bidi_audio_mic_keywords)
    assert payload["override_env"] == "VOICE_MIC_NAME"
    for entry in payload["keywords"]:
        assert entry["override_env"] == payload["override_env"], (
            "override_env drifted between per-entry and top-level surfaces; the two must name the same env-var literal."
        )


def test_list_carries_the_names_list_in_priority_order() -> None:
    """The top-level ``names`` list mirrors the tuple exactly.

    A caller who only wants the string literals (for logging, for
    a shell env-var construction, for a serialised config file)
    reads ``names`` without iterating the ``keywords`` descriptor
    list. The order must match the tuple to keep the priority
    contract; a re-sorted ``names`` field would silently teach a
    caller the wrong walk order.
    """
    payload = _call(g1_list_bidi_audio_mic_keywords)
    assert payload["names"] == list(_DEFAULT_MIC_KEYWORDS)


# ---------------------------------------------------------------------- #
# g1_bidi_audio_mic_keyword_admits: admitted names                       #
# ---------------------------------------------------------------------- #


def test_admits_every_admitted_keyword_in_the_default_list() -> None:
    """Every literal in the snapshot admits with its priority slot.

    A caller comparing an intended override value against the
    snapshot reaches this verb once per candidate keyword; a drift
    between the ``keyword`` field on the payload and the input is a
    shape bug this loop pins per entry rather than only for the
    happy-path first entry.
    """
    for i, keyword in enumerate(_DEFAULT_MIC_KEYWORDS):
        payload = _call(g1_bidi_audio_mic_keyword_admits, keyword=keyword)
        assert payload["status"] == "success", f"{keyword!r} was not admitted; payload: {payload}"
        entry = payload["keyword"]
        assert entry["priority"] == i
        assert entry["keyword"] == keyword
        assert entry["match_case_insensitive"] is True
        assert entry["override_env"] == "VOICE_MIC_NAME"


# ---------------------------------------------------------------------- #
# g1_bidi_audio_mic_keyword_admits: refusal shapes                       #
# ---------------------------------------------------------------------- #


def test_admits_refuses_none_with_a_required_argument_message() -> None:
    """A missing argument reads as required, not silently as a default.

    The bundle's ``_mic_keywords`` accessor never takes ``None`` -
    it either reads the env-var or returns the default list. A
    caller who passed ``None`` intended to test the accessor's
    default path, and this verb refuses that shape with a message
    that names the admitted set so the caller sees the right
    next step (call :func:`g1_list_bidi_audio_mic_keywords` if
    they want the whole list).
    """
    payload = _call(g1_bidi_audio_mic_keyword_admits, keyword=None)
    assert payload["status"] == "error"
    assert "required" in payload["message"]
    assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_bool_arguments_at_the_boundary() -> None:
    """Bool values are refused without silent str-coercion.

    Python's ``str(True) == "True"`` and ``True == 1`` would let a
    boolean quietly compare against the snapshot; the refusal names
    the bool at the boundary so the shape error surfaces before the
    membership test. This is the same guard
    :mod:`~strands_robots.tools.g1.g1_voice_providers` uses at
    the same position.
    """
    for arg in (True, False):
        payload = _call(g1_bidi_audio_mic_keyword_admits, keyword=arg)
        assert payload["status"] == "error"
        assert "bool" in payload["message"]
        assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_non_string_non_bool_arguments() -> None:
    """Ints, floats, lists, dicts are refused with a type-named message.

    A caller who passes a raw PyAudio device index (an int) instead
    of a keyword string reaches this refusal; the message names the
    type so the caller sees the shape error. Comparing an ``int``
    against a ``str`` in Python raises no exception - it just always
    returns ``False`` - so this refusal is what surfaces the type
    mismatch decidably.
    """
    args: tuple[object, ...] = (0, 1, 3.14, [], {}, ("DJI",))
    for arg in args:
        payload = _call(g1_bidi_audio_mic_keyword_admits, keyword=arg)
        assert payload["status"] == "error", f"{arg!r} of type {type(arg).__name__} was not refused"
        assert type(arg).__name__ in payload["message"]
        assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_the_empty_string_decidably() -> None:
    """``""`` is refused rather than treated as a default.

    A caller who passed the empty string intended to test the
    default-path behaviour; the refusal names the admitted set so
    the caller sees the right next step, rather than silently
    matching (which the tuple can't, since none of its entries is
    empty) and returning a confusing off-set refusal instead.
    """
    payload = _call(g1_bidi_audio_mic_keyword_admits, keyword="")
    assert payload["status"] == "error"
    assert "non-empty" in payload["message"]
    assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_a_miscased_keyword_with_the_admitted_casing() -> None:
    """``"dji"`` is refused with a message quoting ``"DJI"`` as the admitted form.

    The bundle's helper lowercases both operands before its ``in``
    check, but the *snapshot* carries the exact casing the bundle
    ships. A caller who intended to store a keyword in a
    ``VOICE_MIC_NAME`` override reaches this verb to see the
    correct casing; silently matching ``"dji"`` here would teach
    the caller a casing that the bundle's ``in`` check would
    accept but that violates the snapshot's own contract.
    """
    payload = _call(g1_bidi_audio_mic_keyword_admits, keyword="dji")
    assert payload["status"] == "error"
    # The refusal message quotes the admitted set so the correct
    # casing is one lookup away.
    assert "DJI" in payload["message"]
    assert "strands-labs/robots#358" in payload["message"]


def test_admits_refuses_an_off_set_keyword_with_the_admitted_set() -> None:
    """An unknown keyword refusal names every admitted literal.

    A caller who intended to admit their observed mic family (say,
    ``"Yeti"``) reads the refusal payload to see the neon-observed
    set; a message that dropped the set would force a second
    lookup against :func:`g1_list_bidi_audio_mic_keywords`.
    """
    payload = _call(g1_bidi_audio_mic_keyword_admits, keyword="Yeti")
    assert payload["status"] == "error"
    for kw in _DEFAULT_MIC_KEYWORDS:
        assert kw in payload["message"], (
            f"refusal message dropped {kw!r} from the admitted-set citation; "
            "the caller must be able to see the correct set without a second "
            "lookup."
        )
    assert "strands-labs/robots#358" in payload["message"]


# ---------------------------------------------------------------------- #
# Shape parity with sibling verbs                                        #
# ---------------------------------------------------------------------- #


def test_admit_shape_matches_list_entry_shape() -> None:
    """The admit-path descriptor names the same fields as the list entries.

    A caller who reads :func:`g1_list_bidi_audio_mic_keywords` and
    then reaches :func:`g1_bidi_audio_mic_keyword_admits` on one
    entry must see the same shape both ways; a widen to the
    descriptor on one path but not the other is a review-time
    signal this test surfaces. The shape must match the ``keyword``
    subkey of each list entry.
    """
    list_payload = _call(g1_list_bidi_audio_mic_keywords)
    list_entry = list_payload["keywords"][0]
    admit_payload = _call(
        g1_bidi_audio_mic_keyword_admits,
        keyword=list_entry["keyword"],
    )
    assert admit_payload["status"] == "success"
    admit_entry = admit_payload["keyword"]
    assert set(list_entry.keys()) == set(admit_entry.keys()), (
        f"list-entry shape {sorted(list_entry.keys())} disagrees with "
        f"admit-entry shape {sorted(admit_entry.keys())}. The two must "
        "name the same fields verbatim so a caller reading either sees "
        "the same descriptor."
    )
    assert list_entry == admit_entry
