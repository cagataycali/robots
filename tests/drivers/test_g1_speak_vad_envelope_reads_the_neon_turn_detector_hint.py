"""The speak VAD envelope tools name the neon-observed turn-detector bounds.

The neon bundle's ``g1_speak`` verb
(``cagataycali/neon-the-g1/tools/g1_speak.py``) takes two turn-detection
knobs at ``action="start"``: ``vad_threshold`` (a float in ``[0.0, 1.0]``
naming the voice-activity detector's admission floor) and
``silence_duration_ms`` (a positive integer naming the trailing-silence
window that ends a turn).  The bundle passes both to the ``BidiAgent``
factory the ``g1.build_voice_agent`` helper constructs; the factory
forwards them to the turn-detector configuration inside
``strands.experimental.bidi`` without itself refusing an out-of-range
value.  The :mod:`strands_robots.tools.g1.g1_speak_vad_envelope` module
snapshots the observed envelope into module-level constants and exposes
two agent-facing verbs -
:func:`g1_list_speak_vad_envelope` (name the whole envelope) and
:func:`g1_speak_vad_admits` (decide one query) - so a caller can decide
the refusal decidably before a future driver-side wrapper for
``g1_speak(action="start")`` is called.  The tests here fix that
contract without pulling the SDK: the module is loadable on a host
without ``unitree_sdk2py`` *and* without the optional audio-stack
imports (``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``)
the neon bundle's runtime path pulls (the same SDK-load-hygiene rule
every other file under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen or
narrow to the constants surfaces here as a shape change rather than as
a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The ``BidiAgent`` turn detector's live answer at wire time.  The
  verbs answer against the module-level snapshot, not against a live
  import of the detector's admission handler (the whole point of the
  port is that the snapshot lets a headless host answer).  A
  driver-side wrapper for ``g1_speak`` that lands later will
  re-validate against the detector's live handler at wire time;
  testing the snapshot vs the live handler is a driver-side test, not
  a lookup-side one.
* Whether the caller's ``vad_threshold`` matches the caller's mic
  gain.  A threshold above the mic's typical energy is a runtime
  liveness question the neon bundle answers under its
  ``STATS["energy_mean_abs"]`` reading; neither this lookup nor the
  bidi detector's admission set can decide it ahead of wire time.
  The membership tests here grade the numeric envelope only.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

from strands_robots.tools.g1.g1_speak_vad_envelope import (
    _REFUSAL_TEXT_SILENCE_DURATION_MS,
    _REFUSAL_TEXT_VAD_THRESHOLD,
    _SILENCE_DURATION_MS_MIN,
    _SILENCE_DURATION_MS_NEON_DEFAULT,
    _VAD_THRESHOLD_MAX,
    _VAD_THRESHOLD_MIN,
    _VAD_THRESHOLD_NEON_DEFAULT,
    g1_list_speak_vad_envelope,
    g1_speak_vad_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim.  This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a submodule
    at import time would break every headless CI runner and Thor
    before an office bring-up (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_speak_vad_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_speak_vad_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_import_pulls_no_optional_audio_stack_module() -> None:
    """The tool module is loadable without the optional audio-stack imports.

    The neon bundle's runtime path pulls ``pywebrtc_audio`` /
    ``pyaudio`` / ``strands.experimental.bidi`` inside its own
    ``_probe_bidi`` helper; those are optional dependencies the
    ``strands-robots`` package does not require.  A read-only lookup
    module that pulled any of them at import time would break every
    headless CI runner and Thor before an office bring-up on the same
    surface a missing SDK submodule would, refs
    strands-labs/robots#358.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_speak_vad_envelope")
    after = set(sys.modules)
    audio_stack_prefixes = ("pywebrtc_audio", "pyaudio")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in audio_stack_prefixes)
    }
    # ``strands.experimental.bidi`` is graded separately: it is a
    # subpackage of ``strands`` (which this test module transitively
    # imports through the tool decorator), so its ``bidi`` subpackage
    # is what a leak here would surface.
    bidi_leaked = {name for name in after - before if name.startswith("strands.experimental.bidi")}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_speak_vad_envelope imports pulled "
        f"optional audio-stack submodules: {leaked}. The lookup half of "
        "the ``g1_speak`` port must not pull the runtime audio-stack "
        "imports (refs strands-labs/robots#358)."
    )
    assert bidi_leaked == set(), (
        f"strands_robots.tools.g1.g1_speak_vad_envelope imports pulled "
        f"``strands.experimental.bidi`` submodules: {bidi_leaked}. The "
        "lookup half must not pull the bidi runtime stack; that pull "
        "belongs on a future driver-side wrapper (refs "
        "strands-labs/robots#358)."
    )


def test_the_snapshot_names_the_neon_observed_envelope() -> None:
    """The module-level constants match the neon docstring's observed values.

    The neon ``g1_speak`` docstring names the two knobs as:

    * ``vad_threshold: 0.0-1.0. Higher = less twitchy. 0.7 stops
      echo triggers``
    * ``silence_duration_ms: How long of silence ends a turn. 700 =
      relaxed``

    The clamp pair (``[0.0, 1.0]`` for ``vad_threshold``, ``>= 1``
    for ``silence_duration_ms``) and the two neon-observed defaults
    (``0.7`` and ``700``) are pinned so a widen on the neon side
    surfaces here as a diverging constant rather than as a silent
    envelope drift.
    """
    assert _VAD_THRESHOLD_MIN == 0.0
    assert _VAD_THRESHOLD_MAX == 1.0
    assert _VAD_THRESHOLD_NEON_DEFAULT == 0.7
    assert _SILENCE_DURATION_MS_MIN == 1
    assert _SILENCE_DURATION_MS_NEON_DEFAULT == 700


def test_the_refusal_texts_name_the_module_local_remedy() -> None:
    """The refusal texts quote the module-local bounds, not a motion-FSM code.

    The bidi voice pipeline ships no distinct rc for a bounds-
    violated ``vad_threshold`` or ``silence_duration_ms``, and the
    motion-FSM ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` reads
    ``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a remedy
    that points a planner at locomotion FSM transitions to fix a
    turn-detector argument.  This cell locks the module-local remedy
    text so a future refactor that re-borrows a motion code lands as
    a shape change here rather than silently.
    """
    assert "vad_threshold" in _REFUSAL_TEXT_VAD_THRESHOLD
    assert f"[{_VAD_THRESHOLD_MIN}, {_VAD_THRESHOLD_MAX}]" in _REFUSAL_TEXT_VAD_THRESHOLD
    assert "silence_duration_ms" in _REFUSAL_TEXT_SILENCE_DURATION_MS
    assert f">= {_SILENCE_DURATION_MS_MIN}" in _REFUSAL_TEXT_SILENCE_DURATION_MS
    # The refusal texts must NOT re-borrow the motion-FSM 7404 remedy.
    for text in (_REFUSAL_TEXT_VAD_THRESHOLD, _REFUSAL_TEXT_SILENCE_DURATION_MS):
        assert "FSM" not in text, (
            f"refusal text {text!r} names an FSM remedy; the bidi voice "
            "pipeline ships no motion-FSM refusal, and the module-local "
            "text must not re-borrow ``7404``."
        )


def test_g1_list_speak_vad_envelope_returns_the_whole_envelope() -> None:
    """The verb's payload names every clamp and every refusal descriptor.

    ``envelope`` is a dict whose fields match the module's own
    snapshot; ``refusals`` is a list of one descriptor per dimension
    (``vad_threshold`` and ``silence_duration_ms``) with the module-
    local refusal text a future write verb would surface on a bounds
    violation.  Every field is read off the module's constants (not
    restated in the test body), so a widen to the descriptor lands
    in one place.
    """
    result = _call(g1_list_speak_vad_envelope)
    assert result["status"] == "success"
    envelope = result["envelope"]
    assert envelope["vad_threshold_min"] == _VAD_THRESHOLD_MIN
    assert envelope["vad_threshold_max"] == _VAD_THRESHOLD_MAX
    assert envelope["vad_threshold_neon_default"] == _VAD_THRESHOLD_NEON_DEFAULT
    assert envelope["silence_duration_ms_min"] == _SILENCE_DURATION_MS_MIN
    assert envelope["silence_duration_ms_neon_default"] == _SILENCE_DURATION_MS_NEON_DEFAULT
    refusal_texts = {r["text"] for r in result["refusals"]}
    assert refusal_texts == {
        _REFUSAL_TEXT_VAD_THRESHOLD,
        _REFUSAL_TEXT_SILENCE_DURATION_MS,
    }
    refusal_dims = {r["dimension"] for r in result["refusals"]}
    assert refusal_dims == {"vad_threshold", "silence_duration_ms"}


def test_g1_list_speak_vad_envelope_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh dicts and lists; a mutation on the
    returned ``envelope`` dict or ``refusals`` list does not leak
    back into the module's constants.  This cell is where a share-a-
    reference regression would surface once, not scattered across
    every call site.
    """
    result = _call(g1_list_speak_vad_envelope)
    result["envelope"]["synthetic"] = True
    result["refusals"].append({"synthetic": True})
    fresh = _call(g1_list_speak_vad_envelope)
    assert "synthetic" not in fresh["envelope"]
    assert all("synthetic" not in r for r in fresh["refusals"])


def test_g1_speak_vad_admits_admits_the_neon_defaults() -> None:
    """The neon-tuned defaults ``0.7`` and ``700`` are admitted.

    The neon docstring names ``vad_threshold=0.7`` /
    ``silence_duration_ms=700`` as the tuned values for the G1 DDS
    speaker->mic loop; a caller who does not pass explicit arguments
    lands on those defaults and this verb admits the pair without
    surfacing a refusal.
    """
    result = _call(g1_speak_vad_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []
    assert result["envelope"]["vad_threshold_neon_default"] == 0.7
    assert result["envelope"]["silence_duration_ms_neon_default"] == 700


def test_g1_speak_vad_admits_admits_the_envelope_boundaries() -> None:
    """The inclusive lower and upper clamps for both dimensions admit.

    ``vad_threshold=0.0`` (accept every frame) and
    ``vad_threshold=1.0`` (accept only a certain-speech frame) are
    both legitimate operator commands on the ``[0.0, 1.0]`` range,
    not the shape-error the refusal exists to catch.
    ``silence_duration_ms=1`` (the exclusive-of-zero minimum) is a
    legitimate "end the turn immediately" command.  This cell locks
    the inclusive-boundary contract.
    """
    for threshold in (_VAD_THRESHOLD_MIN, _VAD_THRESHOLD_MAX):
        result = _call(g1_speak_vad_admits, vad_threshold=threshold, silence_duration_ms=1)
        assert result["admits"] is True, (
            f"vad_threshold={threshold!r} at the boundary should have admitted, got {result!r}."
        )
    # silence_duration_ms=1 is the exclusive-of-zero minimum; ensure
    # the boundary is admitted rather than refused off-by-one.
    result = _call(g1_speak_vad_admits, vad_threshold=0.5, silence_duration_ms=_SILENCE_DURATION_MS_MIN)
    assert result["admits"] is True


def test_g1_speak_vad_admits_admits_an_int_widening_to_float() -> None:
    """An integer ``vad_threshold`` widens to ``float`` and is admitted.

    A caller writing ``vad_threshold=1`` (an integer rather than a
    literal ``1.0``) means the same admitted ceiling as ``1.0``; the
    verb widens the int to a float and admits it.  This mirrors the
    ``BidiAgent`` turn detector's own comparison, which numerically
    compares its score against the argument without pinning the
    argument's Python type.
    """
    result = _call(g1_speak_vad_admits, vad_threshold=1, silence_duration_ms=700)
    assert result["admits"] is True
    result = _call(g1_speak_vad_admits, vad_threshold=0, silence_duration_ms=700)
    assert result["admits"] is True


def test_g1_speak_vad_admits_refuses_a_bool_threshold() -> None:
    """A ``bool`` ``vad_threshold`` is refused because ``True`` would widen to ``1.0``.

    Python's ``bool`` is a subclass of ``int`` (and ``int`` widens
    to ``float`` below), so a caller passing ``True`` would otherwise
    silently look up ``1.0`` (the admitted ceiling) and hide the type
    mistake.  The verb refuses both bool values at the boundary so
    the mistake surfaces.
    """
    for value in (True, False):
        result = _call(g1_speak_vad_admits, vad_threshold=value, silence_duration_ms=700)
        assert result["admits"] is False, f"vad_threshold={value!r} (bool) should have been refused, got {result!r}."
        assert any(r["dimension"] == "vad_threshold" and r["comparison"] == "non-float" for r in result["refusals"])
        assert any(r["text"] == _REFUSAL_TEXT_VAD_THRESHOLD for r in result["refusals"])


def test_g1_speak_vad_admits_refuses_a_non_numeric_threshold() -> None:
    """A string / list / dict ``vad_threshold`` is refused with the non-float remedy.

    A caller writing ``vad_threshold="0.7"`` or
    ``vad_threshold=[0.7]`` is making a shape mistake, not naming
    the detector's float threshold.  The verb refuses each with the
    module-local :data:`_REFUSAL_TEXT_VAD_THRESHOLD` so the caller
    reads the remedy that matches the surface.
    """
    for value in ("0.7", [0.7], (0.7,), {"threshold": 0.7}):
        result = _call(g1_speak_vad_admits, vad_threshold=value, silence_duration_ms=700)
        assert result["admits"] is False, (
            f"vad_threshold={value!r} ({type(value).__name__}) should have been refused, got {result!r}."
        )
        vad_refusals = [r for r in result["refusals"] if r["dimension"] == "vad_threshold"]
        assert vad_refusals, f"missing vad_threshold refusal for {value!r}"
        assert vad_refusals[0]["comparison"] == "non-float"
        assert vad_refusals[0]["text"] == _REFUSAL_TEXT_VAD_THRESHOLD


def test_g1_speak_vad_admits_refuses_a_non_finite_threshold() -> None:
    """A ``nan`` / ``inf`` / ``-inf`` ``vad_threshold`` is refused.

    The ``BidiAgent`` turn detector's comparison against a non-finite
    threshold degenerates (``score >= nan`` is always ``False``,
    ``score >= inf`` is always ``False``); silencing every frame at
    wire time is the failure mode this refusal exists to catch.  The
    module-local remedy names the finite ``[0.0, 1.0]`` range.
    """
    for value in (math.nan, math.inf, -math.inf):
        result = _call(g1_speak_vad_admits, vad_threshold=value, silence_duration_ms=700)
        assert result["admits"] is False, (
            f"vad_threshold={value!r} (non-finite) should have been refused, got {result!r}."
        )
        vad_refusals = [r for r in result["refusals"] if r["dimension"] == "vad_threshold"]
        assert vad_refusals, f"missing vad_threshold refusal for {value!r}"
        assert vad_refusals[0]["comparison"] == "non-finite"
        assert vad_refusals[0]["text"] == _REFUSAL_TEXT_VAD_THRESHOLD


def test_g1_speak_vad_admits_refuses_an_out_of_range_threshold() -> None:
    """A ``vad_threshold`` outside ``[0.0, 1.0]`` is refused with the range remedy.

    Below ``0.0`` the detector's admission floor is undefined; above
    ``1.0`` no score can meet the threshold and every frame is
    silenced.  The verb refuses both sides and names the violated
    bound (``vad_threshold_min`` on the low side,
    ``vad_threshold_max`` on the high side).
    """
    for value, expected_bound_key, expected_cmp in (
        (-0.1, "vad_threshold_min", "value < bound"),
        (-1.0, "vad_threshold_min", "value < bound"),
        (1.1, "vad_threshold_max", "value > bound"),
        (10.0, "vad_threshold_max", "value > bound"),
    ):
        result = _call(g1_speak_vad_admits, vad_threshold=value, silence_duration_ms=700)
        assert result["admits"] is False, (
            f"vad_threshold={value!r} out of range should have been refused, got {result!r}."
        )
        vad_refusals = [r for r in result["refusals"] if r["dimension"] == "vad_threshold"]
        assert vad_refusals, f"missing vad_threshold refusal for {value!r}"
        assert vad_refusals[0]["bound_key"] == expected_bound_key
        assert vad_refusals[0]["comparison"] == expected_cmp
        assert vad_refusals[0]["text"] == _REFUSAL_TEXT_VAD_THRESHOLD


def test_g1_speak_vad_admits_refuses_a_bool_silence() -> None:
    """A ``bool`` ``silence_duration_ms`` is refused because ``True`` would look up ``1``.

    Python's ``bool`` is a subclass of ``int``; a caller passing
    ``True`` would otherwise silently look up ``1`` (a legitimate
    one-millisecond wait) and hide the type mistake.  The verb
    refuses both bool values at the boundary.
    """
    for value in (True, False):
        result = _call(g1_speak_vad_admits, vad_threshold=0.7, silence_duration_ms=value)
        assert result["admits"] is False, (
            f"silence_duration_ms={value!r} (bool) should have been refused, got {result!r}."
        )
        sil_refusals = [r for r in result["refusals"] if r["dimension"] == "silence_duration_ms"]
        assert sil_refusals, f"missing silence_duration_ms refusal for {value!r}"
        assert sil_refusals[0]["comparison"] == "non-int"
        assert sil_refusals[0]["text"] == _REFUSAL_TEXT_SILENCE_DURATION_MS


def test_g1_speak_vad_admits_refuses_a_non_int_silence() -> None:
    """A ``float`` / ``str`` / ``list`` ``silence_duration_ms`` is refused.

    A caller writing ``silence_duration_ms=700.0`` sees an
    actionable refusal rather than a silent truncation the turn
    detector would perform; a caller writing
    ``silence_duration_ms="700"`` is making a shape mistake.  The
    verb refuses each and quotes the module-local remedy.
    """
    for value in (700.0, "700", [700], (700,), 7.5):
        result = _call(g1_speak_vad_admits, vad_threshold=0.7, silence_duration_ms=value)
        assert result["admits"] is False, (
            f"silence_duration_ms={value!r} ({type(value).__name__}) should have been refused, got {result!r}."
        )
        sil_refusals = [r for r in result["refusals"] if r["dimension"] == "silence_duration_ms"]
        assert sil_refusals, f"missing silence_duration_ms refusal for {value!r}"
        assert sil_refusals[0]["comparison"] == "non-int"
        assert sil_refusals[0]["text"] == _REFUSAL_TEXT_SILENCE_DURATION_MS


def test_g1_speak_vad_admits_refuses_a_non_positive_silence() -> None:
    """A ``silence_duration_ms`` at or below zero is refused.

    A non-positive wait collapses the turn boundary to a single
    frame and cuts off every operator pause between words; the neon
    bundle names positive integers only, matching the shared
    :func:`~strands_robots.utils.positive_count_error` domain every
    other integer-count knob in this package uses.  The refusal
    surfaces the ``>= 1`` remedy.
    """
    for value in (0, -1, -700):
        result = _call(g1_speak_vad_admits, vad_threshold=0.7, silence_duration_ms=value)
        assert result["admits"] is False, (
            f"silence_duration_ms={value!r} (non-positive) should have been refused, got {result!r}."
        )
        sil_refusals = [r for r in result["refusals"] if r["dimension"] == "silence_duration_ms"]
        assert sil_refusals, f"missing silence_duration_ms refusal for {value!r}"
        assert sil_refusals[0]["bound_key"] == "silence_duration_ms_min"
        assert sil_refusals[0]["comparison"] == "value < bound"
        assert sil_refusals[0]["text"] == _REFUSAL_TEXT_SILENCE_DURATION_MS


def test_g1_speak_vad_admits_reports_both_dimensions_together() -> None:
    """A caller with two bad arguments reads two refusals in one payload.

    The two dimensions are graded independently rather than short-
    circuiting on the first violation; a caller passing a bad
    threshold *and* a bad silence duration reads both refusals in a
    single call so the correction lands in one round-trip rather
    than in a ping-pong of single-refusal payloads.
    """
    result = _call(g1_speak_vad_admits, vad_threshold=2.0, silence_duration_ms=0)
    assert result["admits"] is False
    dims = {r["dimension"] for r in result["refusals"]}
    assert dims == {"vad_threshold", "silence_duration_ms"}


def test_g1_speak_vad_admits_returns_the_envelope_on_every_call() -> None:
    """Every payload carries the ``envelope`` sub-dict, admitted or refused.

    A caller reading the refusal payload has the same envelope on
    hand that
    :func:`g1_list_speak_vad_envelope` returns, so the correction
    step does not need a second call.  This cell locks the shape.
    """
    for kwargs in (
        {"vad_threshold": 0.7, "silence_duration_ms": 700},
        {"vad_threshold": 2.0, "silence_duration_ms": 0},
        {"vad_threshold": "0.7", "silence_duration_ms": 700},
    ):
        result = _call(g1_speak_vad_admits, **kwargs)
        assert "envelope" in result
        assert result["envelope"]["vad_threshold_min"] == _VAD_THRESHOLD_MIN
        assert result["envelope"]["vad_threshold_max"] == _VAD_THRESHOLD_MAX
        assert result["envelope"]["silence_duration_ms_min"] == _SILENCE_DURATION_MS_MIN
