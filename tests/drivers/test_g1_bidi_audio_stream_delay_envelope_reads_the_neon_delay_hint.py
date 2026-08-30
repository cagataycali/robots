"""The stream-delay-envelope lookup tools name what the neon bidi bundle admits.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) constructs a
WebRTC :class:`AudioProcessor` with a ``stream_delay_ms`` argument
that names the speaker->mic loopback delay AEC must compensate for
on the G1's DDS audio path. The neon bundle names the G1-tuned
default as ``DEFAULT_STREAM_DELAY_MS = 120`` and its ``g1_speak``
verb docstring surfaces the same number as the value "for G1 DDS
path"; the ``AudioProcessor`` constructor itself places no clamp
on the argument beyond WebRTC's internal delay-buffer bound (past
which the echo canceller silently truncates), so the
:mod:`strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
module snapshots that observed range into module-level constants
and exposes two agent-facing verbs -
:func:`g1_list_bidi_audio_stream_delay_envelope` (name the whole
envelope) and :func:`g1_stream_delay_ms_admits` (decide one
query) - so a caller can decide the refusal decidably before a
future audio-processing write path is attempted. The tests here
fix that contract without pulling the SDK or ``pywebrtc_audio``:
the module is loadable on a host without ``unitree_sdk2py`` and
without the WebRTC audio stack (the same SDK-load-hygiene rule
every other file under :mod:`strands_robots.tools.g1` carries,
refs strands-labs/robots#358), and every membership answer is
read off the module's own snapshot rather than restated in the
tests, so a widen or narrow to the observed range surfaces here
as a shape change rather than as a diverging table this file
would need to manually update.

Two things this file's cells deliberately do not pin:

* The WebRTC library's own answer at wire time. The envelope is
  the neon bundle's observed range, not WebRTC's compile-time
  delay-buffer bound (which is an implementation detail of the
  ``AudioProcessor`` C++ layer and not part of the ``pywebrtc_audio``
  Python surface). A driver-side wrapper for the bidi IO that
  lands later will re-check the envelope at wire time and its
  refusal string will surface the same module-local
  :data:`_REFUSAL_TEXT` the admits-verb quotes today.
* The live bidi state. Whether the ``G1BidiAudioIO`` singleton is
  currently constructed, whether the mic autopick has resolved a
  device, whether the far-buffer queue is draining: those are
  live driver-instance reads and belong on a future bidi state
  verb; the envelope surfaces only the numeric bound decision.

One property this file explicitly refuses to pin: the ``7404``
motion-FSM refusal code from
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`. That code
is the driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refusal on ``rt/lowcmd`` writes and its decoded text reads
``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a locomotion
FSM remedy. The ``AudioProcessor`` runs on the mic pre-processing
thread in the Python process itself and never touches
``rt/lowcmd``; the audio-processing pipeline ships no distinct rc
for a bounds-violated stream-delay argument, and the refusal text
this module surfaces is module-local so a planner reading a
stream-delay refusal sees a remedy that matches the surface, not
a re-borrowed motion FSM code. Cells below pin only the
module-local text; a re-borrowing of ``7404`` would fail
``test_the_refusal_text_names_the_stream_delay_envelope_not_the_motion_fsm``.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope import (
    _REFUSAL_TEXT,
    _STREAM_DELAY_MS_MAX,
    _STREAM_DELAY_MS_MIN,
    _STREAM_DELAY_MS_NEON_DEFAULT,
    g1_list_bidi_audio_stream_delay_envelope,
    g1_stream_delay_ms_admits,
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

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358);
    a module that pulled a submodule at import time would break
    every headless CI runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope "
        f"imports pulled SDK submodules: {leaked}. The rule for this "
        f"package is that the SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


def test_the_import_pulls_no_pywebrtc_audio_module() -> None:
    """The tool module is loadable on a host without ``pywebrtc_audio``.

    The neon bundle's ``g1_bidi_audio`` imports ``pywebrtc_audio`` at
    module load to construct the WebRTC :class:`AudioProcessor`; the
    envelope port must not close that dependency on this module so
    a headless CI runner can decide the numeric refusal without
    the audio stack present. Pinned here so a future edit that
    reaches into ``pywebrtc_audio`` at import time (for a compile-time
    delay-buffer bound, say) fails this cell first, not as a
    dependency surprise on a mesh peer that never installs the
    audio stack.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "pywebrtc" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope "
        f"imports pulled pywebrtc_audio submodules: {leaked}. The "
        f"envelope port is numeric-only; the WebRTC library belongs "
        f"inside the driver-side wrapper that lands later, refs "
        f"strands-labs/robots#358."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """The envelope bounds are integers with min <= max.

    An inverted min/max pair (min > max) would reject every integer
    value; a non-integer bound would let a caller passing 120.5
    slip through the type-refusal path. Pins the invariant so a
    widen or narrow of the observed range that inverts a pair
    surfaces here rather than as a silently unreachable envelope
    in production.
    """
    assert isinstance(_STREAM_DELAY_MS_MIN, int) and not isinstance(_STREAM_DELAY_MS_MIN, bool), (
        f"_STREAM_DELAY_MS_MIN is not a plain int: {_STREAM_DELAY_MS_MIN!r}"
    )
    assert isinstance(_STREAM_DELAY_MS_MAX, int) and not isinstance(_STREAM_DELAY_MS_MAX, bool), (
        f"_STREAM_DELAY_MS_MAX is not a plain int: {_STREAM_DELAY_MS_MAX!r}"
    )
    assert _STREAM_DELAY_MS_MIN <= _STREAM_DELAY_MS_MAX, (
        f"stream-delay bounds inverted: min={_STREAM_DELAY_MS_MIN} > "
        f"max={_STREAM_DELAY_MS_MAX}. g1_stream_delay_ms_admits would "
        f"refuse every value."
    )


def test_the_neon_default_sits_inside_the_envelope() -> None:
    """The neon-tuned default is inside the observed clamp pair.

    The neon bundle's ``DEFAULT_STREAM_DELAY_MS = 120`` names the
    G1-DDS-tuned value the ``G1BidiAudioIO`` constructor reaches
    for when the caller does not pass ``stream_delay_ms``.
    Surfacing the default at the envelope layer without also
    keeping it inside the clamp pair would hand a caller a
    "neon-observed" value the admits verb would then refuse -
    contradiction. Pinned so a narrow of the observed range that
    accidentally drops the neon default fails this cell first,
    not as an admit/refuse contradiction the caller has to reason
    about at runtime.
    """
    assert isinstance(_STREAM_DELAY_MS_NEON_DEFAULT, int) and not isinstance(_STREAM_DELAY_MS_NEON_DEFAULT, bool), (
        f"_STREAM_DELAY_MS_NEON_DEFAULT is not a plain int: {_STREAM_DELAY_MS_NEON_DEFAULT!r}"
    )
    assert _STREAM_DELAY_MS_MIN <= _STREAM_DELAY_MS_NEON_DEFAULT <= _STREAM_DELAY_MS_MAX, (
        f"neon default {_STREAM_DELAY_MS_NEON_DEFAULT} is outside "
        f"[{_STREAM_DELAY_MS_MIN}, {_STREAM_DELAY_MS_MAX}]. A caller "
        f"pinning the neon-observed value would then be refused by "
        f"g1_stream_delay_ms_admits."
    )


def test_the_envelope_matches_the_neon_observed_range() -> None:
    """The bounds match the neon-bundle-observed ``[0, 500]`` ms range.

    The neon bundle's ``g1_bidi_audio`` names ``0`` as the floor
    (a negative delay hint is a shape mistake, not a stronger
    compensation request) and WebRTC's ``AudioProcessor``
    truncates every value past ~500 ms to its internal
    delay-buffer bound; the neon-tuned value for the G1 DDS
    speaker->mic loopback is ``120`` ms and it lands at the
    lower-middle of the envelope. Pinning the numbers here
    surfaces a drift in either direction: a widen to ``[0, 1000]``
    (a change in the WebRTC delay-buffer bound) or a narrow to
    ``[60, 200]`` (a caller-side field-note correction) would
    fail this cell first.
    """
    assert _STREAM_DELAY_MS_MIN == 0
    assert _STREAM_DELAY_MS_MAX == 500
    assert _STREAM_DELAY_MS_NEON_DEFAULT == 120


def test_the_refusal_text_names_the_stream_delay_envelope_not_the_motion_fsm() -> None:
    """The refusal text is module-local, not a re-borrowed motion FSM code.

    The G1 driver's :meth:`_check_motion_gates` refuses locomotion
    writes with rc=``7404`` whose text reads ``"Invalid FSM id -
    need FSM in {500, 501, 801}"``. The ``AudioProcessor`` runs
    on the mic pre-processing thread in the Python process itself
    and never touches ``rt/lowcmd``; the audio-processing pipeline
    ships no distinct rc for a bounds-violated stream-delay
    argument. The refusal shape this module surfaces is
    module-local text that names the stream-delay envelope (not
    the motion FSM) so an agent planner reading a stream-delay
    refusal sees a remedy on the same surface the write belongs
    on. Pinned here so a re-borrowing of ``7404`` (or any other
    motion-FSM entry from ``ERR_CODES``) fails this cell first,
    not as a wrong-remedy surprise in production.
    """
    assert isinstance(_REFUSAL_TEXT, str) and _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT is not a non-empty string: {_REFUSAL_TEXT!r}"
    )
    assert "stream_delay_ms" in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT does not name the stream_delay_ms dimension: "
        f"{_REFUSAL_TEXT!r}. A caller reading the refusal must see a "
        f"remedy on the audio-processing surface."
    )
    fsm_text = ERR_CODES[7404]
    assert _REFUSAL_TEXT != fsm_text, (
        f"_REFUSAL_TEXT re-borrows the motion-FSM ``7404`` text "
        f"{fsm_text!r}. The AudioProcessor runs on the mic "
        f"pre-processing thread in-process and never touches "
        f"rt/lowcmd; the refusal shape must be module-local so a "
        f"planner does not read a motion FSM remedy for a "
        f"stream-delay error."
    )
    assert "FSM" not in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT names the motion FSM: {_REFUSAL_TEXT!r}. The "
        f"AudioProcessor runs on the mic pre-processing thread "
        f"in-process; the refusal remedy belongs on the "
        f"audio-processing surface, not the locomotion FSM."
    )


def test_g1_list_bidi_audio_stream_delay_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp and the refusal.

    ``envelope`` carries every clamp constant (including the
    neon-observed default) and ``refusals`` names the module-local
    :data:`_REFUSAL_TEXT` a future driver-side bidi audio wrapper
    would surface on a bounds violation.
    """
    result = _call(g1_list_bidi_audio_stream_delay_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["stream_delay_ms_min"] == _STREAM_DELAY_MS_MIN
    assert env["stream_delay_ms_max"] == _STREAM_DELAY_MS_MAX
    assert env["stream_delay_ms_neon_default"] == _STREAM_DELAY_MS_NEON_DEFAULT
    assert result["refusals"] == [{"text": _REFUSAL_TEXT}]


def test_g1_list_bidi_audio_stream_delay_envelope_refusal_omits_a_borrowed_code() -> None:
    """The list-envelope refusal descriptor names no ``code`` field.

    A ``code`` field on this refusal would only be honest if the
    audio-processing pipeline shipped a distinct rc for a
    bounds-violated stream-delay argument, and it does not;
    borrowing a motion-FSM code from
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` would
    hand a planner a wrong-surface remedy. Pins the omission so a
    future re-introduction of ``code`` fails this cell first.
    """
    result = _call(g1_list_bidi_audio_stream_delay_envelope)
    for refusal in result["refusals"]:
        assert "code" not in refusal, (
            f"refusal descriptor carries a ``code`` field: {refusal!r}. "
            f"The audio-processing pipeline ships no rc for a "
            f"bounds-violated stream-delay argument; borrowing one "
            f"from ERR_CODES puts a wrong-surface remedy on an "
            f"audio-processing refusal."
        )


def test_g1_list_bidi_audio_stream_delay_envelope_returns_fresh_containers() -> None:
    """Successive calls do not share the envelope dict or refusals list.

    A caller mutating one call's ``envelope`` (or ``refusals``)
    must not affect the next call's payload. Pins the isolation
    so a mutation-at-callsite bug lands here rather than as a
    ghost-state-across-calls regression in production.
    """
    first = _call(g1_list_bidi_audio_stream_delay_envelope)
    second = _call(g1_list_bidi_audio_stream_delay_envelope)
    assert first is not second
    assert first["envelope"] is not second["envelope"]
    assert first["refusals"] is not second["refusals"]


def test_g1_stream_delay_ms_admits_the_neon_default() -> None:
    """The neon-tuned default is admitted with an empty refusals list.

    ``stream_delay_ms=120`` matches ``DEFAULT_STREAM_DELAY_MS`` in
    the neon bundle - a caller who reads the envelope's
    ``stream_delay_ms_neon_default`` and passes it to the admits
    verb must see an admit, not a refuse. Pins the round-trip so a
    narrow of the observed range that drops the neon default
    fails this cell first.
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=120)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_stream_delay_ms_admits_a_value_inside_the_envelope() -> None:
    """A value inside the envelope is admitted with an empty refusals list.

    ``stream_delay_ms=250`` sits at the middle of the observed
    range, so a driver-side wrapper for the bidi IO would not
    refuse it on envelope grounds (whether the audio stack is
    currently free is a separate live-read decision the verb does
    not answer).
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=250)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


@pytest.mark.parametrize("stream_delay_ms", [0, 500])
def test_g1_stream_delay_ms_admits_at_the_exact_clamp_boundaries(stream_delay_ms: int) -> None:
    """A value at either clamp boundary is inside, not outside.

    Boundary values ``stream_delay_ms=0`` (disabled delay hint on
    a headphone-out path where loopback is negligible) and
    ``stream_delay_ms=500`` (the WebRTC ceiling) are admitted
    because :func:`g1_stream_delay_ms_admits` refuses on
    ``value < bound`` and ``value > bound`` rather than
    ``<=``/``>= bound`` - a zero-delay hint is a legitimate
    command and off-by-one at the boundary would silently reject
    two valid saturated values.
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=stream_delay_ms)
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_stream_delay_ms_admits_refuses_a_value_below_the_floor() -> None:
    """A ``stream_delay_ms`` below ``stream_delay_ms_min`` reads as one refusal.

    The refusal descriptor names the dimension, the value the
    caller passed, the bound-key on the envelope it violated, and
    the module-local :data:`_REFUSAL_TEXT` a driver-side wrapper
    would quote.
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=-1)
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "stream_delay_ms"
    assert refusal["value"] == -1
    assert refusal["bound_key"] == "stream_delay_ms_min"
    assert refusal["bound"] == _STREAM_DELAY_MS_MIN
    assert refusal["comparison"] == "value < bound"
    assert refusal["text"] == _REFUSAL_TEXT
    assert "code" not in refusal, (
        f"below-floor refusal carries a ``code`` field: {refusal!r}. "
        f"The audio-processing pipeline ships no rc for a "
        f"bounds-violated stream-delay argument."
    )


def test_g1_stream_delay_ms_admits_refuses_a_value_above_the_ceiling() -> None:
    """A ``stream_delay_ms`` above ``stream_delay_ms_max`` reads as one refusal.

    The neon bundle observed ``500`` ms as the WebRTC ceiling; a
    caller passing ``1000`` learns the ceiling and the
    module-local refusal text before a driver-side wrapper fires,
    rather than learning at wire time only that the echo canceller
    silently truncated the argument and the AEC quality degrades
    off-screen.
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=1000)
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "stream_delay_ms"
    assert refusal["value"] == 1000
    assert refusal["bound_key"] == "stream_delay_ms_max"
    assert refusal["bound"] == _STREAM_DELAY_MS_MAX
    assert refusal["comparison"] == "value > bound"
    assert refusal["text"] == _REFUSAL_TEXT
    assert "code" not in refusal


def test_g1_stream_delay_ms_admits_refuses_a_bool_stream_delay_ms() -> None:
    """A ``bool`` ``stream_delay_ms`` reads as refused with a ``non-int`` comparison.

    Python's ``bool`` is a subclass of ``int``, so ``True`` would
    otherwise silently look up ``1`` (a legitimate one-millisecond
    delay hint) and hide the type mistake behind an
    admitted-with-tiny-delay result. Refusing at the boundary
    surfaces the mistake instead.
    """
    for value in (True, False):
        result = _call(g1_stream_delay_ms_admits, stream_delay_ms=value)
        assert result["admits"] is False, f"bool {value!r} was admitted as a stream_delay_ms"
        refusal = result["refusals"][0]
        assert refusal["dimension"] == "stream_delay_ms"
        assert refusal["comparison"] == "non-int"
        assert refusal["text"] == _REFUSAL_TEXT


@pytest.mark.parametrize(
    "stream_delay_ms",
    [120.0, 120.5, Decimal("120"), "120", None, [], (120,)],
)
def test_g1_stream_delay_ms_admits_refuses_a_non_int_stream_delay_ms(
    stream_delay_ms: Any,
) -> None:
    """A non-int-non-bool ``stream_delay_ms`` reads as refused with a ``non-int`` comparison.

    ``float`` values (even integer-valued like ``120.0``) are
    refused because the WebRTC ``AudioProcessor`` constructor
    expects an integer; a caller passing ``120.0`` learns the
    shape mistake here rather than at wire time via a silent
    truncation. ``Decimal`` and ``str`` follow the same rule (a
    str is refused rather than parsed - the verb does not
    fabricate a value the caller did not supply).
    """
    result = _call(g1_stream_delay_ms_admits, stream_delay_ms=stream_delay_ms)
    assert result["admits"] is False, f"{stream_delay_ms!r} was admitted as a stream_delay_ms"
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "stream_delay_ms"
    assert refusal["comparison"] == "non-int"
    assert refusal["text"] == _REFUSAL_TEXT


def test_g1_stream_delay_ms_admits_carries_the_envelope_on_admit_and_refuse() -> None:
    """The verb returns the same envelope shape on admitted and refused paths.

    A caller reading ``envelope`` from a refused result must see
    the same shape as one reading it from an admitted result, so
    the payload does not switch between two schemas depending on
    the verdict.
    """
    admitted = _call(g1_stream_delay_ms_admits, stream_delay_ms=120)
    refused = _call(g1_stream_delay_ms_admits, stream_delay_ms=-1)
    assert admitted["envelope"].keys() == refused["envelope"].keys()
    assert admitted["envelope"]["stream_delay_ms_min"] == refused["envelope"]["stream_delay_ms_min"]
    assert admitted["envelope"]["stream_delay_ms_max"] == refused["envelope"]["stream_delay_ms_max"]
    assert admitted["envelope"]["stream_delay_ms_neon_default"] == refused["envelope"]["stream_delay_ms_neon_default"]


def test_g1_stream_delay_ms_admits_declares_a_non_handle_first_parameter_type() -> None:
    """The ``stream_delay_ms`` parameter is annotated ``int``, not ``Any``.

    ``Any`` is the annotation the derived
    ``TestEveryLiveHandleVerbRefusesAWrongHandle`` scanner in
    ``tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py``
    keys on to grade a verb as a live-handle verb, and a
    live-handle verb owes an ``{"status": "error"}`` envelope on a
    wrong handle - a shape ``g1_stream_delay_ms_admits`` does not
    owe because its first parameter is a numeric bound, not a
    live driver instance.  The same guard is pinned on the sibling
    envelope-verb tests (e.g. ``g1_volume_admits``,
    ``g1_swing_height_admits``, ``g1_velocity_admits``); this cell
    keeps this port in the same shape.

    This guard reads the annotation off the wrapped function so a
    future widen back to ``Any`` fails this cell first, rather
    than re-entering the live-handle population and tripping the
    scanner a second time.
    """
    import inspect

    undecorated = getattr(g1_stream_delay_ms_admits, "__wrapped__", g1_stream_delay_ms_admits)
    signature = inspect.signature(undecorated)
    parameter = signature.parameters["stream_delay_ms"]
    assert parameter.annotation in ("int", int), (
        f"g1_stream_delay_ms_admits.stream_delay_ms annotation is "
        f"{parameter.annotation!r}; Any would re-enter the "
        f"live-handle-verb population and trip the wrong-handle "
        f"scanner in tests/tools/g1/."
    )
