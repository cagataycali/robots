"""The audio-volume-envelope lookup tools name what ``AudioClient.SetVolume`` admits.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) exposes
``SetVolume(volume)`` without clamps of its own: any integer argument
reaches the audio controller unchanged, and the controller's
behaviour above the neon-bundle-observed usable range is undefined.
The :mod:`strands_robots.tools.g1.g1_audio_volume_envelope` module
snapshots that observed range into module-level constants and exposes
two agent-facing verbs - :func:`g1_list_audio_volume_envelope` (name
the whole envelope) and :func:`g1_volume_admits` (decide one query) -
so a caller can decide the refusal decidably before a future audio
write path is attempted. The tests here fix that contract without
pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen or
narrow to the observed range surfaces here as a shape change rather
than as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The envelope is the neon
  bundle's observed range, not the SDK's own clamps (the SDK has
  none). A driver-side wrapper for ``SetVolume`` that lands later
  will re-check the envelope at wire time and its refusal string
  will surface the same module-local :data:`_REFUSAL_TEXT` the
  admits-verb quotes today.
* The live audio state. Whether an ``AudioClient`` singleton is
  currently constructed, whether ``PlayStream`` holds the wire: those
  are live driver-instance reads and belong on a future audio state
  verb; the envelope surfaces only the numeric bound decision.

One property this file explicitly refuses to pin: the ``7404``
motion-FSM refusal code from
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`. That code is
the driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refusal on ``rt/lowcmd`` writes and its decoded text reads
``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a locomotion
FSM remedy. ``AudioClient`` is on a separate RPC channel and the
audio SDK ships no distinct rc for a bounds-violated volume; the
refusal text this module surfaces is module-local so a planner
reading a volume refusal sees a remedy that matches the surface,
not a re-borrowed motion FSM code. Cells below pin only the
module-local text; a re-borrowing of ``7404`` would fail
``test_the_refusal_text_names_the_volume_envelope_not_the_motion_fsm``.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_audio_volume_envelope import (
    _REFUSAL_TEXT,
    _VOLUME_MAX,
    _VOLUME_MIN,
    g1_list_audio_volume_envelope,
    g1_volume_admits,
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
    runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_audio_volume_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_audio_volume_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """The envelope bounds are integers with min <= max.

    An inverted min/max pair (min > max) would reject every integer
    value; a non-integer bound would let a caller passing 50.5 slip
    through the type-refusal path. Pins the invariant so a widen or
    narrow of the observed range that inverts a pair surfaces here
    rather than as a silently unreachable envelope in production.
    """
    assert isinstance(_VOLUME_MIN, int) and not isinstance(_VOLUME_MIN, bool), (
        f"_VOLUME_MIN is not a plain int: {_VOLUME_MIN!r}"
    )
    assert isinstance(_VOLUME_MAX, int) and not isinstance(_VOLUME_MAX, bool), (
        f"_VOLUME_MAX is not a plain int: {_VOLUME_MAX!r}"
    )
    assert _VOLUME_MIN <= _VOLUME_MAX, (
        f"volume bounds inverted: min={_VOLUME_MIN} > max={_VOLUME_MAX}. g1_volume_admits would refuse every value."
    )


def test_the_envelope_matches_the_neon_observed_range() -> None:
    """The bounds match the neon bundle's field-noted ``0-100`` range.

    The neon bundle's ``use_unitree.py`` names the AudioClient volume
    range as ``SetVolume(volume) / GetVolume() → 0-100`` (matching
    Unitree's own service documentation). Pinning the numbers here
    surfaces a drift in either direction: a widen to ``0-127`` (a
    change to the audio SDK's expected range) or a narrow to
    ``10-90`` (a caller-side field-note correction) would fail this
    cell first.
    """
    assert _VOLUME_MIN == 0
    assert _VOLUME_MAX == 100


def test_the_refusal_text_names_the_volume_envelope_not_the_motion_fsm() -> None:
    """The refusal text is module-local, not a re-borrowed motion FSM code.

    The G1 driver's :meth:`_check_motion_gates` refuses locomotion
    writes with rc=``7404`` whose text reads ``"Invalid FSM id - need
    FSM in {500, 501, 801}"``. ``AudioClient`` is on a separate RPC
    channel from ``rt/lowcmd`` and the audio SDK ships no distinct
    rc for a bounds-violated volume; the refusal shape this module
    surfaces is module-local text that names the volume envelope
    (not the motion FSM) so an agent planner reading a volume
    refusal sees a remedy on the same surface the write belongs on.
    Pinned here so a re-borrowing of ``7404`` (or any other
    motion-FSM entry from ``ERR_CODES``) fails this cell first, not
    as a wrong-remedy surprise in production.
    """
    assert isinstance(_REFUSAL_TEXT, str) and _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT is not a non-empty string: {_REFUSAL_TEXT!r}"
    )
    assert "volume" in _REFUSAL_TEXT.lower(), (
        f"_REFUSAL_TEXT does not name the volume dimension: {_REFUSAL_TEXT!r}. "
        f"A caller reading the refusal must see a remedy on the audio surface."
    )
    fsm_text = ERR_CODES[7404]
    assert _REFUSAL_TEXT != fsm_text, (
        f"_REFUSAL_TEXT re-borrows the motion-FSM ``7404`` text {fsm_text!r}. "
        f"AudioClient is on a separate RPC channel and the audio SDK ships "
        f"no distinct rc for a bounds-violated volume; the refusal shape "
        f"must be module-local so a planner does not read a motion FSM "
        f"remedy for a volume error."
    )
    assert "FSM" not in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT names the motion FSM: {_REFUSAL_TEXT!r}. "
        f"AudioClient is on a separate RPC channel; the refusal remedy "
        f"belongs on the audio surface, not the locomotion FSM."
    )


def test_g1_list_audio_volume_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp and the refusal.

    ``envelope`` carries every clamp constant and ``refusals`` names
    the module-local :data:`_REFUSAL_TEXT` a future driver-side
    ``SetVolume`` wrapper would surface on a bounds violation.
    """
    result = _call(g1_list_audio_volume_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["volume_min"] == _VOLUME_MIN
    assert env["volume_max"] == _VOLUME_MAX
    assert result["refusals"] == [{"text": _REFUSAL_TEXT}]


def test_g1_list_audio_volume_envelope_refusal_omits_a_borrowed_code() -> None:
    """The list-envelope refusal descriptor names no ``code`` field.

    A ``code`` field on this refusal would only be honest if the
    audio SDK shipped a distinct rc for a bounds-violated volume,
    and it does not; borrowing a motion-FSM code from
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` would hand
    a planner a wrong-surface remedy. Pins the omission so a future
    re-introduction of ``code`` fails this cell first.
    """
    result = _call(g1_list_audio_volume_envelope)
    for refusal in result["refusals"]:
        assert "code" not in refusal, (
            f"refusal descriptor carries a ``code`` field: {refusal!r}. "
            f"The audio SDK ships no rc for a bounds-violated volume; "
            f"borrowing one from ERR_CODES puts a wrong-surface remedy on "
            f"an audio refusal."
        )


def test_g1_list_audio_volume_envelope_returns_fresh_containers() -> None:
    """Successive calls do not share the envelope dict or refusals list.

    A caller mutating one call's ``envelope`` (or ``refusals``) must
    not affect the next call's payload. Pins the isolation so a
    mutation-at-callsite bug lands here rather than as a
    ghost-state-across-calls regression in production.
    """
    first = _call(g1_list_audio_volume_envelope)
    second = _call(g1_list_audio_volume_envelope)
    assert first is not second
    assert first["envelope"] is not second["envelope"]
    assert first["refusals"] is not second["refusals"]


def test_g1_volume_admits_a_value_inside_the_envelope() -> None:
    """A value inside the envelope is admitted with an empty refusals list.

    ``volume=50`` sits at the middle of the observed range, so a
    driver-side wrapper for ``SetVolume`` would not refuse it on
    envelope grounds (whether the audio bus is currently free is a
    separate live-read decision the verb does not answer).
    """
    result = _call(g1_volume_admits, volume=50)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


@pytest.mark.parametrize("volume", [0, 100])
def test_g1_volume_admits_at_the_exact_clamp_boundaries(volume: int) -> None:
    """A value at either clamp boundary is inside, not outside.

    Boundary values ``volume=0`` (mute) and ``volume=100`` (full) are
    admitted because :func:`g1_volume_admits` refuses on ``value <
    bound`` and ``value > bound`` rather than ``<=``/``>= bound`` -
    a mute is a legitimate command and off-by-one at the boundary
    would silently reject two valid saturated values.
    """
    result = _call(g1_volume_admits, volume=volume)
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_volume_admits_refuses_a_value_below_the_floor() -> None:
    """A ``volume`` below ``volume_min`` reads as one refusal.

    The refusal descriptor names the dimension, the value the caller
    passed, the bound-key on the envelope it violated, and the
    module-local :data:`_REFUSAL_TEXT` a driver-side wrapper would
    quote.
    """
    result = _call(g1_volume_admits, volume=-1)
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "volume"
    assert refusal["value"] == -1
    assert refusal["bound_key"] == "volume_min"
    assert refusal["bound"] == _VOLUME_MIN
    assert refusal["comparison"] == "value < bound"
    assert refusal["text"] == _REFUSAL_TEXT
    assert "code" not in refusal, (
        f"below-floor refusal carries a ``code`` field: {refusal!r}. "
        f"The audio SDK ships no rc for a bounds-violated volume."
    )


def test_g1_volume_admits_refuses_a_value_above_the_ceiling() -> None:
    """A ``volume`` above ``volume_max`` reads as one refusal.

    The neon bundle observed ``100`` as maximum; a caller passing
    ``150`` learns the ceiling and the module-local refusal text
    before a driver-side wrapper fires, rather than learning at
    wire time only that the audio controller's clipping behaviour
    above the observed range is undefined.
    """
    result = _call(g1_volume_admits, volume=150)
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "volume"
    assert refusal["value"] == 150
    assert refusal["bound_key"] == "volume_max"
    assert refusal["bound"] == _VOLUME_MAX
    assert refusal["comparison"] == "value > bound"
    assert refusal["text"] == _REFUSAL_TEXT
    assert "code" not in refusal


def test_g1_volume_admits_refuses_a_bool_volume() -> None:
    """A ``bool`` ``volume`` reads as refused with a ``non-int`` comparison.

    Python's ``bool`` is a subclass of ``int``, so ``True`` would
    otherwise silently look up ``1`` (a legitimate low volume) and
    hide the type mistake behind an admitted-with-quiet-speaker
    result. Refusing at the boundary surfaces the mistake instead.
    """
    for value in (True, False):
        result = _call(g1_volume_admits, volume=value)
        assert result["admits"] is False, f"bool {value!r} was admitted as a volume"
        refusal = result["refusals"][0]
        assert refusal["dimension"] == "volume"
        assert refusal["comparison"] == "non-int"
        assert refusal["text"] == _REFUSAL_TEXT


@pytest.mark.parametrize(
    "volume",
    [50.0, 50.5, Decimal("50"), "50", None, [], (50,)],
)
def test_g1_volume_admits_refuses_a_non_int_volume(volume: Any) -> None:
    """A non-int-non-bool ``volume`` reads as refused with a ``non-int`` comparison.

    ``float`` values (even integer-valued like ``50.0``) are refused
    because the SDK's ``SetVolume`` expects an integer; a caller
    passing ``50.0`` learns the shape mistake here rather than at
    wire time via a silent truncation. ``Decimal`` and ``str`` follow
    the same rule (a str is refused rather than parsed - the verb
    does not fabricate a value the caller did not supply).
    """
    result = _call(g1_volume_admits, volume=volume)
    assert result["admits"] is False, f"{volume!r} was admitted as a volume"
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "volume"
    assert refusal["comparison"] == "non-int"
    assert refusal["text"] == _REFUSAL_TEXT


def test_g1_volume_admits_carries_the_envelope_on_admit_and_refuse() -> None:
    """The verb returns the same envelope shape on admitted and refused paths.

    A caller reading ``envelope`` from a refused result must see the
    same shape as one reading it from an admitted result, so the
    payload does not switch between two schemas depending on the
    verdict.
    """
    admitted = _call(g1_volume_admits, volume=50)
    refused = _call(g1_volume_admits, volume=-1)
    assert admitted["envelope"].keys() == refused["envelope"].keys()
    assert admitted["envelope"]["volume_min"] == refused["envelope"]["volume_min"]
    assert admitted["envelope"]["volume_max"] == refused["envelope"]["volume_max"]


def test_g1_volume_admits_declares_a_non_handle_first_parameter_type() -> None:
    """The ``volume`` parameter is annotated ``int``, not ``Any``.

    ``Any`` is the annotation the derived
    ``TestEveryLiveHandleVerbRefusesAWrongHandle`` scanner in
    ``tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py``
    keys on to grade a verb as a live-handle verb, and a live-handle
    verb owes an ``{"status": "error"}`` envelope on a wrong handle -
    a shape ``g1_volume_admits`` does not owe because its first
    parameter is a numeric bound, not a live driver instance.  The
    reason the wrong-handle scan flagged the verb in a prior fire
    was that ``volume`` shipped as ``Any``; naming ``int`` here fixes
    the wrong classification without weakening the numeric refusal
    (bool, float, str and other non-int shapes are still refused with
    ``admits=False`` and a ``non-int`` comparison, pinned by the
    parametric refusal tests above).

    This guard reads the annotation off the wrapped function so a
    future widen back to ``Any`` fails this cell first, rather than
    re-entering the live-handle population and tripping the scanner
    a second time.
    """
    import inspect

    # Reach the undecorated function through ``getattr`` with a fallback:
    # ``@tool`` returns a ``DecoratedFunctionTool`` that does not declare
    # ``__wrapped__`` statically, and this is the repo-wide convention for
    # reading a tool's real signature (see tests/tools/test_use_lerobot.py,
    # the driver test modules) - resilient to the SDK renaming the
    # attribute, and type-clean without a blanket ignore.
    undecorated = getattr(g1_volume_admits, "__wrapped__", g1_volume_admits)
    signature = inspect.signature(undecorated)
    volume_parameter = signature.parameters["volume"]
    assert volume_parameter.annotation in ("int", int), (
        f"g1_volume_admits.volume annotation is {volume_parameter.annotation!r}; "
        "Any would re-enter the live-handle-verb population and trip the "
        "wrong-handle scanner in tests/tools/g1/."
    )
