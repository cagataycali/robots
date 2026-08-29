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
  will quote the ``7404`` gate-refusal code the driver's
  ``_check_motion_gates`` also quotes.
* The live audio state. Whether an ``AudioClient`` singleton is
  currently constructed, whether ``PlayStream`` holds the wire: those
  are live driver-instance reads and belong on a future audio state
  verb; the envelope surfaces only the numeric bound decision.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_audio_volume_envelope import (
    _GATE_REFUSAL_CODE,
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


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses SDK-shaped writes
    with rc=7404, and the ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side volume wrapper would surface. Pinned here so
    a re-wording of that message lands in one place, not one in the
    driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_audio_volume_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp and the refusal.

    ``envelope`` carries every clamp constant and ``refusals`` names
    the ``7404`` gate-refusal code with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_audio_volume_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["volume_min"] == _VOLUME_MIN
    assert env["volume_max"] == _VOLUME_MAX
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


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
    ``7404`` code a driver-side wrapper would quote.
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
    assert refusal["code"] == _GATE_REFUSAL_CODE
    assert refusal["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_volume_admits_refuses_a_value_above_the_ceiling() -> None:
    """A ``volume`` above ``volume_max`` reads as one refusal.

    The neon bundle observed ``100`` as maximum; a caller passing
    ``150`` learns the ceiling and the ``7404`` refusal code before
    a driver-side wrapper fires, rather than learning at wire time
    only that the audio controller's clipping behaviour above the
    observed range is undefined.
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
