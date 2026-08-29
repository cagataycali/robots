"""The stand-height-envelope lookup tools name what ``LocoClient.SetStandHeight`` walks.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetStandHeight(height_m)`` and its ``HighStand()`` cousin without
clamps of its own: any finite argument reaches the controller
unchanged, and the controller's behaviour above the neon-bundle-
observed walkable range is undefined. The
:mod:`strands_robots.tools.g1.g1_stand_height_envelope` module
snapshots that observed range into module-level constants and exposes
two agent-facing verbs - :func:`g1_list_stand_height_envelope` (name
the whole envelope) and :func:`g1_stand_height_admits` (decide one
query) - so a caller can decide the refusal decidably before a future
locomotion write path is attempted. The tests here fix that contract
without pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen or
narrow to the observed range surfaces here as a shape change rather
than as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The envelope is the neon
  bundle's observed range, not the SDK's own clamps (the SDK has
  none). A driver-side wrapper for ``SetStandHeight`` that lands
  later will re-check the envelope at wire time and its refusal
  string will quote the ``7404`` gate-refusal code the driver's
  ``_check_motion_gates`` also quotes.
* Whether the driver's live ``fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  live driver-instance read and belongs on
  :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; the verb surfaces
  the set as a snapshot so a caller comparing an intended write
  against both conditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_stand_height_envelope import (
    _GATE_REFUSAL_CODE,
    _HIGH_STAND_ROUTE_THRESHOLD,
    _HIGH_STAND_SENTINEL,
    _STAND_HEIGHT_MAX,
    _STAND_HEIGHT_MIN,
    g1_list_stand_height_envelope,
    g1_stand_height_admits,
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
    importlib.import_module("strands_robots.tools.g1.g1_stand_height_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_stand_height_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every clamp is a finite float and the min/max pair is ordered.

    A non-finite bound would let ``g1_stand_height_admits`` admit
    every value on that dimension; an inverted min/max pair
    (min > max) would reject every finite value. Pins the invariant so
    a widen or narrow of the observed range that inverts the pair
    surfaces here rather than as a silently unreachable envelope in
    production.
    """
    for name, value in (
        ("_STAND_HEIGHT_MIN", _STAND_HEIGHT_MIN),
        ("_STAND_HEIGHT_MAX", _STAND_HEIGHT_MAX),
        ("_HIGH_STAND_ROUTE_THRESHOLD", _HIGH_STAND_ROUTE_THRESHOLD),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _STAND_HEIGHT_MIN <= _STAND_HEIGHT_MAX, (
        f"stand-height bounds inverted: min={_STAND_HEIGHT_MIN} > "
        f"max={_STAND_HEIGHT_MAX}. g1_stand_height_admits would refuse "
        f"every non-negative height."
    )


def test_the_high_stand_sentinel_matches_the_sdk_convention() -> None:
    """The HighStand sentinel is ``UINT32_MAX`` as the SDK carries it.

    The Unitree SDK's ``HighStand`` method dispatches by passing
    ``UINT32_MAX`` (``2**32 - 1`` = 4294967295) through the same
    ``SetStandHeight`` handler. This envelope quotes the same
    integer so a caller comparing this lookup's answer to the SDK's
    own constant sees the same number on both sides. Pinned here
    because a firmware release that renumbered the sentinel would
    require a matched update in both the SDK and this snapshot.
    """
    assert _HIGH_STAND_SENTINEL == 2**32 - 1
    assert _HIGH_STAND_SENTINEL == 4294967295


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side stand-height wrapper would surface. Pinned here
    so a re-wording of that message lands in one place, not one in
    the driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_stand_height_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp, the gate set, and the refusal.

    ``envelope`` carries every clamp constant plus the HighStand
    sentinel and route threshold, ``walk_ready_fsm_ids`` quotes
    :data:`WALK_FSMS`, and ``refusals`` names the ``7404``
    gate-refusal code with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_stand_height_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["stand_height_min"] == _STAND_HEIGHT_MIN
    assert env["stand_height_max"] == _STAND_HEIGHT_MAX
    assert env["high_stand_route_threshold"] == _HIGH_STAND_ROUTE_THRESHOLD
    assert env["high_stand_sentinel"] == _HIGH_STAND_SENTINEL
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_stand_height_admits_a_value_inside_the_set_stand_height_range() -> None:
    """A non-negative height inside the range is admitted on the SetStandHeight route.

    The identity case (``height=0.5``, halfway up the envelope) sits
    strictly inside ``[stand_height_min, stand_height_max]``, so a
    driver-side wrapper for ``SetStandHeight`` would not refuse it on
    envelope grounds (whether the FSM gate admits it is a separate
    live-read decision the verb does not answer). The ``route``
    fields names ``"set_stand_height"`` to distinguish the two SDK
    dispatch paths this envelope covers.
    """
    result = _call(g1_stand_height_admits, height=0.5)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "set_stand_height"
    assert result["refusals"] == []
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_stand_height_admits_at_the_exact_clamp_boundaries() -> None:
    """A height at the clamp boundaries is inside, not outside.

    Boundary values ``height == stand_height_min`` and
    ``height == stand_height_max`` are admitted because
    :func:`g1_stand_height_admits` refuses on ``value > bound`` /
    ``value < bound`` rather than ``>= bound`` / ``<= bound`` (the
    neon bundle's docstring records ``0.0`` as LOW-stand and
    ``~0.8`` as walkable ceiling, and off-by-one at the boundary
    would silently reject a saturated command a caller intended
    to send).
    """
    lo = _call(g1_stand_height_admits, height=_STAND_HEIGHT_MIN)
    hi = _call(g1_stand_height_admits, height=_STAND_HEIGHT_MAX)
    assert lo["admits"] is True
    assert lo["route"] == "set_stand_height"
    assert lo["refusals"] == []
    assert hi["admits"] is True
    assert hi["route"] == "set_stand_height"
    assert hi["refusals"] == []


def test_g1_stand_height_admits_a_strictly_negative_height_as_high_stand() -> None:
    """Any strictly-negative height routes to HighStand with admits=True.

    The neon bundle's ``g1_set_stand_height`` verb uses
    ``if height < 0:`` to route to ``LocoClient.HighStand()``; every
    strictly-negative float lands on that branch. The SDK's HighStand
    reads only the sentinel it passes internally, not the caller's
    own magnitude, so ``-1.0`` and ``-1e-300`` both route to the same
    dispatch call. The ``route`` field names ``"high_stand"`` so a
    caller distinguishes the two SDK paths.
    """
    for h in (-1.0, -0.5, -1e-300, -1e10):
        result = _call(g1_stand_height_admits, height=h)
        assert result["admits"] is True, f"height={h!r} should be admitted"
        assert result["route"] == "high_stand", f"height={h!r} should route to HighStand"
        assert result["refusals"] == []


def test_g1_stand_height_admits_treats_negative_zero_as_low_stand_not_high_stand() -> None:
    """``-0.0`` routes as LOW-stand, matching the neon bundle's conditional.

    Python's ``-0.0 < 0.0`` is ``False`` (IEEE-754 comparison ignores
    the sign bit on zero), so the neon bundle's ``if height < 0``
    reads ``-0.0`` as non-negative and routes it to
    ``SetStandHeight(-0.0)``. This envelope preserves that
    convention: a caller who wants HighStand must pass a
    strictly-negative sentinel (any ``-x`` with ``x > 0``), not a
    value near zero.
    """
    result = _call(g1_stand_height_admits, height=-0.0)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "set_stand_height", (
        "-0.0 must route to SetStandHeight (not HighStand) to match the "
        "neon bundle's `if height < 0` conditional, which reads -0.0 as "
        "non-negative"
    )
    assert result["refusals"] == []


def test_g1_stand_height_admits_a_height_above_the_ceiling() -> None:
    """A height above ``stand_height_max`` refuses on that bound.

    The refusal descriptor names ``dimension="height"``, the offending
    value, the bound it violated, and the ``7404`` gate-refusal
    code. ``route`` is ``None`` because a rejected value would not
    reach either SDK dispatch path at wire time.
    """
    over = _STAND_HEIGHT_MAX + 0.1
    result = _call(g1_stand_height_admits, height=over)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "height"
    assert r["value"] == over
    assert r["bound_key"] == "stand_height_max"
    assert r["bound"] == _STAND_HEIGHT_MAX
    assert r["comparison"] == "value > bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


@pytest.mark.parametrize("bad_height", [math.inf, -math.inf, math.nan])
def test_g1_stand_height_admits_refuses_non_finite_input(bad_height: float) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` refuse with ``comparison="non-finite"``.

    A NaN cannot be routed decidably (``nan < 0`` is ``False`` but
    ``nan >= 0`` is also ``False``, so neither branch admits it), and
    an infinity would either overrun ``stand_height_max`` or route as
    an unbounded HighStand sentinel - both are shape violations
    rather than value ones. Named on the refusal descriptor so a
    caller distinguishes a bounds violation from a shape violation.
    """
    result = _call(g1_stand_height_admits, height=bad_height)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "height"
    assert r["comparison"] == "non-finite"
    assert r["code"] == _GATE_REFUSAL_CODE


def test_g1_stand_height_admits_default_call_is_low_stand() -> None:
    """Calling with no arguments admits as the LOW-stand pose.

    The verb's default ``height=0.0`` sits at ``stand_height_min``,
    so a zero-arg invocation ("what would the safest, most-conservative
    stand height do?") returns ``admits=True, route="set_stand_height"``
    with an empty refusals list. Pinned so a change to the default
    that silently shifted the safest-query answer surfaces here.
    """
    result = _call(g1_stand_height_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "set_stand_height"
    assert result["refusals"] == []
