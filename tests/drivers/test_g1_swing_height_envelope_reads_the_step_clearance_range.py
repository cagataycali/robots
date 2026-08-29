"""The swing-height-envelope lookup tools name what the neon ``_Call(7103)`` clamps.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) does not
expose a public ``SetSwingHeight`` method; the setter is reachable
only through the SDK's raw ``_Call`` on API id ``7103``, which the
neon bundle
(``cagataycali/neon-the-g1/tools/_g1_common.py::set_swing_height``)
fronts under a single-writer lock and narrows to
``max(0.0, min(0.2, float(height)))`` before dispatch. The
:mod:`strands_robots.tools.g1.g1_swing_height_envelope` module
snapshots that clamp pair (plus the neon-bundle-documented
recommended interval of ``0.05-0.15`` m and the ``7103`` API id) into
module-level constants and exposes two agent-facing verbs -
:func:`g1_list_swing_height_envelope` (name the whole envelope) and
:func:`g1_swing_height_admits` (decide one query) - so a caller can
decide the refusal decidably before a future driver-side wrapper
fires. The tests here fix that contract without pulling the SDK: the
module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
observed clamps surfaces here as a shape change rather than as a
diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The envelope is the neon
  bundle's observed hard clamp, not the SDK's own bounds (the SDK
  has none; the ``7103`` handler accepts any finite argument). A
  driver-side wrapper for the swing-height setter that lands later
  will re-check the envelope at wire time and its refusal string
  will quote the ``7404`` gate-refusal code the driver's
  ``_check_motion_gates`` also quotes.
* Whether the driver's live ``fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  live driver-instance read and belongs on
  :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; the verb
  surfaces the set as a snapshot so a caller comparing an intended
  write against both conditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_swing_height_envelope import (
    _GATE_REFUSAL_CODE,
    _SWING_HEIGHT_API_ID,
    _SWING_HEIGHT_MAX,
    _SWING_HEIGHT_MIN,
    _SWING_HEIGHT_RECOMMENDED_MAX,
    _SWING_HEIGHT_RECOMMENDED_MIN,
    g1_list_swing_height_envelope,
    g1_swing_height_admits,
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
    importable with the SDK absent (refs strands-labs/robots#358); a
    module that pulled a submodule at import time would break every
    headless CI runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_swing_height_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_swing_height_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every clamp is a finite float and the min/max pairs are ordered.

    A non-finite bound would let :func:`g1_swing_height_admits`
    admit every value on that dimension; an inverted min/max pair
    (min > max) would reject every finite value. Both the hard
    clamp and the recommendation interval are checked so a widen or
    narrow that inverts either pair surfaces here rather than as a
    silently unreachable envelope in production.
    """
    for name, value in (
        ("_SWING_HEIGHT_MIN", _SWING_HEIGHT_MIN),
        ("_SWING_HEIGHT_MAX", _SWING_HEIGHT_MAX),
        ("_SWING_HEIGHT_RECOMMENDED_MIN", _SWING_HEIGHT_RECOMMENDED_MIN),
        ("_SWING_HEIGHT_RECOMMENDED_MAX", _SWING_HEIGHT_RECOMMENDED_MAX),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _SWING_HEIGHT_MIN <= _SWING_HEIGHT_MAX, (
        f"swing-height hard-clamp inverted: min={_SWING_HEIGHT_MIN} > "
        f"max={_SWING_HEIGHT_MAX}. g1_swing_height_admits would refuse "
        f"every finite height."
    )
    assert _SWING_HEIGHT_RECOMMENDED_MIN <= _SWING_HEIGHT_RECOMMENDED_MAX, (
        f"swing-height recommendation inverted: "
        f"min={_SWING_HEIGHT_RECOMMENDED_MIN} > "
        f"max={_SWING_HEIGHT_RECOMMENDED_MAX}."
    )


def test_the_recommended_interval_sits_inside_the_hard_clamp() -> None:
    """The recommendation interval is a subset of the hard clamp.

    A recommended-min below :data:`_SWING_HEIGHT_MIN` or a
    recommended-max above :data:`_SWING_HEIGHT_MAX` would let a
    caller who reads ``inside_recommended`` land on a value the
    hard-clamp refuses at admission time - contradicting the
    "recommendation implies admitted" invariant the returned
    envelope names. Pinned so a widen of one range without a
    matched widen of the other surfaces here.
    """
    assert _SWING_HEIGHT_MIN <= _SWING_HEIGHT_RECOMMENDED_MIN, (
        f"recommendation floor {_SWING_HEIGHT_RECOMMENDED_MIN} sits below "
        f"the hard-clamp floor {_SWING_HEIGHT_MIN}; a value inside the "
        f"recommendation but below the clamp would refuse at admission."
    )
    assert _SWING_HEIGHT_RECOMMENDED_MAX <= _SWING_HEIGHT_MAX, (
        f"recommendation ceiling {_SWING_HEIGHT_RECOMMENDED_MAX} sits "
        f"above the hard-clamp ceiling {_SWING_HEIGHT_MAX}; a value "
        f"inside the recommendation but above the clamp would refuse."
    )


def test_the_swing_height_api_id_matches_the_neon_bundle() -> None:
    """The envelope quotes API id ``7103`` as the neon bundle does.

    The Unitree SDK does not expose a public ``SetSwingHeight``
    method; the setter is reachable only through the raw
    ``_Call(7103, ...)`` path the neon bundle fronts. This envelope
    quotes the same integer so a caller comparing this lookup's
    answer to the neon bundle's own dispatch sees the same number
    on both sides. Pinned here because a firmware release that
    renumbered the setter would require a matched update in both
    the neon bundle and this snapshot.
    """
    assert _SWING_HEIGHT_API_ID == 7103


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries
    the text a driver-side swing-height wrapper would surface.
    Pinned here so a re-wording of that message lands in one place,
    not one in the driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_swing_height_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp, the recommendation, the api id, and the refusal.

    ``envelope`` carries every clamp constant plus the recommended
    interval and the ``7103`` API id, ``walk_ready_fsm_ids`` quotes
    :data:`WALK_FSMS`, and ``refusals`` names the ``7404``
    gate-refusal code with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_swing_height_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["swing_height_min"] == _SWING_HEIGHT_MIN
    assert env["swing_height_max"] == _SWING_HEIGHT_MAX
    assert env["swing_height_recommended_min"] == _SWING_HEIGHT_RECOMMENDED_MIN
    assert env["swing_height_recommended_max"] == _SWING_HEIGHT_RECOMMENDED_MAX
    assert env["swing_height_api_id"] == _SWING_HEIGHT_API_ID
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_swing_height_admits_a_value_inside_the_recommended_interval() -> None:
    """A height inside the recommendation is admitted with inside_recommended=True.

    The identity case (``height=0.1`` m, the centre of the neon
    bundle's documented "Typical safe range: 0.05-0.15 m") sits
    strictly inside both the hard clamp and the recommendation, so
    a driver-side wrapper would not refuse it on envelope grounds
    (whether the FSM gate admits it is a separate live-read
    decision the verb does not answer). ``route`` names
    ``"call_7103"`` to identify the SDK dispatch path.
    """
    result = _call(g1_swing_height_admits, height=0.1)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "call_7103"
    assert result["inside_recommended"] is True
    assert result["refusals"] == []
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_swing_height_admits_at_the_exact_hard_clamp_boundaries() -> None:
    """A height at the hard-clamp boundaries is inside, not outside.

    Boundary values ``height == swing_height_min`` (``0.0``) and
    ``height == swing_height_max`` (``0.2``) are admitted because
    :func:`g1_swing_height_admits` refuses on ``value > bound`` /
    ``value < bound`` rather than ``>= bound`` / ``<= bound`` (the
    neon bundle's clamp is ``max(0.0, min(0.2, height))``, which
    dispatches the boundary value unchanged; off-by-one at the
    boundary would silently reject a saturated command a caller
    intended to send). ``inside_recommended`` is ``False`` at the
    hard-clamp boundary because those values sit outside the
    softer recommendation interval.
    """
    lo = _call(g1_swing_height_admits, height=_SWING_HEIGHT_MIN)
    hi = _call(g1_swing_height_admits, height=_SWING_HEIGHT_MAX)
    assert lo["admits"] is True
    assert lo["route"] == "call_7103"
    assert lo["refusals"] == []
    assert lo["inside_recommended"] is False
    assert hi["admits"] is True
    assert hi["route"] == "call_7103"
    assert hi["refusals"] == []
    assert hi["inside_recommended"] is False


def test_g1_swing_height_admits_at_the_recommendation_boundaries() -> None:
    """Values at the recommendation boundaries are inside the recommendation.

    ``inside_recommended`` reads ``_SWING_HEIGHT_RECOMMENDED_MIN <=
    h <= _SWING_HEIGHT_RECOMMENDED_MAX`` (inclusive on both sides)
    so the boundary values themselves are inside the recommendation.
    Pinned because an off-by-one that turned this into a strict
    interval would drop the neon bundle's own documented range at
    its endpoints.
    """
    lo = _call(g1_swing_height_admits, height=_SWING_HEIGHT_RECOMMENDED_MIN)
    hi = _call(g1_swing_height_admits, height=_SWING_HEIGHT_RECOMMENDED_MAX)
    assert lo["admits"] is True
    assert lo["inside_recommended"] is True
    assert hi["admits"] is True
    assert hi["inside_recommended"] is True


def test_g1_swing_height_admits_a_value_below_the_floor() -> None:
    """A strictly-negative height refuses on ``swing_height_min``.

    The refusal descriptor names ``dimension="height"``, the
    offending value, the bound it violated
    (``bound_key="swing_height_min"``), the comparison
    (``"value < bound"``), and the ``7404`` gate-refusal code.
    ``route`` is ``None`` because a rejected value would not reach
    the ``_Call(7103)`` dispatch path at wire time.
    """
    under = -0.05
    result = _call(g1_swing_height_admits, height=under)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert result["inside_recommended"] is False
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "height"
    assert r["value"] == under
    assert r["bound_key"] == "swing_height_min"
    assert r["bound"] == _SWING_HEIGHT_MIN
    assert r["comparison"] == "value < bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_swing_height_admits_a_value_above_the_ceiling() -> None:
    """A height above ``swing_height_max`` refuses on that bound.

    The refusal descriptor names ``dimension="height"``, the
    offending value, the bound it violated
    (``bound_key="swing_height_max"``), the comparison
    (``"value > bound"``), and the ``7404`` gate-refusal code.
    ``route`` is ``None`` because a rejected value would not reach
    the ``_Call(7103)`` dispatch path at wire time.
    """
    over = _SWING_HEIGHT_MAX + 0.1
    result = _call(g1_swing_height_admits, height=over)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert result["inside_recommended"] is False
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "height"
    assert r["value"] == over
    assert r["bound_key"] == "swing_height_max"
    assert r["bound"] == _SWING_HEIGHT_MAX
    assert r["comparison"] == "value > bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_swing_height_admits_a_value_between_the_clamp_and_the_recommendation() -> None:
    """A value inside the hard clamp but outside the recommendation admits with inside_recommended=False.

    A caller who reads only ``admits`` sees the same answer the
    neon wrapper would give at dispatch time (the value is inside
    the hard clamp, so the wrapper does not refuse it). A caller
    who also reads ``inside_recommended`` sees that the value
    sits outside the neon bundle's documented safety range and
    can decide the softer refusal itself. Pinned because folding
    the recommendation into the admits decision would drop the
    neon bundle's own admitted range at the ``[0.0, 0.05)`` and
    ``(0.15, 0.2]`` sub-intervals.
    """
    # 0.02 sits inside the hard clamp [0.0, 0.2] but below the
    # recommendation floor 0.05.
    result = _call(g1_swing_height_admits, height=0.02)
    assert result["admits"] is True
    assert result["route"] == "call_7103"
    assert result["inside_recommended"] is False
    assert result["refusals"] == []

    # 0.18 sits inside the hard clamp but above the recommendation
    # ceiling 0.15.
    result = _call(g1_swing_height_admits, height=0.18)
    assert result["admits"] is True
    assert result["route"] == "call_7103"
    assert result["inside_recommended"] is False
    assert result["refusals"] == []


@pytest.mark.parametrize("bad_height", [math.inf, -math.inf, math.nan])
def test_g1_swing_height_admits_refuses_non_finite_input(bad_height: float) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` refuse with ``comparison="non-finite"``.

    A NaN cannot be compared decidably (``nan < 0`` is ``False``
    but ``nan > 0.2`` is also ``False``), and an infinity would
    either overrun ``swing_height_max`` or underrun
    ``swing_height_min`` - both are shape violations rather than
    value ones. Named on the refusal descriptor so a caller
    distinguishes a bounds violation from a shape violation.
    """
    result = _call(g1_swing_height_admits, height=bad_height)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert result["inside_recommended"] is False
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "height"
    assert r["comparison"] == "non-finite"
    assert r["code"] == _GATE_REFUSAL_CODE


def test_g1_swing_height_admits_default_call_is_the_recommendation_centre() -> None:
    """Calling with no arguments admits at the centre of the recommendation.

    The verb's default ``height=0.1`` sits at the midpoint of the
    neon bundle's ``0.05-0.15`` m safe range, so a zero-arg
    invocation ("what would the safest, most-typical swing height
    do?") returns ``admits=True, route="call_7103",
    inside_recommended=True`` with an empty refusals list. Pinned
    so a change to the default that silently shifted the
    safest-query answer surfaces here.
    """
    result = _call(g1_swing_height_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "call_7103"
    assert result["inside_recommended"] is True
    assert result["refusals"] == []
