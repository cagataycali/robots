"""The locomotion-duration-envelope lookup tools name what neon ``g1_move_velocity`` clamps and refuses.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration_s)`` as the bounded write for a
velocity command that terminates on its own after ``duration_s``
seconds, and ``Move(vx, vy, vyaw, continous_move=True)`` as the
unbounded cousin the neon bundle
(``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_move_velocity``)
routes to on ``continuous=True``. The SDK itself places *no* clamps
on the ``duration_s`` argument; the neon wrapper narrows it to
``max(0.0, min(10.0, float(duration)))`` before dispatch and then
refuses ``duration <= 0`` outright with the message
``"duration<=0 (non-continuous), refusing"``. The
:mod:`strands_robots.tools.g1.g1_locomotion_duration_envelope`
module snapshots that clamp pair plus the two SDK method names into
module-level constants and exposes two agent-facing verbs -
:func:`g1_list_locomotion_duration_envelope` (name the whole
envelope) and :func:`g1_locomotion_duration_admits` (decide one
query) - so a caller can decide the refusal decidably before a
future driver-side wrapper fires. The tests here fix that contract
without pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off
the module's own snapshot rather than restated in the tests, so a
widen or narrow to the observed clamps surfaces here as a shape
change rather than as a diverging table this file would need to
manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The envelope is the neon
  bundle's observed hard clamp, not the SDK's own bounds (the SDK
  has none; the ``SetVelocity`` handler accepts any finite
  argument). A driver-side wrapper for the velocity setter that
  lands later will re-check the envelope at wire time and its
  refusal string will quote the ``7404`` gate-refusal code the
  driver's ``_check_motion_gates`` also quotes.
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
from strands_robots.tools.g1.g1_locomotion_duration_envelope import (
    _GATE_REFUSAL_CODE,
    _LOCOMOTION_CONTINUOUS_MOVE_METHOD,
    _LOCOMOTION_DURATION_MAX,
    _LOCOMOTION_DURATION_MIN,
    _LOCOMOTION_SET_VELOCITY_METHOD,
    g1_list_locomotion_duration_envelope,
    g1_locomotion_duration_admits,
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
    importlib.import_module("strands_robots.tools.g1.g1_locomotion_duration_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_locomotion_duration_envelope imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every clamp is a finite float and the min/max pair is ordered.

    A non-finite bound would let
    :func:`g1_locomotion_duration_admits` admit every value; an
    inverted ``min > max`` pair would reject every finite value.
    Pinned so a widen or narrow that inverts the pair surfaces
    here rather than as a silently unreachable envelope in
    production.
    """
    for name, value in (
        ("_LOCOMOTION_DURATION_MIN", _LOCOMOTION_DURATION_MIN),
        ("_LOCOMOTION_DURATION_MAX", _LOCOMOTION_DURATION_MAX),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"
    assert _LOCOMOTION_DURATION_MIN < _LOCOMOTION_DURATION_MAX, (
        f"locomotion-duration clamp inverted or degenerate: "
        f"min={_LOCOMOTION_DURATION_MIN} >= max={_LOCOMOTION_DURATION_MAX}. "
        f"g1_locomotion_duration_admits would refuse every finite duration."
    )


def test_the_neon_bundle_bounds_are_the_ones_the_wrapper_dispatches() -> None:
    """The clamps match the neon bundle's own ``max(0.0, min(10.0, ...))`` expression.

    The neon bundle's ``g1_move_velocity`` verb runs
    ``duration = max(0.0, min(10.0, float(duration)))`` before its
    ``if duration <= 0`` refusal. Pinned as the exact numeric pair
    the wrapper dispatches, so a revision on the neon side that
    widens the wall-clock cap without a matched update here
    surfaces as a test drift rather than as a silently-diverging
    admission set.
    """
    assert _LOCOMOTION_DURATION_MIN == 0.0
    assert _LOCOMOTION_DURATION_MAX == 10.0


def test_the_sdk_method_names_match_the_neon_bundle_dispatches() -> None:
    """The envelope quotes ``SetVelocity`` and ``Move`` as the neon bundle does.

    The neon bundle's ``g1_move_velocity`` routes the
    bounded-duration branch to ``LocoClient.SetVelocity`` and the
    unbounded branch to ``LocoClient.Move(..., continous_move=True)``.
    Pinned here because a firmware release that renamed either
    method would require a matched update in both the neon bundle
    and this snapshot; the two names travel together.
    """
    assert _LOCOMOTION_SET_VELOCITY_METHOD == "SetVelocity"
    assert _LOCOMOTION_CONTINUOUS_MOVE_METHOD == "Move"


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a
    locomotion-shaped write on an FSM outside :data:`WALK_FSMS`
    with rc=7404, and the ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries
    the text a driver-side velocity wrapper would surface. Pinned
    here so a re-wording of that message lands in one place, not
    one in the driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_locomotion_duration_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp, the two SDK methods, and the refusal.

    ``envelope`` carries every clamp constant plus the two SDK
    method names, ``walk_ready_fsm_ids`` quotes :data:`WALK_FSMS`,
    and ``refusals`` names the ``7404`` gate-refusal code with the
    decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_locomotion_duration_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["duration_min"] == _LOCOMOTION_DURATION_MIN
    assert env["duration_max"] == _LOCOMOTION_DURATION_MAX
    assert env["set_velocity_method"] == _LOCOMOTION_SET_VELOCITY_METHOD
    assert env["continuous_move_method"] == _LOCOMOTION_CONTINUOUS_MOVE_METHOD
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_locomotion_duration_admits_a_typical_value() -> None:
    """A duration in the middle of the clamp range is admitted.

    The identity case (``duration=1.0`` s, the neon bundle's own
    default for ``g1_move_velocity`` documented as "single 1-second
    impulse for safety") sits strictly inside the hard clamp, so a
    driver-side wrapper would not refuse it on envelope grounds
    (whether the FSM gate admits it is a separate live-read
    decision the verb does not answer). ``route`` names
    ``"set_velocity"`` to identify the SDK dispatch path.
    """
    result = _call(g1_locomotion_duration_admits, duration=1.0)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "set_velocity"
    assert result["refusals"] == []
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_locomotion_duration_admits_at_the_exact_upper_boundary() -> None:
    """A duration equal to ``duration_max`` is inside, not outside.

    The boundary value ``duration == duration_max`` (``10.0`` s) is
    admitted because :func:`g1_locomotion_duration_admits` refuses
    on ``value > bound`` rather than ``>= bound`` (the neon
    bundle's clamp is ``min(10.0, duration)``, which dispatches
    the boundary value unchanged; off-by-one at the boundary
    would silently reject a saturated command a caller intended to
    send).
    """
    result = _call(g1_locomotion_duration_admits, duration=_LOCOMOTION_DURATION_MAX)
    assert result["admits"] is True
    assert result["route"] == "set_velocity"
    assert result["refusals"] == []


def test_g1_locomotion_duration_admits_refuses_at_the_exact_lower_boundary() -> None:
    """A duration equal to ``duration_min`` (``0.0``) refuses strictly.

    The lower bound is **strict** because the neon wrapper's own
    conditional ``if duration <= 0: refuse`` refuses ``0.0``
    outright with the message ``"duration<=0 (non-continuous),
    refusing"`` before the SDK is touched. The refusal descriptor
    quotes ``comparison="value <= bound"`` and
    ``bound_key="duration_min"`` so a caller distinguishes this
    from the above-ceiling refusal. Pinned because reading the
    floor as inclusive would admit a no-op command the neon wrapper
    refuses at admission time.
    """
    result = _call(g1_locomotion_duration_admits, duration=_LOCOMOTION_DURATION_MIN)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "duration"
    assert r["value"] == _LOCOMOTION_DURATION_MIN
    assert r["bound_key"] == "duration_min"
    assert r["bound"] == _LOCOMOTION_DURATION_MIN
    assert r["comparison"] == "value <= bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_locomotion_duration_admits_a_value_below_the_floor() -> None:
    """A strictly-negative duration refuses on ``duration_min``.

    The refusal descriptor names ``dimension="duration"``, the
    offending value, the bound it violated
    (``bound_key="duration_min"``), the comparison
    (``"value <= bound"``, shared with the ``duration == 0``
    boundary case because both fail the same ``duration <= 0``
    conditional the neon wrapper uses), and the ``7404``
    gate-refusal code. ``route`` is ``None`` because a rejected
    value would not reach ``SetVelocity`` at wire time.
    """
    under = -1.5
    result = _call(g1_locomotion_duration_admits, duration=under)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "duration"
    assert r["value"] == under
    assert r["bound_key"] == "duration_min"
    assert r["bound"] == _LOCOMOTION_DURATION_MIN
    assert r["comparison"] == "value <= bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_locomotion_duration_admits_a_value_above_the_ceiling() -> None:
    """A duration above ``duration_max`` refuses on that bound.

    The refusal descriptor names ``dimension="duration"``, the
    offending value, the bound it violated
    (``bound_key="duration_max"``), the comparison
    (``"value > bound"``), and the ``7404`` gate-refusal code.
    ``route`` is ``None`` because a rejected value would not
    reach ``SetVelocity`` at wire time. A caller who wants a
    longer walk should read the envelope's
    ``continuous_move_method`` field and route through the
    ``LocoClient.Move(..., continous_move=True)`` branch instead.
    """
    over = _LOCOMOTION_DURATION_MAX + 5.0
    result = _call(g1_locomotion_duration_admits, duration=over)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "duration"
    assert r["value"] == over
    assert r["bound_key"] == "duration_max"
    assert r["bound"] == _LOCOMOTION_DURATION_MAX
    assert r["comparison"] == "value > bound"
    assert r["code"] == _GATE_REFUSAL_CODE
    assert r["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


@pytest.mark.parametrize("bad_duration", [math.inf, -math.inf, math.nan])
def test_g1_locomotion_duration_admits_refuses_non_finite_input(
    bad_duration: float,
) -> None:
    """``math.inf`` / ``-math.inf`` / ``math.nan`` refuse with ``comparison="non-finite"``.

    A NaN cannot be compared decidably (``nan > 0`` is ``False``
    but ``nan <= 10.0`` is also ``False``), and an infinity would
    either overrun ``duration_max`` or, on the negative branch,
    underrun ``duration_min`` - both are shape violations rather
    than value ones. Named on the refusal descriptor so a caller
    distinguishes a bounds violation from a shape violation.
    """
    result = _call(g1_locomotion_duration_admits, duration=bad_duration)
    assert result["status"] == "success"
    assert result["admits"] is False
    assert result["route"] is None
    assert len(result["refusals"]) == 1
    r = result["refusals"][0]
    assert r["dimension"] == "duration"
    assert r["comparison"] == "non-finite"
    assert r["code"] == _GATE_REFUSAL_CODE


def test_g1_locomotion_duration_admits_default_call_is_the_neon_default() -> None:
    """Calling with no arguments admits at the neon bundle's own default.

    The verb's default ``duration=1.0`` matches the neon bundle's
    own ``g1_move_velocity`` default (documented as "single
    1-second impulse for safety"), so a zero-arg invocation
    ("what would the safest, most-typical duration do?") returns
    ``admits=True, route="set_velocity"`` with an empty refusals
    list. Pinned so a change to the default that silently shifted
    the safest-query answer surfaces here.
    """
    result = _call(g1_locomotion_duration_admits)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["route"] == "set_velocity"
    assert result["refusals"] == []
