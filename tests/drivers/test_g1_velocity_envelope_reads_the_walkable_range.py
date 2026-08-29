"""The velocity-envelope lookup tools name what ``LocoClient.SetVelocity`` walks.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes
``SetVelocity(vx, vy, vyaw, duration)`` and its ``Move(vx, vy, vyaw,
continous_move=True)`` cousin without clamps of its own: any finite
argument reaches the controller unchanged, and the controller's
behaviour above the neon-bundle-observed walkable range is undefined.
The :mod:`strands_robots.tools.g1.g1_velocity_envelope` module snapshots
that observed range into module-level constants and exposes two agent-
facing verbs - :func:`g1_list_velocity_envelope` (name the whole
envelope) and :func:`g1_velocity_admits` (decide one query) - so a
caller can decide the refusal decidably before a future locomotion
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
  none). A driver-side wrapper for ``SetVelocity`` that lands later
  will re-check the envelope at wire time and its refusal string
  will quote the ``7404`` gate-refusal code the driver's
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
from strands_robots.tools.g1.g1_velocity_envelope import (
    _ANGLE_ABS_MAX,
    _DISTANCE_ABS_MAX,
    _DURATION_MAX_SECONDS,
    _DURATION_MIN_SECONDS,
    _GATE_REFUSAL_CODE,
    _SPEED_MAX,
    _SPEED_MIN,
    _VX_ABS_MAX,
    _VY_ABS_MAX,
    _VYAW_ABS_MAX,
    _YAW_RATE_MAX,
    _YAW_RATE_MIN,
    g1_list_velocity_envelope,
    g1_velocity_admits,
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
    importlib.import_module("strands_robots.tools.g1.g1_velocity_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_velocity_envelope imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        f"loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_envelope_bounds_are_finite_and_ordered() -> None:
    """Every clamp is a finite float and the min/max pairs are ordered.

    A non-finite bound would let ``g1_velocity_admits`` admit every
    value on that dimension; an inverted min/max pair (min > max)
    would reject every finite value. Pins the invariant so a widen or
    narrow of the observed range that inverts a pair surfaces here
    rather than as a silently unreachable envelope in production.
    """
    for name, value in (
        ("_VX_ABS_MAX", _VX_ABS_MAX),
        ("_VY_ABS_MAX", _VY_ABS_MAX),
        ("_VYAW_ABS_MAX", _VYAW_ABS_MAX),
        ("_DURATION_MIN_SECONDS", _DURATION_MIN_SECONDS),
        ("_DURATION_MAX_SECONDS", _DURATION_MAX_SECONDS),
        ("_DISTANCE_ABS_MAX", _DISTANCE_ABS_MAX),
        ("_SPEED_MIN", _SPEED_MIN),
        ("_SPEED_MAX", _SPEED_MAX),
        ("_ANGLE_ABS_MAX", _ANGLE_ABS_MAX),
        ("_YAW_RATE_MIN", _YAW_RATE_MIN),
        ("_YAW_RATE_MAX", _YAW_RATE_MAX),
    ):
        assert math.isfinite(value), f"{name} is not finite: {value!r}"

    assert _DURATION_MIN_SECONDS <= _DURATION_MAX_SECONDS, (
        f"duration bounds inverted: min={_DURATION_MIN_SECONDS} > "
        f"max={_DURATION_MAX_SECONDS}. g1_velocity_admits would refuse "
        f"every duration."
    )
    assert _SPEED_MIN <= _SPEED_MAX, f"speed bounds inverted: min={_SPEED_MIN} > max={_SPEED_MAX}"
    assert _YAW_RATE_MIN <= _YAW_RATE_MAX, f"yaw-rate bounds inverted: min={_YAW_RATE_MIN} > max={_YAW_RATE_MAX}"


def test_the_gate_refusal_code_matches_the_driver_constant() -> None:
    """The envelope's refusal code names the driver's gate refusal.

    The driver's ``_check_motion_gates`` refuses a locomotion-shaped
    write on an FSM outside :data:`WALK_FSMS` with rc=7404, and the
    ``7404`` entry in
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries the
    text a driver-side velocity wrapper would surface. Pinned here so
    a re-wording of that message lands in one place, not one in the
    driver and a diverging copy in this envelope.
    """
    assert _GATE_REFUSAL_CODE == 7404
    assert _GATE_REFUSAL_CODE in ERR_CODES, (
        f"envelope quotes rc={_GATE_REFUSAL_CODE} but that code is not in "
        f"ERR_CODES. The refusal string would render as 'unknown'."
    )


def test_g1_list_velocity_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names every clamp, the gate set, and the refusal.

    ``envelope`` carries every clamp constant, ``walk_ready_fsm_ids``
    quotes :data:`WALK_FSMS`, and ``refusals`` names the ``7404``
    gate-refusal code with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_velocity_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["vx_abs_max"] == _VX_ABS_MAX
    assert env["vy_abs_max"] == _VY_ABS_MAX
    assert env["vyaw_abs_max"] == _VYAW_ABS_MAX
    assert env["duration_min_seconds"] == _DURATION_MIN_SECONDS
    assert env["duration_max_seconds"] == _DURATION_MAX_SECONDS
    assert env["distance_abs_max"] == _DISTANCE_ABS_MAX
    assert env["speed_min"] == _SPEED_MIN
    assert env["speed_max"] == _SPEED_MAX
    assert env["angle_abs_max"] == _ANGLE_ABS_MAX
    assert env["yaw_rate_min"] == _YAW_RATE_MIN
    assert env["yaw_rate_max"] == _YAW_RATE_MAX
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert result["refusals"] == [
        {"code": _GATE_REFUSAL_CODE, "text": ERR_CODES[_GATE_REFUSAL_CODE]},
    ]


def test_g1_velocity_admits_a_vector_inside_every_clamp() -> None:
    """A vector inside every clamp is admitted with an empty refusals list.

    The zero-shape vector (``vx=vy=vyaw=0.0, duration=1.0``) is the
    identity case: it sits at the origin on every direction axis and
    inside the duration window, so a driver-side wrapper for
    ``SetVelocity`` would not refuse it on envelope grounds (whether
    the FSM gate admits it is a separate live-read decision the verb
    does not answer).
    """
    result = _call(g1_velocity_admits, vx=0.0, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []
    assert result["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_g1_velocity_admits_at_the_exact_clamp_boundaries() -> None:
    """A vector at the clamp boundaries is inside, not outside.

    Boundary values ``|vx|=vx_abs_max``, ``|vy|=vy_abs_max``,
    ``|vyaw|=vyaw_abs_max``, ``duration=duration_max_seconds`` are
    admitted because :func:`g1_velocity_admits` refuses on ``|value|
    > bound`` rather than ``>= bound`` (the neon bundle's clamp does
    the same, and off-by-one at the boundary would silently reject a
    caller's saturated command).

    ``duration_min_seconds`` is deliberately absent from this cell:
    that bound is exclusive, so its boundary refuses rather than
    admits, and it is pinned by
    :func:`test_g1_velocity_admits_refuses_a_zero_or_negative_duration`.
    """
    result = _call(
        g1_velocity_admits,
        vx=_VX_ABS_MAX,
        vy=-_VY_ABS_MAX,
        vyaw=_VYAW_ABS_MAX,
        duration=_DURATION_MAX_SECONDS,
    )
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_velocity_admits_refuses_a_vx_above_the_clamp() -> None:
    """``vx`` above ``vx_abs_max`` reads as one refusal on that dimension.

    The refusal descriptor names the dimension, the value the caller
    passed, the bound-key on the envelope it violated, and the
    ``7404`` code a driver-side wrapper would quote. Other
    dimensions inside their clamps do not appear in the refusals
    list, so a caller sees only the boundaries that failed.
    """
    result = _call(g1_velocity_admits, vx=_VX_ABS_MAX + 0.5, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["admits"] is False
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "vx"
    assert refusal["value"] == _VX_ABS_MAX + 0.5
    assert refusal["bound_key"] == "vx_abs_max"
    assert refusal["bound"] == _VX_ABS_MAX
    assert refusal["code"] == _GATE_REFUSAL_CODE
    assert refusal["text"] == ERR_CODES[_GATE_REFUSAL_CODE]


def test_g1_velocity_admits_refuses_a_negative_vx_symmetrically() -> None:
    """A negative ``vx`` past the clamp reads the same as a positive one.

    The envelope clamps ``|vx|``, not ``vx`` directly, because reverse
    walking is the same walk-envelope with the sign flipped. This
    cell pins the symmetric behaviour so a caller passing
    ``vx=-vx_abs_max - 0.1`` sees the same refusal shape as one
    passing ``vx=+vx_abs_max + 0.1``.
    """
    result = _call(g1_velocity_admits, vx=-(_VX_ABS_MAX + 0.1), vy=0.0, vyaw=0.0, duration=1.0)
    assert result["admits"] is False
    assert result["refusals"][0]["dimension"] == "vx"


def test_g1_velocity_admits_reports_every_violated_dimension_at_once() -> None:
    """A vector outside three clamps names three refusals in one call.

    A caller learns every boundary they missed in one query rather
    than one per dispatch attempt; the driver's gate at wire time
    would name only the first refusal it hits, so this verb's payload
    is strictly wider than the wire-time refusal.
    """
    result = _call(
        g1_velocity_admits,
        vx=_VX_ABS_MAX + 1.0,
        vy=_VY_ABS_MAX + 1.0,
        vyaw=_VYAW_ABS_MAX + 1.0,
        duration=_DURATION_MAX_SECONDS + 100.0,
    )
    assert result["admits"] is False
    dimensions = {refusal["dimension"] for refusal in result["refusals"]}
    assert dimensions == {"vx", "vy", "vyaw", "duration"}


def test_g1_velocity_admits_refuses_a_non_finite_component() -> None:
    """A non-finite ``vx`` reads as refused with the ``non-finite`` comparison.

    ``math.inf`` and ``math.nan`` are not admissible even against a
    generous clamp; the refusal descriptor's ``comparison`` field
    names ``non-finite`` so a caller distinguishes a bounds-violation
    from a shape-violation.
    """
    result = _call(g1_velocity_admits, vx=math.inf, vy=0.0, vyaw=0.0, duration=1.0)
    assert result["admits"] is False
    assert result["refusals"][0]["dimension"] == "vx"
    assert result["refusals"][0]["comparison"] == "non-finite"

    result_nan = _call(g1_velocity_admits, vx=math.nan, vy=0.0, vyaw=0.0, duration=1.0)
    assert result_nan["admits"] is False
    assert result_nan["refusals"][0]["comparison"] == "non-finite"


@pytest.mark.parametrize("duration", [0.0, -0.0, -1.0])
def test_g1_velocity_admits_refuses_a_zero_or_negative_duration(duration: float) -> None:
    """A duration at or below ``duration_min_seconds`` reads as refused.

    The neon bundle refuses zero-or-negative durations because the
    controller ignores them without raising, which would let a
    silently-dropped command look like a successful walk to the
    planner.

    ``duration=0.0`` is the cell that matters and it is the reason this
    is parametrised rather than asserting a negative alone: the min
    bound *is* ``0.0``, so a refusal written as ``value < bound`` admits
    exactly the no-op it exists to catch, and a negative-only assertion
    cannot see that. A duration computed as distance/speed that rounds
    to zero is the realistic caller, and it must not read as admitted.
    ``-0.0`` is included because it compares equal to the bound while
    carrying a sign bit, so it is decided by the same comparison.
    """
    result = _call(g1_velocity_admits, vx=0.0, vy=0.0, vyaw=0.0, duration=duration)
    assert result["admits"] is False
    assert result["refusals"][0]["dimension"] == "duration"
    assert result["refusals"][0]["bound_key"] == "duration_min_seconds"
    assert result["refusals"][0]["comparison"] == "value <= bound"


def test_g1_velocity_admits_refuses_a_duration_above_the_ceiling() -> None:
    """A duration above the ceiling reads as refused with the ceiling key.

    The neon bundle caps single-shot duration at
    :data:`_DURATION_MAX_SECONDS`; longer commands switch to the
    ``Move(continous_move=True)`` path, which is a separate write and
    outside this envelope. Pinned so a caller trying to walk the
    single-shot path for 100 seconds sees an actionable refusal
    rather than a silently truncated command.
    """
    result = _call(
        g1_velocity_admits,
        vx=0.0,
        vy=0.0,
        vyaw=0.0,
        duration=_DURATION_MAX_SECONDS + 100.0,
    )
    assert result["admits"] is False
    assert result["refusals"][0]["dimension"] == "duration"
    assert result["refusals"][0]["bound_key"] == "duration_max_seconds"
