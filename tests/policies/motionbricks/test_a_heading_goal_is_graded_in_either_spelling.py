"""One facing goal, two spellings, one answer.

``target_heading`` is a well-known goal key with two spellings of the same
quantity: a planar direction vector, or the same direction as an angle in
radians via ``target_heading_angle``. Both are read by
:func:`~strands_robots.policies.motionbricks.observation._heading_to_direction`
and both land in the same ``facing_direction`` field of the control signal the
generator is handed. Only one of them was graded.

The vector spelling is held to the shared component domain, and the argument for
that domain is written down in the sibling grader beside this one
(``test_direction_goal_keys_are_held_to_one_domain.py``): the near-zero fallback
is a magnitude test, so it cannot cover a non-finite component, and a ``nan``
fell straight through it to produce a direction of ``[nan, nan, nan]`` "handed to
the generator, from a call that reported success".

The angle spelling reached the same field through a bare ``float()``, so that
outcome survived in it verbatim - ``target_heading_angle=nan`` produced a
``facing_direction`` of ``[nan, nan, 0.0]``, reported as success. The other value
classes the vector spelling refuses by name diverged too: ``True`` was read as a
1.0 rad (57.3 degree) heading, a numeric string was read as an angle, and
``inf`` reached :func:`math.cos` to raise a bare ``math domain error`` naming
neither the surface nor the key, while ``[1.57]`` and ``10**400`` escaped as
``TypeError`` / ``OverflowError`` from the coercion itself.

The scalar spelling is now held to :func:`~strands_robots.utils.finite_number_error`,
the domain the sibling locomotion family already holds its own scalar goal kwarg
to (:meth:`~strands_robots.policies.wbc.policy.WBCPolicy._validate_height`,
``height``). The cells below grade the two spellings against one roster of value
classes and require the same answer from each, then pin the asymmetry that is
deliberate: a bare number is a legal angle and an illegal direction, and a
two-entry sequence is the reverse.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

import strands_robots.policies.motionbricks.observation as observation
from strands_robots.utils import finite_number_error

_NAN = float("nan")
_INF = float("inf")

#: The two spellings of the one facing goal. Each must name itself when it refuses.
_ANGLE_KEY = "target_heading_angle"
_VECTOR_KEY = "target_heading"

#: Value classes neither spelling can carry, in the shape each spelling takes.
#: A list of tuples rather than a dict because ``True == 1`` and ``0 == False``
#: collapse exactly the distinctions this file is about.
_UNUSABLE_CLASSES: list[tuple[str, Any, Any]] = [
    ("nan", _NAN, [_NAN, 0.0]),
    ("inf", _INF, [_INF, 0.0]),
    ("-inf", -_INF, [-_INF, 0.0]),
    ("bool", True, [True, False]),
    ("numeric-string", "1.5708", ["1.5708", "0.0"]),
    ("past-float64", 10**400, [10**400, 0.0]),
    ("non-number", [1.57], [None, 0.0]),
]


def _signals(**kwargs: Any) -> dict[str, Any]:
    """Build one control-signal dict through the public builder."""
    return observation.build_control_signals(
        mode_idx=0, clip_token_specs=[None], min_token=1, max_token=1, kwargs=kwargs
    )


def _verdict(key: str, value: Any) -> str:
    """What the builder does with ``value`` given under ``key``.

    A refusal counts only when it names the key the caller spelled. Folding any
    exception into "refused" would let these cells pass against the code they
    were written to catch: the bare coercion also raised ``ValueError`` for
    ``inf`` (``math domain error``), naming neither this surface nor the key.

    Returns:
        ``"refused"``, ``"accepted"`` or ``"escaped:<ExceptionType>"``.
    """
    try:
        _signals(target_velocity=[1.0, 0.0], **{key: value})
    except ValueError as exc:
        return "refused" if key in str(exc) else f"escaped:ValueError({str(exc)[:40]})"
    except Exception as exc:  # noqa: BLE001 - any escape is the defect, so report its type
        return f"escaped:{type(exc).__name__}"
    return "accepted"


def _facing(key: str, value: Any) -> list[float]:
    """The facing direction ``key`` resolves to, read from the field it drives."""
    return list(_signals(target_velocity=[1.0, 0.0], **{key: value})["facing_direction"])


class TestNeitherSpellingCarriesAnUnusableValue:
    """One roster, two spellings, the same answer from each."""

    @pytest.mark.parametrize(
        ("scalar", "vector"),
        [pytest.param(s, v, id=name) for name, s, v in _UNUSABLE_CLASSES],
    )
    def test_both_spellings_refuse_it_by_name(self, scalar: Any, vector: Any) -> None:
        assert _verdict(_ANGLE_KEY, scalar) == "refused"
        assert _verdict(_VECTOR_KEY, vector) == "refused"

    def test_a_nan_angle_does_not_reach_the_generator_as_a_facing_direction(self) -> None:
        # The specific outcome the vector spelling's domain exists to prevent,
        # reached through the other spelling of the same key: a facing direction
        # of [nan, nan, 0.0] from a call that reported success.
        with pytest.raises(ValueError, match=_ANGLE_KEY):
            _signals(target_velocity=[1.0, 0.0], target_heading_angle=_NAN)

    def test_a_bool_is_not_read_as_a_one_radian_heading(self) -> None:
        # 57.3 degrees off the movement direction, from a caller who passed a flag.
        with pytest.raises(ValueError, match=_ANGLE_KEY):
            _signals(target_velocity=[1.0, 0.0], target_heading_angle=True)


class TestWhatIsUnchanged:
    """Every angle a caller could legitimately pass still resolves the same way."""

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            pytest.param(0.0, [1.0, 0.0, 0.0], id="zero-faces-plus-x"),
            pytest.param(math.pi / 2, [0.0, 1.0, 0.0], id="quarter-turn"),
            pytest.param(-math.pi, [-1.0, 0.0, 0.0], id="negative-angle"),
            pytest.param(3 * math.pi, [-1.0, 0.0, 0.0], id="past-one-turn"),
            pytest.param(0, [1.0, 0.0, 0.0], id="int"),
            pytest.param(np.float64(math.pi / 2), [0.0, 1.0, 0.0], id="numpy-float64"),
            pytest.param(np.float32(0.0), [1.0, 0.0, 0.0], id="numpy-float32"),
        ],
    )
    def test_a_usable_angle_resolves_unchanged(self, angle: Any, expected: list[float]) -> None:
        assert _facing(_ANGLE_KEY, angle) == pytest.approx(expected, abs=1e-9)

    def test_the_angle_still_overrides_the_movement_direction(self) -> None:
        # The behaviour test_policy.py grades: facing is independent of movement.
        signals = _signals(target_velocity=[1.0, 0.0], target_heading_angle=math.pi / 2)
        assert signals["movement_direction"] == pytest.approx([1.0, 0.0, 0.0])
        assert signals["facing_direction"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)

    def test_omitting_the_angle_still_faces_the_movement_direction(self) -> None:
        assert _signals(target_velocity=[0.0, 2.0])["facing_direction"] == pytest.approx([0.0, 1.0, 0.0])

    def test_the_angle_still_wins_over_the_vector_spelling(self) -> None:
        # Precedence is unchanged: the angle is read first.
        signals = _signals(target_velocity=[1.0, 0.0], target_heading_angle=0.0, target_heading=[0.0, 1.0])
        assert signals["facing_direction"] == pytest.approx([1.0, 0.0, 0.0])


class TestTheAsymmetryThatIsDeliberate:
    """The two spellings take different shapes, and that difference is not a bug."""

    def test_a_bare_number_is_a_legal_angle_and_an_illegal_direction(self) -> None:
        assert _verdict(_ANGLE_KEY, 1.5708) == "accepted"
        assert _verdict(_VECTOR_KEY, 1.5708) == "refused"

    def test_a_two_entry_sequence_is_a_legal_direction_and_an_illegal_angle(self) -> None:
        assert _verdict(_VECTOR_KEY, [0.0, 1.0]) == "accepted"
        assert _verdict(_ANGLE_KEY, [0.0, 1.0]) == "refused"

    def test_the_two_spellings_agree_on_the_direction_they_name(self) -> None:
        # Same quantity, so the same facing direction whichever way it arrives.
        assert _facing(_ANGLE_KEY, math.pi / 2) == pytest.approx(_facing(_VECTOR_KEY, [0.0, 1.0]), abs=1e-9)


class TestTheReaderIsSingleSourced:
    """The domain is consulted, not restated, and the refusal names the key."""

    def test_the_reader_consults_the_shared_scalar_domain(self) -> None:
        # A local finiteness test here would be a second copy of a rule the
        # library already owns, and the two could then disagree about a bool or
        # a numpy scalar.
        source = inspect.getsource(observation._heading_to_direction)
        assert "finite_number_error(" in source
        assert "isfinite" not in source

    def test_the_call_site_passes_the_key_it_reads(self) -> None:
        source = inspect.getsource(observation._heading_to_direction)
        assert f'"{_ANGLE_KEY}"' in source


class TestThePremises:
    """Each of these is a fact the cells above depend on."""

    def test_the_shared_scalar_domain_answers_for_these_values(self) -> None:
        # The roster above expects this guard's verdicts, so the fix must be
        # consulting it rather than restating it.
        for _, scalar, _vector in _UNUSABLE_CLASSES:
            assert finite_number_error(scalar, "p", "c") is not None
        assert finite_number_error(math.pi / 2, "p", "c") is None
        assert finite_number_error(np.float64(0.0), "p", "c") is None

    def test_a_bare_coercion_could_not_have_refused_these(self) -> None:
        # Why grading has to happen before the conversion: float() reinterprets
        # three of the roster's classes rather than refusing them.
        assert float(True) == 1.0
        assert float("1.5708") == pytest.approx(1.5708)
        assert math.isnan(float(_NAN))

    def test_cosine_of_a_non_finite_angle_cannot_report_a_direction(self) -> None:
        # The two ways the old path failed downstream of the coercion.
        assert math.isnan(math.cos(_NAN))
        with pytest.raises(ValueError, match="math domain error"):
            math.cos(_INF)

    def test_the_conversion_left_in_place_can_only_restate_the_value(self) -> None:
        # float() stays after the guard, matching the sibling family's
        # grade-then-convert shape, and for every value the domain admits it is
        # a restatement rather than a decision.
        for value in (1.57, 2, np.float64(1.57), np.float32(1.57)):
            assert math.cos(value) == math.cos(float(value))

    def test_the_sibling_family_holds_its_scalar_goal_kwarg_to_this_domain(self) -> None:
        # The convention this fix follows, read off the sibling rather than
        # asserted: wbc grades its own scalar goal kwarg with the same helper.
        from strands_robots.policies.wbc.policy import WBCPolicy

        with pytest.raises(ValueError, match="height"):
            WBCPolicy._validate_height(_NAN)
