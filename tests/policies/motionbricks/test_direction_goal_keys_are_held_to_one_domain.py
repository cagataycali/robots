"""A planar direction goal is refused by name, and the refusal states the rule.

:func:`~strands_robots.policies.motionbricks.observation._unit_direction` reads two
of the issue #300 well-known goal keys - ``target_velocity`` and
``target_heading`` - and normalises each to a unit 3-vector in the world XY
plane. Three things about that read were not graded anywhere.

The near-zero fallback is a magnitude test, so it cannot cover a non-finite
component. ``nan < 1e-6`` is ``False`` and ``inf`` divided by its own norm is
``nan``, so either one fell straight through the fallback and produced a
``movement_direction`` of ``[nan, nan, nan]`` - handed to the generator, from a
call that reported success. That is exactly the outcome the fallback exists to
prevent, and two shipped cells in ``test_policy.py`` say so in their own words
("instead of fabricating a NaN/garbage heading", "must not yield a NaN
direction") while driving only the inputs where the fallback does fire. The
property was stated and the input that violates it was never passed.

The refusal named neither key. One helper serves two documented keys, and both
answered ``direction vector must have 2 or 3 entries``, so a caller who passed
both could not tell which one was refused. The sibling locomotion family names
its key in every refusal - :meth:`WBCPolicy._validate_velocity` answers
``target_velocity must have at least 3 elements [vx, vy, omega]`` - and the ABC's
stated reason for leaving the component count out of the goal contract is that
"each receiver states its own arity and refuses a shape it cannot use", which
requires the refusal to say which receiver and which key.

And the message stated a rule the check did not enforce. Five statements in the
package give the rule as two or three components; the check tested ``< 2`` alone,
so four, six and twenty components were accepted and read for their first two
entries. The wire validator defers the count here deliberately ("the component
COUNT is not checked against any receiver's arity here"), so this is the only
place it is checked at all.

The per-component domain is now the shared
:func:`~strands_robots.utils.finite_vector_error` the sim setters already use, so
a ``bool`` is refused by name rather than read as ``1.0`` and the wording matches
the rest of the library. The cells below drive both keys and both doors: the
per-call kwarg and the constructor default.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import strands_robots.policies.motionbricks.observation as observation
import strands_robots.policies.motionbricks.policy as policy_module
from strands_robots.utils import finite_vector_error

_NAN = float("nan")
_INF = float("inf")

#: The two well-known goal keys this reader serves. Both must name themselves.
_DIRECTION_KEYS = ("target_velocity", "target_heading")

#: Counts the message states, stated here rather than read off the module so a
#: silent widening of the module's own bounds fails a cell instead of following it.
_ACCEPTED_COUNTS = (2, 3)


def _signals(**kwargs: Any) -> dict[str, Any]:
    """Build one control-signal dict through the public builder."""
    return observation.build_control_signals(
        mode_idx=0, clip_token_specs=[None], min_token=1, max_token=1, kwargs=kwargs
    )


def _direction_for(key: str, value: Any) -> list[float]:
    """The direction ``key`` resolves to, read from the field it drives."""
    if key == "target_velocity":
        return list(_signals(target_velocity=value)["movement_direction"])
    return list(_signals(target_velocity=[1.0, 0.0], target_heading=value)["facing_direction"])


class _StubAgent:
    """A :class:`MotionAgent` that satisfies the protocol and never generates.

    Only the constructor is exercised here, so ``next_qpos`` is never reached;
    the four class attributes are what make this a ``MotionAgent`` structurally,
    which is how the shipped stub in ``test_policy.py`` satisfies the same seam.
    """

    clip_keys: list[str] = ["walk"]
    clip_token_specs: list[list[int] | None] = [None]
    min_token: int = 1
    max_token: int = 1

    def reset(self) -> None:  # pragma: no cover - never called
        return None

    def next_qpos(
        self, control_signals: dict[str, Any], controller_dt: float
    ) -> NDArray[np.float64]:  # pragma: no cover - never called
        raise AssertionError("these cells never step the generator")


def _build(**kwargs: Any) -> policy_module.MotionBricksPolicy:
    """Construct the policy through its real constructor."""
    return policy_module.MotionBricksPolicy(motion_agent=_StubAgent(), style="walk", **kwargs)


# ---------------------------------------------------------------------------
# Regression: a non-finite component is the NaN direction the fallback prevents
# ---------------------------------------------------------------------------
class TestANonFiniteComponentIsRefused:
    """The fallback is a magnitude test, so the domain has to be a domain."""

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([_NAN, 0.0], id="nan-first"),
            pytest.param([0.0, _NAN], id="nan-second"),
            pytest.param([_NAN, _NAN], id="nan-both"),
            pytest.param([_INF, 0.0], id="inf-first"),
            pytest.param([_INF, _INF], id="inf-both"),
            pytest.param([0.0, 0.0, _NAN], id="nan-in-third-entry"),
        ],
    )
    def test_a_non_finite_component_is_refused(self, key: str, value: list[float]) -> None:
        with pytest.raises(ValueError, match="must contain finite numbers"):
            _direction_for(key, value)

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    def test_no_direction_a_caller_can_pass_reaches_the_generator_non_finite(self, key: str) -> None:
        # The harm, stated as the invariant rather than as one input: whatever a
        # caller passes, the resolved direction is finite or the call refused.
        for value in ([_NAN, 0.0], [_INF, 0.0], [0.0, _NAN], [_NAN, _NAN, _NAN]):
            try:
                resolved = _direction_for(key, value)
            except ValueError:
                continue
            assert all(math.isfinite(component) for component in resolved), (
                f"{key}={value!r} resolved to {resolved!r}, which the generator would read"
            )


# ---------------------------------------------------------------------------
# Regression: the refusal names the key it came from
# ---------------------------------------------------------------------------
class TestTheRefusalNamesTheKeyItCameFrom:
    """One helper, two documented keys; a caller who passed both must be told which."""

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([0.5], id="too-few"),
            pytest.param([0.5, 0.0, 0.0, 0.0], id="too-many"),
            pytest.param([_NAN, 0.0], id="non-finite"),
            pytest.param("abc", id="non-numeric"),
            pytest.param([True, False], id="bool"),
        ],
    )
    def test_every_refusal_names_the_key(self, key: str, value: Any) -> None:
        with pytest.raises(ValueError) as caught:
            _direction_for(key, value)
        assert key in str(caught.value)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([0.5], id="too-few"),
            pytest.param([_NAN, 0.0], id="non-finite"),
        ],
    )
    def test_the_two_keys_do_not_share_one_refusal(self, value: Any) -> None:
        # The same bad shape under each key must produce distinguishable text, or
        # naming the key has bought nothing.
        messages = []
        for key in _DIRECTION_KEYS:
            with pytest.raises(ValueError) as caught:
                _direction_for(key, value)
            messages.append(str(caught.value))
        assert messages[0] != messages[1]
        assert "target_heading" not in messages[0]
        assert "target_velocity" not in messages[1]

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    def test_a_non_numeric_value_is_refused_by_name_not_by_numpy(self, key: str) -> None:
        # Previously numpy answered "could not convert string to float: 'abc'",
        # naming neither the key nor the expected shape.
        with pytest.raises(ValueError) as caught:
            _direction_for(key, "abc")
        message = str(caught.value)
        assert "could not convert string to float" not in message
        assert "elements must be numbers" in message


# ---------------------------------------------------------------------------
# Regression: the arity the message states is the arity enforced
# ---------------------------------------------------------------------------
class TestTheStatedArityIsTheEnforcedArity:
    """Five statements said two or three; the check tested only the floor."""

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize(
        "count",
        [
            pytest.param(0, id="0"),
            pytest.param(1, id="1"),
            pytest.param(4, id="4"),
            pytest.param(6, id="6-a-spatial-twist"),
            pytest.param(20, id="20"),
        ],
    )
    def test_a_count_outside_the_stated_range_is_refused(self, key: str, count: int) -> None:
        value = [1.0] + [0.0] * (count - 1) if count else []
        with pytest.raises(ValueError, match="must have 2 or 3 entries"):
            _direction_for(key, value)

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize("count", _ACCEPTED_COUNTS)
    def test_every_count_the_message_states_is_accepted(self, key: str, count: int) -> None:
        resolved = _direction_for(key, [1.0] + [0.0] * (count - 1))
        assert resolved == pytest.approx([1.0, 0.0, 0.0])

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param((component for component in (1.0, 2.0)), id="generator"),
            pytest.param(5.0, id="bare-scalar"),
        ],
    )
    def test_a_value_with_no_readable_length_is_refused_by_the_component_domain(self, key: str, value: Any) -> None:
        # The order of the two checks is observable here, which is why it is
        # pinned. A value whose length cannot be read has no count to compare, so
        # the arity check running first would answer "got None" - reporting a
        # domain check that never ran. The component domain owns this verdict and
        # says what actually happened, which is the rule
        # ``finite_vector_error`` already applies to its own callers.
        with pytest.raises(ValueError) as caught:
            _direction_for(key, value)
        message = str(caught.value)
        assert "must be a list/tuple of numbers" in message
        assert "got None" not in message

    def test_the_module_bounds_are_the_bounds_the_message_states(self) -> None:
        assert (observation._MIN_DIRECTION_ENTRIES, observation._MAX_DIRECTION_ENTRIES) == (
            min(_ACCEPTED_COUNTS),
            max(_ACCEPTED_COUNTS),
        )


# ---------------------------------------------------------------------------
# Regression: the constructor default goes through the same door
# ---------------------------------------------------------------------------
class TestTheConstructorDefaultIsHeldToTheSameDomain:
    """A default the reader cannot honor is reported before a rollout starts."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([_NAN, 0.0], id="non-finite"),
            pytest.param([_INF, 0.0], id="inf"),
            pytest.param([0.5], id="too-few"),
            pytest.param([0.5, 0.0, 0.0, 0.0], id="too-many"),
            pytest.param([True, False], id="bool"),
        ],
    )
    def test_a_default_outside_the_domain_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError) as caught:
            _build(target_velocity=value)
        assert "target_velocity" in str(caught.value)
        assert "MotionBricksPolicy" in str(caught.value)

    def test_an_acceptable_default_is_stored_as_the_callers_own_components(self) -> None:
        # Not the unit vector: every call re-normalises, and the stored default is
        # what a caller reads back.
        built = _build(target_velocity=[0.0, 2.0])
        assert built._default_velocity == [0.0, 2.0]

    def test_omitting_the_default_stores_nothing(self) -> None:
        assert _build()._default_velocity is None


# ---------------------------------------------------------------------------
# Over-reach controls: everything a caller could legitimately pass still works
# ---------------------------------------------------------------------------
class TestWhatIsUnchanged:
    """Every shape the reader legitimately took still resolves the same way."""

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param([0.5, 0.0], [1.0, 0.0, 0.0], id="two-components"),
            pytest.param([0.0, 2.0], [0.0, 1.0, 0.0], id="normalised"),
            pytest.param([0.5, 0.0, 9.0], [1.0, 0.0, 0.0], id="third-entry-projected-away"),
            pytest.param((0.0, 3.0), [0.0, 1.0, 0.0], id="tuple"),
            pytest.param([0, 1], [0.0, 1.0, 0.0], id="ints"),
            pytest.param([np.float64(0.0), np.float32(4.0)], [0.0, 1.0, 0.0], id="numpy-scalars"),
        ],
    )
    def test_an_acceptable_direction_resolves_unchanged(self, key: str, value: Any, expected: list[float]) -> None:
        assert _direction_for(key, value) == pytest.approx(expected)

    @pytest.mark.parametrize("key", _DIRECTION_KEYS)
    def test_a_numpy_array_is_still_accepted(self, key: str) -> None:
        assert _direction_for(key, np.array([0.0, 5.0])) == pytest.approx([0.0, 1.0, 0.0])

    @pytest.mark.parametrize(
        "value",
        [pytest.param([0.0, 0.0], id="all-zero"), pytest.param([1e-300, 1e-300], id="near-zero")],
    )
    def test_the_near_zero_fallback_still_fires(self, value: list[float]) -> None:
        # The behaviour the two shipped cells in test_policy.py grade: a command
        # with no direction walks straight ahead rather than yielding NaN.
        assert _signals(target_velocity=value)["movement_direction"] == [1.0, 0.0, 0.0]

    def test_omitting_both_keys_still_walks_forward(self) -> None:
        signals = _signals()
        assert signals["movement_direction"] == [1.0, 0.0, 0.0]
        assert signals["facing_direction"] == [1.0, 0.0, 0.0]

    def test_a_heading_angle_is_untouched_by_this_domain(self) -> None:
        # target_heading_angle is a scalar, not a direction vector, so it does not
        # go through this reader at all.
        signals = _signals(target_heading_angle=math.pi / 2)
        assert signals["facing_direction"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Regression (structural): one reader, one domain, both keys named
# ---------------------------------------------------------------------------
class TestTheReaderIsSingleSourced:
    """The domain is consulted, not restated, and each call site names its key."""

    def test_the_reader_consults_the_shared_domain_rather_than_restating_it(self) -> None:
        # A local finiteness test here would be a second copy of a rule the
        # library already owns, and the two could then disagree about a bool or a
        # numpy scalar.
        source = inspect.getsource(observation._unit_direction)
        assert "finite_vector_error(" in source
        assert "isfinite" not in source

    def test_both_call_sites_pass_the_key_they_read(self) -> None:
        source = inspect.getsource(observation)
        for key in _DIRECTION_KEYS:
            assert f'param_name="{key}"' in source

    def test_the_constructor_default_goes_through_the_same_reader(self) -> None:
        # Not a second copy of the domain in the policy module: the constructor
        # calls this reader, so the two doors cannot drift apart.
        source = inspect.getsource(policy_module.MotionBricksPolicy.__init__)
        assert "_unit_direction(" in source
        assert "finite_vector_error(" not in source


# ---------------------------------------------------------------------------
# Premises: the facts the fix rests on
# ---------------------------------------------------------------------------
class TestThePremises:
    """Each of these is a fact the regression cells above depend on."""

    def test_the_fallback_cannot_cover_a_non_finite_magnitude(self) -> None:
        # Why a magnitude test is not a domain: this is the comparison the
        # fallback makes, and it is False for nan.
        assert not (_NAN < 1e-6)
        assert math.isnan(_INF / _INF)

    def test_the_shared_component_domain_answers_for_these_values(self) -> None:
        # The regression cells above expect this guard's wording, so the fix must
        # be consulting it rather than restating it.
        assert finite_vector_error("m", "p", [_NAN, 0.0]) is not None
        assert finite_vector_error("m", "p", [True, False]) is not None
        assert finite_vector_error("m", "p", "abc") is not None
        assert finite_vector_error("m", "p", [0.5, 0.0]) is None
        # And it deliberately does not check the count, which is why the arity
        # check above is separate.
        assert finite_vector_error("m", "p", [0.5]) is None
        assert finite_vector_error("m", "p", [0.5, 0.0, 0.0, 0.0]) is None

    def test_the_sibling_locomotion_family_names_its_key_too(self) -> None:
        # The convention this fix follows, read off the sibling rather than
        # asserted: wbc's own arity refusal names target_velocity.
        from strands_robots.policies.wbc.policy import WBCPolicy

        with pytest.raises(ValueError) as caught:
            WBCPolicy._validate_velocity([0.5, 0.0])
        assert "target_velocity" in str(caught.value)
