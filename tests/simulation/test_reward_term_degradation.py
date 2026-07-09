"""Robustness/degradation contract for the locomotion reward-DSL terms.

The ``base_velocity`` and ``base_height`` reward terms read a floating base's
state from ``get_observation``. Their documented contract is that they NEVER
propagate an error and NEVER invent a value: when the backend cannot supply a
usable observation - it raises, or returns something that is not a dict - the
term degrades to ``0.0`` (the same constant a fixed-base arm produces). This
mirrors the wider DSL rule that "predicates never raise".

Separately, the pure-Python world->body rotation ``_quat_rotate_inverse_wxyz``
guards a degenerate (near-zero-norm) quaternion by returning the input vector
unchanged rather than dividing by ~0.

These paths are exercised through the public ``make_predicate`` reward-term
surface with a minimal stub engine, plus a direct call to the documented
degenerate-quaternion guard. They are GL-free and need no display.
"""

import pytest

from strands_robots.simulation.predicates import (
    _quat_rotate_inverse_wxyz,
    make_predicate,
)


class _RaisingEngine:
    """Minimal engine stub whose ``get_observation`` always raises.

    A reward term must swallow this and degrade to 0.0 - it may not let a
    backend fault escape into the reward/benchmark loop.
    """

    def get_observation(self, robot_name=None, skip_images=False):
        raise RuntimeError("backend observation unavailable")


class _NonDictEngine:
    """Engine stub whose ``get_observation`` returns a non-dict.

    Some backends can return ``None`` (or another non-mapping) before a world
    is fully initialised; the term must treat that as "no base" and degrade.
    """

    def __init__(self, value):
        self._value = value

    def get_observation(self, robot_name=None, skip_images=False):
        return self._value


@pytest.mark.parametrize(
    "term_name, kwargs",
    [
        ("base_velocity", {"vx": 0.5}),
        ("base_height", {"target": 0.74}),
    ],
)
def test_reward_term_degrades_to_zero_when_observation_raises(term_name, kwargs):
    """A backend error inside get_observation degrades the term to 0.0, not a raise."""
    term = make_predicate(term_name, **kwargs)
    assert term(_RaisingEngine()) == 0.0


@pytest.mark.parametrize(
    "term_name, kwargs",
    [
        ("base_velocity", {"vx": 0.5}),
        ("base_height", {"target": 0.74}),
    ],
)
@pytest.mark.parametrize("bad_obs", [None, [1, 2, 3], "not-a-dict"])
def test_reward_term_degrades_to_zero_when_observation_not_a_dict(term_name, kwargs, bad_obs):
    """A non-dict observation is treated as "no floating base": the term is 0.0."""
    term = make_predicate(term_name, **kwargs)
    assert term(_NonDictEngine(bad_obs)) == 0.0


def test_quat_rotate_inverse_returns_input_unchanged_for_degenerate_quaternion():
    """A near-zero-norm quaternion cannot define a rotation, so the vector is
    returned unchanged instead of dividing by ~0 (documented guard)."""
    vec = [1.1, -2.2, 3.3]
    assert _quat_rotate_inverse_wxyz([0.0, 0.0, 0.0, 0.0], vec) == pytest.approx(vec, abs=1e-12)
    # A sub-threshold-norm quaternion takes the same guard.
    assert _quat_rotate_inverse_wxyz([1e-12, 0.0, 0.0, 0.0], vec) == pytest.approx(vec, abs=1e-12)
