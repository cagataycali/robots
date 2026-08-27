"""Every WBC observation dimension is held to the shared positive-count domain.

:class:`~strands_robots.policies.wbc.config.WBCConfig` carries five discrete
dimensions - ``num_actions``, ``obs_history_len``, ``single_obs_dim``,
``command_dim`` and ``n_obs_joints`` - and each is consumed as a ``deque``
maxlen, a ``range()`` bound, a slice index or an ``np.zeros`` width. Each used
to be checked with a bare comparison (``< 1``, or ``< 3`` for the command
block), which decides a floor and cannot decide whether the value is an integer
at all. Two spellings got through, and the shared domain's own docstring names
the first of them:

* ``True`` is an ``int`` subclass, so it passed ``obs_history_len < 1`` and
  :class:`~strands_robots.policies.wbc.observation.ObservationHistory` stacked
  ONE frame into an 86-wide network input where the checkpoint expects 516. No
  exception, no warning - just the wrong observation, at the wrong width.
* ``nan`` is below nothing, so it passed every one of the five tests and
  surfaced as a bare numpy ``TypeError`` from the observation builder, after the
  ONNX sessions had loaded and the rollout had started. That is the mid-rollout
  failure the value checks in the same ``__post_init__`` exist to convert into a
  construction-time message naming the field.

The scalars in that same method already went through the shared numeric domains;
only the dimensions did not. These cells pin the fields onto
:func:`~strands_robots.utils.positive_count_error`, and pin what that domain
does NOT decide: the command block's floor of three and the relation between
``n_obs_joints`` and ``num_actions`` are per-field rules that survive it.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc.config import WBCConfig
from strands_robots.policies.wbc.observation import ObservationHistory
from strands_robots.utils import positive_count_error

_POLICY = "p.onnx"

# The five discrete dimensions, named locally so these cells are an independent
# oracle rather than a restatement of the tuple the config iterates.
_DIMENSIONS = ("num_actions", "obs_history_len", "single_obs_dim", "command_dim", "n_obs_joints")

# Spellings a bare ``< 1`` comparison cannot refuse, plus the ones it could.
_REFUSED = [
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(2.5, id="fractional-float"),
    pytest.param(86.0, id="integral-float"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param("86", id="numeric-string"),
    pytest.param(None, id="none"),
    pytest.param([86], id="list"),
    pytest.param(np.int64(86), id="numpy-int"),
    pytest.param(np.float64(86), id="numpy-float"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
]


def _config(**overrides: Any) -> WBCConfig:
    """Build a config, letting every dimension default unless overridden."""
    return WBCConfig(policy_path=_POLICY, **overrides)


class TestTheConsumersNeedThisExactDomain:
    """Premise: the dimension consumers refuse what a bare comparison accepts.

    The domain is strict-``int`` rather than any-integral-real because these
    values reach a C-level API. Asserting that here is what makes the choice a
    measurement rather than a preference: an integral float and a NumPy integer
    are both refused by the consumer, so a domain that accepted them would hand
    the caller a value the next call cannot use.
    """

    def test_the_history_deque_refuses_a_numpy_integer_maxlen(self) -> None:
        # Each value below is typed Any deliberately: mypy states the same
        # refusal statically, which is why the domain is strict-int, and an
        # annotated local keeps the runtime assertion without an ignore code.
        numpy_integer: Any = np.int64(6)
        with pytest.raises(TypeError):
            deque(maxlen=numpy_integer)

    def test_a_frame_width_refuses_a_fractional_float(self) -> None:
        fractional: Any = 86.5
        with pytest.raises(TypeError):
            np.zeros(fractional, dtype=np.float64)

    def test_a_range_bound_refuses_a_non_integer(self) -> None:
        fractional: Any = 2.5
        with pytest.raises(TypeError):
            range(fractional)

    def test_nan_is_below_nothing_so_a_bare_floor_cannot_refuse_it(self) -> None:
        """The arithmetic the old checks rested on, stated once."""
        assert not (math.nan < 1)
        assert not (math.nan < 3)
        assert not (math.nan < 15)


class TestEveryDimensionIsHeldToTheDomain:
    """Regression: each of the five dimensions refuses each unusable spelling."""

    @pytest.mark.parametrize("dimension_name", _DIMENSIONS)
    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_dimension_is_refused_at_construction(self, dimension_name: str, value: Any) -> None:
        with pytest.raises(ValueError) as caught:
            _config(**{dimension_name: value})
        assert dimension_name in str(caught.value)

    @pytest.mark.parametrize("dimension_name", _DIMENSIONS)
    def test_the_refusal_names_the_field_and_the_domain(self, dimension_name: str) -> None:
        with pytest.raises(ValueError, match=rf"{dimension_name} must be a positive integer"):
            _config(**{dimension_name: float("nan")})

    @pytest.mark.parametrize("dimension_name", _DIMENSIONS)
    def test_the_refusal_carries_the_value_the_caller_supplied(self, dimension_name: str) -> None:
        with pytest.raises(ValueError, match=r"got 2\.5"):
            _config(**{dimension_name: 2.5})


class TestTheHistoryIsNoLongerSilentlyTruncated:
    """The headline: a flag read as a history length built the wrong input.

    ``obs_history_len=True`` is the one spelling that produced no error at all.
    The deque took ``maxlen=1`` (``True`` is ``1``), ``num_obs`` became ``86 *
    True == 86``, and ``push`` returned an 86-wide vector for a checkpoint
    expecting 516 - a stacked observation five frames short, reported as
    healthy.
    """

    def test_a_true_history_length_is_refused_rather_than_read_as_one(self) -> None:
        with pytest.raises(ValueError, match=r"obs_history_len must be a positive integer"):
            _config(obs_history_len=True)

    def test_a_healthy_history_still_stacks_the_full_network_input(self) -> None:
        history = ObservationHistory(_config())
        stacked = history.push(np.ones(86, dtype=np.float64))
        assert len(history) == 6
        assert stacked.shape[0] == 516

    def test_a_non_real_width_no_longer_reaches_the_frame_allocation(self) -> None:
        """``nan`` used to arrive at ``np.zeros`` as a bare ``TypeError``."""
        with pytest.raises(ValueError, match=r"single_obs_dim must be a positive integer"):
            _config(single_obs_dim=float("nan"))


class TestTheDomainIsWiredForEveryDiscreteField:
    """Derived: every ``int``-annotated field of the config goes through it.

    Read off the dataclass rather than a list, so a sixth discrete dimension
    added later is held to the same domain the hour it lands instead of
    inheriting an exemption by being absent from a tuple.
    """

    @staticmethod
    def _integer_fields() -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(WBCConfig) if f.type in ("int", int))

    def test_the_derived_set_covers_the_dimensions_these_cells_name(self) -> None:
        """Non-vacuity: the scan really finds the fields, and misses none."""
        derived = self._integer_fields()
        assert set(_DIMENSIONS) <= set(derived), (
            f"the config's int-annotated fields are {derived}, which does not cover {_DIMENSIONS}"
        )

    @pytest.mark.parametrize("value", [True, float("nan"), 2.5])
    def test_every_derived_field_refuses_the_spellings_a_floor_cannot(self, value: Any) -> None:
        for dimension_name in self._integer_fields():
            with pytest.raises(ValueError, match=rf"{dimension_name} must be a positive integer"):
                _config(**{dimension_name: value})

    def test_the_shared_domain_agrees_with_these_cells(self) -> None:
        """The refusals come from the shared domain, not a local restatement."""
        for value in (True, float("nan"), 2.5, np.int64(6), 0):
            assert positive_count_error(value, "d", "C") is not None
        assert positive_count_error(6, "d", "C") is None


class TestWhatTheCountDomainDoesNotDecide:
    """Controls: the per-field floor and the cross-field relation survive it.

    The domain decides positive-integer-ness. It does not decide that the
    command block needs three entries, nor that a controller cannot drive more
    joints than it observes. Both rules are still asked, of values the domain
    accepted, and both keep their own message.
    """

    def test_the_command_block_still_needs_three_entries(self) -> None:
        with pytest.raises(ValueError, match=r"command_dim must be >= 3 \(vx, vy, omega\)"):
            _config(command_dim=2)

    def test_observing_fewer_joints_than_are_driven_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match=r"n_obs_joints \(4\) must be >= num_actions \(5\)"):
            _config(num_actions=5, n_obs_joints=4)

    def test_a_usable_dimension_set_is_still_accepted(self) -> None:
        config = _config(single_obs_dim=95, command_dim=8, num_actions=15, n_obs_joints=29, obs_history_len=6)
        assert config.num_obs == 95 * 6

    def test_the_defaults_are_unchanged(self) -> None:
        config = _config()
        assert (config.single_obs_dim, config.obs_history_len, config.num_obs) == (86, 6, 516)
        assert (config.num_actions, config.n_obs_joints, config.command_dim) == (15, 29, 7)

    def test_a_command_block_of_three_is_the_boundary_and_is_accepted(self) -> None:
        assert _config(command_dim=3).command_dim == 3

    def test_the_domain_answers_before_the_floor_does(self) -> None:
        """A non-count gets the domain's diagnosis, not the floor's.

        ``True`` is below three, so a floor asked first would answer
        "command_dim must be >= 3, got True" - which reads as a value slightly
        too small when the truth is that a flag is not a width at all. The
        domain is asked first so the message names the real fault, and the floor
        keeps its own wording for a value that genuinely is a count below three.
        """
        with pytest.raises(ValueError, match=r"command_dim must be a positive integer"):
            _config(command_dim=True)
        with pytest.raises(ValueError, match=r"command_dim must be >= 3"):
            _config(command_dim=2)
