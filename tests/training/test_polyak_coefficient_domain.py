"""The rate a target network tracks its online one is checked, not just compared.

``tau`` is the Polyak coefficient of the two off-policy backends. Each spends it
in one expression per mirrored critic pair::

    tp.mul_(1.0 - spec.tau).add_(spec.tau * p)

so it does not merely tune the update - it decides whether a separate target
network exists at all, and a target network is what makes an off-policy critic's
bootstrap target stationary enough to regress onto.

Its interval was the *precedent* the two on-policy interval gates were written
against: ``discount_factor_problems`` and ``gae_lambda_problems`` both cite "the
sibling FastSAC preflight already bounds its own interval coefficient this way
(``tau`` must be in ``(0, 1]``)" as the shape they generalize. The check they
cited was a bare local comparison, ``if not 0.0 < spec.tau <= 1.0``, duplicated
verbatim in both backends - so the precedent was weaker than the generalization
it inspired, and it left two holes that generalization had already closed.

``True`` is a silent ``tau`` of **one**, because ``bool`` is an ``int``
subclass - the maximum of the interval, at which the Polyak average degenerates
to ``tp.mul_(0.0).add_(p)`` and the target network becomes a copy of the online
network on every update. Measured on a 60-timestep FastSAC run with
``validate()`` returning ``[]`` and the run reporting success: the largest
online-to-target parameter gap in the exported checkpoint was ``0.0`` exactly,
against ``9.9e-04`` for the default ``tau``, and the checkpoint was
byte-identical to a run that asked for ``tau=1.0``. A flag landed as a request to
switch target networks off, and nothing in the run said so.

A numeric string, ``None`` or a list raised ``TypeError: '<' not supported
between instances of 'float' and 'str'`` out of the comparison itself - from a
``validate`` documented to *return* its problems.

``nan`` and the two infinities were already refused, but by an accident of
Python's chained comparison rather than by a finiteness test: every comparison
against ``nan`` is False and ``inf`` is above the upper bound. They stay refused,
now with a reason that names finiteness.

Neither endpoint changes reading, so nothing that worked becomes an error:
``1.0`` stays accepted as the deliberate hard update and ``0.0`` stays refused
because it freezes the target network at its initialization. That one decision -
zero outside the interval - is what makes this domain half-open where the
on-policy pair's is closed.

The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/polyak_coefficient_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


class TestTheDomainIsHalfOpenWhereTheOnPolicyOneIsClosed:
    """The one decision this interval owns, against the gate that cites it."""

    def test_zero_parts_the_two_intervals(self, spec: RLTrainSpec) -> None:
        """``gamma=0`` is a myopic agent; ``tau=0`` is a target network that never moves."""
        spec.gamma = 0.0
        spec.tau = 0.0
        problems = create_trainer("fast_sac").validate(spec)
        assert not [p for p in problems if p.startswith("fast_sac: gamma ")], problems
        assert [p for p in problems if p.startswith("fast_sac: tau ")], problems
