"""FastSAC's entropy temperature is checked where it is aimed, not only where it starts.

``target_entropy`` is the third and last caller-supplied field of FastSAC's
temperature block, and the only one nothing judged. It is the constant the
temperature is optimized *toward* - the one term of the temperature loss a caller
supplies::

    alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

reaching that expression through an unconditional ``float(spec.target_entropy)``
coercion in ``setup``. Its two neighbours - the temperature's starting value and
the rate that moves it - were both put behind a positive-finite domain, and this
field was explicitly left out of both because it is *signed* by construction: it
defaults to ``-num_actions``, so a positive-finite domain would refuse every
reading of it. The domain it does take is the signed counterpart the on-policy
loss weights already read, ``finite_number_error``: any finite real, either sign,
because no endpoint of a target entropy in nats is decidable while a value that is
not a finite real has no reading at all.

Measured on a 40-timestep run, ``validate()`` returning ``[]`` for all of them.
``True`` is a silent sign flip - a target entropy of ``+1.0`` where the field's own
default is ``-6.0`` for this env - and the run reported success while checkpointing
``log_alpha == -0.0018031001091003418`` against the honored run's
``-0.001801646314561367``, so it drove the temperature somewhere else rather than
harmlessly. ``nan`` / ``inf`` / ``-inf`` poison ``alpha``, which scales the entropy
term of both the critic's TD target and the actor loss, and the next rollout raises
``ValueError`` from inside ``torch.distributions.Normal`` about a tensor of ``nan``
policy means - naming that distribution's parameter rather than the field, after
the env, both networks and a full rollout have been built. A list or a dict raises
``TypeError`` out of the ``float()`` coercion in ``setup``.

Two scoping decisions are pinned as well as the domain. ``None`` is exempt rather
than refused: the field is annotated ``float | None`` and ``None`` is the
documented request for the ``-num_actions`` heuristic, unlike the two neighbours
whose ``float`` annotations carry concrete defaults. And the check is *not*
conditioned on ``autotune_alpha`` even though only the tuning branch spends the
value, because the coercion that reads it is unconditional - with tuning off a list
raises the same ``TypeError`` while ``nan`` reaches a successful run that never
spends it, so a check scoped to the tuning branch would let the raising shape
through for the untuned one.

The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import finite_number_error

# The one backend that optimizes a temperature against a target entropy.
OFF_POLICY = "fast_sac"


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/target_entropy_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _target_entropy_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``target_entropy``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"target_entropy"`` substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: target_entropy ")]


class TestTheNoneSentinelIsExemptRatherThanRefused:
    """``None`` is the documented request for ``-num_actions``, not a value."""

    def test_the_field_declares_the_sentinel_in_its_annotation(self) -> None:
        """Executable premise for the exemption, unlike the two neighbouring fields."""
        assert RLTrainSpec.__annotations__["target_entropy"] == "float | None"
        assert RLTrainSpec.__annotations__["init_alpha"] == "float"
        assert RLTrainSpec.__annotations__["alpha_lr"] == "float"

    def test_the_backend_substitutes_the_heuristic_for_it(self) -> None:
        """Executable premise: ``None`` reaches a substitution, not the coercion."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert (
            "float(spec.target_entropy) if spec.target_entropy is not None else -float(self.env.num_actions)" in setup
        )


class TestBothTemperatureBranchesAreCovered:
    """The coercion that reads the field runs whether or not the temperature is tuned.

    Only the tuning branch *spends* the value, but the ``float()`` that reads it is
    unconditional, so a non-real value raises in ``setup`` either way. Scoping the
    check to ``autotune_alpha`` would let that shape through for the untuned branch.
    """

    @pytest.mark.parametrize("autotune", [True, False])
    def test_a_non_real_value_is_reported_on_either_branch(self, spec: RLTrainSpec, autotune: bool) -> None:
        for value in list[Any]([[-6.0], {}, "-6"]):
            spec.autotune_alpha = autotune
            spec.target_entropy = value
            assert _target_entropy_problems(OFF_POLICY, spec), (autotune, value)

    def test_the_coercion_is_not_under_the_tuning_branch(self) -> None:
        """Executable premise: the read precedes ``if self.autotune_alpha``."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert setup.index("float(spec.target_entropy)") < setup.index("if self.autotune_alpha:")


class TestTheDomainIsTheSharedSignedOne:
    """The verdict is the shared helper's, not a local re-derivation."""

    def test_the_domain_is_signed_rather_than_positive(self) -> None:
        """The premise the two neighbouring gates could not express."""
        from strands_robots.utils import positive_finite_number_error

        assert finite_number_error(-6.0, "target_entropy", OFF_POLICY) is None
        assert positive_finite_number_error(-6.0, "target_entropy", OFF_POLICY) is not None
