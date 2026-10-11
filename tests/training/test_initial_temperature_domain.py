"""FastSAC's entropy temperature is checked where it starts, not only where it moves.

FastSAC does not store the temperature: it stores the temperature's *logarithm*,
so ``init_alpha`` reaches ``torch.log`` on both temperature branches::

    self.log_alpha = torch.tensor(float(torch.log(torch.tensor(spec.init_alpha))), ...)

and ``alpha = log_alpha.exp()`` scales the entropy term in the critic's TD target
and in the actor loss from there on. The temperature's *learning rate* went
through the shared positive-finite gate; the value that rate moves reached
``torch.log`` unchecked - and that gate's own reasoning depends on this one, since
it refuses ``alpha_lr=0`` on the grounds that "the temperature stays at
``init_alpha`` for the whole run".

Measured on a 40-timestep run. ``init_alpha=0`` (and ``0.0``, and ``False``) makes
``log(0) == -inf``, so ``alpha`` is exactly ``0``: the entropy term is gone from
both losses, and the run is no longer the maximum-entropy algorithm that was
asked for. Automatic tuning cannot lift it back - ``log_alpha`` was still ``-inf``
after further gradient steps, because no finite update moves an infinity - and the
run reported ``status="success"`` while checkpointing ``log_alpha == -inf``, so the
unusable temperature outlives the run. ``True`` is a silent temperature of exactly
``1.0``. A negative value, ``nan`` or ``inf`` poisons the actor loss instead and
the first update raises ``ValueError`` from ``torch.distributions.Normal`` about a
tensor of ``nan`` policy means - naming that distribution's parameter rather than
the field, and only inside ``train``, after the env and both networks are built.

Scoped once, not twice: only the off-policy backend holds a temperature, but
within it ``init_alpha`` is read on *both* branches, so unlike the rate this check
must not be conditioned on ``autotune_alpha`` - with tuning off the field is the
temperature for the whole run and nothing can move it afterwards. The refusal
through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

# The one backend that holds an entropy temperature.
OFF_POLICY = "fast_sac"

# Values with no usable logarithm. ``0`` / ``0.0`` / ``False`` give ``-inf`` and a
# temperature of exactly zero that nothing can move; ``True`` is a silent ``1.0``;
# the rest give ``nan`` or ``inf`` and poison the actor loss.
UNUSABLE: list[Any] = [
    0,
    0.0,
    -1e-3,
    -1,
    False,
    True,
    float("nan"),
    float("inf"),
    float("-inf"),
    "1.0",
    None,
    [1.0],
    {},
]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/initial_temperature_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _init_alpha_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``init_alpha``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"init_alpha"`` substring, so an unrelated problem can neither
    mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: init_alpha ")]


class TestBothTemperatureBranchesAreCovered:
    """``init_alpha`` is read whether or not the temperature is tuned.

    This is the one scoping difference from the rate: ``alpha_lr`` guards an
    optimizer only the ``autotune_alpha`` branch constructs, so that gate is
    conditioned on the flag. ``init_alpha`` is read on both branches, and with
    tuning off it is the temperature for the entire run, so conditioning this
    check on the flag would leave the more permanent case unguarded.
    """

    @pytest.mark.parametrize("autotune", [True, False])
    def test_it_is_reported_on_either_branch(self, spec: RLTrainSpec, autotune: bool) -> None:
        for value in UNUSABLE:
            spec.autotune_alpha = autotune
            spec.init_alpha = value
            assert _init_alpha_problems(OFF_POLICY, spec), (
                f"fast_sac accepted init_alpha={value!r} (autotune={autotune})"
            )

    def test_both_branches_read_the_field(self) -> None:
        """Executable premise: the un-tuned branch is not a path that ignores it."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert setup.count("spec.init_alpha") == 2, setup.count("spec.init_alpha")


class TestNoneIsAValueRatherThanASentinel:
    """``init_alpha`` is annotated ``float`` with a concrete default.

    Contrast ``learning_rate``, where ``None`` asks the backend for its own
    default. Here there is nothing to fall back to: ``torch.log(None)`` raises.
    """

    def test_the_field_declares_a_concrete_default(self) -> None:
        import dataclasses

        (field,) = [f for f in dataclasses.fields(RLTrainSpec) if f.name == "init_alpha"]
        assert field.default == 1.0
        assert "None" not in str(field.type)


class TestGuardingTheRateDoesNotGuardTheTemperature:
    """The measured consequence: a temperature of zero that nothing can move.

    This is why the starting value needed its own call rather than being covered
    by the rate. ``log(0)`` is ``-inf``, so ``alpha`` is exactly zero and the
    entropy term is absent from both losses; and because no finite gradient step
    moves an infinity, the automatic tuning the spec asked for cannot recover it.
    """

    def test_a_zero_temperature_is_zero_and_stays_zero(self) -> None:
        torch = pytest.importorskip("torch")

        log_alpha = torch.tensor(float(torch.log(torch.tensor(0.0))), requires_grad=True)
        assert float(log_alpha.detach()) == float("-inf")
        assert float(log_alpha.detach().exp()) == 0.0, "premise: the temperature starts at exactly zero"

        opt = torch.optim.Adam([log_alpha], lr=3e-4)
        for _ in range(5):
            opt.zero_grad()
            (-(log_alpha * torch.tensor(-2.0))).backward()
            opt.step()
        assert float(log_alpha.detach().exp()) == 0.0, (
            f"a usable temperature rate lifted alpha to {float(log_alpha.detach().exp())}, "
            "so the starting value would not need its own check"
        )

    def test_a_usable_temperature_does_move(self) -> None:
        """Control: the same loop moves a temperature that started with a logarithm."""
        torch = pytest.importorskip("torch")

        log_alpha = torch.tensor(float(torch.log(torch.tensor(1.0))), requires_grad=True)
        before = float(log_alpha.detach().exp())
        opt = torch.optim.Adam([log_alpha], lr=3e-4)
        opt.zero_grad()
        (-(log_alpha * torch.tensor(-2.0))).backward()
        opt.step()
        assert float(log_alpha.detach().exp()) != before

    def test_the_temperature_reaches_a_logarithm(self) -> None:
        """Executable premise: the field is not stored as the temperature itself."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert "torch.log(torch.tensor(spec.init_alpha))" in setup
