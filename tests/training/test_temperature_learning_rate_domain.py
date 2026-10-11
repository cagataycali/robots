"""FastSAC's second learning rate is checked against the same domain as its first.

FastSAC builds two optimizers from two separate learning-rate fields::

    self.actor_optimizer = torch.optim.Adam(actor_params, lr=spec.learning_rate)
    self.critic_optimizer = torch.optim.Adam(critic_params, lr=spec.learning_rate)
    ...
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=spec.alpha_lr)

``learning_rate`` went through the shared optimization gate; ``alpha_lr`` - which
``RLTrainSpec`` documents as the "Learning rate for the temperature optimizer
(SAC)" - reached the same constructor unchecked, so every failure mode that gate
documents stayed reachable through the second field. Measured on a 40-timestep
run starting from ``init_alpha=1.0``: ``alpha_lr=0`` builds the optimizer and
moves the temperature by nothing, so ``autotune_alpha=True`` silently behaves
like ``autotune_alpha=False``; ``inf`` sends it to an infinity on the first step
and, because the temperature multiplies the log-probability in the *actor* loss,
the run finished ``status="success"`` with a checkpoint whose largest parameter
magnitude was ``inf``; ``True`` was a silent rate of ``1.0``. A negative value
and ``nan`` are refused by ``torch.optim.Adam``, and a ``str`` / ``None`` /
``list`` raises a bare ``TypeError``, but all four only in ``setup`` - after the
env and both networks are built.

Scoped twice over: only the off-policy backend tunes a temperature, and within it
only the ``autotune_alpha`` branch builds the optimizer, so both the other
backends and a spec that tunes nothing must stay quiet. The refusal through the real ``validate`` entry point is one row of
``tests/training/test_rl_spec_field_domains.py``; this module keeps what the
field does to the run that consumes it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

# The one backend that tunes an entropy temperature.
OFF_POLICY = "fast_sac"

# Values the temperature optimizer cannot be driven by. ``0`` and ``False`` build
# it and never move it; ``inf`` poisons it and the actor with it; ``True`` is a
# silent rate of 1.0; the rest raise, but only once ``setup`` runs.
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
    "3e-4",
    None,
    [3e-4],
    {},
]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/temperature_learning_rate_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _alpha_lr_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``alpha_lr``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"alpha_lr"`` substring, so an unrelated problem can neither
    mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: alpha_lr ")]


class TestTheCheckAppliesOnlyWhenATemperatureIsTuned:
    """``autotune_alpha`` selects whether the field is read at all.

    ``setup`` builds a temperature optimizer only on that branch, so refusing the
    field when it is unset would reject a value the run never reads. Both
    directions are pinned, or "drop the condition" and "drop the check" would each
    pass.
    """

    def test_a_spec_that_tunes_nothing_is_not_reported_on(self, spec: RLTrainSpec) -> None:
        for value in UNUSABLE:
            spec.autotune_alpha = False
            spec.alpha_lr = value
            assert _alpha_lr_problems(OFF_POLICY, spec) == [], f"{value!r}"

    def test_a_spec_that_tunes_a_temperature_is_reported_on(self, spec: RLTrainSpec) -> None:
        for value in UNUSABLE:
            spec.autotune_alpha = True
            spec.alpha_lr = value
            assert _alpha_lr_problems(OFF_POLICY, spec), f"fast_sac accepted alpha_lr={value!r} while tuning"

    def test_tuning_is_on_by_default(self, spec: RLTrainSpec) -> None:
        """Non-vacuity: the guarded branch is the one the default spec takes."""
        assert spec.autotune_alpha is True


class TestNoneIsAValueRatherThanASentinel:
    """``alpha_lr`` has a concrete default, so ``None`` is not a request for one.

    ``TrainSpec.learning_rate`` is annotated ``float | None`` and documents ``None``
    as "use the backend's own default", so its gate exempts it. ``alpha_lr`` is
    annotated ``float`` with no such sentinel anywhere in the hierarchy, so
    ``None`` is a value the temperature optimizer cannot take rather than a request
    for a default. The asymmetry is deliberate and pinned here so it is not
    smoothed away in either direction.
    """

    def test_a_none_learning_rate_is_still_accepted(self, spec: RLTrainSpec) -> None:
        # ``RLTrainSpec`` narrows the annotation to ``float``, but the gate still
        # honours the sentinel the base contract documents - which is the whole
        # asymmetry under test, so the assignment is deliberate.
        spec.learning_rate = None  # type: ignore[assignment]
        problems = [p for p in create_trainer(OFF_POLICY).validate(spec) if "learning_rate" in p]
        assert problems == []

    def test_only_the_first_rate_declares_the_sentinel(self) -> None:
        """Executable premise for the asymmetry above: the annotations differ.

        The basis is the declared type rather than the default - ``RLTrainSpec``
        narrows ``learning_rate`` to a concrete ``1e-4``, so comparing defaults
        would find no asymmetry at all while the gates still, correctly, differ.
        """
        import dataclasses

        from strands_robots.training.base import TrainSpec

        base = {f.name: f for f in dataclasses.fields(TrainSpec)}
        rl = {f.name: f for f in dataclasses.fields(RLTrainSpec)}
        assert base["learning_rate"].type == "float | None"
        assert rl["alpha_lr"].type == "float"
        assert rl["alpha_lr"].default == pytest.approx(3e-4)


class TestGuardingTheFirstRateDoesNotGuardTheSecond:
    """The measured consequence: a run whose temperature never adapts.

    This is why the second field needed its own call rather than being covered by
    the first. ``alpha_lr=0`` leaves ``log_alpha`` where it started, so the
    temperature the spec asked to have tuned stays at ``init_alpha`` - while a
    usable rate moves it - and the pre-fix ``validate`` reported neither.
    """

    def test_a_zero_temperature_rate_freezes_the_temperature(self) -> None:
        torch = pytest.importorskip("torch")
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        def _alpha_after_one_step(alpha_lr: float) -> tuple[float, float]:
            log_alpha = torch.tensor(0.0, requires_grad=True)  # init_alpha = 1.0
            opt = torch.optim.Adam([log_alpha], lr=alpha_lr)
            before = float(log_alpha.detach().exp())
            (-(log_alpha * torch.tensor(-2.0))).backward()
            opt.step()
            return before, float(log_alpha.detach().exp())

        frozen_before, frozen_after = _alpha_after_one_step(0.0)
        tuned_before, tuned_after = _alpha_after_one_step(3e-4)
        assert frozen_before == frozen_after == 1.0, (frozen_before, frozen_after)
        assert tuned_after != tuned_before, (tuned_before, tuned_after)
        # And the field that *was* guarded cannot express this: it drives a
        # different optimizer entirely.
        assert "alpha_lr" in inspect.getsource(FastSacTrainer.setup)

    def test_the_two_rates_drive_different_optimizers(self) -> None:
        """Executable premise: the guarded field cannot stand in for this one."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        source = inspect.getsource(FastSacTrainer.setup)
        assert "lr=spec.learning_rate" in source
        assert "lr=spec.alpha_lr" in source
