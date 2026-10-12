"""Every RL hyperparameter is refused by exactly the backends that read it, in the shared words.

``RLTrainSpec`` carries two dozen fields only the RL backends read, and many are read
by only some of them: PPO alone estimates a GAE trace, clips a trust region and
standardizes advantages, FastSAC alone tunes an entropy temperature, FastTD3 alone
delays its actor and injects deterministic-policy noise, and both off-policy
backends replay and Polyak-average; all three size the run from ``total_timesteps``
and ``rollout_steps``, checkpoint every ``log_interval`` iterations, normalize
observations and place tensors on ``device``. Each field's preflight gate is a one-liner over a shared rule in
:mod:`strands_robots.utils` or :mod:`strands_robots.training._validate`, so the
contract every field shares is one table here, run over every training backend:

* every backend that reads the field reports, for every probe value, exactly the
  shared rule's verdict - one problem in the rule's own words for an unusable value,
  nothing for a usable one, nothing for the shipped default - and setting every
  field of a group unusable at once reports each of them in the same pass;
* every backend that does not read the field - the supervised ones included -
  reports nothing about it, while it still refuses a field it does read (silence
  is scoping, not an empty preflight).

What each refused value does to the run that consumes it (a diverging return, a
never-firing cadence, a frozen temperature) is pinned beside the consumer, in the
field's own ``test_<field>_domain.py``.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training._validate import (
    _clip_bound_error,
    _closed_unit_interval_error,
    _half_open_unit_interval_error,
)
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import (
    boolean_flag_error,
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
    step_cadence_error,
    torch_device_error,
)

SUPERVISED = ("mock", "lerobot_local", "cosmos3", "rsl_rl", "sagemaker", "isaaclab")
RL = ("ppo", "fast_sac", "fast_td3")
BACKENDS = SUPERVISED + RL

# Spellings no numeric rule reads: a bool is a silent 1/0, the rest raise out of a
# bare comparison mid-run.
NOT_A_NUMBER: list[Any] = [True, False, np.bool_(True), "1.0", None, [1.0], {}]
NON_FINITE: list[Any] = [float("nan"), float("inf"), float("-inf"), np.float64(np.nan)]

COUNT_UNUSABLE: list[Any] = [
    *NOT_A_NUMBER,
    0,
    -1,
    -1000,
    0.5,
    2.5,
    2.0,
    100.5,
    *NON_FINITE,
    np.int64(4),
    np.int32(3),
    np.uint8(2),
]
COUNT_USABLE: list[Any] = [1, 2, 8, 256, 5000, 10**6]

RATE_UNUSABLE: list[Any] = [*NOT_A_NUMBER, 0, 0.0, -1e-3, -0.1, -1, *NON_FINITE, 10**400]
RATE_USABLE: list[Any] = [3e-4, 1e-5, 1e-3, 0.2, 0.5, 1.0, 5.0, np.float64(0.3), np.float32(1e-4)]

SIGNED_UNUSABLE: list[Any] = [*NOT_A_NUMBER, "-6.0", *NON_FINITE, 10**400]
SIGNED_USABLE: list[Any] = [
    -6.0,
    -6,
    -3.5,
    -1.0,
    -0.5,
    -1e-8,
    0,
    0.0,
    1e-8,
    1.5,
    2,
    np.float64(-6.0),
    np.float32(0.5),
    np.int64(2),
]

# ``target_entropy=None`` is the sentinel for the heuristic default, not a value.
TARGET_ENTROPY_UNUSABLE: list[Any] = [v for v in SIGNED_UNUSABLE if v is not None]

CLOSED_UNIT_UNUSABLE: list[Any] = [*SIGNED_UNUSABLE, 1.5, 1.0000001, 10.0, 1e6, -0.5, -1, -2.0]
CLOSED_UNIT_USABLE: list[Any] = [0.0, 1.0, 0.99, 0.95, 0.5, 0, 1, np.float64(0.97), np.float32(0.8)]

HALF_OPEN_UNIT_UNUSABLE: list[Any] = [*SIGNED_UNUSABLE, 0.0, 0, -0.005, -1.0, 1.0000001, 2.0, 100.0]
HALF_OPEN_UNIT_USABLE: list[Any] = [0.005, 0.5, 1.0, 1, 1e-8, Fraction(1, 200), np.float64(0.01), np.float32(0.005)]

# ``inf`` is the one carve-out a clip bound has: the spelling of "do not clip".
CLIP_UNUSABLE: list[Any] = [*NOT_A_NUMBER, "inf", 0, 0.0, -0.2, -1.0, -100.0, float("nan"), float("-inf"), 10**400]
CLIP_USABLE: list[Any] = [0.2, 0.1, 1.0, 10.0, 1e-8, 3, np.float64(1.0), np.int64(2), float("inf"), np.float64(np.inf)]

# A cadence in iterations: ``0`` and a negative are the documented "no periodic
# checkpoints" mode, so the domain has no floor.
CADENCE_UNUSABLE: list[Any] = [*(v for v in NOT_A_NUMBER if v is not None), 2.5, 2.0, *NON_FINITE]
CADENCE_USABLE: list[Any] = [1, 100, 5000, 0, -1]

# The opt-out spellings a caller reaches for (every one truthy), falsy values that
# are not ``False`` either, and a 2-element array whose truth value raises.
FLAG_UNUSABLE: list[Any] = [
    "false",
    "no",
    "0",
    "",
    "yes",
    0,
    1,
    float("nan"),
    None,
    [],
    [True],
    np.array([True, False]),
]
FLAG_USABLE: list[Any] = [True, False, np.True_, np.False_]

# Measured against ``torch.device``: ``'gpu'`` is the ordinary mistake, and a
# non-``str`` is refused before torch is consulted (``torch.device(1)`` constructs
# on any host and dies at the first ``.to()``). ``'cuda:7'`` and ``'mps'`` are
# usable on a host without them: the spelling is the domain, not the inventory.
DEVICE_UNUSABLE: list[Any] = ["gpu", "CUDA", "cuda:abc", "  ", 1, 2.0, True, ["cuda"], np.array(["cuda", "cpu"])]
DEVICE_USABLE: list[Any] = ["cpu", "cuda", "cuda:0", "cuda:7", "mps", None, "", 0]


def _stated_device_error(value: Any, param: str, context: str) -> str | None:
    """``torch_device_error`` for a stated device; ``spec.device or ...`` owns a falsy scalar."""
    if value is None or (isinstance(value, str | numbers.Number) and not value):
        return None
    return torch_device_error(value, param, context)


@dataclass(frozen=True)
class FieldDomain:
    """One group of ``RLTrainSpec`` fields that share a rule and a reader set."""

    fields: tuple[str, ...]
    rule: Callable[[Any, str, str], str | None]
    readers: tuple[str, ...]
    unusable: list[Any]
    usable: list[Any]
    # The shared rule ``rule`` narrows (an interval) or widens (the ``inf`` clip
    # carve-out); every value it refuses, ``rule`` refuses in the same words.
    base: Callable[[Any, str, str], str | None] | None = None
    # The rule reads its domain from torch, and passes everything without it.
    needs_torch: bool = False

    @property
    def silent(self) -> tuple[str, ...]:
        return tuple(p for p in BACKENDS if p not in self.readers)


DOMAINS = {
    "gamma": FieldDomain(
        ("gamma",),
        _closed_unit_interval_error,
        ("ppo", "fast_sac", "fast_td3"),
        CLOSED_UNIT_UNUSABLE,
        CLOSED_UNIT_USABLE,
    ),
    "lam": FieldDomain(
        ("lam",), _closed_unit_interval_error, ("ppo",), CLOSED_UNIT_UNUSABLE, CLOSED_UNIT_USABLE, finite_number_error
    ),
    "num_learning_epochs": FieldDomain(
        ("num_learning_epochs",), positive_count_error, ("ppo",), COUNT_UNUSABLE, COUNT_USABLE
    ),
    "max_grad_norm": FieldDomain(
        ("max_grad_norm",), _clip_bound_error, ("ppo",), CLIP_UNUSABLE, CLIP_USABLE, positive_finite_number_error
    ),
    "clip_param": FieldDomain(
        ("clip_param",), _clip_bound_error, ("ppo",), CLIP_UNUSABLE, CLIP_USABLE, positive_finite_number_error
    ),
    "loss_weights": FieldDomain(
        ("value_loss_coef", "entropy_coef"), finite_number_error, ("ppo",), SIGNED_UNUSABLE, SIGNED_USABLE
    ),
    "alpha_lr": FieldDomain(("alpha_lr",), positive_finite_number_error, ("fast_sac",), RATE_UNUSABLE, RATE_USABLE),
    "init_alpha": FieldDomain(("init_alpha",), positive_finite_number_error, ("fast_sac",), RATE_UNUSABLE, RATE_USABLE),
    "target_entropy": FieldDomain(
        ("target_entropy",), finite_number_error, ("fast_sac",), TARGET_ENTROPY_UNUSABLE, SIGNED_USABLE
    ),
    "policy_delay": FieldDomain(("policy_delay",), positive_count_error, ("fast_td3",), COUNT_UNUSABLE, COUNT_USABLE),
    "td3_noise": FieldDomain(
        ("exploration_noise_std", "target_noise_std", "target_noise_clip"),
        positive_finite_number_error,
        ("fast_td3",),
        RATE_UNUSABLE,
        RATE_USABLE,
    ),
    "tau": FieldDomain(
        ("tau",),
        _half_open_unit_interval_error,
        ("fast_sac", "fast_td3"),
        HALF_OPEN_UNIT_UNUSABLE,
        HALF_OPEN_UNIT_USABLE,
    ),
    "run_size": FieldDomain(
        ("total_timesteps", "rollout_steps"), positive_count_error, RL, COUNT_UNUSABLE, COUNT_USABLE
    ),
    "log_interval": FieldDomain(("log_interval",), step_cadence_error, RL, CADENCE_UNUSABLE, CADENCE_USABLE),
    "normalize_obs": FieldDomain(("normalize_obs",), boolean_flag_error, RL, FLAG_UNUSABLE, FLAG_USABLE),
    "normalize_advantage": FieldDomain(
        ("normalize_advantage",), boolean_flag_error, ("ppo",), FLAG_UNUSABLE, FLAG_USABLE
    ),
    "autotune_alpha": FieldDomain(("autotune_alpha",), boolean_flag_error, ("fast_sac",), FLAG_UNUSABLE, FLAG_USABLE),
    "device": FieldDomain(
        ("device",), _stated_device_error, RL, DEVICE_UNUSABLE, DEVICE_USABLE, torch_device_error, needs_torch=True
    ),
    "replay": FieldDomain(
        ("buffer_size", "batch_size", "gradient_steps"),
        positive_count_error,
        ("fast_sac", "fast_td3"),
        COUNT_UNUSABLE,
        COUNT_USABLE,
    ),
}


def _spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(output_dir="/tmp/rl_spec_field_domains", env_factory=lambda: None)  # type: ignore[arg-type,return-value]


def _reports(provider: str, spec: RLTrainSpec, field: str) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``field``.

    Filtered on the shared rules' ``"{context}: {param} "`` / ``"{context}: {param}="``
    message shapes rather than on a bare substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    trainer = create_trainer(provider)
    return [p for p in trainer.validate(spec) if _names(p, f"{trainer.provider_name}: {field}")]


def _names(problem: str, prefix: str) -> bool:
    return problem.startswith((f"{prefix} ", f"{prefix}="))


@pytest.mark.parametrize("group", DOMAINS)
def test_every_reader_reports_exactly_the_shared_verdict(group: str) -> None:
    domain = DOMAINS[group]
    if domain.needs_torch:
        pytest.importorskip("torch")
    for provider in domain.readers:
        default = _spec()
        for field in domain.fields:
            assert _reports(provider, default, field) == [], f"{provider} refused the shipped {field}"
            for value in [*domain.unusable, *domain.usable]:
                spec = _spec()
                setattr(spec, field, value)
                verdict = domain.rule(value, field, provider)
                shared = domain.base(value, field, provider) if domain.base else None
                assert None in (shared, verdict) or verdict == shared, (
                    f"{field}={value!r} departs from the shared words"
                )
                assert (verdict is None) is any(value is v for v in domain.usable), f"{field}={value!r}"
                assert _reports(provider, spec, field) == ([] if verdict is None else [verdict]), (
                    f"{provider} {field}={value!r}"
                )
        # Every field of the group is reported in the same pass, not one round at a time.
        spec = _spec()
        for field in domain.fields:
            setattr(spec, field, domain.unusable[0])
        problems = create_trainer(provider).validate(spec)
        for field in domain.fields:
            assert sum(_names(p, f"{provider}: {field}") for p in problems) == 1, (field, problems)


@pytest.mark.parametrize("group", DOMAINS)
def test_a_backend_that_does_not_read_the_field_stays_silent(group: str) -> None:
    domain = DOMAINS[group]
    assert domain.silent, group
    for provider in domain.silent:
        for field in domain.fields:
            for value in domain.unusable:
                spec = _spec()
                setattr(spec, field, value)
                assert _reports(provider, spec, field) == [], f"{provider} read {field}={value!r}"


@pytest.mark.parametrize("provider", BACKENDS)
def test_the_silence_is_scoping_rather_than_an_empty_preflight(provider: str) -> None:
    """Every backend still refuses the one rate they all read."""
    spec = _spec()
    spec.learning_rate = 0.0
    assert _reports(provider, spec, "learning_rate"), provider


@pytest.mark.parametrize("group", DOMAINS)
def test_a_refused_field_stops_the_run_before_anything_is_built(group: str, tmp_path: Any) -> None:
    """``train`` is fail-closed on ``validate``: no env, no checkpoint.

    The probe is each group's first unusable value: ``True`` for every numeric
    group, the spelling every bare reading took silently (a rate, a count, a
    weight or a delay of one), ``"false"`` for a posture flag and ``"gpu"`` for
    the device.
    """
    pytest.importorskip("torch")
    domain = DOMAINS[group]
    built: list[str] = []

    def env_factory() -> None:
        built.append("env")
        raise AssertionError("the refused field reached the environment")

    for field in domain.fields:
        spec = RLTrainSpec(output_dir=str(tmp_path / field), env_factory=env_factory)  # type: ignore[arg-type]
        setattr(spec, field, domain.unusable[0])
        result = create_trainer(domain.readers[0]).train(spec)
        assert result.status == "error", field
        assert any(f"{field}{sep}" in (result.message or "") for sep in " ="), result.message
        assert not result.checkpoint_dir, field
    assert built == []
    assert not list(tmp_path.rglob("*.pt"))
