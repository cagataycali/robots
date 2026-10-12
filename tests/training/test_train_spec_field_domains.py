"""Every ``TrainSpec`` field is refused by exactly the backends that read it, in the shared words.

The spec fields every trainer may read - the run size, the rate, the seed, the
checkpoint cadence, the held-out split, the launch topology, the posture flags and
the LoRA adapter shape - are each read by only some backends: Cosmos and Isaac Lab
never stream a dataset, the mock backend seeds nothing, the RL trainers size a run
in timesteps rather than steps. Each backend's preflight gate is a one-liner over a
shared rule in :mod:`strands_robots.utils`, so the contract every field shares is
one table here:

* every backend that reads the field reports, for every probe value, exactly the
  shared rule's verdict - one problem in the rule's own words, prefixed with the
  backend's name, for an unusable value; nothing for a usable one, for the shipped
  default, or for the documented ``None`` "use the backend's own default" sentinel -
  and setting every field of a group unusable at once reports each in one pass;
* every backend that does not read the field reports nothing about it.

What each refused value does to the run that consumes it (a cadence that
checkpoints every step, a split that holds out a count nobody asked for) is
pinned beside the consumer, in the field's own
``test_<field>_domain.py``. The RL-only hyperparameters have the same table in
``test_rl_spec_field_domains.py``.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import (
    boolean_flag_error,
    non_negative_count_error,
    positive_count_error,
    positive_finite_number_error,
    step_cadence_error,
)
from tests.training.test_rl_spec_field_domains import (
    BACKENDS,
    CADENCE_UNUSABLE,
    CADENCE_USABLE,
    COUNT_UNUSABLE,
    COUNT_USABLE,
    FLAG_UNUSABLE,
    FLAG_USABLE,
    RATE_UNUSABLE,
    RATE_USABLE,
    RL,
    SUPERVISED,
)

# Zero is a seed; ``None`` is the unset sentinel.
SEED_UNUSABLE: list[Any] = [v for v in COUNT_UNUSABLE if v is not None and not (type(v) is int and v == 0)]
SEED_USABLE: list[Any] = [0, *COUNT_USABLE, 2**32 - 1]
# A split is bounded by the dataset too (10 episodes here), so usable counts stay below it.
SPLIT_USABLE: list[Any] = [1, 2, 8]


@dataclass(frozen=True)
class FieldDomain:
    """One group of ``TrainSpec`` fields that share a rule and a reader set."""

    fields: tuple[str, ...]
    rule: Callable[[Any, str, str], str | None]
    readers: tuple[str, ...]
    unusable: list[Any]
    usable: list[Any]
    # ``None`` spells "use the backend's own default" and is never refused.
    unset_is_none: bool = False
    # Read only under this ``method`` (the LoRA adapter shape).
    method: str = "full"

    @property
    def silent(self) -> tuple[str, ...]:
        return tuple(p for p in BACKENDS if p not in self.readers)


_NOT_NONE = [v for v in COUNT_UNUSABLE if v is not None]

DOMAINS = {
    "learning_rate": FieldDomain(
        ("learning_rate",),
        positive_finite_number_error,
        BACKENDS,
        [v for v in RATE_UNUSABLE if v is not None],
        RATE_USABLE,
        unset_is_none=True,
    ),
    "seed": FieldDomain(("seed",), non_negative_count_error, BACKENDS[1:], SEED_UNUSABLE, SEED_USABLE, True),
    "steps": FieldDomain(("steps",), positive_count_error, SUPERVISED, COUNT_UNUSABLE, COUNT_USABLE),
    "global_batch_size": FieldDomain(
        ("global_batch_size",), positive_count_error, SUPERVISED[:-1], COUNT_UNUSABLE, COUNT_USABLE
    ),
    "save_freq": FieldDomain(("save_freq",), step_cadence_error, SUPERVISED[1:], CADENCE_UNUSABLE, CADENCE_USABLE),
    "val_episodes": FieldDomain(
        ("val_episodes",), positive_count_error, ("lerobot_local", "sagemaker"), _NOT_NONE, SPLIT_USABLE, True
    ),
    "launch_topology": FieldDomain(
        ("num_gpus", "num_nodes"),
        positive_count_error,
        ("lerobot_local", "cosmos3", "sagemaker", "isaaclab"),
        COUNT_UNUSABLE,
        [1, 2, 8],
    ),
    "resume": FieldDomain(
        ("resume",),
        boolean_flag_error,
        ("lerobot_local", "rsl_rl", "sagemaker", "isaaclab"),
        FLAG_UNUSABLE,
        FLAG_USABLE,
    ),
    "streaming": FieldDomain(
        ("streaming",), boolean_flag_error, ("lerobot_local", "sagemaker"), FLAG_UNUSABLE, FLAG_USABLE
    ),
    "lora": FieldDomain(
        ("lora_r", "lora_alpha"),
        positive_count_error,
        ("lerobot_local", "sagemaker"),
        _NOT_NONE,
        COUNT_USABLE,
        unset_is_none=True,
        method="lora",
    ),
}


@pytest.fixture
def make_spec(tmp_path: pathlib.Path) -> Callable[[str], TrainSpec]:
    """An otherwise-valid spec for ``provider``, so only the field under test varies."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10, "fps": 30}))
    out = tmp_path / "out"
    out.mkdir()

    def make(provider: str, method: str = "full") -> TrainSpec:
        if provider in RL:
            return RLTrainSpec(output_dir=str(out), env_factory=lambda: None)  # type: ignore[arg-type,return-value]
        return TrainSpec(
            dataset_root=str(root),
            base_model="lerobot/act",
            output_dir=str(out),
            method=method,
            embodiment="so101",
            extra={"task": "Isaac-Cartpole"} if provider == "isaaclab" else {},
        )

    return make


def _reports(provider: str, spec: TrainSpec, field: str) -> list[str]:
    """Problems ``validate`` reports about ``field``, on the shared ``"{context}: {param} "`` shape."""
    trainer = create_trainer(provider)
    return [p for p in trainer.validate(spec) if p.startswith(f"{trainer.provider_name}: {field} ")]


@pytest.mark.parametrize("group", DOMAINS)
def test_every_reader_reports_exactly_the_shared_verdict(group: str, make_spec: Any) -> None:
    domain = DOMAINS[group]
    for provider in domain.readers:
        for field in domain.fields:
            assert _reports(provider, make_spec(provider, domain.method), field) == [], f"{provider} default {field}"
            if domain.unset_is_none:
                spec = make_spec(provider, domain.method)
                setattr(spec, field, None)
                assert _reports(provider, spec, field) == [], f"{provider} refused the unset {field}"
            for value in [*domain.unusable, *domain.usable]:
                spec = make_spec(provider, domain.method)
                setattr(spec, field, value)
                verdict = domain.rule(value, field, create_trainer(provider).provider_name)
                assert (verdict is None) is any(value is v for v in domain.usable), f"{field}={value!r}"
                assert _reports(provider, spec, field) == ([] if verdict is None else [verdict]), (
                    f"{provider} {field}={value!r}"
                )
        spec = make_spec(provider, domain.method)
        for field in domain.fields:
            setattr(spec, field, domain.unusable[0])
        problems = create_trainer(provider).validate(spec)
        for field in domain.fields:
            assert sum(f": {field} " in p for p in problems) == 1, (provider, field, problems)


@pytest.mark.parametrize("group", [g for g, d in DOMAINS.items() if d.silent])
def test_a_backend_that_does_not_read_the_field_stays_silent(group: str, make_spec: Any) -> None:
    domain = DOMAINS[group]
    for provider in domain.silent:
        for field in domain.fields:
            for value in domain.unusable:
                spec = make_spec(provider, domain.method)
                setattr(spec, field, value)
                assert _reports(provider, spec, field) == [], f"{provider} read {field}={value!r}"
