# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every training spec field is named by the surface that governs it.

An agent populates :class:`~strands_robots.training.base.TrainSpec` (or its RL
subclass) field by field from the class docstring, and a backend that reads a
field preflights it through the matching ``Trainer._*_problems`` gate. Both
surfaces are prose, so both can silently stop naming a field the code still
reads - an undocumented field is one an agent cannot populate, and a gate that
does not name the field it grades sends a refused caller to the wrong knob.

This scans the docstrings against the dataclass fields and the ``_validate``
gates, so a field added without an ``Attributes`` entry, or a gate whose
docstring no longer names its field, fails here.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.rl.base_algo import RLTrainSpec

# ``Attributes:`` entries are indented one level past the section header, so a
# field entry is the only thing that opens at that depth.
_ATTRIBUTE = re.compile(r"^\s{4}(\w+):", re.MULTILINE)

# The field(s) each shared preflight grades. A gate MUST name every field it
# reports on, because the message a caller gets names that field.
_GATE_FIELDS: dict[str, tuple[str, ...]] = {
    "_run_size_problems": ("steps", "global_batch_size"),
    "_rl_run_size_problems": ("total_timesteps", "rollout_steps"),
    "_rl_replay_problems": ("buffer_size", "batch_size", "gradient_steps"),
    "_learning_rate_problems": ("learning_rate",),
    "_launch_topology_problems": ("num_gpus", "num_nodes"),
    "_seed_problems": ("seed",),
    "_checkpoint_cadence_problems": ("save_freq",),
    "_rl_checkpoint_interval_problems": ("log_interval",),
    "_validation_episodes_problems": ("val_episodes",),
    "_resume_problems": ("resume",),
    "_streaming_problems": ("streaming",),
    "_lora_hyperparameter_problems": ("lora_r", "lora_alpha"),
    "_discount_factor_problems": ("gamma",),
    "_gae_lambda_problems": ("lam",),
    "_optimization_epochs_problems": ("num_learning_epochs",),
    "_temperature_learning_rate_problems": ("alpha_lr",),
    "_initial_temperature_problems": ("init_alpha",),
    "_target_entropy_problems": ("target_entropy",),
    "_polyak_coefficient_problems": ("tau",),
    "_spec_device_problems": ("device",),
    "_gradient_clip_problems": ("max_grad_norm",),
    "_loss_weight_problems": ("value_loss_coef", "entropy_coef"),
    "_clip_range_problems": ("clip_param",),
    "_policy_delay_problems": ("policy_delay",),
    "_td3_noise_problems": ("exploration_noise_std", "target_noise_std", "target_noise_clip"),
    "_network_width_problems": ("hidden_dims",),
}


def _documented_attributes(cls: type) -> set[str]:
    """Field names the class's own ``Attributes:`` section names."""
    doc = inspect.cleandoc(cls.__doc__ or "")
    _, _, attributes = doc.partition("Attributes:")
    return set(_ATTRIBUTE.findall(attributes))


@pytest.mark.parametrize("cls", [TrainSpec, RLTrainSpec], ids=lambda c: c.__name__)
def test_every_spec_field_has_an_attributes_entry(cls: type) -> None:
    """A field an agent must populate is named where the agent reads."""
    documented = _documented_attributes(cls)
    for base in cls.__mro__[1:]:
        if dataclasses.is_dataclass(base):
            documented |= _documented_attributes(base)
    declared = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
    assert not declared - documented, (
        f"{cls.__name__} declares {sorted(declared - documented)} but its docstring "
        "names no Attributes entry for them, so an agent reading the class cannot "
        "learn the field exists or what it accepts."
    )


@pytest.mark.parametrize("gate, fields", sorted(_GATE_FIELDS.items()))
def test_every_gate_names_the_fields_it_grades(gate: str, fields: tuple[str, ...]) -> None:
    """The refusal names a field, so the docstring must name it too."""
    doc = inspect.getdoc(getattr(Trainer, gate)) or ""
    unnamed = [name for name in fields if name not in doc]
    assert not unnamed, (
        f"Trainer.{gate} reports on {unnamed} but its docstring never names them, "
        "so a caller refused by that gate cannot find the knob from the method it came from."
    )


def test_the_gate_table_covers_every_shared_preflight() -> None:
    """A new gate joins the table instead of shipping unscanned."""
    graded = {name for name in vars(Trainer) if name.endswith("_problems") and name not in ("_security_problems",)}
    assert graded == set(_GATE_FIELDS), (
        f"gates not in the table: {sorted(graded - set(_GATE_FIELDS))}; "
        f"table entries that are not gates: {sorted(set(_GATE_FIELDS) - graded)}"
    )
