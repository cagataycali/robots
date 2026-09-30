# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an rsl_rl actor - what Isaac Lab trains - as a strands deployable checkpoint.

Isaac Lab's ``isaaclab train`` (and so the ``isaaclab`` trainer) writes rsl_rl
checkpoints, ``model_<iteration>.pt``. Their actor is not strands' PPO actor:
rsl_rl 5.x builds an ``MLPModel`` of ``Linear`` layers with a configurable
activation (the Isaac Lab task configs use ELU, strands' PPO uses Tanh) and an
optional ``obs_normalizer``, stored under ``actor_state_dict``. Loading those
weights into strands' Tanh network would either fail on key names or, with the
keys renamed, compute a different function without saying so.

:func:`convert_checkpoint` turns one ``model_<iteration>.pt`` into the
``policy.pt`` + ``policy_meta.json`` pair
:func:`~strands_robots.training.rl.checkpoint.load_deployable_actor` reads, with
``provider="rsl_rl"`` selecting :func:`build_actor_critic` here. rsl_rl's
observation normalizer keeps the same buffers and the same whitening
(``(x - mean) / (std + 1e-2)``) as
:class:`~strands_robots.training.rl.normalization.EmpiricalNormalization`, so it
is carried over as is. The architecture (layer widths, activation) is read from
the weights and from the run's ``params/agent.yaml``; nothing is guessed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

#: Activations rsl_rl's ``resolve_nn_activation`` names that this reader rebuilds.
ACTIVATIONS: tuple[str, ...] = ("elu", "selu", "relu", "lrelu", "tanh", "sigmoid", "identity")

_LINEAR_WEIGHT_RE = re.compile(r"^mlp\.(\d+)\.weight\Z")
_MODEL_RE = re.compile(r"^model_(\d+)\.pt\Z")


def build_actor_critic(
    num_actor_obs: int,
    num_critic_obs: int,
    num_actions: int,
    *,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    activation: str = "elu",
) -> Any:
    """Construct rsl_rl's deterministic MLP actor (torch required).

    The module keeps rsl_rl's parameter names (``mlp.0.weight`` ...), so an
    ``actor_state_dict`` loads without renaming. Only the actor is rebuilt: a
    deployed policy never evaluates the critic, whose width is kept in the
    metadata for completeness.

    Args:
        num_actor_obs: Width of the actor observation vector.
        num_critic_obs: Width of the critic observation vector (unused).
        num_actions: Number of action outputs.
        hidden_dims: Hidden layer widths, input side first.
        activation: One of :data:`ACTIVATIONS`, as the run's ``agent.yaml`` names it.

    Returns:
        An ``nn.Module`` whose ``act_inference`` returns the mean action.

    Raises:
        ValueError: If ``activation`` is not one this reader rebuilds.
    """
    import torch.nn as nn

    del num_critic_obs
    layers = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    if activation not in layers:
        raise ValueError(f"rsl_rl activation {activation!r} is not supported; expected one of {list(ACTIVATIONS)}")

    class RslRlActor(nn.Module):
        """``Linear -> activation -> ... -> Linear``, named as rsl_rl names it."""

        def __init__(self) -> None:
            super().__init__()
            modules: list[nn.Module] = []
            width = num_actor_obs
            for hidden in hidden_dims:
                modules += [nn.Linear(width, int(hidden)), layers[activation]()]
                width = int(hidden)
            modules.append(nn.Linear(width, num_actions))
            self.mlp = nn.Sequential(*modules)

        def act_inference(self, x: Any) -> Any:
            return self.mlp(x)

        def forward(self, x: Any) -> Any:
            return self.mlp(x)

    return RslRlActor()


def read_agent_activation(run_dir: str) -> str:
    """Return the actor activation the run's ``params/agent.yaml`` names.

    Read with a pattern rather than a YAML parser: the file is Isaac Lab's
    dump of the agent config, and only ``actor.activation`` is needed.

    Raises:
        FileNotFoundError: If the run has no ``params/agent.yaml``.
        ValueError: If the file names no actor activation.
    """
    path = Path(run_dir) / "params" / "agent.yaml"
    text = path.read_text(encoding="utf-8")
    block = re.search(r"^actor:\s*\n((?:[ \t]+.*\n?)*)", text, re.M)
    match = re.search(r"^[ \t]+activation:[ \t]*([A-Za-z_]+)", block.group(1) if block else "", re.M)
    if not match:
        raise ValueError(f"{path} names no actor activation (expected 'actor:' with 'activation:' under it)")
    return match.group(1).lower()


def latest_model(run_dir: str) -> str | None:
    """Return the ``model_<iteration>.pt`` with the highest iteration in *run_dir*."""
    models = []
    for path in Path(run_dir).glob("model_*.pt"):
        match = _MODEL_RE.match(path.name)
        if match:
            models.append((int(match.group(1)), path))
    return str(max(models)[1]) if models else None


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    activation: str | None = None,
    action_keys: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    """Write *model_path*'s actor as ``policy.pt`` + ``policy_meta.json`` under *out_dir*.

    Args:
        model_path: An rsl_rl ``model_<iteration>.pt``.
        out_dir: Directory to write into (created).
        activation: The actor activation; read from ``params/agent.yaml`` next
            to the checkpoint when omitted.
        action_keys: Names of the action outputs, in order, when known. Left
            empty, the deploying robot's action keys bind them, and a width
            mismatch is refused there.
        extra_meta: More fields for ``policy_meta.json`` (task, physics...).

    Returns:
        *out_dir*, which ``create_policy("rl", checkpoint_dir=...)`` loads.

    Raises:
        ValueError: If the file is not an rsl_rl actor checkpoint, or its layers
            do not form one MLP.
    """
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    actor = state.get("actor_state_dict") if isinstance(state, dict) else None
    if not isinstance(actor, dict):
        raise ValueError(
            f"{model_path} is not an rsl_rl checkpoint with an 'actor_state_dict' (rsl_rl 5.x, as Isaac Lab "
            f"3.0 writes it); keys: {sorted(state) if isinstance(state, dict) else type(state).__name__}"
        )
    linear = sorted((int(m.group(1)), key) for key in actor if (m := _LINEAR_WEIGHT_RE.match(key)))
    if not linear:
        raise ValueError(f"{model_path}: actor_state_dict has no mlp.<i>.weight layers")
    widths = [tuple(actor[key].shape) for _, key in linear]
    for (out_a, _), (_, in_b) in zip(widths, widths[1:], strict=False):
        if out_a != in_b:
            raise ValueError(f"{model_path}: actor layers {widths} do not chain into one MLP")
    num_actor_obs = int(widths[0][1])
    num_actions = int(widths[-1][0])
    hidden_dims = [int(w[0]) for w in widths[:-1]]
    run_dir = os.path.dirname(os.path.abspath(model_path))
    activation = (activation or read_agent_activation(run_dir)).lower()
    critic = state.get("critic_state_dict") or {}
    critic_in = next((tuple(v.shape)[1] for k, v in critic.items() if k == "mlp.0.weight"), num_actor_obs)

    module_state = {key: value for key, value in actor.items() if key.startswith("mlp.")}
    payload: dict[str, Any] = {"actor_critic": module_state, "iteration": int(state.get("iter", 0) or 0)}
    norm = {
        key.removeprefix("obs_normalizer."): value for key, value in actor.items() if key.startswith("obs_normalizer.")
    }
    if norm:
        payload["actor_norm"] = norm
    # Rebuild once before writing, so a checkpoint this reader cannot load is
    # refused here rather than at deployment.
    module = build_actor_critic(
        num_actor_obs, int(critic_in), num_actions, hidden_dims=tuple(hidden_dims), activation=activation
    )
    module.load_state_dict(module_state)

    contract = (extra_meta or {}).get("deploy_contract")
    if isinstance(contract, dict):
        from strands_robots.training.rl.deploy_contract import complete_obs_layout, contract_problems

        if problems := contract_problems(contract, num_actor_obs=num_actor_obs, num_actions=num_actions):
            raise ValueError(f"{model_path}: the run's deploy contract does not fit its actor: {'; '.join(problems)}")
        contract = complete_obs_layout(contract, num_actor_obs=num_actor_obs)
        extra_meta = {**(extra_meta or {}), "deploy_contract": contract}
        action_keys = list(contract["action_keys"])

    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, os.path.join(out_dir, "policy.pt"))
    meta = {
        "provider": "rsl_rl",
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": int(critic_in),
        "num_actions": num_actions,
        # The actor reads one concatenated vector: Isaac Lab's "policy"
        # observation group. An observation may carry it whole as
        # ``policy_obs`` or element by element as ``policy_obs.<i>``.
        "actor_obs_keys": [f"policy_obs.{i}" for i in range(num_actor_obs)],
        "action_keys": list(action_keys or []),
        "hidden_dims": hidden_dims,
        "activation": activation,
        "obs_normalization": bool(norm),
        "iteration": payload["iteration"],
        "source_checkpoint": os.path.abspath(model_path),
        **(extra_meta or {}),
    }
    with open(os.path.join(out_dir, "policy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return out_dir
