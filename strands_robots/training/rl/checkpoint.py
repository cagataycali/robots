# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load an RL checkpoint's deterministic actor for deployment.

:meth:`~strands_robots.training.rl.base_algo.BaseRLAlgo.save_checkpoint` writes
``policy.pt`` + ``policy_meta.json`` and calls the pair "a deployable
checkpoint". This module is the reader that makes that true: it rebuilds the
network the ``state_dict`` was saved from, restores the weights and the frozen
observation normalizer, and hands back the deterministic action function plus
the vocabularies the actor was trained against.

The reader lives beside the writer because the two share one format. The three
backends do not share one *network*: PPO's actor emits ``num_actions`` raw
Gaussian means, FastTD3's emits ``num_actions`` through a ``tanh``, and
FastSAC's emits ``2 * num_actions`` (a mean/log-std pair) and squashes the mean.
``policy_meta.json`` records ``num_actions`` for all three, so the output width
is not derivable from the metadata alone -- ``provider`` is what selects the
architecture, and each backend's own ``act_inference`` supplies the squash. A
reader that guessed instead would silently mis-scale a FastSAC actor's commands.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

#: Module owning the ``build_actor_critic`` for each RL backend that writes a
#: checkpoint, keyed by the ``provider`` field ``save_checkpoint`` stamps.
_BACKEND_MODULES: dict[str, str] = {
    "ppo": "strands_robots.training.rl.ppo",
    "fast_sac": "strands_robots.training.rl.fast_sac",
    "fast_td3": "strands_robots.training.rl.fast_td3",
}

#: ``policy_meta.json`` fields a deployment needs. Absent any one of them the
#: actor cannot be rebuilt or bound to a robot, so the read refuses by name
#: rather than substituting a default that would command the wrong joints.
_REQUIRED_META_FIELDS = (
    "provider",
    "num_actor_obs",
    "num_critic_obs",
    "num_actions",
    "actor_obs_keys",
    "hidden_dims",
)


@dataclass(frozen=True)
class DeployableActor:
    """An RL checkpoint's deterministic actor plus the vocabularies it was trained on.

    Attributes:
        provider: Trainer that wrote the checkpoint (``"ppo"``, ``"fast_sac"``,
            ``"fast_td3"``).
        actor_obs_keys: Ordered observation keys the actor input is concatenated
            from. The order is part of the trained weights, not a preference.
        action_keys: Ordered action keys the outputs drive - the robot's
            ``robot_action_keys``, not its joint list. Empty when the checkpoint
            was written without a robot bound.
        num_actions: Number of action outputs.
        iteration: Training iteration the checkpoint was saved at, or ``None``.
        module: The loaded ``nn.Module``, in eval mode.
        normalizer: The frozen observation normalizer, or ``None`` when the run
            trained with ``normalize_obs=False``.
    """

    provider: str
    actor_obs_keys: list[str]
    action_keys: list[str]
    num_actions: int
    iteration: int | None
    module: Any
    normalizer: Any

    def act(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action for a batch of actor observations.

        Whitens with the frozen normalizer (when the run trained with one) and
        calls the backend's own ``act_inference``, so the squash a backend
        applies to its deployable action is applied here too.

        Args:
            actor_obs: Batched observations, shape ``(batch, num_actor_obs)``.

        Returns:
            Actions of shape ``(batch, num_actions)``.
        """
        import torch

        with torch.no_grad():
            x = actor_obs if self.normalizer is None else self.normalizer(actor_obs, update=False)
            return self.module.act_inference(x)


def read_checkpoint_meta(checkpoint_dir: str) -> dict[str, Any]:
    """Read and validate ``policy_meta.json`` from a checkpoint directory.

    Args:
        checkpoint_dir: Directory holding ``policy.pt`` + ``policy_meta.json``,
            as returned by ``TrainResult.checkpoint_dir``.

    Returns:
        The parsed metadata mapping.

    Raises:
        FileNotFoundError: When ``policy_meta.json`` is absent - the checkpoint
            carries weights but nothing naming the observations they expect.
        ValueError: When the metadata is not a mapping, omits a required field,
            or names a ``provider`` with no known architecture. Each is a
            checkpoint that cannot be deployed correctly, and a default would
            bind the actor to the wrong observations or actuators.
    """
    path = os.path.join(checkpoint_dir, "policy_meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no policy_meta.json in checkpoint dir: {checkpoint_dir}; "
            "the file records the observation and action keys the actor was trained on, "
            "so weights alone cannot be bound to a robot"
        )
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"policy_meta.json in {checkpoint_dir} must be a JSON object, got {type(meta).__name__}")
    missing = [field for field in _REQUIRED_META_FIELDS if field not in meta]
    if missing:
        raise ValueError(f"policy_meta.json in {checkpoint_dir} omits required field(s): {', '.join(missing)}")
    provider = meta["provider"]
    if provider not in _BACKEND_MODULES:
        raise ValueError(
            f"policy_meta.json in {checkpoint_dir} names provider {provider!r}, "
            f"which has no known actor architecture; expected one of {sorted(_BACKEND_MODULES)}"
        )
    return meta


def load_deployable_actor(checkpoint_dir: str, device: str = "cpu") -> DeployableActor:
    """Load the deterministic actor from an RL training checkpoint.

    Rebuilds the backend's network through its own ``build_actor_critic`` - so
    the deployed graph is the one that was trained, not a reimplementation -
    restores ``policy.pt`` into it, and freezes both the module and the
    observation normalizer into eval mode.

    Args:
        checkpoint_dir: Directory holding ``policy.pt`` + ``policy_meta.json``.
        device: Torch device to load onto.

    Returns:
        A :class:`DeployableActor` whose ``act`` is the deployed command.

    Raises:
        FileNotFoundError: When ``policy.pt`` or ``policy_meta.json`` is absent.
        ValueError: When the metadata is unusable, per :func:`read_checkpoint_meta`.
    """
    import torch

    from strands_robots.training.rl.normalization import EmpiricalNormalization

    meta = read_checkpoint_meta(checkpoint_dir)
    weights = os.path.join(checkpoint_dir, "policy.pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"no policy.pt in checkpoint dir: {checkpoint_dir}")

    num_actor_obs = int(meta["num_actor_obs"])
    build = import_module(_BACKEND_MODULES[meta["provider"]]).build_actor_critic
    module = build(
        num_actor_obs,
        int(meta["num_critic_obs"]),
        int(meta["num_actions"]),
        hidden_dims=tuple(int(h) for h in meta["hidden_dims"]),
    ).to(device)
    # weights_only=True: the payload is state_dicts + an int + a str, matching
    # the loader BaseRLAlgo.load_checkpoint uses, so the legacy unpickler's
    # arbitrary-code-execution surface stays closed.
    state = torch.load(weights, map_location=device, weights_only=True)
    module.load_state_dict(state["actor_critic"])
    module.eval()

    normalizer = None
    if "actor_norm" in state:
        normalizer = EmpiricalNormalization(num_actor_obs, device)
        normalizer.load_state_dict(state["actor_norm"])
        # eval() is what freezes the running statistics: EmpiricalNormalization
        # only folds a batch while training, so a deployed actor whitens every
        # observation with the statistics the run finished on.
        normalizer.eval()

    return DeployableActor(
        provider=str(meta["provider"]),
        actor_obs_keys=[str(k) for k in meta["actor_obs_keys"]],
        action_keys=[str(k) for k in meta.get("action_keys") or []],
        num_actions=int(meta["num_actions"]),
        iteration=meta.get("iteration"),
        module=module,
        normalizer=normalizer,
    )
