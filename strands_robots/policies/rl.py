# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Roll out an actor trained by the RL trainers (``create_policy("rl")``).

The inference half of the RL loop. ``create_trainer("ppo" | "fast_sac" |
"fast_td3")`` trains against a :class:`~strands_robots.training.rl.env.SimEnv`
and writes ``policy.pt`` + ``policy_meta.json``; this provider loads that pair
and presents it as an ordinary :class:`~strands_robots.policies.base.Policy`, so
a trained actor drives a robot through the same
:meth:`~strands_robots.simulation.base.SimEngine.run_policy` /
:meth:`~strands_robots.simulation.base.SimEngine.eval_policy` path as every
other provider::

    result = create_trainer("ppo").train(spec)
    sim.run_policy(robot_name="so101", policy_provider="rl",
                   policy_config={"checkpoint_dir": result.checkpoint_dir})

``checkpoint_dir`` is spelled as the trainer spells it (``TrainResult.checkpoint_dir``,
``BaseRLAlgo.load_checkpoint``, ``latest_checkpoint``), so the value a caller
already holds is the value this provider takes.

The checkpoint's ``actor_obs_keys`` are read from the observation by name, in the
trained order, because that order is part of the weights: an actor trained on
``["1", "2", "1.vel"]`` fed ``["1", "1.vel", "2"]`` is being given a different
input. A key the observation does not carry is refused rather than defaulted -
substituting a zero would command a real robot from a fabricated state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands_robots.training.rl.checkpoint import DeployableActor

logger = logging.getLogger(__name__)


class RLCheckpointPolicy(Policy):
    """Deterministic rollout of an RL training checkpoint's actor.

    Args:
        checkpoint_dir: Directory holding ``policy.pt`` + ``policy_meta.json``,
            as returned by ``TrainResult.checkpoint_dir``.
        device: Torch device to load the actor onto (default ``"cpu"``; PPO on
            MuJoCo declares no GPU floor).
        **kwargs: Ignored, for factory uniformity.

    Raises:
        ValueError: If ``checkpoint_dir`` is missing or blank. There is no
            default checkpoint: without one there is no trained actor to run.
        FileNotFoundError: If the directory holds no ``policy.pt`` /
            ``policy_meta.json``.
    """

    def __init__(self, checkpoint_dir: str = "", device: str = "cpu", **kwargs: Any) -> None:
        if not checkpoint_dir or not str(checkpoint_dir).strip():
            raise ValueError(
                "checkpoint_dir is required for the 'rl' policy provider: pass the "
                "directory a trainer wrote (TrainResult.checkpoint_dir), e.g. "
                "create_policy('rl', checkpoint_dir=result.checkpoint_dir)"
            )
        from strands_robots.training.rl.checkpoint import load_deployable_actor

        self._actor: DeployableActor = load_deployable_actor(str(checkpoint_dir).strip(), device=device)
        self._device = device
        self.robot_state_keys: list[str] = []
        logger.info(
            "RL checkpoint policy loaded: provider=%s iteration=%s actor_obs=%d actions=%d",
            self._actor.provider,
            self._actor.iteration,
            len(self._actor.actor_obs_keys),
            self._actor.num_actions,
        )

    @property
    def provider_name(self) -> str:
        """Provider name for identification (always ``"rl"``)."""
        return "rl"

    @property
    def requires_images(self) -> bool:
        """RL actors trained through ``SimEnv`` consume scalar state only."""
        return False

    @property
    def trained_by(self) -> str:
        """Trainer that wrote the loaded checkpoint (``"ppo"``, ``"fast_sac"``, ``"fast_td3"``)."""
        return self._actor.provider

    @property
    def actor_obs_keys(self) -> list[str]:
        """Ordered observation keys the loaded actor was trained on."""
        return list(self._actor.actor_obs_keys)

    @property
    def action_keys(self) -> list[str]:
        """Ordered action keys the loaded actor's outputs drive.

        The checkpoint's own ``action_keys`` when it recorded them, else the keys
        :meth:`set_robot_state_keys` supplied.
        """
        return list(self._actor.action_keys or self.robot_state_keys)

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered action keys, used only if the checkpoint has none.

        A checkpoint written with a robot bound already names the actuators its
        outputs drive, and those win: they are what the actor was trained
        against. This is the fallback for a checkpoint saved without one.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return the actor's deterministic action for the current observation.

        A one-tick chunk: an RL actor is a per-step controller trained on the
        state it is given, so it has no horizon to predict over.

        Args:
            observation_dict: The observation, carrying one scalar per key named
                in the checkpoint's ``actor_obs_keys``.
            instruction: Ignored - an RL actor is trained against a reward
                function, not conditioned on language.
            **kwargs: Ignored.

        Returns:
            A single-element list holding one ``{action_key: float}`` dict.

        Raises:
            ValueError: If the observation omits a key the actor was trained on,
                or if no action keys are known. Both would otherwise command the
                robot from a fabricated state or bind outputs to the wrong
                actuators.
        """
        import torch

        missing = [key for key in self._actor.actor_obs_keys if key not in observation_dict]
        if missing:
            raise ValueError(
                f"observation omits actor_obs_keys the {self._actor.provider} checkpoint was "
                f"trained on: {missing}; present keys: {sorted(observation_dict)}"
            )
        action_keys = self.action_keys
        if not action_keys:
            raise ValueError(
                "no action keys: the checkpoint recorded none (it was trained without a robot "
                "bound) and set_robot_state_keys was not called, so the actor's "
                f"{self._actor.num_actions} outputs cannot be bound to actuators"
            )
        if len(action_keys) != self._actor.num_actions:
            raise ValueError(
                f"the {self._actor.provider} checkpoint's actor emits {self._actor.num_actions} "
                f"actions but {len(action_keys)} action keys are bound ({action_keys}); "
                "the robot does not match the one the actor was trained on"
            )

        obs = torch.tensor(
            [[float(observation_dict[key]) for key in self._actor.actor_obs_keys]],
            dtype=torch.float32,
            device=self._device,
        )
        action = self._actor.act(obs)[0]
        return [{key: float(action[i]) for i, key in enumerate(action_keys)}]
