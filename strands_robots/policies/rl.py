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
from strands_robots.utils import boolean_flag_error, name_list_error, sequence_length

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
        raw_actions: For an Isaac Lab export only. ``False`` (the default)
            turns the actor's output into joint position targets through the
            run's deploy contract - joints bound by name, ``offset + scale *
            action`` - and refuses an export that carries no contract. ``True``
            returns the network's raw outputs bound by position, for parity
            checks against Isaac Lab's own exported policy; they are not joint
            angles.
        joint_map: For an Isaac Lab export only: ``{contract_joint: robot_key}``
            where the deploying robot names a joint differently. Without it a
            contract joint binds to the robot key of the same name, or of the
            same name without its ``_joint`` suffix (Isaac Lab's
            ``FL_hip_joint`` is the MuJoCo Go2 actuator ``FL_hip``) when that
            pairing is one-to-one.
        command: For an Isaac Lab export whose actor reads a command term
            (``generated_commands``, e.g. a velocity command ``[vx, vy, wz]``):
            the command the observation is built with. Default: zeros, i.e.
            stand still.
        obs_terms: For an Isaac Lab export: ``{term: values}`` for an
            observation term strands cannot compute (a camera, a contact
            sensor), or to replace one it can (``height_scan`` off flat ground).
        **kwargs: Ignored, for factory uniformity.

    Raises:
        ValueError: If ``checkpoint_dir`` is missing or blank (there is no
            default checkpoint: without one there is no trained actor to run),
            or it is an Isaac Lab export whose deploy contract is missing or
            does not fit the actor and ``raw_actions`` was not asked for.
        FileNotFoundError: If the directory holds no ``policy.pt`` /
            ``policy_meta.json``.
    """

    def __init__(
        self,
        checkpoint_dir: str = "",
        device: str = "cpu",
        raw_actions: bool = False,
        joint_map: dict[str, str] | None = None,
        command: list[float] | None = None,
        obs_terms: dict[str, list[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        if not checkpoint_dir or not str(checkpoint_dir).strip():
            raise ValueError(
                "checkpoint_dir is required for the 'rl' policy provider: pass the "
                "directory a trainer wrote (TrainResult.checkpoint_dir), e.g. "
                "create_policy('rl', checkpoint_dir=result.checkpoint_dir)"
            )
        from strands_robots.training.rl.checkpoint import load_deployable_actor

        if error := boolean_flag_error(raw_actions, "raw_actions", "rl"):
            raise ValueError(error)
        self._actor: DeployableActor = load_deployable_actor(str(checkpoint_dir).strip(), device=device)
        self._device = device
        if joint_map is not None and not (
            isinstance(joint_map, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in joint_map.items())
        ):
            raise ValueError(f"joint_map must be a dict of contract joint name -> robot key, got {joint_map!r}")
        self._joint_map = dict(joint_map or {})
        self._command = [float(v) for v in command] if command is not None else None
        self._obs_terms = {str(k): [float(x) for x in v] for k, v in (obs_terms or {}).items()}
        # The actor's previous raw output, which Isaac Lab's ``last_action``
        # term feeds back; zero at the start of every episode (see reset()).
        self._last_raw_action: list[float] = [0.0] * self._actor.num_actions
        self._contract = None if raw_actions else self._actor.deploy_contract
        if not raw_actions and self._actor.isaaclab_task:
            self._check_contract(str(checkpoint_dir).strip())
        self.robot_state_keys: list[str] = []
        logger.info(
            "RL checkpoint policy loaded: provider=%s iteration=%s actor_obs=%d actions=%d",
            self._actor.provider,
            self._actor.iteration,
            len(self._actor.actor_obs_keys),
            self._actor.num_actions,
        )

    def _check_contract(self, checkpoint_dir: str) -> None:
        """Refuse an Isaac Lab export whose outputs this policy cannot turn into joint targets."""
        from strands_robots.training.rl.deploy_contract import contract_problems

        task = self._actor.isaaclab_task
        if self._contract is None:
            raise ValueError(
                f"{checkpoint_dir} is an Isaac Lab {task} actor exported without its deploy contract, so "
                "nothing records which joint each output drives (the order differs between the PhysX "
                "and Newton presets), nor the action scale and default-pose offset Isaac Lab applies: "
                "commanding its raw outputs as joint angles puts a Go2 up to 1.7 rad off where Isaac "
                "Lab would, and it falls. Re-export the run with train_policy(action='export'), which "
                "now writes the contract from the run's IO descriptors - or pass raw_actions=True for "
                "the network's raw outputs, bound by position, which are not joint angles."
            )
        problems = contract_problems(
            self._contract, num_actor_obs=len(self._actor.actor_obs_keys), num_actions=self._actor.num_actions
        )
        if problems:
            raise ValueError(f"{checkpoint_dir}: the deploy contract cannot drive this actor: {'; '.join(problems)}")

    #: ``False``: the actor was trained against a reward function, not language,
    #: so the task envelopes say the instruction they echo was never read.
    reads_instruction: bool = False
    #: The words the task envelope uses for what the actor commands instead.
    instruction_free_actions: str | None = "the trained actor's per-step commands"

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

        The checkpoint's own ``action_keys`` when it recorded them (for an Isaac
        Lab export, the deploy contract's joints in the run's order), else the
        keys :meth:`set_robot_state_keys` supplied.
        """
        if self._contract is not None:
            return list(self._contract["action_keys"])
        return list(self._actor.action_keys or self.robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode: the ``last_action`` an Isaac Lab actor reads is zero again."""
        super().reset(seed)
        self._last_raw_action = [0.0] * self._actor.num_actions

    def _build_policy_obs(self, observation: dict[str, Any]) -> list[float]:
        """The actor's ``policy_obs`` vector built from a strands observation, per the deploy contract.

        See :func:`~strands_robots.training.rl.deploy_contract.build_policy_obs`.
        Joints are read under the robot key each contract joint binds to
        (:func:`bind_contract_joints` against the observation's own joint keys).
        """
        from strands_robots.training.rl.deploy_contract import build_policy_obs

        assert self._contract is not None
        joints = list(self._contract["action_keys"])
        present = [key for key in observation if isinstance(key, str) and not key.endswith(".vel")]
        try:
            keys = bind_contract_joints(
                joints,
                [k for k in present if k in joints or _bare_joint_name(k) in {_bare_joint_name(j) for j in joints}],
                self._joint_map,
            )
            return build_policy_obs(
                self._contract,
                observation,
                joint_keys=keys,
                last_action=self._last_raw_action,
                command=self._command,
                obs_terms=self._obs_terms,
            )
        except ValueError as exc:  # DeployContractError is a ValueError
            raise ValueError(
                f"the Isaac Lab {self._actor.isaaclab_task} actor's observation cannot be built from this "
                f"robot's state: {exc}"
            ) from None

    def contract_joint_binding(self) -> dict[str, str]:
        """``{contract_joint: robot_key}`` for the robot bound by :meth:`set_robot_state_keys`.

        What an engine needs to run the contract's actuator model on the right
        actuators. Empty without a contract.
        """
        if self._contract is None:
            return {}
        return self._contract_binding(list(self._contract["action_keys"]))

    def _contract_binding(self, contract_joints: list[str]) -> dict[str, str]:
        """The robot key each contract joint drives (see :func:`bind_contract_joints`)."""
        try:
            return bind_contract_joints(contract_joints, list(self.robot_state_keys), self._joint_map)
        except ValueError as exc:
            raise ValueError(f"the Isaac Lab {self._actor.isaaclab_task} actor cannot bind: {exc}") from None

    @property
    def deploy_contract(self) -> dict[str, Any] | None:
        """The Isaac Lab deploy contract the actions are produced through, or ``None``.

        Its ``obs_layout`` names the observation terms, in order, the actor's
        ``policy_obs`` vector is concatenated from; ``quat_order`` and
        ``base_velocity_frame`` say which conventions those terms use.
        """
        return self._contract

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

        if (
            self._contract is not None
            and "policy_obs" not in observation_dict
            and not any(key in observation_dict for key in self._actor.actor_obs_keys[:1])
        ):
            observation_dict = {**observation_dict, "policy_obs": self._build_policy_obs(observation_dict)}
        observation_dict = _expand_vector_observations(observation_dict, self._actor.actor_obs_keys)
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
        self._last_raw_action = [float(v) for v in action]
        if self._contract is not None:
            from strands_robots.training.rl.deploy_contract import apply_action_contract

            targets = apply_action_contract(self._contract, [float(v) for v in action])
            binding = self._contract_binding(action_keys)
            return [{binding[joint]: value for joint, value in targets.items()}]
        return [{key: float(action[i]) for i, key in enumerate(action_keys)}]


def _bare_joint_name(name: str) -> str:
    """A joint name without the ``_joint`` suffix URDF/USD exports add, lowercased."""
    lowered = name.lower()
    return lowered[: -len("_joint")] if lowered.endswith("_joint") else lowered


def bind_contract_joints(
    contract_joints: list[str], robot_keys: list[str], joint_map: dict[str, str]
) -> dict[str, str]:
    """Pair each contract joint with the robot key it drives, by name only.

    Order: *joint_map*, then the identical name, then the name without its
    ``_joint`` suffix (case-insensitive) when exactly one robot key has it.
    An empty *robot_keys* (no robot bound) keeps the contract's own names.

    Raises:
        ValueError: A contract joint pairs with no robot key, or two pair with
            the same one. Position is never used to fill the gap.
    """
    if not robot_keys:
        return {joint: joint_map.get(joint, joint) for joint in contract_joints}
    present = set(robot_keys)
    by_bare: dict[str, list[str]] = {}
    for key in robot_keys:
        by_bare.setdefault(_bare_joint_name(key), []).append(key)
    binding: dict[str, str] = {}
    unbound: list[str] = []
    for joint in contract_joints:
        if joint in joint_map and joint_map[joint] in present:
            binding[joint] = joint_map[joint]
        elif joint in present:
            binding[joint] = joint
        elif len(by_bare.get(_bare_joint_name(joint), [])) == 1:
            binding[joint] = by_bare[_bare_joint_name(joint)][0]
        else:
            unbound.append(joint)
    doubled = sorted({key for key in binding.values() if list(binding.values()).count(key) > 1})
    if unbound or doubled:
        raise ValueError(
            f"the actor drives joints {unbound or contract_joints} that this robot's action keys {robot_keys} "
            f"do not name one-to-one{f' ({doubled} would be driven twice)' if doubled else ''}; its outputs are "
            "bound by joint name, never by position - pass joint_map={contract_joint: robot_key} for the rest"
        )
    return binding


def _expand_vector_observations(observation_dict: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Let an observation carry ``name.0 .. name.<n-1>`` as one vector under ``name``.

    An exported Isaac Lab actor reads one concatenated observation group
    (``policy_obs.<i>``); a caller holding that vector passes it whole. Only a
    key the observation lacks is filled, and only from a 1-D sequence whose
    length covers every index the actor reads, so a short vector is still
    reported as missing keys rather than padded.
    """
    wanted: dict[str, list[int]] = {}
    for key in keys:
        if key in observation_dict:
            continue
        base, _, index = key.rpartition(".")
        if base and index.isdigit():
            wanted.setdefault(base, []).append(int(index))
    if not wanted:
        return observation_dict
    expanded = dict(observation_dict)
    for base, indices in wanted.items():
        vector = observation_dict.get(base)
        if vector is None or isinstance(vector, (str, bytes)) or sequence_length(vector) is None:
            continue
        try:
            values = [float(v) for v in vector]
        except (TypeError, ValueError):
            continue
        if max(indices) < len(values):
            expanded.update({f"{base}.{i}": values[i] for i in indices})
    return expanded
