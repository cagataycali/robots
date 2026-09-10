# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An RL training checkpoint is deployable through the policy factory.

The RL trainers write ``policy.pt`` + ``policy_meta.json`` and
:meth:`BaseRLAlgo.save_checkpoint` calls the pair "a deployable-policy metadata
file", with ``act_inference`` docstringed on all three backends as "the
deployable policy". Nothing read them: no provider in
``strands_robots/registry/policies.json`` loaded an RL checkpoint, so
``create_policy("rl")`` raised ``Unknown policy provider`` and a trained actor
could not drive a robot through ``run_policy`` / ``eval_policy`` like every
other provider.

These cells pin the reader end to end: the provider resolves, each backend's
architecture is rebuilt from its own ``build_actor_critic`` (so a FastSAC
actor's ``2 * num_actions`` output and its ``tanh`` squash are honoured rather
than guessed), the observation normalizer is restored frozen, the trained
``actor_obs_keys`` order is respected, and every way the pair can be unusable is
refused by name instead of defaulted into a fabricated command.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from strands_robots.policies import create_policy

torch = pytest.importorskip("torch")


def _write_checkpoint(
    directory: str,
    *,
    provider: str = "ppo",
    actor_obs_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    hidden_dims: tuple[int, ...] = (8, 8),
    normalizer_mean: list[float] | None = None,
    meta_overrides: dict[str, Any] | None = None,
    drop_meta_fields: tuple[str, ...] = (),
    write_weights: bool = True,
) -> str:
    """Write a checkpoint pair shaped exactly as ``save_checkpoint`` writes one.

    Builds the real backend module for ``provider`` so the saved ``state_dict``
    is the one that backend produces, rather than a hand-rolled approximation
    the loader could not distinguish from a genuine checkpoint.
    """
    from importlib import import_module

    keys = actor_obs_keys if actor_obs_keys is not None else ["a", "b"]
    acts = action_keys if action_keys is not None else ["j1", "j2", "j3"]
    module = import_module(f"strands_robots.training.rl.{provider}").build_actor_critic(
        len(keys), len(keys), len(acts), hidden_dims=hidden_dims
    )
    os.makedirs(directory, exist_ok=True)
    state: dict[str, Any] = {"actor_critic": module.state_dict(), "iteration": 7, "provider": provider}
    if normalizer_mean is not None:
        from strands_robots.training.rl.normalization import EmpiricalNormalization

        norm = EmpiricalNormalization(len(keys), "cpu")
        norm.train()
        norm.update(torch.tensor([normalizer_mean], dtype=torch.float32).repeat(64, 1))
        state["actor_norm"] = norm.state_dict()
    if write_weights:
        torch.save(state, os.path.join(directory, "policy.pt"))

    meta: dict[str, Any] = {
        "provider": provider,
        "num_actor_obs": len(keys),
        "num_critic_obs": len(keys),
        "num_actions": len(acts),
        "actor_obs_keys": keys,
        "action_keys": acts,
        "hidden_dims": list(hidden_dims),
        "iteration": 7,
    }
    meta.update(meta_overrides or {})
    for field in drop_meta_fields:
        meta.pop(field, None)
    with open(os.path.join(directory, "policy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return directory


def _rl_policy(**kwargs: Any) -> Any:
    """Resolve the provider through the factory and narrow to its concrete class.

    Going through ``create_policy`` is what pins the registry wiring; the
    ``isinstance`` narrows the ``Policy`` return type onto the provider-specific
    surface the cells read.
    """
    from strands_robots.policies.rl import RLCheckpointPolicy

    policy = create_policy("rl", **kwargs)
    assert isinstance(policy, RLCheckpointPolicy)
    return policy


async def _act(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    chunk = await policy.get_actions(observation, "")
    assert len(chunk) == 1, "an RL actor is a per-step controller: one tick per call"
    return chunk[0]


def test_rl_provider_is_registered_and_resolvable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``create_policy("rl")`` yields a Policy bound to the checkpoint's identity."""
    from strands_robots.policies import list_providers

    assert "rl" in list_providers()
    ckpt = _write_checkpoint(str(tmp_path / "ck"), provider="ppo")
    policy = _rl_policy(checkpoint_dir=ckpt)
    assert policy.provider_name == "rl"
    assert policy.trained_by == "ppo"
    assert policy.requires_images is False
    assert policy.actor_obs_keys == ["a", "b"]
    assert policy.action_keys == ["j1", "j2", "j3"]


@pytest.mark.parametrize("provider", ["ppo", "fast_sac", "fast_td3"])
def test_every_backend_checkpoint_rolls_out(tmp_path, provider: str) -> None:  # type: ignore[no-untyped-def]
    """Each backend's saved actor loads and commands one finite float per action key.

    FastSAC's actor emits ``2 * num_actions`` (a mean/log-std pair) where PPO's
    and FastTD3's emit ``num_actions``, and ``policy_meta.json`` records
    ``num_actions`` for all three - so a loader that sized the network from the
    metadata alone would build the wrong graph for FastSAC.
    """
    ckpt = _write_checkpoint(str(tmp_path / provider), provider=provider)
    policy = _rl_policy(checkpoint_dir=ckpt)
    action = pytest.importorskip("asyncio").run(_act(policy, {"a": 0.25, "b": -0.5}))

    assert list(action) == ["j1", "j2", "j3"]
    assert all(isinstance(v, float) for v in action.values())
    assert all(abs(v) < 1e6 for v in action.values())
    if provider in ("fast_sac", "fast_td3"):
        # Both off-policy backends squash through tanh, so a deployed command is
        # inside the action interval they were trained to emit.
        assert all(-1.0 <= v <= 1.0 for v in action.values()), action


def test_loaded_normalizer_is_frozen_at_the_statistics_the_run_finished_on(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The restored observation normalizer never folds a deployed observation in.

    ``EmpiricalNormalization`` only updates while training, so a deployed actor
    must hold it in eval mode: otherwise every rollout would shift the whitening
    statistics the trained weights expect, and the same state would command a
    drifting action. Read through the public reader the provider is built on.
    """
    from strands_robots.training.rl import load_deployable_actor

    ckpt = _write_checkpoint(str(tmp_path / "ck"), normalizer_mean=[3.0, -1.0])
    actor = load_deployable_actor(ckpt)

    normalizer = actor.normalizer
    assert normalizer is not None
    assert normalizer.training is False
    count_before = int(normalizer.count)
    mean_before = normalizer.mean.clone()

    extreme = torch.full((4, 2), 900.0)
    assert torch.allclose(actor.act(extreme), actor.act(extreme))
    assert int(normalizer.count) == count_before
    assert torch.allclose(normalizer.mean, mean_before)


def test_action_is_deterministic_across_calls(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The same observation commands the same action every time.

    A deployed actor is the deterministic (mean) action with gradients disabled,
    so a rollout is reproducible rather than sampling a fresh command per tick.
    """
    asyncio = pytest.importorskip("asyncio")
    ckpt = _write_checkpoint(str(tmp_path / "ck"), normalizer_mean=[3.0, -1.0])
    policy = _rl_policy(checkpoint_dir=ckpt)

    first = asyncio.run(_act(policy, {"a": 900.0, "b": 900.0}))
    second = asyncio.run(_act(policy, {"a": 900.0, "b": 900.0}))
    assert first == second


def test_actor_obs_keys_are_read_in_the_trained_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The trained key order drives the input vector, not the observation's order.

    The order ``actor_obs_keys`` records is part of the weights: an actor trained
    on ``["a", "b"]`` fed ``[b, a]`` is being given a different input. So a
    reshuffled observation *dict* must produce the same action, while genuinely
    swapping the two values must not.
    """
    asyncio = pytest.importorskip("asyncio")
    ckpt = _write_checkpoint(str(tmp_path / "ck"), actor_obs_keys=["a", "b"])
    policy = _rl_policy(checkpoint_dir=ckpt)

    in_order = asyncio.run(_act(policy, {"a": 1.0, "b": -1.0}))
    reordered_dict = asyncio.run(_act(policy, {"b": -1.0, "a": 1.0}))
    swapped_values = asyncio.run(_act(policy, {"a": -1.0, "b": 1.0}))

    assert in_order == reordered_dict, "binding is by name, so dict order is irrelevant"
    assert in_order != swapped_values, "swapping the values is a different observation"


def test_extra_observation_keys_are_ignored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A richer observation than the actor was trained on still drives it.

    ``run_policy`` hands a policy every scalar the robot reports; an actor
    trained on a subset reads its own keys and ignores the rest.
    """
    asyncio = pytest.importorskip("asyncio")
    ckpt = _write_checkpoint(str(tmp_path / "ck"), actor_obs_keys=["a", "b"])
    policy = _rl_policy(checkpoint_dir=ckpt)

    lean = asyncio.run(_act(policy, {"a": 0.5, "b": 0.5}))
    rich = asyncio.run(_act(policy, {"a": 0.5, "b": 0.5, "c": 99.0, "b.vel": -3.0}))
    assert lean == rich


def test_missing_observation_key_is_refused_by_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An absent trained key refuses instead of defaulting to zero.

    Substituting a value would command the robot from a state it is not in,
    under a successful-looking rollout.
    """
    asyncio = pytest.importorskip("asyncio")
    ckpt = _write_checkpoint(str(tmp_path / "ck"), actor_obs_keys=["a", "b"])
    policy = _rl_policy(checkpoint_dir=ckpt)

    with pytest.raises(ValueError, match="omits actor_obs_keys"):
        asyncio.run(_act(policy, {"a": 0.5}))


def test_action_key_count_must_match_the_actors_width(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A robot with a different actuator count than the actor is refused.

    Binding 3 outputs onto 2 actuators would silently drop the last command;
    onto 4 it would leave one actuator unnamed.
    """
    asyncio = pytest.importorskip("asyncio")
    # The network keeps its 3 outputs; only the recorded vocabulary is empty,
    # which is what a checkpoint saved without a robot bound looks like.
    ckpt = _write_checkpoint(str(tmp_path / "ck"), meta_overrides={"action_keys": []})
    policy = _rl_policy(checkpoint_dir=ckpt)

    # No keys at all: the checkpoint recorded none and none were supplied.
    with pytest.raises(ValueError, match="no action keys"):
        asyncio.run(_act(policy, {"a": 0.0, "b": 0.0}))

    policy.set_robot_state_keys(["only", "two"])
    with pytest.raises(ValueError, match="emits 3 actions but 2 action keys"):
        asyncio.run(_act(policy, {"a": 0.0, "b": 0.0}))


def test_checkpoint_action_keys_win_over_the_robots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A checkpoint that names its actuators is not overridden by the robot's keys.

    ``set_robot_state_keys`` is the fallback for a checkpoint saved without a
    robot bound; the trained vocabulary is what the outputs mean.
    """
    asyncio = pytest.importorskip("asyncio")
    ckpt = _write_checkpoint(str(tmp_path / "ck"), action_keys=["j1", "j2", "j3"])
    policy = _rl_policy(checkpoint_dir=ckpt)
    policy.set_robot_state_keys(["other_a", "other_b", "other_c"])

    action = asyncio.run(_act(policy, {"a": 0.0, "b": 0.0}))
    assert list(action) == ["j1", "j2", "j3"]


@pytest.mark.parametrize(
    ("kwargs_factory", "expected_error", "match"),
    [
        pytest.param(lambda p: {"checkpoint_dir": ""}, ValueError, "checkpoint_dir is required", id="blank-dir"),
        pytest.param(lambda p: {"checkpoint_dir": "   "}, ValueError, "checkpoint_dir is required", id="whitespace"),
        pytest.param(
            lambda p: {"checkpoint_dir": str(p)}, FileNotFoundError, "no policy_meta.json", id="no-meta-at-all"
        ),
    ],
)
def test_unusable_checkpoint_arguments_are_refused(tmp_path, kwargs_factory, expected_error, match) -> None:  # type: ignore[no-untyped-def]
    """Every way the pair can be absent refuses, naming what is missing."""
    with pytest.raises(expected_error, match=match):
        create_policy("rl", **kwargs_factory(tmp_path))


def test_weights_without_metadata_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``policy.pt`` alone cannot be bound to a robot, so it is refused.

    The metadata is what names the observations the weights expect and the
    actuators they drive; weights alone are not a deployable policy.
    """
    ckpt = _write_checkpoint(str(tmp_path / "ck"))
    os.remove(os.path.join(ckpt, "policy_meta.json"))
    with pytest.raises(FileNotFoundError, match="no policy_meta.json"):
        create_policy("rl", checkpoint_dir=ckpt)


def test_metadata_without_weights_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Metadata describing weights that are not there refuses."""
    ckpt = _write_checkpoint(str(tmp_path / "ck"), write_weights=False)
    with pytest.raises(FileNotFoundError, match="no policy.pt"):
        create_policy("rl", checkpoint_dir=ckpt)


@pytest.mark.parametrize(
    ("dropped", "match"),
    [
        pytest.param(("provider",), "provider", id="provider"),
        pytest.param(("num_actor_obs",), "num_actor_obs", id="num_actor_obs"),
        pytest.param(("num_actions",), "num_actions", id="num_actions"),
        pytest.param(("actor_obs_keys",), "actor_obs_keys", id="actor_obs_keys"),
        pytest.param(("hidden_dims",), "hidden_dims", id="hidden_dims"),
    ],
)
def test_incomplete_metadata_names_the_missing_field(tmp_path, dropped, match) -> None:  # type: ignore[no-untyped-def]
    """A metadata field the deployment needs is reported, not defaulted.

    Each of these decides the network's shape or what its inputs and outputs
    mean, so a default would build a different actor or bind it to the wrong
    joints while reporting success.
    """
    ckpt = _write_checkpoint(str(tmp_path / "ck"), drop_meta_fields=dropped)
    with pytest.raises(ValueError, match=f"omits required field.*{match}"):
        create_policy("rl", checkpoint_dir=ckpt)


def test_unknown_trainer_in_metadata_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A provider with no known architecture refuses rather than guessing one.

    ``provider`` is what selects the actor architecture, because the output
    width is not derivable from ``num_actions``.
    """
    ckpt = _write_checkpoint(str(tmp_path / "ck"), meta_overrides={"provider": "some_future_algo"})
    with pytest.raises(ValueError, match="no known actor architecture"):
        create_policy("rl", checkpoint_dir=ckpt)


def test_metadata_that_is_not_an_object_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A ``policy_meta.json`` holding a non-object refuses with its type."""
    ckpt = _write_checkpoint(str(tmp_path / "ck"))
    with open(os.path.join(ckpt, "policy_meta.json"), "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    with pytest.raises(ValueError, match="must be a JSON object, got list"):
        create_policy("rl", checkpoint_dir=ckpt)


def test_loaded_actor_matches_the_trainers_own_inference(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The deployed command equals what the training graph would have produced.

    The point of rebuilding through the backend's own ``build_actor_critic`` is
    that no reimplementation can drift from it: the same weights and the same
    observation give the same action on both sides.
    """
    asyncio = pytest.importorskip("asyncio")
    from strands_robots.training.rl.ppo import build_actor_critic

    ckpt = _write_checkpoint(str(tmp_path / "ck"), actor_obs_keys=["a", "b"], hidden_dims=(8, 8))
    reference = build_actor_critic(2, 2, 3, hidden_dims=(8, 8))
    state = torch.load(os.path.join(ckpt, "policy.pt"), map_location="cpu", weights_only=True)
    reference.load_state_dict(state["actor_critic"])
    reference.eval()
    with torch.no_grad():
        expected = reference.act_inference(torch.tensor([[0.3, -0.7]], dtype=torch.float32))[0]

    action = asyncio.run(_act(_rl_policy(checkpoint_dir=ckpt), {"a": 0.3, "b": -0.7}))
    for i, key in enumerate(["j1", "j2", "j3"]):
        assert action[key] == pytest.approx(float(expected[i]), abs=1e-6)


pytest.importorskip("mujoco")


def test_a_trained_checkpoint_drives_the_robot_it_was_trained_on(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End to end: train in sim, then roll the checkpoint out through run_policy.

    The closed loop the checkpoint format was written for -
    ``create_trainer("ppo").train(...)`` to ``run_policy(policy_provider="rl")``
    - with no hand-built glue in between.
    """
    import strands_robots as sr
    from strands_robots.training import create_trainer
    from strands_robots.training.rl import RLTrainSpec, SimEnv

    target = 0.4

    def reward(engine: Any) -> float:
        observation = engine.get_observation(skip_images=True)
        observation = observation.get("observation") or observation
        return -abs(float(observation["Elbow"]) - target)

    def make_env() -> Any:
        return SimEnv(
            sr.Robot("so100", mode="sim"),
            actor_obs_keys=["Elbow", "Elbow.vel"],
            reward_terms=[reward],
            max_episode_steps=10,
        )

    result = create_trainer("ppo").train(
        RLTrainSpec(
            env_factory=make_env,
            output_dir=str(tmp_path),
            total_timesteps=40,
            rollout_steps=20,
            num_mini_batches=2,
            num_learning_epochs=1,
            hidden_dims=(16, 16),
            seed=0,
        )
    )
    assert result.status == "success"

    policy = _rl_policy(checkpoint_dir=result.checkpoint_dir)
    assert policy.trained_by == "ppo"
    assert policy.actor_obs_keys == ["Elbow", "Elbow.vel"]

    sim = sr.Robot("so100", mode="sim")
    rollout = sim.run_policy(
        robot_name="so100", policy_object=policy, n_steps=20, control_frequency=50.0, control_substeps=5
    )
    assert rollout["status"] == "success"
    # The actor commanded the robot's own actuators, so the arm left its start
    # pose rather than sitting at zero under a successful-looking rollout.
    observation = sim.get_observation(skip_images=True)
    observation = observation.get("observation") or observation
    assert any(abs(float(observation[key])) > 1e-6 for key in policy.action_keys)
