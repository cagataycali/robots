"""Deterministic unit tests for the off-policy RL trainer (FastTD3).

These run in CI: they need ``torch`` but no MuJoCo, no model downloads and no
convergence assumptions - every env is a scripted fake engine behind the real
:class:`~strands_robots.training.rl.env.SimEnv`, so the TD3-specific contracts
(clipped double-Q wiring, delayed policy updates, bounded deterministic
actions, the truncation bootstrap through ``VecSimEnv``'s ``terminal_obs``, and
the checkpoint round-trip) are pinned in the fast CI lane.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from strands_robots.training import create_trainer, list_trainers
from strands_robots.training.base import TrainSpec

torch = pytest.importorskip("torch")

from strands_robots.training.rl import RLTrainSpec, SimEnv, VecSimEnv  # noqa: E402
from strands_robots.training.rl.fast_td3 import FastTd3Trainer, _build_actor_critic  # noqa: E402


class _FakeEngine:
    """One-joint fake engine: enough surface for ``SimEnv`` to wrap."""

    def __init__(self) -> None:
        self._j = 0.0
        self._v = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        # These fakes are duck-typed rather than ``SimEngine`` subclasses, so
        # they do not inherit the default that mirrors the joint names. This
        # robot's one joint is its one actuator, so the two vocabularies agree -
        # which is the shape ``SimEnv`` sizes its action head from.
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        self._v = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._v}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._v = 0.1 * a
        self._j += self._v
        return {"status": "success"}


def _make_env():  # type: ignore[no-untyped-def]
    return SimEnv(
        _FakeEngine(),
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: -abs(float(e.get_observation(skip_images=True)["J"]) - 0.2)],
        action_dim=1,
        max_episode_steps=8,
    )


def _spec(**overrides: Any) -> RLTrainSpec:
    """A small, fast, otherwise-valid TD3 spec over the fake env."""
    base: dict[str, Any] = {
        "env_factory": _make_env,
        "output_dir": "/tmp/fast_td3_tests",
        "total_timesteps": 64,
        "rollout_steps": 8,
        "learning_starts": 16,
        "batch_size": 16,
        "gradient_steps": 2,
        "hidden_dims": (16,),
        "seed": 0,
    }
    base.update(overrides)
    return RLTrainSpec(**base)


def test_fast_td3_registered_and_created() -> None:
    assert "fast_td3" in list_trainers()
    trainer = create_trainer("fast_td3")
    assert trainer.provider_name == "fast_td3"


def test_validate_rejects_bad_specs() -> None:
    trainer = create_trainer("fast_td3")

    # A plain (non-RL) spec is rejected with a clear message.
    problems = trainer.validate(TrainSpec(output_dir="/tmp/x"))
    assert any("RLTrainSpec" in p for p in problems)

    # Missing env_factory.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x"))
    assert any("env_factory" in p for p in problems)

    # learning_starts must be >= batch_size so the first update has a full batch.
    problems = trainer.validate(_spec(batch_size=256, learning_starts=10))
    assert any("learning_starts" in p for p in problems)

    # tau out of range.
    problems = trainer.validate(_spec(tau=2.0))
    assert any("tau" in p for p in problems)


def test_validate_accepts_a_vectorized_env_count() -> None:
    """FastTD3 collects through VecSimEnv, so any positive num_envs is usable.

    This is the deliberate contrast with the single-env FastSAC, whose
    ``validate`` refuses ``num_envs != 1``.
    """
    trainer = create_trainer("fast_td3")
    assert trainer.validate(_spec(num_envs=4)) == []
    assert not any("single-env" in p for p in trainer.validate(_spec(num_envs=4)))
    # A non-count is still refused by the shared domain.
    assert any("num_envs" in p for p in trainer.validate(_spec(num_envs=0)))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"output_dir": ""}, "output_dir is required"),
        ({"total_timesteps": 0}, "total_timesteps must be a positive integer"),
        ({"rollout_steps": 0}, "rollout_steps must be a positive integer"),
        ({"buffer_size": 0}, "buffer_size must be a positive integer"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"gradient_steps": 0}, "gradient_steps must be a positive integer"),
        ({"policy_delay": 0}, "policy_delay must be a positive integer"),
        ({"exploration_noise_std": 0.0}, "exploration_noise_std must be > 0"),
        ({"target_noise_std": 0.0}, "target_noise_std must be > 0"),
        ({"target_noise_clip": 0.0}, "target_noise_clip must be > 0"),
    ],
)
def test_validate_flags_each_out_of_range_field(overrides: dict[str, Any], expected: str) -> None:
    """Each numeric guard in the FastTD3 preflight names the offending field.

    ``validate`` is a pure, read-only preflight the ``train`` entry point runs
    before touching torch or the env, so a misconfigured spec fails with an
    actionable message instead of a deep stack trace. One otherwise-valid spec
    per case isolates a single bad field.
    """
    trainer = create_trainer("fast_td3")
    problems = trainer.validate(_spec(**overrides))
    assert any(expected in p for p in problems), problems


def test_train_rejects_non_rl_spec() -> None:
    trainer = create_trainer("fast_td3")
    result = trainer.train(TrainSpec(output_dir="/tmp/x"))
    assert result.status == "error"
    assert "RLTrainSpec" in result.message


def test_td3_actor_action_bounded_and_finite_under_saturation() -> None:
    """The tanh-bounded deterministic action stays inside [-1, 1] at the extremes."""
    spec = RLTrainSpec(hidden_dims=(16,))
    ac = _build_actor_critic(num_actor_obs=3, num_critic_obs=3, num_actions=2, spec=spec)
    obs = torch.full((8, 3), 50.0)  # drive the pre-tanh output far out
    action = ac.act_inference(obs)
    assert action.shape == (8, 2)
    assert torch.isfinite(action).all()
    assert (action.abs() <= 1.0).all()
    # The target policy is a distinct network with the same bound.
    target_action = ac.act_target(obs)
    assert (target_action.abs() <= 1.0).all()


def test_targets_start_as_copies_and_do_not_require_grad() -> None:
    """Every target network initializes as a copy of its live network, frozen."""
    spec = RLTrainSpec(hidden_dims=(8,))
    ac = _build_actor_critic(num_actor_obs=2, num_critic_obs=2, num_actions=1, spec=spec)
    for live, target in ((ac.actor, ac.actor_target), (ac.q1, ac.q1_target), (ac.q2, ac.q2_target)):
        for p, tp in zip(live.parameters(), target.parameters()):
            assert torch.equal(p, tp)
            assert tp.requires_grad is False


def test_fast_td3_smoke_train_produces_loadable_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = create_trainer("fast_td3")
    spec = _spec(output_dir=str(tmp_path))
    assert trainer.validate(spec) == []

    result = trainer.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None

    policy_pt = os.path.join(result.checkpoint_dir, "policy.pt")
    assert os.path.isfile(policy_pt)
    assert result.exported_model == policy_pt

    state = torch.load(policy_pt, weights_only=True)
    assert "actor_critic" in state and "actor_norm" in state
    assert "log_alpha" not in state  # TD3 holds no entropy temperature

    with open(os.path.join(result.checkpoint_dir, "policy_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["provider"] == "fast_td3"
    assert meta["num_actions"] == 1
    assert meta["actor_obs_keys"] == ["J", "J.vel"]
    # The field names what the ``num_actions`` outputs drive, so its width is
    # part of the contract - a length that disagreed would mis-bind a
    # deployment's outputs onto the robot.
    assert meta["action_keys"]  # non-empty
    assert len(meta["action_keys"]) == meta["num_actions"]

    assert trainer.latest_checkpoint(str(tmp_path)) == result.checkpoint_dir


def test_checkpoint_round_trip_restores_the_actor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A fresh trainer loads a saved policy.pt bit-identically via the shared loader."""
    first = FastTd3Trainer()
    spec = _spec(output_dir=str(tmp_path))
    result = first.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None

    second = FastTd3Trainer()
    second.setup(_spec(output_dir=str(tmp_path)))
    second.load_checkpoint(result.checkpoint_dir)
    for p1, p2 in zip(first.actor_critic.parameters(), second.actor_critic.parameters()):
        assert torch.equal(p1, p2)

    # The eval entry point reaches the same loader from a spec alone.
    ev = FastTd3Trainer().evaluate(spec=_spec(output_dir=str(tmp_path)), num_episodes=2)
    assert ev["num_episodes"] == 2
    assert len(ev["returns"]) == 2


def test_update_is_noop_until_buffer_reaches_batch_size() -> None:
    """``update`` returns zero-loss metrics while the buffer is below a batch."""
    trainer = FastTd3Trainer()
    trainer.setup(_spec())
    assert trainer.buffer.size < trainer.spec.batch_size  # nothing collected yet

    metrics = trainer.update()
    assert metrics["critic_loss"] == 0.0
    assert metrics["actor_loss"] == 0.0
    assert metrics["latest_loss"] == 0.0


def test_actor_updates_follow_the_policy_delay() -> None:
    """The actor moves only every ``policy_delay``-th gradient step.

    6 gradient steps at the shipped delay of 2 take exactly 3 actor updates,
    and the cadence spans ``update()`` calls (the counter is persistent) rather
    than restarting inside each call.
    """
    trainer = FastTd3Trainer()
    trainer.setup(_spec(gradient_steps=3, policy_delay=2, learning_starts=16, batch_size=16))
    trainer.collect_rollout()
    trainer.collect_rollout()  # 16 transitions -> warmup satisfied

    first = trainer.update()  # updates 1..3 -> actor fires at 2
    second = trainer.update()  # updates 4..6 -> actor fires at 4 and 6
    assert int(first["actor_updates"]) == 1
    assert int(second["actor_updates"]) == 2
    assert trainer._update_count == 6


def test_delayed_step_moves_actor_and_targets_together() -> None:
    """A delayed step Polyak-moves the actor target and both critic targets."""
    trainer = FastTd3Trainer()
    trainer.setup(_spec(gradient_steps=2, policy_delay=2, tau=0.5))
    trainer.collect_rollout()
    trainer.collect_rollout()

    before = [p.clone() for p in trainer.actor_critic.actor_target.parameters()]
    metrics = trainer.update()  # 2 gradient steps -> one delayed actor step
    assert int(metrics["actor_updates"]) == 1
    after = list(trainer.actor_critic.actor_target.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "actor target never moved"


class _FakeTermEnv:
    """Minimal ``SimEnv``-shaped fake whose ``step`` emits a scripted terminated/done.

    Mirrors the FastSAC truncation-contract fake: ``step`` always reports
    ``done=1`` with ``info["terminated"]`` scripted, so a test can drive a
    time-out (done=1, terminated=0) or a genuine terminal (done=1,
    terminated=1). Observations count the steps taken since the last reset, so
    a pre-reset terminal observation (1.0) is distinguishable from a fresh
    post-reset one (0.0).
    """

    def __init__(self, terminated_flag: bool, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.num_actor_obs = 2
        self.num_critic_obs = 2
        self.num_actions = 1
        self._terminated = terminated_flag
        self._steps = 0
        self.resets = 0
        self.closed = False

    def _obs(self) -> dict:
        val = float(self._steps)
        return {
            "actor_obs": torch.full((1, self.num_actor_obs), val, device=self.device),
            "critic_obs": torch.full((1, self.num_critic_obs), val, device=self.device),
        }

    def reset(self) -> dict:
        self.resets += 1
        self._steps = 0
        return self._obs()

    def step(self, action):  # type: ignore[no-untyped-def]
        self._steps += 1
        done = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        info = {"time_out": (not self._terminated), "terminated": self._terminated}
        return self._obs(), reward, done, info

    def close(self) -> None:
        self.closed = True


def _td3_trainer_on_fake(terminated_flag: bool, num_envs: int = 1):  # type: ignore[no-untyped-def]
    """A FastTD3 trainer set up on scripted fake envs, warmup branch only."""
    from typing import cast

    trainer = FastTd3Trainer()
    spec = RLTrainSpec(
        env_factory=lambda: cast("SimEnv", _FakeTermEnv(terminated_flag)),
        output_dir="/tmp/td3_truncation_contract",
        device="cpu",
        rollout_steps=1,
        num_envs=num_envs,
        batch_size=16,
        # Keep the buffer below learning_starts so collect_rollout takes the
        # warmup branch and never needs a live actor forward pass.
        learning_starts=64,
    )
    trainer.setup(spec)
    return trainer


@pytest.mark.parametrize("num_envs", [1, 3])
def test_collect_rollout_stores_terminal_not_done_on_timeout(num_envs: int) -> None:
    """A time-out is bootstrapped: the stored done flag is 0, on both paths."""
    trainer = _td3_trainer_on_fake(terminated_flag=False, num_envs=num_envs)
    trainer.collect_rollout()
    assert trainer.buffer.size == num_envs
    assert trainer.buffer._dones[:num_envs].sum().item() == 0.0


@pytest.mark.parametrize("num_envs", [1, 3])
def test_collect_rollout_stores_terminal_on_genuine_terminal(num_envs: int) -> None:
    """A genuine terminal zeroes the bootstrap: the stored done flag is 1."""
    trainer = _td3_trainer_on_fake(terminated_flag=True, num_envs=num_envs)
    trainer.collect_rollout()
    assert trainer.buffer.size == num_envs
    assert trainer.buffer._dones[:num_envs].sum().item() == float(num_envs)


def test_vectorized_collect_stores_the_pre_reset_terminal_obs() -> None:
    """The stored next-obs of a done env is the captured terminal, not the reset.

    ``VecSimEnv`` auto-resets a done sub-env and returns the FRESH observation
    (0.0 on this fake); the TRUE terminal observation (1.0) survives only in
    ``infos[i]["terminal_obs"]``. Storing the fresh one would bootstrap the
    episode's last TD target across the reset boundary.
    """
    trainer = _td3_trainer_on_fake(terminated_flag=False, num_envs=3)
    trainer.collect_rollout()
    stored = trainer.buffer._next_actor_obs[:3]
    assert torch.equal(stored, torch.ones_like(stored)), stored


def test_vectorized_collect_pushes_n_transitions_per_tick() -> None:
    """N envs over T steps put exactly N*T transitions in the buffer."""
    trainer = FastTd3Trainer()
    T, N = 8, 4
    trainer.setup(_spec(num_envs=N, rollout_steps=T, learning_starts=64, batch_size=16))
    assert isinstance(trainer.env, VecSimEnv)
    assert trainer._vectorized is True
    metrics = trainer.collect_rollout()
    assert trainer.buffer.size == T * N
    assert metrics["buffer_size"] == float(T * N)
    assert "mean_reward" in metrics and "mean_episode_return" in metrics
    trainer.env.close()


def test_num_envs_1_uses_single_path() -> None:
    trainer = FastTd3Trainer()
    trainer.setup(_spec(num_envs=1))
    assert trainer._vectorized is False
    assert not isinstance(trainer.env, VecSimEnv)
    trainer.collect_rollout()
    assert trainer.buffer.size == trainer.spec.rollout_steps


def test_vectorized_smoke_train_and_evaluate() -> None:
    """Full vectorized loop: train succeeds, closes the pool, and eval still runs."""
    trainer = FastTd3Trainer()
    spec = _spec(
        output_dir="/tmp/td3_vec_smoke",
        num_envs=4,
        total_timesteps=8 * 4 * 3,  # T*N*iters
        learning_starts=32,
        batch_size=16,
        gradient_steps=1,
    )
    assert trainer.validate(spec) == []
    result = trainer.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "policy.pt"))
    # The finally in train() shut the VecSimEnv's thread pool down...
    assert isinstance(trainer.env, VecSimEnv)
    assert trainer.env._executor is None
    # ...and evaluate still works on the same instance (serial fallback).
    ev = trainer.evaluate(num_episodes=2)
    assert ev["num_episodes"] == 2
    assert isinstance(ev["mean_return"], float)


def test_setup_reconciles_env_device_to_learner_device() -> None:
    """The learner device is authoritative over the env device (GPU-host guard).

    Mirrors the PPO / FastSAC regression: on a GPU host the learner resolves to
    ``cuda`` while ``SimEnv`` keeps its default ``cpu`` device, so observation
    tensors would mix devices. ``setup`` must reconcile the env onto the
    learner device. Reproduced on CPU with the storage-free ``meta`` device.
    """

    def factory():  # type: ignore[no-untyped-def]
        env = _make_env()
        env.device = torch.device("meta")
        return env

    trainer = FastTd3Trainer()
    trainer.setup(_spec(env_factory=factory, device="cpu"))

    assert trainer.env.device == trainer.device
    assert trainer._obs["actor_obs"].device == trainer.device
    assert trainer.buffer.device == trainer.device
