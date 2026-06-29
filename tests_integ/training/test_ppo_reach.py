"""End-to-end PPO convergence proof: train an SO-100 reach policy in MuJoCo.

NOT run in CI (no convergence assertions belong in the fast suite). Run with::

    MUJOCO_GL=egl pytest tests_integ/training/test_ppo_reach.py -v

Asserts that PPO, trained from a reward function alone, both improves its mean
return and produces a deterministic policy that drives the joint to the target.
Requires ``torch`` + ``[sim-mujoco]``; ~1-2 min on CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

TARGET = 0.2


def _elbow_reward(engine):  # type: ignore[no-untyped-def]
    return -abs(float(engine.get_observation(skip_images=True)["Elbow"]) - TARGET)


def _make_env():  # type: ignore[no-untyped-def]
    import strands_robots as sr
    from strands_robots.training.rl import SimEnv

    engine = sr.Robot("so100", mode="sim")
    return SimEnv(
        engine,
        actor_obs_keys=["Elbow", "Elbow.vel"],
        reward_terms=[_elbow_reward],
        action_dim=6,
        max_episode_steps=50,
    )


def test_ppo_learns_to_reach_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from strands_robots.training.rl import PpoTrainer, RLTrainSpec

    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        total_timesteps=250 * 150,
        rollout_steps=250,
        num_mini_batches=4,
        num_learning_epochs=5,
        learning_rate=1e-3,
        init_noise_std=0.8,
        seed=0,
    )
    trainer.setup(spec)

    returns = []
    for _ in range(150):
        metrics = trainer.collect_rollout()
        trainer.update()
        returns.append(metrics["mean_episode_return"])

    # 1. Mean return improved over training (reward gap closed).
    assert sum(returns[-10:]) / 10 > sum(returns[:10]) / 10

    # 2. The deterministic (mean) policy drives the joint to the target.
    trainer.actor_critic.eval()
    trainer.env.reset()
    final_elbow = 0.0
    for _ in range(50):
        actor_obs = trainer._norm_actor(trainer.env._obs_dict()["actor_obs"], update=False)
        with torch.no_grad():
            action = trainer.actor_critic.act_inference(actor_obs)
        trainer.env.step(action)
        final_elbow = float(trainer.env.engine.get_observation(skip_images=True)["Elbow"])

    assert abs(final_elbow - TARGET) < 0.06, f"reached {final_elbow}, target {TARGET}"
