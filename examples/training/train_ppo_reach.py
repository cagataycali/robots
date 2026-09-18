"""Train an SO-100 arm to reach a target joint angle with PPO, from scratch.

A minimal, dependency-light demonstration of the from-scratch RL trainer
(:mod:`strands_robots.training.rl`): no demonstration dataset, just a reward
function. PPO learns a Gaussian-MLP policy that drives the SO-100 ``Elbow`` joint
to a target angle in MuJoCo, then the deterministic (mean) policy is rolled out
and the joint trajectory is reported.

Run (headless on a GPU box, or plain CPU - PPO here trains fine on CPU)::

    MUJOCO_GL=egl python examples/training/train_ppo_reach.py

Requires the ``[sim-mujoco]`` extra plus ``torch``. The same recipe generalizes
to locomotion / whole-body control by swapping the robot and the reward terms
(see ``docs/training/rl.md``).
"""

from __future__ import annotations

import tempfile

import strands_robots as sr
from strands_robots.simulation.observers import RunPolicyEvent, RunPolicyStep
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv

TARGET_ELBOW = 0.2  # radians, within the SO-100 Elbow reachable range


def elbow_reach_reward(engine: object) -> float:
    """Dense reach reward: negative distance of the Elbow joint from the target.

    A reward term is any ``Callable[[SimEngine], float]``; this one reads the
    current ``Elbow`` joint angle and rewards closing the gap to ``TARGET_ELBOW``.
    """
    elbow = engine.get_observation(skip_images=True)["Elbow"]  # type: ignore[attr-defined]
    return -abs(float(elbow) - TARGET_ELBOW)


def make_env() -> SimEnv:
    """Build the reach environment: observe the Elbow, reward closing on TARGET."""
    engine = sr.Robot("so100", mode="sim")
    return SimEnv(
        engine,
        actor_obs_keys=["Elbow", "Elbow.vel"],
        reward_terms=[elbow_reach_reward],
        action_dim=6,
        max_episode_steps=50,
    )


def rollout_and_report(checkpoint_dir: str) -> None:
    """Roll the deterministic actor out on the trained robot, printing the trace.

    ``run_policy(policy_provider="rl")`` presents the checkpoint's actor as an
    ordinary policy, so it drives the robot ``make_env`` trained against. The
    trace is what the reward bought: ``mean_reward`` does not say where the
    joint ended up.
    """
    trace: list[float] = []

    def record(event: RunPolicyEvent) -> None:
        if isinstance(event, RunPolicyStep):
            trace.append(float(event.observation["Elbow"]))

    outcome = sr.Robot("so100", mode="sim").run_policy(
        robot_name="so100",
        policy_provider="rl",
        policy_config={"checkpoint_dir": checkpoint_dir},
        duration=4.0,
        observer=record,
    )
    # run_policy does not raise: every failure, a policy that will not construct
    # included, comes back as a status envelope, and an empty trace is what a
    # failed rollout leaves behind. Read the verdict before reading the trace.
    if outcome["status"] != "success":
        raise SystemExit(f"rollout failed: {outcome['content'][0]['text']}")
    if not trace:
        raise SystemExit("rollout produced no steps, so there is no trace to report")
    print(f"Elbow trace (target {TARGET_ELBOW}): {[round(v, 3) for v in trace[::20]]}")
    print(f"Elbow final={trace[-1]:.4f} rad |error|={abs(trace[-1] - TARGET_ELBOW):.4f} rad")


def main() -> None:
    trainer = create_trainer("ppo")
    output_dir = tempfile.mkdtemp(prefix="ppo_reach_")
    spec = RLTrainSpec(
        env_factory=make_env,
        output_dir=output_dir,
        total_timesteps=250 * 150,
        rollout_steps=250,
        num_mini_batches=5,
        num_learning_epochs=5,
        learning_rate=1e-3,
        init_noise_std=0.8,
        seed=0,
    )

    problems = trainer.validate(spec)
    if problems:
        raise SystemExit(f"spec invalid: {problems}")

    result = trainer.train(spec)
    print(f"status={result.status}")
    print(f"checkpoint={result.checkpoint_dir}")
    print(f"exported policy={result.exported_model}")
    print(f"final metrics={ {k: round(v, 4) if isinstance(v, float) else v for k, v in result.metrics.items()} }")
    if result.status != "success" or result.checkpoint_dir is None:
        raise SystemExit(f"training ended with status={result.status!r}; there is no checkpoint to roll out")
    rollout_and_report(result.checkpoint_dir)


if __name__ == "__main__":
    main()
