"""Stage 4: Reinforcement Learning (PPO + GAE) on sim success (section 4.2).

Optimizes closed-loop task success using the existing simulation as the env
(``PolicyRunner.evaluate`` + ``BenchmarkProtocol``): N parallel envs, sparse
binary success reward, action-chunk-level credit assignment. Flow-matching
log-prob via the single-step SDE surrogate (Song et al. 2021), a stop-gradient
value head at ~20x the actor LR. Produces **Qwen-VLA-Instruct**.

The PPO numerics (GAE, ratio, clipping) are pure NumPy and unit-tested; the
rollout collection and gradient step are torch + sim coupled and injected via
the model / env objects passed to :func:`run_rl`.
"""

import logging
from typing import Any

import numpy as np

from strands_robots.training.qwen_vla.config import RLConfig
from strands_robots.training.qwen_vla.ppo.rollout import RolloutBuffer

logger = logging.getLogger(__name__)


def clipped_ppo_objective(
    advantages: np.ndarray,
    new_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    *,
    clip_epsilon: float = 0.2,
) -> float:
    """Compute the clipped PPO surrogate objective (to maximize), pure NumPy.

    ``L = mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))`` over chunks.

    Args:
        advantages: ``(T,)`` GAE advantages.
        new_logprobs: ``(T,)`` log-probs under the current policy.
        old_logprobs: ``(T,)`` log-probs under the behaviour policy.
        clip_epsilon: PPO clip range.

    Returns:
        Scalar surrogate objective (higher is better).

    Raises:
        ValueError: If lengths mismatch.
    """
    a = np.asarray(advantages, dtype=np.float64)
    nlp = np.asarray(new_logprobs, dtype=np.float64)
    olp = np.asarray(old_logprobs, dtype=np.float64)
    if not (len(a) == len(nlp) == len(olp)):
        raise ValueError("advantages, new_logprobs, old_logprobs must match length")
    ratio = np.exp(nlp - olp)
    unclipped = ratio * a
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * a
    return float(np.minimum(unclipped, clipped).mean())


def run_rl(config: RLConfig, *, model: Any = None, env: Any = None) -> dict[str, Any]:
    """Run the Stage-4 PPO training loop.

    Args:
        config: Stage-4 config. ``sft_checkpoint`` loads the SFT policy.
        model: A loaded Qwen-VLA training model with a value head. Required.
        env: A sim env exposing seeded rollouts + sparse success reward
            (``PolicyRunner.evaluate`` / ``BenchmarkProtocol`` adapter). Required.

    Returns:
        Summary dict for the RL run.

    Raises:
        ImportError: If the ``qwen-vla-train`` extra (torch) is missing.
        ValueError: If *model* or *env* is None or the config is invalid.
    """
    config.validate()
    if model is None:
        raise ValueError("run_rl requires a loaded model with a value head")
    if env is None:
        raise ValueError("run_rl requires a sim env with seeded success-scored rollouts")

    from strands_robots.utils import require_optional

    require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA RL training")

    if config.sft_checkpoint:
        model.load_checkpoint(config.sft_checkpoint)
    logger.info(
        "RL start: envs=%d, iterations=%d, gamma=%.2f, lambda=%.2f, clip=%.2f, ppo_epochs=%d",
        config.num_envs,
        config.num_iterations,
        config.gamma,
        config.gae_lambda,
        config.clip_epsilon,
        config.ppo_epochs,
    )

    buffer = RolloutBuffer()
    last_objective = float("nan")
    for iteration in range(config.num_iterations):
        buffer.clear()
        # Reproducible per-iteration seeding (the #187 reset contract).
        model.reset(seed=config.seed + iteration)
        env.reset(seed=config.seed + iteration)
        for _ in range(config.num_envs):
            traj = env.rollout(model)  # one episode -> chunks + sparse success
            for chunk in traj:
                buffer.add(log_prob=chunk["log_prob"], value=chunk["value"], reward=chunk["reward"], done=chunk["done"])
        advantages, returns = buffer.compute_advantages(gamma=config.gamma, gae_lambda=config.gae_lambda)
        for _ in range(config.ppo_epochs):
            new_logprobs = model.recompute_logprobs(buffer)
            obj = clipped_ppo_objective(
                advantages, new_logprobs, np.array(buffer.log_probs), clip_epsilon=config.clip_epsilon
            )
            model.ppo_step(objective=obj, returns=returns)
            last_objective = obj

    return {
        "stage": "rl",
        "iterations": config.num_iterations,
        "final_objective": last_objective,
        "output_dir": config.output_dir,
    }


__all__ = ["clipped_ppo_objective", "run_rl"]
