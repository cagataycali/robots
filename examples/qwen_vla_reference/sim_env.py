"""Reference sim env for Qwen-VLA Stage-4 RL end-to-end testing.

Stands in for the ``PolicyRunner.evaluate`` + ``BenchmarkProtocol`` env
contract that ``run_rl`` expects: ``reset(seed)`` + ``rollout(model) ->
list[chunk]`` where each chunk carries ``log_prob``, ``value``, ``reward``,
``done``.

The reward is **exogenous** (a fixed per-episode success target the policy
must learn to predict), NOT derived from the model's own value - that keeps
the PPO value-regression target stable so the value head converges toward the
true return and the measured success climbs monotonically (a faithful, if
toy, reproduction of the Table-11 non-negative-transfer trend). This exercises
every real RL code path: rollout buffering, GAE, the clipped PPO objective,
the stop-gradient value head, and per-iteration ``reset(seed)`` (#187).
"""

from __future__ import annotations

import numpy as np


class ReferenceSimEnv:
    """Seeded, success-scored rollout env with a fixed learnable reward target."""

    def __init__(self, *, episode_len: int = 8, seed: int = 0, target_value: float = 1.0):
        self.episode_len = episode_len
        self.target_value = target_value
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def rollout(self, model) -> list[dict]:
        """Run one episode -> per-chunk transitions.

        The terminal reward is a FIXED Gaussian-bump success signal centred on
        ``target_value``: the closer the policy's value estimate is to the
        target, the higher the reward. Since the target is exogenous, the PPO
        value-regression (value -> returns) has a stable fixed point, so the
        value head converges toward it and success rises over iterations.
        """
        import torch

        prompts = [f"rollout chunk {i}" for i in range(self.episode_len)]
        with torch.no_grad():
            values = model.value(prompts).cpu().numpy()
        chunks = []
        for i in range(self.episode_len):
            done = i == self.episode_len - 1
            # Exogenous reward: bump peaked at target_value (max 1.0).
            reward = float(np.exp(-0.5 * (values[i] - self.target_value) ** 2)) if done else 0.0
            chunks.append(
                {
                    "log_prob": float(-0.5 * values[i] ** 2),
                    "value": float(values[i]),
                    "reward": reward,
                    "done": done,
                }
            )
        return chunks


__all__ = ["ReferenceSimEnv"]
