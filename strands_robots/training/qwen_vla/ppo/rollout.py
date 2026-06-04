"""Client-server rollout buffer + GAE for Qwen-VLA Stage-4 RL (section 4.2).

The RL env is the existing simulation: ``PolicyRunner.evaluate`` +
``BenchmarkProtocol`` yield seeded, reproducible, success-scored rollouts. The
reward is sparse binary success ``R in {0, 1}`` assigned at the action-chunk
level (one scalar reward + advantage per H=16 chunk).

The GAE computation (Schulman et al. 2016) is pure NumPy so the advantage /
return math unit-tests deterministically, independent of any policy or sim.
"""

from dataclasses import dataclass, field

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation advantages + returns.

    Operates at the action-chunk level: each element of *rewards* is the scalar
    reward for one H-step action chunk (section 4.2 credit assignment).

    Args:
        rewards: ``(T,)`` per-chunk rewards.
        values: ``(T,)`` value-head estimates for each chunk's state.
        dones: ``(T,)`` boolean/float episode-termination flags (1 = terminal).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        last_value: Bootstrap value for the step after the last (0 if terminal).

    Returns:
        ``(advantages, returns)`` each ``(T,)``.

    Raises:
        ValueError: If inputs are not 1-D or lengths mismatch.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    if not (r.ndim == v.ndim == d.ndim == 1):
        raise ValueError("rewards, values, dones must all be 1-D")
    if not (len(r) == len(v) == len(d)):
        raise ValueError(f"length mismatch: rewards={len(r)}, values={len(v)}, dones={len(d)}")

    t = len(r)
    advantages = np.zeros(t, dtype=np.float64)
    gae = 0.0
    next_value = last_value
    for i in reversed(range(t)):
        non_terminal = 1.0 - d[i]
        delta = r[i] + gamma * next_value * non_terminal - v[i]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[i] = gae
        next_value = v[i]
    returns = advantages + v
    return advantages, returns


@dataclass
class RolloutBuffer:
    """Accumulates per-chunk transitions for a PPO update.

    One entry per H-step action chunk: the chunk's log-prob under the behaviour
    policy, the value estimate, the sparse success reward, and the done flag.
    """

    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    def add(self, *, log_prob: float, value: float, reward: float, done: bool) -> None:
        """Append one chunk-level transition."""
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(1.0 if done else 0.0)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_advantages(
        self, *, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages + returns over the buffered transitions."""
        if not self.rewards:
            raise ValueError("cannot compute advantages on an empty buffer")
        return compute_gae(
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones),
            gamma=gamma,
            gae_lambda=gae_lambda,
            last_value=last_value,
        )

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


__all__ = ["compute_gae", "RolloutBuffer"]
