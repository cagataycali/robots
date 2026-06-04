"""Flow-matching log-probability for PPO (section 4.2 / Song et al. 2021).

PPO needs a tractable ``log pi(a | s)`` for the importance ratio, but a
flow-matching policy defines its density implicitly via a probability-flow ODE.
Following Song et al. (2021), the ODE is converted to an SDE and the log-prob
is estimated with a single random denoising step per rollout (the paper's
efficiency choice, section 4.2).

The Gaussian-surrogate log-prob math (the per-chunk score the PPO ratio uses)
is pure NumPy and unit-tested; the model-coupled denoising step is provided by
the training model and injected.
"""

import numpy as np


def gaussian_logprob(action: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> float:
    """Diagonal-Gaussian log-density of *action* under ``N(mean, exp(log_std)^2)``.

    Used as the single-step SDE surrogate for the flow-matching log-prob: the
    denoising step yields a Gaussian transition kernel whose log-density is the
    per-chunk score entering the PPO importance ratio.

    Args:
        action: ``(...,)`` sampled action (flattened chunk).
        mean: Same-shape predicted mean.
        log_std: Same-shape log standard deviation.

    Returns:
        Scalar summed log-probability over all action dims.

    Raises:
        ValueError: If shapes mismatch.
    """
    a = np.asarray(action, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    ls = np.asarray(log_std, dtype=np.float64)
    if not (a.shape == m.shape == ls.shape):
        raise ValueError(f"shape mismatch: action {a.shape}, mean {m.shape}, log_std {ls.shape}")
    var = np.exp(2.0 * ls)
    log_2pi = np.log(2.0 * np.pi)
    per_dim = -0.5 * (log_2pi + 2.0 * ls + (a - m) ** 2 / var)
    return float(per_dim.sum())


def ppo_ratio(new_logprob: float, old_logprob: float) -> float:
    """Importance ratio ``exp(new_logprob - old_logprob)`` for the PPO objective."""
    return float(np.exp(new_logprob - old_logprob))


__all__ = ["gaussian_logprob", "ppo_ratio"]
