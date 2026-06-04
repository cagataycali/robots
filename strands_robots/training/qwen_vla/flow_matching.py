"""Conditional flow-matching objective + timestep distributions (section 3 / 5.2.1).

Qwen-VLA's action expert is trained with conditional flow matching (eqs. 1-2):
the DiT regresses the velocity field ``v_theta(x_t, t, cond)`` toward the target
``(x_1 - x_0)`` along a straight-line interpolation ``x_t = (1-t) x_0 + t x_1``.
The per-channel loss is masked by the unified zero-padding mask (section 2.4)
and averaged with the paper's two-level scheme (per-sample over valid cells,
then over the batch).

Two timestep-sampling distributions matter for the ablation (section 5.2.1):
**Sigmoid-Normal** (best for T2A) and **Beta** (best for CPT), plus uniform.

The numeric core (masked loss, interpolation, timestep sampling) is pure NumPy
so it unit-tests without torch. A thin ``torch_flow_matching_loss`` wrapper is
provided for the actual training loop and is import-guarded behind the
``qwen-vla-train`` extra.
"""

import numpy as np

from strands_robots.training.qwen_vla.config import TimestepDist


def sample_timesteps(
    n: int,
    dist: TimestepDist,
    *,
    rng: np.random.Generator | None = None,
    beta_a: float = 1.5,
    beta_b: float = 1.0,
    sigmoid_loc: float = 0.0,
    sigmoid_scale: float = 1.0,
) -> np.ndarray:
    """Sample *n* flow-matching timesteps in ``[0, 1]`` from *dist*.

    Args:
        n: Number of timesteps to draw.
        dist: One of :class:`TimestepDist`.
        rng: Optional NumPy generator (for reproducibility).
        beta_a, beta_b: Beta distribution shape params (CPT default skews toward
            larger t, emphasizing the harder denoising region).
        sigmoid_loc, sigmoid_scale: Normal params before the sigmoid squashing
            (T2A's Sigmoid-Normal: ``t = sigmoid(loc + scale * z)``, ``z ~ N(0,1)``).

    Returns:
        ``(n,)`` float64 array of timesteps in ``[0, 1]``.

    Raises:
        ValueError: If *n* is negative or *dist* is unknown.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    rng = rng or np.random.default_rng()

    if dist == TimestepDist.UNIFORM:
        return rng.uniform(0.0, 1.0, size=n)
    if dist == TimestepDist.BETA:
        return rng.beta(beta_a, beta_b, size=n)
    if dist == TimestepDist.SIGMOID_NORMAL:
        z = rng.normal(sigmoid_loc, sigmoid_scale, size=n)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown timestep dist {dist!r}")


def interpolate(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Straight-line flow interpolation ``x_t = (1 - t) x0 + t x1`` (eq. 1).

    Args:
        x0: Noise sample, shape ``(B, H, K)``.
        x1: Target action, shape ``(B, H, K)``.
        t: Per-sample timestep, shape ``(B,)``.

    Returns:
        ``x_t`` with shape ``(B, H, K)``.

    Raises:
        ValueError: If shapes are incompatible.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x0.shape != x1.shape:
        raise ValueError(f"x0 {x0.shape} and x1 {x1.shape} must match")
    if t.shape[0] != x0.shape[0]:
        raise ValueError(f"t batch {t.shape[0]} != x0 batch {x0.shape[0]}")
    t_b = t.reshape((-1,) + (1,) * (x0.ndim - 1))
    return (1.0 - t_b) * x0 + t_b * x1


def target_velocity(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Flow-matching target velocity ``v* = x1 - x0`` (eq. 2)."""
    return np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)


def masked_flow_matching_loss(
    pred_velocity: np.ndarray,
    target_velocity_arr: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Masked, two-level-averaged flow-matching loss (section 3, eq. 2).

    Computes the squared velocity error, zeroes the padded cells via *mask*,
    averages per-sample over the valid cell count, then averages over the
    batch. This matches the paper's "two-level per-channel averaging with
    mask" so a wide-but-mostly-padded embodiment does not dominate the loss.

    Args:
        pred_velocity: ``(B, H, K)`` predicted velocity field.
        target_velocity_arr: ``(B, H, K)`` target ``x1 - x0``.
        mask: ``(B, H, K)`` or ``(H, K)`` binary validity mask. A 2-D mask is
            broadcast across the batch.

    Returns:
        Scalar loss (float).

    Raises:
        ValueError: If shapes are incompatible or no cells are valid.
    """
    pred = np.asarray(pred_velocity, dtype=np.float64)
    tgt = np.asarray(target_velocity_arr, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if pred.shape != tgt.shape:
        raise ValueError(f"pred {pred.shape} and target {tgt.shape} must match")
    if m.ndim == 2:
        m = np.broadcast_to(m, pred.shape)
    if m.shape != pred.shape:
        raise ValueError(f"mask {m.shape} incompatible with pred {pred.shape}")

    sq_err = (pred - tgt) ** 2 * m
    # Per-sample valid-cell count (avoid div-by-zero for fully-padded samples).
    per_sample_valid = m.reshape(m.shape[0], -1).sum(axis=1)
    per_sample_sum = sq_err.reshape(sq_err.shape[0], -1).sum(axis=1)

    valid_samples = per_sample_valid > 0
    if not valid_samples.any():
        raise ValueError("mask selects no valid cells; loss is undefined")

    per_sample_loss = np.zeros_like(per_sample_sum)
    per_sample_loss[valid_samples] = per_sample_sum[valid_samples] / per_sample_valid[valid_samples]
    # Batch-level average over samples that had at least one valid cell.
    return float(per_sample_loss[valid_samples].mean())


def torch_flow_matching_loss(pred_velocity, target_velocity_arr, mask):
    """Torch equivalent of :func:`masked_flow_matching_loss` for the train loop.

    Import-guarded: requires the ``qwen-vla-train`` extra (torch). Keeps the
    same two-level masked averaging so training matches the validated NumPy
    reference exactly.
    """
    from strands_robots.utils import require_optional

    torch = require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA flow-matching training")

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).expand_as(pred_velocity)
    sq_err = (pred_velocity - target_velocity_arr) ** 2 * mask
    per_sample_valid = mask.reshape(mask.shape[0], -1).sum(dim=1)
    per_sample_sum = sq_err.reshape(sq_err.shape[0], -1).sum(dim=1)
    valid = per_sample_valid > 0
    if not bool(valid.any()):
        raise ValueError("mask selects no valid cells; loss is undefined")
    per_sample_loss = torch.zeros_like(per_sample_sum)
    per_sample_loss[valid] = per_sample_sum[valid] / per_sample_valid[valid]
    return per_sample_loss[valid].mean()


__all__ = [
    "sample_timesteps",
    "interpolate",
    "target_velocity",
    "masked_flow_matching_loss",
    "torch_flow_matching_loss",
]
