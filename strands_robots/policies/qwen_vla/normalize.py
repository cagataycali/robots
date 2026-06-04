"""Qwen-VLA per-dataset quantile normalization + channel masking.

Implements the two numerically-sensitive pieces of the Qwen-VLA I/O contract
that must be byte-identical between training and inference:

1. **Per-dataset quantile normalization** (section 5, eq. 5): map each action
   channel from its [1st, 99th] percentile range to ``[-1, 1]`` (and the
   inverse for un-normalizing model output back to robot units). Quantile
   (rather than min/max) clipping is robust to teleop outliers.

2. **Unified zero-padding channel mask** (section 2.4 / 5.2.2): heterogeneous
   embodiments share one fixed-width action tensor ``Y in R^{H x K}``; only the
   leading ``c`` channels are valid, the tail is zero-padded, and a per-channel
   binary mask ``M`` excludes the padding from the loss/gradient (training) and
   from the unpacked robot action (inference).

Pure NumPy - no torch, no model - so it unit-tests in isolation and is reused
verbatim by the training adapter.
"""

import numpy as np

# Paper default clipping quantiles (eq. 5): 1st and 99th percentile.
_DEFAULT_Q_LOW = 0.01
_DEFAULT_Q_HIGH = 0.99
# Guard against divide-by-zero when a channel is constant (q_high == q_low).
_EPS = 1e-6


def compute_quantile_stats(
    actions: np.ndarray,
    *,
    q_low: float = _DEFAULT_Q_LOW,
    q_high: float = _DEFAULT_Q_HIGH,
) -> dict[str, np.ndarray]:
    """Compute per-channel quantile stats from a dataset of actions.

    Args:
        actions: ``(N, c)`` array of raw actions (N samples, c channels).
        q_low: Lower quantile (default 0.01 = 1st percentile, per eq. 5).
        q_high: Upper quantile (default 0.99 = 99th percentile, per eq. 5).

    Returns:
        ``{"q_low": (c,), "q_high": (c,)}`` per-channel quantile arrays.

    Raises:
        ValueError: If *actions* is not 2-D or the quantiles are out of order.
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"actions must be 2-D (N, c), got shape {arr.shape}")
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError(f"require 0 <= q_low < q_high <= 1, got q_low={q_low}, q_high={q_high}")

    lo = np.quantile(arr, q_low, axis=0)
    hi = np.quantile(arr, q_high, axis=0)
    return {"q_low": lo.astype(np.float32), "q_high": hi.astype(np.float32)}


def normalize(actions: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize raw actions to ``[-1, 1]`` via quantile stats (eq. 5).

    ``x_norm = clip(2 * (x - q_low) / (q_high - q_low) - 1, -1, 1)``

    Args:
        actions: ``(..., c)`` raw actions.
        stats: ``{"q_low": (c,), "q_high": (c,)}`` from
            :func:`compute_quantile_stats`.

    Returns:
        Normalized actions in ``[-1, 1]`` with the same shape as *actions*.

    Raises:
        ValueError: If the trailing dim of *actions* does not match the stats.
    """
    arr = np.asarray(actions, dtype=np.float32)
    lo = np.asarray(stats["q_low"], dtype=np.float32)
    hi = np.asarray(stats["q_high"], dtype=np.float32)
    if arr.shape[-1] != lo.shape[-1]:
        raise ValueError(f"action channel dim {arr.shape[-1]} != stats dim {lo.shape[-1]}")
    span = np.maximum(hi - lo, _EPS)
    norm = 2.0 * (arr - lo) / span - 1.0
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def unnormalize(norm_actions: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Invert :func:`normalize`: map ``[-1, 1]`` back to raw action units.

    ``x = (x_norm + 1) / 2 * (q_high - q_low) + q_low``

    Note: because :func:`normalize` *clips* to ``[-1, 1]``, values that were
    originally outside the ``[q_low, q_high]`` band are not recoverable - the
    round-trip is exact only for inputs already within the quantile band
    (the common case for action chunks the model emits). Channels with a
    near-constant value (``q_high ~= q_low``) round-trip to ``q_low``.

    Args:
        norm_actions: ``(..., c)`` normalized actions in ``[-1, 1]``.
        stats: ``{"q_low": (c,), "q_high": (c,)}``.

    Returns:
        Raw-unit actions with the same shape as *norm_actions*.

    Raises:
        ValueError: If the trailing dim does not match the stats.
    """
    arr = np.asarray(norm_actions, dtype=np.float32)
    lo = np.asarray(stats["q_low"], dtype=np.float32)
    hi = np.asarray(stats["q_high"], dtype=np.float32)
    if arr.shape[-1] != lo.shape[-1]:
        raise ValueError(f"action channel dim {arr.shape[-1]} != stats dim {lo.shape[-1]}")
    span = hi - lo
    return ((arr + 1.0) / 2.0 * span + lo).astype(np.float32)


def build_channel_mask(c: int, k: int, h_task: int, h: int) -> np.ndarray:
    """Build the unified zero-padding channel mask ``M`` (section 2.4).

    Qwen-VLA emits a fixed-width ``Y in R^{H x K}`` tensor for every
    embodiment. For a given task only the leading ``c`` channels (action
    dims) and leading ``h_task`` timesteps are valid; everything else is
    zero-padded and excluded from loss/unpacking by this binary mask.

    Args:
        c: Number of valid action channels for this embodiment (``c <= K``).
        k: Fixed model channel dim ``K`` (widest embodiment).
        h_task: Number of valid timesteps for this task (``h_task <= H``;
            e.g. 8 for navigation waypoints, 16 for manipulation).
        h: Fixed model horizon ``H``.

    Returns:
        ``(H, K)`` float32 mask with 1.0 on valid ``[:h_task, :c]`` cells,
        0.0 elsewhere.

    Raises:
        ValueError: If any dimension is non-positive or a valid extent
            exceeds its fixed bound (``c > K`` or ``h_task > H``).
    """
    if c <= 0 or k <= 0 or h_task <= 0 or h <= 0:
        raise ValueError(f"all dims must be positive, got c={c}, k={k}, h_task={h_task}, h={h}")
    if c > k:
        raise ValueError(f"valid channels c={c} cannot exceed model channel dim K={k}")
    if h_task > h:
        raise ValueError(f"valid horizon h_task={h_task} cannot exceed model horizon H={h}")

    mask = np.zeros((h, k), dtype=np.float32)
    mask[:h_task, :c] = 1.0
    return mask


def pad_to_width(actions: np.ndarray, k: int) -> np.ndarray:
    """Zero-pad an ``(H, c)`` action chunk to the fixed ``(H, K)`` width.

    Args:
        actions: ``(H, c)`` valid actions.
        k: Fixed model channel dim ``K`` (``c <= K``).

    Returns:
        ``(H, K)`` float32 array with the tail channels zero-filled.

    Raises:
        ValueError: If *actions* is not 2-D or ``c > K``.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions must be 2-D (H, c), got shape {arr.shape}")
    h, c = arr.shape
    if c > k:
        raise ValueError(f"valid channels c={c} cannot exceed model channel dim K={k}")
    if c == k:
        return arr
    padded = np.zeros((h, k), dtype=np.float32)
    padded[:, :c] = arr
    return padded


def unpad_from_width(padded: np.ndarray, c: int) -> np.ndarray:
    """Slice an ``(H, K)`` model output down to the valid ``(H, c)`` channels.

    Args:
        padded: ``(H, K)`` model output.
        c: Number of valid channels to keep.

    Returns:
        ``(H, c)`` float32 array (the leading valid channels).

    Raises:
        ValueError: If *padded* is not 2-D or ``c`` exceeds its width.
    """
    arr = np.asarray(padded, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"padded must be 2-D (H, K), got shape {arr.shape}")
    if c > arr.shape[1]:
        raise ValueError(f"valid channels c={c} cannot exceed padded width {arr.shape[1]}")
    return arr[:, :c].astype(np.float32)


__all__ = [
    "compute_quantile_stats",
    "normalize",
    "unnormalize",
    "build_channel_mask",
    "pad_to_width",
    "unpad_from_width",
]
