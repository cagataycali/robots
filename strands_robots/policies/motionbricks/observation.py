"""Control-signal builder for the MotionBricks policy.

MotionBricks is driven not by camera frames but by a small *control signal*
dict, exactly as the upstream demo controllers
(``motionbricks/motion_backbone/demo/controllers.py``) construct it:

    {
        "movement_direction": [x, y, z],   # global mujoco frame, unit-ish
        "facing_direction":   [x, y, z],   # global mujoco frame, unit-ish
        "mode":               int,         # index into the clip set (CLIPS)
        "allowed_pred_num_tokens": [int],  # per-mode horizon mask
    }

This module reproduces that mapping as **pure python/NumPy** helpers (no torch,
no ``motionbricks`` import) so the style resolution and command assembly are
unit-testable on any machine. The policy converts the returned plain dict into
torch tensors and injects the motion context just before calling the generator.

The well-known goal kwargs a caller passes through ``get_actions`` are:

* ``style`` / ``mode`` - clip mode, an ``int`` index or a ``str`` name.
* ``target_velocity`` - ``[vx, vy]`` (or ``[vx, vy, vz]``) desired planar
  movement direction in the world frame; only the direction is used (the clip's
  ``avg_root_vel`` sets the speed). Absent -> walk straight ahead (``+x``).
* ``target_heading`` - facing direction as ``[hx, hy]`` (or an angle in radians
  via ``target_heading_angle``). Absent -> face the movement direction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Default movement / facing direction: walk straight ahead along +x (world).
_DEFAULT_DIRECTION = (1.0, 0.0, 0.0)


def resolve_mode(style: int | str, clip_keys: list[str]) -> int:
    """Resolve a style (mode index or name) to its integer index in ``clip_keys``.

    Args:
        style: A clip mode index (``int``) or name (``str``, e.g. ``"walk"``).
        clip_keys: Ordered clip mode names (the generator's ``CLIPS`` keys).

    Returns:
        The integer mode index into ``clip_keys``.

    Raises:
        ValueError: If ``style`` is an out-of-range index or an unknown name.
            The message lists the available modes (no silent clamp - a wrong
            mode silently picks the wrong motion).
    """
    if isinstance(style, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"MotionBricks style must be a mode index or name, not a bool: {style!r}")
    if isinstance(style, int):
        if style < 0 or style >= len(clip_keys):
            raise ValueError(
                f"MotionBricks style index {style} out of range [0, {len(clip_keys)}); available modes: {clip_keys}"
            )
        return style
    if isinstance(style, str):
        if style not in clip_keys:
            raise ValueError(f"MotionBricks style {style!r} is not a known mode; available modes: {clip_keys}")
        return clip_keys.index(style)
    raise ValueError(f"MotionBricks style must be an int index or str name, got {type(style).__name__}")


def _unit_direction(vec: Any, default: tuple[float, float, float]) -> list[float]:
    """Normalise a 2- or 3-vector to a unit 3-vector in the world XY plane.

    A near-zero vector falls back to ``default`` (so an all-zero command does
    not produce a NaN direction).
    """
    raw = np.asarray(vec, dtype=np.float64).ravel()
    if raw.shape[0] < 2:
        raise ValueError(f"direction vector must have 2 or 3 entries, got {raw.shape[0]}")
    # Project onto the world XY plane (ignore any z component).
    planar = np.array([float(raw[0]), float(raw[1]), 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(planar))
    if norm < 1e-6:
        return list(default)
    return (planar / norm).tolist()


def _heading_to_direction(kwargs: dict[str, Any], movement_direction: list[float]) -> list[float]:
    """Resolve the facing direction from kwargs, defaulting to the movement direction."""
    if kwargs.get("target_heading_angle") is not None:
        angle = float(kwargs["target_heading_angle"])
        return [math.cos(angle), math.sin(angle), 0.0]
    if kwargs.get("target_heading") is not None:
        return _unit_direction(kwargs["target_heading"], tuple(movement_direction))  # type: ignore[arg-type]
    return list(movement_direction)


def allowed_pred_num_tokens(
    mode_idx: int,
    clip_token_specs: list[list[int] | None],
    min_token: int,
    max_token: int,
) -> list[int]:
    """Per-mode horizon mask, reproducing the upstream controller helper.

    Mirrors ``base_controller.get_default_allowed_pred_num_tokens``: a mode
    either declares an explicit ``allowed_pred_num_tokens`` mask, or defaults to
    an all-ones mask of width ``max_token - min_token + 1``.

    Args:
        mode_idx: Clip mode index.
        clip_token_specs: Per-mode explicit masks (``None`` where a mode
            declares none), indexed by mode.
        min_token: Generator min token count.
        max_token: Generator max token count.

    Returns:
        The horizon mask as a plain ``list[int]``.

    Raises:
        ValueError: If ``mode_idx`` is out of range, or the token range is
            degenerate (``max_token < min_token``).
    """
    if mode_idx < 0 or mode_idx >= len(clip_token_specs):
        raise ValueError(f"mode index {mode_idx} out of range [0, {len(clip_token_specs)})")
    if max_token < min_token:
        raise ValueError(f"max_token ({max_token}) must be >= min_token ({min_token})")
    spec = clip_token_specs[mode_idx]
    if spec is not None:
        return [int(x) for x in spec]
    return [1] * (max_token - min_token + 1)


def build_control_signals(
    *,
    mode_idx: int,
    clip_token_specs: list[list[int] | None],
    min_token: int,
    max_token: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the plain (torch-free) control-signal dict for one generation step.

    Args:
        mode_idx: Resolved clip mode index (see :func:`resolve_mode`).
        clip_token_specs: Per-mode explicit horizon masks (``None`` where none).
        min_token: Generator min token count.
        max_token: Generator max token count.
        kwargs: The well-known goal kwargs (``target_velocity`` /
            ``target_heading`` / ``target_heading_angle``).

    Returns:
        A dict with ``movement_direction`` / ``facing_direction`` (each a
        ``list[float]`` of length 3), ``mode`` (``int``), and
        ``allowed_pred_num_tokens`` (``list[int]``). All python-native so it is
        JSON-serialisable and torch-independent; the policy wraps it in tensors.
    """
    movement = (
        _unit_direction(kwargs["target_velocity"], _DEFAULT_DIRECTION)
        if kwargs.get("target_velocity") is not None
        else list(_DEFAULT_DIRECTION)
    )
    facing = _heading_to_direction(kwargs, movement)
    return {
        "movement_direction": movement,
        "facing_direction": facing,
        "mode": int(mode_idx),
        "allowed_pred_num_tokens": allowed_pred_num_tokens(mode_idx, clip_token_specs, min_token, max_token),
    }


__all__ = ["resolve_mode", "allowed_pred_num_tokens", "build_control_signals"]
