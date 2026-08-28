"""Observation builder for the Pollen Microduck locomotion policies.

Assembles the flat, float32 observation vector the exported ONNX policies
consume, from the runtime observation dict produced by
:meth:`~strands_robots.simulation.base.SimEngine.get_observation`.

The layout is a fixed concatenation (measured off Pollen's reference
``microduck_rl/scripts/infer_policy.py`` and baked into every shipped ONNX's
``observation_names`` metadata)::

    base_ang_vel        (3)
    projected_gravity   (3)   world -Z rotated into the base frame
    joint_pos_relative  (14)  current joint pos - DEFAULT_POSE, contract order
    joint_vel           (14)  contract order
    last_action         (14)  the PREVIOUS raw ONNX output (not the motor target)
    command             (C)   unified command vector, C set by the policy

Total width is ``48 + C``: ``C = 13`` (``twist(3) + head_pose(4) + body_pose(6)``)
for the shipped alpha policies (61-D), ``C = 3`` for legacy twist-only policies
(51-D). The width is a parameter, never a hardcoded magic number, and unused
command slots stay PRESENT and zero (the dead-weight rule) so one obs layout
serves every policy in a bundle.

CRITICAL: ``EmpiricalNormalization`` is baked INTO the exported ONNX graph, so
the vector built here is fed RAW to the session. This module never normalises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: World gravity direction, rotated into the base frame to form the
#: ``projected_gravity`` observation block.
_WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

#: Component count of the ``base_ang_vel`` observation block (x, y, z).
_BASE_ANG_VEL_LEN = 3

#: Component count of the ``base_quat`` observation block (w, x, y, z).
_BASE_QUAT_LEN = 4


def quat_rotate_inverse(quat: NDArray[np.float32], vec: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rotate ``vec`` by the inverse of quaternion ``quat`` (``[w, x, y, z]``).

    Byte-for-byte the same formula Pollen's ``infer_policy.py`` uses to derive
    ``projected_gravity`` from the trunk orientation, so a rollout driven here
    feeds the network the same gravity block it saw in training.
    """
    q = np.asarray(quat, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    w = q[0]
    xyz = q[1:4]
    t = np.cross(xyz, v) * 2.0
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


def projected_gravity(base_quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """World ``-Z`` expressed in the base frame, from the base quaternion (wxyz)."""
    return quat_rotate_inverse(base_quat, _WORLD_GRAVITY)


def _require_base_block(observation_dict: dict[str, Any], key: str, expected_len: int) -> NDArray[np.float32]:
    """Read a fixed-width floating-base block out of the observation dict.

    The two base blocks are the only :func:`build_observation` inputs that arrive
    from the caller's observation dict rather than from the policy's own state, so
    this is the only place that holds them to a width. ``last_action`` and
    ``command`` are checked at the policy seam instead - the graph's output width
    in :meth:`MicroduckPolicy.get_actions` and a ``command`` override in
    ``_apply_command_kwargs`` - and neither seam ever sees these two.

    A wrong width used to be taken by a truncating slice, which is silent in both
    directions. One component short, ``q[1:4]`` is a 2-vector that ``np.cross``
    reads as planar, so ``base_quat`` silently loses its ``z``: the gravity block
    is still three finite components at the documented width, 7.5 degrees from the
    truth for a small-yaw pose at a norm of 0.991 - inside a percent of unity, and
    this module normalises nothing - rising to 20.7 degrees at 0.935. Over-long, a
    7-element ``[base_pos, base_quat]`` slice is read as a quaternion made of
    positions, 70.9 degrees off. A short ``base_ang_vel`` instead narrowed the
    returned vector below the documented ``48 + len(command)``, handing the graph
    fewer values than its own ``observation_names`` metadata declares. NumPy 2.0
    deprecated the 2-vector cross the short read relies on, so that path is on its
    way to raising from inside numpy rather than answering.

    The sibling locomotion observation builders hold their own sub-vectors the
    same way (``strands_robots.policies.wbc.observation._require_len``).

    Raises:
        KeyError: If ``key`` is absent from the observation dict.
        ValueError: If the block is not ``expected_len`` components wide.
    """
    block = np.asarray(observation_dict[key], dtype=np.float32).reshape(-1)
    if block.shape[0] != expected_len:
        raise ValueError(
            f"build_observation: observation_dict[{key!r}] has {block.shape[0]} "
            f"component(s) but the {key} observation block is {expected_len} wide. "
            f"A different width cannot be used: the block is read into a fixed slot "
            f"of the vector the ONNX graph consumes, so it either changes the "
            f"returned width away from the documented 48 + len(command) or re-reads "
            f"the block's own components from the wrong positions."
        )
    return block


def build_observation(
    observation_dict: dict[str, Any],
    *,
    joint_names: list[str],
    default_pose: NDArray[np.float32],
    last_action: NDArray[np.float32],
    command: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Assemble the raw float32 observation vector for one control tick.

    Args:
        observation_dict: Runtime observation. Reads the per-joint scalar
            ``<joint>`` (position) and ``<joint>.vel`` (velocity) keys in
            ``joint_names`` order, plus ``base_ang_vel`` (3) and ``base_quat``
            (4, wxyz).
        joint_names: The 14 actuated joints in CONTRACT order (never permute).
        default_pose: Per-joint neutral pose (rad), ``joint_names`` order; the
            ``joint_pos`` block is measured relative to it.
        last_action: The previous RAW ONNX output (14), zeros on the first tick.
        command: The unified command vector (width C, zero-padded dead weight).

    Returns:
        A 1-D ``float32`` array of length ``48 + len(command)``.

    Raises:
        KeyError: If a required joint / base key is absent from the obs dict.
        ValueError: If ``base_ang_vel`` or ``base_quat`` is not the width its
            observation block defines (3 and 4). Both are read into fixed slots
            of the returned vector, so another width either changes the returned
            width away from ``48 + len(command)`` or re-reads the block's own
            components from the wrong positions.
    """
    base_ang_vel = _require_base_block(observation_dict, "base_ang_vel", _BASE_ANG_VEL_LEN)
    grav = projected_gravity(_require_base_block(observation_dict, "base_quat", _BASE_QUAT_LEN))

    joint_pos = np.array([float(observation_dict[name]) for name in joint_names], dtype=np.float32)
    joint_pos_rel = joint_pos - np.asarray(default_pose, dtype=np.float32)
    joint_vel = np.array([float(observation_dict[f"{name}.vel"]) for name in joint_names], dtype=np.float32)

    return np.concatenate(
        [
            base_ang_vel,
            grav,
            joint_pos_rel,
            joint_vel,
            np.asarray(last_action, dtype=np.float32).reshape(-1),
            np.asarray(command, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def decode_action(
    raw_action: NDArray[np.float32],
    *,
    default_pose: NDArray[np.float32],
    action_scale: float,
) -> NDArray[np.float32]:
    """Decode a raw ONNX action into per-joint motor targets (rad).

    ``motor_target = DEFAULT_POSE + action * action_scale`` - the exact decode
    Pollen applies before the servo current-limit clip (a driver/sim concern,
    not part of this decode).
    """
    return (
        np.asarray(default_pose, dtype=np.float32)
        + np.asarray(raw_action, dtype=np.float32).reshape(-1) * float(action_scale)
    ).astype(np.float32)
