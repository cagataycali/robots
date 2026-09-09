"""Observation builder for the Pollen Microduck locomotion policies.

Assembles the flat, float32 observation vector the exported ONNX policies
consume, from the runtime observation dict produced by
:meth:`~strands_robots.simulation.base.SimEngine.get_observation`.

The layout is a fixed concatenation (measured off Pollen's reference
``microduck_rl/scripts/infer_policy.py`` and baked into every shipped ONNX's
``observation_names`` metadata)::

    base_ang_vel        (3)
    projected_gravity   (3)   a UNIT gravity direction in the base frame. A
                              ``projected_gravity`` export rotates world -Z
                              from ``base_quat``; a ``gravity_source="raw_accel"``
                              export estimates the same direction from
                              ``base_acc`` (3, m/s^2). The choice is
                              training-time and baked into the ONNX metadata.
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
the vector built here is fed RAW to the session. This module never rescales the
assembled vector. The two normalisations it performs are both inside slot two,
and neither is a scaling choice: :func:`raw_accel_gravity` returns the unit
direction the ``raw_accel`` export's own estimator returns, and
:func:`quat_rotate_inverse` normalises the orientation it reads because its
formula is a rotation only for a unit quaternion - a scaled quaternion encodes
the same rotation and has to answer the same.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from strands_robots.utils import MIN_QUATERNION_NORM

#: World gravity direction, rotated into the base frame to form the
#: ``projected_gravity`` observation block.
_WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

#: Component count of the ``base_ang_vel`` observation block (x, y, z).
_BASE_ANG_VEL_LEN = 3

#: Component count of the ``base_quat`` observation block (w, x, y, z).
_BASE_QUAT_LEN = 4

#: Component count of the ``base_acc`` observation block (accelerometer, m/s^2).
_BASE_ACC_LEN = 3

#: Component count of the projected-gravity block ``base_quat`` is reduced to.
_GRAVITY_LEN = 3

#: Magnitude below which the negated accelerometer carries no usable direction
#: and :func:`raw_accel_gravity` rotates ``base_quat`` instead. Pollen's own
#: threshold; free fall reads ``|accel| = 0`` and lands here.
_RAW_ACCEL_MIN_MAGNITUDE = 0.1

#: The two ONNX ``gravity_source`` values ``build_observation`` accepts. Slot two
#: of the vector is a UNIT gravity direction in the base frame either way: the
#: ``projected_gravity`` branch rotates world ``-Z`` from ``base_quat``, and the
#: ``raw_accel`` branch estimates the same direction from the accelerometer (see
#: :func:`raw_accel_gravity`). They are two estimators of ONE quantity, not two
#: quantities - on a settled duck they agree to 1e-6. Older Pollen exports and
#: some backlash variants ship with ``raw_accel``; every currently-shipped alpha
#: policy is ``projected_gravity``, which is why that stays the default when the
#: metadata is silent. Both spellings are exactly the two
#: ``self.use_projected_gravity`` branches Pollen's own ``get_observations``
#: selects between.
GRAVITY_SOURCE_PROJECTED = "projected_gravity"
GRAVITY_SOURCE_RAW_ACCEL = "raw_accel"
_GRAVITY_SOURCES: tuple[str, ...] = (GRAVITY_SOURCE_PROJECTED, GRAVITY_SOURCE_RAW_ACCEL)


def quat_rotate_inverse(quat: NDArray[np.float32], vec: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rotate ``vec`` by the inverse of quaternion ``quat`` (``[w, x, y, z]``).

    Byte-for-byte the same formula Pollen's ``infer_policy.py`` uses to derive
    ``projected_gravity`` from the trunk orientation, so a rollout driven here
    feeds the network the same gravity block it saw in training - read from the
    rotation the quaternion encodes rather than from the quaternion itself.

    The formula mixes a term quadratic in the components with a linear one, so
    scaling does not cancel: it is a rotation only for a UNIT quaternion. A
    quaternion scaled by any positive factor encodes the same rotation, which is
    why the library's orientation domain
    (:func:`~strands_robots.utils.coerce_orientation_quaternion`) accepts any
    magnitude - on the stated ground that "every consumer either normalizes or
    is scale-invariant". This is one of those consumers, so it normalizes: read
    unnormalized, a base orientation 2x off unit turns world ``-Z`` 41 deg away
    from the direction it encodes and hands the graph a slot-two block of
    magnitude 1.56, where the layout declares a unit direction. Pollen's own
    code needs no such step because it reads a fused-IMU orientation that is
    already unit; the values reaching a provider are not all of that kind - an
    IMU reading drifts off unit, and an orientation obtained by interpolating
    two samples is short by up to ~8%.

    A norm below :data:`~strands_robots.utils.MIN_QUATERNION_NORM` carries no
    direction at all and cannot be repaired by scaling, so it is refused rather
    than read. That case is the ``[0, 0, 0, 0]`` an unwritten or dropped
    orientation field spells, and the formula answers it with ``vec``
    unchanged - for world ``-Z``, exactly the gravity block of a PERFECTLY
    UPRIGHT base, the one attitude a locomotion policy most needs to tell apart
    from a fall. It passes the width guard in :func:`_require_base_block` and
    the finiteness pass in :func:`_non_finite_observation_error`, so this is the
    only place that sees it. The same-layer siblings refuse it the same way
    (:func:`strands_robots.policies.wbc.control.quat_rotate_inverse`,
    :func:`strands_robots.policies.protomotions.state_utils.quat_rotate_inverse`).

    Args:
        quat: Base orientation ``[w, x, y, z]``. Any magnitude is accepted and
            normalized here, so a scaled quaternion answers the same as the
            unit one it encodes.
        vec: The world-frame 3-vector to express in the base frame.

    Returns:
        ``vec`` expressed in the base frame, ``float32`` - unit-length whenever
        ``vec`` is. A ``nan``/``inf`` component propagates instead of being
        refused here, so the assembled-vector pass names it through the block it
        becomes.

    Raises:
        ValueError: If ``quat`` has no direction to recover (norm below
            :data:`~strands_robots.utils.MIN_QUATERNION_NORM`).
    """
    q = np.asarray(quat, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    # float64 for the norm alone: the components stay in the caller's float32,
    # matching the dtype the ONNX graph is fed.
    norm = float(np.linalg.norm(np.asarray(q, dtype=np.float64)))
    if norm < MIN_QUATERNION_NORM:
        raise ValueError(
            f"quat_rotate_inverse: the base orientation {np.asarray(q, dtype=np.float64).tolist()} "
            f"has norm {norm!r} and describes no rotation, so the gravity block it reduces "
            f"to carries no direction either. An all-zero quaternion is what an orientation "
            f"that was never written spells, and rotating world -Z by it returns world -Z - "
            f"the gravity block of a perfectly upright base. Supply the base's real wxyz "
            f"orientation (MuJoCo reports it as base_quat in get_observation)."
        )
    q = (q / np.float32(norm)).astype(np.float32)
    w = q[0]
    xyz = q[1:4]
    t = np.cross(xyz, v) * 2.0
    return (v - w * t + np.cross(xyz, t)).astype(np.float32)


def projected_gravity(base_quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """World ``-Z`` expressed in the base frame, from the base quaternion (wxyz).

    A UNIT direction for any ``base_quat`` magnitude: the rotation is read from
    the orientation the quaternion encodes, not from its components (see
    :func:`quat_rotate_inverse`).

    Args:
        base_quat: Base orientation ``[w, x, y, z]``, any magnitude.

    Returns:
        The unit gravity direction in the base frame (``float32``, 3).

    Raises:
        ValueError: If ``base_quat`` describes no rotation (~zero norm).
    """
    return quat_rotate_inverse(base_quat, _WORLD_GRAVITY)


def raw_accel_gravity(base_acc: NDArray[np.float32], base_quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """Gravity direction estimated from the accelerometer, as Pollen derives it.

    A resting IMU measures the reaction to gravity, so the NEGATED reading points
    along gravity, and normalising it gives the same UNIT direction
    :func:`projected_gravity` derives from the orientation. Pollen's
    ``get_raw_accelerometer`` is exactly that - negate, normalise, and, when the
    magnitude is too small to carry a direction, rotate world ``-Z`` instead:

        accel_negated = -accel_raw
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            return self.quat_rotate_inverse(quat, world_gravity)

    So the two ``gravity_source`` branches are two estimators of ONE quantity. On
    a settled duck (``|accel| = 9.81``) they agree to 1e-6. Writing ``base_acc``
    into slot two unchanged instead hands the network a vector 9.81x too long
    with every component sign-flipped, and a robot in free fall
    (``|accel| = 0``) a zero vector carrying no direction at all - both finite,
    both the documented width, and neither what the export was trained on.

    Args:
        base_acc: Accelerometer reading (3, m/s^2) in the base frame.
        base_quat: Base orientation (4, wxyz). Read only for the degenerate
            fallback, which is why this path requires it.

    Returns:
        A unit ``float32`` gravity direction in the base frame.

    Raises:
        ValueError: If the reading is degenerate AND ``base_quat`` describes no
            rotation, so neither estimator carries a direction.
    """
    negated = -np.asarray(base_acc, dtype=np.float32)
    magnitude = float(np.linalg.norm(negated))
    if magnitude > _RAW_ACCEL_MIN_MAGNITUDE:
        return (negated / magnitude).astype(np.float32)
    return projected_gravity(base_quat)


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
    is still three finite components at the documented width, 8.1 degrees from the
    truth for a small-yaw pose and 28.0 for a roll-then-yaw one - at EXACTLY unit
    length both times, because :func:`quat_rotate_inverse` normalises the orientation
    it is handed, so the truncation leaves no trace in the magnitude for anything
    downstream to screen. Over-long, a
    7-element ``[base_pos, base_quat]`` slice is read as a quaternion made of
    positions, 70.9 degrees off. A short ``base_ang_vel`` instead narrowed the
    returned vector below the documented ``48 + len(command)``, handing the graph
    fewer values than its own ``observation_names`` metadata declares.

    The short-``base_quat`` reading is numpy-version dependent, and both bands are
    inside the declared ``numpy>=1.21,<3.0``: the 2-vector cross it relies on was
    deprecated through the 2.0-2.4 line and removed in 2.5.0, so from there the
    same unguarded read raises out of ``np.cross`` - ``Both input arrays must be
    (arrays of) 3-dimensional vectors`` - naming neither this key, nor its width,
    nor the caller who supplied it. Silently wrong below that boundary and
    unattributable above it, so this guard is what names the block on either, and
    it is not a guard a newer numpy makes redundant.

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


def _non_finite_observation_error(
    observation: NDArray[np.float32],
    joint_names: list[str],
    command_len: int,
    gravity_source: str = GRAVITY_SOURCE_PROJECTED,
) -> str | None:
    """Name the blocks of an assembled observation that carry a non-finite value.

    ``build_observation`` holds its two floating-base blocks to a width, and a
    width is the only thing it held them to: a ``nan`` or ``inf`` component
    passed straight into the vector the ONNX graph reads. That is silent when
    the graph tolerates it, and misattributed when it does not - the value
    propagates and
    :meth:`~strands_robots.policies.microduck.MicroduckPolicy.get_actions`
    refuses it as ``'the ONNX action'``, blaming the checkpoint's graph for a
    number the caller supplied.

    The check runs on the ASSEMBLED vector rather than on each input, because
    the assembled vector is the one place every input path meets: the two base
    blocks, the ``len(joint_names)`` position and velocity scalars, the previous
    action and the command. One pass covers all of them.

    It is a plain :func:`numpy.isfinite` rather than
    :func:`~strands_robots.utils.finite_vector_error` because at this point the
    value is a ``float32`` 1-D array this function built, so none of the
    spellings that shared domain exists to judge - a nested sequence, a
    ``bool``, a 0-d scalar, a non-numeric - can reach it, and the two agree on
    everything that can. On the shipped 51-wide observation the shared domain
    costs 40.68 us against a 37.92 us build, where this costs 1.27 us.

    Args:
        observation: The assembled 1-D ``float32`` observation vector.
        joint_names: The joints the position / velocity blocks were read for, in
            the order they were read; used to name the offending joint.
        command_len: Width of the trailing command block.

    Returns:
        A message naming each offending block, or ``None`` when every component
        is finite. Built only on the refusal path, so the happy path pays for
        the :func:`numpy.isfinite` pass alone.
    """
    if bool(np.isfinite(observation).all()):
        return None
    bad = np.flatnonzero(~np.isfinite(observation))

    nj = len(joint_names)
    slot_two_name = "projected_gravity (from base_quat)" if gravity_source == GRAVITY_SOURCE_PROJECTED else "base_acc"
    layout: list[tuple[str, int]] = [
        ("base_ang_vel", _BASE_ANG_VEL_LEN),
        (slot_two_name, _GRAVITY_LEN),
        ("joint_pos", nj),
        ("joint_vel", nj),
        ("last_action", nj),
        ("command", command_len),
    ]
    offenders: list[str] = []
    start = 0
    for name, width in layout:
        hits = [int(i) - start for i in bad if start <= int(i) < start + width]
        if hits:
            if name in ("joint_pos", "joint_vel"):
                joints = ", ".join(joint_names[h] for h in hits)
                offenders.append(f"{name} ({joints})")
            else:
                offenders.append(f"{name} at {hits}")
        start += width
    return (
        f"build_observation: the assembled observation carries {bad.size} non-finite "
        f"component(s) in {', '.join(offenders)}. A nan/inf here is read into the "
        f"vector the ONNX graph consumes: the graph propagates it and get_actions "
        f"then refuses 'the ONNX action', reporting the checkpoint's graph for a "
        f"number this observation supplied."
    )


def build_observation(
    observation_dict: dict[str, Any],
    *,
    joint_names: list[str],
    default_pose: NDArray[np.float32],
    last_action: NDArray[np.float32],
    command: NDArray[np.float32],
    gravity_source: str = GRAVITY_SOURCE_PROJECTED,
) -> NDArray[np.float32]:
    """Assemble the raw float32 observation vector for one control tick.

    Args:
        observation_dict: Runtime observation. Reads the per-joint scalar
            ``<joint>`` (position) and ``<joint>.vel`` (velocity) keys in
            ``joint_names`` order, plus ``base_ang_vel`` (3) and either
            ``base_quat`` (4, wxyz) or ``base_acc`` (3, m/s^2) depending on
            ``gravity_source``.
        joint_names: The 14 actuated joints in CONTRACT order (never permute).
        default_pose: Per-joint neutral pose (rad), ``joint_names`` order; the
            ``joint_pos`` block is measured relative to it.
        last_action: The previous RAW ONNX output (14), zeros on the first tick.
        command: The unified command vector (width C, zero-padded dead weight).
        gravity_source: Which reading fills slot two of the vector.
            ``"projected_gravity"`` (default) rotates world ``-Z`` into the
            base frame from ``base_quat`` (the shipped alpha convention);
            ``"raw_accel"`` estimates the same unit direction from ``base_acc``
            (older exports, some backlash variants) - see
            :func:`raw_accel_gravity`.  Both are the two branches Pollen's own
            ``get_observations`` selects between under
            ``self.use_projected_gravity``; the choice is a training-time flag
            baked into the export, so ``MicroduckPolicy`` reads it out of the
            ONNX ``custom_metadata_map`` (``gravity_source`` key) and threads
            it here.  Any other value is refused, naming the shipped pair, so
            a caller who mistypes ``"gravity"`` learns from the raise rather
            than from a silently-wrong rollout - slot two would keep the
            documented width and stay finite while its meaning did not match
            what the network was trained on, exactly the shape of drift
            harness#388 was filed against.

    Returns:
        A 1-D ``float32`` array of length ``48 + len(command)``.

    Raises:
        KeyError: If a required joint / base key is absent from the obs dict.
            ``raw_accel`` needs BOTH ``base_acc`` and ``base_quat``, because the
            degenerate-reading fallback is the rotation; requiring it up front
            refuses at the first tick rather than at the moment the robot leaves
            the ground.
        ValueError: If ``base_ang_vel`` or the ``gravity_source`` block is not
            the width its observation block defines (3 and 4 for ``base_quat``,
            3 for ``base_acc``). Both are read into fixed slots of the returned
            vector, so another width either changes the returned width away
            from ``48 + len(command)`` or re-reads the block's own components
            from the wrong positions. Or if ``gravity_source`` is neither of
            the two shipped spellings. Or if any component of the assembled
            vector is not finite - the offending blocks are named, and a joint
            block names the joint. A ``nan``/``inf`` reaches the graph
            otherwise, which propagates it and makes ``get_actions`` refuse it
            as ``'the ONNX action'``. Or if ``base_quat`` describes no rotation
            (~zero norm, the spelling of an orientation that was never written):
            the width and finiteness passes both accept it while the gravity
            block it reduces to reads as a perfectly upright base, so
            :func:`quat_rotate_inverse` refuses it instead.
    """
    if gravity_source not in _GRAVITY_SOURCES:
        raise ValueError(
            f"build_observation: gravity_source={gravity_source!r} is not one of "
            f"{list(_GRAVITY_SOURCES)}. This value is a training-time flag baked into "
            f"the ONNX export (Pollen's use_projected_gravity), and the two branches "
            f"read different base keys (base_quat vs base_acc); a third spelling has "
            f"no defined slot-two contract."
        )

    base_ang_vel = _require_base_block(observation_dict, "base_ang_vel", _BASE_ANG_VEL_LEN)
    if gravity_source == GRAVITY_SOURCE_PROJECTED:
        grav = projected_gravity(_require_base_block(observation_dict, "base_quat", _BASE_QUAT_LEN))
    else:
        # ``raw_accel``: the accelerometer is a second ESTIMATOR of the same unit
        # gravity direction, not a second quantity - Pollen's
        # ``get_raw_accelerometer`` negates, normalises, and falls back to the
        # rotation for a degenerate reading. See :func:`raw_accel_gravity`. This
        # is the one normalisation this module performs, and it is the export's
        # own contract rather than a scaling choice: the fused
        # ``EmpiricalNormalization`` inside the graph is still the only thing
        # that rescales the assembled vector.
        grav = raw_accel_gravity(
            _require_base_block(observation_dict, "base_acc", _BASE_ACC_LEN),
            _require_base_block(observation_dict, "base_quat", _BASE_QUAT_LEN),
        )

    joint_pos = np.array([float(observation_dict[name]) for name in joint_names], dtype=np.float32)
    joint_pos_rel = joint_pos - np.asarray(default_pose, dtype=np.float32)
    joint_vel = np.array([float(observation_dict[f"{name}.vel"]) for name in joint_names], dtype=np.float32)

    command_block = np.asarray(command, dtype=np.float32).reshape(-1)
    observation = np.concatenate(
        [
            base_ang_vel,
            grav,
            joint_pos_rel,
            joint_vel,
            np.asarray(last_action, dtype=np.float32).reshape(-1),
            command_block,
        ]
    ).astype(np.float32)
    if reason := _non_finite_observation_error(observation, joint_names, command_block.shape[0], gravity_source):
        raise ValueError(reason)
    return observation


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
