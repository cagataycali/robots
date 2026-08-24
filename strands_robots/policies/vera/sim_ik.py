"""Inverse-kinematics bridge: VERA EE-delta action chunk -> MuJoCo joint targets.

VERA's ``mimicgen`` (``eef_delta``) and ``droid`` (``cartesian_delta``)
embodiments emit, per step, a **6-DoF end-effector delta** (translation +
rotation) plus an optional gripper column. MuJoCo arm actuators are commanded in
**joint space**, so closing the sim loop needs an IK step that maps each
Cartesian *delta* onto an absolute target pose and solves it to joint angles.

The generic damped-least-squares solver wrapper is the shared
:class:`strands_robots.simulation.ik.MinkIKBridge` (one home for the mink
``FrameTask`` + ``PostureTask`` solve loop; the cosmos3 provider - which
decodes *absolute* EE pose trajectories in
:mod:`~strands_robots.policies.cosmos3.sim_ik` - and the simulation motion
primitives use the same class). This module subclasses it only to brand the
install errors with the ``sim-mujoco`` extra, and keeps the VERA-specific
decode glue (:func:`decode_vera_delta_chunk_to_targets`) local so a change to
one model's action semantics can never silently break the other.

``mink`` + ``mujoco`` are imported lazily so importing the VERA provider in the
light base env (no torch / no sim) stays cheap; a missing stack raises an
actionable install error rather than a silent default.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from strands_robots.simulation.ik import MinkIKBridge as _SharedMinkIKBridge
from strands_robots.simulation.ik import resolve_qp_solver
from strands_robots.utils import finite_number_error, positive_finite_number_error

logger = logging.getLogger(__name__)


def _install_hint() -> str:
    """Actionable message when the IK stack (mink + mujoco) is not importable."""
    return (
        "The VERA eef-delta IK-to-MuJoCo bridge needs 'mink' + 'mujoco', which "
        "were not importable. Install the sim extra:\n"
        "  uv pip install 'strands-robots[sim-mujoco]'\n"
        "This turns VERA's end-effector delta chunk (mimicgen/droid) into joint "
        "targets the MuJoCo arm can track. For joint_position embodiments "
        "(allegro) no IK is needed - the action maps directly to joints."
    )


_NO_BACKEND_MSG = (
    "No qpsolvers backend is installed; the VERA IK bridge needs one "
    "(e.g. 'daqp' or 'quadprog'). Install the sim extra: "
    "uv pip install 'strands-robots[sim-mujoco]'."
)


def _resolve_qp_solver(requested: str | None) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    Delegates to the shared :func:`strands_robots.simulation.ik.resolve_qp_solver`
    with VERA-branded errors: the install hint and no-backend message name the
    ``sim-mujoco`` extra so a clean-install user is pointed at the right
    dependency set (no silent fallback to an unrequested solver).
    """
    return resolve_qp_solver(requested, install_hint=_install_hint(), no_backend_msg=_NO_BACKEND_MSG)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6D rotation representation into a ``(3, 3)`` matrix.

    The 6D rep (Zhou et al. 2019) is the first two columns of the rotation
    matrix; the third is their cross product. Robust to non-orthonormal input.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = r[:3], r[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert an axis-angle 3-vector (rotation vector) to a ``(3, 3)`` matrix."""
    v = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


#: The rotation-delta encodings :func:`delta_to_matrix` implements, as the width
#: of the rotation block: 3 for an axis-angle delta, 6 for rot6d. Single owner of
#: that vocabulary - both the dispatch's refusal and
#: :func:`coerce_rotation_dim` are built from it, so the guard and the dispatch
#: cannot describe different sets.
ROTATION_DIMS: tuple[int, int] = (3, 6)


def coerce_rotation_dim(value: Any, param: str, context: str) -> tuple[int | None, str | None]:
    """Validate a rotation-delta encoding width and normalize it to an ``int``.

    The accepted set is an *enumeration* rather than an interval, and a closed
    one: :func:`delta_to_matrix` implements exactly the two encodings in
    :data:`ROTATION_DIMS` and raises for every other width, so there is no third
    parameterization a caller could ask for and no endpoint left to decide.

    Every other value is refused where it is supplied, because none of them is
    refused usefully further down:

    * ``0``, ``-3``, ``2`` and ``4`` reach the dispatch and raise there - which
      on the policy path means mid-rollout, inside ``get_actions``, after the
      server handshake and the IK bridge build rather than at the call that
      supplied them.
    * ``2.7`` and ``True`` were truncated by an ``int()`` coercion first, so that
      refusal named ``2`` and ``1``: a width the caller never supplied.
    * a numeric string, a non-integral float and ``nan`` reach the per-step
      rotation slice ``step[3 : 3 + rotation_dim]`` and raise ``TypeError: slice
      indices must be integers``, naming neither the parameter nor the surface,
      and ``inf`` reports needing ``>= inf`` pose dims.

    Normalizing is load-bearing rather than cosmetic: the width indexes that
    slice, so an integral float - what a JSON or YAML config read produces, and
    which the dispatch accepts - has to arrive there as an index. Validation
    decides whether the value can be honored; the conversion makes the honored
    one consumable.

    Numeric-ness, ``bool`` rejection, finiteness and the float64 range are
    :func:`~strands_robots.utils.finite_number_error`'s, whose domain this is a
    finite subset of; only membership is decided here.

    Args:
        value: The caller-supplied encoding width.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method or function name.

    Returns:
        ``(width, None)`` when usable, else ``(None, message)``.
    """
    if (err := finite_number_error(value, param, context)) is not None:
        return None, err
    numeric = float(value)
    if numeric != int(numeric) or int(numeric) not in ROTATION_DIMS:
        accepted = " or ".join(
            f"{dim} ({label})" for dim, label in zip(ROTATION_DIMS, ("axis-angle", "rot6d"), strict=True)
        )
        return None, f"{context}: {param} must be {accepted}, got {value!r}."
    return int(numeric), None


def delta_to_matrix(rot_delta: np.ndarray, rotation_dim: int) -> np.ndarray:
    """Map a rotation delta (``rotation_dim`` ∈ {3 axis-angle, 6 rot6d}) -> (3,3)."""
    if rotation_dim == 6:
        return rot6d_to_matrix(rot_delta)
    if rotation_dim == 3:
        return axis_angle_to_matrix(rot_delta)
    raise ValueError(
        f"unsupported rotation_dim {rotation_dim!r}; use {ROTATION_DIMS[0]} (axis-angle) or {ROTATION_DIMS[1]} (rot6d)"
    )


class MinkIKBridge(_SharedMinkIKBridge):
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    The VERA branding of the shared
    :class:`strands_robots.simulation.ik.MinkIKBridge` (same solver, tasks, and
    convergence behavior): a missing ``mink``/``qpsolvers`` stack raises the
    ``sim-mujoco`` install hint. See the shared class for the full
    constructor/solve contract.
    """

    _INSTALL_HINT: ClassVar[str] = _install_hint()
    _NO_BACKEND_MSG: ClassVar[str] = _NO_BACKEND_MSG
    _LOG_LABEL: ClassVar[str] = "VERA MinkIKBridge"


#: The ``gripper_dim_index`` value meaning "whichever column is last". VERA's
#: server metadata uses it as the default, so it is a wire-level sentinel rather
#: than a convenience: an explicit index and this sentinel are the only two
#: things a caller can mean. Single owner of that vocabulary -
#: :func:`coerce_gripper_dim_index` and the resolution in
#: :func:`decode_vera_delta_chunk_to_targets` are both built from it.
GRIPPER_INDEX_LAST: int = -1


def coerce_gripper_dim_index(value: Any, param: str, context: str) -> tuple[int | None, str | None]:
    """Validate a gripper-column index and normalize it to an ``int``.

    The accepted set is :data:`GRIPPER_INDEX_LAST` (the trailing column) or a
    column index ``>= 0``. It is closed for the same reason the encoding width's
    is: the value selects one column of the action chunk, so there is no third
    thing it could mean, and no endpoint left to decide.

    Every other value is refused where it is supplied, because none of them is
    refused usefully further down - the index reaches
    ``action_chunk[:, gidx]`` and ``np.delete(action_chunk, gidx, axis=1)``:

    * a negative other than the sentinel - ``-5``, ``-99`` - and ``nan`` all fail
      the ``>= 0`` test the resolution used, so each was answered with the
      *default*: the trailing column, exactly as if the sentinel had been
      supplied. A request no column satisfies became the documented default with
      nothing logged and nothing returned to say so, which is the one outcome a
      caller cannot detect - the value they meant and the value that was used
      differ, and both calls report the same clean joint targets.
    * a non-integral float, ``inf``, ``True`` and a numeric string reach the
      index itself and raise ``IndexError: only integers, slices ...`` or
      ``ValueError: boolean array argument obj to delete ...``, naming neither
      the parameter nor the surface.
    * ``None`` and a list fail the ``>= 0`` comparison with ``TypeError: '>='
      not supported between instances of ...``, which is not the ``ValueError``
      channel this function documents.

    Normalizing is load-bearing rather than cosmetic, exactly as for
    :func:`coerce_rotation_dim`: the value indexes a column, so an integral
    float - what a JSON or YAML config read produces, and what
    ``int(meta["gripper_dim_index"])`` produced on the provider path - has to
    arrive there as an index. Validation decides whether the value can be
    honored; the conversion makes the honored one consumable.

    Numeric-ness, ``bool`` rejection, finiteness and the float64 range are
    :func:`~strands_robots.utils.finite_number_error`'s, whose domain this is a
    subset of; only the sentinel and the sign are decided here. Whether an
    in-range index actually addresses a column of *this* chunk needs the chunk's
    width, so it is checked by
    :func:`decode_vera_delta_chunk_to_targets` where that width is known.

    Args:
        value: The caller-supplied gripper-column index.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method or function name.

    Returns:
        ``(index, None)`` when usable, else ``(None, message)``.
    """
    if (err := finite_number_error(value, param, context)) is not None:
        return None, err
    numeric = float(value)
    index = int(numeric)
    if numeric != index or (index < 0 and index != GRIPPER_INDEX_LAST):
        return None, (
            f"{context}: {param} must be {GRIPPER_INDEX_LAST} (the trailing column) "
            f"or a column index >= 0, got {value!r}."
        )
    return index, None


def decode_vera_delta_chunk_to_targets(
    action_chunk: np.ndarray,
    ik_bridge: MinkIKBridge,
    q_init: np.ndarray,
    *,
    rotation_dim: int = 3,
    has_gripper: bool = True,
    gripper_dim_index: int = -1,
    translation_scale: float = 1.0,
) -> dict[str, Any]:
    """Turn a VERA EE-**delta** action chunk into MuJoCo joint targets via IK.

    VERA emits, per step, ``[translation(3), rotation(rotation_dim), gripper?]``
    as a delta on the *current* end-effector pose. We re-anchor each delta on the
    arm's **achieved** EE pose (closed loop - the FK of the previous IK solve),
    mirroring how robot deploy servers anchor on the observed pose so per-step
    tracking error stays bounded instead of compounding down the chunk.

    Args:
        action_chunk: ``[T, D]`` VERA action chunk (per-step EE delta + gripper).
        ik_bridge: A :class:`MinkIKBridge` over the target arm's MuJoCo model.
        q_init: Seed joint config (length ``model.nq``) - the robot's current pose.
        rotation_dim: 3 (axis-angle) or 6 (rot6d) rotation delta encoding -
            the two the decoder implements. Any other width is refused up
            front; see the ``Raises`` section.
        has_gripper: Whether the chunk carries a trailing gripper column.
        gripper_dim_index: Index of the gripper column, or
            :data:`GRIPPER_INDEX_LAST` for the trailing one; the value read out
            of it is passed through (binarized by caller). Read only when
            ``has_gripper``, and then it must address a column of this chunk;
            see the ``Raises`` section.
        translation_scale: Multiplier on the translation delta, composed on
            top of the OSC position scale (units match). Must be a positive
            finite number; see the ``Raises`` section for why.

    Raises:
        ValueError: If ``action_chunk`` is not ``[T, D]``, if it carries too
            few pose dims, if ``rotation_dim`` is not one of the encodings
            :func:`delta_to_matrix` implements, if ``translation_scale`` is
            not a positive finite number, or if ``gripper_dim_index`` names no
            column of the chunk - see :func:`coerce_gripper_dim_index` for why
            an index outside that set is not refused by anything downstream. An unusable ``rotation_dim`` is
            refused here for the same reason the scale is: it indexes the
            rotation block of every step, so a non-integral or non-numeric
            width raised ``TypeError: slice indices must be integers`` out of
            that slice - naming neither the parameter nor this function - and
            ``inf`` reported needing ``>= inf`` pose dims. The scale multiplies every translation delta in
            the chunk, so an unusable one is not refused by anything
            downstream: ``0`` discards the translation half of every action
            and returns only the rotation, a negative value inverts it, and
            ``nan``/``inf`` make **every** returned joint target non-finite
            (along with the ``tracking_error`` that would otherwise report
            it). ``send_action`` then refuses each of those targets for
            being non-finite, which reads as a wrong-embodiment action-key
            mismatch rather than as the scale that caused it.

    Returns:
        ``{"qpos": [T, nq], "gripper": [T] | None, "tracking_error": {...}}``.
    """
    # Refused before the first IK solve. This multiplies every translation
    # delta in the chunk, so an unusable value is applied rather than
    # rejected - see Raises. The domain matches the two sibling action
    # multipliers (``SimEnv.action_scale``, ``WBCConfig.action_scale``).
    if (
        err := positive_finite_number_error(
            translation_scale, "translation_scale", "decode_vera_delta_chunk_to_targets"
        )
    ) is not None:
        raise ValueError(err)
    # Same place, same reason: the width indexes the rotation block of every
    # step, so an unusable one is not refused downstream - it slices the wrong
    # columns out of the pose, or fails to slice at all. Normalizing to ``int``
    # is what lets an integral float reach that slice, which needs an index.
    dim, dim_err = coerce_rotation_dim(rotation_dim, "rotation_dim", "decode_vera_delta_chunk_to_targets")
    if dim is None:
        raise ValueError(dim_err)
    rotation_dim = dim
    # Third of the three, same place, same reason: the index selects the gripper
    # column of every step. Checked only when a gripper column is claimed, since
    # ``has_gripper=False`` means nothing reads it - refusing a value this call
    # never consumes would be a false rejection. Whether an in-range index
    # addresses a column of *this* chunk needs ``D``, so that half is below.
    if has_gripper:
        gidx_value, gidx_err = coerce_gripper_dim_index(
            gripper_dim_index, "gripper_dim_index", "decode_vera_delta_chunk_to_targets"
        )
        if gidx_value is None:
            raise ValueError(gidx_err)
        gripper_dim_index = gidx_value
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must be [T, D]; got {action_chunk.shape}")
    T, D = action_chunk.shape

    # Split gripper column off.
    gripper = None
    pose_block = action_chunk
    if has_gripper:
        gidx = D - 1 if gripper_dim_index == GRIPPER_INDEX_LAST else gripper_dim_index
        if gidx >= D:
            raise ValueError(
                f"decode_vera_delta_chunk_to_targets: gripper_dim_index {gripper_dim_index} "
                f"addresses no column of a {D}-wide action chunk (columns 0..{D - 1}, "
                f"or {GRIPPER_INDEX_LAST} for the trailing one)."
            )
        gripper = action_chunk[:, gidx].copy()
        pose_block = np.delete(action_chunk, gidx, axis=1)

    expected = 3 + rotation_dim
    if pose_block.shape[1] < expected:
        raise ValueError(
            f"VERA eef-delta needs >= {expected} pose dims (3 trans + {rotation_dim} rot); "
            f"got {pose_block.shape[1]} after removing gripper. Check rotation_dim/action_space."
        )

    q = np.asarray(q_init, dtype=np.float64).copy()
    achieved = ik_bridge.ee_pose(q)
    qpos_list: list[np.ndarray] = []
    err_list: list[float] = []
    for step in pose_block:
        # Robosuite OSC_POSE maps the policy's [-1,1] action to metric deltas via
        # output_max: translation *= 0.05 m, rotation *= 0.5 rad (control_delta=true,
        # input_max=1). VERA emits these normalized OSC actions, so we apply the
        # same scaling before IK -- without it the raw [-1,1] values are treated as
        # ~0.4 m steps, producing unreachable IK targets (track err > 1 m) and the
        # arm never descends to the object. translation_scale composes on top of
        # the OSC position scale for callers that need a further tweak.
        _OSC_POS_SCALE = 0.05
        _OSC_ROT_SCALE = 0.5
        trans = step[:3] * (_OSC_POS_SCALE * float(translation_scale))
        rot = step[3 : 3 + rotation_dim] * _OSC_ROT_SCALE
        # VERA/MimicGen eef_delta follows robosuite OSC_POSE: translation deltas
        # are in the WORLD/base frame (added to the EE position), not the tool
        # frame. Rotation deltas premultiply (world-frame) the current EE
        # orientation. Composing translation in the tool frame (achieved @ delta)
        # rotates a "move down" command by the gripper's orientation, so the arm
        # barely descends -- the cube never gets reached. Apply world-frame.
        rot_delta = delta_to_matrix(rot, rotation_dim)
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = rot_delta @ achieved[:3, :3]  # world-frame rotation delta
        target[:3, 3] = achieved[:3, 3] + trans  # world-frame translation delta
        q = ik_bridge.solve(target, q)
        achieved_new = ik_bridge.ee_pose(q)
        err_list.append(float(np.linalg.norm(achieved_new[:3, 3] - target[:3, 3])))
        achieved = achieved_new
        qpos_list.append(q.copy())

    nq = ik_bridge.model.nq
    qpos = np.stack(qpos_list) if qpos_list else np.empty((0, nq), dtype=np.float64)
    err_arr = np.asarray(err_list, dtype=np.float64)
    tracking = {
        "mean_mm": float(err_arr.mean() * 1000.0) if err_arr.size else 0.0,
        "max_mm": float(err_arr.max() * 1000.0) if err_arr.size else 0.0,
    }
    return {"qpos": qpos, "gripper": gripper, "tracking_error": tracking}
