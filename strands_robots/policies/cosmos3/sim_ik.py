"""Inverse-kinematics bridge: Cosmos 3 EE-pose trajectory -> MuJoCo joint targets.

A Cosmos 3 action chunk decodes (see :mod:`action_decode`) to an absolute
end-effector pose trajectory in Cartesian space. MuJoCo arm actuators are
commanded in **joint space**, so closing the sim loop needs an IK step that
maps each Cartesian target to joint angles.

:class:`MinkIKBridge` wraps `mink <https://github.com/kevinzakka/mink>`_, a
differential-IK library that works directly on the same ``mujoco.MjModel`` (no
URDF or second kinematics engine). Per pose it runs a damped least-squares
``solve_ik`` with a Cartesian :class:`mink.FrameTask` on the end-effector body
plus a :class:`mink.PostureTask` regularizer, integrating the joint velocity
over the control timestep. ``mink`` + ``mujoco`` are imported lazily so the
``cosmos3-diffusers`` extra alone (no sim) stays importable; a missing stack
raises an actionable install error (AGENTS.md key convention #6, no silent
default).

This is a geometric post-step applied *after* Cosmos, not part of the model.
The Cosmos "modes" (``policy`` / ``forward_dynamics`` / ``inverse_dynamics``)
are world-model conditioning modes, not robot kinematics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco

    from .embodiments import Cosmos3Embodiment

logger = logging.getLogger(__name__)


def _install_hint() -> str:
    """Actionable message when the IK stack (mink + mujoco) is not importable."""
    return (
        "Cosmos 3 IK-to-MuJoCo bridge needs the 'cosmos3-sim' extra (mink + "
        "mujoco), which was not importable. Install it with:\n"
        "  uv pip install strands-robots[cosmos3-sim]\n"
        "This pulls mink (differential IK on the MuJoCo model) and mujoco. It "
        "turns the Cosmos end-effector pose trajectory into joint targets the "
        "MuJoCo arm can track."
    )


_PREFERRED_QP_SOLVERS = ("daqp", "quadprog", "osqp", "proxqp", "cvxopt", "scs")


def _resolve_qp_solver(requested: str | None) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    ``mink`` defaults to (and pins) ``daqp``, but environments commonly ship
    only ``quadprog``. Auto-selecting from ``qpsolvers.available_solvers``
    (preferring daqp, then quadprog) keeps the IK bridge working everywhere
    without forcing an extra QP dependency. An explicit ``requested`` name is
    honoured when installed; if it is not, we fail with an actionable error that
    lists what *is* available (AGENTS.md #6 - no silent fallback to a solver the
    caller did not ask for, but also no opaque KeyError deep in qpsolvers).
    """
    try:
        from qpsolvers import available_solvers
    except ImportError as e:
        raise ImportError(_install_hint()) from e
    available = list(available_solvers)
    if not available:
        raise RuntimeError(
            "No qpsolvers backend is installed; the Cosmos 3 IK bridge needs one "
            "(e.g. 'daqp' or 'quadprog'). Install the cosmos3-sim extra: "
            "uv pip install 'strands-robots[cosmos3-sim]'."
        )
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested qpsolvers backend {requested!r} is not installed. "
                f"Available: {available}. Install it (e.g. pip install "
                f"'qpsolvers[{requested}]') or pass an available solver / None."
            )
        return requested
    for name in _PREFERRED_QP_SOLVERS:
        if name in available:
            return name
    return available[0]


class MinkIKBridge:
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    Args:
        model: The ``mujoco.MjModel`` for the arm being controlled.
        ee_frame_name: Name of the end-effector frame (a body or site) the
            Cartesian task tracks (e.g. ``"hand"`` for a Franka/Panda).
        ee_frame_type: ``"body"`` (default), ``"site"``, or ``"geom"`` - the
            ``mink.FrameTask`` frame type for ``ee_frame_name``.
        position_cost: Cartesian position task weight.
        orientation_cost: Cartesian orientation task weight.
        posture_cost: Posture (joint-regularizer) task weight - keeps the solve
            near the current configuration so it stays smooth and avoids
            flipping between IK branches.
        solver: ``qpsolvers`` backend name passed to ``mink.solve_ik``.
            ``None`` (default) auto-selects an installed backend - preferring
            ``"daqp"`` (what ``mink`` pins), then ``"quadprog"``, then whatever
            ``qpsolvers.available_solvers`` reports - so the bridge runs whether
            the env has daqp or only quadprog. Pass an explicit name to force one.
        damping: Levenberg-Marquardt damping for ``solve_ik``.
        max_iters: Max differential-IK iterations per target pose.
        dt: Integration timestep for each IK iteration (s).
        pos_threshold: Convergence threshold on position error (m).
        ori_threshold: Convergence threshold on orientation error (rad).

    Raises:
        ImportError: If ``mink``/``mujoco`` are not importable (with an
            actionable install hint).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_frame_name: str,
        ee_frame_type: str = "body",
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        solver: str | None = None,
        damping: float = 1e-3,
        max_iters: int = 20,
        dt: float = 1e-2,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
    ) -> None:
        try:
            import mink
        except ImportError as e:
            raise ImportError(_install_hint()) from e

        self._mink = mink
        self.model = model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_type = ee_frame_type
        self.solver = _resolve_qp_solver(solver)
        self.damping = damping
        self.max_iters = max_iters
        self.dt = dt
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold

        self._configuration = mink.Configuration(model)
        self._frame_task = mink.FrameTask(
            frame_name=ee_frame_name,
            frame_type=ee_frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self._tasks = [self._frame_task, self._posture_task]
        logger.info(
            "MinkIKBridge ready [ee=%s/%s solver=%s nq=%d]",
            ee_frame_type,
            ee_frame_name,
            solver,
            model.nq,
        )

    def ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Forward kinematics: ``(4, 4)`` EE pose at a joint configuration.

        Args:
            qpos: Joint configuration of length ``model.nq``.

        Returns:
            The end-effector frame's absolute ``(4, 4)`` homogeneous pose.
        """
        self._configuration.update(np.asarray(qpos, dtype=np.float64))
        transform = self._configuration.get_transform_frame_to_world(self.ee_frame_name, self.ee_frame_type)
        return np.asarray(transform.as_matrix(), dtype=np.float32)

    def solve(self, target_pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Solve IK for a single Cartesian target from a seed configuration.

        Args:
            target_pose: Desired EE ``(4, 4)`` homogeneous pose.
            q_init: Seed joint configuration (length ``model.nq``); the solve is
                warm-started here and the posture task regularizes toward it.

        Returns:
            The solved joint configuration (length ``model.nq``, ``float64``).
        """
        mink = self._mink
        q = np.asarray(q_init, dtype=np.float64).copy()
        self._configuration.update(q)
        self._posture_task.set_target(q)

        target = mink.SE3.from_matrix(np.asarray(target_pose, dtype=np.float64))
        self._frame_task.set_target(target)

        for _ in range(self.max_iters):
            velocity = mink.solve_ik(self._configuration, self._tasks, self.dt, self.solver, self.damping)
            self._configuration.integrate_inplace(velocity, self.dt)
            err = self._frame_task.compute_error(self._configuration)
            if np.linalg.norm(err[:3]) <= self.pos_threshold and np.linalg.norm(err[3:]) <= self.ori_threshold:
                break
        return np.asarray(self._configuration.q, dtype=np.float64).copy()

    def solve_trajectory(self, poses: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Solve IK for an EE-pose trajectory, warm-starting each step.

        Args:
            poses: Absolute EE poses of shape ``[N, 4, 4]``.
            q_init: Seed joint configuration for the first pose; each subsequent
                solve warm-starts from the previous solution so the joint
                trajectory stays continuous.

        Returns:
            Joint configurations of shape ``[N, model.nq]`` (``float64``).
        """
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must be [N, 4, 4]; got {poses.shape}")
        q = np.asarray(q_init, dtype=np.float64).copy()
        out = []
        for pose in poses:
            q = self.solve(pose, q)
            out.append(q.copy())
        return np.stack(out) if out else np.empty((0, self.model.nq), dtype=np.float64)

    def tracking_error(self, poses: np.ndarray, qpos_traj: np.ndarray) -> dict[str, float]:
        """Cartesian position tracking error between targets and solved poses.

        Args:
            poses: Target EE poses ``[N, 4, 4]``.
            qpos_traj: Solved joint configs ``[N, nq]`` (from
                :meth:`solve_trajectory`).

        Returns:
            ``{"mean_mm": float, "max_mm": float}`` - mean / max Euclidean
            position error in millimetres across the trajectory.
        """
        poses = np.asarray(poses, dtype=np.float32)
        errs = []
        for target, q in zip(poses, np.asarray(qpos_traj), strict=True):
            achieved = self.ee_pose(q)
            errs.append(float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])))
        errs_arr = np.asarray(errs, dtype=np.float32)
        if errs_arr.size == 0:
            return {"mean_mm": 0.0, "max_mm": 0.0}
        return {"mean_mm": float(errs_arr.mean() * 1000.0), "max_mm": float(errs_arr.max() * 1000.0)}


def decode_cosmos_chunk_to_targets(
    action_chunk: np.ndarray,
    embodiment: Cosmos3Embodiment,
    ik_bridge: MinkIKBridge,
    q_init: np.ndarray,
    *,
    stats: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Turn a Cosmos 3 raw action chunk into MuJoCo joint targets via IK.

    The full sim-loop bridge for the in-process ``diffusers`` backend, composing
    the three honest steps (no fabricated joint targets): de-normalize ->
    decode relative EE poses to an absolute trajectory -> inverse kinematics.

    1. **De-normalize** the model's ``[-1, 1]`` quantile-normalized action back
       to physical units with the embodiment's bundled ``q01``/``q99`` stats
       (:func:`~strands_robots.policies.cosmos3.action_decode.denormalize_quantile`).
    2. **Decode** the per-step ``[translation(3), rot6d(6)]`` pose block into an
       absolute ``(T+1, 4, 4)`` EE-pose trajectory anchored at the robot's
       current pose
       (:func:`~strands_robots.policies.cosmos3.action_decode.decode_pose_trajectory`).
    3. **Solve IK** for each Cartesian target via :class:`MinkIKBridge`, warm-
       starting from ``q_init`` so the joint trajectory stays continuous.

    Args:
        action_chunk: Raw unified action ``[T, raw_action_dim]`` from the
            diffusers backend (normalized ``[-1, 1]``; last column is grasp for
            gripper embodiments).
        embodiment: Active :class:`Cosmos3Embodiment` (provides ``domain_name``
            for the stats lookup, ``raw_action_layout`` for the gripper column,
            and ``normalization``).
        ik_bridge: A :class:`MinkIKBridge` over the target arm's MuJoCo model.
        q_init: Seed joint configuration (length ``model.nq``) - the robot's
            current pose; the EE trajectory is anchored at its forward
            kinematics and each IK solve warm-starts from the previous step.
        stats: Optional explicit ``{"q01", "q99"}`` stats override. When ``None``
            the bundled per-domain stats are loaded for ``embodiment.domain_name``.

    Returns:
        ``{"qpos": np.ndarray[T, nq], "gripper": np.ndarray[T] | None,
        "poses": np.ndarray[T, 4, 4], "tracking_error": {"mean_mm", "max_mm"}}``.
        ``gripper`` is ``None`` for grasp-less embodiments.

    Raises:
        ValueError: If ``embodiment.normalization`` is not ``"quantile"`` (the
            only method the current Cosmos 3 domains and bundled stats use).
    """
    from .action_decode import decode_pose_trajectory, denormalize_quantile, load_action_stats

    if embodiment.normalization != "quantile":
        raise ValueError(
            f"decode_cosmos_chunk_to_targets supports normalization='quantile' "
            f"(the bundled Cosmos 3 stats), not {embodiment.normalization!r}."
        )
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must be [T, D]; got {action_chunk.shape}")

    if stats is None:
        stats = load_action_stats(embodiment.domain_name)
    denorm = denormalize_quantile(action_chunk, stats["q01"], stats["q99"])

    # Split off a trailing grasp/gripper column when the layout has one.
    layout = embodiment.raw_action_layout
    has_grasp = bool(layout) and layout[-1] == "grasp"
    pose_block = denorm[:, :-1] if has_grasp else denorm
    gripper = denorm[:, -1] if has_grasp else None

    q0 = np.asarray(q_init, dtype=np.float64)
    initial_pose = ik_bridge.ee_pose(q0).astype(np.float64)
    abs_poses = decode_pose_trajectory(pose_block, initial_pose, rotation_dim=6)
    target_poses = abs_poses[1:]  # drop the anchor frame
    qpos = ik_bridge.solve_trajectory(target_poses, q0)
    return {
        "qpos": qpos,
        "gripper": gripper,
        "poses": target_poses,
        "tracking_error": ik_bridge.tracking_error(target_poses, qpos),
    }
