"""MuJoCo torque shim for :class:`WBCLatentPolicy`: SONIC's PD law on all 29 G1 joints.

The stock Menagerie Unitree G1 scene drives every joint with a uniform
position servo (``kp = 500``). The SONIC decoder was trained against the
armature-derived per-joint gains in
:data:`~strands_robots.policies.wbc_latent.constants.SONIC_KPS` /
:data:`~strands_robots.policies.wbc_latent.constants.SONIC_KDS` (14 to 99 N m/rad),
so writing its joint targets straight to ``data.ctrl`` would track them with
gains 5x to 35x stiffer than the ones the network expects and the robot falls.
The shim mirrors :class:`~strands_robots.policies.wbc.sim_control.WBCTorqueController`:
flip the actuators to torque mode, compute ``tau = kp * (target - q) - kd * dq``
every 0.005 s physics substep, four substeps per 50 Hz control tick
(``owns_stepping = True``), and restore the actuators on uninstall. Unlike the
locomotion shim it drives all 29 joints from one gain table; there is no
"arm hold" branch because the decoder commands the arms too.

``MuJoCoSimEngine._maybe_install_action_controller`` installs this shim when a
``WBCLatentPolicy`` is anywhere in the policy tree and the driven actuators are
position servos, so every rollout surface (``run_policy``, ``eval_policy``,
``evaluate_benchmark``) gets the same physics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.utils import positive_whole_number_error

from .constants import NUM_JOINTS, SONIC_BASE_HEIGHT, SONIC_DEFAULT_ANGLES, SONIC_JOINT_NAMES, SONIC_KDS, SONIC_KPS
from .policy import GRIPPER_KEYS, WBCLatentPolicy

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

#: Training configuration of the decoder (low_latency/config.yaml: sim_dt 0.005, decimation 4).
_SIM_DT = 0.005
_CONTROL_DECIMATION = 4


class WBCLatentTorqueController:
    """Track the decoder's 29 joint targets with SONIC's per-joint PD torques.

    Attributes:
        owns_stepping: ``True`` - :meth:`apply` advances physics itself.
        target_q: The targets currently tracked (hardware order), updated by :meth:`apply`.
        last_tau: Torques of the most recent substep (hardware order).
    """

    owns_stepping = True

    def __init__(
        self,
        policy: WBCLatentPolicy,
        *,
        actuator_ids: Sequence[int],
        qpos_addrs: Sequence[int],
        dof_addrs: Sequence[int],
        saved_actuator_gains: dict[int, tuple[Any, Any, Any, Any, Any]],
        model: Any,
        world: Any,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
        physics_substeps_per_control: int = _CONTROL_DECIMATION,
    ) -> None:
        """Bind the controller to resolved actuators; use :meth:`from_sim` to build one.

        Raises:
            ValueError: the three index lists are not all 29 long, or
                ``physics_substeps_per_control`` is not a positive whole number.
        """
        if not (len(actuator_ids) == len(qpos_addrs) == len(dof_addrs) == NUM_JOINTS):
            raise ValueError(
                f"WBCLatentTorqueController: expected {NUM_JOINTS} actuators/qpos/dof addresses, got "
                f"{len(actuator_ids)}/{len(qpos_addrs)}/{len(dof_addrs)}"
            )
        if error := positive_whole_number_error(
            physics_substeps_per_control, "physics_substeps_per_control", "WBCLatentTorqueController"
        ):
            raise ValueError(error)
        self.policy = policy
        self.actuator_ids = list(actuator_ids)
        self.qpos_addrs = np.asarray(qpos_addrs, dtype=int)
        self.dof_addrs = np.asarray(dof_addrs, dtype=int)
        self._saved_actuator_gains = dict(saved_actuator_gains)
        self._model = model
        self._world = world
        self.kps = np.asarray(SONIC_KPS if kps is None else kps, dtype=np.float64)
        self.kds = np.asarray(SONIC_KDS if kds is None else kds, dtype=np.float64)
        self.physics_substeps_per_control = int(physics_substeps_per_control)
        self.target_q = np.asarray(SONIC_DEFAULT_ANGLES, dtype=np.float64).copy()
        self.last_tau = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.last_grippers: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # install / uninstall
    # ------------------------------------------------------------------

    @classmethod
    def from_sim(cls, sim: SimEngine, policy: WBCLatentPolicy, robot_name: str) -> WBCLatentTorqueController:
        """Resolve the 29 joints of ``robot_name``, flip their actuators to torque, seed the stance.

        Raises:
            RuntimeError: no compiled world, or a SONIC joint / its actuator is
                missing from the model (the scene is not a 29-joint G1).
        """
        import mujoco as mj

        world = getattr(sim, "_world", None)
        if world is None or getattr(world, "_model", None) is None:
            raise RuntimeError("WBCLatentTorqueController.from_sim: no compiled world/model on the sim.")
        model = world._model
        robot = world.robots.get(robot_name) if getattr(world, "robots", None) else None
        pfx = robot.namespace if robot is not None else ""

        act_ids: list[int] = []
        qpos_addrs: list[int] = []
        dof_addrs: list[int] = []
        for name in SONIC_JOINT_NAMES:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + name)
            if jid < 0:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(
                    f"WBCLatentTorqueController: joint {name!r} not found in the model (looked for "
                    f"{pfx + name!r} and {name!r}); the SONIC decoder drives the 29-joint Unitree G1 only."
                )
            ai = next((a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == jid), -1)
            if ai < 0:
                raise RuntimeError(f"WBCLatentTorqueController: joint {name!r} (id {jid}) has no driving actuator.")
            act_ids.append(int(ai))
            qpos_addrs.append(int(model.jnt_qposadr[jid]))
            dof_addrs.append(int(model.jnt_dofadr[jid]))

        saved: dict[int, tuple[Any, Any, Any, Any, Any]] = {}
        for ai in act_ids:
            saved[ai] = (
                int(model.actuator_gaintype[ai]),
                int(model.actuator_biastype[ai]),
                np.array(model.actuator_gainprm[ai], copy=True),
                np.array(model.actuator_biasprm[ai], copy=True),
                np.array(model.actuator_ctrlrange[ai], copy=True),
            )
            model.actuator_gaintype[ai] = mj.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[ai] = mj.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[ai][:3] = [1.0, 0.0, 0.0]
            model.actuator_biasprm[ai][:3] = [0.0, 0.0, 0.0]
            model.actuator_ctrlrange[ai] = [-1000.0, 1000.0]
        model.opt.timestep = _SIM_DT

        data = world._data
        for adr, angle in zip(qpos_addrs, SONIC_DEFAULT_ANGLES, strict=True):
            data.qpos[adr] = float(angle)
        free_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + "floating_base_joint")
        if free_jid < 0:
            free_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if free_jid >= 0 and int(model.jnt_type[free_jid]) == int(mj.mjtJoint.mjJNT_FREE):
            base_adr = int(model.jnt_qposadr[free_jid])
            data.qpos[base_adr + 2] = SONIC_BASE_HEIGHT
            data.qpos[base_adr + 3 : base_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        policy.decoder.reset()
        logger.info(
            "WBCLatentTorqueController installed on %r: %d actuators -> torque mode, dt=%.4f, decim=%d",
            robot_name,
            len(act_ids),
            _SIM_DT,
            _CONTROL_DECIMATION,
        )
        return cls(
            policy,
            actuator_ids=act_ids,
            qpos_addrs=qpos_addrs,
            dof_addrs=dof_addrs,
            saved_actuator_gains=saved,
            model=model,
            world=world,
            kps=policy.decoder.kps,
            kds=policy.decoder.kds,
        )

    def uninstall(self) -> None:
        """Drop this controller's registration first, then restore the actuator gains.

        Same order and reasoning as
        :meth:`~strands_robots.policies.wbc.sim_control.WBCTorqueController.uninstall`:
        a registration left behind is read as a manual install that wins, and
        would dispatch PD torques into actuators that are servos again.
        """
        backend_state = getattr(self._world, "_backend_state", None) if self._world is not None else None
        if isinstance(backend_state, dict) and backend_state.get("action_controller") is self:
            del backend_state["action_controller"]
        model = self._model
        for ai, (gaintype, biastype, gainprm, biasprm, ctrlrange) in self._saved_actuator_gains.items():
            model.actuator_gaintype[ai] = gaintype
            model.actuator_biastype[ai] = biastype
            model.actuator_gainprm[ai] = gainprm
            model.actuator_biasprm[ai] = biasprm
            model.actuator_ctrlrange[ai] = ctrlrange

    # ------------------------------------------------------------------
    # per-tick
    # ------------------------------------------------------------------

    def compute_torques(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """``tau = kp * (target_q - q) - kd * dq`` on the current targets (hardware order)."""
        return self.kps * (self.target_q - np.asarray(q, dtype=np.float64)) - self.kds * np.asarray(
            dq, dtype=np.float64
        )

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,  # noqa: ARG002 - hook signature parity
    ) -> None:
        """Refresh the targets from ``action_dict`` and advance physics one control tick.

        A joint the dict omits or gives a non-number keeps its previous target
        (a hold, not an abort). The gripper keys are recorded on
        :attr:`last_grippers`; the Menagerie G1 has no gripper actuators.
        """
        import mujoco as mj

        for i, name in enumerate(SONIC_JOINT_NAMES):
            v = action_dict.get(name)
            if v is None:
                continue
            try:
                self.target_q[i] = float(v)
            except (TypeError, ValueError):
                continue
        try:
            self.last_grippers = (
                float(action_dict.get(GRIPPER_KEYS[0], self.last_grippers[0])),
                float(action_dict.get(GRIPPER_KEYS[1], self.last_grippers[1])),
            )
        except (TypeError, ValueError):
            # An unusable gripper value keeps the last commanded one; the arm
            # joints above were already filtered the same way.
            pass
        for _ in range(self.physics_substeps_per_control):
            q = data.qpos[self.qpos_addrs]
            dq = data.qvel[self.dof_addrs]
            tau = self.compute_torques(q, dq)
            for ai, t in zip(self.actuator_ids, tau, strict=True):
                data.ctrl[ai] = float(t)
            self.last_tau = tau
            mj.mj_step(model, data)


def wbc_latent_uses_position_servo(sim: SimEngine, robot_name: str) -> bool:
    """True when at least one SONIC joint of ``robot_name`` is driven by a position servo.

    Mirrors :func:`~strands_robots.policies.wbc.sim_control.wbc_uses_position_servo`:
    ``False`` when the world is absent, the joints do not resolve, or the
    actuators are already torque motors (nothing to install).
    """
    import mujoco as mj

    world = getattr(sim, "_world", None)
    if world is None or getattr(world, "_model", None) is None:
        return False
    model = world._model
    robot = world.robots.get(robot_name) if getattr(world, "robots", None) else None
    pfx = robot.namespace if robot is not None else ""
    for name in SONIC_JOINT_NAMES:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + name)
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        for ai in range(model.nu):
            if int(model.actuator_trnid[ai, 0]) == jid:
                if int(model.actuator_biastype[ai]) == int(mj.mjtBias.mjBIAS_AFFINE):
                    return True
                break
    return False


def install_wbc_latent_torque_control(
    sim: SimEngine, policy: WBCLatentPolicy, robot_name: str
) -> WBCLatentTorqueController:
    """Build the controller and register it in ``world._backend_state["action_controller"]``.

    Raises:
        RuntimeError: the world is absent, has no ``_backend_state`` dict, or
            the joints cannot be resolved.
    """
    controller = WBCLatentTorqueController.from_sim(sim, policy, robot_name)
    world = getattr(sim, "_world", None)
    if world is None:
        raise RuntimeError("install_wbc_latent_torque_control: no world on the sim.")
    backend_state = getattr(world, "_backend_state", None)
    if not isinstance(backend_state, dict):
        raise RuntimeError("install_wbc_latent_torque_control: world has no _backend_state dict.")
    backend_state["action_controller"] = controller
    return controller


__all__ = ["WBCLatentTorqueController", "install_wbc_latent_torque_control", "wbc_latent_uses_position_servo"]
