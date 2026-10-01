"""Close an Isaac Lab policy's PD loop on MuJoCo torque motors, every physics step.

An Isaac Lab ``JointPositionAction`` produces joint POSITION targets, and the
run's actuator model turns them into torque on every physics step - for the
Go2, a ``DCMotor`` with stiffness 25 N m/rad, damping 0.5 N m s/rad and a
23.7 / 45.43 N m limit, stepped four times per 50 Hz policy step. strands'
MuJoCo Go2 (the menagerie model) has TORQUE motors: a position target sent to
them is read as newton-metres. With the deploy contract applied but no PD the
Go2 still fell on its side; with Isaac Lab's PD it stood.

:class:`ContractPDController` is that loop, installed through the
``world._backend_state["action_controller"]`` seam the WBC torque shim uses:
``tau = kp * (target - q) - kd * qd``, clipped to the effort limit and the
actuator's ``ctrlrange``, recomputed every ``mj_step`` for one control period
(``control_dt`` / timestep physics steps). It never changes an actuator's gains,
so :meth:`ContractPDController.uninstall` only drops the registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _is_torque_motor(model: Any, actuator_id: int, mujoco: Any) -> bool:
    """A ``<motor>``: direct gain, no bias - ``ctrl`` is the joint torque."""
    return bool(
        model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_NONE
        and model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
    )


class ContractPDController:
    """Isaac Lab's joint PD, run on MuJoCo torque motors for one robot.

    Build with :meth:`from_sim`. ``apply(action_dict, model, data, robot_name)``
    reads each action as a joint position target and advances physics itself
    for one control period (``owns_stepping``).
    """

    owns_stepping: bool = True

    def __init__(
        self,
        *,
        joints: dict[str, tuple[int, int, int, float, float, float]],
        substeps: int,
        world: Any = None,
    ) -> None:
        """Bind the controller to one robot's actuators.

        Args:
            joints: ``{action_key: (actuator_id, qpos_adr, dof_adr, kp, kd, effort_limit)}``.
            substeps: Physics steps per applied action (the run's control period).
            world: The world whose ``_backend_state`` registered this controller.
        """
        self._joints = joints
        self._substeps = max(1, int(substeps))
        self._world = world
        self._targets: dict[str, float] = {}

    @classmethod
    def from_sim(
        cls, sim: Any, robot_name: str, contract: Mapping[str, Any], binding: Mapping[str, str]
    ) -> ContractPDController | None:
        """Resolve the robot's actuators for *contract*'s joints; ``None`` when this loop does not apply.

        It applies when every bound actuator is a torque motor and the contract
        records a stiffness for its joint. Position-servo actuators already
        close a loop of their own, so they are left alone.

        Args:
            sim: The MuJoCo engine.
            robot_name: The robot being driven.
            contract: The policy's deploy contract (``actuators``, ``control_dt``).
            binding: ``{contract_joint: robot_action_key}``.
        """
        import mujoco

        world = sim._world
        model = world._model
        robot = world.robots.get(robot_name) if hasattr(world.robots, "get") else None
        prefix = getattr(robot, "namespace", "") or ""
        actuators = contract.get("actuators") or {}
        joints: dict[str, tuple[int, int, int, float, float, float]] = {}
        for joint, key in binding.items():
            spec = actuators.get(joint) or {}
            if spec.get("stiffness") is None:
                return None
            aid = -1
            for name in (prefix + key, key):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    break
            if aid < 0 or not _is_torque_motor(model, aid, mujoco):
                return None
            jid = int(model.actuator_trnid[aid, 0])
            effort = spec.get("effort_limit")
            joints[key] = (
                aid,
                int(model.jnt_qposadr[jid]),
                int(model.jnt_dofadr[jid]),
                float(spec["stiffness"]),
                float(spec.get("damping") or 0.0),
                float(effort) if effort is not None else float("inf"),
            )
        control_dt = contract.get("control_dt")
        substeps = round(float(control_dt) / float(model.opt.timestep)) if control_dt else 1
        return cls(joints=joints, substeps=substeps, world=world)

    def apply(self, action_dict: Mapping[str, Any], model: Any, data: Any, robot_name: str) -> None:
        """Hold the action's targets through one control period of PD-controlled physics."""
        import mujoco

        del robot_name
        for key, value in action_dict.items():
            if key in self._joints:
                self._targets[key] = float(value)
        for _ in range(self._substeps):
            for key, target in self._targets.items():
                aid, qadr, dadr, kp, kd, limit = self._joints[key]
                tau = kp * (target - float(data.qpos[qadr])) - kd * float(data.qvel[dadr])
                tau = max(-limit, min(limit, tau))
                if model.actuator_ctrllimited[aid]:
                    low, high = model.actuator_ctrlrange[aid]
                    tau = max(float(low), min(float(high), tau))
                data.ctrl[aid] = tau
            mujoco.mj_step(model, data)

    def uninstall(self) -> None:
        """Drop this controller's registration (no gain was changed)."""
        state = getattr(self._world, "_backend_state", None) if self._world is not None else None
        if isinstance(state, dict) and state.get("action_controller") is self:
            del state["action_controller"]


__all__ = ["ContractPDController"]
