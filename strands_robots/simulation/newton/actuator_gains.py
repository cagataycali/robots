"""MJCF position-servo gains Newton's MJCF importer does not carry over.

An MJCF ``<position>`` actuator states its servo two ways. ``kp`` is a plain
attribute; the damping is usually written as ``dampratio``, which the MuJoCo
compiler expands into a velocity gain ``kv`` (stored as ``-kv`` in
``actuator_biasprm[:, 2]``) scaled by the joint's own effective inertia, and the
torque ceiling is written as ``forcerange``. Newton's importer reads ``kp`` into
``joint_target_ke`` and stops there: a ``dampratio`` servo arrives with
``joint_target_kd == 0`` and ``joint_effort_limit == 1e6``.

A P-only servo with no torque ceiling does not track. On the shipped SO-100
(``<position kp="50" dampratio="1" forcerange="-3.5 3.5"/>``) a constant
``Rotation = 0.5`` command oscillates between 0.05 and 0.96 rad indefinitely on
Newton while MuJoCo settles on 0.4997, so the same command means two different
motions on the two backends the engine contract promises are interchangeable.

MuJoCo is already a dependency of this backend (the ``[sim-newton]`` extra
narrows it), so the gains are read from the compiled model rather than
re-derived from the XML: the compiler is the authority on what ``dampratio``
means for a given joint inertia.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class JointServo(NamedTuple):
    """The position-servo gains MuJoCo compiled for one joint.

    Attributes:
        kp: Position gain (``actuator_gainprm[0]``).
        kd: Velocity gain, ``-actuator_biasprm[2]``. ``0.0`` when the actuator
            declares neither ``kv`` nor ``dampratio``.
        effort_limit: Torque ceiling from ``forcerange``, or ``None`` when the
            actuator is not force-limited.
    """

    kp: float
    kd: float
    effort_limit: float | None


def mjcf_joint_servos(model_path: str) -> dict[str, JointServo]:
    """Read the compiled position-servo gains of every joint-driving actuator.

    Args:
        model_path: Path to an MJCF file.

    Returns:
        ``{joint name: JointServo}`` for each actuator whose transmission is a
        single joint. Actuators on a tendon, site or body are skipped - they
        drive no one joint, so there is no DOF to write their gains onto.

    Raises:
        ImportError: MuJoCo is not installed.
        ValueError: MuJoCo cannot compile ``model_path``.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    servos: dict[str, JointServo] = {}
    for actuator in range(model.nu):
        if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not name:
            continue
        # A position actuator's bias is [0, -kp, -kv]; a plain motor has no
        # bias at all, which reads as kd == 0 - correct, it has no servo.
        kd = -float(model.actuator_biasprm[actuator, 2])
        limit: float | None = None
        if int(model.actuator_forcelimited[actuator]):
            low, high = (float(bound) for bound in model.actuator_forcerange[actuator])
            # The usable symmetric magnitude: a servo pushing either way is
            # capped by the tighter side.
            limit = min(abs(low), abs(high))
        servos[name] = JointServo(
            kp=float(model.actuator_gainprm[actuator, 0]),
            kd=kd if kd > 0.0 else 0.0,
            effort_limit=limit,
        )
    return servos


def apply_joint_servos(
    builder: Any,
    servos: dict[str, JointServo],
    joint_labels: list[str],
    first_joint: int,
    short_name: Any,
) -> dict[str, JointServo]:
    """Write a servo's velocity gain and torque ceiling onto builder DOFs.

    Only the two values Newton's importer drops are written. ``kp`` is left
    alone: the importer already carries it into ``joint_target_ke``, and
    overwriting it would silently substitute this reader for Newton's own.

    Args:
        builder: Newton ``ModelBuilder`` holding the just-imported robot.
        servos: Gains keyed by MJCF joint name (:func:`mjcf_joint_servos`).
        joint_labels: ``builder.joint_label`` after the import.
        first_joint: Index of this robot's first joint in ``joint_labels``.
        short_name: Callable reducing a Newton joint label to its short name.

    Returns:
        The servos actually written, keyed by short joint name.
    """
    applied: dict[str, JointServo] = {}
    for joint in range(first_joint, len(joint_labels)):
        servo = servos.get(short_name(joint_labels[joint]))
        if servo is None:
            continue
        dof = int(builder.joint_qd_start[joint])
        if dof >= len(builder.joint_target_kd):
            continue
        if servo.kd > 0.0:
            builder.joint_target_kd[dof] = servo.kd
        if servo.effort_limit is not None:
            builder.joint_effort_limit[dof] = servo.effort_limit
        applied[short_name(joint_labels[joint])] = servo
    return applied
