"""Site-actuated free bodies - the aerial robots - on the Isaac backend.

A MuJoCo quadrotor (``crazyflie``, ``skydio_x2``) has no joints besides its free
base: it is one rigid body, and its actuators are ``<motor site=...>`` thrusters
whose ``gear`` is a 6-D wrench in the site frame. PhysX builds no articulation
from a jointless body, so the articulation path cannot load it; it is driven here
as a rigid body instead, and each tick every motor contributes

    force  = R_site @ gear[:3] * ctrl      (applied at the site's world position)
    torque = R_site @ gear[3:] * ctrl

which is exactly MuJoCo's site transmission for a free body (see
``mj_actuator_moment``). The functions here are pure so that parity is testable
without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SiteDriveError(ValueError):
    """A site actuator this backend cannot drive as MuJoCo would.

    The rigid-body path applies ``force = gear * ctrl`` per motor and nothing
    else. A site actuator with a gain other than 1, a bias (a ``<position>`` or
    ``<velocity>`` servo on a site) or activation dynamics (``dyntype``, the
    rotor-lag model) has a different force law in MuJoCo; flying it with the
    motor law would diverge from MuJoCo with every value valid and nothing
    refusing, so it is refused here by name instead.
    """


@dataclass(frozen=True)
class SiteActuator:
    """One ``<motor site=...>`` of a free body, in that body's frame."""

    name: str
    site_pos: tuple[float, float, float]
    site_quat: tuple[float, float, float, float]  # wxyz, site frame in body frame
    gear: tuple[float, float, float, float, float, float]
    ctrlrange: tuple[float, float] | None


@dataclass
class SiteDrive:
    """What a site-actuated robot needs at run time: its body and its motors."""

    body_name: str
    actuators: list[SiteActuator]
    ctrl: dict[str, float] = field(default_factory=dict)
    body_prim_path: str | None = None
    body_int: int | None = None
    handle: object | None = None

    def set_ctrl(self, name: str, value: float) -> float:
        """Set motor *name*'s control, clipped to its ``ctrlrange`` as MuJoCo clips it; returns the value set."""
        act = next(a for a in self.actuators if a.name == name)
        v = float(value)
        if act.ctrlrange is not None:
            v = float(np.clip(v, act.ctrlrange[0], act.ctrlrange[1]))
        self.ctrl[name] = v
        return v


def mjcf_site_drive(mjcf_path: str) -> SiteDrive | None:
    """The :class:`SiteDrive` for *mjcf_path*, or None when it is an articulated robot.

    A model qualifies when it has no hinge or slide joint, at least one actuator,
    and every actuator is a site transmission on one and the same free body.
    Anything else - a joint, a tendon or joint actuator, motors on two bodies -
    is left to the articulation path, which reports on it itself.

    Args:
        mjcf_path: The MJCF file the robot was loaded from.

    Returns:
        The drive, or None when the model is not a site-actuated free body.

    Raises:
        SiteDriveError: If a site actuator is not the plain ``<motor>`` shape
            (``force = gear * ctrl``): a servo, a scaled gain or activation
            dynamics would fly differently from MuJoCo with no signal.
    """
    try:
        import mujoco
    except ImportError:
        return None
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    except Exception:  # noqa: BLE001 - not ours to report; the load that follows names it
        return None
    if model.nu == 0:
        return None
    types = set(int(t) for t in model.jnt_type)
    if types - {int(mujoco.mjtJoint.mjJNT_FREE)}:
        return None
    if any(int(t) != int(mujoco.mjtTrn.mjTRN_SITE) for t in model.actuator_trntype):
        return None
    bodies = {int(model.site_bodyid[int(model.actuator_trnid[i, 0])]) for i in range(model.nu)}
    if len(bodies) != 1:
        return None
    body = bodies.pop()
    if not any(int(model.jnt_bodyid[j]) == body for j in range(model.njnt)):
        return None  # a welded body cannot fly; nothing to drive
    actuators = []
    for i in range(model.nu):
        _refuse_unless_motor_law(model, i)
        site = int(model.actuator_trnid[i, 0])
        limited = bool(model.actuator_ctrllimited[i])
        actuators.append(
            SiteActuator(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}",
                site_pos=tuple(float(v) for v in model.site_pos[site]),  # type: ignore[arg-type]
                site_quat=tuple(float(v) for v in model.site_quat[site]),  # type: ignore[arg-type]
                gear=tuple(float(v) for v in model.actuator_gear[i]),  # type: ignore[arg-type]
                ctrlrange=(float(model.actuator_ctrlrange[i, 0]), float(model.actuator_ctrlrange[i, 1]))
                if limited
                else None,
            )
        )
    return SiteDrive(body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "", actuators=actuators)


def _refuse_unless_motor_law(model: object, i: int) -> None:
    """Raise :class:`SiteDriveError` unless actuator *i* is the plain ``<motor>`` shape.

    MuJoCo's ``<motor>`` is ``gaintype=fixed`` with ``gainprm[0] == 1``,
    ``biastype=none`` and ``dyntype=none``: its force is exactly ``gear * ctrl``,
    which is what :func:`site_wrenches` applies. The site transmission alone does
    not say so - a ``<position site=...>`` servo or a ``<general dyntype="filter">``
    rotor is also a site transmission - so each of the three is checked.
    """
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
    problems: list[str] = []
    dyntype = int(model.actuator_dyntype[i])  # type: ignore[attr-defined]
    if dyntype != int(mujoco.mjtDyn.mjDYN_NONE):
        problems.append(f"dyntype={mujoco.mjtDyn(dyntype).name} (MuJoCo integrates an activation state)")
    gaintype = int(model.actuator_gaintype[i])  # type: ignore[attr-defined]
    gain = float(model.actuator_gainprm[i, 0])  # type: ignore[attr-defined]
    if gaintype != int(mujoco.mjtGain.mjGAIN_FIXED):
        problems.append(f"gaintype={mujoco.mjtGain(gaintype).name}")
    elif gain != 1.0:
        problems.append(f"gainprm[0]={gain:g} (MuJoCo scales ctrl by it)")
    biastype = int(model.actuator_biastype[i])  # type: ignore[attr-defined]
    if biastype != int(mujoco.mjtBias.mjBIAS_NONE):
        problems.append(f"biastype={mujoco.mjtBias(biastype).name} (a servo: MuJoCo adds a bias term)")
    if problems:
        raise SiteDriveError(
            f"site actuator {name!r} is not a plain <motor>: {'; '.join(problems)}. The Isaac rigid-body path "
            "applies force = gear * ctrl per motor and nothing else, so this robot would fly differently from "
            "MuJoCo with no error. Model it as <motor site=...> (fixed unit gain, no bias, no dynamics), or "
            "load it on MuJoCo."
        )


def _quat_to_matrix(q: tuple[float, ...] | np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def site_wrenches(
    drive: SiteDrive, body_pos: np.ndarray, body_quat_wxyz: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(force, torque, point)`` in world frame for every motor with a non-zero control."""
    r_body = _quat_to_matrix(body_quat_wxyz)
    pos = np.asarray(body_pos, dtype=float).reshape(3)
    out = []
    for act in drive.actuators:
        ctrl = drive.ctrl.get(act.name, 0.0)
        if ctrl == 0.0:
            continue
        r_site = r_body @ _quat_to_matrix(act.site_quat)
        gear = np.asarray(act.gear, dtype=float)
        out.append(
            (
                r_site @ (gear[:3] * ctrl),
                r_site @ (gear[3:] * ctrl),
                pos + r_body @ np.asarray(act.site_pos, dtype=float),
            )
        )
    return out
