"""MuJoCo model and data as Foxglove 3D messages: ``/tf`` transforms and a mesh scene.

Every registry robot is a compiled MuJoCo model, so its 3D view can come from
the model itself rather than from a URDF: each mesh geom becomes a
``TriangleList`` in its body's frame, published once and ``frame_locked``, and
every body pose becomes one ``FrameTransform`` from ``world`` per step. That is
the spike's route and it needs no asset handler and no ``package://`` scheme.

Two conventions to hold on to:

* MuJoCo quaternions are ``w, x, y, z``; Foxglove's are ``x, y, z, w``.
* Collision meshes sit in geom group 3 by Menagerie convention and every
  visual mesh in groups 0-2, so :func:`scene_update` keeps ``geom_group <= 2``
  and the arm is drawn once rather than twice.

``foxglove`` is imported inside the functions: this module is reachable from an
install without the ``[foxglove]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Highest geom group treated as visual; group 3 is the collision layer.
VISUAL_GROUP_MAX = 2

#: The one frame every transform hangs off.
WORLD_FRAME = "world"

_PALETTE = ((0.22, 0.22, 0.24), (0.00, 0.48, 0.24), (0.55, 0.55, 0.58))


def body_name(model: Any, body_id: int) -> str:
    """The body's MJCF name, or ``body_<id>`` for an unnamed body."""
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def robot_of_body(name: str) -> str | None:
    """The robot namespace of a body name (``so101/base`` -> ``so101``), or ``None`` for the world."""
    head, sep, _ = name.partition("/")
    return head if sep else None


def _timestamp(stamp_ns: int) -> Any:
    from foxglove.messages import Timestamp

    return Timestamp(sec=stamp_ns // 1_000_000_000, nsec=stamp_ns % 1_000_000_000)


def frame_transforms(model: Any, data: Any, stamp_ns: int) -> Any:
    """Every body pose as a ``FrameTransforms`` message, parent ``world``.

    Args:
        model: The live ``mujoco.MjModel``.
        data: The matching ``mujoco.MjData`` after ``mj_forward``.
        stamp_ns: Wall-clock nanoseconds for the message timestamps.

    Returns:
        A ``foxglove.messages.FrameTransforms`` with ``nbody - 1`` transforms.
    """
    from foxglove.messages import FrameTransform, FrameTransforms, Quaternion, Vector3

    stamp = _timestamp(stamp_ns)
    transforms = []
    for b in range(1, model.nbody):
        p = data.xpos[b]
        q = data.xquat[b]
        transforms.append(
            FrameTransform(
                timestamp=stamp,
                parent_frame_id=WORLD_FRAME,
                child_frame_id=body_name(model, b),
                translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                rotation=Quaternion(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0])),
            )
        )
    return FrameTransforms(transforms=transforms)


def _geom_color(model: Any, geom_id: int, body_id: int) -> tuple[float, float, float, float]:
    rgba = model.geom_rgba[geom_id]
    if rgba[3] <= 0.0 or bool(np.all(rgba[:3] == 0.5)):
        r, g, b = _PALETTE[body_id % len(_PALETTE)]
        return r, g, b, 1.0
    return float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])


def _mesh_triangles(model: Any, geom_id: int, body_id: int) -> Any:
    from foxglove.messages import Color, Point3, Pose, Quaternion, TriangleListPrimitive, Vector3

    mesh_id = int(model.geom_dataid[geom_id])
    v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    verts = model.mesh_vert[v0 : v0 + nv]
    faces = model.mesh_face[f0 : f0 + nf].reshape(-1)
    pos = model.geom_pos[geom_id]
    quat = model.geom_quat[geom_id]
    r, g, b, a = _geom_color(model, geom_id, body_id)
    return TriangleListPrimitive(
        pose=Pose(
            position=Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(x=float(quat[1]), y=float(quat[2]), z=float(quat[3]), w=float(quat[0])),
        ),
        points=[Point3(x=float(x), y=float(y), z=float(z)) for x, y, z in verts.tolist()],
        indices=[int(i) for i in faces.tolist()],
        color=Color(r=r, g=g, b=b, a=a),
    )


def _primitive(model: Any, geom_id: int, body_id: int) -> tuple[str, Any] | None:
    """A box, sphere, cylinder or capsule geom as its Foxglove primitive, or ``None``."""
    import mujoco
    from foxglove.messages import (
        Color,
        CubePrimitive,
        CylinderPrimitive,
        Pose,
        Quaternion,
        SpherePrimitive,
        Vector3,
    )

    size = model.geom_size[geom_id]
    pos = model.geom_pos[geom_id]
    quat = model.geom_quat[geom_id]
    r, g, b, a = _geom_color(model, geom_id, body_id)
    pose = Pose(
        position=Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=Quaternion(x=float(quat[1]), y=float(quat[2]), z=float(quat[3]), w=float(quat[0])),
    )
    color = Color(r=r, g=g, b=b, a=a)
    kind = int(model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return "cubes", CubePrimitive(
            pose=pose, size=Vector3(x=2 * float(size[0]), y=2 * float(size[1]), z=2 * float(size[2])), color=color
        )
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        d = 2 * float(size[0])
        return "spheres", SpherePrimitive(pose=pose, size=Vector3(x=d, y=d, z=d), color=color)
    if kind in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        d = 2 * float(size[0])
        return "cylinders", CylinderPrimitive(
            pose=pose,
            size=Vector3(x=d, y=d, z=2 * float(size[1])),
            bottom_scale=1.0,
            top_scale=1.0,
            color=color,
        )
    return None


def scene_update(model: Any, stamp_ns: int, *, robot: str | None = None) -> Any | None:
    """Every visual geom as a ``SceneUpdate`` of frame-locked entities, one per body.

    Args:
        model: The live ``mujoco.MjModel``.
        stamp_ns: Wall-clock nanoseconds for the entity timestamps.
        robot: When given, only bodies in that robot's namespace
            (``<robot>/...``); ``None`` selects the bodies of no robot (the
            floor, task objects, a cube an agent added).

    Returns:
        A ``foxglove.messages.SceneUpdate``, or ``None`` when no body of that
        selection carries a visual geom. Meshes become ``TriangleList``
        primitives; boxes, spheres, cylinders and capsules become their own
        primitives; planes and height fields are skipped (the 3D panel draws
        its own grid).
    """
    import mujoco
    from foxglove.messages import SceneEntity, SceneUpdate

    stamp = _timestamp(stamp_ns)
    entities = []
    for b in range(1, model.nbody):
        name = body_name(model, b)
        if robot_of_body(name) != robot:
            continue
        parts: dict[str, list[Any]] = {"triangles": [], "cubes": [], "spheres": [], "cylinders": []}
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != b or int(model.geom_group[g]) > VISUAL_GROUP_MAX:
                continue
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                parts["triangles"].append(_mesh_triangles(model, g, b))
                continue
            prim = _primitive(model, g, b)
            if prim is not None:
                parts[prim[0]].append(prim[1])
        if any(parts.values()):
            entities.append(
                SceneEntity(
                    timestamp=stamp,
                    frame_id=name,
                    id=name,
                    frame_locked=True,
                    triangles=parts["triangles"],
                    cubes=parts["cubes"],
                    spheres=parts["spheres"],
                    cylinders=parts["cylinders"],
                )
            )
    return SceneUpdate(entities=entities) if entities else None


def joint_states(model: Any, data: Any, joint_names: list[str], stamp_ns: int, *, robot: str) -> Any:
    """Positions and velocities of ``joint_names`` as a ``JointStates`` message.

    Joint names arrive un-namespaced (``robot_joint_names``) while the model
    carries ``<robot>/<joint>``; both spellings are tried.
    """
    import mujoco
    from foxglove.messages import JointState, JointStates

    states = []
    for joint in joint_names:
        jid = -1
        for candidate in (f"{robot}/{joint}", joint):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if jid >= 0:
                break
        if jid < 0:
            continue
        states.append(
            JointState(
                name=joint,
                position=float(data.qpos[model.jnt_qposadr[jid]]),
                velocity=float(data.qvel[model.jnt_dofadr[jid]]),
            )
        )
    return JointStates(timestamp=_timestamp(stamp_ns), joints=states)
