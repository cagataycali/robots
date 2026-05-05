"""MjSpec-based MJCF builder - programmatic scene construction via the MuJoCo AST.

Replaces the string-concat ``MJCFBuilder`` (see ``mjcf_builder.py``) with direct
manipulation of ``mujoco.MjSpec`` - the official editable MJCF AST shipped with
MuJoCo 3.2+. Benefits:

* No f-strings, no regex-validated names, no XML injection surface - MuJoCo's
  own compiler validates every element.
* No hand-rolled ``_camera_xyaxes_from_target`` - we convert target direction
  to a quaternion via MuJoCo's own ``mju_mat2Quat``.
* Live mutation via ``spec.recompile(model, data)`` instead of the tmpdir +
  ET.parse + mj_saveLastXML round-trip.

This module is opt-in behind the ``STRANDS_SIM_USE_MJSPEC`` env var (see
``Simulation._use_mjspec``). The legacy ``MJCFBuilder`` stays in place until
every downstream caller migrates.

See ``IDEA.md`` at repo root for the full staged refactor plan (GH #121 tracks
execution).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from strands_robots.simulation.models import SimCamera, SimObject, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


# MuJoCo geom-type enum mapping. Populated lazily on first call so module
# import doesn't require mujoco to be installed (mirrors mjcf_builder's
# pattern via _ensure_mujoco).
_GEOM_TYPE_CACHE: dict[str, int] | None = None


def _geom_type(shape: str) -> int:
    """Map our shape-name vocabulary to MuJoCo's ``mjtGeom`` enum.

    Raises ValueError for shapes unsupported by the current pipeline. New
    shapes (ellipsoid, hfield) can be added here without touching the rest
    of the builder.
    """
    global _GEOM_TYPE_CACHE
    if _GEOM_TYPE_CACHE is None:
        mujoco = _ensure_mujoco()
        _GEOM_TYPE_CACHE = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "mesh": mujoco.mjtGeom.mjGEOM_MESH,
            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
        }
    try:
        return _GEOM_TYPE_CACHE[shape]
    except KeyError as e:
        supported = ", ".join(sorted(_GEOM_TYPE_CACHE.keys()))
        raise ValueError(f"Unsupported shape {shape!r}. Supported: {supported}.") from e


def _normalize_size(shape: str, size: list[float]) -> list[float]:
    """Convert SimObject ``size`` convention to MuJoCo's per-type size vector.

    MuJoCo's geom-size conventions (all in the LOCAL frame):

    * ``box``:      half-extents ``[hx, hy, hz]``
    * ``sphere``:   ``[radius]``   (extra slots ignored)
    * ``cylinder``: ``[radius, half-height]``
    * ``capsule``:  ``[radius, half-height]``   (cap hemisphere radius = radius)
    * ``ellipsoid``:``[rx, ry, rz]``
    * ``plane``:    ``[hx, hy, grid_spacing]``   (hx/hy are half-sizes)
    * ``mesh``:     ``[]``   (mesh asset dictates extent; size is ignored)

    SimObject stores a single ``size: list[float]`` that we map per shape.
    This matches the existing ``MJCFBuilder._object_xml`` semantics byte-for-
    byte (box full-extents -> halved; sphere size[0] -> radius/2 via
    ``size[0]/2`` which matches the existing code path).
    """
    if shape == "box":
        sx, sy, sz = size if len(size) >= 3 else (0.1, 0.1, 0.1)
        return [sx / 2, sy / 2, sz / 2]
    if shape == "sphere":
        # Current MJCFBuilder uses size[0]/2 as radius -> we match exactly.
        radius = size[0] / 2 if size else 0.025
        # MuJoCo wants a 3-vec size for all geoms; only the first is used.
        return [radius, 0.0, 0.0]
    if shape in ("cylinder", "capsule"):
        radius = size[0] / 2 if size else 0.025
        half_h = size[2] / 2 if len(size) > 2 else 0.05
        return [radius, half_h, 0.0]
    if shape == "ellipsoid":
        sx, sy, sz = size if len(size) >= 3 else (0.05, 0.05, 0.05)
        return [sx / 2, sy / 2, sz / 2]
    if shape == "plane":
        sx = size[0] if size else 1.0
        sy = size[1] if len(size) > 1 else sx
        return [sx, sy, 0.01]
    if shape == "mesh":
        return [0.0, 0.0, 0.0]
    raise ValueError(f"Cannot normalize size for shape {shape!r}.")


def _target_quat(position: list[float], target: list[float]) -> list[float] | None:
    """Compute the camera orientation quaternion that makes ``position`` look
    at ``target`` with world +Z as the up vector.

    Uses the same camera convention as ``MJCFBuilder._camera_xyaxes_from_target``:

    * Forward (cam local -Z) = normalize(target - position)
    * Right   (cam local +X) = normalize(forward x up)
    * Image-up (cam local +Y) = normalize(right x forward)

    Then constructs a 3x3 rotation matrix whose columns are [right, image-up,
    -forward] (MuJoCo's camera Z axis points OUT of the scene, so world-frame
    column 3 is -forward), and converts it to a quaternion via MuJoCo's own
    ``mju_mat2Quat`` so no hand-rolled quaternion math is involved.

    Returns ``None`` on a degenerate case (target == position, or forward
    parallel to up). Callers should surface a clear error in that case -
    ``MJCFBuilder._camera_xyaxes_from_target`` has the same contract.
    """
    mujoco = _ensure_mujoco()

    fwd = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    flen = float(np.linalg.norm(fwd))
    if flen < 1e-9:
        return None
    fwd /= flen

    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    rlen = float(np.linalg.norm(right))
    if rlen < 1e-9:
        # forward parallel to up - caller should reject this upstream.
        return None
    right /= rlen
    image_up = np.cross(right, fwd)
    image_up /= float(np.linalg.norm(image_up))

    # Columns of R are [right, image_up, -forward] - the camera's +X, +Y, +Z
    # basis vectors expressed in world frame. MuJoCo expects row-major here.
    rot = np.column_stack([right, image_up, -fwd])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return quat.tolist()


class SpecBuilder:
    """Builds ``mujoco.MjSpec`` trees from ``SimWorld`` state.

    This is the MjSpec equivalent of ``MJCFBuilder``. Every public method
    returns the MjSpec (not a string), and callers are expected to call
    ``spec.compile()`` themselves - that gives them a handle to both the
    spec (for later mutation) and the compiled MjModel in one clean pass.
    """

    @staticmethod
    def build(world: SimWorld) -> Any:
        """Build a fresh ``mujoco.MjSpec`` that mirrors ``MJCFBuilder.build_objects_only``.

        Output equivalence target (post-compile, not XML-string):

        * Same ``nbody`` count (ground + main_light + fill_light + one body
          per object).
        * Same ``ngeom`` count (1 ground + 1 per object).
        * Same ``ncam`` count (one per world.cameras).
        * Same ``nu`` == 0 (objects have no actuators).
        * Same ``option.timestep`` and ``option.gravity``.

        Returns the **MjSpec**; caller should call ``spec.compile()`` to get
        an MjModel.
        """
        mujoco = _ensure_mujoco()

        spec = mujoco.MjSpec()
        spec.modelname = "strands_sim"

        # Global compiler + simulation options.
        # In MjSpec 3.8 these are plain attribute writes, not dict updates.
        spec.compiler.degree = False  # radians
        spec.compiler.autolimits = True

        spec.option.timestep = float(world.timestep)
        spec.option.gravity = list(world.gravity)

        # Visual / offscreen framebuffer size - needed for render() over the
        # default 640x480 without MuJoCo's cryptic offwidth/offheight error.
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.quality.shadowsize = 4096

        # Ground texture + material - only used when world.ground_plane.
        # These live at spec level; ``add_texture`` / ``add_material``.
        grid_tex = spec.add_texture(
            name="grid_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=512,
            height=512,
            rgb1=[0.9, 0.9, 0.9],
            rgb2=[0.7, 0.7, 0.7],
        )
        grid_mat = spec.add_material(
            name="grid_mat",
            texrepeat=[8, 8],
            reflectance=0.1,
        )
        # Bind texture to the material's textures[2D] slot via set API (MjSpec
        # uses a typed index; 2D index is mjTEXROLE_RGB == 1 per MuJoCo headers,
        # but the pythonic way is the setter below).
        grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = grid_tex.name
        # ^ set by role name string; MjSpec resolves by name at compile.

        # Mesh assets for objects that declare ``shape == "mesh"``.
        for obj in world.objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)

        # Lights (world body children).
        spec.worldbody.add_light(
            name="main_light",
            pos=[0.0, 0.0, 3.0],
            dir=[0.0, 0.0, -1.0],
            diffuse=[1.0, 1.0, 1.0],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            name="fill_light",
            pos=[1.0, 1.0, 2.0],
            dir=[-0.5, -0.5, -1.0],
            diffuse=[0.5, 0.5, 0.5],
        )

        # Ground plane.
        if world.ground_plane:
            spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[5.0, 5.0, 0.01],
                material="grid_mat",
                conaffinity=1,
                condim=3,
            )

        # Cameras.
        for cam in world.cameras.values():
            SpecBuilder._add_camera(spec, cam)

        # Objects.
        for obj in world.objects.values():
            SpecBuilder._add_object(spec, obj)

        return spec

    # ------------------------------------------------------------------ camera
    @staticmethod
    def _add_camera(spec: Any, cam: SimCamera) -> None:
        """Add a world-fixed camera to ``spec``.

        If ``cam.target`` is set and differs from ``cam.position``, we convert
        the look-at direction to a quaternion via ``_target_quat`` and attach
        that to the camera's ``quat`` attribute - so the camera actually looks
        at its target. Otherwise the camera gets MuJoCo's default -Z look
        direction.
        """
        mujoco = _ensure_mujoco()
        pos = list(cam.position)
        kwargs: dict[str, Any] = {
            "name": cam.name,
            "pos": pos,
            "fovy": float(cam.fov),
            "mode": mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        }
        target = getattr(cam, "target", None)
        if target is not None:
            quat = _target_quat(pos, list(target))
            if quat is not None:
                kwargs["quat"] = quat
        spec.worldbody.add_camera(**kwargs)

    # ------------------------------------------------------------------ object
    @staticmethod
    def _add_object(spec: Any, obj: SimObject) -> None:
        """Add a SimObject to ``spec.worldbody``.

        Mirrors ``MJCFBuilder._object_xml`` semantics:

        * Dynamic objects (``is_static=False``) get a freejoint + an explicit
          inertial block (diagonal 0.001 inertia, user-supplied mass).
        * Static objects skip the freejoint and inertial.
        * Box/sphere/cylinder/capsule/plane/mesh are handled; ellipsoid is a
          bonus enabled by the new SHAPE_MAP.
        """
        body = spec.worldbody.add_body(
            name=obj.name,
            pos=list(obj.position),
            quat=list(obj.orientation),
        )

        if not obj.is_static:
            body.add_freejoint(name=f"{obj.name}_joint")
            # Explicit inertial matches the legacy builder: diag 0.001, mass
            # from SimObject. MuJoCo will auto-compute from geom density
            # otherwise, which can differ. In MjSpec the inertial properties
            # are body-level attributes, not a child element - ``explicitinertial``
            # tells the compiler to honour them verbatim.
            body.mass = float(obj.mass)
            body.inertia = [0.001, 0.001, 0.001]
            body.ipos = [0.0, 0.0, 0.0]
            body.explicitinertial = True

        geom_kwargs: dict[str, Any] = {
            "name": f"{obj.name}_geom",
            "type": _geom_type(obj.shape),
            "rgba": list(obj.color),
            "condim": 3,
        }
        if obj.shape == "mesh":
            geom_kwargs["meshname"] = f"mesh_{obj.name}"
        else:
            geom_kwargs["size"] = _normalize_size(obj.shape, list(obj.size))

        # Legacy code adds ``friction="1 0.5 0.001"`` on box only; we keep
        # that parity to avoid behavioural drift (other shapes use MuJoCo's
        # default 1 0.005 0.0001).
        if obj.shape == "box":
            geom_kwargs["friction"] = [1.0, 0.5, 0.001]

        body.add_geom(**geom_kwargs)


__all__ = ["SpecBuilder", "_geom_type", "_normalize_size", "_target_quat"]
