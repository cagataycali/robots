"""Export a compiled MuJoCo model as browser-renderable geometry.

The dashboard's 3D view does NOT run physics in the browser. The server is the
single source of truth: it runs ``mj_step``/``mj_forward`` and the browser is a
thin renderer. To make that work we ship two things:

1. **Static geometry** (once, via :func:`export_scene_geometry`): for every geom
   in the model -- its type, size, colour, and (for mesh geoms) the baked
   vertex/face arrays in the geom's local frame. This is everything Three.js
   needs to build the scene graph.

2. **Live geom poses** (per frame, via :func:`geom_pose_frame`): the world
   position (``geom_xpos``) and orientation (``geom_xmat``) of every geom.
   These are exactly the arrays MuJoCo's own renderer consumes, so applying
   them in Three.js yields a pixel-faithful view. 384 floats for the so100
   (32 geoms x 12) -- ~1.5 KB/frame, vs a JPEG camera stream.

Why geom-space (not joint-space) on the wire: shipping ``geom_xpos`` means the
browser never needs the kinematic tree, joint limits, or a physics engine. It
just sets transforms. The authoritative forward-kinematics already ran on the
server.

Coordinate note: MuJoCo is Z-up, right-handed. ``geom_xmat`` is a row-major
3x3. Three.js is Y-up; the viewer applies a single Z-up->Y-up rotation on the
scene root rather than transforming every geom here.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MuJoCo mjtGeom enum values we serialise. Primitive geoms (plane/sphere/
# capsule/ellipsoid/cylinder/box) are reconstructed from ``size`` in the
# browser; mesh geoms carry baked vertices/faces.
_GEOM_PLANE = 0
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_ELLIPSOID = 4
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7

_GEOM_TYPE_NAMES = {
    _GEOM_PLANE: "plane",
    _GEOM_SPHERE: "sphere",
    _GEOM_CAPSULE: "capsule",
    _GEOM_ELLIPSOID: "ellipsoid",
    _GEOM_CYLINDER: "cylinder",
    _GEOM_BOX: "box",
    _GEOM_MESH: "mesh",
}


def export_scene_geometry(model: Any, *, visual_only: bool = True, precision: int = 5) -> dict[str, Any]:
    """Bake a compiled ``mjModel`` into a JSON-serialisable scene description.

    Returns a dict::

        {
            "ngeom": int,
            "meshes": [ {"vert": [...], "face": [...], "normal": [...]} , ... ],
            "geoms": [
                {
                    "type": "mesh" | "box" | ...,
                    "mesh": <mesh-index into "meshes"> | None,
                    "size": [sx, sy, sz],
                    "rgba": [r, g, b, a],
                    "group": int,
                },
                ...
            ],
        }

    Visual filtering is left to the browser (it hides ``group==3`` collision
    geoms by default). ``vert``/``face`` are flat lists for compact JSON.
    """
    import numpy as np

    ngeom = int(model.ngeom)

    # First pass: decide which geoms we keep and which mesh ids they use.
    # ``visual_only`` drops MuJoCo collision group 3 (the duplicate hull
    # geoms) -- they are never rendered in the reference viewer either.
    keep_gid: list[int] = []
    used_mesh: set[int] = set()
    for gid in range(ngeom):
        group = int(model.geom_group[gid])
        if visual_only and group == 3:
            continue
        keep_gid.append(gid)
        if int(model.geom_type[gid]) == _GEOM_MESH:
            did = int(model.geom_dataid[gid])
            if did >= 0:
                used_mesh.add(did)

    # Bake only the referenced meshes; remap their ids to a compact range.
    mesh_remap: dict[int, int] = {}
    meshes: list[dict[str, Any]] = []
    for mid in sorted(used_mesh):
        vadr = int(model.mesh_vertadr[mid])
        vnum = int(model.mesh_vertnum[mid])
        fadr = int(model.mesh_faceadr[mid])
        fnum = int(model.mesh_facenum[mid])
        verts = np.round(
            np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float32), precision
        )
        faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int32)
        mesh_remap[mid] = len(meshes)
        meshes.append(
            {
                "vert": verts.reshape(-1).tolist(),
                "face": faces.reshape(-1).tolist(),
            }
        )

    # ``geoms`` is indexed to match the FULL model geom list so pose frames
    # (which carry all ngeom xpos/xmat) line up. Dropped geoms get a null
    # entry the browser skips, preserving index alignment without shipping
    # their (collision) geometry.
    geoms: list[dict[str, Any] | None] = [None] * ngeom
    for gid in keep_gid:
        gtype = int(model.geom_type[gid])
        dataid = int(model.geom_dataid[gid])
        mesh_idx = mesh_remap.get(dataid) if (gtype == _GEOM_MESH and dataid >= 0) else None
        geoms[gid] = {
            "type": _GEOM_TYPE_NAMES.get(gtype, "unknown"),
            "mesh": mesh_idx,
            "size": [round(float(x), precision) for x in model.geom_size[gid]],
            "rgba": [round(float(x), 3) for x in model.geom_rgba[gid]],
            "group": int(model.geom_group[gid]),
        }

    return {"ngeom": ngeom, "meshes": meshes, "geoms": geoms}


def geom_pose_frame(model: Any, data: Any) -> dict[str, Any]:
    """Snapshot every geom's world pose for one render frame.

    Returns ``{"xpos": [ngeom*3 floats], "xmat": [ngeom*9 floats]}`` -- the
    same arrays MuJoCo's renderer reads. The caller is responsible for holding
    the sim lock; this only reads ``data``.
    """
    import numpy as np

    xpos = np.round(np.asarray(data.geom_xpos, dtype=np.float32), 5).reshape(-1).tolist()
    xmat = np.round(np.asarray(data.geom_xmat, dtype=np.float32), 5).reshape(-1).tolist()
    return {"xpos": xpos, "xmat": xmat}
