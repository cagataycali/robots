"""What the browser twin needs from a compiled ``MjModel``: geoms, meshes, cameras.

The twin does not run physics and does not parse MJCF. It draws the geoms
MuJoCo already compiled, at the world poses MuJoCo already computed
(``geom_xpos``/``geom_xmat`` arrive over the telemetry socket as binary rows).
Mesh vertices come out of ``mesh_vert``/``mesh_face`` - the compiled model
holds them, so nothing here opens a file, and there is no path to contain.

Everything reads ``MjModel`` only, which is immutable after load and safe to
read from a request thread while the worker steps ``MjData``.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

# mjtGeom, by number - the names the browser switches on.
GEOM_TYPES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
    8: "sdf",
}

MESH_MAGIC = b"SRM1"  # header: magic, nvert u32, nface u32; then f32[nvert*3], u32[nface*3]

#: Floats per geom in a binary pose frame: ``[x y z | 3x3 row-major]``. Published
#: in :func:`describe` so the browser strides by it instead of restating it, and
#: graded against ``static/twin.js`` and the packer that fills the rows
#: (:func:`strands_robots.dashboard.sim_session._pack_poses`).
POSE_ROW_FLOATS = 12


def _name(model: Any, kind: str, index: int) -> str | None:
    import mujoco

    obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind}")
    return mujoco.mj_id2name(model, obj, int(index)) or None


def describe(model: Any) -> dict[str, Any]:
    """Geoms with type/size/color/mesh/body, meshes with vertex counts, cameras and lights."""
    geoms = []
    for i in range(int(model.ngeom)):
        matid = int(model.geom_matid[i])
        rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[i]
        geoms.append(
            {
                "id": i,
                "name": _name(model, "GEOM", i),
                "type": GEOM_TYPES.get(int(model.geom_type[i]), "unknown"),
                "size": [float(x) for x in model.geom_size[i]],
                "rgba": [round(float(x), 4) for x in rgba],
                "group": int(model.geom_group[i]),
                "mesh": int(model.geom_dataid[i]) if int(model.geom_type[i]) == 7 else None,
                "body": _name(model, "BODY", int(model.geom_bodyid[i])),
            }
        )
    meshes = [
        {
            "id": i,
            "name": _name(model, "MESH", i),
            "vertices": int(model.mesh_vertnum[i]),
            "faces": int(model.mesh_facenum[i]),
            "url": f"mesh/{i}",
        }
        for i in range(int(model.nmesh))
    ]
    cameras = [
        {
            "id": i,
            "name": _name(model, "CAMERA", i),
            "pos": [float(x) for x in model.cam_pos0[i]] if hasattr(model, "cam_pos0") else None,
            "fovy": float(model.cam_fovy[i]),
        }
        for i in range(int(model.ncam))
    ]
    return {
        "ngeom": int(model.ngeom),
        "geoms": geoms,
        "meshes": meshes,
        "cameras": cameras,
        "nlight": int(model.nlight),
        "pose_row_floats": POSE_ROW_FLOATS,
        "mesh_format": MESH_MAGIC.decode(),
    }


def mesh_bytes(model: Any, index: int) -> bytes:
    """One compiled mesh as ``SRM1 | nvert | nface | f32 verts | u32 faces`` (little-endian).

    Raises:
        IndexError: for a mesh id the model does not have.
    """
    if not 0 <= index < int(model.nmesh):
        raise IndexError(index)
    va, vn = int(model.mesh_vertadr[index]), int(model.mesh_vertnum[index])
    fa, fn = int(model.mesh_faceadr[index]), int(model.mesh_facenum[index])
    verts = np.ascontiguousarray(model.mesh_vert[va : va + vn], dtype="<f4")
    # faces index into this mesh's own vertices already (mjModel stores them local)
    faces = np.ascontiguousarray(model.mesh_face[fa : fa + fn], dtype="<u4")
    return MESH_MAGIC + struct.pack("<II", vn, fn) + verts.tobytes() + faces.tobytes()
