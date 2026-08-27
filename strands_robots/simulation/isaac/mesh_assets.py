"""Mesh asset utilities for the Isaac backend (OBJ/STL/MSH -> USD, cached).

The Isaac backend realizes custom mesh objects (``add_object(shape="mesh")``,
and the LIBERO scene objects extracted by
:func:`strands_robots.simulation.isaac.loaders.load_mjcf_scene_objects`) by
referencing a USD asset onto the live stage. LIBERO ships its object meshes as
OBJ/STL/MSH inside the upstream ``libero`` package (the compiled MJCFs
reference the binary ``.msh`` form for the object visuals), so those need a one-time
conversion to USD first. This module owns that conversion plus the pure-stdlib
triangle-mesh parsing it is built on:

* :func:`load_mesh_geometry` - parse an OBJ, STL or legacy MuJoCo binary MSH
  file into ``(points, face_vertex_counts, face_vertex_indices)`` with no
  third-party dependency (``xml``-free, ``trimesh``-free), so the geometry
  half is unit-testable on CPU-only CI without Isaac Sim or the
  ``sim-isaac`` extra. MSH is load-bearing, not a nicety: LIBERO's
  robosuite-compiled scenes declare the bowl/plate *visual* meshes as
  ``.msh`` (the OBJ sources live beside them but the MJCF references the
  compiled form), so without it the exact objects #2459 names would stay
  box proxies.
* :func:`mesh_aabb` - the axis-aligned bounding box of a mesh asset (used by
  the MJCF scene loader so a mesh-only LIBERO body gets the mesh's real
  bounds as its collision proxy instead of a hardcoded 0.05 m box, and by
  ``add_object`` to report the compiled extent the way the MuJoCo backend
  does).
* :func:`convert_mesh_to_usd` - author a ``UsdGeom.Mesh`` USD file from the
  parsed geometry via ``pxr`` (the ``usd-core`` wheel from the ``sim-isaac``
  extra; no Kit / ``omni.kit.asset_converter`` session needed), cached under
  ``$STRANDS_BASE_DIR/asset_cache/usd_meshes/<sha256>.usda`` keyed on the
  asset's *content* - the same content-addressed pattern as the LIBERO
  scene cache. Scale is never baked into the converted USD: callers apply
  scale on the referencing prim's xform, so one cache entry serves every
  scale.

Failure semantics (the loaders' fail-loud contract): a missing file raises
:class:`FileNotFoundError`, an unsupported extension, an asset with no
triangle geometry or one declaring a vertex coordinate that is not finite
raises :class:`ValueError`, and a missing ``pxr`` raises
:class:`ImportError` with an install hint. Nothing here degrades to a silent
default box.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from itertools import chain

from strands_robots.utils import get_base_dir

__all__ = [
    "MESH_EXTENSIONS",
    "USD_EXTENSIONS",
    "load_mesh_geometry",
    "mesh_aabb",
    "mesh_usd_cache_dir",
    "convert_mesh_to_usd",
]

#: Triangle-mesh formats this module can parse and convert to USD.
MESH_EXTENSIONS: tuple[str, ...] = (".obj", ".stl", ".msh")

#: Formats that are already USD and need no conversion (referenced verbatim).
USD_EXTENSIONS: tuple[str, ...] = (".usd", ".usda", ".usdc", ".usdz")


def _require_mesh_file(mesh_path: str) -> str:
    """Validate ``mesh_path`` exists and carries a parseable extension.

    Returns the lowercase extension. Raises :class:`FileNotFoundError` for a
    missing / non-file path and :class:`ValueError` for an extension outside
    :data:`MESH_EXTENSIONS` - never a silent fallback.
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"mesh asset not found: {mesh_path}")
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext not in MESH_EXTENSIONS:
        raise ValueError(
            f"unsupported mesh format {ext!r} for {mesh_path}: expected one of {MESH_EXTENSIONS} "
            f"(USD assets {USD_EXTENSIONS} need no conversion and are referenced directly)"
        )
    return ext


def _non_finite_vertex_error(mesh_path: str, index: int, vertex: tuple[float, float, float]) -> ValueError:
    """The refusal for a vertex whose coordinates are not all finite.

    A NaN or infinite coordinate is not a position, and neither of the two
    ways this module measures ``points`` reports that. ``min``/``max``
    order a NaN as neither smaller nor larger than anything, so
    :func:`mesh_aabb`'s running comparison drops it and returns the bounds
    of the vertices that are finite - the same numbers a mesh declaring
    only those would produce, under no error at all - which is exactly the
    outcome :func:`_unterminated_facet_error` exists to refuse, reached
    from the other side: there the vertices are in the bounds and their
    triangle is in no face, here the vertex is in a face and not in the
    bounds. The extent :func:`convert_mesh_to_usd` authors is built by a
    different spelling (``min`` over an iterable rather than a running
    comparison) which keeps a leading NaN and drops a trailing one, so the
    same asset can be measured two ways by one module, decided by the
    order its vertices happen to be declared in. An infinite coordinate
    needs no such subtlety: it makes the reported extent unbounded, and a
    scene object sized from it is a collision proxy with no bounds.

    MuJoCo - the owner of the ``.msh`` format and a reader of the other two -
    refuses the same input rather than measuring around it
    (``vertex coordinate N is not finite``), so refusing is the format
    family's own disposition, and it is the one this module's fail-loud
    contract already states.

    One wording for all four parse paths, raised where they meet, so a
    format cannot drift into tolerating what its siblings refuse. The
    locator is the vertex index rather than a file line because two of the
    four paths are binary and have no lines, and it is the locator MuJoCo
    reports for exactly this refusal.
    """
    return ValueError(
        f"mesh {mesh_path}: vertex {index} has a coordinate that is not finite ({vertex!r}) - "
        f"a mesh measured around it reports bounds for geometry the file does not declare, so the "
        f"asset is refused instead of being sized"
    )


def _non_finite_scale_error(mesh_path: str, scale: tuple[float, float, float]) -> ValueError:
    """The refusal for a mesh scale whose components are not all finite.

    :func:`load_mesh_geometry` already refuses a vertex that is not finite,
    for the reason :func:`_non_finite_vertex_error` states: ``min``/``max``
    order a NaN as neither smaller nor larger than anything, so the running
    comparison drops it and reports the bounds of what remains. ``scale``
    reaches that same comparison - :func:`mesh_aabb` measures
    ``vertex * scale`` - so a non-finite component there poisons every
    vertex on that axis and the comparison drops all of them, leaving the
    axis at its ``inf``/``-inf`` seed. The asset's own coordinates being
    finite is therefore not enough: the transform applied to them has to be
    finite too, and it arrives from the caller rather than from the file.

    The two components fail differently and neither is screenable:

    * A NaN axis reports a centre of NaN and, because the reported extent is
      floored at ``1e-4``, a **finite** size of 0.1 mm. A consumer that
      screens the reported fields for non-finite values catches the centre
      and not the size, so a metre-scale asset passes that screen as a
      plausible sub-millimetre one.
    * An infinite axis reports an infinite centre and a size of NaN, because
      the extent is ``inf - inf``.

    A scale of ``0.0`` stays accepted: it is a finite request to flatten the
    asset on that axis, and the ``1e-4`` floor is doing the job it is there
    for. A negative scale stays accepted too - it mirrors the asset, and the
    running comparison is order-independent, so the bound is right.
    """
    return ValueError(
        f"mesh {mesh_path}: scale has a component that is not finite ({scale!r}) - a mesh measured "
        f"through it reports a centre that is not a position and an extent floored at 1e-4, so the "
        f"asset is refused instead of being sized"
    )


def load_mesh_geometry(
    mesh_path: str,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse an OBJ, STL or legacy MuJoCo MSH asset into USD-shaped triangle-mesh arrays.

    Parameters
    ----------
    mesh_path : str
        Filesystem path to a ``.obj``, ``.stl`` or ``.msh`` (legacy MuJoCo binary mesh) file.

    Returns
    -------
    tuple
        ``(points, face_vertex_counts, face_vertex_indices)`` matching the
        ``UsdGeom.Mesh`` attribute layout: ``points`` is a list of xyz
        tuples, ``face_vertex_counts`` one entry per face (OBJ polygons are
        kept as n-gons; STL and MSH are always triangles), ``face_vertex_indices``
        the flattened 0-based vertex indices.

    Raises
    ------
    FileNotFoundError
        If ``mesh_path`` doesn't exist.
    ValueError
        If the extension is unsupported, the file is malformed, the
        asset declares no vertices / no faces (an empty mesh renders
        nothing, which downstream would misread as a scene property), or
        a vertex coordinate is not finite (a mesh measured around one
        reports bounds for geometry the file does not declare, so it is
        refused rather than sized).
    """
    ext = _require_mesh_file(mesh_path)
    if ext == ".obj":
        points, counts, indices = _parse_obj(mesh_path)
    elif ext == ".msh":
        points, counts, indices = _parse_msh(mesh_path)
    else:
        points, counts, indices = _parse_stl(mesh_path)
    # One finiteness rule for every format, applied where the four parse
    # paths meet, so the two consumers that measure ``points`` -
    # :func:`mesh_aabb` and the extent :func:`convert_mesh_to_usd` authors -
    # cannot disagree about the same file. The fast path is a single
    # C-level ``map`` over the flattened coordinates (measurably ~4% of the
    # parse it follows on the largest shipped meshes); the per-vertex walk
    # that names the offender runs only when that map has already failed.
    if not all(map(math.isfinite, chain.from_iterable(points))):
        index, vertex = next((i, v) for i, v in enumerate(points) if not all(map(math.isfinite, v)))
        raise _non_finite_vertex_error(mesh_path, index, vertex)
    return points, counts, indices


def _parse_obj(mesh_path: str) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse a Wavefront OBJ: ``v`` positions + ``f`` faces (n-gons kept).

    OBJ is whitespace-delimited, so a keyword may be followed by a tab as
    legitimately as by a space (MuJoCo reads either form identically). The
    keyword is therefore read off the split fields rather than matched as a
    ``"v "`` prefix: matching the prefix skipped a tab-separated ``v`` line,
    which either dropped the vertex silently - leaving a mesh whose bounds
    are the ones the file did not declare - or made a later face reference
    fall out of range and blamed the face for a vertex the parser never
    recognised. ``vn`` / ``vt`` still fall out naturally: they are different
    keywords, not the vertex keyword followed by a separator.
    """
    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []
    with open(mesh_path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            parts = line.split()
            if not parts:
                continue
            keyword = parts[0]
            if keyword == "v":
                if len(parts) < 4:
                    raise ValueError(f"OBJ {mesh_path}:{lineno}: vertex with fewer than 3 components: {line!r}")
                try:
                    points.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError as e:
                    raise ValueError(f"OBJ {mesh_path}:{lineno}: non-numeric vertex component: {line!r}") from e
            elif keyword == "f":
                refs = parts[1:]
                if len(refs) < 3:
                    raise ValueError(f"OBJ {mesh_path}:{lineno}: face with fewer than 3 vertices: {line!r}")
                face: list[int] = []
                for ref in refs:
                    # ``i``, ``i/t``, ``i//n`` or ``i/t/n``; only the vertex
                    # index is consumed. Negative indices are relative to the
                    # vertices declared so far, per the OBJ spec.
                    head = ref.split("/", 1)[0]
                    try:
                        idx = int(head)
                    except ValueError as e:
                        raise ValueError(f"OBJ {mesh_path}:{lineno}: non-integer face index {ref!r}") from e
                    resolved = idx - 1 if idx > 0 else len(points) + idx
                    if resolved < 0 or resolved >= len(points):
                        raise ValueError(
                            f"OBJ {mesh_path}:{lineno}: face index {idx} out of range ({len(points)} vertices declared)"
                        )
                    face.append(resolved)
                counts.append(len(face))
                indices.extend(face)
    if not points or not counts:
        raise ValueError(f"mesh {mesh_path} has no triangle geometry (empty vertices/faces)")
    return points, counts, indices


def _parse_stl(mesh_path: str) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse an STL (binary or ASCII), deduplicating shared vertices.

    Binary detection goes by the declared triangle count matching the file
    size (the robust test - binary STL headers may legally start with the
    bytes ``solid`` that otherwise mark the ASCII form).
    """
    with open(mesh_path, "rb") as fh:
        data = fh.read()
    if len(data) >= 84:
        (tri_count,) = struct.unpack_from("<I", data, 80)
        if 84 + 50 * tri_count == len(data) and tri_count > 0:
            return _parse_stl_binary(data, tri_count)
    return _parse_stl_ascii(mesh_path, data)


def _parse_stl_binary(data: bytes, tri_count: int) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    points: list[tuple[float, float, float]] = []
    index_of: dict[tuple[float, float, float], int] = {}
    counts: list[int] = []
    indices: list[int] = []
    for rec in struct.iter_unpack("<12fH", data[84 : 84 + 50 * tri_count]):
        # rec[0:3] is the facet normal (ignored); rec[3:12] the 3 vertices.
        for base in (3, 6, 9):
            vert = (float(rec[base]), float(rec[base + 1]), float(rec[base + 2]))
            idx = index_of.get(vert)
            if idx is None:
                idx = len(points)
                index_of[vert] = idx
                points.append(vert)
            indices.append(idx)
        counts.append(3)
    return points, counts, indices


def _unterminated_facet_error(mesh_path: str, lineno: int, vertex_count: int, before: str) -> ValueError:
    """The refusal for an ASCII-STL facet still open when something else ends it.

    One wording for both sites - a following ``facet`` keyword, and the end
    of the file - so the two cannot drift, and each names the line the open
    facet's first vertex is on, which is the only line that identifies it.
    """
    return ValueError(
        f"STL {mesh_path}:{lineno}: facet left unterminated - {vertex_count} vertices read and no 'endfacet' "
        f"before {before}, so the triangle they form would be dropped while their coordinates stay in the bounds"
    )


def _parse_stl_ascii(mesh_path: str, data: bytes) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse an ASCII STL, deduplicating shared vertices.

    A facet is flushed on its ``endfacet``, so a facet still open when the
    file ends - or when another ``facet`` keyword arrives - is refused rather
    than left pending. Its vertices are already in ``points``, and therefore
    in the bounds :func:`mesh_aabb` reports and the ``extent``
    :func:`convert_mesh_to_usd` authors, while the triangle they form is in
    no face: accepting one returns bounds for geometry the mesh does not
    carry, under no error at all. The two binary parsers in this module
    already refuse a truncated file by reconciling their declared counts
    against the byte length; this is the same refusal for the format whose
    triangle count is not declared anywhere but in its structure.
    """
    text = data.decode("utf-8", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise ValueError(f"STL {mesh_path}: neither a well-formed binary STL nor an ASCII one (no 'solid' header)")
    points: list[tuple[float, float, float]] = []
    index_of: dict[tuple[float, float, float], int] = {}
    counts: list[int] = []
    indices: list[int] = []
    facet: list[int] = []
    facet_lineno = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        lowered = line.lower()
        if lowered.startswith("vertex"):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"STL {mesh_path}:{lineno}: vertex with fewer than 3 components: {line!r}")
            try:
                vert = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError as e:
                raise ValueError(f"STL {mesh_path}:{lineno}: non-numeric vertex component: {line!r}") from e
            idx = index_of.get(vert)
            if idx is None:
                idx = len(points)
                index_of[vert] = idx
                points.append(vert)
            if not facet:
                facet_lineno = lineno
            facet.append(idx)
        elif lowered.startswith("endfacet"):
            if len(facet) != 3:
                raise ValueError(f"STL {mesh_path}:{lineno}: facet with {len(facet)} vertices (expected 3)")
            indices.extend(facet)
            counts.append(3)
            facet = []
        elif lowered.startswith("facet") and facet:
            raise _unterminated_facet_error(mesh_path, facet_lineno, len(facet), f"the facet starting on line {lineno}")
    if facet:
        raise _unterminated_facet_error(mesh_path, facet_lineno, len(facet), "the end of the file")
    if not points or not counts:
        raise ValueError(f"mesh {mesh_path} has no triangle geometry (empty vertices/faces)")
    return points, counts, indices


def _parse_msh(mesh_path: str) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Parse a legacy MuJoCo binary mesh (``.msh``).

    The format MuJoCo <= 3.x compiled and LIBERO ships for its object
    *visual* meshes: a 16-byte header of four little-endian int32s
    ``(nvertex, nnormal, ntexcoord, nface)`` followed by densely packed
    float32 vertex positions (3 per vertex), float32 normals (3 per
    normal), float32 texture coordinates (2 per texcoord) and int32 face
    vertex indices (3 per face - the format is triangles-only). Only
    positions and faces are consumed here. The declared counts must
    account for the file's exact byte length - the same
    sizes-must-reconcile test the binary-STL sniffing uses - so a
    truncated or foreign file fails loud instead of parsing as garbage
    geometry.
    """
    with open(mesh_path, "rb") as fh:
        data = fh.read()
    if len(data) < 16:
        raise ValueError(f"MSH {mesh_path}: shorter than the 16-byte (nvertex, nnormal, ntexcoord, nface) header")
    nvertex, nnormal, ntexcoord, nface = struct.unpack_from("<4i", data, 0)
    if nvertex <= 0 or nface <= 0 or nnormal < 0 or ntexcoord < 0:
        raise ValueError(
            f"MSH {mesh_path}: non-positive geometry counts (nvertex={nvertex}, nface={nface}, "
            f"nnormal={nnormal}, ntexcoord={ntexcoord})"
        )
    expected = 16 + 4 * (3 * nvertex + 3 * nnormal + 2 * ntexcoord + 3 * nface)
    if expected != len(data):
        raise ValueError(
            f"MSH {mesh_path}: declared counts need {expected} bytes but the file holds {len(data)} - "
            f"truncated, or not a legacy MuJoCo binary mesh"
        )
    points: list[tuple[float, float, float]] = [
        (float(x), float(y), float(z)) for x, y, z in struct.iter_unpack("<3f", data[16 : 16 + 12 * nvertex])
    ]
    face_start = expected - 12 * nface
    indices: list[int] = []
    for tri in struct.iter_unpack("<3i", data[face_start:expected]):
        for idx in tri:
            if idx < 0 or idx >= nvertex:
                raise ValueError(f"MSH {mesh_path}: face index {idx} out of range ({nvertex} vertices declared)")
            indices.append(idx)
    return points, [3] * nface, indices


def mesh_aabb(
    mesh_path: str, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned bounding box of a mesh asset, in the mesh's local frame.

    Parameters
    ----------
    mesh_path : str
        Filesystem path to a ``.obj``, ``.stl`` or ``.msh`` (legacy MuJoCo binary mesh) file.
    scale : tuple[float, float, float]
        Per-axis scale applied to the vertices before measuring (an MJCF
        ``<mesh scale=...>``). Defaults to unit scale. Every component must
        be finite; it is checked before the asset is parsed, because a
        transform that can never be applied is not worth reading a mesh
        for, and the check needs nothing from the file.

    Returns
    -------
    tuple
        ``(center, size)`` where ``size`` is the FULL extent per axis (not
        half-extents), matching :class:`~strands_robots.simulation.isaac.loaders.SceneObject`.

    Raises
    ------
    FileNotFoundError / ValueError
        Same contract as :func:`load_mesh_geometry`, plus
        :func:`_non_finite_scale_error` when a ``scale`` component is not
        finite. The asset's vertices being finite does not make the bound
        finite: ``scale`` reaches the same running comparison, and the
        reported extent's ``1e-4`` floor makes a NaN axis look like a
        plausible sub-millimetre asset rather than a failure.
    """
    if not all(map(math.isfinite, scale)):
        raise _non_finite_scale_error(mesh_path, scale)
    points, _counts, _indices = load_mesh_geometry(mesh_path)
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for p in points:
        for i in range(3):
            v = p[i] * scale[i]
            mins[i] = min(mins[i], v)
            maxs[i] = max(maxs[i], v)
    center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))
    size = tuple(max(maxs[i] - mins[i], 1e-4) for i in range(3))
    return center, size  # type: ignore[return-value]


def mesh_usd_cache_dir() -> str:
    """Cache directory for converted USD meshes (created on first use).

    ``$STRANDS_BASE_DIR/asset_cache/usd_meshes`` - typically
    ``~/.strands_robots/asset_cache/usd_meshes`` - following the LIBERO
    scene cache's location convention.
    """
    cache_dir = os.path.join(str(get_base_dir()), "asset_cache", "usd_meshes")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def convert_mesh_to_usd(mesh_path: str, cache_dir: str | None = None) -> str:
    """Convert an OBJ/STL/MSH asset to a USD file, content-addressed and cached.

    A ``.usd``/``.usda``/``.usdc``/``.usdz`` input is returned unchanged (it
    is already referenceable). Anything else must be a :data:`MESH_EXTENSIONS`
    format; the parsed geometry is authored as a single ``UsdGeom.Mesh`` at
    ``/Mesh`` (the default prim), unit scale, Z-up, meters - callers apply
    pose / scale on the prim that references it. The output lands at
    ``<cache_dir>/<sha256(file bytes)>.usda`` and is written atomically
    (temp file + ``os.replace``) so a crashed conversion never leaves a
    torn cache entry that a later call would trust.

    Parameters
    ----------
    mesh_path : str
        Filesystem path to the source asset.
    cache_dir : str, optional
        Override the cache location (tests). Defaults to
        :func:`mesh_usd_cache_dir`.

    Returns
    -------
    str
        Path to the converted (or passed-through) USD file.

    Raises
    ------
    FileNotFoundError / ValueError
        Same contract as :func:`load_mesh_geometry`.
    ImportError
        If ``pxr`` is unavailable (install the ``sim-isaac`` extra).
    """
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in USD_EXTENSIONS:
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"mesh asset not found: {mesh_path}")
        return mesh_path
    _require_mesh_file(mesh_path)

    with open(mesh_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    out_dir = cache_dir if cache_dir is not None else mesh_usd_cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{digest}.usda")
    if os.path.isfile(out_path):
        return out_path

    # Read the asset before probing for ``pxr``, the way the extension /
    # existence check above already precedes it: an install without the
    # ``sim-isaac`` extra should still learn what is wrong with the mesh it
    # handed over, not only what is missing from the environment. The cache
    # hit above still short-circuits both, and a cache entry only exists
    # because some earlier call parsed the same bytes successfully.
    points, counts, indices = load_mesh_geometry(mesh_path)

    try:
        from pxr import Gf, Usd, UsdGeom, Vt  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "converting a mesh asset to USD requires Pixar USD (pxr). "
            "Install via: pip install 'strands-robots[sim-isaac]' "
            "or directly: pip install 'usd-core>=25.5,<27.0.0'"
        ) from e

    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]

    # Author to a sibling temp path, then rename into place. ``CreateNew``
    # refuses an existing file, so the temp name carries the pid; the final
    # ``os.replace`` is atomic on POSIX.
    tmp_path = os.path.join(out_dir, f".{digest}.{os.getpid()}.tmp.usda")
    stage = Usd.Stage.CreateNew(tmp_path)
    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*mins), Gf.Vec3f(*maxs)]))
        # Polygonal asset, not a subdivision surface: without this RTX/Hydra
        # would smooth the control cage (catmullClark is USD's default).
        mesh.CreateSubdivisionSchemeAttr("none")
        stage.SetDefaultPrim(mesh.GetPrim())
        stage.GetRootLayer().Save()
    finally:
        # Drop the stage handle before the rename so the layer is closed.
        del stage
    os.replace(tmp_path, out_path)
    return out_path
