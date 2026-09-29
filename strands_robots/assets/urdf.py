"""Build a MuJoCo asset (``robot.xml`` + ``scene.xml`` + meshes) from a URDF.

``robot_descriptions`` ships 120 URDF descriptions and 84 of them have no MJCF
sibling, so the MuJoCo backend could not load them: MuJoCo reads a URDF, but a
URDF has no actuators, no floor, no light, its meshes are often Collada
(``.dae``), which MuJoCo does not decode, and their paths are ``package://``
URIs that only a ROS workspace resolves. This module closes each of those gaps
once, at asset-resolution time, and persists the result in the same layout a
Menagerie robot has, so :func:`strands_robots.assets.manager.resolve_model_path`
finds it with no new resolver code and the SAME artifact feeds the Python sim,
the docs viewer and the thumbnail renderer.

Pipeline (:func:`build_urdf_asset`):

1. parse the URDF, resolve every ``<mesh filename>`` (``package://``, ``file://``,
   absolute or relative) against the description's package and repository;
2. copy ``.stl``/``.obj``/``.msh`` meshes, convert anything else trimesh can read
   to binary STL, refuse the rest with one fixed sentence;
3. rewrite the URDF (mesh paths relative, ROS-only elements dropped, a
   ``<mujoco><compiler>`` block that keeps visual meshes and link names);
4. ``mujoco.MjSpec.from_file`` on the rewritten URDF, then edit the spec: a
   freejoint for floating-base robots, one position actuator per hinge/slide
   joint sized from the URDF ``effort`` limit, damping and armature defaults,
   the root lifted so a floating robot stands on the floor;
5. compile, write ``robot.xml`` (the model) and ``scene.xml`` (floor, light,
   skybox, ``<include file="robot.xml"/>``), and ``urdf_asset.json`` with the
   provenance and the counts the registry reports (``joints`` = ``model.nu``).

Everything here is heavy (it imports the description module, which clones the
upstream repository on first use) and is reached only from the asset paths that
are already allowed to download: :func:`strands_robots.assets.download.auto_download_robot`
when the registry entry says ``asset.source.type == "urdf"``.

trimesh is optional (``pip install strands-robots[sim-urdf]``): without it a
description whose meshes are all STL/OBJ still builds, and one that needs a
conversion is refused with the sentence that names the extra.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils import log_safe

logger = logging.getLogger(__name__)

#: Mesh extensions MuJoCo decodes itself.
MUJOCO_MESH_EXTENSIONS: frozenset[str] = frozenset({".stl", ".obj", ".msh"})

#: Mesh extensions trimesh can read and this module converts to binary STL.
CONVERTIBLE_MESH_EXTENSIONS: frozenset[str] = frozenset({".dae", ".ply", ".glb", ".gltf", ".off", ".3mf", ".wrl"})

#: ``robot_descriptions`` tags that mean "the base is not bolted to the world".
FLOATING_TAGS: frozenset[str] = frozenset({"humanoid", "biped", "quadruped", "wheeled", "mobile_manipulator", "drone"})

#: ``robot_descriptions`` tag -> curated registry category (docs family).
#: Order matters: the first tag present wins, so a base tag outranks an arm tag
#: (PR2 is ``{"dual_arm", "mobile_manipulator"}`` and is a mobile manipulator).
CATEGORY_BY_TAG: dict[str, str] = {
    "mobile_manipulator": "mobile_manip",
    "humanoid": "humanoid",
    "biped": "humanoid",
    "quadruped": "mobile",
    "wheeled": "mobile",
    "drone": "aerial",
    "dual_arm": "bimanual",
    "end_effector": "hand",
    "arm": "arm",
    "educational": "arm",
}

#: Joints whose name says they drive a finger: their actuator gain is capped so
#: a URDF ``effort`` written for a whole arm does not crush the gripper.
_FINGER_RE = re.compile(r"finger|gripper|thumb|index|middle|ring|pinky|knuckle|jaw|hand_joint", re.IGNORECASE)

#: URDF elements MuJoCo has no use for and that reference files it cannot open
#: (Gazebo plugins ship as ``.so`` in ``filename=`` attributes, for example).
_ROS_ONLY_ELEMENTS: tuple[str, ...] = ("gazebo", "transmission", "ros2_control", "sensor", "plugin")

#: MuJoCo's mjMINVAL: the smallest mass or inertia a moving body may carry.
_MJ_MINVAL = 1e-15

#: What a link with a broken inertial gets: 1 g and a 1e-6 kg m^2 per kg diagonal.
_DEFAULT_MASS, _DEFAULT_INERTIA_PER_KG = 0.001, 1e-6

#: Gap left between the lowest geom and the floor when the root is lifted (m).
_FLOOR_CLEARANCE = 0.01

#: Gain bounds for the position actuators (N m / rad or N / m per unit error).
_KP_MIN, _KP_MAX = 5.0, 2000.0
_KP_DEFAULT_HINGE, _KP_DEFAULT_SLIDE, _KP_FINGER_MAX = 50.0, 500.0, 20.0

#: Files this module writes into the asset directory.
ROBOT_URDF, ROBOT_XML, SCENE_XML, ASSET_JSON, MESH_DIR = (
    "robot.urdf",
    "robot.xml",
    "scene.xml",
    "urdf_asset.json",
    "meshes",
)

#: Fixed refusal sentences, one per failure class (the ledger's ``reason`` column).
REFUSAL_MESH_MISSING = "a mesh the URDF names is not in the package or repository"
REFUSAL_MESH_FORMAT = "mesh format {ext} is not loadable by MuJoCo and has no converter"
REFUSAL_MESH_CONVERT = "trimesh could not read {file} ({error})"
REFUSAL_NO_TRIMESH = "mesh format {ext} needs trimesh: pip install 'strands-robots[sim-urdf]'"
REFUSAL_COMPILE = "MuJoCo refused the compiled spec: {error}"
REFUSAL_NO_JOINTS = "the compiled model has no actuated joint"
REFUSAL_CLONE = "upstream clone failed or URDF_PATH missing after import"
REFUSAL_XACRO = "the description ships xacro only; rendering it needs xacrodoc: pip install 'strands-robots[sim-urdf]'"


class UrdfBuildError(RuntimeError):
    """The URDF could not become a MuJoCo asset; ``str(exc)`` is the fixed sentence."""


@dataclass
class MeshRecord:
    """One mesh reference the URDF made and what became of it."""

    source: str
    ext: str
    output: str
    converted: bool


@dataclass
class UrdfAssetInfo:
    """Provenance and counts of a built asset, persisted as ``urdf_asset.json``."""

    name: str
    module: str
    urdf_path: str
    category: str
    floating: bool
    nq: int
    nu: int
    nmesh: int
    joints: list[str]
    meshes: list[MeshRecord] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    root_offset_z: float = 0.0
    repository: str | None = None
    commit: str | None = None

    @property
    def converted(self) -> int:
        """How many meshes went through trimesh."""
        return sum(1 for m in self.meshes if m.converted)

    @property
    def streamable(self) -> bool:
        """Whether every source mesh is a format a browser MuJoCo can fetch upstream."""
        return all(not m.converted for m in self.meshes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, with the derived fields spelled out."""
        d = asdict(self)
        d["converted"] = self.converted
        d["streamable"] = self.streamable
        return d


def category_for_tags(tags: Iterable[str]) -> str:
    """Map ``robot_descriptions`` tags to one curated registry category.

    The first tag with a mapping wins in the order of :data:`CATEGORY_BY_TAG`,
    so ``{"mobile_manipulator", "dual_arm"}`` is a mobile manipulator (the base
    decides the family, the arm count does not). No known tag gives ``"arm"``.
    """
    present = set(tags)
    for tag in CATEGORY_BY_TAG:
        if tag in present:
            return CATEGORY_BY_TAG[tag]
    return "arm"


def is_floating(tags: Iterable[str]) -> bool:
    """Whether the tags say the base moves through the room."""
    return bool(FLOATING_TAGS.intersection(tags))


def resolve_mesh_uri(uri: str, urdf_dir: Path, package_dir: Path | None, repo_dir: Path | None) -> Path | None:
    """Resolve a URDF mesh ``filename`` to a file on disk, or ``None``.

    ``package://<pkg>/<rest>`` is looked up as ``<rest>`` under the description's
    package directory when its name is ``<pkg>``, under ``<pkg>`` inside the
    package's parent and the repository, and finally under any directory named
    ``<pkg>`` in the repository, then as ``<rest>`` under the package directory
    and the URDF's directory whatever they are called. ``file://`` is stripped;
    a relative path is relative to the URDF's own directory.
    """
    if uri.startswith("package://"):
        rest = uri[len("package://") :]
        pkg, _, tail = rest.partition("/")
        roots = [r for r in (package_dir, package_dir.parent if package_dir else None, repo_dir) if r is not None]
        for root in roots:
            if root.name == pkg and (root / tail).is_file():
                return root / tail
            if (root / pkg / tail).is_file():
                return root / pkg / tail
        if repo_dir is not None and repo_dir.is_dir():
            for d in repo_dir.rglob(pkg):
                if d.is_dir() and (d / tail).is_file():
                    return d / tail
        # The package name in the URI is the ROS package, which need not be the
        # directory's name (a description checked out as ``urdf/`` or ``v1/``
        # still names its own package): the description's own package and the
        # URDF's directory are the last places to look.
        for fallback in (package_dir, urdf_dir):
            if fallback is not None and (fallback / tail).is_file():
                return fallback / tail
        return None
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    p = Path(os.path.expanduser(uri))
    if p.is_absolute():
        return p if p.is_file() else None
    # A relative path is relative to the URDF in the spec; in the wild it is as
    # often relative to the package (``meshes/x.stl`` next to ``urdf/``).
    for base in (urdf_dir, package_dir, repo_dir):
        if base is not None and (base / p).is_file():
            return base / p
    return None


#: MuJoCo's STL decoder refuses more faces than this; bigger meshes go to OBJ.
_STL_MAX_FACES = 200_000


def _convert_mesh(src: Path, dst_stem: Path) -> Path:
    """Convert *src* with trimesh; returns the file written (``.stl`` or ``.obj``).

    Binary STL is the compact choice; MuJoCo's STL decoder stops at 200,000
    faces, so a denser mesh is written as OBJ instead. Scenes are flattened to
    one mesh.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise UrdfBuildError(REFUSAL_NO_TRIMESH.format(ext=src.suffix.lower())) from exc
    try:
        loaded = trimesh.load(str(src), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.to_mesh()
        if loaded.is_empty and src.suffix.lower() == ".dae":
            loaded = _load_collada_tolerant(src)
        if loaded.is_empty:
            raise ValueError("empty mesh")
        ext = ".obj" if len(getattr(loaded, "faces", ())) > _STL_MAX_FACES else ".stl"
        dst = dst_stem.with_suffix(ext)
        loaded.export(str(dst), file_type=ext[1:])
        other = dst_stem.with_suffix(".obj" if ext == ".stl" else ".stl")
        if other.exists():
            other.unlink()
        return dst
    except UrdfBuildError:
        raise
    except Exception as exc:
        raise UrdfBuildError(REFUSAL_MESH_CONVERT.format(file=src.name, error=str(exc)[:120])) from exc


def _load_collada_tolerant(src: Path) -> Any:
    """Read a Collada file whose materials are broken, keeping every triangle.

    pycollada stops at the first ``DaeError`` by default and a texture image that
    is not in the repository (a common state of a ROS description) then reads as
    an empty scene through trimesh. Loading with the reference errors ignored
    and walking the bound scene (node transforms applied) recovers the geometry;
    the material is irrelevant here, MuJoCo gets a plain mesh either way.
    """
    import collada  # type: ignore[import-untyped]
    import numpy as np
    import trimesh

    doc = collada.Collada(
        str(src),
        ignore=[collada.common.DaeBrokenRefError, collada.common.DaeUnsupportedError, collada.common.DaeMalformedError],
    )
    parts = []
    scene = doc.scene
    bound = list(scene.objects("geometry")) if scene is not None else []
    for geom in bound:
        for prim in geom.primitives():
            tri = prim.triangleset() if hasattr(prim, "triangleset") else prim
            verts = getattr(tri, "vertex", None)
            idx = getattr(tri, "vertex_index", None)
            if verts is None or idx is None or len(idx) == 0:
                continue
            parts.append(trimesh.Trimesh(vertices=np.asarray(verts, dtype=float), faces=np.asarray(idx).reshape(-1, 3)))
    if not bound:  # no scene: fall back to the raw geometries, untransformed
        for geometry in doc.geometries:
            for prim in geometry.primitives:
                tri = prim.triangleset() if hasattr(prim, "triangleset") else prim
                verts = getattr(tri, "vertex", None)
                idx = getattr(tri, "vertex_index", None)
                if verts is None or idx is None or len(idx) == 0:
                    continue
                parts.append(
                    trimesh.Trimesh(vertices=np.asarray(verts, dtype=float), faces=np.asarray(idx).reshape(-1, 3))
                )
    if not parts:
        return trimesh.Trimesh()
    mesh = trimesh.util.concatenate(parts)
    # Collada files are commonly authored in Y-up; the asset's <up_axis> says so
    # and trimesh's own loader applies the same correction.
    if str(getattr(doc.assetInfo, "upaxis", "")).upper().startswith("Y"):
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
    return mesh


def _stl_is_loadable(path: Path) -> bool:
    """Whether MuJoCo's STL decoder accepts *path*: binary, 1..200,000 faces."""
    try:
        with path.open("rb") as fh:
            head = fh.read(84)
        if len(head) < 84 or head[:5].lower() == b"solid" and not head[80:84]:
            return False
        faces = int.from_bytes(head[80:84], "little")
        binary_size = 84 + 50 * faces
        return 0 < faces <= _STL_MAX_FACES and path.stat().st_size == binary_size
    except OSError:
        return False


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem) or "mesh"


def _is_fresh(src: Path, out: Path) -> bool:
    """A cached output is reused while the source is unchanged (mtime, size)."""
    try:
        return out.is_file() and out.stat().st_mtime >= src.stat().st_mtime and out.stat().st_size > 0
    except OSError:
        return False


_PREFIX_RE = re.compile(r"<(\w+):\w+")
_ROBOT_TAG_RE = re.compile(r"<robot\b")


def _parse_urdf(urdf_path: Path) -> ET.Element:
    """Parse the URDF; a file that still carries xacro-style prefixes gets them bound.

    A hand-exported URDF sometimes keeps ``<xacro:...>`` or ``<gazebo:...>``
    elements without the ``xmlns:`` declaration that makes them XML. Those
    elements are dropped anyway, so binding the prefix to a placeholder
    namespace is enough to read the rest.
    """
    text = urdf_path.read_text(encoding="utf-8", errors="replace")
    try:
        return ET.fromstring(text)  # noqa: S314 - the URDF is the asset being built; no external entities
    except ET.ParseError as exc:
        if "unbound prefix" not in str(exc):
            raise
    prefixes = sorted(set(_PREFIX_RE.findall(text)) - {"xml"})
    decls = "".join(f' xmlns:{p}="urn:strands-robots:{p}"' for p in prefixes)
    text = _ROBOT_TAG_RE.sub(f"<robot{decls}", text, count=1)
    return ET.fromstring(text)  # noqa: S314 - same file, prefixes now declared


def _inertia_is_positive(inertial: ET.Element) -> bool:
    """MuJoCo's rule: mass and every inertia eigenvalue above ``mjMINVAL`` (1e-15).

    A 1e-22 eigenvalue is positive to linear algebra and "not positive" to the
    compiler, which is what a CAD export of a sensor frame produces.
    """
    import numpy as np

    mass_el = inertial.find("mass")
    inertia = inertial.find("inertia")
    try:
        mass = float(mass_el.get("value", 0)) if mass_el is not None else 0.0
        if inertia is None:
            return mass > 0
        g = {k: float(inertia.get(k, 0) or 0) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    except ValueError:
        return False
    if mass <= _MJ_MINVAL:
        return False
    m = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    return bool(np.all(np.linalg.eigvalsh(m) > _MJ_MINVAL))


def _geom_is_degenerate(geometry: ET.Element) -> bool:
    """A primitive with a zero dimension, which MuJoCo refuses as ``size 0``."""
    for child in geometry:
        try:
            if child.tag == "box":
                return any(float(v) <= 0 for v in child.get("size", "1 1 1").split())
            if child.tag == "sphere":
                return float(child.get("radius", 1)) <= 0
            if child.tag == "cylinder":
                return float(child.get("radius", 1)) <= 0 or float(child.get("length", 1)) <= 0
            if child.tag == "capsule":
                return float(child.get("radius", 1)) <= 0
        except ValueError:
            return True
    return False


def _repair_urdf(root: ET.Element) -> list[str]:
    """Fix, in place, the URDF defects that make MuJoCo refuse a model.

    * an ``<inertial>`` whose mass is zero or whose inertia matrix is not
      positive definite becomes a small positive one (the link keeps its mass
      when that is positive; the inertia is what position control never
      notices); a link with no ``<inertial>`` gets that default too;
    * a visual or collision whose primitive has a zero dimension is removed;
    * a visual with several ``<material>`` children keeps the first;
    * a ``<material>`` re-defined under the same name becomes a reference to
      the first definition (MuJoCo refuses a repeated definition).

    Returns one line per repair for the log.
    """
    repairs: list[str] = []
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            # A link with no inertial is massless, which MuJoCo refuses once a
            # joint moves it; the URDF spec says the same link is a frame with
            # no dynamics, so it gets the smallest inertial that compiles.
            inertial = ET.SubElement(link, "inertial")
            repairs.append(f"link {link.get('name')}: no inertial, added the default")
        if not _inertia_is_positive(inertial):
            mass_el = inertial.find("mass")
            try:
                mass = float(mass_el.get("value", 0)) if mass_el is not None else 0.0
            except ValueError:
                mass = 0.0
            # A mass at the mjMINVAL edge (4e-15 kg lidar frames) would make the
            # derived inertia fall under it again; a link this light is a frame.
            mass = mass if mass >= _DEFAULT_MASS else _DEFAULT_MASS
            for child in list(inertial):
                if child.tag in ("mass", "inertia"):
                    inertial.remove(child)
            ET.SubElement(inertial, "mass", {"value": f"{mass:g}"})
            i = f"{mass * _DEFAULT_INERTIA_PER_KG:g}"
            ET.SubElement(inertial, "inertia", {"ixx": i, "iyy": i, "izz": i, "ixy": "0", "ixz": "0", "iyz": "0"})
            repairs.append(f"link {link.get('name')}: inertia was not positive definite, replaced")
        for shape in list(link):
            if shape.tag not in ("visual", "collision"):
                continue
            geometry = shape.find("geometry")
            if geometry is not None and _geom_is_degenerate(geometry):
                link.remove(shape)
                repairs.append(f"link {link.get('name')}: dropped a {shape.tag} with a zero-size primitive")
                continue
            materials = shape.findall("material")
            for extra in materials[1:]:  # one material per visual; MuJoCo refuses a second
                shape.remove(extra)
            if len(materials) > 1:
                repairs.append(
                    f"link {link.get('name')}: kept the first of {len(materials)} materials in a {shape.tag}"
                )
    seen: set[str] = set()
    for material in root.iter("material"):
        name = material.get("name")
        if not name:
            continue
        if name in seen and len(material):
            for child in list(material):
                material.remove(child)
            repairs.append(f"material {name}: repeated definition turned into a reference")
        elif len(material):
            seen.add(name)
    return repairs


def rewrite_urdf(
    urdf_path: Path,
    dest: Path,
    *,
    package_dir: Path | None,
    repo_dir: Path | None,
) -> tuple[Path, list[MeshRecord], dict[str, tuple[float, float]]]:
    """Resolve meshes into ``dest/meshes`` and write ``dest/robot.urdf``.

    Returns the rewritten URDF path, the mesh records and the per-joint
    ``(effort, damping)`` the URDF declared (MjSpec keeps neither). Repairs the
    rewrite makes (see :func:`_repair_urdf`) are logged, one line each.

    Raises:
        UrdfBuildError: a mesh is missing, unconvertible or of an unknown format.
    """
    root = _parse_urdf(urdf_path)
    repairs = _repair_urdf(root)
    for tag in _ROS_ONLY_ELEMENTS:
        for el in list(root.iter(tag)):
            for parent in root.iter():
                if el in list(parent):
                    parent.remove(el)
                    break
    mesh_dir = dest / MESH_DIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    records: list[MeshRecord] = []
    by_source: dict[Path, str] = {}
    taken: dict[str, Path] = {}  # output stem -> source, so one stem never serves two files
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename", "")
        src = resolve_mesh_uri(uri, urdf_path.parent, package_dir, repo_dir)
        if src is None:
            raise UrdfBuildError(f"{REFUSAL_MESH_MISSING}: {uri}")
        src = src.resolve()
        if src in by_source:
            mesh.set("filename", by_source[src])
            continue
        ext = src.suffix.lower()
        convert = ext not in MUJOCO_MESH_EXTENSIONS
        if convert and ext not in CONVERTIBLE_MESH_EXTENSIONS:
            raise UrdfBuildError(REFUSAL_MESH_FORMAT.format(ext=ext or "(none)"))
        stem = _safe_stem(src)
        # Two different sources with one stem (left/right meshes in sibling
        # folders) must not overwrite each other.
        while stem in taken and taken[stem] != src:
            stem = f"{stem}_{len(taken)}"
        taken[stem] = src
        if convert:
            cached = [c for c in (mesh_dir / f"{stem}.stl", mesh_dir / f"{stem}.obj") if _is_fresh(src, c)]
            out = cached[0] if cached else _convert_mesh(src, mesh_dir / stem)
        elif ext == ".stl" and not _stl_is_loadable(src):
            # An ASCII STL or one above MuJoCo's face cap is re-encoded.
            cached = [c for c in (mesh_dir / f"{stem}.stl", mesh_dir / f"{stem}.obj") if _is_fresh(src, c)]
            out = cached[0] if cached else _convert_mesh(src, mesh_dir / stem)
            convert = True
        else:
            out = mesh_dir / f"{stem}{ext}"
            if not _is_fresh(src, out):
                shutil.copy2(src, out)
        rel = f"{MESH_DIR}/{out.name}"
        by_source[src] = rel
        records.append(MeshRecord(source=str(src), ext=ext, output=rel, converted=convert))
        mesh.set("filename", rel)

    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        lim = joint.find("limit")
        dyn = joint.find("dynamics")
        effort = float(lim.get("effort", 0) or 0) if lim is not None else 0.0
        damping = float(dyn.get("damping", 0) or 0) if dyn is not None else 0.0
        limits[str(joint.get("name"))] = (effort, damping)

    mj = root.find("mujoco")
    if mj is None:
        mj = ET.SubElement(root, "mujoco")
    for old in list(mj.findall("compiler")):
        mj.remove(old)
    ET.SubElement(
        mj,
        "compiler",
        {
            "meshdir": ".",
            "discardvisual": "false",
            "fusestatic": "false",
            "balanceinertia": "true",
            "strippath": "false",
        },
    )
    out_urdf = dest / ROBOT_URDF
    out_urdf.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    for line in repairs:
        logger.info("urdf %s: %s", urdf_path.name, line)
    return out_urdf, records, limits


def _add_actuators(spec: Any, limits: dict[str, tuple[float, float]], mujoco: Any) -> list[str]:
    """One position actuator per hinge/slide joint; returns the joint names driven."""
    driven: list[str] = []
    # Enum values compared as int() to int(): a mujoco enum on the left of ==
    # (which is what a tuple membership test produces) stops matching numpy
    # fields on mujoco 3.12, silently, so the membership is spelled by value.
    hinge, slide_t = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    for joint in spec.joints:
        if int(joint.type) not in (hinge, slide_t):
            continue
        effort, _ = limits.get(joint.name, (0.0, 0.0))
        slide = int(joint.type) == slide_t
        kp = min(max(effort, _KP_MIN), _KP_MAX) if effort > 0 else (_KP_DEFAULT_SLIDE if slide else _KP_DEFAULT_HINGE)
        if _FINGER_RE.search(joint.name):
            kp = min(kp, _KP_FINGER_MAX)
        # MjSpec keeps damping per dof (a 3-vector); the URDF's own value, when
        # it gave one, already sits in it.
        if float(joint.damping[0]) <= 0:
            joint.damping = [kp / (10.0 if slide else 20.0)] * 3
        if not slide and float(joint.armature) <= 0:
            joint.armature = 0.01
        act = spec.add_actuator()
        act.name = joint.name
        act.target = joint.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if float(joint.range[0]) < float(joint.range[1]):
            act.ctrlrange = joint.range
            act.ctrllimited = True
        if effort > 0:
            act.forcerange = [-effort, effort]
            act.forcelimited = True
        driven.append(joint.name)
    return driven


def _lowest_point(model: Any, data: Any) -> float:
    """World z of the lowest geom AABB corner at the current pose (planes excluded)."""
    import mujoco
    import numpy as np

    lowest = 0.0
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):  # infinite, and it is the floor itself
            continue
        center = model.geom_aabb[g, :3]
        half = model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        # Extent of the rotated box along world z: sum of |R_z . e_i| * half_i.
        extent = float(np.abs(rot[2]).dot(half))
        z = float(data.geom_xpos[g][2] + rot[2].dot(center)) - extent
        lowest = min(lowest, z)
    return lowest


def _scene_xml(model_name: str, extent: float) -> str:
    ext = max(float(extent), 0.5)
    return (
        f'<mujoco model="{model_name} scene">\n'
        f'  <include file="{ROBOT_XML}"/>\n'
        f'  <statistic center="0 0 {ext / 2:.3f}" extent="{ext * 1.6:.3f}"/>\n'
        "  <visual>\n"
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="130" elevation="-20" offwidth="1280" offheight="960"/>\n'
        "  </visual>\n"
        "  <asset>\n"
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"'
        ' markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
        "  </asset>\n"
        "  <worldbody>\n"
        f'    <light pos="0 0 {ext * 2:.2f}" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def build_from_urdf(
    urdf_path: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    name: str,
    module: str = "",
    tags: Iterable[str] = (),
    floating: bool | None = None,
    package_dir: str | os.PathLike[str] | None = None,
    repo_dir: str | os.PathLike[str] | None = None,
    repository: str | None = None,
    commit: str | None = None,
) -> UrdfAssetInfo:
    """Build ``robot.xml``, ``scene.xml`` and the meshes for one URDF into *dest*.

    Pure with respect to the network: everything it reads is on disk. The
    ``robot_descriptions`` import that fetches the description is
    :func:`build_urdf_asset`'s job.

    Args:
        urdf_path: The source URDF.
        dest: Asset directory to write (created).
        name: Registry name of the robot (``modelname`` of the MJCF).
        module: ``robot_descriptions`` module, for provenance.
        tags: ``robot_descriptions`` tags; decide floating vs fixed and the category.
        floating: Override the tag heuristic (``None`` = decide from *tags*).
        package_dir: The description's ``PACKAGE_PATH`` (``package://`` root).
        repo_dir: The description's ``REPOSITORY_PATH``.
        repository: ``owner/repo`` of the upstream, for the docs viewer pin.
        commit: Upstream commit the description is pinned to.

    Returns:
        The :class:`UrdfAssetInfo` also written as ``urdf_asset.json``.

    Raises:
        UrdfBuildError: with one of the fixed refusal sentences.
    """
    import mujoco

    urdf = Path(urdf_path)
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    pkg = Path(package_dir) if package_dir else urdf.parent
    repo = Path(repo_dir) if repo_dir else None
    rewritten, meshes, limits = rewrite_urdf(urdf, out, package_dir=pkg, repo_dir=repo)

    try:
        spec = mujoco.MjSpec.from_file(str(rewritten))
    except Exception as exc:
        raise UrdfBuildError(REFUSAL_COMPILE.format(error=str(exc).splitlines()[0][:160])) from exc
    spec.modelname = name
    roots = list(spec.worldbody.bodies)
    is_free = is_floating(tags) if floating is None else bool(floating)
    has_free = any(int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE) for j in spec.joints)
    if is_free and not has_free and roots:
        roots[0].add_freejoint()
    is_free = is_free or has_free
    driven = _add_actuators(spec, limits, mujoco)
    if not driven:
        raise UrdfBuildError(REFUSAL_NO_JOINTS)

    try:
        model = spec.compile()
    except Exception as exc:
        raise UrdfBuildError(REFUSAL_COMPILE.format(error=str(exc).splitlines()[0][:160])) from exc

    # A URDF has no floor, so its root frame sits wherever the author put it:
    # a quadruped's trunk at z=0 with the legs below, a hand's palm at z=0
    # with the fingers below. Lift the root until the lowest geom clears the
    # floor by a centimetre; a robot already above it is left where it is.
    root_offset = 0.0
    if roots:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        lowest = _lowest_point(model, data)
        if lowest < 0.0:
            root_offset = -lowest + _FLOOR_CLEARANCE
            roots[0].pos[2] = float(roots[0].pos[2]) + root_offset
            try:
                model = spec.compile()
            except Exception as exc:  # pragma: no cover - the same spec compiled a line ago
                raise UrdfBuildError(REFUSAL_COMPILE.format(error=str(exc).splitlines()[0][:160])) from exc

    (out / ROBOT_XML).write_text(spec.to_xml(), encoding="utf-8")
    (out / SCENE_XML).write_text(_scene_xml(name, model.stat.extent), encoding="utf-8")
    try:
        scene = mujoco.MjModel.from_xml_path(str(out / SCENE_XML))
    except Exception as exc:
        raise UrdfBuildError(REFUSAL_COMPILE.format(error=str(exc).splitlines()[0][:160])) from exc

    info = UrdfAssetInfo(
        name=name,
        module=module,
        urdf_path=str(urdf),
        category=category_for_tags(tags),
        floating=is_free,
        nq=int(scene.nq),
        nu=int(scene.nu),
        nmesh=int(scene.nmesh),
        joints=driven,
        meshes=meshes,
        root_offset_z=root_offset,
        repository=repository,
        commit=commit,
    )
    (out / ASSET_JSON).write_text(json.dumps(info.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "urdf asset built: %s -> %s (nq %d, nu %d, %d meshes, %d converted)",
        log_safe(name),
        out,
        info.nq,
        info.nu,
        len(meshes),
        info.converted,
    )
    return info


def _description_pin(mod: Any) -> tuple[str | None, str | None]:
    """``owner/repo`` and commit of a ``robot_descriptions`` module's upstream.

    ``robot_descriptions`` keeps the pins in ``_repositories.REPOSITORIES``; the
    module's ``REPOSITORY_PATH`` ends in the entry's key or its ``cache_path``.
    """
    repo_path = getattr(mod, "REPOSITORY_PATH", None)
    if not repo_path:
        return None, None
    try:
        from robot_descriptions._repositories import REPOSITORIES  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - the module import already needed the package
        return None, None
    leaf = Path(str(repo_path)).name
    # Keyed by description name, but the directory on disk is ``cache_path``,
    # which differs for some (Universal_Robots_ROS2_Description clones into
    # ``ur_description``): match on either.
    entry = REPOSITORIES.get(leaf)
    if entry is None:
        entry = next((r for r in REPOSITORIES.values() if getattr(r, "cache_path", None) == leaf), None)
    if entry is None:
        return None, None
    m = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?/?\Z", str(getattr(entry, "url", "")))
    return (m.group(1) if m else None), (str(entry.commit) if getattr(entry, "commit", None) else None)


def _urdf_path_of(mod: Any) -> str | None:
    """``URDF_PATH`` of a description, rendering a xacro-only one through robot_descriptions.

    23 descriptions (the Universal Robots family, Kinova Jaco, xArm, Franka
    FER/FR3 v2, Stretch SE3) ship xacro and no URDF; ``robot_descriptions``
    renders and caches the URDF with ``xacrodoc`` when asked through its own
    ``_xacro.get_urdf_path``. Without ``xacrodoc`` installed the render raises
    and the description is refused with the sentence that names the package.
    """
    direct = getattr(mod, "URDF_PATH", None)
    if direct:
        return str(direct)
    if not getattr(mod, "XACRO_PATH", None):
        return None
    try:
        from robot_descriptions._xacro import get_urdf_path  # type: ignore[import-not-found]

        return str(get_urdf_path(mod))
    except ModuleNotFoundError as exc:
        raise UrdfBuildError(REFUSAL_XACRO) from exc
    except Exception as exc:
        raise UrdfBuildError(f"{REFUSAL_CLONE}: xacro render failed ({str(exc)[:120]})") from exc


def build_urdf_asset(name: str, module: str, dest_dir: str | os.PathLike[str]) -> UrdfAssetInfo:
    """Import the ``robot_descriptions`` URDF *module* and build its asset.

    Heavy: the import clones the upstream repository on first use (serialized by
    :func:`strands_robots._description_cache.import_description`). Writes into
    ``<dest_dir>/<module>/``.

    Raises:
        UrdfBuildError: the description has no readable ``URDF_PATH``, or any
            build refusal from :func:`build_from_urdf`.
    """
    from .._description_cache import import_description

    try:
        mod = import_description(module)
    except Exception as exc:
        # ImportError for a missing module; GitPython's GitCommandError when the
        # upstream repository or the pinned commit is gone (eve_r3's Halodi repo
        # was deleted): both are the same fact to the caller.
        detail = str(exc).strip().splitlines()
        raise UrdfBuildError(f"{REFUSAL_CLONE}: {detail[-1][:160] if detail else type(exc).__name__}") from exc
    urdf_path = _urdf_path_of(mod)
    if not urdf_path or not os.path.isfile(str(urdf_path)):
        raise UrdfBuildError(REFUSAL_CLONE)
    tags: set[str] = set()
    try:
        from robot_descriptions._descriptions import DESCRIPTIONS  # type: ignore[import-not-found]

        tags = set(getattr(DESCRIPTIONS.get(module), "tags", ()) or ())
    except ImportError:  # pragma: no cover - the module import above already needed the package
        pass
    repo, commit = _description_pin(mod)
    return build_from_urdf(
        str(urdf_path),
        Path(dest_dir) / module,
        name=name,
        module=module,
        tags=tags,
        package_dir=getattr(mod, "PACKAGE_PATH", None),
        repo_dir=getattr(mod, "REPOSITORY_PATH", None),
        repository=repo,
        commit=commit,
    )


def read_asset_info(asset_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The ``urdf_asset.json`` a build left in *asset_dir*, or ``None``."""
    path = Path(asset_dir) / ASSET_JSON
    if not path.is_file():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded
