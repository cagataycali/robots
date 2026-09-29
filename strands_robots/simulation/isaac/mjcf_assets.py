"""Convert an MJCF robot description to USD, content-addressed and cached.

Isaac Sim ships an MJCF importer - ``isaacsim.asset.importer.mjcf``, exposing
``MJCFImporter`` and ``MJCFImporterConfig`` - and this module is the seam that
turns it into a plain ``mjcf -> usd path`` function so the Isaac backend can load
the *same* description file the MuJoCo backend loads.

That matters because it is what makes this repository's parity claim true rather
than aspirational. ``docs/learn/simulation/isaac.md`` promises that "the joint-name and
observation contract matches the MuJoCo backend, [so] policies and observation
mappings transfer unchanged between backends", and the only way to keep that
promise is for both backends to read one file. Measured on
``nvcr.io/nvidia/isaac-sim:6.0.1`` (A10G), converting the descriptions
``strands_robots.simulation.model_registry.resolve_model`` already resolves and
loading the result through ``add_robot(usd_path=...)`` reproduces MuJoCo's joint
vocabulary exactly:

===========================  ===============================================  =========
asset                        joint names                                      vs MuJoCo
===========================  ===============================================  =========
``franka_emika_panda``       ``joint1``..``joint7``, ``finger_joint1/2``       9 of 9
``trs_so_arm100``            ``Rotation Pitch Elbow Wrist_Pitch Wrist_Roll     6 of 6
                             Jaw``
===========================  ===============================================  =========

Both wired a live articulation, reset, stepped, and reported a non-empty
``get_observation()``.

Three properties of the vendor importer shape this module, all measured rather
than read off the documentation:

* ``usd_path`` is an output **directory root**, not a file name. The importer
  writes ``<usd_path>/<stem>/<stem>.usda`` and returns that path, so the return
  value is what a caller must reference - deriving the path itself would guess.
* Given no ``usd_path`` it writes **beside the source**, which fails with
  ``OSError: [Errno 30] Read-only file system`` for a description that lives in a
  shared read-only checkout. Every description this package resolves is such a
  file - ``~/.strands_robots/assets/<dir>`` is a symlink into
  ``~/.cache/robot_descriptions/`` - so an explicit destination is mandatory here
  rather than merely tidy.
* It writes a USD *file* and does **not** add anything to the live stage, so the
  result still has to be referenced in. ``IsaacSimulation._load_usd_robot``
  already does exactly that, which is why this module converts and stops.

The importer only exists inside a running Kit application, because it is a Kit
extension rather than a library. ``add_robot`` is always called on a live
simulation, so that holds wherever this is reached in production; a unit test
stands the importer in.
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

from strands_robots.utils import get_base_dir

__all__ = [
    "MJCF_EXTENSIONS",
    "USD_EXTENSIONS",
    "convert_mjcf_to_usd",
    "robot_usd_cache_dir",
]

#: Extensions this module treats as an MJCF description. MuJoCo itself accepts
#: any name, but every description in the shipped registry is ``.xml``, and
#: restricting the set is what lets a USD input be passed through below without
#: ambiguity.
MJCF_EXTENSIONS: tuple[str, ...] = (".xml", ".mjcf")

#: Already-USD inputs, referenced verbatim with no conversion. Spelled here
#: rather than imported from :mod:`strands_robots.simulation.isaac.mesh_assets`
#: so the two modules' vocabularies can diverge if USD ever gains a robot-only
#: container format; the values are deliberately identical today.
USD_EXTENSIONS: tuple[str, ...] = (".usd", ".usda", ".usdc", ".usdz")


def robot_usd_cache_dir() -> str:
    """Cache directory for converted robot USD assets (created on first use).

    ``$STRANDS_BASE_DIR/asset_cache/usd_robots`` - typically
    ``~/.strands_robots/asset_cache/usd_robots`` - a sibling of the mesh cache
    :func:`~strands_robots.simulation.isaac.mesh_assets.mesh_usd_cache_dir`
    writes to, following the same convention.
    """
    cache_dir = os.path.join(str(get_base_dir()), "asset_cache", "usd_robots")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


#: MJCF ``<asset>`` children that name a file on disk. ``<mesh>`` is the one that
#: matters for PhysX, and the rest are hashed for the same reason: a texture or a
#: heightfield swapped under a byte-identical entry directory is a different
#: robot, and nothing downstream would report it.
_ASSET_FILE_TAGS: tuple[str, ...] = ("mesh", "texture", "hfield", "skin")


#: The file whose presence makes a cache entry usable. Written INSIDE the staging
#: directory and published by the rename, so a reader never observes an entry
#: without one.
_MARKER_NAME = ".converted"


def _read_marker(marker: str) -> str | None:
    """The USD path a completed cache entry records, or ``None``.

    ``None`` covers every not-usable state without distinguishing them, because
    the caller's action is the same for all of them: an absent marker, an
    unreadable one, an empty one, and one naming a file that is no longer there.
    """
    try:
        with open(marker, encoding="utf-8") as fh:
            cached = fh.read().strip()
    except OSError:
        return None
    return cached if cached and os.path.isfile(cached) else None


def _referenced_files(mjcf_path: str) -> list[str]:
    """Every file *mjcf_path* pulls in, transitively and absolute.

    An MJCF reaches outside its own directory in two ways, and the shipped
    registry uses both (AGENTS.md > Registry conventions records the nested
    layouts):

    * ``<include file="../so_arm100/so_arm100.xml"/>`` - ``lekiwi``'s entry point
      does this, so the whole arm's joints and geoms live in a sibling directory.
    * ``<compiler meshdir="../assets/meshes"/>`` - ``asimov_v0`` does this, so
      every mesh PhysX simulates is outside the entry directory.

    Resolution follows MuJoCo's own rules, matching
    :func:`strands_robots.simulation.isaac.loaders._mjcf_model_toplevel` and
    :func:`~strands_robots.simulation.isaac.loaders._parse_mjcf_mesh_assets`: an
    include path is relative to the *including* file, while ``<compiler>`` and
    ``<asset>`` are model-global so a mesh directory declared in an included
    fragment still resolves against the *entry* file's directory, and the last
    declaration in document order wins - per ATTRIBUTE.

    The base is per asset KIND, which is the part that is easy to get wrong:
    MuJoCo resolves ``mesh`` / ``hfield`` / ``skin`` against ``meshdir`` and
    ``texture`` against ``texturedir``, with ``assetdir`` the fallback for both and
    the model file's own directory the fallback for that. Collapsing the three into
    one "last directory seen" made ``texturedir`` beat ``meshdir`` inside a single
    ``<compiler meshdir=... texturedir=...>``, so every mesh resolved into the
    texture directory, missed, and contributed ``<unreadable>`` instead of its
    bytes - the digest then did not move when the geometry PhysX simulates changed,
    which is exactly the staleness this closure exists to prevent.

    A missing, unreadable, malformed or cyclic reference contributes its path and
    no bytes rather than raising: this is a cache key, and refusing to compute one
    would fail a conversion the vendor importer is perfectly able to report on
    itself. The path still enters the manifest, so two trees differing only in a
    broken reference do not collide.
    """
    entry = os.path.normpath(os.path.abspath(mjcf_path))
    entry_dir = os.path.dirname(entry)

    includes: list[str] = []
    # One slot per attribute, NOT one list of every directory seen. Flattening
    # them into a single list and keeping the last entry made ``texturedir`` beat
    # ``meshdir`` whenever one ``<compiler>`` carried both - the reverse of
    # MuJoCo's rule and of this function's own docstring - so a mesh resolved into
    # the texture directory, contributed no bytes, and left the real geometry out
    # of the key. ``<compiler meshdir=... texturedir=...>`` is an ordinary
    # Menagerie shape, so that is the common case rather than a corner.
    declared_dirs: dict[str, str] = {}
    # Paired with its tag, because the base to resolve against is per KIND:
    # MuJoCo reads ``mesh`` / ``hfield`` / ``skin`` against ``meshdir`` and
    # ``texture`` against ``texturedir``. One shared base cannot be right for both
    # whenever a model declares them separately.
    asset_files: list[tuple[str, str]] = []

    def _walk(path: str, base_dir: str, seen: frozenset[str]) -> None:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            return
        for element in root.iter():
            if element.tag == "compiler":
                # Last declaration in document order wins, per attribute, which is
                # what ``_parse_mjcf_mesh_assets`` does for the pair it reads.
                for attr in ("meshdir", "assetdir", "texturedir"):
                    value = element.get(attr)
                    if value:
                        declared_dirs[attr] = value
            elif element.tag in _ASSET_FILE_TAGS:
                value = element.get("file")
                if value:
                    asset_files.append((element.tag, value))
            elif element.tag == "include":
                value = element.get("file")
                if not value:
                    continue
                target = os.path.normpath(
                    os.path.abspath(value if os.path.isabs(value) else os.path.join(base_dir, value))
                )
                if target in seen:
                    continue
                includes.append(target)
                _walk(target, os.path.dirname(target), seen | {target})

    _walk(entry, entry_dir, frozenset({entry}))

    def _base_for(tag: str) -> str:
        """The directory MuJoCo resolves a ``tag`` asset's relative file against.

        ``meshdir`` / ``texturedir`` are the kind-specific declarations and
        ``assetdir`` is the fallback for both; with none declared, a relative
        asset resolves against the model file's own directory. Resolved against
        ``entry_dir`` rather than the including file's directory because
        ``<compiler>`` is model-global - a mesh directory declared in an included
        fragment still resolves from the entry file.
        """
        specific = "texturedir" if tag == "texture" else "meshdir"
        declared = declared_dirs.get(specific) or declared_dirs.get("assetdir")
        if not declared:
            return entry_dir
        return declared if os.path.isabs(declared) else os.path.join(entry_dir, declared)

    resolved = list(includes)
    for tag, name in asset_files:
        base = _base_for(tag)
        resolved.append(os.path.normpath(os.path.abspath(name if os.path.isabs(name) else os.path.join(base, name))))
    return resolved


def _asset_digest(mjcf_path: str) -> str:
    """A digest covering the description *and the files it pulls in*.

    An MJCF is not self-contained: Menagerie's ``scene.xml`` is a handful of
    lines that ``<include>`` the robot body and reference a ``meshdir`` of STLs,
    and the geometry PhysX ends up simulating lives in those. Keying the cache on
    ``scene.xml`` alone would therefore hand back a stale USD after any change
    that did not touch the one file named - which is most of them.

    So the digest covers every file under the description's directory, by
    relative path and content. That is a full read of the asset directory on each
    call (~35 MB for ``franka_emika_panda``, tens of milliseconds), paid even on
    a cache hit; it is bounded by the size of one robot description and is
    negligible beside the conversion it guards, which takes seconds.

    **And every file it references from OUTSIDE that directory**, because the
    directory walk alone is not the file set the conversion reads. The shipped
    registry's nested layouts reach out of it by design: ``lekiwi``'s entry point
    ``<include>``s ``../so_arm100/so_arm100.xml``, and ``asimov_v0`` declares
    ``meshdir="../assets/meshes"`` - so a change to the arm's joints or to any
    mesh left the key identical and served the USD built from the *old*
    description, silently, under ``status: success``. That is the exact staleness
    the paragraph above says this digest exists to prevent, and the wrong
    ``so_arm100.xml`` is a wrong joint vocabulary, which defeats the
    joint-name-parity guarantee this module is here to provide. The collision
    holds in the other direction too: two trees whose entry directories are
    byte-identical but whose siblings differ shared one key.

    The referenced closure is a *superset* of the walk rather than a replacement
    for it. A file inside the entry directory is already covered by relative path
    and content, so only the outside ones are added, and each contributes the
    path it resolved to relative to that directory (``../so_arm100/so_arm100.xml``)
    - a relative spelling so two machines with the same layout agree, which is the
    same reason the walk below sorts.

    Sorting the manifest is what makes the digest reproducible: :func:`os.walk`
    does not promise an order, so an unsorted manifest would key the same bytes
    differently on two machines and miss every cache entry.
    """
    root = os.path.dirname(os.path.abspath(mjcf_path))
    manifest = hashlib.sha256()

    def _fold(full: str, label: str) -> None:
        manifest.update(label.encode("utf-8", "surrogateescape"))
        try:
            with open(full, "rb") as fh:
                for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):  # type: ignore[misc]
                    manifest.update(chunk)
        except OSError:
            # A file that cannot be read cannot contribute its bytes, and
            # skipping it silently would let two different trees share a
            # digest. Fold the failure itself in, so an unreadable file is a
            # distinct key rather than an absent one.
            manifest.update(b"\x00<unreadable>")

    walked: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            walked.add(os.path.normpath(os.path.abspath(full)))
            _fold(full, os.path.relpath(full, root))

    # Then whatever the description reaches for outside that tree. Sorted for the
    # same reproducibility reason, and de-duplicated so a mesh named twice does
    # not fold twice.
    for referenced in sorted(set(_referenced_files(mjcf_path)) - walked):
        _fold(referenced, os.path.relpath(referenced, root))

    # The description's own path within the tree matters: one directory can hold
    # several entry points (Menagerie ships ``scene.xml`` beside the bare robot
    # body), and they convert to different USD.
    manifest.update(os.path.relpath(os.path.abspath(mjcf_path), root).encode("utf-8", "surrogateescape"))
    return manifest.hexdigest()


def _importer_version() -> str:
    """Version of the toolchain that writes the USD, folded into the cache key.

    The converter's output layout is version-specific: Isaac Sim 6.0.x
    (mujoco-usd-converter 0.2.0) flattens physics inline, 6.1.x (0.5.0) puts it
    behind a ``Physics`` variantSet. A cache keyed on the MJCF bytes alone served
    a 6.1 conversion to a 6.0 process sharing ``~/.strands_robots``.
    """
    from importlib.metadata import PackageNotFoundError, version

    parts = []
    for dist in ("isaacsim", "mujoco-usd-converter"):
        try:
            parts.append(f"{dist}={version(dist)}")
        except PackageNotFoundError:
            parts.append(f"{dist}=?")
    return ",".join(parts)


#: Bumped when the post-import fix-ups below change what a cache entry holds.
_POSTPROCESS_VERSION = "drives-v3"


def _position_servo_gains(mjcf_path: str) -> dict[str, tuple[float, float, float | None]]:
    """``{joint: (kp, kd, force_limit)}`` for every MuJoCo position servo in *mjcf_path*.

    Read from the COMPILED model, not the XML: ``dampratio`` and class-default
    ``kp`` are resolved only by MuJoCo's compiler (a ``dampratio="1"`` servo is
    authored as ``biasprm[2]=+1`` and compiles to ``-kd``). A position servo is
    ``gaintype=fixed``, ``biastype=affine``, ``biasprm[1] == -gainprm[0]``;
    anything else (motors, velocity servos) is left out. ``{}`` when MuJoCo is
    not importable or the model does not compile - the vendor conversion then
    stands as it is.
    """
    try:
        import mujoco
    except ImportError:
        return {}
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    except (ValueError, OSError, RuntimeError):
        return {}
    gains: dict[str, tuple[float, float, float | None]] = {}
    for i in range(model.nu):
        if int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_gaintype[i]) != int(mujoco.mjtGain.mjGAIN_FIXED):
            continue
        if int(model.actuator_biastype[i]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        kp = float(model.actuator_gainprm[i][0])
        bias = model.actuator_biasprm[i]
        if kp <= 0 or float(bias[1]) != -kp:
            continue
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i][0]))
        if not joint:
            continue
        force = float(model.actuator_forcerange[i][1]) if bool(model.actuator_forcelimited[i]) else None
        # The joint's own passive damping (``<joint damping=>``) is velocity
        # damping MuJoCo applies alongside the servo; a PhysX drive has one
        # damping term, so it carries both. so101 declares kv=0 and puts all of
        # its damping on the joint - without this its drives had damping 0 and
        # the arm rang around its targets.
        joint_id = int(model.actuator_trnid[i][0])
        passive = float(model.dof_damping[int(model.jnt_dofadr[joint_id])])
        gains[joint] = (kp, max(0.0, -float(bias[2])) + passive, force)
    return gains


def _author_position_drives(usd_file: str, mjcf_path: str) -> list[str]:
    """Give the converted robot's PhysX joint drives its MJCF position-servo gains.

    The Isaac Sim MJCF importer (6.0.x and 6.1.x) converts a ``<position>``
    actuator to a PhysX drive only when ``biasprm[2] < 0`` - i.e. an explicit
    ``kv``. A servo that states ``dampratio`` (every Menagerie arm: so100/so101,
    panda, ...) stores ``biasprm[2] = +dampratio``, fails that check, and the
    drive is left at ``stiffness=0, damping=0``: measured on so100, every joint
    hung limp under gravity (Wrist_Pitch 1.3 rad off a 0.3 rad target, Jaw
    resting on its lower limit). The vendor path also writes MuJoCo's N*m/rad
    gains unconverted into USD's per-DEGREE drive units, 57x too stiff.

    Authors ``stiffness = kp * pi/180`` and ``damping = kd * pi/180`` on each
    revolute joint's angular drive (prismatic joints: linear drive, m units, no
    conversion) as opinions in the entry's ROOT layer, which are stronger than
    the ``Physics`` variant the drives live in. Returns the joints written.
    """
    import math

    gains = _position_servo_gains(mjcf_path)
    if not gains:
        return []
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    stage = Usd.Stage.Open(usd_file)
    # Compose with the PhysX flavour selected (session layer: not saved), so the
    # joints the simulation will see are the ones found and edited.
    default = stage.GetDefaultPrim()
    if default and default.GetVariantSets().HasVariantSet("Physics"):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            vset = default.GetVariantSets().GetVariantSet("Physics")
            if not vset.GetVariantSelection():
                vset.SetVariantSelection("physx" if "physx" in vset.GetVariantNames() else "physics")
    stage.SetEditTarget(stage.GetRootLayer())
    # A joint whose MJCF name is not a valid USD identifier (so101: "1") is
    # transcoded by the converter (``tn__1_``); decode the prim names onto the
    # MJCF vocabulary the gains are keyed by.
    from strands_robots.simulation.isaac.joint_names import demangle_usd_joint_names

    joint_prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    decoded, _ = demangle_usd_joint_names([p.GetName() for p in joint_prims], list(gains))
    written: list[str] = []
    for prim, name in zip(joint_prims, decoded, strict=True):
        if name not in gains:
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            instance, scale = "angular", math.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            instance, scale = "linear", 1.0
        else:
            continue
        kp, kd, force = gains[name]
        drive = UsdPhysics.DriveAPI.Apply(prim, instance)
        drive.CreateStiffnessAttr().Set(kp * scale)
        drive.CreateDampingAttr().Set(kd * scale)
        if force is not None:
            drive.CreateMaxForceAttr().Set(force)
        written.append(name)
    if written:
        stage.GetRootLayer().Save()
    return written


def convert_mjcf_to_usd(
    mjcf_path: str,
    cache_dir: str | None = None,
    *,
    fix_base: bool | None = None,
    import_scene: bool = False,
) -> str:
    """Convert an MJCF description to a referenceable USD file.

    A ``.usd``/``.usda``/``.usdc``/``.usdz`` input is returned unchanged - it is
    already referenceable - so a caller can hand this either format. Anything
    else must be an :data:`MJCF_EXTENSIONS` file.

    The result is cached under ``<cache_dir>/<digest>/`` keyed on
    :func:`_asset_digest` together with the two options that change the produced
    USD, so a robot is converted once per (description, posture) rather than once
    per ``add_robot`` call. A conversion that fails leaves no directory behind,
    because a torn cache entry is one a later call would trust.

    Parameters
    ----------
    mjcf_path : str
        Filesystem path to the MJCF description, or to a USD file to pass
        through.
    cache_dir : str, optional
        Override the cache location (tests). Defaults to
        :func:`robot_usd_cache_dir`.
    fix_base : bool or None, optional
        Whether to weld the root body to the world. ``None`` - the default, and
        the vendor default - honours whatever the description itself says, which
        is the only choice that keeps a floating-base robot floating: every
        shipped quadruped and humanoid declares its base with ``<freejoint>``,
        and ``True`` would silently bolt such a robot to the ground while
        reporting success. ``True`` welds it, ``False`` frees it.
    import_scene : bool, optional
        Whether to import the description's ``<worldbody>`` furniture (floors,
        lights) alongside the robot. Defaults to ``False``: ``add_robot`` is
        asking for a robot, and the world it goes into already has a ground
        plane and lighting of its own.

    Returns
    -------
    str
        Path to the converted (or passed-through) USD file.

    Raises
    ------
    FileNotFoundError
        If *mjcf_path* does not exist.
    ValueError
        If *mjcf_path* is neither an MJCF nor a USD extension.
    ImportError
        If the Isaac MJCF importer is unavailable, with ``name`` set to the
        module that is missing. The importer is a Kit extension, so this is also
        what a call from outside a running Isaac Sim application reports.
    """
    ext = os.path.splitext(mjcf_path)[1].lower()
    if ext in USD_EXTENSIONS:
        if not os.path.isfile(mjcf_path):
            raise FileNotFoundError(f"robot asset not found: {mjcf_path}")
        return mjcf_path
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"MJCF description not found: {mjcf_path}")
    if ext not in MJCF_EXTENSIONS:
        raise ValueError(
            f"cannot convert {mjcf_path!r} to USD: expected an MJCF "
            f"({', '.join(MJCF_EXTENSIONS)}) or a USD file "
            f"({', '.join(USD_EXTENSIONS)}), got {ext or 'no extension'!r}"
        )

    # The posture options are part of the key, not just the payload: the same
    # description converted with fix_base=True is a different robot from the
    # same description converted with fix_base=None, and a cache keyed on the
    # bytes alone would serve whichever was built first.
    key = hashlib.sha256(
        f"{_asset_digest(mjcf_path)}|fix_base={fix_base}|import_scene={import_scene}|importer={_importer_version()}|post={_POSTPROCESS_VERSION}".encode()
    ).hexdigest()
    out_dir = cache_dir if cache_dir is not None else robot_usd_cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    target_root = os.path.join(out_dir, key)

    stem = os.path.splitext(os.path.basename(mjcf_path))[0]
    # The layout the importer produces, recorded by the marker below rather than
    # recomputed: the vendor writes ``<root>/<stem>/<stem>.usda`` today, and a
    # future release that changes that would make a derived path silently wrong.
    marker = os.path.join(target_root, _MARKER_NAME)
    cached = _read_marker(marker)
    if cached is not None:
        return cached

    try:
        from isaacsim.asset.importer.mjcf import (  # type: ignore[import-not-found]
            MJCFImporter,
            MJCFImporterConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "converting an MJCF description to USD requires Isaac Sim's MJCF "
            "importer extension (isaacsim.asset.importer.mjcf). It is a Kit "
            "extension, so it resolves only inside a running Isaac Sim "
            "application - install the runtime (see docs/learn/simulation/isaac.md) "
            "and call this from a live simulation.",
            name="isaacsim.asset.importer.mjcf",
        ) from exc

    # Convert into a sibling staging directory, then rename into place, so a
    # crashed or half-written conversion is never visible under the real key. The
    # marker is written inside staging before that rename, so the rename publishes
    # a complete entry and no reader can see a directory without one -
    # :func:`_install_entry` owns the rest of that contract, including what to do
    # when another process has already published this key.
    #
    # One staging directory per CONVERSION, not per process: ``mkdtemp`` creates a
    # name no concurrent caller can derive. A name carrying only the pid collides
    # between THREADS of one process, and the collision is not benign - the winner
    # renames the shared directory onto the key, so the loser's importer output
    # disappears mid-call and it raises "reported success but wrote no USD file"
    # for a conversion that in fact succeeded.
    staging = tempfile.mkdtemp(prefix=f".{key}.{os.getpid()}.", suffix=".tmp", dir=out_dir)
    try:
        config = MJCFImporterConfig()
        config.mjcf_path = os.path.abspath(mjcf_path)
        config.usd_path = staging
        config.fix_base = fix_base
        config.import_scene = import_scene
        produced = MJCFImporter(config=config).import_mjcf()
    except BaseException:
        _remove_tree(staging)
        raise

    resolved = _resolve_produced(produced, staging, stem)
    if resolved is None:
        _remove_tree(staging)
        raise RuntimeError(
            f"the Isaac MJCF importer reported success for {mjcf_path!r} but wrote no "
            f"USD file under {staging!r} (it returned {produced!r}). Refusing to "
            f"return a path nothing can reference."
        )

    try:
        _author_position_drives(resolved, mjcf_path)
    except BaseException:
        _remove_tree(staging)
        raise

    final = os.path.join(target_root, os.path.relpath(resolved, staging))
    # The marker goes INSIDE staging, naming the path it will have once installed,
    # so the rename below publishes a COMPLETE entry in one step. Written after the
    # rename instead, there was a window in which ``target_root`` existed with no
    # marker: a concurrent reader saw an unusable entry and converted again for
    # nothing, and a crash in the window left a torn entry behind permanently.
    with open(os.path.join(staging, _MARKER_NAME), "w", encoding="utf-8") as fh:
        fh.write(final)

    return _install_entry(staging, target_root, marker, final, mjcf_path)


def _install_entry(staging: str, target_root: str, marker: str, final: str, mjcf_path: str) -> str:
    """Publish *staging* as the entry at *target_root*, or defer to the winner.

    **Never deletes a completed entry.** The cache root is shared cross-process -
    ``~/.strands_robots/asset_cache/usd_robots`` - and the per-conversion staging
    directory says concurrent converters are an intended case, so two callers that
    both miss the marker for one key both convert. The install used to
    ``_remove_tree(target_root)`` before renaming, which means the loser deleted
    the winner's finished entry *after* the winner had returned its path and
    referenced that USD into a live stage. USD composes payloads lazily, so a read
    landing in the delete-then-rename window failed, or composed the robot without
    its meshes, in a process whose own conversion was entirely correct - a
    nondeterministic stage-load error attributable to nobody.

    Both converters produce identical content for a given key, so the winner's
    entry is always the right answer and losing the race costs only the staging
    directory. ``os.rename`` onto an existing non-empty directory is refused by
    the OS (``ENOTEMPTY``, or ``FileExistsError`` on Windows), which is what makes
    that check atomic rather than a test-then-act.

    A torn ``target_root`` - a directory with no usable marker, left by a crash or
    by an older build that wrote the marker after renaming - would otherwise make
    the entry permanently uninstallable, since nothing deletes it any more. It is
    moved aside to a pid-unique quarantine path with a single ``os.rename``, which
    is itself atomic: whichever process gets there first moves it, and the others
    see their rename fail and re-read the marker. The quarantined directory is then
    removed, because at that point no live reader can reach it by name.
    """
    try:
        os.rename(staging, target_root)
        return final
    except OSError:
        pass

    # Lost the race, or something is already at the target.
    winner = _read_marker(marker)
    if winner is not None:
        _remove_tree(staging)
        return winner

    # Nothing usable is there, so it is torn. Move it aside and try once more.
    quarantine = f"{target_root}.torn.{os.getpid()}"
    try:
        os.rename(target_root, quarantine)
    except OSError:
        # Another process moved it, or installed over it, in the meantime.
        winner = _read_marker(marker)
        if winner is not None:
            _remove_tree(staging)
            return winner
    else:
        _remove_tree(quarantine)

    try:
        os.rename(staging, target_root)
        return final
    except OSError:
        winner = _read_marker(marker)
        if winner is not None:
            _remove_tree(staging)
            return winner
        _remove_tree(staging)
        raise RuntimeError(
            f"converted {mjcf_path!r} but could not install the cache entry at "
            f"{target_root!r}, and no other process left a usable one there. "
            f"Refusing to return a path that may not survive."
        ) from None


def _resolve_produced(produced: object, staging: str, stem: str) -> str | None:
    """The USD file the importer actually wrote, or ``None`` if there is none.

    Prefers the importer's own return value, because the layout it writes is its
    business and has changed across releases. Falls back to a search of the
    staging tree so a release that returns ``None`` (or a directory) still works,
    and prefers a file named after the description when several are present -
    a converted asset can carry sibling payload layers.
    """
    if isinstance(produced, str) and os.path.isfile(produced):
        return produced
    candidates = [
        os.path.join(dirpath, filename)
        for dirpath, _dirnames, filenames in os.walk(staging)
        for filename in sorted(filenames)
        if os.path.splitext(filename)[1].lower() in USD_EXTENSIONS
    ]
    if not candidates:
        return None
    for candidate in candidates:
        if os.path.splitext(os.path.basename(candidate))[0] == stem:
            return candidate
    return candidates[0]


def _remove_tree(path: str) -> None:
    """Delete *path* if present, tolerating its absence.

    A local helper rather than a bare ``shutil.rmtree(..., ignore_errors=True)``
    so a failure to clean a staging directory cannot be mistaken for a
    conversion failure, and so the ``ignore_errors`` blanket is not applied to
    the real cache entry.
    """
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            # Best-effort, and now uniformly so: no caller removes a *completed*
            # cache entry any more (:func:`_install_entry` renames rather than
            # deletes), so every path through here passes either a staging
            # directory or a quarantined torn one. A failed unlink on either
            # leaks a temp path instead of corrupting the cache, which is the
            # distinction this helper exists to keep - and the install's own
            # rename reports separately if it could not publish.
            pass


@functools.lru_cache(maxsize=32)
def _motor_joints_cached(mjcf_path: str, mtime: float) -> dict[str, tuple[float, float | None, float | None]]:
    del mtime  # part of the cache key only: an edited file is read again
    try:
        import mujoco
    except ImportError:
        return {}
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    except (ValueError, OSError, RuntimeError):
        return {}
    out: dict[str, tuple[float, float | None, float | None]] = {}
    for i in range(model.nu):
        if int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_gaintype[i]) != int(mujoco.mjtGain.mjGAIN_FIXED):
            continue
        if int(model.actuator_biastype[i]) != int(mujoco.mjtBias.mjBIAS_NONE):
            continue
        if int(model.actuator_dyntype[i]) != int(mujoco.mjtDyn.mjDYN_NONE):
            continue
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i][0]))
        if not joint:
            continue
        gear = float(model.actuator_gear[i][0]) * float(model.actuator_gainprm[i][0])
        limited = bool(model.actuator_ctrllimited[i])
        lo, hi = (float(v) for v in model.actuator_ctrlrange[i]) if limited else (None, None)
        out[joint] = (gear, lo, hi)
    return out


def mjcf_motor_joints(mjcf_path: str | None) -> dict[str, tuple[float, float | None, float | None]]:
    """``{joint: (torque per unit ctrl, ctrl_lo, ctrl_hi)}`` for every MJCF ``<motor>`` actuator.

    A ``<motor>`` (fixed gain, no bias, no dynamics, on one joint) makes its
    ``ctrl`` a torque, ``gear * gain * ctrl``, clipped to ``ctrlrange`` when the
    actuator is ``ctrllimited``. The Isaac MJCF importer turns it into a PhysX
    FORCE drive with ``stiffness=0, damping=0``, so a joint position target sent
    to it moves nothing - every menagerie quadruped and humanoid (go2, unitree_h1,
    openarm, ...) is one. The engine reads this table to apply such an action as
    a joint effort instead. ``{}`` when MuJoCo is not importable or the file does
    not compile.
    """
    if not mjcf_path or not os.path.isfile(mjcf_path):
        return {}
    return _motor_joints_cached(os.path.abspath(mjcf_path), os.path.getmtime(mjcf_path))


@functools.lru_cache(maxsize=8)
def _compiled_model_cached(mjcf_path: str, mtime: float) -> Any:
    del mtime  # part of the cache key only
    import mujoco

    return mujoco.MjModel.from_xml_path(mjcf_path)


def compiled_mjcf_model(mjcf_path: str | None) -> Any:
    """The robot's source MJCF compiled by MuJoCo, cached by path and mtime; ``None`` when unavailable.

    The converted USD drops the actuator table (ranges, gains, gears). What
    reads it on Isaac - a policy's ``set_sim_context`` - reads it from here.
    """
    if not mjcf_path or not os.path.isfile(mjcf_path):
        return None
    try:
        return _compiled_model_cached(os.path.abspath(mjcf_path), os.path.getmtime(mjcf_path))
    except (ImportError, ValueError, OSError, RuntimeError):
        return None
