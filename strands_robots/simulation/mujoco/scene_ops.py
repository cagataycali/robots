"""XML round-trip injection/ejection for scene modification.

Shared helper `_reload_scene_from_xml` handles the common pattern:
save XML → patch paths → modify → reload → copy state → re-discover joints.
"""

import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import Any

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.mujoco.mjcf_builder import MJCFBuilder, _camera_xyaxes_from_target, _sanitize_name

logger = logging.getLogger(__name__)


def _patch_xml_paths(xml_content: str, robot_base_dir: str) -> str:
    """Patch meshdir/texturedir in XML to absolute paths for tmpdir loading.

    Uses ElementTree for consistent XML manipulation throughout scene_ops.
    Falls back to the original string if ET parsing fails (e.g. XML fragments).
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        # Fallback for malformed fragments — use regex as last resort
        logger.debug("ET parse failed for _patch_xml_paths, using regex fallback")
        meshdir_match = re.search(r'meshdir="([^"]*)"', xml_content)
        if meshdir_match:
            abs_meshdir = os.path.normpath(os.path.join(robot_base_dir, meshdir_match.group(1)))
            xml_content = re.sub(r'meshdir="[^"]*"', f'meshdir="{abs_meshdir}"', xml_content)
        texdir_match = re.search(r'texturedir="([^"]*)"', xml_content)
        if texdir_match:
            abs_texdir = os.path.normpath(os.path.join(robot_base_dir, texdir_match.group(1)))
            xml_content = re.sub(r'texturedir="[^"]*"', f'texturedir="{abs_texdir}"', xml_content)
        return xml_content

    compiler = root.find("compiler")
    if compiler is None:
        # No compiler element — add one with meshdir
        compiler = ET.SubElement(root, "compiler")
        # Insert at beginning (after root tag)
        root.remove(compiler)
        root.insert(0, compiler)

    existing_meshdir = compiler.get("meshdir", "")
    compiler.set("meshdir", os.path.normpath(os.path.join(robot_base_dir, existing_meshdir)))

    existing_texdir = compiler.get("texturedir", "")
    if existing_texdir or compiler.get("texturedir") is not None:
        compiler.set("texturedir", os.path.normpath(os.path.join(robot_base_dir, existing_texdir)))
    else:
        compiler.set("texturedir", robot_base_dir)

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def _get_abs_meshdir(root: ET.Element) -> str:
    """Extract the absolute meshdir from a parsed XML root.

    Returns empty string if no compiler/meshdir is set.
    """
    compiler = root.find("compiler")
    if compiler is not None:
        return compiler.get("meshdir", "")
    return ""


def _rewrite_mesh_paths(
    robot_asset: ET.Element,
    robot_meshdir: str,
    scene_meshdir: str,
) -> None:
    """Rewrite mesh ``file=`` attributes so they resolve under scene_meshdir.

    When merging robot assets into the scene XML, the scene's ``<compiler
    meshdir="...">`` governs where MuJoCo looks for mesh files.  If the
    robot's meshdir differs (e.g. ``robot_base/assets/`` vs ``robot_base/``),
    each ``<mesh file="X.stl">`` must be adjusted to be correct relative to
    the scene's meshdir.

    Strategy: convert each mesh file to an absolute path (via robot_meshdir),
    then make it relative to scene_meshdir.  If they share no common prefix,
    fall back to absolute paths.
    """
    if not robot_meshdir or not scene_meshdir:
        return
    # Normalize: ensure trailing sep for consistent joining
    robot_meshdir = os.path.normpath(robot_meshdir)
    scene_meshdir = os.path.normpath(scene_meshdir)

    if robot_meshdir == scene_meshdir:
        return  # No rewriting needed — meshdirs match

    for child in robot_asset:
        if child.tag != "mesh":
            continue
        file_attr = child.get("file")
        if not file_attr:
            continue
        # Build absolute path of the mesh file under robot's meshdir
        abs_mesh = os.path.normpath(os.path.join(robot_meshdir, file_attr))
        # Make it relative to the scene's meshdir
        try:
            rel_path = os.path.relpath(abs_mesh, scene_meshdir)
        except ValueError:
            # On Windows, relpath fails across drives — use absolute
            rel_path = abs_mesh
        child.set("file", rel_path)

    # Also rewrite texture file paths that reference files on disk
    for child in robot_asset:
        if child.tag != "texture":
            continue
        file_attr = child.get("file")
        if not file_attr:
            continue
        abs_tex = os.path.normpath(os.path.join(robot_meshdir, file_attr))
        try:
            rel_path = os.path.relpath(abs_tex, scene_meshdir)
        except ValueError:
            rel_path = abs_tex
        child.set("file", rel_path)


def _reload_scene_from_xml(world: SimWorld, scene_path: str) -> bool:
    """Reload MuJoCo model from modified XML, preserving state.

    Copies qpos, qvel, ctrl from old model and re-discovers robot joint/actuator IDs.

    before copying existing state into the new MjData we explicitly call
    ``mj_resetData`` so that joints NOT present in ``old_model`` (i.e. the
    freshly-injected robot's joints) start from a well-defined zero state
    rather than whatever garbage pybind11 happened to hand us from fresh
    allocation. Old state is then layered on top per-joint-by-name so
    previously-existing robots/objects keep their positions.
    """
    mj = _ensure_mujoco()
    new_model = mj.MjModel.from_xml_path(str(scene_path))
    new_data = mj.MjData(new_model)

    # zero the whole state buffer before copying old-state on top.
    # Without this, freshly-added robots show nonzero qpos/qvel/ctrl from
    # uninitialised memory and any observation taken before reset() is garbage.
    mj.mj_resetData(new_model, new_data)

    # Copy state per-joint by name to handle layout shifts when injected
    # bodies land earlier in the body-tree traversal.  Flat-index copies
    # (qpos[:old_nq]) are unsafe because MuJoCo allocates qpos in
    # recursive body-tree order — a new body can shift existing entries.
    old_model = world._model
    old_data = world._data
    for i in range(old_model.njnt):
        jnt_name = mj.mj_id2name(old_model, mj.mjtObj.mjOBJ_JOINT, i)
        if not jnt_name:
            continue
        new_jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
        if new_jid < 0:
            continue  # joint removed from scene
        # Defensive: skip copy if joint type changed (extremely unlikely in
        # inject/eject flow, but prevents stride mismatch → silent corruption).
        if old_model.jnt_type[i] != new_model.jnt_type[new_jid]:
            continue
        # qpos: width depends on joint type (free=7, ball=4, hinge/slide=1)
        jnt_type = old_model.jnt_type[i]
        qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}.get(int(jnt_type), 1)
        old_adr = old_model.jnt_qposadr[i]
        new_adr = new_model.jnt_qposadr[new_jid]
        new_data.qpos[new_adr : new_adr + qpos_width] = old_data.qpos[old_adr : old_adr + qpos_width]
        # qvel: width = joint DoF (free=6, ball=3, hinge/slide=1)
        dof_width = {0: 6, 1: 3, 2: 1, 3: 1}.get(int(jnt_type), 1)
        old_dof = old_model.jnt_dofadr[i]
        new_dof = new_model.jnt_dofadr[new_jid]
        new_data.qvel[new_dof : new_dof + dof_width] = old_data.qvel[old_dof : old_dof + dof_width]

    # Copy ctrl per-actuator by name (actuator order may also shift)
    for i in range(old_model.nu):
        act_name = mj.mj_id2name(old_model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if not act_name:
            continue
        new_aid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)
        if new_aid >= 0:
            new_data.ctrl[new_aid] = old_data.ctrl[i]

    mj.mj_forward(new_model, new_data)

    world._model = new_model
    world._data = new_data

    # Persist the current scene XML so subsequent mj_saveLastXML calls can
    # reset the MuJoCo global state. Without this, any render/renderer
    # creation poisons mj_saveLastXML for inject/eject round-trips.
    try:
        with open(scene_path) as _f:
            world._backend_state["xml"] = _f.read()
    except OSError:
        # Best-effort — don't fail the reload just because we can't read back.
        pass

    # Re-discover robot joints/actuators (IDs may shift).
    # Try namespaced name first (multi-robot case), fall back to raw.
    for robot in world.robots.values():
        robot.joint_ids = []
        robot.actuator_ids = []
        pfx = robot.namespace or ""
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        for i in range(new_model.nu):
            jnt_id = new_model.actuator_trnid[i, 0]
            if jnt_id in robot.joint_ids:
                robot.actuator_ids.append(i)
        if not robot.actuator_ids:
            # Last-resort fallback: all actuators (single-robot scenes).
            if len(world.robots) == 1:
                for i in range(new_model.nu):
                    robot.actuator_ids.append(i)

    return True


def _get_robot_base_dir(world: SimWorld) -> str | None:
    """Get the directory of the first robot model file.

    For multi-robot scenes with different asset directories, use
    ``_get_all_robot_base_dirs()`` instead.
    """
    if world._backend_state.get("robot_base_xml", ""):
        return os.path.dirname(os.path.abspath(world._backend_state.get("robot_base_xml", "")))
    return None


def _get_all_robot_base_dirs(world: SimWorld) -> list[str]:
    """Return a deduplicated list of directories containing robot model files.

    Each robot's ``urdf_path`` points to its MJCF/URDF source.  The directory
    of each path may contain mesh assets that the scene XML references.
    """
    dirs: list[str] = []
    seen: set[str] = set()
    for robot in world.robots.values():
        d = os.path.dirname(os.path.abspath(robot.urdf_path))
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    # Also include the legacy single-robot path if set.
    legacy = _get_robot_base_dir(world)
    if legacy and legacy not in seen:
        dirs.append(legacy)
    return dirs


def _save_and_patch_xml(world: SimWorld, tmpdir: str, filename: str) -> str:
    """Save current model to XML in tmpdir and patch asset paths.

    Note: MuJoCo's ``mj_saveLastXML`` is a global function that always
    writes the *last loaded* model's XML, ignoring the ``model`` argument.
    Any renderer creation (``mj.Renderer``) or ancillary model load between
    our last scene compile and this save will poison the global → we get
    some *other* model's XML and the inject/eject XML round-trip fails
    silently (e.g. "Body 'cube' not found in MJCF XML").

    To work around this, we first reload our own stored scene XML into the
    MuJoCo global state (via ``MjModel.from_xml_string``). The resulting
    ``_tmp`` model is discarded — its only purpose is to reset
    ``mj_saveLastXML``'s internal pointer.

    Multi-robot note: uses the first robot's base dir for compiler paths.
    Individual robot mesh paths are rewritten to absolute during
    inject_robot_into_scene (via _rewrite_mesh_paths), so the scene-level
    meshdir only needs to resolve for the primary robot. Future enhancement:
    convert all mesh paths to absolute during injection to eliminate
    first-wins coupling entirely.
    """
    mj = _ensure_mujoco()
    scene_path = os.path.join(tmpdir, filename)

    stored_xml = world._backend_state.get("xml")
    if stored_xml:
        _tmp = mj.MjModel.from_xml_string(stored_xml)  # noqa: F841
        mj.mj_saveLastXML(scene_path, _tmp)
    else:
        mj.mj_saveLastXML(scene_path, world._model)

    robot_base_dir = _get_robot_base_dir(world)
    if robot_base_dir and os.path.isdir(robot_base_dir):
        with open(scene_path) as f:
            xml_content = f.read()
        xml_content = _patch_xml_paths(xml_content, robot_base_dir)
        with open(scene_path, "w") as f:
            f.write(xml_content)

    return scene_path


def _prefix_robot_names(robot_root: Any, prefix: str) -> None:
    """Prefix every named element and reference in a robot MJCF so that
    multiple robots with the same ``data_config`` can coexist in one scene.

    Without this, two ``so101`` robots share body names (``base``, ``gripper``,
    ...), joint names (``shoulder_pan``, ...), actuator names, etc. MuJoCo
    requires all top-level names to be globally unique and rejects the merged
    XML with ``"repeated name 'base' in body"``.

    The prefix is applied in-place to:
      - element ``name`` attributes (bodies, joints, actuators, sites, geoms,
        sensors, tendons, equality constraints, keyframes)
      - reference attributes that point *into* the robot namespace:
        ``joint``, ``body``, ``site``, ``geom``, ``tendon``, ``actuator``,
        ``body1``, ``body2``, ``joint1``, ``joint2``

    Asset references (mesh, material, texture, hfield) and class references
    are NOT prefixed — they are shared by same-config robots (which is the
    whole point of the dedupe in assets/defaults).

    Args:
        robot_root: The parsed ``<mujoco>`` root of the robot XML.
        prefix: The robot instance name, used as a namespace prefix.
    """
    pfx = f"{prefix}/"

    # Tags whose "name" attribute identifies a unique element in the merged
    # scene. Each instance must get prefixed.
    _NAMED_TAGS = {
        "body",
        "joint",
        "geom",
        "site",
        "camera",
        "light",
        "actuator",
        "general",
        "motor",
        "position",
        "velocity",
        "sensor",
        "force",
        "torque",
        "jointpos",
        "jointvel",
        "framepos",
        "framequat",
        "frameangvel",
        "framelinvel",
        "framelinacc",
        "frameangacc",
        "accelerometer",
        "gyro",
        "magnetometer",
        "rangefinder",
        "touch",
        "subtreecom",
        "subtreelinvel",
        "subtreeangmom",
        "velocimeter",
        "user",
        "tendon",
        "fixed",
        "spatial",
        "equality",
        "connect",
        "weld",
        "joint_equality",
        "tendon_equality",
        "key",  # keyframes
    }

    # Attributes that reference named elements (in the robot namespace).
    _REF_ATTRS = {
        "joint",
        "body",
        "site",
        "geom",
        "tendon",
        "actuator",
        "body1",
        "body2",
        "joint1",
        "joint2",
        "childclass",  # default classes — prefixed too since we keep per-robot ones? No — keep shared.
        "target",
    }
    # We don't prefix "childclass" because classes are shared (deduped) across
    # same-config robots. Remove it from the set.
    _REF_ATTRS.discard("childclass")

    def visit(elem: Any) -> None:
        # Rename ``name`` attribute if this tag is in the named set.
        if elem.tag in _NAMED_TAGS:
            orig = elem.get("name", "")
            if orig and not orig.startswith(pfx):
                elem.set("name", pfx + orig)

        # Rewrite reference attributes (they point to robot-local elements).
        for attr in _REF_ATTRS:
            val = elem.get(attr)
            if val and not val.startswith(pfx):
                elem.set(attr, pfx + val)

        for child in elem:
            visit(child)

    # We only want to prefix elements inside:
    #  - worldbody (bodies, their children)
    #  - actuator
    #  - sensor
    #  - equality
    #  - tendon
    #  - keyframe
    # We do NOT prefix contents of <default>, <asset>, <compiler>, <option>
    # because these are shared across same-config robot instances.
    for section in ("worldbody", "actuator", "sensor", "equality", "tendon", "keyframe", "contact"):
        sec = robot_root.find(section)
        if sec is not None:
            for child in sec:
                visit(child)


def _collect_existing_class_names(scene_default: Any | None) -> set[str]:
    """Walk a <default> subtree and return every ``class="X"`` ever declared."""
    names: set[str] = set()
    if scene_default is None:
        return names
    stack = list(scene_default)
    while stack:
        node = stack.pop()
        cls = node.get("class", "")
        if cls:
            names.add(cls)
        stack.extend(list(node))
    return names


def _namespace_robot_default_classes(robot_root: Any, namespace: str, skip: set[str]) -> dict[str, str]:
    """Rename ``<default class="X">`` blocks to ``<default class="{namespace}__X">``.

    MuJoCo flattens all nested ``<default class="X">`` names into a single
    global namespace at compile time. Two robots that each declare a nested
    class named ``visual`` (common in MuJoCo Menagerie models) collide with
    ``"repeated default class name"`` even though they live in different
    parent ``<default>`` blocks in the source XML.

    This helper renames every class declared in the robot's ``<default>``
    tree to a namespaced form, EXCEPT for classes listed in ``skip`` (names
    that already exist in the merged scene from a robot sharing the same
    ``data_config`` — those we want to reuse, not duplicate).

    It then rewrites every ``class=`` and ``childclass=`` attribute in the
    robot's other sections (``worldbody``, ``actuator``, ``sensor``, etc.)
    so the references still resolve to the renamed classes.

    Args:
        robot_root: The <mujoco> root of the robot's canonical MJCF.
        namespace: A prefix unique to this robot's ``data_config`` — typically
            the data_config key itself (e.g. ``"h1"`` or ``"so100"``).
        skip: Class names that already exist in the scene (leave them alone).

    Returns:
        Mapping from old → new class names (only for classes we renamed).
    """
    robot_default = robot_root.find("default")
    if robot_default is None:
        return {}

    mapping: dict[str, str] = {}
    stack = list(robot_default)
    while stack:
        node = stack.pop()
        cls = node.get("class", "")
        if cls and cls not in skip and cls not in mapping:
            mapping[cls] = f"{namespace}__{cls}"
        stack.extend(list(node))

    if not mapping:
        return {}

    # Apply the rename everywhere in the robot tree: <default class=..>, and
    # class=/childclass= on body/geom/joint/site/camera/... references.
    def rewrite(elem: Any) -> None:
        for attr in ("class", "childclass"):
            v = elem.get(attr)
            if v and v in mapping:
                elem.set(attr, mapping[v])
        for child in elem:
            rewrite(child)

    rewrite(robot_root)
    return mapping


def inject_robot_into_scene(
    world: SimWorld,
    robot: SimRobot,
    robot_xml_path: str,
) -> bool:
    """Inject a robot into a running simulation via XML round-trip.

    Loads the robot XML, extracts its bodies/actuators/assets/sensors, and
    merges them into the existing world scene XML.  This preserves all
    existing world state (gravity, objects, cameras, other robots).

    The approach:
    1. Save current world model to XML.
    2. Load the robot XML into a *temporary* MjModel just to get its
       canonical MJCF (handles URDF→MJCF conversion).
    3. Parse both XMLs with ElementTree.
    4. Merge robot assets, worldbody children, actuators, and sensors
       into the world XML.  Mesh ``file=`` paths are rewritten so they
       resolve correctly under the scene's ``meshdir``.
    5. Reload the combined scene and re-discover joint/actuator IDs.

    Note: MuJoCo's ``mj_saveLastXML`` is a global function that always
    saves the XML from the most recently loaded model, regardless of which
    ``MjModel`` is passed.  We must therefore convert the robot FIRST
    (step 2), then reload the world model to reset the global state before
    saving the scene XML (step 1).
    """
    mj = _ensure_mujoco()
    if world._model is None:
        return False

    tmpdir = tempfile.mkdtemp(prefix="strands_robot_inject_")
    try:
        # Step 2 (done first): Convert robot file to canonical MJCF via
        # MuJoCo round-trip.  We do this *before* saving the scene because
        # mj_saveLastXML is a global that always emits the last-loaded XML.
        robot_model = mj.MjModel.from_xml_path(str(robot_xml_path))
        robot_mjcf_path = os.path.join(tmpdir, f"robot_{_sanitize_name(robot.name)}.xml")
        mj.mj_saveLastXML(robot_mjcf_path, robot_model)

        # Step 1: Save the current world scene to XML.
        # Re-derive the scene XML from the stored backend XML string so
        # that mj_saveLastXML emits the *scene* (not the robot we just
        # loaded above).
        stored_xml = world._backend_state.get("xml")
        if stored_xml:
            # Reload from stored XML to reset mj_saveLastXML global state,
            # then save.  The intermediate model is discarded.
            _tmp = mj.MjModel.from_xml_string(stored_xml)  # noqa: F841
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_with_robot.xml")

        # Patch robot MJCF asset paths to absolute
        robot_base_dir = os.path.dirname(os.path.abspath(robot_xml_path))
        with open(robot_mjcf_path) as f:
            robot_xml_content = f.read()
        robot_xml_content = _patch_xml_paths(robot_xml_content, robot_base_dir)
        with open(robot_mjcf_path, "w") as f:
            f.write(robot_xml_content)

        # Step 3: Parse both XMLs
        scene_tree = ET.parse(scene_path)
        scene_root = scene_tree.getroot()
        robot_root = ET.fromstring(robot_xml_content)

        # Step 3a: Prefix all names/references inside the robot XML with the
        # robot's instance name. Required so that multiple robots with the
        # same ``data_config`` (e.g. three so101s) can coexist — otherwise
        # MuJoCo rejects the merged XML with "repeated name 'base' in body".
        _prefix_robot_names(robot_root, robot.name)

        scene_worldbody = scene_root.find("worldbody")
        robot_worldbody = robot_root.find("worldbody")
        if scene_worldbody is None or robot_worldbody is None:
            logger.error("Missing <worldbody> in scene or robot XML")
            return False

        # Step 4a: Merge assets (meshes, textures, materials)
        # Robot and scene may have different meshdirs (e.g. robot uses
        # meshdir="<base>/assets/" while scene uses meshdir="<base>/").
        # Rewrite robot mesh file= attributes so they resolve under
        # the scene's meshdir.
        scene_asset = scene_root.find("asset")
        robot_asset = robot_root.find("asset")

        robot_meshdir = _get_abs_meshdir(robot_root)

        if robot_asset is not None:
            # Rewrite mesh/texture file= paths to absolute before merging.
            # This eliminates the first-wins coupling: each robot's assets
            # resolve independently regardless of scene-level meshdir.
            if robot_meshdir:
                for child in robot_asset:
                    if child.tag in ("mesh", "texture"):
                        file_attr = child.get("file")
                        if file_attr and not os.path.isabs(file_attr):
                            child.set("file", os.path.normpath(os.path.join(robot_meshdir, file_attr)))
            # NOTE: The elif was unreachable (robot_meshdir is falsy in else
            # branch, making `scene_meshdir and robot_meshdir` always False).
            # Absolutizing file= attrs above handles all cases correctly.

            if scene_asset is None:
                scene_asset = ET.SubElement(scene_root, "asset")
            # Collect existing asset names to avoid duplicates
            existing_assets: set[str] = set()
            for child in scene_asset:
                name = child.get("name", "")
                if name:
                    existing_assets.add(name)
            for child in robot_asset:
                name = child.get("name", "")
                if name and name not in existing_assets:
                    scene_asset.append(child)
                    existing_assets.add(name)
                elif not name:
                    # Unnamed assets (rare) — append unconditionally
                    scene_asset.append(child)

        # Step 4b: Merge worldbody children (robot bodies, lights, etc.)
        # Skip ground planes and lights from robot XML to avoid duplicates
        _SKIP_GROUND_TYPES = {"plane"}
        for child in robot_worldbody:
            if child.tag == "geom" and child.get("type") in _SKIP_GROUND_TYPES:
                continue  # Skip duplicate ground planes
            if child.tag == "light":
                continue  # Skip duplicate lights
            scene_worldbody.append(child)

        # Step 4c: Merge actuators (dedupe by name — multiple same-config
        # robots would clash on e.g. "shoulder_pan" actuator).
        scene_actuator = scene_root.find("actuator")
        robot_actuator = robot_root.find("actuator")
        if robot_actuator is not None:
            if scene_actuator is None:
                scene_actuator = ET.SubElement(scene_root, "actuator")
            existing_actuators: set[str] = {c.get("name", "") for c in scene_actuator if c.get("name")}
            for child in robot_actuator:
                n = child.get("name", "")
                if n and n in existing_actuators:
                    continue
                scene_actuator.append(child)
                if n:
                    existing_actuators.add(n)

        # Step 4d: Merge sensors (dedupe by name)
        scene_sensor = scene_root.find("sensor")
        robot_sensor = robot_root.find("sensor")
        if robot_sensor is not None:
            if scene_sensor is None:
                scene_sensor = ET.SubElement(scene_root, "sensor")
            existing_sensors: set[str] = {c.get("name", "") for c in scene_sensor if c.get("name")}
            for child in robot_sensor:
                n = child.get("name", "")
                if n and n in existing_sensors:
                    continue
                scene_sensor.append(child)
                if n:
                    existing_sensors.add(n)

        # Step 4e: Merge default classes.
        # - Robots that share a data_config reuse the same classes (dedupe).
        # - Robots with DIFFERENT data_configs often declare colliding class
        #   names (e.g. every MuJoCo Menagerie model has its own nested
        #   ``<default class="visual">``). Namespace those classes per
        #   data_config so both can coexist.
        scene_default = scene_root.find("default")
        robot_default = robot_root.find("default")

        merged_configs = world._backend_state.setdefault("merged_configs", set())
        robot_cfg = robot.data_config or robot.name
        if robot_default is not None and robot_cfg not in merged_configs:
            existing_class_names = _collect_existing_class_names(scene_default)
            _namespace_robot_default_classes(robot_root, robot_cfg, existing_class_names)
            # Re-fetch after in-place rewrite.
            robot_default = robot_root.find("default")
            merged_configs.add(robot_cfg)
        elif robot_cfg in merged_configs:
            # Same config already merged — drop this robot's <default> entirely,
            # and rewrite class/childclass on its bodies to point at the
            # already-merged, already-namespaced classes so references resolve.
            if robot_default is not None:
                for node in list(robot_default):
                    pass  # no-op; we'll strip robot_default below

            # Walk once to rewrite references using the existing scheme:
            # classes were namespaced as "{cfg}__{origname}" the first time.
            def _rewrite_refs(elem: Any) -> None:
                for attr in ("class", "childclass"):
                    v = elem.get(attr)
                    if v and "__" not in v:
                        elem.set(attr, f"{robot_cfg}__{v}")
                for child in elem:
                    _rewrite_refs(child)

            _rewrite_refs(robot_root)
            # Zero out robot_default so the merge below is a no-op.
            robot_default = None

        if robot_default is not None:
            if scene_default is None:
                scene_default = ET.SubElement(scene_root, "default")
                # Insert after compiler/option
                scene_root.remove(scene_default)
                insert_idx = 0
                for i, child in enumerate(scene_root):
                    if child.tag in ("compiler", "option", "size"):
                        insert_idx = i + 1
                scene_root.insert(insert_idx, scene_default)

            existing_classes: set[str] = set()
            for child in scene_default:
                cls = child.get("class", "")
                if cls:
                    existing_classes.add(cls)
                elif child.tag == "default":
                    # MJCF nested default blocks use <default class="name">
                    nested_cls = child.get("class", "") or ""
                    if nested_cls:
                        existing_classes.add(nested_cls)
            for child in robot_default:
                cls = child.get("class", "")
                if cls and cls in existing_classes:
                    continue  # already merged from a previous same-config robot
                scene_default.append(child)
                if cls:
                    existing_classes.add(cls)

        # Step 4f: Merge equality constraints
        scene_equality = scene_root.find("equality")
        robot_equality = robot_root.find("equality")
        if robot_equality is not None:
            if scene_equality is None:
                scene_equality = ET.SubElement(scene_root, "equality")
            for child in robot_equality:
                scene_equality.append(child)

        # Step 4g: Merge tendon elements
        scene_tendon = scene_root.find("tendon")
        robot_tendon = robot_root.find("tendon")
        if robot_tendon is not None:
            if scene_tendon is None:
                scene_tendon = ET.SubElement(scene_root, "tendon")
            for child in robot_tendon:
                scene_tendon.append(child)

        # Remove keyframes — adding joints changes qpos size
        for keyframe_elem in scene_root.findall("keyframe"):
            scene_root.remove(keyframe_elem)

        # Step 5: Write merged XML and reload
        scene_tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)

    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Robot injection failed for '%s': %s", robot.name, e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def inject_object_into_scene(world: SimWorld, obj: SimObject) -> bool:
    """Inject object into a running simulation via XML round-trip.

    Uses ElementTree for XML manipulation (consistent with eject_body_from_scene).
    """
    _ensure_mujoco()
    if world._model is None:
        return False

    tmpdir = tempfile.mkdtemp(prefix="strands_sim_")
    try:
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_with_objects.xml")

        tree = ET.parse(scene_path)
        root = tree.getroot()

        # Find <worldbody> and append the object element
        worldbody = root.find("worldbody")
        if worldbody is None:
            logger.error("No <worldbody> found in scene XML")
            return False

        obj_xml_str = MJCFBuilder._object_xml(obj, indent=4)
        obj_elem = ET.fromstring(f"<_wrapper>{obj_xml_str}</_wrapper>")
        for child in obj_elem:
            worldbody.append(child)

        # Remove keyframes — adding a freejoint changes qpos size
        for keyframe_elem in root.findall("keyframe"):
            root.remove(keyframe_elem)

        tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Object injection reload failed: %s", e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def eject_body_from_scene(world: SimWorld, body_name: str) -> bool:
    """Remove a named body from the scene via XML round-trip."""
    tmpdir = tempfile.mkdtemp(prefix="strands_eject_")
    try:
        # Use helper so we honour the mj_saveLastXML global-state workaround.
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_ejected.xml")

        tree = ET.parse(scene_path)
        root = tree.getroot()

        # Patch paths
        robot_base_dir = _get_robot_base_dir(world)
        if robot_base_dir:
            compiler = root.find("compiler")
            if compiler is not None:
                existing_meshdir = compiler.get("meshdir", "")
                compiler.set("meshdir", os.path.normpath(os.path.join(robot_base_dir, existing_meshdir)))
                existing_texdir = compiler.get("texturedir", "")
                compiler.set("texturedir", os.path.normpath(os.path.join(robot_base_dir, existing_texdir)))

        # Remove target body
        removed = False
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "body" and child.get("name") == body_name:
                    parent.remove(child)
                    removed = True

        if not removed:
            logger.warning(f"Body '{body_name}' not found in MJCF XML — skipping ejection.")

        # Remove keyframes
        for keyframe_elem in root.findall("keyframe"):
            root.remove(keyframe_elem)

        tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Body ejection failed for '%s': %s", body_name, e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Inject a camera into a running simulation via XML round-trip.

    Uses ElementTree for XML manipulation (consistent with eject_body_from_scene).
    """
    _ensure_mujoco()
    if world._model is None:
        return False

    tmpdir = tempfile.mkdtemp(prefix="strands_cam_")
    try:
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_with_cameras.xml")

        tree = ET.parse(scene_path)
        root = tree.getroot()

        worldbody = root.find("worldbody")
        if worldbody is None:
            logger.error("No <worldbody> found in scene XML")
            return False

        px, py, pz = cam.position
        cam_elem = ET.SubElement(worldbody, "camera")
        cam_elem.set("name", _sanitize_name(cam.name))
        cam_elem.set("pos", f"{px} {py} {pz}")
        cam_elem.set("fovy", str(cam.fov))
        cam_elem.set("mode", "fixed")
        # write xyaxes so the camera actually LOOKS at cam.target.
        # Without this the `target` parameter is cosmetic and all custom
        # cameras share the MuJoCo default orientation -> identical frames.
        target = getattr(cam, "target", None)
        if target:
            xyaxes = _camera_xyaxes_from_target(cam.position, target)
            if xyaxes:
                cam_elem.set("xyaxes", xyaxes)
            else:
                # Degenerate (target == position): leave unoriented but log.
                logger.warning(
                    "inject_camera: camera '%s' has target == position; xyaxes not emitted",
                    cam.name,
                )

        tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Camera injection reload failed: %s", e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
