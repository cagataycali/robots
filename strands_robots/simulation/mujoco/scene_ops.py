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

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.mujoco.mjcf_builder import MJCFBuilder, _sanitize_name

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
    """
    mj = _ensure_mujoco()
    new_model = mj.MjModel.from_xml_path(str(scene_path))
    new_data = mj.MjData(new_model)

    # Copy state from old model
    old_nq = min(world._data.qpos.shape[0], new_data.qpos.shape[0])
    old_nv = min(world._data.qvel.shape[0], new_data.qvel.shape[0])
    new_data.qpos[:old_nq] = world._data.qpos[:old_nq]
    new_data.qvel[:old_nv] = world._data.qvel[:old_nv]
    old_nu = min(world._data.ctrl.shape[0], new_data.ctrl.shape[0])
    new_data.ctrl[:old_nu] = world._data.ctrl[:old_nu]

    mj.mj_forward(new_model, new_data)

    world._model = new_model
    world._data = new_data

    # Re-discover robot joints/actuators (IDs may shift)
    for robot in world.robots.values():
        robot.joint_ids = []
        robot.actuator_ids = []
        for jnt_name in robot.joint_names:
            jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        for i in range(new_model.nu):
            jnt_id = new_model.actuator_trnid[i, 0]
            if jnt_id in robot.joint_ids:
                robot.actuator_ids.append(i)
        if not robot.actuator_ids:
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
    """Save current model to XML in tmpdir and patch asset paths."""
    mj = _ensure_mujoco()
    scene_path = os.path.join(tmpdir, filename)
    mj.mj_saveLastXML(scene_path, world._model)

    robot_base_dir = _get_robot_base_dir(world)
    if robot_base_dir and os.path.isdir(robot_base_dir):
        with open(scene_path) as f:
            xml_content = f.read()
        xml_content = _patch_xml_paths(xml_content, robot_base_dir)
        with open(scene_path, "w") as f:
            f.write(xml_content)

    return scene_path


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

        scene_meshdir = _get_abs_meshdir(scene_root)
        robot_meshdir = _get_abs_meshdir(robot_root)

        if robot_asset is not None:
            # Rewrite mesh/texture file= paths before merging
            if scene_meshdir and robot_meshdir:
                _rewrite_mesh_paths(robot_asset, robot_meshdir, scene_meshdir)

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

        # Step 4c: Merge actuators
        scene_actuator = scene_root.find("actuator")
        robot_actuator = robot_root.find("actuator")
        if robot_actuator is not None:
            if scene_actuator is None:
                scene_actuator = ET.SubElement(scene_root, "actuator")
            for child in robot_actuator:
                scene_actuator.append(child)

        # Step 4d: Merge sensors
        scene_sensor = scene_root.find("sensor")
        robot_sensor = robot_root.find("sensor")
        if robot_sensor is not None:
            if scene_sensor is None:
                scene_sensor = ET.SubElement(scene_root, "sensor")
            for child in robot_sensor:
                scene_sensor.append(child)

        # Step 4e: Merge default classes
        scene_default = scene_root.find("default")
        robot_default = robot_root.find("default")
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
            for child in robot_default:
                scene_default.append(child)

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
    mj = _ensure_mujoco()

    tmpdir = tempfile.mkdtemp(prefix="strands_eject_")
    try:
        scene_path = os.path.join(tmpdir, "scene_ejected.xml")
        mj.mj_saveLastXML(scene_path, world._model)

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

        tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Camera injection reload failed: %s", e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
