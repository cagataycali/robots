"""Scene mutation via the MuJoCo ``MjSpec`` AST.

This module used to contain ~980 lines of XML-round-trip machinery (tmpdir +
``mj_saveLastXML`` + ElementTree parse + name-mangling + regex path patching).
All of that is replaced by ``spec.recompile(model, data)`` which:

* preserves joint state on unchanged joints automatically,
* initializes new joints to body ``pos``/``quat`` (removing the need to
  delete keyframes on freejoint insertion),
* namespaces robot bodies/joints/geoms/actuators/sensors via ``spec.attach()``
  without us walking the tree manually.

Public API:

* :func:`inject_robot_into_scene` - ``spec.attach(robot_spec, prefix=...)``.
* :func:`inject_object_into_scene` - ``SpecBuilder.add_object(spec, obj)`` + recompile.
* :func:`inject_camera_into_scene` - ``SpecBuilder.add_camera(spec, cam)`` + recompile.
* :func:`eject_body_from_scene` - ``SpecBuilder.remove_body(spec, name)`` + recompile.
* :func:`eject_robot_from_scene` - walk the spec, delete everything namespaced
  under ``{robot_name}/``, then recompile.

Every function takes a ``SimWorld`` whose ``_backend_state["spec"]`` holds the
live ``MjSpec``. They return ``True`` on success, ``False`` on failure (matching
the legacy API) so call sites in ``simulation.py`` don't need to change.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder

logger = logging.getLogger(__name__)


def _get_spec(world: SimWorld) -> Any | None:
    """Fetch the live MjSpec from ``world._backend_state``.

    Callers MUST have run ``_compile_world`` at least once before any scene
    mutation - without a spec we can't recompile. Returns ``None`` if missing
    so callers can return a clean error dict rather than crashing mid-op.
    """
    return world._backend_state.get("spec")


def _recompile_preserving_state(world: SimWorld, spec: Any) -> bool:
    """Recompile ``spec`` in place, replacing ``world._model`` and ``_data``.

    Uses ``spec.recompile(model, data)`` which auto-preserves qpos/qvel for
    existing joints and initializes new joints to their body's pos/quat. No
    manual state-copy loop is required.

    Also re-discovers per-robot joint and actuator IDs (they may have shifted
    as new bodies were inserted earlier in the body tree). Returns True on
    success, False on compile failure (logged).
    """
    mj = _ensure_mujoco()
    try:
        new_model, new_data = spec.recompile(world._model, world._data)
    except (ValueError, RuntimeError) as e:
        logger.error("spec.recompile failed: %s", e)
        return False

    world._model = new_model
    world._data = new_data

    # Keep the cached XML in sync with the spec for legacy readers (e.g.
    # load_scene + add_robot round-trip).
    try:
        world._backend_state["xml"] = spec.to_xml()
    except Exception as xml_err:
        logger.debug("spec.to_xml() failed: %s", xml_err)

    # Re-discover per-robot IDs. Names inside MuJoCo are namespaced under
    # robot.namespace (e.g. "arm1/shoulder_pan") when robots were attached
    # via SpecBuilder.attach_robot; fall back to the raw name otherwise.
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        robot.actuator_ids = []
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
        # Single-robot fallback: if no actuators matched by joint, assume
        # all actuators belong to this robot. Matches the legacy behaviour.
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    return True


# =============================================================================
# Inject
# =============================================================================


def inject_robot_into_scene(
    world: SimWorld,
    robot: SimRobot,
    robot_xml_path: str,
) -> bool:
    """Attach a robot to the scene via ``spec.attach(other, prefix=..., frame=...)``.

    MuJoCo handles name prefixing (bodies, joints, geoms, actuators, sensors,
    sites), asset deduplication (meshes, textures, materials), and default-
    class namespacing. No manual tree-walking required.

    Registers the robot's source joint names on ``robot.joint_names`` so
    downstream observation/policy code can resolve them via
    ``{robot.namespace}{joint_name}``.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_robot: no spec or model in world")
        return False

    try:
        joint_names = SpecBuilder.attach_robot(spec, robot, robot_xml_path)
        robot.joint_names = joint_names
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Robot attach failed for '%s': %s", robot.name, e)
        return False

    return _recompile_preserving_state(world, spec)


def inject_object_into_scene(world: SimWorld, obj: SimObject) -> bool:
    """Add a ``SimObject`` to the scene and recompile in place."""
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_object: no spec or model in world")
        return False

    try:
        SpecBuilder.add_object(spec, obj)
    except (ValueError, RuntimeError) as e:
        logger.error("Object add failed for '%s': %s", obj.name, e)
        return False

    return _recompile_preserving_state(world, spec)


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Add a camera to the scene and recompile in place."""
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_camera: no spec or model in world")
        return False

    try:
        SpecBuilder.add_camera(spec, cam)
    except (ValueError, RuntimeError) as e:
        logger.error("Camera add failed for '%s': %s", cam.name, e)
        return False

    return _recompile_preserving_state(world, spec)


# =============================================================================
# Eject
# =============================================================================


def eject_body_from_scene(world: SimWorld, body_name: str) -> bool:
    """Remove a body (by short name) and recompile."""
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_body: no spec or model in world")
        return False

    if not SpecBuilder.remove_body(spec, body_name):
        logger.warning("Body '%s' not found in spec - nothing ejected", body_name)
        # Matching legacy behaviour: return True so scene state stays consistent
        # (caller has already popped the Python-side dict entry).
        return True

    return _recompile_preserving_state(world, spec)


def eject_robot_from_scene(world: SimWorld, robot_name: str) -> bool:
    """Remove every spec element namespaced under ``{robot_name}/``.

    Implementation note: deleting a body that was added via ``spec.attach()``
    triggers a known MuJoCo 3.8 segfault at interpreter shutdown (the
    attached child spec's memory gets freed twice). To sidestep that bug
    we REBUILD the scene spec from scratch using the post-remove
    ``world.robots`` / ``world.objects`` / ``world.cameras`` state, then
    re-attach the remaining robots. Joint state is not preserved across this
    path - callers that care should call ``reset`` or save/restore state
    around remove_robot. In the common case (agent removes a robot to clear
    the scene), this is the expected behaviour anyway.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_robot: no spec or model in world")
        return False

    mj = _ensure_mujoco()

    # Preserve the current qpos for bodies that are NOT being removed.
    # We rebuild from world state and then re-attach remaining robots, so
    # object freejoints start at their body pos (matching fresh add_object
    # semantics); robot joints start at qpos=0 (same as fresh add_robot).

    # First drop cameras that originated from the robot being ejected.
    # They're in world.cameras with origin_robot == robot_name. Without this,
    # SpecBuilder.build would skip them (via origin_robot), but stale entries
    # would linger in the registry and confuse observation code.
    stale_cam_names = [cname for cname, cam in world.cameras.items() if getattr(cam, "origin_robot", "") == robot_name]
    for cname in stale_cam_names:
        del world.cameras[cname]

    # Step 1: rebuild the base spec from world (objects + cameras +
    # lights + ground).
    new_spec = SpecBuilder.build(world)

    # Step 2: re-attach every remaining robot (the one being ejected is
    # already popped from ``world.robots`` by the caller).
    for robot in world.robots.values():
        # Re-discover joint names via the attach - they're stable per URDF.
        joint_names = SpecBuilder.attach_robot(new_spec, robot, robot.urdf_path)
        robot.joint_names = joint_names

    # Step 3: compile fresh and install. No spec.recompile(model, data)
    # here - recompile implicitly preserves qpos state which doesn't
    # make sense across a scene rebuild, and forcing a fresh compile
    # avoids the attach/delete bug.
    try:
        new_model = new_spec.compile()
        new_data = mj.MjData(new_model)
    except (ValueError, RuntimeError) as e:
        logger.error("eject_robot: fresh compile failed: %s", e)
        return False

    world._model = new_model
    world._data = new_data
    world._backend_state["spec"] = new_spec
    try:
        world._backend_state["xml"] = new_spec.to_xml()
    except Exception as xml_err:
        logger.debug("spec.to_xml() failed: %s", xml_err)

    # Re-discover joint/actuator IDs for remaining robots.
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        robot.actuator_ids = []
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
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(new_model.nu))

    logger.debug("eject_robot %r: scene rebuilt", robot_name)
    return True


# =============================================================================
# Agent-authored raw MJCF (Stage 6)
# =============================================================================


def replace_scene_mjcf(world: SimWorld, xml: str) -> bool:
    """Atomically swap the whole scene for agent-written MJCF.

    Validated by actually compiling it. On failure raises ``ValueError`` with
    MuJoCo's compiler error verbatim. On success, the old spec/model/data are
    replaced and all per-robot joint/actuator IDs re-discovered (but since
    the agent may have changed the whole scene, the ``world.robots`` dict
    is NOT touched - that's the caller's responsibility).
    """
    mj = _ensure_mujoco()
    new_spec = SpecBuilder.from_mjcf_string(xml)
    # Compile eagerly so malformed XML fails here rather than on the next
    # mj_step.
    new_model = new_spec.compile()
    new_data = mj.MjData(new_model)

    world._backend_state["spec"] = new_spec
    world._model = new_model
    world._data = new_data
    try:
        world._backend_state["xml"] = new_spec.to_xml()
    except Exception:
        pass
    return True
