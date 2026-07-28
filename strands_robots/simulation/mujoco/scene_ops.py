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
* :func:`reposition_body_in_scene` - edit a body's spec ``pos``/``quat`` + recompile.
* :func:`eject_robot_from_scene` - walk the spec, delete everything namespaced
  under ``{robot_name}/``, then recompile.

Every function takes a ``SimWorld`` whose ``_backend_state["spec"]`` holds the
live ``MjSpec``. They return ``True`` on success, ``False`` on failure (matching
the legacy API) so call sites in :mod:`simulation` don't need to change.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimWorld
from strands_robots.simulation.mujoco.backend import (
    _actuator_target_joint,
    _ensure_mujoco,
    filter_mujoco_attach_noise,
)
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder

logger = logging.getLogger(__name__)


def _scalar(value: Any) -> float:
    """Read an mjSpec numeric field that may be a scalar or a 1-element array.

    The mjSpec Python bindings expose some joint fields as bare floats
    (``MjsJoint.damping``, ``.armature`` in mujoco >= 3.3) and others as
    ndarrays (``.range``). Reading the wrong shape raises ``TypeError``, which
    is easy to bury in a broad handler, so normalise here instead of assuming
    either layout at the call site.
    """
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _set_mjspec_scalar(obj: Any, field: str, value: float) -> None:
    """Write a numeric mjSpec joint field that may be a bare float OR an ndarray.

    The mjSpec Python bindings are inconsistent across mujoco versions on how a
    1-DOF joint's ``damping`` / ``armature`` is exposed: some builds present a
    bare ``float`` (assign a scalar; item-assignment raises), others a
    ``[3, 1]`` ndarray (assign ``[0]``; a scalar assign raises "incompatible
    function arguments"). Probe the live shape and write the matching form so
    the same surgery compiles on both, instead of hard-coding one layout.
    """
    current = getattr(obj, field)
    try:
        # ndarray / buffer layout: has a settable element 0.
        current[0] = value
    except (TypeError, IndexError):
        setattr(obj, field, value)


def _rediscover_robot_ids(world: SimWorld) -> None:
    """Re-resolve every robot's ``joint_ids`` / ``actuator_ids`` against the model.

    Names inside MuJoCo are namespaced under ``robot.namespace``
    (e.g. ``arm1/shoulder_pan``) when the robot was attached via
    ``SpecBuilder.attach_robot``; the raw name is the fallback. A joint that no
    longer resolves is simply dropped, so the id lists always describe the LIVE
    model.

    Every path that recompiles must call this, because the cached ids are plain
    integer indices. ``patch_scene_mjcf`` did not, and its ``delete_body`` op
    accepts any body name - including a robot link - so deleting one left the
    registry describing a robot that no longer exists:

        add_robot("a")                       nbody=12 njnt=9 nu=8
        patch delete_body "a/link5"          nbody=6  njnt=4 nu=4   (status=success)
        world.robots["a"].joint_ids          [0..8]   -> 5 of them out of range
        zero_dynamics(robot_name="a")        IndexError: index 4 is out of bounds

    i.e. stale ids indexing past the end of the new, smaller arrays - raising out
    of the tool contract rather than returning a structured error.

    ``joint_names`` is pruned to the joints that still resolve, for the same
    reason: it is the ordering contract for ``set_joint_positions``' list form and
    for ``get_features``, so a name for a deleted joint made those disagree with
    each other (a 9-value list was rejected for naming 5 non-joints, while a
    4-value list was rejected for not being 9 long). A no-op on every ordinary
    path, where all names resolve.
    """
    mj = _ensure_mujoco()
    model = world._model
    if model is None:
        return
    for robot in world.robots.values():
        pfx = robot.namespace or ""
        robot.joint_ids = []
        robot.actuator_ids = []
        live_names: list[str] = []
        for jnt_name in robot.joint_names:
            jid = -1
            if pfx:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jid < 0:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
                live_names.append(jnt_name)
        if len(live_names) != len(robot.joint_names):
            logger.warning(
                "robot %r: %d of %d joints are no longer in the model (%s); "
                "dropping them from the registry so it describes the live scene",
                robot.name,
                len(robot.joint_names) - len(live_names),
                len(robot.joint_names),
                sorted(set(robot.joint_names) - set(live_names)),
            )
            robot.joint_names = live_names
        robot.actuator_ids = _robot_actuator_ids(model, robot.joint_ids)
        # Single-robot fallback: if no actuators matched by joint, assume all
        # actuators belong to this robot. Matches the legacy behaviour.
        if not robot.actuator_ids and len(world.robots) == 1:
            robot.actuator_ids = list(range(model.nu))


def _robot_actuator_ids(model: Any, joint_ids: list[int]) -> list[int]:
    """Actuator ids that drive any of ``joint_ids``, directly or via a tendon.

    A robot owns an actuator when the actuator's transmission targets one of
    the robot's joints. Two transmission shapes matter:

    * JOINT / JOINTINPARENT - ``trnid[0]`` is the joint id.
    * TENDON - ``trnid[0]`` is a *tendon* id; the actuator belongs to the robot
      when any joint wrapped by that tendon is the robot's (a parallel gripper
      driven by one tendon over two finger joints).

    Comparing the raw ``trnid[0]`` against joint ids - as this did - conflates
    those id spaces, so a tendon gripper was attributed to whichever robot
    happened to own the joint whose id equalled the tendon id, and was missing
    from its real owner. See :func:`_actuator_target_joint`.
    """
    mj = _ensure_mujoco()
    wanted = set(joint_ids)
    out: list[int] = []
    for act_id in range(int(model.nu)):
        jnt_id = _actuator_target_joint(model, act_id)
        if jnt_id >= 0:
            if jnt_id in wanted:
                out.append(act_id)
            continue
        if int(model.actuator_trntype[act_id]) != int(mj.mjtTrn.mjTRN_TENDON):
            continue
        # Walk the tendon's fixed-joint wraps; any robot joint claims it.
        ten_id = int(model.actuator_trnid[act_id, 0])
        adr = int(model.tendon_adr[ten_id])
        for w in range(adr, adr + int(model.tendon_num[ten_id])):
            if int(model.wrap_type[w]) == int(mj.mjtWrap.mjWRAP_JOINT) and int(model.wrap_objid[w]) in wanted:
                out.append(act_id)
                break
    return out


def _get_spec(world: SimWorld) -> Any | None:
    """Fetch the live MjSpec from ``world._backend_state``.

    Callers MUST have run ``_compile_world`` at least once before any scene
    mutation - without a spec we can't recompile. Returns ``None`` if missing
    so callers can return a clean error dict rather than crashing mid-op.
    """
    return world._backend_state.get("spec")


def _sync_cached_xml(world: SimWorld, spec: Any) -> None:
    """Refresh the legacy XML cache in ``world._backend_state["xml"]`` from ``spec``.

    Some readers (the ``load_scene`` + ``add_robot`` round-trip) consume the
    cached XML string rather than the live ``MjSpec``, so it must be refreshed
    after every recompile. ``spec.to_xml()`` can fail on specs MuJoCo cannot
    serialise; that is never fatal to the mutation (the live model/spec are
    already updated), so we leave the previous cache in place - but we always
    log the reason at debug so a silently stale cache stays diagnosable rather
    than being swallowed.
    """
    try:
        with filter_mujoco_attach_noise():
            world._backend_state["xml"] = spec.to_xml()
    except Exception as xml_err:
        logger.debug("spec.to_xml() failed; cached XML left stale: %s", xml_err)


def _install_model(world: SimWorld, model: Any, data: Any) -> None:
    """Install a new model/data pair and bump the recompile generation.

    ``_recompile_generation`` is the ONLY discriminator consumers have for "the
    arrays I cached no longer describe this model": ``load_state`` uses it to
    reject a stale checkpoint whose nq/nv/na/nu happen to be unchanged, and
    ``randomization._dr_baseline`` uses it to re-snapshot the un-randomised
    ``model.*`` arrays. Every path that swaps ``world._model`` must therefore go
    through here - four of the five swap sites used to bump nothing, so a
    ``remove_robot`` (full rebuild), ``replace_scene_mjcf``, ``patch_scene_mjcf``
    or ``remove_camera`` left the baseline indexing arrays of the WRONG LENGTH:

        add_robot a, add_robot b, add_object(0.05kg), randomize, remove_robot b
        6x randomize -> mass 0.6441 kg    (legal window is [0.025, 0.100])
        baseline body_mass len 24 vs live nbody 13

    and when the model GREW instead of shrank, an ``IndexError`` escaped
    ``randomize`` uncaught rather than merely corrupting the sample.

    Also runs the forward pass every swap needs so ``mjData``'s derived world
    transforms (``xpos``, ``cam_xpos``/``cam_xmat``, ``light_xpos``) are populated
    before anything reads them. ``spec.recompile`` preserves qpos/qvel but leaves
    those derived arrays zeroed, and ``render`` consumes them directly without a
    forward of its own, so ``remove_camera`` - the one recompile site that had no
    ``mj_forward`` - returned a near-black frame:

        render() frame mean 122.26 -> 25.49 after remove_camera
        cam_xpos [0,0,0], cam_xmat all-zero, every light_xpos [0,0,0]

    Owning it here means a new swap site cannot reintroduce the omission.
    """
    mj = _ensure_mujoco()
    world._model = model
    world._data = data
    world._recompile_generation += 1
    mj.mj_forward(model, data)


def _recompile_preserving_state(world: SimWorld, spec: Any, *, raise_on_refusal: bool = False) -> bool:
    """Recompile ``spec`` in place, replacing ``world._model`` and ``_data``.

    Uses ``spec.recompile(model, data)`` which auto-preserves qpos/qvel for
    existing joints and initializes new joints to their body's pos/quat. No
    manual state-copy loop is required.

    Also re-discovers per-robot joint and actuator IDs (they may have shifted
    as new bodies were inserted earlier in the body tree). Returns True on
    success, False on compile failure (logged).

    Args:
        world: The scene whose model/data are replaced on success.
        spec: The ``MjSpec`` to compile.
        raise_on_refusal: Re-raise the compiler's exception instead of folding
            it into a ``False`` return. A refusal message names what the
            compiler could not honor, and a ``bool`` cannot carry it: a caller
            that reports the reason to a user needs it, whereas a caller that
            only rolls back does not. Off by default so existing callers keep
            the ``bool`` contract.
    """
    try:
        with filter_mujoco_attach_noise():
            new_model, new_data = spec.recompile(world._model, world._data)
    except (ValueError, RuntimeError) as e:
        logger.error("spec.recompile failed: %s", e)
        # Stash MuJoCo's own reason so the caller can put it in the tool result.
        # This function returns bool and several callers rely on that, so the
        # message travels out-of-band rather than changing the signature. Without
        # it every rejection collapsed to "spec recompile refused", hiding the
        # actionable cause - "mass and inertia of moving bodies must be larger
        # than mjMINVAL", "repeated name 'table' in body" - which the agent needs
        # in order to correct its own call.
        world._backend_state["last_recompile_error"] = str(e)
        if raise_on_refusal:
            raise
        return False
    world._backend_state.pop("last_recompile_error", None)

    # Bumps the recompile generation so save_state/load_state can detect stale
    # checkpoints even when nq/nv/na/nu happen to stay the same (e.g. remove one
    # free-jointed object, add another of the same shape).
    # Also runs the forward pass newly-injected bodies need for valid
    # xpos/xquat and camera xforms; without it the next render() after
    # add_object / add_robot / add_camera is a 100% black frame.
    _install_model(world, new_model, new_data)

    # Keep the cached XML in sync with the spec for legacy readers (e.g.
    # load_scene + add_robot round-trip).
    _sync_cached_xml(world, spec)

    _rediscover_robot_ids(world)

    return True


# Persist


_NO_SPEC_REASON = "the scene has no live spec, so the change cannot be recorded durably"


def _spec_element_by_id(elements: Any, entity_id: int, kind: str) -> tuple[Any | None, str | None]:
    """Return the spec element a compiled entity id was built from.

    The compiler emits one model entity per spec element in declaration order and
    records the resulting index on the element, so a compiled id indexes the
    spec's element list directly. That mapping is what lets an UNNAMED element be
    addressed at all: most geoms in a robot scene carry no name, so resolving by
    name would silently cover almost none of them.

    The recorded ``id`` is verified rather than assumed. Writing a property onto
    the wrong entity is a worse outcome than not recording it, so a spec that no
    longer agrees with the compiled model is reported instead of written to.

    Args:
        elements: The spec's element list (``spec.geoms`` / ``spec.bodies``).
        entity_id: Compiled index of the entity, as resolved by the caller.
        kind: Entity name used in the reason text (``"geom"`` / ``"body"``).

    Returns:
        ``(element, None)`` when the element was located, otherwise
        ``(None, reason)`` naming why it was not.
    """
    count = len(elements)
    if entity_id < 0 or entity_id >= count:
        return None, f"{kind} id {entity_id} is outside the scene spec's {count} {kind}(s)"
    element = elements[entity_id]
    if element.id != entity_id:
        return None, (
            f"the scene spec no longer agrees with the compiled model: the {kind} at index "
            f"{entity_id} reports id {element.id}"
        )
    return element, None


def persist_geom_properties(
    world: SimWorld,
    geom_id: int,
    *,
    color: list[float] | None = None,
    friction: list[float] | None = None,
    size: list[float] | None = None,
) -> str | None:
    """Record a runtime geom property write in the spec the model is compiled from.

    ``world._model`` is DERIVED state: every scene mutation recompiles the spec
    over it (see :func:`_recompile_preserving_state`), so a value written only
    into the model is discarded by the next ``add_object`` / ``add_camera`` /
    ``add_robot`` call and the geom reverts to whatever it was compiled with -
    after the setter already reported the new value. Writing the same value into
    the spec is what makes that reported result durable.

    Nothing is written when a reason is returned, so a caller can refuse before it
    touches the model and keep the two representations in step.

    Args:
        world: The scene holding the live spec.
        geom_id: Compiled geom index, already resolved by the caller.
        color: RGBA components, already validated.
        friction: The three friction coefficients, already validated.
        size: Half-extents, already validated. Only the components the caller
            supplied are written, matching the model write: the unused tail of
            the spec's 3-wide row keeps its declared value.

    Returns:
        ``None`` once the value is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    spec_geom, reason = _spec_element_by_id(spec.geoms, geom_id, "geom")
    if spec_geom is None:
        return reason

    if color is not None:
        spec_geom.rgba = list(color)
    if friction is not None:
        spec_geom.friction[:] = friction
    if size is not None:
        spec_geom.size[: len(size)] = size
    return None


def persist_body_mass(world: SimWorld, body_id: int, *, mass_ratio: float) -> str | None:
    """Record a runtime body mass change in the spec the model is compiled from.

    Recording the change as a scale is what keeps the two representations equal:
    ``set_body_properties`` documents a mass change as a uniform density change at
    fixed geometry, and both mass and inertia are linear in density, so applying
    one ratio reproduces exactly the inertial the setter reported.

    A body's compiled inertial comes from one of two places, and only the one in
    force is writable. A body that declares an explicit ``<inertial>`` carries its
    own mass and inertia (``explicitinertial``); every other body has both
    integrated from its geoms, and there assigning ``mass`` on the body element is
    silently ignored by the compiler - the geoms' mass or density is what has to
    move.

    Args:
        world: The scene holding the live spec.
        body_id: Compiled body index, already resolved by the caller.
        mass_ratio: The new mass divided by the compiled mass. Finite and ``> 0``,
            which the caller guarantees by refusing a body with no mass to scale.

    Returns:
        ``None`` once the change is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    spec_body, reason = _spec_element_by_id(spec.bodies, body_id, "body")
    if spec_body is None:
        return reason

    if spec_body.explicitinertial:
        spec_body.mass *= mass_ratio
        # A body states its inertia as either the principal diagonal or the six
        # unique components of the full tensor; whichever form is not in use is
        # all zeros, so scaling both is exact and needs no branch.
        spec_body.inertia[:] = [value * mass_ratio for value in spec_body.inertia]
        spec_body.fullinertia[:] = [value * mass_ratio for value in spec_body.fullinertia]
        return None

    spec_geoms = list(spec_body.geoms)
    if not spec_geoms:
        return (
            "the body declares no explicit inertial and owns no geom, so it holds "
            "nothing whose mass the change could scale"
        )
    for spec_geom in spec_geoms:
        # A geom states either an explicit mass or a density, and the compiler
        # uses the mass only when it is set (an unset mass reads as nan, which
        # fails every comparison). Scaling whichever one is in force scales that
        # geom's contribution, so the body total scales by the same ratio.
        if spec_geom.mass > 0:
            spec_geom.mass *= mass_ratio
        else:
            spec_geom.density *= mass_ratio
    return None


def persist_world_option(
    world: SimWorld,
    *,
    gravity: list[float] | None = None,
    timestep: float | None = None,
) -> str | None:
    """Record a runtime physics-option write in the spec the model is compiled from.

    ``model.opt`` is compiled from ``spec.option``, so a gravity or timestep
    written only into the model is restored to the scene's declared value by the
    next recompile - putting a lunar-gravity world back to 9.81 m/s^2 on the next
    ``add_object``.

    Args:
        world: The scene holding the live spec.
        gravity: The three gravity components, already validated.
        timestep: The integration step in seconds, already validated.

    Returns:
        ``None`` once the value is recorded, otherwise the reason it could not be.
    """
    spec = _get_spec(world)
    if spec is None:
        return _NO_SPEC_REASON
    if gravity is not None:
        spec.option.gravity[:] = gravity
    if timestep is not None:
        spec.option.timestep = timestep
    return None


# Inject


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
        with filter_mujoco_attach_noise():
            joint_names = SpecBuilder.attach_robot(spec, robot, robot_xml_path)
        robot.joint_names = joint_names
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Robot attach failed for '%s': %s", robot.name, e)
        return False

    return _recompile_preserving_state(world, spec)


def inject_object_into_scene(world: SimWorld, obj: SimObject) -> bool:
    """Add a ``SimObject`` to the scene and recompile in place.

    A ``shape="mesh"`` object references a mesh asset that must be registered
    on the spec before the geom that names it can compile. The full-scene
    ``SpecBuilder.build`` registers those assets in its own pass, but the
    incremental path (``SpecBuilder.add_object``) does not, so this function
    registers the mesh (``spec.add_mesh(name=f"mesh_{obj.name}", ...)``) itself
    before adding the body. Without this, ``add_object(shape="mesh")`` at
    runtime always failed to recompile even for a valid mesh file.

    ``SpecBuilder.add_object`` mutates the spec (adds the body + geom) before
    the recompile that validates it. If that recompile is refused - e.g. the
    mesh file cannot be loaded - the just-added body AND its mesh asset are
    deleted again before returning ``False`` so the spec stays compilable.
    Without the rollback the orphan lingers and every later scene mutation,
    including a corrected retry under the same name, keeps failing to recompile
    (``repeated name`` collisions), bricking the whole scene after one bad add.

    The same rollback applies when ``SpecBuilder.add_object`` itself raises
    part-way through, which it can do *after* inserting the body (the geom's
    type lookup rejects an unsupported shape; the mass write rejects a
    non-numeric value). That error is then re-raised rather than folded into a
    ``False`` return, so the caller can report the actual reason - a swallowed
    ``ValueError`` left the caller with nothing but "spec recompile refused"
    while the actionable message went to the log.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_object: no spec or model in world")
        return False

    try:
        # Meshes need their asset registered before the geom references it.
        # build() registers meshes in a separate pass, so add_object does not;
        # the incremental path must register it here.
        if obj.shape == "mesh" and obj.mesh_path:
            spec.add_mesh(name=f"mesh_{obj.name}", file=obj.mesh_path)
        SpecBuilder.add_object(spec, obj)
    except (ValueError, RuntimeError):
        # add_object is atomic over its own body mutation: a raise there (an
        # unsupported shape, or a name that collides with an existing scene
        # body) rolls the half-built body back out itself, so only the mesh
        # asset registered here still needs undoing. Removing it keeps the spec
        # compilable, then the error propagates: the caller turns it into a
        # structured result, and the reason - e.g. the exact unsupported shape
        # and the supported list - is what a caller needs instead of a generic
        # recompile refusal.
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        raise

    # Roll the just-added body (and any mesh asset) back out so the spec
    # returns to its last good, compilable state (a worldbody body delete is
    # safe - the attach/delete segfault only affects spec.attach() child specs).
    #
    # Ask for the compiler's own reason rather than a bare False. Because the
    # object's mass is declared on its geom, MuJoCo integrates the inertia from
    # the shape, so its "mass and inertia of moving bodies must be larger than
    # mjMINVAL" floor is shape-dependent: a mass above mjMINVAL can still
    # integrate to an inertia below it on a small geom. add_object's numeric
    # pre-check cannot express that floor without duplicating the compiler's
    # per-shape integration, so the residual case has to arrive as the reason
    # the compiler gives. Folded into a False it became "spec recompile
    # refused." with the actionable text left in the log - the same dead end
    # the unsupported-shape path was fixed to stop producing.
    try:
        recompiled = _recompile_preserving_state(world, spec, raise_on_refusal=True)
    except (ValueError, RuntimeError):
        SpecBuilder.remove_body(spec, obj.name)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        raise
    if not recompiled:
        SpecBuilder.remove_body(spec, obj.name)
        SpecBuilder.remove_mesh(spec, f"mesh_{obj.name}")
        return False
    return True


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Add a camera to the scene and recompile in place.

    Mirrors :func:`inject_object_into_scene`: ``SpecBuilder.add_camera`` mutates
    the spec before the validating recompile, so a refused recompile rolls the
    just-added camera back out to keep the spec compilable for later edits.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("inject_camera: no spec or model in world")
        return False

    try:
        SpecBuilder.add_camera(spec, cam)
    except (ValueError, RuntimeError) as e:
        logger.error("Camera add failed for '%s': %s", cam.name, e)
        return False

    if not _recompile_preserving_state(world, spec):
        SpecBuilder.remove_camera(spec, cam.name)
        return False
    return True


# Eject


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

    # Objects added at runtime register a mesh asset named f"mesh_{name}".
    # Delete it too so the name is fully reusable and unused assets do not
    # accumulate across remove/re-add cycles (safe no-op for primitives).
    SpecBuilder.remove_mesh(spec, f"mesh_{body_name}")

    return _recompile_preserving_state(world, spec)


def reposition_body_in_scene(
    world: SimWorld,
    body_name: str,
    position: list[float] | None = None,
    orientation: list[float] | None = None,
) -> bool:
    """Reposition a body (by short name) by editing its spec pose and recompiling.

    Used for STATIC objects, which have no freejoint and therefore cannot be
    moved through ``data.qpos`` at runtime - MuJoCo welds a static body to the
    worldbody with no DOF. Editing the spec body ``pos``/``quat`` and
    recompiling (preserving other joints' state) is the only way to move a
    welded fixture, mirroring :func:`inject_object_into_scene` /
    :func:`eject_body_from_scene`.

    ``position`` / ``orientation`` are applied only when provided (a ``None``
    leaves that component untouched). Returns ``True`` on success, ``False`` if
    the spec/body is missing or the recompile fails (both logged), so the
    caller can surface a clean error instead of a silent no-op.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("reposition_body: no spec or model in world")
        return False

    try:
        body = spec.body(body_name)
    except (KeyError, ValueError):
        body = None
    if body is None:
        logger.warning("Body '%s' not found in spec - nothing repositioned", body_name)
        return False

    if position is not None:
        body.pos = list(position)
    if orientation is not None:
        body.quat = list(orientation)

    return _recompile_preserving_state(world, spec)


def _snapshot_joint_state(world: SimWorld) -> dict[str, tuple[list[float], list[float]]]:
    """Snapshot per-joint ``(qpos, qvel)`` slices keyed by fully-qualified
    MuJoCo joint name.

    Used by :func:`eject_robot_from_scene` to preserve the state of surviving
    robots and object freejoints across a scene rebuild. Flat-index slicing
    is unsafe here because the body-tree order may shift when a robot is
    removed (see AGENTS.md "Per-name state copy" rule).

    Returns a dict mapping ``<joint_name> -> (qpos_slice, qvel_slice)`` where
    each slice has the appropriate width for the joint type (1 for hinge/
    slide, 4 for ball, 7 for free).
    """
    if world._model is None or world._data is None:
        return {}
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    snap: dict[str, tuple[list[float], list[float]]] = {}
    for jid in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])
        jtype = int(model.jnt_type[jid])
        # qpos width: free=7, ball=4, hinge/slide=1
        # qvel width: free=6, ball=3, hinge/slide=1
        if jtype == mj.mjtJoint.mjJNT_FREE:
            qpos_w, qvel_w = 7, 6
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            qpos_w, qvel_w = 4, 3
        else:
            qpos_w, qvel_w = 1, 1
        snap[name] = (
            [float(x) for x in data.qpos[qpos_adr : qpos_adr + qpos_w]],
            [float(x) for x in data.qvel[qvel_adr : qvel_adr + qvel_w]],
        )
    return snap


def _restore_joint_state(
    world: SimWorld,
    snapshot: dict[str, tuple[list[float], list[float]]],
) -> int:
    """Restore per-joint state from a snapshot into ``world._data`` by name.

    Joints that no longer exist in the compiled model (e.g. those belonging
    to the ejected robot) are silently skipped. Joints that exist in the
    new model but were not in the snapshot keep their fresh-compile defaults
    (body pos/quat for freejoints, 0 for hinge/slide).

    Returns the number of joints actually restored, for logging.
    """
    if world._model is None or world._data is None:
        return 0
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    restored = 0
    for name, (qpos_vals, qvel_vals) in snapshot.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue  # joint no longer exists (expected for ejected robot)
        qpos_adr = int(model.jnt_qposadr[jid])
        qvel_adr = int(model.jnt_dofadr[jid])
        # Width sanity check: if joint type changed (should not happen for
        # same-name joints across an eject), skip to avoid corrupting state.
        jtype = int(model.jnt_type[jid])
        if jtype == mj.mjtJoint.mjJNT_FREE:
            expect_qp, expect_qv = 7, 6
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            expect_qp, expect_qv = 4, 3
        else:
            expect_qp, expect_qv = 1, 1
        if len(qpos_vals) != expect_qp or len(qvel_vals) != expect_qv:
            logger.warning(
                "_restore_joint_state: width mismatch for %r (qpos %d!=%d or qvel %d!=%d), skipping",
                name,
                len(qpos_vals),
                expect_qp,
                len(qvel_vals),
                expect_qv,
            )
            continue
        for i, v in enumerate(qpos_vals):
            data.qpos[qpos_adr + i] = v
        for i, v in enumerate(qvel_vals):
            data.qvel[qvel_adr + i] = v
        restored += 1
    return restored


def _snapshot_actuation(world: SimWorld) -> dict[str, Any]:
    """Snapshot the actuation half of ``MjData`` that a fresh compile discards.

    :func:`eject_robot_from_scene` deliberately does a fresh ``spec.compile()``
    + ``mj.MjData(new_model)`` to dodge the MuJoCo attach/delete segfault, so
    unlike the sibling ``spec.recompile(model, data)`` path it starts from a
    zeroed ``MjData``. ``_snapshot_joint_state`` carries only ``qpos``/``qvel``,
    so every surviving position servo lost its commanded target and was driven
    from its held pose toward 0 on the next step.

    ``ctrl`` and ``act`` are keyed by ACTUATOR NAME, never flat index: the fresh
    compile shifts actuator ids exactly as it shifts joint ids (AGENTS.md
    "Per-name state copy"). ``qfrc_applied`` is keyed by joint name for the same
    reason - it holds latched ``apply_force`` torques.

    The engine's own state contract already says actuation is part of the state:
    ``PhysicsMixin.save_state`` uses ``mjSTATE_INTEGRATION`` precisely because
    ``mjSTATE_FULLPHYSICS`` "silently excluded ``ctrl`` and ``qfrc_applied``, so
    the first step after a restore drove toward the pre-restore targets".

    Returns:
        Dict with ``ctrl`` / ``act`` (``{actuator name: value}``) and
        ``qfrc_applied`` (``{joint name: [per-DOF values]}``). Empty when there
        is no compiled model.
    """
    if world._model is None or world._data is None:
        return {}
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    ctrl: dict[str, float] = {}
    act: dict[str, float] = {}
    for aid in range(int(model.nu)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, aid)
        if not name:
            continue
        ctrl[name] = float(data.ctrl[aid])
        # Stateful actuators (integrator / filter dyntype) carry an internal
        # activation; a zeroed act restarts a filtered servo from scratch.
        act_adr = int(model.actuator_actadr[aid])
        if act_adr >= 0 and act_adr < len(data.act):
            act[name] = float(data.act[act_adr])
    qfrc_applied: dict[str, list[float]] = {}
    for jid in range(int(model.njnt)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        width = _joint_dof_width(mj, int(model.jnt_type[jid]))
        values = [float(x) for x in data.qfrc_applied[dof_adr : dof_adr + width]]
        if any(values):
            qfrc_applied[name] = values
    return {"ctrl": ctrl, "act": act, "qfrc_applied": qfrc_applied}


def _restore_actuation(world: SimWorld, snapshot: dict[str, Any]) -> int:
    """Write a :func:`_snapshot_actuation` result back through fresh name lookups.

    Names that no longer resolve (the ejected robot's actuators and joints) are
    skipped, exactly as :func:`_restore_joint_state` skips vanished joints.

    Args:
        world: The world holding the freshly compiled model and data.
        snapshot: Result of :func:`_snapshot_actuation`.

    Returns:
        Number of actuators whose ``ctrl`` was restored, for logging.
    """
    if not snapshot or world._model is None or world._data is None:
        return 0
    mj = _ensure_mujoco()
    model = world._model
    data = world._data
    restored = 0
    for name, value in (snapshot.get("ctrl") or {}).items():
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue  # actuator belonged to the ejected robot
        data.ctrl[aid] = value
        restored += 1
    for name, value in (snapshot.get("act") or {}).items():
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        act_adr = int(model.actuator_actadr[aid])
        if act_adr >= 0 and act_adr < len(data.act):
            data.act[act_adr] = value
    for name, values in (snapshot.get("qfrc_applied") or {}).items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        if _joint_dof_width(mj, int(model.jnt_type[jid])) != len(values):
            logger.warning(
                "_restore_actuation: DOF width changed for joint %r, dropping its latched qfrc_applied",
                name,
            )
            continue
        for offset, value in enumerate(values):
            data.qfrc_applied[dof_adr + offset] = value
    return restored


def _joint_dof_width(mj: Any, jnt_type: int) -> int:
    """Return the number of ``qvel``/``qfrc`` entries a joint type occupies."""
    if jnt_type == mj.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mj.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def eject_robot_from_scene(world: SimWorld, robot_name: str) -> bool:
    """Remove every spec element namespaced under ``{robot_name}/``.

    Implementation note: deleting a body that was added via ``spec.attach()``
    triggers a known MuJoCo 3.8 segfault at interpreter shutdown (the
    attached child spec's memory gets freed twice). To sidestep that bug
    we REBUILD the scene spec from scratch using the post-remove
    ``world.robots`` / ``world.objects`` / ``world.cameras`` state, then
    re-attach the remaining robots.

    Joint state preservation: before the rebuild we snapshot every joint's
    ``(qpos, qvel)`` keyed by fully-qualified name; after the fresh compile
    we restore state for every joint that still exists in the new model.
    Joints belonging to the ejected robot are naturally dropped (their name
    no longer resolves). This keeps surviving robots at their current pose
    and object freejoints at their current world pose - the behaviour the
    agent expects when calling ``remove_robot`` mid-scene.

    Flat-index slicing is **not** safe here: removing a robot shifts every
    body/joint index that comes after it in the kinematic tree, so
    ``data.qpos[:]`` copies across compiles would mis-assign DOFs. Per-name
    lookup is the only correct approach (see AGENTS.md).
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("eject_robot: no spec or model in world")
        return False

    mj = _ensure_mujoco()

    # Snapshot joint state BEFORE we rebuild. Keyed by the fully-qualified
    # MuJoCo joint name (prefix/joint for attached robots, bare name for
    # object freejoints).
    state_snapshot = _snapshot_joint_state(world)
    # ... and the actuation state, which the fresh MjData in Step 3 zeroes. A
    # position servo whose ctrl is gone is driven from its held pose toward 0.
    actuation_snapshot = _snapshot_actuation(world)

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
        with filter_mujoco_attach_noise():
            joint_names = SpecBuilder.attach_robot(new_spec, robot, robot.urdf_path)
        robot.joint_names = joint_names

    # Step 2-cameras: body-mounted cameras could not be added by
    # ``SpecBuilder.build`` because their parent body (e.g. ``a/hand``) only
    # exists after the attaches above. Add them now; a wrist camera on a
    # SURVIVING robot used to take the whole rebuild down with an uncaught
    # ValueError out of build().
    SpecBuilder.add_deferred_cameras(new_spec, world)

    # Step 2a: replay any recorded actuate_robot surgery onto the fresh spec.
    # SpecBuilder.attach_robot rebuilds each robot from its URDF, which carries
    # no runtime servos - so removing an UNRELATED robot stripped a surviving
    # robot's position actuators, reverted the integrator to Euler and zeroed its
    # damping/armature/gravcomp, leaving send_action with unresolved keys and the
    # arm undrivable mid-episode.
    for robot in world.robots.values():
        actuation = getattr(robot, "actuation", None)
        if not actuation:
            continue
        try:
            _apply_actuation_to_spec(
                new_spec,
                mj,
                robot.name,
                robot.namespace or "",
                dict(actuation.get("kp_by_joint") or {}),
                damping=float(actuation.get("damping", 0.0)),
                armature=float(actuation.get("armature", 0.0)),
                gravity_compensation=bool(actuation.get("gravity_compensation")),
                disable_self_collision=bool(actuation.get("disable_self_collision")),
            )
        except (ValueError, RuntimeError, TypeError) as e:
            logger.error("eject_robot: could not replay actuation for %r: %s", robot.name, e)

    # Step 2b: re-add the runtime weld equalities of any attachment that
    # SURVIVES the rebuild. ``SpecBuilder.build`` reconstructs only
    # objects/cameras/lights/ground, so a weld added at runtime by
    # ``attach_bodies`` was silently destroyed by an UNRELATED robot removal:
    # removing robot 'b' dropped the weld holding a cube to robot 'a's hand
    # (neq 3 -> 1), and the "grasped" object drifted 0.30 m away while the
    # registry still reported it attached. Attachments referencing the ejected
    # robot are skipped - their bodies no longer exist - and their stale records
    # are dropped so they cannot block a later remove/re-attach.
    registry = world._backend_state.get("attachments")
    if isinstance(registry, dict):
        for child, record in list(registry.items()):
            if record.get("mode") != "weld":
                continue
            parent = str(record.get("parent", ""))
            if parent == robot_name or parent.startswith(f"{robot_name}/"):
                registry.pop(child, None)
                logger.info(
                    "eject_robot %r: dropped weld record for %r (its parent belonged to the ejected robot)",
                    robot_name,
                    child,
                )
                continue
            _readd_weld(new_spec, mj, record, child, parent)

    # Step 3: compile fresh and install. No spec.recompile(model, data)
    # here - recompile implicitly preserves qpos state which doesn't
    # make sense across a scene rebuild, and forcing a fresh compile
    # avoids the attach/delete bug.
    try:
        with filter_mujoco_attach_noise():
            new_model = new_spec.compile()
        new_data = mj.MjData(new_model)
    except (ValueError, RuntimeError) as e:
        logger.error("eject_robot: fresh compile failed: %s", e)
        return False

    _install_model(world, new_model, new_data)
    world._backend_state["spec"] = new_spec
    _sync_cached_xml(world, new_spec)

    # Step 4: restore state for every joint that survived the rebuild. Joints
    # belonging to the ejected robot simply don't resolve and get skipped.
    restored = _restore_joint_state(world, state_snapshot)
    # Restore commanded targets alongside the pose. Both must land before
    # mj_forward below, so the derived quantities reflect the real command.
    restored_ctrl = _restore_actuation(world, actuation_snapshot)

    # Step 5: run a forward pass so derived quantities (xpos, cam xforms)
    # reflect the restored state. Without this, the next render() call can
    # produce stale frames because MjData was freshly allocated in Step 3.
    mj.mj_forward(new_model, new_data)

    # Re-discover joint/actuator IDs for remaining robots.
    _rediscover_robot_ids(world)

    logger.debug(
        "eject_robot %r: scene rebuilt, restored state for %d/%d joints and ctrl for %d/%d actuators",
        robot_name,
        restored,
        len(state_snapshot),
        restored_ctrl,
        len(actuation_snapshot.get("ctrl") or {}),
    )
    return True


# Runtime attach / actuate primitives (GH #1533, PR 1)


def _readd_weld(spec: Any, mj: Any, record: dict[str, Any], child: str, parent: str) -> None:
    """Re-create a runtime weld equality on a freshly rebuilt spec.

    Used by :func:`eject_robot_from_scene`, whose rebuild reconstructs only the
    declarative scene and would otherwise discard every weld ``attach_bodies``
    added at runtime. Best-effort: a weld that cannot be re-created is logged and
    skipped rather than aborting the ejection, since the model itself is valid
    without it.
    """
    eq_name = str(record.get("eq_name") or f"attach_weld_{child}")
    relpos = record.get("relpos") or [0.0, 0.0, 0.0]
    relquat = record.get("relquat") or [1.0, 0.0, 0.0, 0.0]
    eq = spec.add_equality()
    try:
        eq.name = eq_name
        eq.type = mj.mjtEq.mjEQ_WELD
        eq.objtype = mj.mjtObj.mjOBJ_BODY
        eq.name1 = parent
        eq.name2 = child
        data = [0.0] * 11
        data[3:6] = [float(v) for v in relpos]
        data[6:10] = [float(v) for v in relquat]
        data[10] = float(record.get("torquescale", 1.0) or 1.0)
        eq.data = data
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error("eject_robot: could not re-add weld %r: %s", eq_name, e)
        spec.delete(eq)


def add_weld_constraint(
    world: SimWorld,
    *,
    name: str,
    parent: str,
    child: str,
    relpos: list[float],
    relquat: list[float],
    torquescale: float = 1.0,
) -> bool:
    """Add a named weld equality constraint between two bodies and recompile.

    ``relpos`` / ``relquat`` are the pose of ``child`` expressed in ``parent``'s
    frame - callers capture the CURRENT runtime relative pose so the weld holds
    the bodies exactly where they are (MuJoCo's all-zero ``relpose`` quat would
    instead bake in the compile-time ``qpos0`` pose, which is wrong for a
    runtime grasp-attach). The equality is stored on the live spec so it
    survives later recompiles (``add_object`` etc.).

    Returns ``True`` on success. On ANY failure - a rejected attribute write (a
    duplicate ``name``) or a failed recompile - the just-added equality is
    deleted again so the spec stays compilable, and ``False`` is returned. This
    function never raises.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("add_weld_constraint: no spec or model in world")
        return False
    mj = _ensure_mujoco()

    eq = spec.add_equality()
    # Every attribute write can raise - a duplicate ``name`` raises ValueError
    # from MuJoCo. Without this guard the exception escaped past the documented
    # "returns True/False" contract AND left the half-initialized equality on the
    # live spec (``eq.type`` never assigned, so it stayed mjEQ_CONNECT), which
    # made the world permanently un-mutable: every later add_object /
    # export_xml / detach failed on "repeated name" or "connect constraint
    # supports only sites and bodies", with no recovery route.
    try:
        eq.name = name
        eq.type = mj.mjtEq.mjEQ_WELD
        eq.objtype = mj.mjtObj.mjOBJ_BODY
        eq.name1 = parent
        eq.name2 = child
        # mjEQ_WELD data layout (mjNEQDATA=11): [anchor(3), relpose pos(3),
        # relpose quat(4), torquescale(1)]. anchor stays zero - relpose fully
        # determines the held configuration.
        data = [0.0] * 11
        data[3:6] = [float(v) for v in relpos]
        data[6:10] = [float(v) for v in relquat]
        data[10] = float(torquescale)
        eq.data = data
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error("add_weld_constraint(%r): %s", name, e)
        spec.delete(eq)
        return False

    if not _recompile_preserving_state(world, spec):
        spec.delete(eq)
        return False
    return True


def remove_equality_constraint(world: SimWorld, name: str) -> bool:
    """Delete a named equality constraint from the live spec and recompile.

    Returns ``False`` (logged) when the constraint is missing or the recompile
    fails, so callers can surface a clean error instead of a silent no-op.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("remove_equality_constraint: no spec or model in world")
        return False
    for eq in spec.equalities:
        if eq.name == name:
            spec.delete(eq)
            return _recompile_preserving_state(world, spec)
    logger.warning("Equality constraint '%s' not found in spec - nothing removed", name)
    return False


def _apply_actuation_to_spec(
    spec: Any,
    mj: Any,
    robot_name: str,
    pfx: str,
    kp_by_joint: dict[str, float],
    *,
    damping: float,
    armature: float,
    gravity_compensation: bool,
    disable_self_collision: bool,
) -> None:
    """Apply the ``actuate_robot`` spec surgery: servos + damping/armature/gravcomp.

    Factored out of :func:`actuate_robot_in_scene` so
    :func:`eject_robot_from_scene` can REPLAY a recorded surgery onto its freshly
    rebuilt spec. The rebuild reconstructs each surviving robot from its URDF,
    which carries none of this, so without a replay removing an unrelated robot
    silently stripped this robot's actuators and reverted its integrator.

    Raises on a rejected spec write; callers decide whether that is fatal.
    """
    # Bare URDF chains (no damping/armature) diverge under the default Euler
    # integrator once stiff position servos are added; implicitfast integrates
    # joint damping implicitly and stays stable.
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    for body in spec.bodies:
        body_name = body.name or ""
        if pfx and body_name.startswith(pfx):
            if gravity_compensation:
                body.gravcomp = 1.0
            if disable_self_collision:
                for geom in body.geoms:
                    geom.contype = 0
                    geom.conaffinity = 0

    for joint in spec.joints:
        joint_name = joint.name or ""
        if not (pfx and joint_name.startswith(pfx)):
            continue
        short = joint_name[len(pfx) :]
        if short not in kp_by_joint:
            continue
        # ``MjsJoint.damping`` / ``.armature`` are a bare float on some mujoco
        # builds and a ``[3, 1]`` ndarray on others; write shape-agnostically so
        # the surgery compiles on both (see _set_mjspec_scalar).
        _set_mjspec_scalar(joint, "damping", max(_scalar(joint.damping), damping))
        _set_mjspec_scalar(joint, "armature", max(_scalar(joint.armature), armature))

    for short, kp in kp_by_joint.items():
        act = spec.add_actuator()
        act.name = f"{robot_name}_act_{short}"
        act.target = f"{pfx}{short}"
        act.trntype = mj.mjtTrn.mjTRN_JOINT
        jnt_range_defined = False
        for joint in spec.joints:
            if (joint.name or "") == f"{pfx}{short}":
                jnt_range_defined = bool(float(joint.range[0]) < float(joint.range[1]))
                break
        act.set_to_position(kp=float(kp), dampratio=1.0, inheritrange=jnt_range_defined)


def actuate_robot_in_scene(
    world: SimWorld,
    robot: SimRobot,
    kp_by_joint: dict[str, float],
    *,
    damping: float,
    armature: float,
    gravity_compensation: bool,
    disable_self_collision: bool,
) -> bool:
    """Add position-servo actuators to a robot's joints and recompile.

    The supported form of the private-spec surgery the ``so101_curobo`` example
    performed by hand (GH #1533): converts an actuator-less (URDF-loaded) arm
    into a position-controlled one so ``send_action`` / ``run_policy`` can
    drive it. Per joint in ``kp_by_joint`` (SHORT joint names, values = kp):

    * a position actuator (``set_to_position``: fixed gain kp, affine bias with
      ``dampratio=1.0`` for ~critical damping, ctrlrange inherited from the
      joint's range when it declares one),
    * joint ``damping`` / ``armature`` floors (bare URDFs ship none, which
      blows up explicit integration),

    plus, scene-wide, the stable ``implicitfast`` integrator, and optionally
    gravity compensation on the robot's bodies and self-collision disable on
    the robot's own geoms (cuRobo-style planners ignore adjacent-link
    contacts, which otherwise block planned motion).

    Atomicity: the spec is snapshotted (XML round-trip) before surgery; any
    failure restores the snapshot so no partial edit (integrator flip, gravcomp,
    half the actuators) lingers on the live spec. Returns ``True`` on success.
    """
    spec = _get_spec(world)
    if spec is None or world._model is None:
        logger.error("actuate_robot_in_scene: no spec or model in world")
        return False
    mj = _ensure_mujoco()
    pfx = robot.namespace or ""

    # Snapshot for atomic rollback (mirrors patch_scene_mjcf): the surgery
    # touches option/bodies/joints/actuators/geoms, too many objects to undo
    # piecewise.
    # Use the native deep copy, NOT an XML round-trip: to_xml() emits mesh refs
    # as bare filenames and drops meshdir/assets, so a restored spec for any
    # mesh-based robot cannot compile - which left the world with a permanently
    # un-mutable spec (every later add_object / actuate failed) while the model
    # kept stepping. spec.copy() also preserves joint state, which the
    # round-trip silently reset.
    try:
        backup_spec = spec.copy()
    except (ValueError, RuntimeError) as e:  # pragma: no cover - copy of a live spec is fine
        logger.error("actuate_robot_in_scene: failed to snapshot spec: %s", e)
        return False

    try:
        _apply_actuation_to_spec(
            spec,
            mj,
            robot.name,
            pfx,
            kp_by_joint,
            damping=damping,
            armature=armature,
            gravity_compensation=gravity_compensation,
            disable_self_collision=disable_self_collision,
        )
    except (ValueError, RuntimeError, TypeError) as e:
        logger.error("actuate_robot_in_scene: spec surgery failed for '%s': %s", robot.name, e)
        world._backend_state["spec"] = backup_spec
        return False

    if not _recompile_preserving_state(world, spec):
        # Restore the pre-surgery spec so the failed edit doesn't poison
        # later scene mutations.
        world._backend_state["spec"] = backup_spec
        return False
    # Record the surgery so a FULL rebuild can re-apply it. eject_robot_from_scene
    # reconstructs every surviving robot from its URDF, which carries none of
    # this - so removing an UNRELATED robot silently stripped this robot's
    # position servos, reverted the integrator to Euler and zeroed its
    # damping/armature/gravcomp, leaving send_action with unresolved keys.
    robot.actuation = {
        "kp_by_joint": dict(kp_by_joint),
        "damping": float(damping),
        "armature": float(armature),
        "gravity_compensation": bool(gravity_compensation),
        "disable_self_collision": bool(disable_self_collision),
    }
    return True


# Agent-authored raw MJCF (Stage 6)


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
    with filter_mujoco_attach_noise():
        new_model = new_spec.compile()
    new_data = mj.MjData(new_model)

    world._backend_state["spec"] = new_spec
    # Mark the scene as agent-authored. The registries (world.objects / cameras)
    # cannot describe raw MJCF, and the FULL rebuild path
    # (``eject_robot_from_scene``, triggered by removing ANY robot) reconstructs
    # the scene from exactly those registries - so it silently discards
    # everything this XML introduced. Measured, for a scene with two hand-written
    # bodies, a sensor and a site:
    #
    #     after replace_scene_mjcf   nbody=25 nsensor=1 nsite=1
    #     after remove_robot         nbody=12 nsensor=0 nsite=0
    #
    # with ``remove_robot`` reporting only "Robot 'b' removed.". The flag lets it
    # warn instead of destroying the agent's scene without a word.
    world._backend_state["raw_mjcf_scene"] = True
    # Installs and runs the forward pass geom positions / camera xforms need;
    # without it the first render() here hits `data.xpos == 0 for all bodies`
    # and the renderer dumps a 100% black frame.
    _install_model(world, new_model, new_data)

    _sync_cached_xml(world, new_spec)
    return True


# Structured-op patching of the live spec (Stage 6, part 2 - GH #125)

# Supported ops for patch_scene_mjcf. Kept narrow on purpose - adding unchecked
# attribute setters would make the tool an arbitrary-code hole. Agents that
# need exotic MJCF should go through replace_scene_mjcf with a full XML.
_PATCH_OPS = {
    "add_body",
    "add_geom",
    "add_site",
    "set_body_pos",
    "set_body_quat",
    "delete_body",
}


def _find_body(spec: Any, name: str, new_bodies: dict[str, Any]) -> Any:
    """Locate a body by name in a live spec, checking batch-local additions.

    MuJoCo 3.8 ``spec.body(name)`` only resolves bodies that existed at the
    last ``compile()`` / ``recompile()`` call. Bodies added mid-batch are
    not visible through that lookup but ARE present on the spec - we track
    their handles in ``new_bodies`` so ``add_geom`` / ``add_site`` /
    ``set_body_pos`` etc. can reference them within the same patch.
    """
    if name == "world":
        return spec.worldbody
    if name in new_bodies:
        return new_bodies[name]
    b = spec.body(name)
    if b is not None:
        return b
    # Fallback: scan all bodies. Catches bodies introduced via spec.attach()
    # (e.g. robots composed into the scene) that aren't in new_bodies because
    # we didn't create them in this batch.
    for body in spec.bodies:
        if body.name == name:
            return body
    return None


def _sync_patched_object_pose(world: SimWorld, op: dict[str, Any]) -> None:
    """Mirror a ``set_body_pos`` / ``set_body_quat`` patch onto its ``SimObject``.

    ``SpecBuilder.build`` (the full-rebuild path) reconstructs object bodies from
    ``world.objects``, so a pose that lived only on the spec was silently reverted
    by an unrelated ``remove_robot``. Only tracked objects are mirrored - a patch
    against a robot link or a body the patch itself created has nothing to update,
    and the spec edit already stands for the current model.
    """
    kind = op.get("op")
    if kind not in ("set_body_pos", "set_body_quat"):
        return
    name = op.get("name")
    if not isinstance(name, str):
        return
    obj = world.objects.get(name)
    if obj is None:
        return
    if kind == "set_body_pos":
        pos = op.get("pos")
        if pos is not None:
            obj.position = [float(v) for v in pos]
    else:
        quat = op.get("quat")
        if quat is not None:
            obj.orientation = [float(v) for v in quat]


def _apply_patched_free_body_poses(world: SimWorld, ops: list[dict[str, Any]]) -> None:
    """Write the live ``qpos`` for every pose-patched FREE-jointed body.

    Called after the batch recompile. A free body's pose lives in its freejoint's
    ``qpos``, which the recompile preserves - so editing the body's rest pose in
    the spec moved a *static* body but left a *dynamic* one exactly where it was
    until something re-seeded ``qpos`` from ``qpos0`` (the next ``reset``). Writing
    the freejoint slice makes the patch take effect immediately, matching both the
    static-body behaviour and what ``list_objects`` already reported.

    The body's velocity is zeroed for the same reason ``move_object`` does it: a
    teleport that keeps its old momentum shoots off on the next step. Bodies with
    no freejoint (static fixtures, robot links) and unresolvable names are skipped
    - the rest-pose edit already positions those.
    """
    mj = _ensure_mujoco()
    model, data = world._model, world._data
    if model is None or data is None:
        return
    for op in ops:
        kind = op.get("op")
        if kind not in ("set_body_pos", "set_body_quat"):
            continue
        name = op.get("name")
        if not isinstance(name, str):
            continue
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        jnt_id = -1
        for i in range(int(model.njnt)):
            if int(model.jnt_bodyid[i]) == body_id and int(model.jnt_type[i]) == int(mj.mjtJoint.mjJNT_FREE):
                jnt_id = i
                break
        if jnt_id < 0:
            continue  # static fixture / robot link: the rest-pose edit is enough
        adr = int(model.jnt_qposadr[jnt_id])
        if kind == "set_body_pos":
            pos = op.get("pos")
            if pos is not None:
                data.qpos[adr : adr + 3] = [float(v) for v in pos]
        else:
            quat = op.get("quat")
            if quat is not None:
                data.qpos[adr + 3 : adr + 7] = [float(v) for v in quat]
        dof = int(model.jnt_dofadr[jnt_id])
        data.qvel[dof : dof + 6] = 0.0
    mj.mj_forward(model, data)


def _finite_vec(op_kind: str, field: str, value: Any, expect: int) -> list[float]:
    """Coerce a patch op's numeric vector, rejecting a non-finite component.

    MuJoCo's compiler does not reject a ``nan``/``inf`` body pose, so an
    LLM-supplied one was written straight into the model and the patch still
    reported success. The corruption then spread on the first step:

        patch add_body pos=[nan, 0, 0.3]   status=success
        model.body_pos                     [nan, 0, 0.3]
        data.xpos                          non-finite
        step(n_steps=20)                   status=success
        data.qpos / data.qvel              non-finite
        get_observation("a")["joint1"]      nan

    with MuJoCo only muttering "Nan, Inf or huge value in QPOS" to stderr. Every
    numeric field of every op goes through here so the whole batch is rejected
    before it touches the spec, matching the up-front all-or-nothing validation
    ``send_action`` and ``set_joint_positions`` already do.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(f"{op_kind}: '{field}' must be a list of {expect} numbers, got {type(value).__name__}")
    values = list(value)
    if len(values) != expect:
        raise ValueError(f"{op_kind}: '{field}' must have {expect} components, got {len(values)}")
    out: list[float] = []
    for i, raw in enumerate(values):
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{op_kind}: '{field}'[{i}] is not a number ({raw!r})") from e
        if not math.isfinite(numeric):
            raise ValueError(
                f"{op_kind}: '{field}'[{i}] is not finite ({numeric}). A non-finite pose corrupts "
                "the whole world - MuJoCo propagates it into qpos/qvel on the next step and every "
                "observation becomes nan. Nothing was applied."
            )
        out.append(numeric)
    return out


def _finite_quat(op_kind: str, value: Any) -> list[float]:
    """Like :func:`_finite_vec` for a wxyz quaternion, also rejecting zero norm.

    An all-zero quaternion passes a finiteness check but is not a rotation;
    MuJoCo normalises it into ``nan``, reintroducing the same corruption.
    """
    quat = _finite_vec(op_kind, "quat", value, 4)
    norm = math.sqrt(sum(component * component for component in quat))
    if norm < 1e-9:
        raise ValueError(f"{op_kind}: 'quat' has (near-)zero norm {norm:.3e}; it does not describe a rotation.")
    return quat


def _apply_patch_op(spec: Any, op: dict[str, Any], new_bodies: dict[str, Any]) -> None:
    """Apply a single structured op to a live MjSpec.

    Raises ``ValueError`` with a human-readable message on bad input;
    MuJoCo compile errors surface on the enclosing ``recompile`` call. Every
    numeric field is validated through :func:`_finite_vec` / :func:`_finite_quat`
    first - the compiler accepts a non-finite pose, so an unvalidated one poisoned
    the world while reporting success. ``new_bodies`` is a batch-local cache of
    body handles added earlier in the same patch (see ``_find_body``).
    """
    if not isinstance(op, dict):
        raise ValueError(f"each op must be a dict, got {type(op).__name__}")

    kind = op.get("op")
    if kind not in _PATCH_OPS:
        raise ValueError(f"unknown op '{kind}'. Supported: {sorted(_PATCH_OPS)}")

    if kind == "add_body":
        parent = op.get("parent", "world")
        name = op.get("name")
        if not name:
            raise ValueError("add_body requires 'name'")
        pos = _finite_vec("add_body", "pos", op.get("pos", [0.0, 0.0, 0.0]), 3)
        quat = _finite_quat("add_body", op.get("quat", [1.0, 0.0, 0.0, 0.0]))
        parent_body = _find_body(spec, parent, new_bodies)
        if parent_body is None:
            raise ValueError(f"add_body: parent '{parent}' not found")
        new_body = parent_body.add_body(name=name, pos=pos, quat=quat)
        new_bodies[name] = new_body
        return

    if kind == "add_geom":
        body_name = op.get("body")
        if not body_name:
            raise ValueError("add_geom requires 'body'")
        body = _find_body(spec, body_name, new_bodies)
        if body is None:
            raise ValueError(f"add_geom: body '{body_name}' not found")

        shape = op.get("type", "box")
        from strands_robots.simulation.mujoco.spec_builder import (
            _geom_type,
            _normalize_size,
        )

        geom_kwargs: dict[str, Any] = {
            "type": _geom_type(shape),
            "size": _normalize_size(shape, op.get("size", [0.1, 0.1, 0.1])),
            "rgba": op.get("rgba", [0.5, 0.5, 0.5, 1.0]),
        }
        if "name" in op:
            geom_kwargs["name"] = op["name"]
        if "pos" in op:
            geom_kwargs["pos"] = _finite_vec("add_geom", "pos", op["pos"], 3)
        if "quat" in op:
            geom_kwargs["quat"] = _finite_quat("add_geom", op["quat"])
        body.add_geom(**geom_kwargs)
        return

    if kind == "add_site":
        body_name = op.get("body", "world")
        body = _find_body(spec, body_name, new_bodies)
        if body is None:
            raise ValueError(f"add_site: body '{body_name}' not found")
        name = op.get("name")
        if not name:
            raise ValueError("add_site requires 'name'")
        site_kwargs: dict[str, Any] = {
            "name": name,
            "pos": _finite_vec("add_site", "pos", op.get("pos", [0.0, 0.0, 0.0]), 3),
        }
        if "size" in op:
            site_kwargs["size"] = op["size"]
        if "rgba" in op:
            site_kwargs["rgba"] = op["rgba"]
        body.add_site(**site_kwargs)
        return

    if kind == "set_body_pos":
        name = op.get("name")
        if not name:
            raise ValueError("set_body_pos requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"set_body_pos: body '{name}' not found")
        body.pos = _finite_vec("set_body_pos", "pos", op.get("pos", [0.0, 0.0, 0.0]), 3)
        return

    if kind == "set_body_quat":
        name = op.get("name")
        if not name:
            raise ValueError("set_body_quat requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"set_body_quat: body '{name}' not found")
        body.quat = _finite_quat("set_body_quat", op.get("quat", [1.0, 0.0, 0.0, 0.0]))
        return

    if kind == "delete_body":
        name = op.get("name")
        if not name:
            raise ValueError("delete_body requires 'name'")
        body = _find_body(spec, name, new_bodies)
        if body is None:
            raise ValueError(f"delete_body: body '{name}' not found")
        spec.delete(body)
        new_bodies.pop(name, None)
        return


def patch_scene_mjcf(world: SimWorld, ops: list[dict[str, Any]]) -> int:
    """Apply a sequence of structured ops to the live spec in order.

    Each op is a small dict like::

        {"op": "add_body", "parent": "world", "name": "foo", "pos": [0,0,1]}
        {"op": "add_geom", "body": "foo", "type": "sphere", "size": [0.1]}
        {"op": "set_body_pos", "name": "foo", "pos": [1,0,1]}
        {"op": "delete_body", "name": "foo"}

    The list is applied atomically: if any op raises, the whole patch is
    rejected and the world is left in its original state. After all ops
    succeed, ``spec.recompile(model, data)`` is called once, so joint
    qpos/qvel for unchanged joints are preserved automatically.

    Returns the number of ops applied (same as ``len(ops)`` on success).
    """
    if not isinstance(ops, list):
        raise ValueError(f"ops must be a list, got {type(ops).__name__}")
    if not ops:
        return 0

    spec = world._backend_state.get("spec")
    if spec is None:
        raise RuntimeError("world has no spec; patch_scene_mjcf requires a compiled world")

    # Snapshot so a failed op can be atomically rejected. Use the native deep
    # copy: an XML round-trip is NOT safe here, because to_xml() writes mesh refs
    # as bare filenames and drops meshdir/assets, so restoring it for any
    # mesh-based robot yields a spec that cannot compile - leaving the world
    # permanently un-mutable (every later scene op fails) while the model keeps
    # stepping. The round-trip also silently reset joint state.
    try:
        backup_spec = spec.copy()
    except (ValueError, RuntimeError) as e:  # pragma: no cover - copy of a live spec is fine
        raise RuntimeError(f"failed to snapshot spec before patch: {e}") from e

    applied = 0
    new_bodies: dict[str, Any] = {}
    # Snapshot the object poses the mirror below may touch, so a rejected batch
    # rolls the mirror back with the spec. Without this the spec reverted while
    # the SimObject kept the refused pose, and the next full rebuild applied a
    # patch the caller was told had failed.
    pose_backup = {name: (list(obj.position), list(obj.orientation)) for name, obj in world.objects.items()}
    try:
        for op in ops:
            _apply_patch_op(spec, op, new_bodies)
            # A pose patch edits the SPEC. The incremental recompile below keeps
            # that, but a later FULL rebuild (eject_robot_from_scene, triggered by
            # removing ANY robot) reconstructs objects from ``world.objects`` and
            # so reverted the patch: a body moved to [1.2, 0.3, 0.5] snapped back
            # to its original [0.4, 0, 0.3]. Mirror the new pose onto the tracked
            # object so the declarative rebuild reproduces it.
            _sync_patched_object_pose(world, op)
            applied += 1
    except Exception as err:
        # Restore the pre-patch spec so a rejected batch leaves nothing behind.
        world._backend_state["spec"] = backup_spec
        for name, (pos, quat) in pose_backup.items():
            obj = world.objects.get(name)
            if obj is not None:
                obj.position = pos
                obj.orientation = quat
        raise ValueError(f"patch op #{applied + 1} failed: {err}") from err

    # One recompile for the whole batch - preserves qpos/qvel for unchanged joints.
    with filter_mujoco_attach_noise():
        patched_model, patched_data = spec.recompile(world._model, world._data)
    # Installs and forwards so new bodies' xpos / xquat / cam_xmat are populated
    # for the very next render() or get_body_state() call.
    _install_model(world, patched_model, patched_data)

    # A pose patch on a FREE-JOINTED body needs its live qpos written too. The
    # recompile faithfully preserves the old qpos, and for a free body that qpos
    # - not the body's rest pose - is what positions it, so the patch had no
    # visible effect until the next reset happened to re-seed qpos from qpos0:
    #
    #     patch set_body_pos c -> [0.7, 0.2, 0.4]   status=success
    #     get_body_state("c")  -> pos [0.4, 0.0, 0.3]     unmoved
    #     list_objects()       -> "c: box at [0.7, 0.2, 0.4]"
    #     reset()              -> pos [0.7, 0.2, 0.4]     moves only now
    #
    # Two tool actions therefore disagreed about where the object was. Static
    # bodies were always fine (no freejoint, so the rest pose IS the pose), which
    # is what made the dynamic case easy to miss.
    _apply_patched_free_body_poses(world, ops)

    # A patch can add or DELETE bodies, so the cached per-robot joint/actuator
    # ids may no longer describe the model. ``delete_body`` accepts any body name
    # including a robot link, and deleting one left every id stale - some of them
    # past the end of the new, smaller arrays, which raised IndexError out of
    # ``zero_dynamics``. Re-resolve by name so the ids always match the model.
    _rediscover_robot_ids(world)

    _sync_cached_xml(world, spec)
    return applied
