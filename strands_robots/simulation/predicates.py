"""Named-predicate library for declarative ``BenchmarkProtocol`` specs.

Each entry in :data:`PREDICATE_REGISTRY` is a factory ``(**kwargs) -> callable``
where the returned callable takes a ``SimEngine`` and returns ``bool`` (for
success/failure predicates) or ``float`` (for reward terms).

The registry is a closed set: the YAML/JSON spec loader refuses names not in
this registry, so spec files are safe to parse from untrusted / LLM-authored
input and no ``eval`` is ever called. User-defined predicates must be
registered via :func:`register_predicate` before loading the spec.

Predicates only call ``SimEngine`` methods (abstract) or probe for MuJoCo-only
methods via ``getattr``, returning a safe fallback (``False`` / ``0.0``) when
the backend does not support them. When the backend supports a lookup but the
referenced ``body``/``joint`` name cannot be resolved (almost always a spec
typo), the term still degrades to a constant but the name is logged once at
WARNING.

Available predicates (bool):

    body_above_z(body, z)
    body_below_z(body, z)
    joint_above(joint, value)
    joint_below(joint, value)
    distance_less_than(body_a, body_b, threshold)
    inside_region(body, min, max)
    contact_between(geom_a, geom_b)
    contact_any()
    body_on(body_a, body_b, z_offset=0.02, xy_tol=0.15)
    body_inside(body, container, xy_tol=0.15, z_tol=0.15)
    body_upright(body, tol=0.15)
    grasped(body, gripper_prefix)
    base_tipped(tol=0.15, robot=None)
    base_below_z(z, robot=None)
    base_beyond_x(x, robot=None)
    base_beyond_y(y, robot=None)
    base_yaw_beyond(yaw, robot=None)

Available reward terms (float):

    distance_neg(body_a, body_b, weight=1.0)
    joint_progress(joint, target, weight=1.0)
    base_velocity(vx=0.0, vy=0.0, wz=0.0, weight=1.0, robot=None)
    base_velocity_tracking(vx=0.0, vy=0.0, wz=0.0, lin_weight=1.0, ang_weight=0.5, tracking_sigma=0.25, robot=None)
    base_height(target, weight=1.0, robot=None)
    base_orientation(weight=1.0, robot=None)
    base_lin_vel_z(weight=1.0, robot=None)
    base_ang_vel_xy(weight=1.0, robot=None)
    staged_reward(stages)
    constant(value)
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

BoolPredicate = Callable[["SimEngine"], bool]
RewardTerm = Callable[["SimEngine"], float]
PredicateFactory = Callable[..., Callable[["SimEngine"], Any]]


# Names the DSL has already warned about, so a broken spec cannot spam the
# reward/eval hot loop. Keyed by (kind, name); process-global and deduplicated.
_RESOLUTION_WARNED: set[tuple[str, str]] = set()


def _warn_unresolved(kind: str, name: str, tried: tuple[str, ...] = ()) -> None:
    """Warn once that a spec references an entity the sim cannot resolve.

    Called only when the backend supports the lookup but the named body/joint
    is not found - almost always a spec typo. The term still degrades to a
    constant (bool -> False, reward -> 0.0); this only surfaces the name. A
    missing lookup method (unsupported backend) is a capability gap and stays
    silent.
    """
    key = (kind, name)
    if key in _RESOLUTION_WARNED:
        return
    _RESOLUTION_WARNED.add(key)
    extra = f" (tried {list(tried)})" if len(tried) > 1 else ""
    logger.warning(
        "predicate/reward DSL: %s %r is not present in the simulation%s; the "
        "referencing term degrades to a constant (bool predicate -> False, "
        "reward -> 0.0), which silently prevents success / yields a dead reward. "
        "Check the name against the loaded scene / benchmark spec.",
        kind,
        name,
        extra,
    )


def _reset_resolution_warnings() -> None:
    """Clear the one-time-warning dedup cache (test isolation)."""
    _RESOLUTION_WARNED.clear()


# Helpers for digging values out of the structured ``{"status", "content"}``
# dicts that MuJoCo-backend methods return. Defensive against empty content
# lists and missing keys - predicates should never crash the eval loop.


def _extract_json(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``json`` content block payload, or ``{}`` if absent."""
    if not isinstance(result, dict):
        return {}
    for block in result.get("content", []) or []:
        if isinstance(block, dict):
            payload = block.get("json")
            if isinstance(payload, dict):
                # Copy into a new dict so mypy keeps it typed as dict[str, Any].
                return dict(payload)
    return {}


#: Minimum contact normal force (newtons) to count as touching; rejects only
#: the exact-zero records MuJoCo emits for geom pairs inside ``margin``/``gap``.
_CONTACT_FORCE_EPS = 1e-9


def _load_bearing_contacts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The contacts in a ``get_contacts`` payload that actually carry load.

    Drops solver-excluded records (``exclude != 0``) and zero-force proximity
    records, both of which MuJoCo reports whenever a geom declares ``margin``/
    ``gap``; without the filter, contact-gated predicates can fire in mid-air.
    Payloads lacking the ``exclude``/``normal_force`` keys degrade to
    geometry-only behaviour so older engines keep working.
    """
    contacts = payload.get("contacts")
    if not isinstance(contacts, list):
        return []
    out: list[dict[str, Any]] = []
    for c in contacts:
        if not isinstance(c, dict):
            continue
        if int(c.get("exclude", 0) or 0) != 0:
            continue
        force = c.get("normal_force")
        if force is not None and abs(float(force)) <= _CONTACT_FORCE_EPS:
            continue
        out.append(c)
    return out


def _contact_bodies(contact: dict[str, Any]) -> set[str]:
    """Every name a contact can be addressed by: geom names plus parent bodies.

    ``get_contacts`` synthesizes names like ``"plate_1/geom_3"`` for unnamed
    geoms, so the explicit ``body1``/``body2`` fields are included to let
    body-level checks match without parsing synthesized strings.
    """
    names = {contact.get("geom1"), contact.get("geom2"), contact.get("body1"), contact.get("body2")}
    return {n for n in names if isinstance(n, str) and n}


def _body_position(sim: SimEngine, body: str) -> list[float] | None:
    """Best-effort body-position lookup. Returns ``None`` on any failure.

    Requires the backend to implement ``get_body_state`` (MuJoCo only at time
    of writing). Tries the bare name first, then the LIBERO ``<name>_main``
    root-body convention (BDDL names objects without the ``_main`` suffix the
    MJCF root body carries); warns once if neither resolves.
    """
    get_body_state = getattr(sim, "get_body_state", None)
    if get_body_state is None:
        return None

    def _try(name: str) -> list[float] | None:
        try:
            result = get_body_state(body_name=name)
        except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
            logger.debug("body_position(%r) failed: %s", name, e)
            return None
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        payload = _extract_json(result)
        pos = payload.get("position")
        if isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos):
            return [float(c) for c in pos]
        return None

    # Bare name first, then the LIBERO ``<name>_main`` root-body convention
    # (skip if already suffixed).
    pos = _try(body)
    if pos is not None:
        return pos
    tried = [body]
    if not body.endswith("_main"):
        tried.append(f"{body}_main")
        pos = _try(f"{body}_main")
        if pos is not None:
            return pos
    _warn_unresolved("body", body, tuple(tried))
    return None


def _joint_position(sim: SimEngine, joint: str) -> float | None:
    """Best-effort joint-position lookup, preferring a direct joint read.

    Probes the backend's ``get_joint_state`` first because ``get_observation``
    only enumerates a *registered robot's* joints: an articulated scene joint (a
    drawer slide, a door or cabinet hinge - the objects these predicates exist to
    score) never appears there, so the lookup returned ``None`` and the term
    silently degraded to ``False`` / ``0.0``. A physically open drawer therefore
    failed its own success threshold, and every LIBERO ``(open X)`` /
    ``(closed X)`` goal - which compiles to these predicates - scored a
    permanent 0%.

    Falls back to the observation dict when the backend has no direct accessor,
    so a backend without one keeps working (robot joints appear in both).
    Returns ``None`` when neither resolves, warning once if the backend answered
    but the joint is absent (a typo, not a capability gap).
    """
    get_joint_state = getattr(sim, "get_joint_state", None)
    if get_joint_state is not None:
        try:
            result = get_joint_state(joint)
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("get_joint_state(%r) failed: %s", joint, e)
        else:
            if isinstance(result, dict) and result.get("status") == "success":
                pos = _extract_json(result).get("position")
                # A multi-DOF joint reports a list; these predicates compare a
                # scalar, so only a 1-DOF joint is answerable here.
                if isinstance(pos, (int, float)) and not isinstance(pos, bool):
                    return float(pos)

    try:
        obs = sim.get_observation(skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive
        logger.debug("get_observation() failed: %s", e)
        return None
    if not isinstance(obs, dict):
        return None
    val = obs.get(joint)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if obs and joint not in obs:
        _warn_unresolved("joint", joint)
    return None


def _body_quaternion(sim: SimEngine, body: str) -> list[float] | None:
    """Best-effort quaternion lookup. Returns ``None`` on any failure.

    Quaternion convention: MuJoCo reports ``[w, x, y, z]``. Name resolution
    mirrors ``_body_position`` (bare name, then the LIBERO ``<name>_main``
    root-body fallback); warns once if neither resolves.
    """
    get_body_state = getattr(sim, "get_body_state", None)
    if get_body_state is None:
        return None

    def _try(name: str) -> list[float] | None:
        try:
            result = get_body_state(body_name=name)
        except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
            logger.debug("body_quaternion(%r) failed: %s", name, e)
            return None
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        payload = _extract_json(result)
        quat = payload.get("quaternion")
        if isinstance(quat, list) and len(quat) == 4 and all(isinstance(c, (int, float)) for c in quat):
            return [float(c) for c in quat]
        return None

    quat = _try(body)
    if quat is not None:
        return quat
    tried = [body]
    if not body.endswith("_main"):
        tried.append(f"{body}_main")
        quat = _try(f"{body}_main")
        if quat is not None:
            return quat
    _warn_unresolved("body", body, tuple(tried))
    return None


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Simple 3D Euclidean distance; no numpy so predicates stay dependency-free."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _quat_rotate_inverse_wxyz(quat_wxyz: list[float], vec: list[float]) -> list[float]:
    """Express a WORLD-frame 3-vector in the body frame given a (w,x,y,z) quaternion.

    Computes ``R(q)^T @ vec`` in pure Python (no numpy). A near-zero-norm
    quaternion returns ``vec`` unchanged.
    """
    w, x, y, z = (float(c) for c in quat_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm < 1e-8:
        return [float(v) for v in vec]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = (float(c) for c in vec)
    two_w = 2.0 * w
    s = 2.0 * w * w - 1.0
    # b = cross(q_vec, v); term = v * s - b * 2w + q_vec * (q_vec . v) * 2
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    d = 2.0 * (x * vx + y * vy + z * vz)
    return [
        vx * s - cx * two_w + x * d,
        vy * s - cy * two_w + y * d,
        vz * s - cz * two_w + z * d,
    ]


def _base_twist(sim: SimEngine, robot: str | None) -> tuple[float, float, float] | None:
    """Return a floating base's BODY-frame planar twist ``(vx, vy, wz)``, or None.

    ``base_lin_vel`` (world frame) is rotated into the base frame via
    ``base_quat`` so ``vx``/``vy`` are the forward/lateral velocity in the
    robot's own heading; ``base_ang_vel`` is already body-frame so its z is
    the yaw rate. This is the frame a locomotion velocity command is expressed
    against (IsaacLab / legged_gym convention). Returns None (warning once)
    when the robot exposes no floating base (e.g. a fixed-base arm).
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_velocity get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    lin = obs.get("base_lin_vel")
    quat = obs.get("base_quat")
    ang = obs.get("base_ang_vel")
    if not (
        isinstance(lin, list)
        and len(lin) == 3
        and isinstance(quat, list)
        and len(quat) == 4
        and isinstance(ang, list)
        and len(ang) == 3
    ):
        # No floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    v_body = _quat_rotate_inverse_wxyz(quat, lin)
    return float(v_body[0]), float(v_body[1]), float(ang[2])


def _base_body_velocity(sim: SimEngine, robot: str | None) -> tuple[list[float], list[float]] | None:
    """Return a floating base's BODY-frame ``(linear_velocity, angular_velocity)``, or None.

    Both twists are expressed in the base frame: world-frame ``base_lin_vel``
    is rotated via ``base_quat`` (so its z is the vertical velocity along the
    base's OWN up-axis) and ``base_ang_vel`` is already body-frame (its xy are
    the roll/pitch rates). Returns None (warning once) when the robot exposes
    no floating base.
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base motion get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    lin = obs.get("base_lin_vel")
    quat = obs.get("base_quat")
    ang = obs.get("base_ang_vel")
    if not (
        isinstance(lin, list)
        and len(lin) == 3
        and isinstance(quat, list)
        and len(quat) == 4
        and isinstance(ang, list)
        and len(ang) == 3
    ):
        # No floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    v_body = _quat_rotate_inverse_wxyz(quat, lin)
    return (
        [float(v_body[0]), float(v_body[1]), float(v_body[2])],
        [float(ang[0]), float(ang[1]), float(ang[2])],
    )


def _base_position(sim: SimEngine, robot: str | None) -> list[float] | None:
    """Return a floating base's WORLD position ``[x, y, z]``, or None.

    Reads ``get_observation``'s ``base_pos`` signal. Returns None (warning
    once) when the robot exposes no floating base (e.g. a fixed-base arm).
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_height get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    pos = obs.get("base_pos")
    if not (isinstance(pos, list) and len(pos) == 3):
        # No floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _base_quaternion(sim: SimEngine, robot: str | None) -> list[float] | None:
    """Return a floating base's orientation quaternion ``[w, x, y, z]``, or None.

    Reads ``get_observation``'s ``base_quat`` (both backends report
    ``[w, x, y, z]``). Returns None (warning once) when the robot exposes no
    floating base (e.g. a fixed-base arm).
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_orientation get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    quat = obs.get("base_quat")
    if not (isinstance(quat, list) and len(quat) == 4):
        # No floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


# Predicate factories


def _body_above_z(body: str, z: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        return pos is not None and pos[2] > float(z)

    return check


def _body_below_z(body: str, z: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        return pos is not None and pos[2] < float(z)

    return check


def _joint_above(joint: str, value: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        q = _joint_position(sim, joint)
        return q is not None and q > float(value)

    return check


def _joint_below(joint: str, value: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        q = _joint_position(sim, joint)
        return q is not None and q < float(value)

    return check


def _distance_less_than(body_a: str, body_b: str, threshold: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return False
        return _euclidean_distance(pos_a, pos_b) < float(threshold)

    return check


def _inside_region(body: str, min: list[float], max: list[float]) -> BoolPredicate:  # noqa: A002 - DSL keyword
    if not (isinstance(min, list) and len(min) == 3 and isinstance(max, list) and len(max) == 3):
        raise ValueError("inside_region: 'min' and 'max' must each be a list of 3 numbers")
    lo = [float(c) for c in min]
    hi = [float(c) for c in max]
    if any(lo[i] > hi[i] for i in range(3)):
        raise ValueError(f"inside_region: 'min' {lo} must be component-wise <= 'max' {hi}")

    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        if pos is None:
            return False
        return all(lo[i] <= pos[i] <= hi[i] for i in range(3))

    return check


def _contact_between(geom_a: str, geom_b: str) -> BoolPredicate:
    """Pairwise contact predicate; order-insensitive. Requires ``get_contacts()`` (MuJoCo)."""

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("contact_between(%r,%r) failed: %s", geom_a, geom_b, e)
            return False
        want = {geom_a, geom_b}
        for c in _load_bearing_contacts(_extract_json(result)):
            if want <= _contact_bodies(c):
                return True
        return False

    return check


def _contact_any() -> BoolPredicate:
    """Sparse "any contact" predicate - matches the legacy ``success_fn='contact'`` path."""

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("contact_any() failed: %s", e)
            return False
        return bool(_load_bearing_contacts(_extract_json(result)))

    return check


def _body_contact(sim: SimEngine, body_a: str, body_b: str) -> bool | None:
    """Best-effort body-contact lookup.

    Returns ``True``/``False`` when ``sim.get_contacts()`` is available AND
    any geom of ``body_a`` touches any geom of ``body_b``; returns ``None``
    when the lookup is unavailable (or the payload malformed) so callers can
    degrade to geometric-only checks. Contacts are matched on the explicit
    parent bodies first, then on the ``<body>_g...`` / ``<body>/...`` geom
    naming conventions. Used by the contact-aware branch of :func:`_body_on`.
    """
    get_contacts = getattr(sim, "get_contacts", None)
    if get_contacts is None:
        return None
    try:
        result = get_contacts()
    except Exception as e:  # noqa: BLE001 - defensive
        logger.debug("body_contact(%r, %r) get_contacts raised: %s", body_a, body_b, e)
        return None
    if not isinstance(result, dict) or result.get("status") != "success":
        # Error stub / malformed payload: "unknown", not False, so the caller
        # can degrade to the geometric-only check.
        return None
    payload = _extract_json(result)
    if not isinstance(payload.get("contacts"), list):
        return None
    contacts = _load_bearing_contacts(payload)

    def _matches(name: str, body: str) -> bool:
        return name == body or name.startswith(f"{body}_g") or name.startswith(f"{body}/")

    for c in contacts:
        names = _contact_bodies(c)
        if any(_matches(n, body_a) for n in names) and any(_matches(n, body_b) for n in names):
            return True
    return False


def _body_on(
    body_a: str,
    body_b: str,
    z_offset: float = 0.02,
    xy_tol: float = 0.15,
    require_contact: bool = False,
) -> BoolPredicate:
    """Approximate ``(on A B)`` predicate - A resting on top of B.

    True when ``A.z > B.z + z_offset`` AND horizontal distance
    ``|A.xy - B.xy| < xy_tol``. With ``require_contact=True`` it also requires
    physics contact between A and B (upstream LIBERO ``check_ontop``
    semantics); an engine without ``get_contacts`` skips the contact check and
    keeps the geometric verdict. ``z_offset`` accounts for B's half-height
    plus a small buffer; tune per scene. For full geometric fidelity, register
    a scene-specific predicate via :func:`register_predicate`.
    """

    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return False
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        if (dx * dx + dy * dy) ** 0.5 > float(xy_tol):
            return False
        if not (pos_a[2] > pos_b[2] + float(z_offset)):
            return False
        if require_contact:
            in_contact = _body_contact(sim, body_a, body_b)
            # None => engine cannot check contacts; keep the geometric verdict.
            if in_contact is False:
                return False
        return True

    return check


def _body_inside(body: str, container: str, xy_tol: float = 0.15, z_tol: float = 0.15) -> BoolPredicate:
    """Approximate ``(in A B)`` predicate - A contained within B's volume.

    True when A's position is within an axis-aligned box centered on B with
    half-extents (``xy_tol``, ``xy_tol``, ``z_tol``). Defaults are tuned for
    table-top manipulation; register a scene-specific predicate for richer
    geometry.
    """

    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body)
        pos_b = _body_position(sim, container)
        if pos_a is None or pos_b is None:
            return False
        return (
            abs(pos_a[0] - pos_b[0]) <= float(xy_tol)
            and abs(pos_a[1] - pos_b[1]) <= float(xy_tol)
            and abs(pos_a[2] - pos_b[2]) <= float(z_tol)
        )

    return check


def _body_upright(body: str, tol: float = 0.15) -> BoolPredicate:
    """True when ``body``'s local +Z axis is within ``tol`` of world +Z.

    For a unit quaternion ``R[2,2] = 1 - 2*(x^2 + y^2)``, so the check is
    ``2*(x^2 + y^2) < tol`` - monotonic in tilt, so a small tol (0.01-0.2)
    bounds the maximum allowed tilt directly.
    """
    t = float(tol)
    if t < 0:
        raise ValueError(f"body_upright: 'tol' must be >= 0, got {t}")

    def check(sim: SimEngine) -> bool:
        quat = _body_quaternion(sim, body)
        if quat is None:
            return False
        # MuJoCo quat layout is (w, x, y, z).
        _, x, y, _ = quat
        return 2.0 * (x * x + y * y) < t

    return check


def _geom_belongs_to_body(geom: str, body: str) -> bool:
    """True when geom name ``geom`` is one of ``body``'s geoms.

    Matches the exact body name or the ``<body>_g`` prefix, which covers both
    ``<body>_geom`` (strands ``add_object``) and ``<body>_g<idx>``
    (LIBERO/robosuite). The ``_g`` boundary keeps distinct names apart
    (``cube_1_g`` does not match ``cube_10_g0``).
    """
    return geom == body or geom.startswith(f"{body}_g")


def _grasped(body: str, gripper_prefix: str) -> BoolPredicate:
    """True when ``body`` contacts any geom whose name starts with ``gripper_prefix``.

    The gripper is treated as a set of geoms (fingers, pads, tip sites), so
    the prefix (e.g. ``"robot0_gripper"`` for Panda) covers all of them. Body
    geoms are matched via :func:`_geom_belongs_to_body`'s naming conventions.
    Backends without ``get_contacts()`` return ``False``.
    """

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("grasped(%r, %r) failed: %s", body, gripper_prefix, e)
            return False
        payload = _extract_json(result)
        contacts = payload.get("contacts")
        if not isinstance(contacts, list):
            return False
        for c in contacts:
            if not isinstance(c, dict):
                continue
            g1 = c.get("geom1") or ""
            g2 = c.get("geom2") or ""
            # One side must be a geom of the body, the other a gripper geom.
            body_match = _geom_belongs_to_body(g1, body) or _geom_belongs_to_body(g2, body)
            gripper_match = any(isinstance(g, str) and g.startswith(gripper_prefix) for g in (g1, g2))
            if body_match and gripper_match:
                return True
        return False

    return check


def _base_tipped(tol: float = 0.15, robot: str | None = None) -> BoolPredicate:
    """True when a floating base has tilted more than ``tol`` from level.

    The fall-over termination for locomotion tasks - put it in a ``failure``
    clause so a rollout ends when the robot topples. Reads ``base_quat`` from
    ``get_observation``, so it needs no base body name (unlike
    ``body_upright``). Tipped when ``2*(x**2 + y**2) > tol``, the complement
    of ``body_upright``'s check; ``2*(x**2 + y**2) = 1 - cos(theta)`` for a
    roll/pitch of ``theta``, so ``tol=0.15`` trips at ~32 deg and ``tol=1.0``
    at 90 deg - a fall-over termination typically uses ~0.7-1.0. A fixed-base
    arm has no base orientation, so the predicate degrades to ``False``
    (logged once). ``robot`` selects the robot in a multi-robot scene
    (default: the sole robot).
    """
    t = float(tol)
    if t < 0:
        raise ValueError(f"base_tipped: 'tol' must be >= 0, got {t}")
    rname = robot

    def check(sim: SimEngine) -> bool:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return False
        # base_quat layout is (w, x, y, z) on both backends.
        _, x, y, _ = quat
        return 2.0 * (x * x + y * y) > t

    return check


def _ground_height(sim: SimEngine, x: float, y: float) -> float:
    """Local terrain surface height (world z) beneath ``(x, y)``; ``0.0`` on flat ground.

    Lets height/fall terms measure clearance above the terrain beneath the
    base instead of absolute world z (which misses a collapse on a raised
    heightfield). Reads the backend's ``_ground_height_at`` hook; a sim
    lacking the hook degrades to ``0.0``. Never raises.
    """
    fn = getattr(sim, "_ground_height_at", None)
    if fn is None:
        return 0.0
    try:
        return float(fn(x, y))
    except Exception as e:  # noqa: BLE001 - predicates never raise
        logger.debug("_ground_height_at(%.3f, %.3f) failed: %s", x, y, e)
        return 0.0


def _base_below_z(z: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's height ABOVE THE LOCAL GROUND drops below ``z``.

    The collapse counterpart of :func:`_base_tipped`; put both in a
    ``failure`` clause so a rollout ends when the robot falls over OR drops to
    the floor::

        failure:
          any:
            - {predicate: base_tipped, tol: 0.7}
            - {predicate: base_below_z, z: 0.3}

    Reads ``base_pos`` from ``get_observation``, so it needs no base body name
    (unlike ``body_below_z``). The height is measured above the local terrain
    beneath the base (``0.0`` on flat ground), so a collapse on a raised
    heightfield is still caught. ``z`` is the collapse clearance in metres;
    set it well below the standing base height (a G1 pelvis stands ~0.74 m, so
    ``z=0.3`` catches a collapse). A fixed-base arm has no base position, so
    the predicate degrades to ``False`` (logged once). ``robot`` selects the
    robot in a multi-robot scene (default: the sole robot).
    """
    zt = float(z)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        # Height above the local terrain, not absolute world z.
        return (pos[2] - _ground_height(sim, pos[0], pos[1])) < zt

    return check


def _base_beyond_x(x: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's world x-position has passed forward of ``x``.

    The forward-progress SUCCESS predicate for a walk-forward locomotion task;
    pair it with the fall predicates (``base_tipped`` / ``base_below_z``) in a
    ``failure`` clause. Reads ``base_pos`` from ``get_observation``, so it
    needs no base body name (unlike ``inside_region``). ``x`` is an ABSOLUTE
    world x-threshold in metres, not a displacement: the canonical scene
    spawns near the origin facing +x, so ``base_beyond_x(x=D)`` reads "walked
    ~D metres forward"; set ``x`` relative to the known spawn x. Pure
    x-position test - height and orientation do not affect it. A fixed-base
    arm has no base position, so the predicate degrades to ``False`` (logged
    once). ``robot`` selects the robot in a multi-robot scene (default: the
    sole robot).
    """
    xt = float(x)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        return pos[0] > xt

    return check


def _base_beyond_y(y: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's world y-position has passed beyond ``y``.

    The lateral (strafe-left) counterpart of :func:`_base_beyond_x`: world +y
    is the robot's left for the identity spawn orientation the locomotion
    scenes use. ``y`` is an ABSOLUTE world y-threshold in metres, not a
    displacement; set it relative to the known spawn y. Pure y-position test -
    height and orientation do not affect it, so pair it with the fall
    predicates in a ``failure`` clause. A fixed-base arm degrades to ``False``
    (logged once). ``robot`` selects the robot in a multi-robot scene
    (default: the sole robot).
    """
    yt = float(y)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        return pos[1] > yt

    return check


def _base_yaw_beyond(yaw: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base has turned past a heading of ``yaw`` radians.

    The heading (turn-in-place) SUCCESS counterpart of :func:`_base_beyond_x`
    / :func:`_base_beyond_y`, extracted from ``base_quat`` (``w, x, y, z``)::

        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    ``yaw`` is an ABSOLUTE world heading in radians measured from the identity
    spawn (yaw 0); positive is a LEFT (counter-clockwise) turn, so ``yaw=1.0``
    reads "turned ~1 rad (~57 deg) to the left". The yaw wraps at +-pi, so a
    goal at or beyond pi is not a well-defined single-turn heading. Pure
    heading test - roll/pitch, position and height do not affect it; pair it
    with ``base_tipped`` in a ``failure`` clause since the yaw of a toppled
    base is ill-defined. A fixed-base arm degrades to ``False`` (logged once).
    ``robot`` selects the robot in a multi-robot scene (default: the sole
    robot).
    """
    yt = float(yaw)
    rname = robot

    def check(sim: SimEngine) -> bool:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return False
        # base_quat layout is (w, x, y, z) on both backends.
        w, x, y, z = quat
        heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return heading > yt

    return check


# Reward terms (float-valued)


def _distance_neg(body_a: str, body_b: str, weight: float = 1.0) -> RewardTerm:
    """Negative Euclidean distance between two bodies: ``weight * -dist(a, b)``.

    The canonical "reach" reward - monotonic, pulls the bodies together.
    """
    w = float(weight)

    def term(sim: SimEngine) -> float:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return 0.0
        return -w * _euclidean_distance(pos_a, pos_b)

    return term


def _joint_progress(joint: str, target: float, weight: float = 1.0) -> RewardTerm:
    """Negative absolute distance from a joint to its target, weighted.

    Dense signal for drawer/door tasks where success is "joint near target".
    """
    w = float(weight)
    t = float(target)

    def term(sim: SimEngine) -> float:
        q = _joint_position(sim, joint)
        if q is None:
            return 0.0
        return -w * abs(q - t)

    return term


def _constant(value: float) -> RewardTerm:
    """Constant reward per step. Useful for shaping a survival bonus."""
    v = float(value)

    def term(_sim: SimEngine) -> float:
        return v

    return term


def _base_velocity(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    weight: float = 1.0,
    robot: str | None = None,
) -> RewardTerm:
    """Negative base velocity-tracking error - unbounded L2 locomotion reward.

    ``-weight * ||(v_body_x, v_body_y, w_body_z) - (vx, vy, wz)||`` with the
    command in the BODY frame: ``vx`` forward, ``vy`` lateral (m/s in the
    robot's own heading) and ``wz`` the yaw rate (rad/s). 0 at perfect
    tracking, more negative with error. The tracked twist is heading-relative
    (see ``_base_twist``). A fixed-base arm has no base twist, so the term
    degrades to ``0.0`` (logged once). ``robot`` selects the robot in a
    multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    tvx, tvy, twz = float(vx), float(vy), float(wz)
    rname = robot

    def term(sim: SimEngine) -> float:
        twist = _base_twist(sim, rname)
        if twist is None:
            return 0.0
        bvx, bvy, bwz = twist
        dvx, dvy, dwz = bvx - tvx, bvy - tvy, bwz - twz
        return -w * float((dvx * dvx + dvy * dvy + dwz * dwz) ** 0.5)

    return term


def _base_velocity_tracking(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    lin_weight: float = 1.0,
    ang_weight: float = 0.5,
    tracking_sigma: float = 0.25,
    robot: str | None = None,
) -> RewardTerm:
    """Bounded exponential-kernel velocity-tracking reward (legged_gym convention).

    A POSITIVE, BOUNDED signal that peaks when the base matches a commanded
    BODY-frame velocity (``vx`` forward, ``vy`` lateral in m/s, ``wz`` yaw
    rate in rad/s)::

        lin_weight * exp(-((v_body_x - vx)**2 + (v_body_y - vy)**2) / tracking_sigma)
      + ang_weight * exp(-(w_body_z - wz)**2 / tracking_sigma)

    the sum of legged_gym's ``tracking_lin_vel`` and ``tracking_ang_vel``
    terms with their canonical defaults (weights 1.0 / 0.5, sigma 0.25).
    Bounded to ``[0, lin_weight + ang_weight]``, maximal at perfect tracking -
    unlike the unbounded :func:`_base_velocity`, it saturates near the command
    and stays well-scaled against the bounded regularizer terms. Command frame
    and floating-base handling match ``base_velocity`` (a fixed-base arm
    degrades to ``0.0``, logged once). Raises ``ValueError`` if
    ``tracking_sigma`` <= 0.
    """
    tvx, tvy, twz = float(vx), float(vy), float(wz)
    lw, aw = float(lin_weight), float(ang_weight)
    sigma = float(tracking_sigma)
    if sigma <= 0:
        raise ValueError(f"base_velocity_tracking: 'tracking_sigma' must be > 0, got {sigma}")
    rname = robot

    def term(sim: SimEngine) -> float:
        twist = _base_twist(sim, rname)
        if twist is None:
            return 0.0
        bvx, bvy, bwz = twist
        lin_err = (bvx - tvx) ** 2 + (bvy - tvy) ** 2
        ang_err = (bwz - twz) ** 2
        return lw * math.exp(-lin_err / sigma) + aw * math.exp(-ang_err / sigma)

    return term


def _base_height(target: float, weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared base-height error - anti-crouch locomotion regularizer.

    ``-weight * ((base_z - ground_z) - target) ** 2``, the legged_gym /
    IsaacLab ``base_height`` term; pairs with ``base_velocity``, which alone
    rewards diving/crouching. The height is measured ABOVE THE LOCAL GROUND
    beneath the base (``0.0`` on flat ground), so raised terrain is not
    spuriously penalised. ``target`` is the desired base height in metres
    above the ground (e.g. G1 pelvis ~0.74 m, Go2 trunk ~0.34 m). A fixed-base
    arm has no base position, so the term degrades to ``0.0`` (logged once).
    ``robot`` selects the robot in a multi-robot scene (default: the sole
    robot).
    """
    w = float(weight)
    tgt = float(target)
    rname = robot

    def term(sim: SimEngine) -> float:
        pos = _base_position(sim, rname)
        if pos is None:
            return 0.0
        # Height above the local terrain, not absolute world z.
        d = (pos[2] - _ground_height(sim, pos[0], pos[1])) - tgt
        return -w * d * d

    return term


def _base_orientation(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative flat-orientation error - anti-lean locomotion regularizer.

    ``-weight * (g_x ** 2 + g_y ** 2)`` where ``(g_x, g_y, g_z)`` is gravity
    expressed in the base frame ("projected gravity"): 0 when level, growing
    as ``sin(theta) ** 2`` for a roll/pitch of ``theta``. Invariant to YAW, so
    the robot may turn freely. The legged_gym / IsaacLab ``orientation`` term;
    pairs with ``base_height`` to stop a velocity-tracking policy cheating by
    leaning or crouching. A fixed-base arm has no base orientation, so the
    term degrades to ``0.0`` (logged once). ``robot`` selects the robot in a
    multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return 0.0
        gx, gy, _gz = _quat_rotate_inverse_wxyz(quat, [0.0, 0.0, -1.0])
        return -w * float(gx * gx + gy * gy)

    return term


def _base_lin_vel_z(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared vertical base velocity - anti-bounce regularizer.

    ``-weight * v_body_z ** 2`` where ``v_body_z`` is the base's linear
    velocity along its OWN up-axis. The legged_gym / IsaacLab ``lin_vel_z``
    term; complements ``base_height``, which penalises a static height offset
    but misses bouncing whose mean height error is ~0. A fixed-base arm has no
    base velocity, so the term degrades to ``0.0`` (logged once). ``robot``
    selects the robot in a multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        motion = _base_body_velocity(sim, rname)
        if motion is None:
            return 0.0
        vz = motion[0][2]
        return -w * vz * vz

    return term


def _base_ang_vel_xy(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared roll/pitch angular velocity - anti-wobble regularizer.

    ``-weight * (w_body_x ** 2 + w_body_y ** 2)`` from body-frame
    ``base_ang_vel`` (the IMU-gyro reading). Invariant to the yaw rate, so
    turning is free. The legged_gym / IsaacLab ``ang_vel_xy`` term;
    complements ``base_orientation``, which penalises a static tilt but misses
    oscillation whose mean tilt is ~0. A fixed-base arm has no base velocity,
    so the term degrades to ``0.0`` (logged once). ``robot`` selects the robot
    in a multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        motion = _base_body_velocity(sim, rname)
        if motion is None:
            return 0.0
        wx, wy = motion[1][0], motion[1][1]
        return -w * (wx * wx + wy * wy)

    return term


# Stateful reward terms (declarative phase machine). ``staged_reward`` is the
# single generic stateful primitive: it composes EXISTING registry predicates
# into a forward-only phase machine, so tasks stay authored as data and inside
# the closed-registry / no-eval contract.


class StatefulRewardTerm:
    """A reward term that carries per-episode state and must be ``reset()``.

    Duck-typed by consumers: anything with ``__call__(sim) -> float`` AND a
    zero-arg ``reset()`` is treated as episode-stateful. ``SimEnv.reset`` and
    ``DeclarativeBenchmark.on_episode_start`` call ``reset()`` on any reward
    term that has it, so stateless plain-function terms are unaffected.
    """

    def __call__(self, sim: SimEngine) -> float:  # pragma: no cover - interface
        """Score the current sim state, returning this step's reward contribution."""
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        """Clear per-episode state at an episode boundary so accumulated
        progress does not leak across episodes."""
        raise NotImplementedError


class _StagedReward(StatefulRewardTerm):
    """Monotonic multi-stage (phase-machine) reward built from sub-predicates.

    Each stage declares ``reward`` (dense signal while IN the stage),
    ``advance_when`` (bool gate; the first True awards ``bonus`` once and
    advances - phases never regress), and ``bonus`` (one-time scalar, default
    0.0). The last stage has no gate. Per step the emitted reward is
    ``current_stage.reward(sim) + (bonus if this step advanced else 0.0)``.
    """

    def __init__(
        self,
        stages: list[tuple[RewardTerm, BoolPredicate | None, float]],
    ) -> None:
        self._stages = stages
        self._phase = 0

    def reset(self) -> None:
        self._phase = 0

    @property
    def phase(self) -> int:
        """Current stage index (0-based). Exposed for logging / tests."""
        return self._phase

    def __call__(self, sim: SimEngine) -> float:
        if not self._stages:
            return 0.0
        phase = min(self._phase, len(self._stages) - 1)
        reward_fn, advance_fn, bonus = self._stages[phase]
        r = float(reward_fn(sim))
        # Advance (and award the one-time bonus) only if there IS a next stage
        # and this stage declares a gate that now fires.
        if self._phase < len(self._stages) - 1 and advance_fn is not None and bool(advance_fn(sim)):
            self._phase += 1
            return r + float(bonus)
        return r


def reject_non_finite_kwargs(kwargs: dict[str, Any], *, context: str, pred_name: str) -> None:
    """Refuse a ``nan`` / ``inf`` numeric anywhere in a predicate call's kwargs.

    The predicate registry is closed, so a spec cannot name an unknown predicate -
    but kwargs are forwarded to the factory VERBATIM and no factory checks them, so
    a non-finite threshold or weight reaches the reward. JSON has no ``inf``
    literal but ``1e999`` parses to one; YAML spells it ``.inf``.

    Lives here rather than in ``benchmark_spec`` because ``staged_reward``
    NESTS predicate calls and compiles them through :func:`make_predicate`
    directly, bypassing the spec loader's own gate - so validating only there let
    a nested ``{"reward": {"predicate": "constant", "value": 1e999}}`` through to
    ``avg_reward = inf``. ``benchmark_spec`` imports this so both paths share one
    rule (and ``benchmark_spec`` imports from this module, never the reverse).

    Lists are checked element-wise (``bounds`` and friends take sequences of
    floats). ``bool`` is an ``int`` subclass and always finite.
    """
    for key, value in kwargs.items():
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for index, item in enumerate(candidates):
            if isinstance(item, bool) or not isinstance(item, numbers.Real):
                continue
            if math.isfinite(float(item)):
                continue
            where = f"{key}[{index}]" if isinstance(value, (list, tuple)) else key
            raise ValueError(
                f"{context}: predicate {pred_name!r} kwarg '{where}' is not finite ({item}). "
                "A non-finite threshold or weight propagates into the reward - an eval run "
                "reports 'Avg reward: inf' while every call still returns success."
            )


def _staged_reward(stages: list[Any]) -> RewardTerm:
    """Factory: compile a declared stage list into a :class:`_StagedReward`.

    Each stage's ``reward`` and ``advance_when`` are compiled through
    :func:`make_predicate`, so specs stay inside the closed-registry /
    no-``eval`` contract. Stage shape::

        {
            "reward": {"predicate": <float-term name>, **kwargs},
            "advance_when": {"predicate": <bool-pred name>, **kwargs},  # omit on last stage
            "bonus": <float>,   # optional, default 0.0
        }

    Raises:
        ValueError: stages is not a non-empty list, a stage is malformed, a
            non-final stage omits ``advance_when``, or ``bonus`` is non-numeric.
        TypeError: surfaced from :func:`make_predicate` for bad sub-kwargs.
    """
    if not isinstance(stages, list) or not stages:
        raise ValueError("staged_reward: 'stages' must be a non-empty list of stage dicts")

    compiled: list[tuple[RewardTerm, BoolPredicate | None, float]] = []
    n = len(stages)
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"staged_reward: stage[{i}] must be a dict, got {type(stage).__name__}")
        unknown = set(stage.keys()) - {"reward", "advance_when", "bonus"}
        if unknown:
            raise ValueError(
                f"staged_reward: stage[{i}] has unknown keys {sorted(unknown)}; allowed: reward, advance_when, bonus"
            )

        reward_call = stage.get("reward")
        if not isinstance(reward_call, dict) or "predicate" not in reward_call:
            raise ValueError(
                f"staged_reward: stage[{i}].reward must be a predicate-call dict "
                "like {predicate: distance_neg, body_a: ..., body_b: ...}"
            )
        reward_name = reward_call["predicate"]
        reward_kwargs = {k: v for k, v in reward_call.items() if k != "predicate"}
        reject_non_finite_kwargs(reward_kwargs, context=f"staged_reward: stage[{i}].reward", pred_name=reward_name)
        reward_fn = make_predicate(reward_name, **reward_kwargs)

        advance_call = stage.get("advance_when")
        advance_fn: BoolPredicate | None
        if advance_call is None:
            if i != n - 1:
                raise ValueError(
                    f"staged_reward: stage[{i}] is not the final stage and must declare "
                    "'advance_when' (a bool predicate gating the transition to the next stage)"
                )
            advance_fn = None
        else:
            if not isinstance(advance_call, dict) or "predicate" not in advance_call:
                raise ValueError(
                    f"staged_reward: stage[{i}].advance_when must be a predicate-call dict "
                    "like {predicate: distance_less_than, body_a: ..., body_b: ..., threshold: ...}"
                )
            advance_name = advance_call["predicate"]
            if isinstance(advance_name, str) and predicate_kind(advance_name) == "float":
                raise ValueError(
                    f"staged_reward: stage[{i}].advance_when predicate {advance_name!r} is a "
                    "reward term (float-valued); advance_when gates the stage transition and "
                    "must be a bool predicate. Reward terms belong in the stage's 'reward' field."
                )
            advance_kwargs = {k: v for k, v in advance_call.items() if k != "predicate"}
            reject_non_finite_kwargs(
                advance_kwargs, context=f"staged_reward: stage[{i}].advance_when", pred_name=str(advance_name)
            )
            advance_fn = make_predicate(advance_name, **advance_kwargs)

        bonus_raw = stage.get("bonus", 0.0)
        if isinstance(bonus_raw, bool) or not isinstance(bonus_raw, (int, float)):
            raise ValueError(f"staged_reward: stage[{i}].bonus must be a number, got {bonus_raw!r}")
        # "is a number" is not enough: the bonus is added to the reward the step a
        # stage advances, so a non-finite one makes that step's reward non-finite
        # (measured: staged term returned [inf, 0.0, 0.0] with bonus=1e999).
        if not math.isfinite(float(bonus_raw)):
            raise ValueError(f"staged_reward: stage[{i}].bonus must be finite, got {bonus_raw!r}")

        compiled.append((reward_fn, advance_fn, float(bonus_raw)))

    return _StagedReward(compiled)


# Registry

PREDICATE_REGISTRY: dict[str, PredicateFactory] = {
    # bool-valued
    "body_above_z": _body_above_z,
    "body_below_z": _body_below_z,
    "joint_above": _joint_above,
    "joint_below": _joint_below,
    "distance_less_than": _distance_less_than,
    "inside_region": _inside_region,
    "contact_between": _contact_between,
    "contact_any": _contact_any,
    "body_on": _body_on,
    "body_inside": _body_inside,
    "body_upright": _body_upright,
    "grasped": _grasped,
    "base_tipped": _base_tipped,
    "base_below_z": _base_below_z,
    "base_beyond_x": _base_beyond_x,
    "base_beyond_y": _base_beyond_y,
    "base_yaw_beyond": _base_yaw_beyond,
    # float-valued
    "distance_neg": _distance_neg,
    "joint_progress": _joint_progress,
    "base_velocity": _base_velocity,
    "base_velocity_tracking": _base_velocity_tracking,
    "base_height": _base_height,
    "base_orientation": _base_orientation,
    "base_lin_vel_z": _base_lin_vel_z,
    "base_ang_vel_xy": _base_ang_vel_xy,
    "constant": _constant,
    # stateful (phase machine)
    "staged_reward": _staged_reward,
}


def register_predicate(name: str, factory: PredicateFactory) -> None:
    """Register a user-defined predicate factory.

    Must be called before loading a spec that references ``name``. Runtime
    factories are NOT sandboxed - registering opts into running the factory
    with kwargs parsed from the spec, so only register from trusted code
    paths; anything LLM-authored should use the built-in DSL exclusively.

    Raises:
        ValueError: If ``name`` shadows a built-in predicate.
        TypeError: If ``factory`` is not callable.
    """
    if name in PREDICATE_REGISTRY:
        raise ValueError(f"register_predicate: '{name}' shadows a built-in predicate; pick a different name")
    if not callable(factory):
        raise TypeError(f"register_predicate: factory must be callable, got {type(factory).__name__}")
    PREDICATE_REGISTRY[name] = factory


def make_predicate(name: str, **kwargs: Any) -> Callable[[SimEngine], Any]:
    """Instantiate a predicate from its name + kwargs.

    The single entry point the DSL loader uses - it never touches ``eval`` or
    ``exec``. ``kwargs`` are forwarded verbatim to the factory; the result is
    a callable ``(sim) -> bool`` or ``(sim) -> float``.

    Raises:
        ValueError: If ``name`` is unknown (the message lists the valid set).
        TypeError: If required factory kwargs are missing.
    """
    factory = PREDICATE_REGISTRY.get(name)
    if factory is None:
        valid = sorted(PREDICATE_REGISTRY.keys())
        raise ValueError(f"Unknown predicate '{name}'. Valid: {valid}")
    return factory(**kwargs)


def predicate_kind(name: str) -> str:
    """Classify a registered predicate as ``"bool"``, ``"float"``, or ``"unknown"``.

    Success/failure clauses require ``"bool"``; ``dense_reward`` terms are
    ``"float"``. The kind is read from the factory's ``-> BoolPredicate`` /
    ``-> RewardTerm`` return annotation, so it cannot drift from the registry.
    A factory without a recognizable annotation classifies as ``"unknown"``
    and is exempt from kind validation.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    factory = PREDICATE_REGISTRY.get(name)
    if factory is None:
        valid = sorted(PREDICATE_REGISTRY.keys())
        raise ValueError(f"Unknown predicate '{name}'. Valid: {valid}")
    annotation = str(getattr(factory, "__annotations__", {}).get("return", ""))
    if "Bool" in annotation:
        return "bool"
    if "Reward" in annotation:
        return "float"
    return "unknown"


def supports_body_lookup(sim: SimEngine) -> bool:
    """Whether *sim*'s backend can resolve body names at all.

    Body-referencing predicates resolve through ``get_body_state`` (MuJoCo
    only at time of writing). On a backend without that method every
    body-referencing predicate degrades to a constant ``False``, so callers
    that arm such a predicate (e.g. ``run_policy(stop_when=...)``) should
    check this up front and reject rather than silently never fire.

    Args:
        sim: The engine to probe.

    Returns:
        ``True`` when the backend implements ``get_body_state``.
    """
    return getattr(sim, "get_body_state", None) is not None


def can_resolve_body(sim: SimEngine, body: str) -> bool:
    """Whether *body* resolves in *sim* right now, via the predicate DSL's own lookup.

    Uses the exact resolution path the body-referencing predicates use at
    evaluation time (:func:`_body_position`), including the LIBERO
    ``<name>_main`` fallback - so a ``True`` here means the predicate will
    genuinely be evaluable against the live scene, and a ``False`` means it
    would degrade to a constant ``False`` forever (a typo'd name, or a
    backend without body lookups).

    Args:
        sim: The engine to probe.
        body: The body name a predicate clause references.

    Returns:
        ``True`` when the lookup resolves to a position.
    """
    return _body_position(sim, body) is not None


def can_resolve_joint(sim: SimEngine, joint: str) -> bool:
    """Whether *joint* resolves in *sim* right now, via the predicate DSL's own lookup.

    Mirrors :func:`can_resolve_body` for joint-referencing predicates
    (``joint_above`` / ``joint_below``), using the same ``get_observation``
    path the predicates use at evaluation time.

    Args:
        sim: The engine to probe.
        joint: The joint name a predicate clause references.

    Returns:
        ``True`` when the lookup resolves to a value.
    """
    return _joint_position(sim, joint) is not None


__all__ = [
    "PREDICATE_REGISTRY",
    "BoolPredicate",
    "PredicateFactory",
    "RewardTerm",
    "StatefulRewardTerm",
    "can_resolve_body",
    "can_resolve_joint",
    "make_predicate",
    "predicate_kind",
    "register_predicate",
    "supports_body_lookup",
]
