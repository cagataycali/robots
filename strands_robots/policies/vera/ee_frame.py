"""End-effector frame auto-discovery for VERA eef/cartesian-delta IK.

VERA's ``eef_delta`` (mimicgen) and ``cartesian_delta`` (droid) embodiments emit
end-effector pose *deltas*; driving a MuJoCo arm needs an IK target frame (the
body/site the Cartesian task tracks). The robot registry does **not** record an
ee-frame, so we discover it from the compiled ``mujoco.MjModel`` with a robust,
namespace-aware heuristic - making eef-delta embodiments **zero-config**.

Heuristic (first match wins), scoped to the robot's ``namespace`` (``<robot>/``):
  1. A **site** whose name hints at the tool point:
     ``attachment_site`` | ``grasp`` | ``ee`` | ``tcp`` | ``pinch`` | ``flange``
     - these are the conventional MuJoCo IK targets (e.g. menagerie Panda ships
     ``attachment_site``). Sites are preferred: they are the intended TCP.
  2. A **body** whose name hints at the hand/tool:
     ``hand`` | ``gripper`` | ``tool`` | ``ee`` | ``tcp`` | ``wrist`` | ``flange``.
  3. The **leaf body** of the robot's kinematic chain (the descendant of the
     robot's joints that has no child body) - the last link, where a tool mounts.

Returns ``(frame_name, frame_type)`` ready for ``mink.FrameTask`` /
``MinkIKBridge`` (frame_type ∈ {``"site"``, ``"body"``}); names keep the robot
namespace so they resolve in the shared world model. Returns ``None`` when the
arm cannot be resolved (caller then warns + asks for an explicit frame).

``mujoco`` is imported lazily so importing this module in the light base env is
free.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SITE_HINTS = ("attachment_site", "attachment", "grasp", "pinch", "tcp", "ee_site", "ee", "flange")
_BODY_HINTS = ("hand", "gripper", "tool", "tcp", "ee", "wrist", "flange", "end_effector", "eef")


def _names_of(model: Any, obj_type: int) -> list[tuple[int, str]]:
    """Return ``[(id, name), ...]`` for all objects of ``obj_type`` in the model."""
    import mujoco as mj

    out: list[tuple[int, str]] = []
    n = {
        mj.mjtObj.mjOBJ_SITE: model.nsite,
        mj.mjtObj.mjOBJ_BODY: model.nbody,
    }[obj_type]
    for i in range(n):
        nm = mj.mj_id2name(model, obj_type, i)
        if nm:
            out.append((i, nm))
    return out


def _scoped(name: str, namespace: str | None) -> bool:
    """True when ``name`` belongs to the robot's namespace (or no namespace set)."""
    if not namespace:
        return True
    return name.startswith(namespace)


def _basename(name: str, namespace: str | None) -> str:
    """Strip the robot namespace prefix for hint matching."""
    if namespace and name.startswith(namespace):
        return name[len(namespace) :]
    return name


def discover_ee_frame(model: Any, namespace: str | None = None) -> tuple[str, str] | None:
    """Discover an IK end-effector frame ``(name, type)`` for a robot.

    Args:
        model: The compiled ``mujoco.MjModel`` (the shared world model).
        namespace: The robot's body/site namespace prefix (e.g. ``"panda/"``).
            Discovery is scoped to this so multi-robot worlds resolve correctly.

    Returns:
        ``(frame_name, frame_type)`` where ``frame_type`` ∈ {``"site"``,
        ``"body"``}, names keep the namespace; or ``None`` if nothing resolves.
    """
    try:
        import mujoco  # noqa: F401  (lazy availability check)
    except ImportError:
        logger.debug("mujoco not importable; cannot auto-discover ee-frame")
        return None

    # 1) Prefer a TCP-like SITE.
    sites = [(i, n) for i, n in _names_of(model, _site_obj()) if _scoped(n, namespace)]
    for hint in _SITE_HINTS:
        for _i, name in sites:
            if hint in _basename(name, namespace).lower():
                logger.info("VERA ee-frame: site %r (hint %r)", name, hint)
                return name, "site"

    # 2) A hand/tool BODY.
    bodies = [(i, n) for i, n in _names_of(model, _body_obj()) if _scoped(n, namespace)]
    for hint in _BODY_HINTS:
        for _i, name in bodies:
            if hint in _basename(name, namespace).lower():
                logger.info("VERA ee-frame: body %r (hint %r)", name, hint)
                return name, "body"

    # 3) Leaf body of the namespace's kinematic chain.
    leaf = _leaf_body(model, namespace, bodies)
    if leaf is not None:
        logger.info("VERA ee-frame: leaf body %r (kinematic chain tail)", leaf)
        return leaf, "body"

    logger.warning(
        "VERA ee-frame: could not auto-discover an end-effector frame for "
        "namespace %r; pass ee_frame_name explicitly to set_ik_target(...).",
        namespace,
    )
    return None


def _site_obj() -> int:
    import mujoco as mj

    return mj.mjtObj.mjOBJ_SITE


def _body_obj() -> int:
    import mujoco as mj

    return mj.mjtObj.mjOBJ_BODY


def _leaf_body(model: Any, namespace: str | None, bodies: list[tuple[int, str]]) -> str | None:
    """The deepest body in the namespace's chain (a body with no in-namespace child).

    MuJoCo stores ``body_parentid``; the leaf (no children within the namespace)
    that sits furthest from the world is the tool-mount link. Among multiple
    leaves we pick the one with the greatest depth from the world body.
    """
    if not bodies:
        return None
    ids = {i for i, _ in bodies}
    id_to_name = {i: n for i, n in bodies}
    # Children count within the namespace.
    has_child = set()
    for i in ids:
        parent = int(model.body_parentid[i])
        if parent in ids:
            has_child.add(parent)
    leaves = [i for i in ids if i not in has_child]
    if not leaves:
        return None

    # Depth from world for tie-break (more joints between world and body = tip).
    def depth(bi: int) -> int:
        d, cur = 0, bi
        seen = set()
        while cur not in seen:
            seen.add(cur)
            p = int(model.body_parentid[cur])
            if p == cur or p == 0:
                break
            cur = p
            d += 1
        return d

    leaves.sort(key=depth, reverse=True)
    return id_to_name[leaves[0]]
