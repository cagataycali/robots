"""The arithmetic behind ``Simulation.set_pose``: where a body is now, where it is asked to go, and which record
holds that (the compiled model, or the scene entry when the body is not built yet or is rebuilt from the scene).
Called under the world lock."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import moved_frame, vec
from ..types import Pose


def _requested(cur: Pose, position: Any, orientation: Any) -> Pose:
    return Pose(
        cur.position if position is None else np.asarray(vec(position, 3, "position")),
        cur.orientation if orientation is None else np.asarray(vec(orientation, 4, "orientation (w, x, y, z)")),
    )


def _entry(ops: Any, name: str) -> Any:
    """The scene record of a robot (by id) or an object (by name)."""
    if name in ops._robots:
        return ops.scene.robot(name)
    return next(o for o in ops.scene.objects if o.name == name)


def teleport(ops: Any, name: str, position: Any, orientation: Any) -> Pose:
    """Move ``name`` to the requested pose, velocities zeroed; returns where it is."""
    try:
        cur = ops.backend.body_pose(name)
    except KeyError:
        if not ops._editing:
            raise
        entry = _entry(ops, name)  # added in this block, not built yet: the spawn frame is where it is
        pose = _requested(
            Pose(np.asarray(entry.position, float), np.asarray(entry.orientation, float)), position, orientation
        )
        entry.position, entry.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())
        ops._reload()  # the block's one rebuild places it from the scene
        return pose
    pose = _requested(cur, position, orientation)
    fixed = name in ops._robots and not ops._robots[name].description.floating
    if fixed:
        sr = ops.scene.robot(name)  # a fixed base is rebuilt from the scene, so the scene moves first
        f = moved_frame(Pose(np.asarray(sr.position), np.asarray(sr.orientation)), cur, pose)
        sr.position, sr.orientation = tuple(f.position.tolist()), tuple(f.orientation.tolist())
    for o in ops.scene.objects:
        if o.name == name and not o.free:
            o.position, o.orientation = tuple(pose.position.tolist()), tuple(pose.orientation.tolist())
            fixed = True
    if fixed and ops._editing:  # the scene holds the pose: the block's one rebuild places it, not a second
        ops._reload()
        return pose
    ops.backend.set_pose(name, pose, zero_velocity=True)
    ops.revision += 1
    return ops.backend.body_pose(name)
