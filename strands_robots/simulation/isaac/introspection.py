"""Scene introspection the shared ``describe()`` contract advertises.

``SimEngine.describe()`` - the call agents are told to make FIRST "instead of
guessing method names" - lists ``get_robot_state`` and ``get_features`` among
"the methods most commonly needed", and the Isaac backend inherits
``describe()`` unchanged. Isaac implemented neither, so an agent that followed
the advertisement got ``AttributeError: 'IsaacSimulation' object has no
attribute 'get_robot_state'`` (measured on a live GPU run). MuJoCo and
Newton both carry them, plus ``list_objects`` / ``list_cameras``, which this
mixin adds with the same envelope shapes.

Everything here is a read of state the backend already owns (the articulation
observation, the object/camera registries, ``get_body_state``); no new Kit
surface is touched, so the methods behave identically on 6.0.x and 6.1.x.
"""

from __future__ import annotations

from typing import Any

from strands_robots.simulation.models import registered

_NO_WORLD = "No world. Call create_world first."


class IsaacIntrospectionMixin:
    """``get_robot_state`` / ``get_features`` / ``list_objects`` / ``list_cameras``."""

    # Provided by IsaacSimulation.
    _lock: Any
    _robots: dict[str, Any]
    _objects: dict[str, Any]
    _cameras: dict[str, Any]
    _world_created: bool
    _physics_view_stale: bool
    _sim_time: float

    def _single_robot(self, robot_name: str | None) -> tuple[str | None, dict[str, Any] | None]:
        if not self._world_created:
            return None, {"status": "error", "content": [{"text": _NO_WORLD}]}
        if robot_name is None:
            if len(self._robots) != 1:
                return None, {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"robot_name is required: {len(self._robots)} robots are loaded "
                                f"({sorted(self._robots)})."
                            )
                        }
                    ],
                }
            robot_name = next(iter(self._robots))
        if not registered(self._robots, robot_name):
            return None, {
                "status": "error",
                "content": [{"text": f"Robot '{robot_name}' not found. Loaded: {sorted(self._robots)}."}],
            }
        return robot_name, None

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        """Per-joint position and velocity, in the MuJoCo/Newton envelope.

        The ``json`` block is ``{"state": {joint: {"position", "velocity"}}}``
        (radians and radians/second for revolute joints), read from the same
        articulation observation ``get_observation`` returns. A scene whose
        tensor view is stale (a dynamic body was added or removed since the
        last ``reset()``) cannot be read, and says so rather than answering an
        empty state.
        """
        name, err = self._single_robot(robot_name)
        if err is not None or name is None:
            return err or {"status": "error", "content": [{"text": "get_robot_state: no robot resolved."}]}
        if getattr(self, "_physics_view_stale", False):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "get_robot_state: a dynamic body was added or removed since the last reset(), "
                            "so PhysX's tensor view no longer covers the scene. Call reset() first."
                        )
                    }
                ],
            }
        obs = self.get_observation(name, skip_images=True)  # type: ignore[attr-defined]
        joints = list(self._robots[name].joint_names)
        missing = [j for j in joints if j not in obs]
        if missing:
            return {
                "status": "error",
                "content": [
                    {"text": f"get_robot_state: '{name}' has no readable articulation state for joints {missing}."}
                ],
            }
        state = {j: {"position": float(obs[j]), "velocity": float(obs.get(f"{j}.vel", 0.0))} for j in joints}
        text = f"'{name}' state (t={self._sim_time:.3f}s):\n" + "".join(
            f"{j}: pos={v['position']:.4f}, vel={v['velocity']:.4f}\n" for j, v in state.items()
        )
        return {"status": "success", "content": [{"text": text}, {"json": {"state": state}}]}

    def get_features(self, robot_name: str | None = None) -> dict[str, Any]:
        """Joints / cameras / robots summary, in the MuJoCo/Newton ``features`` schema.

        Isaac drives joints through PhysX joint drives rather than named
        actuators, so ``actuator_names`` echoes the joint names (Newton's
        convention).
        """
        if not self._world_created:
            return {"status": "error", "content": [{"text": _NO_WORLD}]}
        with self._lock:
            if robot_name is not None:
                if not registered(self._robots, robot_name):
                    return {
                        "status": "error",
                        "content": [{"text": f"Robot '{robot_name}' not found. Loaded: {sorted(self._robots)}."}],
                    }
                scoped = {robot_name: self._robots[robot_name]}
            else:
                scoped = dict(self._robots)
            joint_names = [j for robot in scoped.values() for j in robot.joint_names]
            robots_info = {
                rname: {
                    "joint_names": list(robot.joint_names),
                    "n_joints": len(robot.joint_names),
                    "data_config": getattr(robot, "data_config", None),
                    "source": getattr(robot, "description_path", None),
                }
                for rname, robot in scoped.items()
            }
            cameras = list(self._cameras)
        timestep = self.physics_timestep()  # type: ignore[attr-defined]
        features = {
            "n_joints": len(joint_names),
            "n_dofs": len(joint_names),
            "timestep": timestep,
            "solver": "physx",
            "joint_names": joint_names,
            "actuator_names": list(joint_names),
            "camera_names": cameras,
            "robots": robots_info,
        }
        lines = [
            "Simulation Features (Isaac Sim / PhysX)",
            f"Joints ({len(joint_names)}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"Cameras: {', '.join(cameras) or 'none'}",
        ]
        if timestep:
            lines.append(f"Timestep: {timestep}s ({1 / timestep:.0f}Hz)")
        lines += [f"{rname}: {info['n_joints']} joints" for rname, info in robots_info.items()]
        return {"status": "success", "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}]}

    def list_cameras(self) -> list[str]:
        """Names of the cameras added with ``add_camera``, in insertion order."""
        with self._lock:
            return list(self._cameras)

    def list_objects(self) -> dict[str, Any]:
        """Objects with their shape and LIVE pose (read back, not the spawn request)."""
        if not self._world_created:
            return {"status": "error", "content": [{"text": _NO_WORLD}]}
        with self._lock:
            objects = dict(self._objects)
        if not objects:
            return {"status": "success", "content": [{"text": "No objects."}, {"json": {"objects": {}}}]}
        listing: dict[str, Any] = {}
        lines = ["Objects:"]
        for name, obj in objects.items():
            pose = self.get_body_state(body_name=name)  # type: ignore[attr-defined]
            payload = next((b["json"] for b in pose.get("content", []) if "json" in b), None)
            position = payload.get("position") if pose.get("status") == "success" and payload else None
            listing[name] = {
                "shape": getattr(obj, "shape", None),
                "is_static": bool(getattr(obj, "is_static", False)),
                "prim_path": getattr(obj, "prim_path", None),
                "position": position,
            }
            where = f"[{', '.join(f'{v:.3f}' for v in position)}]" if position else "pose unavailable"
            lines.append(
                f"  {name}: {listing[name]['shape']} at {where}{' (static)' if listing[name]['is_static'] else ''}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}, {"json": {"objects": listing}}]}
