"""Physics mixin — advanced MuJoCo physics introspection and manipulation.

Exposes the deep MuJoCo C API through clean Python methods:
- Raycasting (mj_ray)
- Jacobians (mj_jacBody, mj_jacSite, mj_jacGeom)
- Energy computation (mj_energyPos, mj_energyVel)
- External forces (mj_applyFT, xfrc_applied)
- Mass matrix (mj_fullM)
- State checkpointing (mj_getState, mj_setState)
- Inverse dynamics (mj_inverse)
- Body/joint introspection (poses, velocities, accelerations)
- Direct joint position/velocity control (qpos, qvel)
- Runtime model modification (mass, friction, color, size)
- Sensor readout (sensordata)
- Contact force analysis (mj_contactForce)
"""

import json
import logging
from typing import Any

import numpy as np

from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class PhysicsMixin:
    """Advanced physics capabilities for Simulation.

    Expects: self._world (SimWorld with _model, _data)

    Naming: methods match action names in tool_spec.json for direct dispatch.
    """

    # ── State Checkpointing ──

    def save_state(self, name: str = "default") -> dict[str, Any]:
        """Save the full physics state (qpos, qvel, act, time) to a named checkpoint.

        Uses mj_getState with mjSTATE_PHYSICS for complete state capture.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        state_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_PHYSICS)
        state = np.zeros(state_size)
        mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_PHYSICS)

        if not hasattr(self._world, "_checkpoints"):
            self._world._checkpoints = {}

        self._world._checkpoints[name] = {
            "state": state.copy(),
            "sim_time": self._world.sim_time,
            "step_count": self._world.step_count,
        }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"💾 State '{name}' saved\n"
                        f"  t={self._world.sim_time:.4f}s, step={self._world.step_count}\n"
                        f"  State vector: {state_size} floats\n"
                        f"  Checkpoints: {list(self._world._checkpoints.keys())}"
                    )
                }
            ],
        }

    def load_state(self, name: str = "default") -> dict[str, Any]:
        """Restore physics state from a named checkpoint."""
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        checkpoints = getattr(self._world, "_checkpoints", {})
        if name not in checkpoints:
            available = list(checkpoints.keys()) if checkpoints else ["none"]
            return {
                "status": "error",
                "content": [{"text": f"❌ Checkpoint '{name}' not found. Available: {available}"}],
            }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        checkpoint = checkpoints[name]

        mj.mj_setState(model, data, checkpoint["state"], mj.mjtState.mjSTATE_PHYSICS)
        mj.mj_forward(model, data)

        self._world.sim_time = checkpoint["sim_time"]
        self._world.step_count = checkpoint["step_count"]

        return {
            "status": "success",
            "content": [
                {"text": f"📂 State '{name}' restored (t={self._world.sim_time:.4f}s, step={self._world.step_count})"}
            ],
        }

    # ── External Forces ──

    def apply_force(
        self,
        body_name: str,
        force: list[float] = None,
        torque: list[float] = None,
        point: list[float] = None,
    ) -> dict[str, Any]:
        """Apply external force and/or torque to a body.

        Uses mj_applyFT for precise force application at a world-frame point.
        Forces persist for one timestep — call before each step for continuous force.

        Args:
            body_name: Target body name.
            force: [fx, fy, fz] in world frame (Newtons).
            torque: [tx, ty, tz] in world frame (N·m).
            point: [px, py, pz] world-frame point of force application.
                   Defaults to body CoM if not specified.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"❌ Body '{body_name}' not found."}]}

        f = np.array(force or [0, 0, 0], dtype=np.float64)
        t = np.array(torque or [0, 0, 0], dtype=np.float64)
        p = np.array(point, dtype=np.float64) if point else data.xipos[body_id].copy()

        mj.mj_applyFT(model, data, f, t, p, body_id, data.qfrc_applied)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"💨 Force applied to '{body_name}' (body {body_id})\n"
                        f"  Force: {f.tolist()} N\n"
                        f"  Torque: {t.tolist()} N·m\n"
                        f"  Point: {p.tolist()}"
                    )
                }
            ],
        }

    # ── Raycasting ──

    def raycast(
        self,
        origin: list[float],
        direction: list[float],
        exclude_body: int = -1,
        include_static: bool = True,
    ) -> dict[str, Any]:
        """Cast a ray and find the first geom intersection.

        Uses mj_ray for precise distance sensing / obstacle detection.

        Args:
            origin: [x, y, z] ray start point in world frame.
            direction: [dx, dy, dz] ray direction (auto-normalized).
            exclude_body: Body ID to exclude from intersection (-1 = none).
            include_static: Whether to include static geoms.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        pnt = np.array(origin, dtype=np.float64)
        vec = np.array(direction, dtype=np.float64)
        # Normalize direction
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        geomid = np.array([-1], dtype=np.int32)
        dist = mj.mj_ray(
            model,
            data,
            pnt,
            vec,
            None,  # geom group filter (None = all)
            1 if include_static else 0,
            exclude_body,
            geomid,
        )

        hit = dist >= 0
        geom_name = None
        if hit and geomid[0] >= 0:
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomid[0])

        result = {
            "hit": hit,
            "distance": float(dist) if hit else None,
            "geom_id": int(geomid[0]) if hit else None,
            "geom_name": geom_name,
            "hit_point": (pnt + vec * dist).tolist() if hit else None,
        }

        if hit:
            text = f"🎯 Ray hit '{geom_name or geomid[0]}' at dist={dist:.4f}m, point={result['hit_point']}"
        else:
            text = "🎯 Ray: no intersection"

        return {"status": "success", "content": [{"text": text}, {"text": json.dumps(result, default=str)}]}

    # ── Jacobians ──

    def get_jacobian(
        self,
        body_name: str = None,
        site_name: str = None,
        geom_name: str = None,
    ) -> dict[str, Any]:
        """Compute the Jacobian (position + rotation) for a body, site, or geom.

        The Jacobian maps joint velocities to Cartesian velocities:
            v = J @ dq

        Returns both positional (3×nv) and rotational (3×nv) Jacobians.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        if body_name:
            obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            if obj_id < 0:
                return {"status": "error", "content": [{"text": f"❌ Body '{body_name}' not found."}]}
            mj.mj_jacBody(model, data, jacp, jacr, obj_id)
            label = f"body '{body_name}'"
        elif site_name:
            obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
            if obj_id < 0:
                return {"status": "error", "content": [{"text": f"❌ Site '{site_name}' not found."}]}
            mj.mj_jacSite(model, data, jacp, jacr, obj_id)
            label = f"site '{site_name}'"
        elif geom_name:
            obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
            if obj_id < 0:
                return {"status": "error", "content": [{"text": f"❌ Geom '{geom_name}' not found."}]}
            mj.mj_jacGeom(model, data, jacp, jacr, obj_id)
            label = f"geom '{geom_name}'"
        else:
            return {"status": "error", "content": [{"text": "❌ Specify body_name, site_name, or geom_name."}]}

        return {
            "status": "success",
            "content": [
                {"text": f"🧮 Jacobian for {label}: pos={jacp.shape}, rot={jacr.shape}, nv={model.nv}"},
                {
                    "text": json.dumps(
                        {
                            "jacp": jacp.tolist(),
                            "jacr": jacr.tolist(),
                            "nv": model.nv,
                        },
                        default=str,
                    )
                },
            ],
        }

    # ── Energy ──

    def get_energy(self) -> dict[str, Any]:
        """Compute potential and kinetic energy of the system."""
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        mj.mj_energyPos(model, data)
        mj.mj_energyVel(model, data)

        potential = float(data.energy[0])
        kinetic = float(data.energy[1])
        total = potential + kinetic

        return {
            "status": "success",
            "content": [
                {"text": f"⚡ Energy: potential={potential:.4f}J, kinetic={kinetic:.4f}J, total={total:.4f}J"},
                {"text": json.dumps({"potential": potential, "kinetic": kinetic, "total": total}, default=str)},
            ],
        }

    # ── Mass Matrix ──

    def get_mass_matrix(self) -> dict[str, Any]:
        """Compute the full mass (inertia) matrix M(q).

        M is nv×nv where nv is the number of DoFs.
        Useful for dynamics analysis, impedance control, etc.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        nv = model.nv
        M = np.zeros((nv, nv))
        mj.mj_fullM(model, M, data.qM)
        rank = int(np.linalg.matrix_rank(M))
        cond = float(np.linalg.cond(M)) if rank > 0 else float("inf")

        return {
            "status": "success",
            "content": [
                {"text": f"🧮 Mass matrix: {nv}×{nv}, rank={rank}, cond={cond:.2e}"},
                {
                    "text": json.dumps(
                        {
                            "shape": [nv, nv],
                            "rank": rank,
                            "condition_number": cond,
                            "diagonal": np.diag(M).tolist(),
                            "total_mass": float(np.sum(model.body_mass)),
                        },
                        default=str,
                    )
                },
            ],
        }

    # ── Inverse Dynamics ──

    def inverse_dynamics(self) -> dict[str, Any]:
        """Compute inverse dynamics: given qacc, what forces are needed?

        Runs mj_inverse to compute qfrc_inverse — the generalized forces
        that would produce the current accelerations.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        mj.mj_inverse(model, data)

        # Build named force mapping
        forces = {}
        for i in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if name:
                dof_adr = model.jnt_dofadr[i]
                forces[name] = float(data.qfrc_inverse[dof_adr])

        return {
            "status": "success",
            "content": [
                {"text": f"🔄 Inverse dynamics: {len(forces)} joint forces computed"},
                {"text": json.dumps({"qfrc_inverse": forces}, default=str)},
            ],
        }

    # ── Body Introspection ──

    def get_body_state(
        self,
        body_name: str,
    ) -> dict[str, Any]:
        """Get the full state of a body: position, orientation, velocity, acceleration.

        Returns Cartesian pose + 6D spatial velocity (linear + angular).
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"❌ Body '{body_name}' not found."}]}

        # Position and orientation
        pos = data.xpos[body_id].tolist()
        quat = data.xquat[body_id].tolist()
        rotmat = data.xmat[body_id].reshape(3, 3).tolist()

        # Velocity (6D: angular then linear in world frame)
        vel = np.zeros(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        linvel = vel[3:].tolist()
        angvel = vel[:3].tolist()

        # Mass and inertia
        mass = float(model.body_mass[body_id])
        com = data.xipos[body_id].tolist()

        state = {
            "position": pos,
            "quaternion": quat,
            "rotation_matrix": rotmat,
            "linear_velocity": linvel,
            "angular_velocity": angvel,
            "mass": mass,
            "center_of_mass": com,
        }

        text = (
            f"🏷️ Body '{body_name}' (id={body_id}):\n"
            f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n"
            f"  linvel: [{linvel[0]:.4f}, {linvel[1]:.4f}, {linvel[2]:.4f}]\n"
            f"  angvel: [{angvel[0]:.4f}, {angvel[1]:.4f}, {angvel[2]:.4f}]\n"
            f"  mass: {mass:.4f}kg, com: {com}"
        )

        return {"status": "success", "content": [{"text": text}, {"text": json.dumps(state, default=str)}]}

    # ── Direct Joint Control ──

    def set_joint_positions(
        self,
        positions: dict[str, float] = None,
        robot_name: str = None,
    ) -> dict[str, Any]:
        """Set joint positions directly (bypassing actuators).

        Writes to qpos and runs mj_forward to update kinematics.
        Useful for teleportation, IK solutions, or keyframe setting.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if positions is None:
            return {"status": "error", "content": [{"text": "❌ positions dict required."}]}

        set_count = 0
        for jnt_name, value in positions.items():
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr] = float(value)
                set_count += 1
            else:
                logger.warning("Joint '%s' not found, skipping", jnt_name)

        mj.mj_forward(model, data)

        return {
            "status": "success",
            "content": [{"text": f"🎯 Set {set_count}/{len(positions)} joint positions, FK updated"}],
        }

    def set_joint_velocities(
        self,
        velocities: dict[str, float] = None,
    ) -> dict[str, Any]:
        """Set joint velocities directly.

        Writes to qvel. Useful for initializing dynamics.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if velocities is None:
            return {"status": "error", "content": [{"text": "❌ velocities dict required."}]}

        set_count = 0
        for jnt_name, value in velocities.items():
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                dof_adr = model.jnt_dofadr[jnt_id]
                data.qvel[dof_adr] = float(value)
                set_count += 1

        return {
            "status": "success",
            "content": [{"text": f"💨 Set {set_count}/{len(velocities)} joint velocities"}],
        }

    # ── Sensor Readout ──

    def get_sensor_data(self, sensor_name: str = None) -> dict[str, Any]:
        """Read sensor values from the simulation.

        MuJoCo supports: jointpos, jointvel, accelerometer, gyro, force,
        torque, touch, rangefinder, framequat, subtreecom, clock, etc.

        Args:
            sensor_name: Specific sensor name, or None for all sensors.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if model.nsensor == 0:
            return {"status": "success", "content": [{"text": "📡 No sensors in model."}]}

        mj.mj_forward(model, data)

        sensors = {}
        for i in range(model.nsensor):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            if not name:
                name = f"sensor_{i}"

            adr = model.sensor_adr[i]
            dim = model.sensor_dim[i]
            values = data.sensordata[adr : adr + dim].tolist()

            if sensor_name and name != sensor_name:
                continue

            sensors[name] = {
                "values": values if dim > 1 else values[0],
                "dim": int(dim),
                "type": int(model.sensor_type[i]),
            }

        if sensor_name and sensor_name not in sensors:
            return {"status": "error", "content": [{"text": f"❌ Sensor '{sensor_name}' not found."}]}

        lines = [f"📡 Sensors ({len(sensors)}/{model.nsensor}):"]
        for name, info in sensors.items():
            lines.append(f"  {name}: {info['values']} (dim={info['dim']})")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"text": json.dumps({"sensors": sensors}, default=str)}],
        }

    # ── Runtime Model Modification ──

    def set_body_properties(
        self,
        body_name: str,
        mass: float = None,
    ) -> dict[str, Any]:
        """Modify body properties at runtime (no recompile needed).

        Changes take effect on the next mj_step.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model = self._world._model
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"❌ Body '{body_name}' not found."}]}

        changes = []
        if mass is not None:
            old_mass = float(model.body_mass[body_id])
            model.body_mass[body_id] = mass
            changes.append(f"mass: {old_mass:.3f} → {mass:.3f}")

        return {
            "status": "success",
            "content": [{"text": f"🔧 Body '{body_name}': {', '.join(changes)}"}],
        }

    def set_geom_properties(
        self,
        geom_name: str = None,
        geom_id: int = None,
        color: list[float] = None,
        friction: list[float] = None,
        size: list[float] = None,
    ) -> dict[str, Any]:
        """Modify geom properties at runtime (no recompile needed).

        Changes take effect immediately for rendering (color) or next step (friction, size).
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model = self._world._model

        gid = geom_id
        if geom_name:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
        if gid is None or gid < 0:
            return {"status": "error", "content": [{"text": f"❌ Geom '{geom_name or geom_id}' not found."}]}

        label = geom_name or f"geom_{gid}"
        changes = []

        if color is not None:
            model.geom_rgba[gid] = color[:4] if len(color) >= 4 else color[:3] + [1.0]
            changes.append(f"color → {model.geom_rgba[gid].tolist()}")

        if friction is not None:
            fric = friction[:3] if len(friction) >= 3 else friction + [0.0] * (3 - len(friction))
            model.geom_friction[gid] = fric
            changes.append(f"friction → {fric}")

        if size is not None:
            n = min(len(size), 3)
            model.geom_size[gid, :n] = size[:n]
            changes.append(f"size → {model.geom_size[gid].tolist()}")

        return {
            "status": "success",
            "content": [{"text": f"🔧 Geom '{label}': {', '.join(changes)}"}],
        }

    # ── Contact Force Analysis ──

    def get_contact_forces(self) -> dict[str, Any]:
        """Get detailed contact forces for all active contacts.

        Uses mj_contactForce for each active contact pair.
        Returns normal and friction forces.
        """
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
            g2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"

            # Get contact force (normal + friction in contact frame)
            force = np.zeros(6)
            mj.mj_contactForce(model, data, i, force)

            contacts.append(
                {
                    "geom1": g1,
                    "geom2": g2,
                    "distance": float(c.dist),
                    "position": c.pos.tolist(),
                    "normal_force": float(force[0]),
                    "friction_force": force[1:3].tolist(),
                    "full_wrench": force.tolist(),
                }
            )

        if not contacts:
            return {"status": "success", "content": [{"text": "💥 No active contacts."}]}

        lines = [f"💥 {len(contacts)} contacts:"]
        for c in contacts[:15]:
            lines.append(f"  {c['geom1']} ↔ {c['geom2']}: normal={c['normal_force']:.3f}N, dist={c['distance']:.4f}m")
        if len(contacts) > 15:
            lines.append(f"  ... and {len(contacts) - 15} more")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"text": json.dumps({"contacts": contacts}, default=str)}],
        }

    # ── Multi-Ray (batch raycasting) ──

    def multi_raycast(
        self,
        origin: list[float],
        directions: list[list[float]],
        exclude_body: int = -1,
    ) -> dict[str, Any]:
        """Cast multiple rays from a single origin (e.g., for LIDAR simulation).

        Efficiently casts N rays using individual mj_ray calls.
        Returns array of distances and hit geoms.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        pnt = np.array(origin, dtype=np.float64)
        results = []

        for d in directions:
            vec = np.array(d, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            geomid = np.array([-1], dtype=np.int32)
            dist = mj.mj_ray(model, data, pnt, vec, None, 1, exclude_body, geomid)
            results.append(
                {
                    "distance": float(dist) if dist >= 0 else None,
                    "geom_id": int(geomid[0]) if dist >= 0 else None,
                }
            )

        hit_count = sum(1 for r in results if r["distance"] is not None)
        return {
            "status": "success",
            "content": [
                {"text": f"🎯 Multi-ray: {hit_count}/{len(directions)} hits from {origin}"},
                {"text": json.dumps({"rays": results}, default=str)},
            ],
        }

    # ── Forward Kinematics (explicit) ──

    def forward_kinematics(self) -> dict[str, Any]:
        """Run forward kinematics to update all body positions/orientations.

        Usually called implicitly by mj_step, but useful after manually
        setting qpos to see updated Cartesian positions.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_camlight(model, data)

        # Build body position summary
        bodies = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            bodies[name] = {
                "position": data.xpos[i].tolist(),
                "quaternion": data.xquat[i].tolist(),
            }

        return {
            "status": "success",
            "content": [
                {"text": f"🦴 FK computed for {model.nbody} bodies"},
                {"text": json.dumps({"bodies": bodies}, default=str)},
            ],
        }

    # ── Total Mass ──

    def get_total_mass(self) -> dict[str, Any]:
        """Get total mass and per-body mass breakdown."""
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model = self._world._model

        total = float(mj.mj_getTotalmass(model))
        bodies = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            m = float(model.body_mass[i])
            if m > 0:
                bodies[name] = m

        return {
            "status": "success",
            "content": [
                {"text": f"⚖️ Total mass: {total:.4f}kg ({len(bodies)} bodies with mass)"},
                {"text": json.dumps({"total_mass": total, "bodies": bodies}, default=str)},
            ],
        }

    # ── Export Model XML ──

    def export_xml(self, output_path: str = None) -> dict[str, Any]:
        """Export the current model to MJCF XML.

        Uses mj_saveLastXML — exports the exact model currently loaded,
        including any runtime modifications.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()

        if output_path:
            mj.mj_saveLastXML(output_path, self._world._model)
            return {"status": "success", "content": [{"text": f"📄 Model exported to {output_path}"}]}
        else:
            # Return XML string via saveLastXML to temp file
            import os
            import tempfile

            tmpfile = tempfile.mktemp(suffix=".xml")
            mj.mj_saveLastXML(tmpfile, self._world._model)
            with open(tmpfile) as f:
                xml = f.read()
            os.unlink(tmpfile)
            return {
                "status": "success",
                "content": [
                    {"text": f"📄 Model XML ({len(xml)} chars):\n{xml[:2000]}{'...' if len(xml) > 2000 else ''}"}
                ],
            }
