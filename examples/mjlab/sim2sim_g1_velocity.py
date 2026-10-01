"""Sim-to-sim: play an mjlab-trained Unitree G1 velocity actor on both strands-robots engines.

Conditions = engine x actuator gains x command:

* engine  ``mjlab`` (MuJoCo-Warp) and ``mujoco`` (classic)
* gains   ``stock``   = the ``g1.xml`` shipped with strands-robots (``kp=500 dampratio=1``)
          ``trained`` = the per-joint kp/kd the actor was trained with, read from the
                        ONNX metadata (``joint_stiffness`` / ``joint_damping``) and
                        written into a patched copy of the same MJCF
* command ``[vx, vy, wz]`` in the base frame, held for the whole episode

Per episode: ticks survived (fall = base height below ``FALL_Z`` or the body z axis
tilted past ``FALL_TILT``), mean planar-velocity tracking error, mean yaw-rate
error and mean base height over the survived ticks. ``--baseline hold`` plays a
zero action (hold the default pose) through the same decode for reference.

Usage: python examples/mjlab/sim2sim_g1_velocity.py <g1.onnx> --out s2s_g1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from strands_robots import Robot
from strands_robots.assets import resolve_model_path, resolve_robot_name
from strands_robots.policies import create_policy
from strands_robots.policies.rsl_rl_onnx.policy import RslRlOnnxPolicy, _quat_rotate_inverse

ROBOT = "unitree_g1"
FALL_Z = 0.45
FALL_TILT = 0.5  # projected gravity z > -0.5  <=>  tilt > 60 deg
COMMANDS = {
    "stand": [0.0, 0.0, 0.0],
    "fwd_0.5": [0.5, 0.0, 0.0],
    "fwd_1.0": [1.0, 0.0, 0.0],
    "yaw_0.5": [0.0, 0.0, 0.5],
}


def patched_mjcf(stiffness: dict[str, float], damping: dict[str, float]) -> str:
    """Copy the g1 asset dir and rewrite every <position> actuator with the trained kp/kv."""
    src = Path(resolve_model_path(resolve_robot_name(ROBOT)))
    dst_dir = Path(tempfile.mkdtemp(prefix="g1_trained_gains_"))
    shutil.copytree(src.parent, dst_dir, dirs_exist_ok=True)
    xml = (dst_dir / src.name).read_text()

    def repl(m: re.Match) -> str:
        joint = m.group(2)
        kp, kv = stiffness[joint], damping[joint]
        return f'{m.group(1)}kp="{kp}" kv="{kv}"/>'

    xml, n = re.subn(r'(<position class="g1" name="[^"]+" joint="([^"]+)")\s*/>', repl, xml)
    if n != len(stiffness):
        raise RuntimeError(f"patched {n} actuators, expected {len(stiffness)}")
    # drop the class-level dampratio so the per-actuator kv is what MuJoCo uses
    xml = xml.replace('<position kp="500" dampratio="1" inheritrange="1"/>', '<position inheritrange="1"/>')
    out = dst_dir / src.name
    out.write_text(xml)
    return str(out)


def gains_from_onnx(onnx: str) -> tuple[dict[str, float], dict[str, float]]:
    import onnxruntime as ort

    md = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    names = md["joint_names"].split(",")
    kp = [float(x) for x in md["joint_stiffness"].split(",")]
    kd = [float(x) for x in md["joint_damping"].split(",")]
    return dict(zip(names, kp)), dict(zip(names, kd))


def _base(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    quat = np.asarray(obs["base_quat"], dtype=np.float64)
    lin_b = _quat_rotate_inverse(quat, np.asarray(obs["base_lin_vel"], dtype=np.float64))
    ang_b = np.asarray(obs["base_ang_vel"], dtype=np.float64)
    grav = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    return lin_b, ang_b, grav, float(obs["base_pos"][2])


async def rollout(sim, policy, command, ticks: int, hz: float) -> dict:
    sim.reset()
    if policy is not None:
        policy.reset()
    n_sub = max(1, int(round(sim.physics_timestep() ** -1 / hz)))
    v_err, w_err, z_hist = [], [], []
    survived = 0
    t0 = time.perf_counter()
    for _ in range(ticks):
        obs = sim.get_observation(ROBOT)
        lin_b, ang_b, grav, z = _base(obs)
        if z < FALL_Z or grav[2] > -FALL_TILT:
            break
        survived += 1
        v_err.append(float(np.linalg.norm(lin_b[:2] - np.asarray(command[:2]))))
        w_err.append(abs(float(ang_b[2]) - command[2]))
        z_hist.append(z)
        if policy is None:
            action = HOLD
        else:
            chunk = await policy.get_actions(obs, "", target_velocity=command)
            action = chunk[0]
        sim.send_action(action, ROBOT, n_substeps=n_sub)
    return {
        "survived_ticks": survived,
        "survived_s": survived / hz,
        "fell": survived < ticks,
        "v_xy_err_mean": float(np.mean(v_err)) if v_err else None,
        "w_z_err_mean": float(np.mean(w_err)) if w_err else None,
        "base_z_mean": float(np.mean(z_hist)) if z_hist else None,
        "wall_s": time.perf_counter() - t0,
    }


HOLD: dict[str, float] = {}


async def evaluate(
    onnx: str, backend: str, gains: str, mjcf: str | None, ticks: int, hz: float, baseline: bool
) -> dict:
    global HOLD
    t0 = time.perf_counter()
    kw = {"num_envs": 1} if backend == "mjlab" else {}
    if mjcf:
        kw["urdf_path"] = mjcf
    sim = Robot(ROBOT, backend=backend, **kw)
    build_s = time.perf_counter() - t0
    policy: RslRlOnnxPolicy | None = None
    if baseline:
        md = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
        HOLD = {j: float(q) for j, q in zip(md.spec.joint_names, md.spec.default_joint_pos)}
    else:
        policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
    eps = {}
    for cname, cmd in COMMANDS.items():
        eps[cname] = await rollout(sim, policy, cmd, ticks, hz)
        print(
            f"  {backend:6s} {gains:7s} {'hold' if baseline else 'actor':5s} {cname:8s} "
            f"survived {eps[cname]['survived_s']:5.1f}s  v_err {eps[cname]['v_xy_err_mean']}  z {eps[cname]['base_z_mean']}",
            flush=True,
        )
    sim.cleanup()
    return {
        "backend": backend,
        "gains": gains,
        "policy": "hold" if baseline else "rsl_rl_onnx",
        "build_s": round(build_s, 2),
        "episodes": eps,
        "survived_all": all(not e["fell"] for e in eps.values()),
        "survived_s_total": sum(e["survived_s"] for e in eps.values()),
        "ticks_per_s": round(
            sum(e["survived_ticks"] for e in eps.values()) / max(1e-9, sum(e["wall_s"] for e in eps.values())), 1
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--out", required=True)
    p.add_argument("--ticks", type=int, default=500, help="10 s at 50 Hz (the task's control rate)")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--backends", default="mjlab,mujoco")
    p.add_argument("--gains", default="stock,trained")
    p.add_argument("--baseline", default="hold", help="'' to skip the hold-pose baseline")
    a = p.parse_args()

    kp, kd = gains_from_onnx(a.onnx)
    mjcf_trained = patched_mjcf(kp, kd)
    out = {
        "onnx": a.onnx,
        "ticks": a.ticks,
        "hz": a.hz,
        "fall_z": FALL_Z,
        "fall_tilt": FALL_TILT,
        "commands": COMMANDS,
        "trained_mjcf": mjcf_trained,
        "runs": [],
    }
    for backend in a.backends.split(","):
        for gains in a.gains.split(","):
            mjcf = mjcf_trained if gains == "trained" else None
            out["runs"].append(asyncio.run(evaluate(a.onnx, backend, gains, mjcf, a.ticks, a.hz, False)))
            if a.baseline and gains == "stock":
                out["runs"].append(asyncio.run(evaluate(a.onnx, backend, gains, mjcf, a.ticks, a.hz, True)))
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
