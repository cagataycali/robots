"""Native reference for sim2sim_g1_velocity.py: play the same ONNX inside the mjlab
task env it was trained in (play cfg, 1 env, fixed command per episode), and report
survival / tracking with the same definitions. This is the denominator for the
transfer gap measured on the strands-robots engines.

Usage: python examples/mjlab/native_g1_velocity_reference.py <g1.onnx> --out native.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim2sim_g1_velocity import COMMANDS, FALL_TILT, FALL_Z  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-G1")
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    import onnxruntime as ort
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from strands_robots.training import mjlab_tasks

    mjlab_tasks.register_all()
    cfg = load_env_cfg(a.task, play=True)
    cfg.scene.num_envs = 1
    # Hold the command for the whole episode; no pushes / randomisation noise beyond the play cfg.
    tw = cfg.commands["twist"] if isinstance(cfg.commands, dict) else cfg.commands.twist
    tw.resampling_time_range = (1e9, 1e9)
    tw.rel_standing_envs = 0.0
    tw.rel_heading_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    robot = env.scene["robot"]
    out = {"onnx": a.onnx, "task": a.task, "ticks": a.ticks, "episodes": {}}
    try:
        for cname, cmd in COMMANDS.items():
            obs, _ = env.reset()
            twist = env.command_manager.get_term("twist")
            twist.vel_command_b[:] = torch.tensor(cmd, device=a.device)
            obs = env.observation_manager.compute(
                update_history=True
            )  # re-read with the held command (history_length 0)
            v_err, w_err, z_hist = [], [], []
            survived = 0
            t0 = time.perf_counter()
            for _ in range(a.ticks):
                twist.vel_command_b[:] = torch.tensor(cmd, device=a.device)
                o = obs["actor"]
                act = sess.run(None, {in_name: o.detach().cpu().numpy().astype(np.float32)})[0]
                z = float(robot.data.root_link_pos_w[0, 2])
                grav = float(robot.data.projected_gravity_b[0, 2])
                if z < FALL_Z or grav > -FALL_TILT:
                    break
                survived += 1
                lin_b = robot.data.root_link_lin_vel_b[0, :2].detach().cpu().numpy()
                v_err.append(float(np.linalg.norm(lin_b - np.asarray(cmd[:2]))))
                w_err.append(abs(float(robot.data.root_link_ang_vel_b[0, 2]) - cmd[2]))
                z_hist.append(z)
                obs, _, terminated, truncated, _ = env.step(torch.as_tensor(act, device=a.device))
                if bool(terminated[0]) and not bool(truncated[0]):
                    break
            hz = 1.0 / env.step_dt
            out["episodes"][cname] = {
                "survived_ticks": survived,
                "survived_s": survived / hz,
                "fell": survived < a.ticks,
                "v_xy_err_mean": float(np.mean(v_err)) if v_err else None,
                "w_z_err_mean": float(np.mean(w_err)) if w_err else None,
                "base_z_mean": float(np.mean(z_hist)) if z_hist else None,
                "wall_s": time.perf_counter() - t0,
            }
            e = out["episodes"][cname]
            print(
                f"  native {cname:8s} survived {e['survived_s']:5.1f}s  v_err {e['v_xy_err_mean']}  z {e['base_z_mean']}",
                flush=True,
            )
    finally:
        env.close()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
