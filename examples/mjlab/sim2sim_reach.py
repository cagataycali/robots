"""Sim-to-sim: play an mjlab-trained so101 reach actor on both strands-robots engines.

Same N seeded targets, same tick rate, both ``backend="mjlab"`` (MuJoCo-Warp)
and ``backend="mujoco"`` (classic). Success = gripper site within
``SUCCESS_M`` of the target at the end of the episode; also reports the
minimum error along the way. Writes JSON; every number in REPORT.md points here.

Usage: python examples/mjlab/sim2sim_reach.py <reach.onnx> --out s2s.json [--n 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

from strands_robots import Robot
from strands_robots.policies import create_policy
from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
from strands_robots.training.mjlab_tasks.so101_reach import (
    SUCCESS_M,
    TARGET_X,
    TARGET_Y,
    TARGET_Z,
)

JOINTS = ["1", "2", "3", "4", "5", "6"]


def sample_targets(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.array([TARGET_X[0], TARGET_Y[0], TARGET_Z[0]])
    hi = np.array([TARGET_X[1], TARGET_Y[1], TARGET_Z[1]])
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


async def rollout(sim, policy, fk: _SiteFK, target, ticks: int, hz: float) -> dict:
    sim.reset()
    policy.reset()
    errs = []
    n_sub = max(1, int(round(sim.physics_timestep() ** -1 / hz)))
    t0 = time.perf_counter()
    for _ in range(ticks):
        obs = sim.get_observation("so101")
        q = [float(obs[j]) for j in JOINTS]
        errs.append(float(np.linalg.norm(fk.site_pos(q) - target)))
        chunk = await policy.get_actions(obs, "", target_pose=target.tolist())
        sim.send_action(chunk[0], "so101", n_substeps=n_sub)
    obs = sim.get_observation("so101")
    q = [float(obs[j]) for j in JOINTS]
    final = float(np.linalg.norm(fk.site_pos(q) - target))
    return {
        "final_err_m": final,
        "min_err_m": float(min(errs + [final])),
        "success": final < SUCCESS_M,
        "wall_s": time.perf_counter() - t0,
    }


async def evaluate(onnx: str, backend: str, targets: np.ndarray, ticks: int, hz: float, policy_provider: str) -> dict:
    t0 = time.perf_counter()
    sim = Robot("so101", backend=backend, num_envs=1) if backend == "mjlab" else Robot("so101", backend=backend)
    build_s = time.perf_counter() - t0
    if policy_provider == "rsl_rl_onnx":
        policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot="so101")
    else:
        policy = create_policy(policy_provider)
    fk = _SiteFK("so101", "gripper", JOINTS)
    eps = [await rollout(sim, policy, fk, t, ticks, hz) for t in targets]
    sim.cleanup()
    succ = sum(e["success"] for e in eps)
    return {
        "backend": backend,
        "policy": policy_provider,
        "build_s": round(build_s, 2),
        "episodes": eps,
        "success": f"{succ}/{len(eps)}",
        "success_rate": succ / len(eps),
        "final_err_median_m": float(np.median([e["final_err_m"] for e in eps])),
        "min_err_median_m": float(np.median([e["min_err_m"] for e in eps])),
        "ticks_per_s": round(ticks * len(eps) / sum(e["wall_s"] for e in eps), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=150, help="3 s at 50 Hz, the task's control rate")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--backends", default="mjlab,mujoco")
    p.add_argument("--baselines", default="mock", help="comma list of extra baseline providers (e.g. mock)")
    a = p.parse_args()

    targets = sample_targets(a.n, a.seed)
    out = {"onnx": a.onnx, "n": a.n, "seed": a.seed, "ticks": a.ticks, "hz": a.hz, "success_m": SUCCESS_M, "runs": []}
    for backend in a.backends.split(","):
        for prov in ["rsl_rl_onnx"] + [b for b in a.baselines.split(",") if b]:
            r = asyncio.run(evaluate(a.onnx, backend, targets, a.ticks, a.hz, prov))
            out["runs"].append(r)
            print(
                f"{backend:7s} {prov:12s} success {r['success']:>6s} "
                f"final median {r['final_err_median_m']:.3f} m  min median {r['min_err_median_m']:.3f} m  "
                f"{r['ticks_per_s']} ticks/s (build {r['build_s']} s)",
                flush=True,
            )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
