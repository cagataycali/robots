"""Step 6 benchmark: vectorized policy evaluation on mjlab vs the classic MuJoCo backend.

Same ``rsl_rl_onnx`` reach actor, same seeded per-episode targets, same 150 ticks
at 50 Hz, same LeRobot v3 output. The classic row runs ``run_policy`` one episode
at a time on ``backend="mujoco"``; the mjlab rows drive N worlds in lockstep
(``vec_eval.vec_rollout``) with per-world physics randomization and flush the N
episodes through :class:`BatchedLeRobotRecorder`. Success = gripper site within
``SUCCESS_M`` of the target at the last tick, from the robot's own MJCF FK.

    python examples/mjlab/vec_eval_bench.py --onnx train/reach_final.onnx --out scratch/vec_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from strands_robots.policies import create_policy
from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
from strands_robots.simulation import create_simulation
from strands_robots.training.mjlab_tasks.vec_eval import BatchedLeRobotRecorder, open_recorder, vec_rollout

JOINTS = ["1", "2", "3", "4", "5", "6"]  # strands-robots so101 MJCF joint names
TARGET_X, TARGET_Y, TARGET_Z = (0.10, 0.30), (-0.20, 0.20), (0.05, 0.30)
SUCCESS_M = 0.02


def sample_targets(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.array([TARGET_X[0], TARGET_Y[0], TARGET_Z[0]])
    hi = np.array([TARGET_X[1], TARGET_Y[1], TARGET_Z[1]])
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def dataset_truth(root: Path) -> dict:
    import pyarrow.parquet as pq

    files = sorted((root / "data").rglob("*.parquet"))
    frames = sum(pq.read_metadata(f).num_rows for f in files)
    meta = json.loads((root / "meta" / "info.json").read_text()) if (root / "meta" / "info.json").exists() else {}
    return {
        "parquet_files": len(files),
        "frames": frames,
        "info_episodes": meta.get("total_episodes"),
        "info_frames": meta.get("total_frames"),
    }


def classic(onnx: str, targets: np.ndarray, ticks: int, hz: float, root: Path) -> dict:
    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("so101")
    policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot="so101")
    fk = _SiteFK("so101", "gripper", JOINTS)
    rec_root = root / "classic"
    # cameras=[] : proprio only, the same schema the mjlab rows write (no renderer there).
    r = sim.start_recording(repo_id="bench/classic", fps=int(hz), root=str(rec_root), task="reach", cameras=[])
    assert r["status"] == "success", r
    t0 = time.perf_counter()
    successes = []
    finals = []
    for t in targets:
        sim.reset()
        policy.reset()
        res = sim.run_policy(
            "so101",
            policy_object=policy,
            instruction="reach",
            policy_kwargs={"target_pose": t.tolist()},
            max_steps=ticks,
            control_frequency=hz,
            fast_mode=True,
            reset_between=False,
        )
        assert res["status"] == "success", res
        obs = sim.get_observation("so101")
        err = float(np.linalg.norm(fk.site_pos([float(obs[j]) for j in JOINTS]) - t))
        finals.append(err)
        successes.append(err < SUCCESS_M)
    wall = time.perf_counter() - t0
    stop = sim.stop_recording()
    sim.cleanup()
    return {
        "backend": "mujoco",
        "num_envs": 1,
        "episodes": len(targets),
        "wall_s": round(wall, 2),
        "episodes_per_minute": round(60 * len(targets) / wall, 2),
        "success": int(sum(successes)),
        "final_err_median_m": round(float(np.median(finals)), 4),
        "stop_recording": stop["content"][0]["text"][:200],
        "dataset": dataset_truth(rec_root),
    }


async def vectorized(
    onnx: str, n: int, targets: np.ndarray, ticks: int, hz: float, root: Path, randomize: bool, record: bool
) -> dict:
    sim = create_simulation("mjlab", num_envs=n)
    sim.create_world()
    sim.add_robot("so101")
    sim.reset()  # compile + JIT outside the timed region (one-off per process/N)
    policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot="so101")
    fk = _SiteFK("so101", "gripper", JOINTS)
    dr = None
    t_all = time.perf_counter()
    if randomize:
        r = sim.randomize(randomize_physics=True, seed=int(n))
        assert r["status"] == "success", r
        dr = {k: (np.array(v).shape if isinstance(v, list) else v) for k, v in r["content"][1]["json"].items()}
        dr = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dr.items()}
    rec_root = root / f"mjlab_{n}"
    rec_buf = BatchedLeRobotRecorder(n, JOINTS, sim.robot_action_keys("so101"), "reach") if record else None
    kw = [{"target_pose": t.tolist()} for t in targets[:n]]
    res = await vec_rollout(
        sim,
        policy,
        robot_name="so101",
        ticks=ticks,
        instruction="reach",
        control_hz=hz,
        kwargs_per_world=kw,
        recorder=rec_buf,
    )
    finals = [
        float(np.linalg.norm(fk.site_pos([f[j] for j in JOINTS]) - t))
        for f, t in zip(res.final_obs, targets[:n], strict=True)
    ]
    flush = None
    truth = None
    if rec_buf is not None:
        recorder = open_recorder(sim, f"bench/mjlab_{n}", rec_root, int(hz), "reach")
        flush = rec_buf.flush(recorder)
        recorder.finalize()
        truth = dataset_truth(rec_root)
    wall_total = time.perf_counter() - t_all
    sim.cleanup()
    out = res.summary()
    out.update(
        {
            "backend": "mjlab",
            "episodes": n,
            "wall_total_s": round(wall_total, 2),
            "episodes_per_minute_incl_flush": round(60 * n / wall_total, 2),
            "success": int(sum(e < SUCCESS_M for e in finals)),
            "final_err_median_m": round(float(np.median(finals)), 4),
            "randomized": dr,
            "flush": flush,
            "dataset": truth,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--classic-episodes", type=int, default=16)
    ap.add_argument("--sizes", default="16,64,256")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",") if s]
    targets = sample_targets(max([a.classic_episodes, *sizes]), a.seed)
    root = Path(tempfile.mkdtemp(prefix="vec_eval_"))
    rows = []
    try:
        row = classic(a.onnx, targets[: a.classic_episodes], a.ticks, a.hz, root)
        print(json.dumps(row), flush=True)
        rows.append(row)
        for n in sizes:
            for randomize in (False, True):
                row = asyncio.run(
                    vectorized(
                        a.onnx,
                        n,
                        targets,
                        a.ticks,
                        a.hz,
                        root / ("dr" if randomize else "plain"),
                        randomize,
                        not a.no_record,
                    )
                )
                print(json.dumps(row), flush=True)
                rows.append(row)
        # Unbatched policy path once, to price the batched ONNX call.
        sim = create_simulation("mjlab", num_envs=sizes[0])
        sim.create_world()
        sim.add_robot("so101")
        sim.reset()
        policy = create_policy("rsl_rl_onnx", onnx_path=a.onnx, robot="so101")
        kw = [{"target_pose": t.tolist()} for t in targets[: sizes[0]]]
        res = asyncio.run(
            vec_rollout(
                sim,
                policy,
                robot_name="so101",
                ticks=a.ticks,
                control_hz=a.hz,
                kwargs_per_world=kw,
                force_unbatched=True,
            )
        )
        row = res.summary()
        row.update({"backend": "mjlab", "note": "force_unbatched: N get_actions calls per tick"})
        print(json.dumps(row), flush=True)
        rows.append(row)
        sim.cleanup()
    finally:
        Path(a.out).write_text(
            json.dumps({"ticks": a.ticks, "hz": a.hz, "seed": a.seed, "onnx": a.onnx, "rows": rows}, indent=1)
        )
        if a.keep:
            print("datasets kept at", root)
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
