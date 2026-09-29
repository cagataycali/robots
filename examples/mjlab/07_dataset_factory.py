"""Dataset factory: how many LeRobot v3 episodes per hour do 1024 mjlab worlds produce?

Runs the so101 reach scene with N worlds in lockstep for a wall-clock budget,
every batch with fresh seeded targets and fresh per-world physics randomisation
(``MjlabEngine.randomize``), and streams every episode through
:class:`BatchedLeRobotRecorder` into ONE LeRobot v3 dataset. Two drivers:

* ``--policy scripted``: a batched damped-least-squares IK expert built on the
  robot's own MJCF forward kinematics (``_SiteFK``), the "expert data" row.
* ``--policy onnx``: the rsl_rl reach actor trained by the mjlab trainer, the
  "policy data" row (self-play / DAgger-style collection).

Every batch is timed separately (rollout, flush, reset) so the JSON says where
the hour goes; the writer, not the physics, was the ceiling in the deep lane and
this measures it at scale. Success per episode is the gripper site within
``SUCCESS_M`` of the target at the last tick. With ``--push`` the dataset goes to
the Hub as a private repo and ``repo_info`` is read back so the reported episode
count is the Hub's, not this process's.

Usage::

    python examples/mjlab/07_dataset_factory.py --policy scripted --minutes 30 --num-envs 1024 --root runs/factory_scripted --out factory_scripted.json
    python examples/mjlab/07_dataset_factory.py --policy onnx --onnx runs/reach.onnx --minutes 30 --num-envs 1024 --root runs/factory_onnx --out factory_onnx.json --push cagataydev/mjlab-factory-so101-20260929

Install (the lerobot extra first, then this one: mjlab needs torch>=2.14)::

    uv pip install "strands-robots[sim-mjlab,rl]" huggingface_hub
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np

ROBOT = "so101"
JOINTS = ["1", "2", "3", "4", "5", "6"]


class ScriptedReachExpert:
    """Batched damped-least-squares IK on the gripper site: one Jacobian per world per tick."""

    def __init__(self, fk, gain: float = 1.0, damping: float = 1e-3, max_step: float = 0.2) -> None:
        self.fk, self.gain, self.damping, self.max_step = fk, gain, damping, max_step
        self.eps = 1e-4

    def reset(self) -> None:
        return None

    def _jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p0 = np.asarray(self.fk.site_pos(list(q)))
        J = np.zeros((3, len(q)))
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += self.eps
            J[:, i] = (np.asarray(self.fk.site_pos(list(dq))) - p0) / self.eps
        return p0, J

    def _one(self, obs: dict, target: np.ndarray) -> dict:
        q = np.array([float(obs[j]) for j in JOINTS])
        p, J = self._jacobian(q)
        err = target - p
        JJt = J @ J.T + self.damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err) * self.gain
        norm = float(np.linalg.norm(dq))
        if norm > self.max_step:
            dq *= self.max_step / norm
        q_new = q + dq
        q_new[5] = q[5]  # the gripper does not help a reach
        return {j: float(v) for j, v in zip(JOINTS, q_new, strict=True)}

    async def get_actions_batch(self, observations, instruction: str = "", kwargs_per_world=None):
        kws = kwargs_per_world or [{}] * len(observations)
        return [
            self._one(o, np.asarray(kw["target_pose"], dtype=float)) for o, kw in zip(observations, kws, strict=True)
        ]

    async def get_actions(self, observation, instruction: str = "", **kwargs):
        return [self._one(observation, np.asarray(kwargs["target_pose"], dtype=float))]


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


async def factory(
    policy_kind: str,
    onnx: str | None,
    minutes: float,
    n: int,
    ticks: int,
    hz: float,
    root: Path,
    seed: int,
    push: str | None,
) -> dict:
    from strands_robots.policies import create_policy
    from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
    from strands_robots.simulation import create_simulation
    from strands_robots.training.mjlab_tasks.so101_reach import SUCCESS_M, TARGET_X, TARGET_Y, TARGET_Z
    from strands_robots.training.mjlab_tasks.vec_eval import BatchedLeRobotRecorder, open_recorder, vec_rollout

    if root.exists():
        raise SystemExit(
            f"{root} exists; the factory refuses to append to an older dataset (delete it or pick a new --root)"
        )
    rng = np.random.default_rng(seed)
    lo = np.array([TARGET_X[0], TARGET_Y[0], TARGET_Z[0]])
    hi = np.array([TARGET_X[1], TARGET_Y[1], TARGET_Z[1]])
    sim = create_simulation("mjlab", num_envs=n)
    sim.create_world()
    sim.add_robot(ROBOT)
    sim.reset()  # compile + JIT once
    fk = _SiteFK(ROBOT, "gripper", JOINTS)
    policy = (
        ScriptedReachExpert(fk)
        if policy_kind == "scripted"
        else create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
    )
    task = f"reach ({policy_kind})"
    repo_id = push or f"local/mjlab_factory_{policy_kind}"
    recorder = open_recorder(sim, repo_id, root, int(hz), task)
    action_keys = sim.robot_action_keys(ROBOT)
    batches = []
    t_begin = time.perf_counter()
    budget_s = minutes * 60.0
    k = 0
    while time.perf_counter() - t_begin < budget_s:
        t0 = time.perf_counter()
        r = sim.randomize(randomize_physics=True, seed=seed * 1000 + k)
        assert r["status"] == "success", r
        targets = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        buf = BatchedLeRobotRecorder(n, JOINTS, action_keys, task)
        t1 = time.perf_counter()
        res = await vec_rollout(
            sim,
            policy,
            robot_name=ROBOT,
            ticks=ticks,
            instruction=task,
            control_hz=hz,
            kwargs_per_world=[{"target_pose": t.tolist()} for t in targets],
            recorder=buf,
        )
        t2 = time.perf_counter()
        finals = [
            float(np.linalg.norm(fk.site_pos([f[j] for j in JOINTS]) - t))
            for f, t in zip(res.final_obs, targets, strict=True)
        ]
        flush = buf.flush(recorder)
        t3 = time.perf_counter()
        batches.append(
            {
                "batch": k,
                "randomize_s": round(t1 - t0, 2),
                "rollout_s": round(t2 - t1, 2),
                "policy_s": round(float(getattr(res, "policy_s", 0.0)), 2),
                "flush_s": round(t3 - t2, 2),
                "episodes": flush["episodes"],
                "frames": flush["frames"],
                "success": int(sum(e < SUCCESS_M for e in finals)),
                "final_err_median_m": round(float(np.median(finals)), 4),
                "elapsed_min": round((t3 - t_begin) / 60.0, 2),
            }
        )
        print(json.dumps(batches[-1]), flush=True)
        k += 1
    recorder.finalize()
    wall = time.perf_counter() - t_begin
    sim.cleanup()
    episodes = sum(b["episodes"] for b in batches)
    frames = sum(b["frames"] for b in batches)
    size = _dir_bytes(root)
    rec = {
        "policy": policy_kind,
        "onnx": onnx,
        "num_envs": n,
        "ticks": ticks,
        "hz": hz,
        "minutes_budget": minutes,
        "wall_min": round(wall / 60.0, 2),
        "batches": len(batches),
        "episodes": episodes,
        "frames": frames,
        "success": sum(b["success"] for b in batches),
        "success_rate": round(sum(b["success"] for b in batches) / max(1, episodes), 4),
        "dataset_bytes": size,
        "dataset_gb": round(size / 1e9, 3),
        "episodes_per_hour": round(episodes / (wall / 3600.0)),
        "frames_per_hour": round(frames / (wall / 3600.0)),
        "gb_per_hour": round(size / 1e9 / (wall / 3600.0), 3),
        "time_split_s": {
            "rollout": round(sum(b["rollout_s"] for b in batches), 1),
            "flush": round(sum(b["flush_s"] for b in batches), 1),
            "randomize": round(sum(b["randomize_s"] for b in batches), 1),
        },
        "per_batch": batches,
        "root": str(root),
    }
    if push:
        from huggingface_hub import HfApi

        pushed = recorder.push_to_hub(tags=["mjlab", "so101", "reach", "synthetic"], private=True)
        info = HfApi().repo_info(push, repo_type="dataset", files_metadata=True)
        rec["hub"] = {
            "push": {k2: v for k2, v in pushed.items() if k2 != "content"},
            "repo": push,
            "sha": info.sha,
            "private": info.private,
            "files": len(info.siblings or []),
            "bytes": sum((s.size or 0) for s in (info.siblings or [])),
        }
    return rec


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=("scripted", "onnx"), required=True)
    p.add_argument("--onnx", help="rsl_rl_onnx actor for --policy onnx")
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--ticks", type=int, default=150)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--root", required=True, help="fresh directory for the LeRobot v3 dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--push", help="private Hub repo id to push the dataset to (read back with repo_info)")
    a = p.parse_args(argv)
    if a.policy == "onnx" and not a.onnx:
        p.error("--policy onnx needs --onnx")
    os.environ.setdefault("MUJOCO_GL", "egl")
    rec = asyncio.run(factory(a.policy, a.onnx, a.minutes, a.num_envs, a.ticks, a.hz, Path(a.root), a.seed, a.push))
    Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "per_batch"}, indent=1))


if __name__ == "__main__":
    main()
