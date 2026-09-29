"""Curriculum over the goal box: one goal-conditioned so101 policy on a box 3x the reach task's.

The stock reach task samples targets from a fixed box (``TARGET_X/Y/Z``). This
example keeps the same observations, actions and rewards, but every world
carries its own curriculum ``level``; the target box of a world scales about the
stock box's centre by 1.0, 1.5, 2.0, 2.5, 3.0 for levels 0..4 (3x the stock
box at the top, which reaches past the so101 workspace on purpose: the report
states how many targets per stage are reachable at all). A world is promoted on
every episode it ends at the goal and demoted on every episode it ends more than
15 cm away, the same shape as mjlab's terrain curriculum, and the schedule is
logged per iteration as the mean level plus the at-goal fraction per level.

After training the actor is exported to ONNX and evaluated on classic MuJoCo on
20 seeded targets per stage box, next to the stock reach actor (trained on the
1x box only, from example 04 ``--dr none``) on the same targets. The question:
does one goal-conditioned policy with a curriculum cover the 3x box, and what
does the 1x policy do when asked for targets outside its training box?

Usage::

    python examples/mjlab/05_curriculum_goal_box.py train --num-envs 1024 --iterations 400 --run-dir runs/reach_curriculum
    python examples/mjlab/05_curriculum_goal_box.py export --checkpoint runs/reach_curriculum/model_399.pt --onnx runs/reach_curriculum.onnx
    python examples/mjlab/05_curriculum_goal_box.py eval --curriculum runs/reach_curriculum.onnx --baseline runs/reach_none.onnx --out curriculum_eval.json

Install (the lerobot extra first, then this one: mjlab needs torch>=2.14)::

    uv pip install "strands-robots[sim-mjlab,rl]"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROBOT = "so101"
JOINTS = ["1", "2", "3", "4", "5", "6"]
TASK_ID = "Strands-Reach-So101-GoalBoxCurriculum"
LEVEL_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
PROMOTE_AT_GOAL = True
DEMOTE_ERR_M = 0.15
METRIC = "Metrics/reach/at_goal"


# ------------------------------------------------------------------- task


def box_for(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The stock target box scaled about its centre (floor-clipped at z = 2 cm)."""
    from strands_robots.training.mjlab_tasks.so101_reach import TARGET_X, TARGET_Y, TARGET_Z

    lo = np.array([TARGET_X[0], TARGET_Y[0], TARGET_Z[0]])
    hi = np.array([TARGET_X[1], TARGET_Y[1], TARGET_Z[1]])
    c, h = (lo + hi) / 2, (hi - lo) / 2 * scale
    lo2, hi2 = c - h, c + h
    lo2[2] = max(lo2[2], 0.02)
    return lo2, hi2


def curriculum_env_cfg(play: bool = False):
    import torch
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    from strands_robots.training.mjlab_tasks import so101_reach as base

    boxes = [box_for(s) for s in LEVEL_SCALES]
    lo_t = torch.as_tensor(np.stack([b[0] for b in boxes]), dtype=torch.float32)
    hi_t = torch.as_tensor(np.stack([b[1] for b in boxes]), dtype=torch.float32)

    class CurriculumReachCommand(base.ReachCommand):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.lo, self.hi = lo_t.to(self.device), hi_t.to(self.device)
            self.metrics["level"] = torch.zeros(self.num_envs, device=self.device)

        def _resample_command(self, env_ids: torch.Tensor) -> None:
            lo, hi = self.lo[self.level[env_ids]], self.hi[self.level[env_ids]]
            self.target_pos_b[env_ids] = lo + torch.rand_like(lo) * (hi - lo)

        def _update_metrics(self) -> None:
            super()._update_metrics()
            self.metrics["level"] = self.level.float()

    @dataclass(kw_only=True)
    class CurriculumReachCommandCfg(base.ReachCommandCfg):
        def build(self, env):
            return CurriculumReachCommand(self, env)

    def goal_box_levels(env, env_ids, command_name: str = "reach"):
        """Promote a resetting world that ends at the goal, demote one that ends far away; returns the mean level."""
        cmd = env.command_manager.get_term(command_name)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(cmd.num_envs, device=cmd.device)[env_ids]
        # Only worlds that actually ran an episode move (the very first reset has zero metrics).
        env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
        if len(env_ids):
            err = cmd.metrics["position_error"][env_ids]
            up = err < cmd.cfg.success_threshold
            down = err > DEMOTE_ERR_M
            lvl = cmd.level[env_ids]
            lvl = torch.where(up, lvl + 1, lvl)
            lvl = torch.where(down, lvl - 1, lvl)
            cmd.level[env_ids] = lvl.clamp(0, len(LEVEL_SCALES) - 1)
        return cmd.level.float().mean()

    cfg = base.so101_reach_env_cfg(play=play)
    cfg.commands = {"reach": CurriculumReachCommandCfg(resampling_time_range=(3.0, 5.0), debug_vis=play)}
    if not play:
        cfg.curriculum = {"goal_box": CurriculumTermCfg(func=goal_box_levels, params={"command_name": "reach"})}
    return cfg


def register() -> str:
    from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task

    from strands_robots.training.mjlab_tasks.so101_reach import register as register_base

    base = register_base()
    if TASK_ID not in list_tasks():
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=curriculum_env_cfg(),
            play_env_cfg=curriculum_env_cfg(play=True),
            rl_cfg=load_rl_cfg(base),
        )
    return TASK_ID


# ------------------------------------------------------------------ train


def train(num_envs: int, iterations: int, run_dir: Path, seed: int) -> None:
    import torch  # noqa: F401  (rsl_rl needs torch first)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    tid = register()
    env_cfg = load_env_cfg(tid)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(tid)
    agent_cfg.max_iterations = iterations
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = 100
    run_dir.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=str(run_dir), device="cuda:0")
    cmd = env.command_manager.get_term("reach")
    schedule: list[dict] = []
    logger = runner.logger
    original_log = logger.log

    def hooked_log(*, it: int, collect_time: float, learn_time: float, **kw) -> None:
        # Instantaneous per-level at-goal fraction over all worlds + the level histogram.
        at_goal = cmd.metrics["at_goal"]
        row = {"it": it, "mean_level": round(float(cmd.level.float().mean()), 3), "levels": {}, "at_goal_by_level": {}}
        for k in range(len(LEVEL_SCALES)):
            m = cmd.level == k
            n = int(m.sum())
            row["levels"][str(k)] = n
            row["at_goal_by_level"][str(k)] = round(float(at_goal[m].mean()), 3) if n else None
        schedule.append(row)
        original_log(it=it, collect_time=collect_time, learn_time=learn_time, **kw)

    logger.log = hooked_log  # type: ignore[method-assign]
    t0 = time.monotonic()
    runner.learn(iterations, init_at_random_ep_len=True)
    (run_dir / "curriculum_schedule.json").write_text(
        json.dumps(
            {
                "task": tid,
                "num_envs": num_envs,
                "iterations": iterations,
                "wall_s": round(time.monotonic() - t0, 1),
                "level_scales": LEVEL_SCALES,
                "schedule": schedule,
            }
        ),
        encoding="utf-8",
    )
    env.close()


def export(checkpoint: Path, onnx: Path) -> None:
    from strands_robots.training.mjlab_tasks.export import export_checkpoint

    print(export_checkpoint(register(), checkpoint, onnx, run_name="reach_curriculum"))


# ------------------------------------------------------------------- eval


def reachable_fraction(fk, targets: np.ndarray, tol_m: float, seed: int = 0, n_cloud: int = 20000) -> float:
    """Share of targets within ``tol_m`` of some FK sample of random joint configurations (workspace check)."""
    import mujoco

    m = fk.model
    rng = np.random.default_rng(seed)
    lo, hi = [], []
    for j in JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo.append(m.jnt_range[jid, 0] if m.jnt_limited[jid] else -np.pi)
        hi.append(m.jnt_range[jid, 1] if m.jnt_limited[jid] else np.pi)
    cloud = np.stack([fk.site_pos(list(q)) for q in rng.uniform(lo, hi, size=(n_cloud, len(JOINTS)))])
    d = np.linalg.norm(targets[:, None, :] - cloud[None, :, :], axis=-1).min(1)
    return float((d < tol_m).mean())


async def evaluate(onnx_by_name: dict[str, str], n: int, seed: int, ticks: int, hz: float, out: Path) -> dict:
    import sys

    from strands_robots import Robot
    from strands_robots.policies import create_policy
    from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
    from strands_robots.training.mjlab_tasks.so101_reach import SUCCESS_M

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sim2sim_reach import rollout

    fk = _SiteFK(ROBOT, "gripper", JOINTS)
    rng = np.random.default_rng(seed)
    stages = {}
    for k, s in enumerate(LEVEL_SCALES):
        lo, hi = box_for(s)
        targets = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        stage = {
            "scale": s,
            "box_lo": np.round(lo, 3).tolist(),
            "box_hi": np.round(hi, 3).tolist(),
            "reachable_fraction": round(reachable_fraction(fk, targets, SUCCESS_M), 3),
            "policies": {},
        }
        for label, onnx in onnx_by_name.items():
            policy = create_policy("rsl_rl_onnx", onnx_path=onnx, robot=ROBOT)
            sim = Robot(ROBOT, backend="mujoco")
            eps = [await rollout(sim, policy, fk, t, ticks, hz) for t in targets]
            sim.cleanup()
            succ = sum(e["success"] for e in eps)
            stage["policies"][label] = {
                "success": f"{succ}/{n}",
                "success_rate": succ / n,
                "final_err_median_m": round(float(np.median([e["final_err_m"] for e in eps])), 4),
                "episodes": eps,
            }
            print(
                f"level {k} (x{s}) {label:10s} {succ:>2d}/{n}  median final err {stage['policies'][label]['final_err_median_m']:.4f} m  reachable {stage['reachable_fraction']:.2f}",
                flush=True,
            )
        stages[str(k)] = stage
    rec = {
        "backend": "mujoco",
        "n_per_stage": n,
        "seed": seed,
        "ticks": ticks,
        "hz": hz,
        "success_m": SUCCESS_M,
        "policies": onnx_by_name,
        "level_scales": LEVEL_SCALES,
        "stages": stages,
    }
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--num-envs", type=int, default=1024)
    t.add_argument("--iterations", type=int, default=400)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--onnx", required=True)
    v = sub.add_parser("eval")
    v.add_argument("--curriculum", required=True, help="ONNX of the curriculum actor")
    v.add_argument("--baseline", help="ONNX of the stock 1x-box reach actor (optional)")
    v.add_argument("--out", required=True)
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--seed", type=int, default=11)
    v.add_argument("--ticks", type=int, default=200)
    v.add_argument("--hz", type=float, default=50.0)
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.cmd == "train":
        train(a.num_envs, a.iterations, Path(a.run_dir), a.seed)
    elif a.cmd == "export":
        export(Path(a.checkpoint), Path(a.onnx))
    else:
        policies = {"curriculum": a.curriculum}
        if a.baseline:
            policies["baseline_1x"] = a.baseline
        asyncio.run(evaluate(policies, a.n, a.seed, a.ticks, a.hz, Path(a.out)))


if __name__ == "__main__":
    main()
