"""Scale sweep: how far does one GPU take the so101 reach task?

Trains ``Strands-Reach-SO101`` with rsl_rl PPO at several ``num_envs`` and
records, per run: env-steps/s (policy steps, collection only and end to end),
wall clock and iterations to the first logged ``Metrics/reach/at_goal >= target``,
peak GPU memory (torch allocator high water mark plus the Warp mempool high
water mark, which is where MuJoCo Warp allocates), with and without CUDA graph
capture. Everything goes to one JSON, the plot helper renders a PNG from it.

Usage::

    python examples/mjlab/01_scale_sweep.py --num-envs 64 256 1024 4096 \
        --graph both --out sweep.json
    python examples/mjlab/01_scale_sweep.py --plot sweep.json --png assets/scale_sweep.png

One process per (num_envs, graph) pair is the honest way to read memory
(``--num-envs`` with one value); the driver loop in the lane's scratch script
does exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

TASK = "Strands-Reach-SO101"
METRIC = "Metrics/reach/at_goal"


class _Converged(Exception):
    """Raised from the log hook once the success metric clears the target."""


def _gpu_mem_gb() -> dict[str, float]:
    import torch
    import warp as wp

    dev = wp.get_device("cuda:0")
    out = {"torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
    try:
        out["warp_mempool_high_gb"] = wp.get_mempool_used_mem_high(dev) / 2**30
    except Exception:  # pragma: no cover - older warp
        out["warp_mempool_high_gb"] = float("nan")
    return out


def run_one(num_envs: int, use_graph: bool, *, target: float, max_iterations: int, seed: int, log_root: Path) -> dict:
    """Train one configuration until ``at_goal >= target`` or ``max_iterations``."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.sim import sim as mjlab_sim
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from strands_robots.training import mjlab_tasks

    mjlab_tasks.register_all()
    if not use_graph:
        # mjlab decides graph capture from the driver; there is no config knob, so the
        # example flips the decision (documented in FINDINGS as a wanted upstream knob).
        mjlab_sim.Simulation._should_use_cuda_graph = lambda self: False  # type: ignore[method-assign]

    env_cfg = load_env_cfg(TASK)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(TASK)
    agent_cfg.max_iterations = max_iterations
    agent_cfg.seed = seed
    agent_cfg.save_interval = max_iterations  # only the final checkpoint
    agent_cfg.logger = "tensorboard"
    log_dir = log_root / f"n{num_envs}_{'graph' if use_graph else 'nograph'}"
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t_build = time.monotonic()
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=str(log_dir), device="cuda:0")
    build_s = time.monotonic() - t_build

    steps_per_it = agent_cfg.num_steps_per_env * num_envs
    history: list[dict] = []
    state: dict = {
        "first_hit_it": None,
        "first_hit_wall_s": None,
        "converged_it": None,
        "converged_wall_s": None,
        "t0": None,
    }
    logger = runner.logger
    original_log = logger.log

    def hooked_log(*, it: int, collect_time: float, learn_time: float, **kw) -> None:
        vals = [float(torch.as_tensor(e[METRIC]).float().mean()) for e in logger.ep_extras if METRIC in e]
        at_goal = sum(vals) / len(vals) if vals else float("nan")
        wall = time.monotonic() - state["t0"]
        history.append(
            {
                "it": it,
                "wall_s": round(wall, 2),
                "collect_s": round(collect_time, 3),
                "learn_s": round(learn_time, 3),
                "at_goal": round(at_goal, 4),
                "fps_collect": int(steps_per_it / collect_time),
                "fps_total": int(steps_per_it / (collect_time + learn_time)),
            }
        )
        original_log(it=it, collect_time=collect_time, learn_time=learn_time, **kw)
        if state["first_hit_it"] is None and at_goal >= target:
            state["first_hit_it"] = it
            state["first_hit_wall_s"] = round(wall, 2)
        # Converged = the mean of the last 5 logged values clears the target (one value is
        # noisy at small N: only the episodes that ended this iteration contribute).
        recent = [h["at_goal"] for h in history[-5:] if h["at_goal"] == h["at_goal"]]
        if len(recent) == 5 and sum(recent) / 5 >= target:
            state["converged_it"] = it
            state["converged_wall_s"] = round(wall, 2)
            raise _Converged

    logger.log = hooked_log  # type: ignore[method-assign]
    state["t0"] = time.monotonic()
    try:
        runner.learn(max_iterations, init_at_random_ep_len=True)
    except _Converged:
        runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
    train_s = time.monotonic() - state["t0"]
    mem = _gpu_mem_gb()
    graph_active = bool(getattr(env.sim, "use_cuda_graph", False))
    env.close()

    done_its = len(history)
    total_steps = done_its * steps_per_it
    tail = history[-min(10, done_its) :] if history else []
    rec = {
        "num_envs": num_envs,
        "cuda_graph": bool(use_graph),
        "cuda_graph_active": graph_active,
        "build_s": round(build_s, 1),
        "iterations": done_its,
        "env_steps": total_steps,
        "train_wall_s": round(train_s, 1),
        "env_steps_per_s_end_to_end": int(total_steps / train_s) if train_s else 0,
        "env_steps_per_s_collect_steady": int(sum(h["fps_collect"] for h in tail) / len(tail)) if tail else 0,
        "env_steps_per_s_total_steady": int(sum(h["fps_total"] for h in tail) / len(tail)) if tail else 0,
        "at_goal_final": history[-1]["at_goal"] if history else None,
        "at_goal_target": target,
        "first_hit_it": state["first_hit_it"],
        "first_hit_wall_s": state["first_hit_wall_s"],
        "converged_it": state["converged_it"],
        "converged_wall_s": state["converged_wall_s"],
        "seed": seed,
        "log_dir": str(log_dir),
        "history": history,
        **mem,
    }
    (log_dir / "result.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def plot(results: list[dict], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for graph, marker, label in ((True, "o", "CUDA graph"), (False, "s", "no graph")):
        rows = sorted(
            (r for r in results if r["cuda_graph"] == graph and "error" not in r), key=lambda r: r["num_envs"]
        )
        if not rows:
            continue
        n = [r["num_envs"] for r in rows]
        axes[0].plot(n, [r["env_steps_per_s_collect_steady"] for r in rows], marker=marker, label=f"collect, {label}")
        axes[0].plot(
            n,
            [r["env_steps_per_s_total_steady"] for r in rows],
            marker=marker,
            ls="--",
            label=f"collect+learn, {label}",
        )
        hit = [(r["num_envs"], r["converged_wall_s"]) for r in rows if r.get("converged_wall_s") is not None]
        if hit:
            axes[1].plot([h[0] for h in hit], [h[1] / 60 for h in hit], marker=marker, label=label)
        axes[2].plot(
            n,
            [r["torch_max_allocated_gb"] + r.get("warp_mempool_high_gb", 0.0) for r in rows],
            marker=marker,
            label=label,
        )
    for ax, title, ylabel in (
        (axes[0], "throughput", "env-steps/s"),
        (axes[1], f"wall clock to at_goal >= {results[0]['at_goal_target']} (5-iteration mean)", "minutes"),
        (axes[2], "peak GPU memory (torch + warp)", "GB"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("num_envs")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.suptitle("so101 reach on mjlab, Jetson AGX Thor: scale sweep")
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, nargs="*", default=[64, 256, 1024, 4096])
    p.add_argument("--graph", choices=("on", "off", "both"), default="both")
    p.add_argument("--target", type=float, default=0.6)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="scale_sweep.json")
    p.add_argument("--log-root", default="runs/scale_sweep")
    p.add_argument("--plot", help="existing results JSON to plot instead of training")
    p.add_argument("--png", default="examples/mjlab/assets/scale_sweep.png")
    a = p.parse_args(argv)

    if a.plot:
        plot(json.loads(Path(a.plot).read_text(encoding="utf-8")), Path(a.png))
        print(a.png)
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    out = Path(a.out)
    results: list[dict] = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else []
    graphs = {"on": (True,), "off": (False,), "both": (True, False)}[a.graph]
    for n in a.num_envs:
        for g in graphs:
            try:
                rec = run_one(
                    n, g, target=a.target, max_iterations=a.max_iterations, seed=a.seed, log_root=Path(a.log_root)
                )
            except Exception as exc:  # out of memory and friends are results too
                rec = {"num_envs": n, "cuda_graph": g, "error": repr(exc)[:400], "at_goal_target": a.target}
            results = [r for r in results if not (r["num_envs"] == n and r["cuda_graph"] == g)] + [rec]
            out.write_text(json.dumps(results, indent=1), encoding="utf-8")
            summary = {k: v for k, v in rec.items() if k != "history"}
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
