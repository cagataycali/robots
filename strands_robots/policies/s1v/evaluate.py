"""Closed-loop evaluation of S1V arms against the scripted expert and a random actor.

Protocol (shared with the Laya lane so the tables line up): ``N`` episodes per task
on fixed seeds ``seed0 .. seed0+N-1``; every arm sees the same block poses, light
and camera jitter. Metrics per arm and task: success rate, mean final distance,
mean steps to success (successes only), mean block lift (pick), ticks per
episode, tick latency p50/p95, and for learned arms the gate analysis from the
recorded ``progress_if_executed`` / ``safe`` answers against the expert's labels.

Usage::

    python -m strands_robots.policies.s1v.evaluate OUT.json \
        --arm s1v=PATH_TO_CKPT --arm vision-only=PATH2 --arm no-temp=PATH:notemp \
        --arm scripted --arm random --n 20 --seed 5000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .expert import TASKS, run_episode
from .primitives import SO101_PRIMITIVES, Primitive
from .scene import So101Scene


def random_actor(seed: int):
    """Uniform random primitive per tick (the floor every learned arm must clear)."""
    rng = random.Random(seed)

    def act(_obs: dict[str, Any], _priv: Any, _expert: Primitive) -> Primitive:
        return rng.choice(SO101_PRIMITIVES)

    return act


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of ``run_episode`` results into the report numbers."""
    succ = [r for r in results if r["success"]]
    return {
        "n": len(results),
        "success_rate": len(succ) / max(1, len(results)),
        "successes": len(succ),
        "mean_final_distance_cm": 100.0 * float(np.mean([r["final_distance"] for r in results])),
        "mean_steps_to_success": float(np.mean([r["steps_to_success"] for r in succ])) if succ else None,
        "mean_cube_lift_cm": 100.0 * float(np.mean([r["cube_lift"] for r in results])),
        "mean_ticks": float(np.mean([r["ticks"] for r in results])),
    }


def gate_analysis(decisions: list[dict[str, Any]], expert_labels: list[dict[str, float]]) -> dict[str, Any]:
    """How a hold-threshold on the noul answers would have sorted good and bad steps.

    A step is *bad* when the expert's post-hoc label says it did not progress or
    was unsafe. For thresholds on ``progress_if_executed`` and ``safe`` report how
    many bad steps would have been held (true positives) and how many good steps
    wrongly held (false positives), plus Brier score of each answer against its
    label and the AUC.
    """
    if not decisions:
        return {}
    p_prog = np.array([d["progress_if_executed"] for d in decisions])
    p_safe = np.array([d["safe"] for d in decisions])
    p_jaws = np.array([d["cube_in_jaws"] for d in decisions])
    y_prog = np.array([lab["progress_if_executed"] for lab in expert_labels])
    y_safe = np.array([lab["safe"] for lab in expert_labels])
    y_jaws = np.array([lab["cube_in_jaws"] for lab in expert_labels])
    out: dict[str, Any] = {
        "ticks": int(len(decisions)),
        "brier": {
            "progress_if_executed": float(np.mean((p_prog - y_prog) ** 2)),
            "safe": float(np.mean((p_safe - y_safe) ** 2)),
            "cube_in_jaws": float(np.mean((p_jaws - y_jaws) ** 2)),
        },
        "auc": {
            "progress_if_executed": _auc(p_prog, y_prog),
            "safe": _auc(p_safe, y_safe),
            "cube_in_jaws": _auc(p_jaws, y_jaws),
        },
        "base_rate": {
            "progress_if_executed": float(y_prog.mean()),
            "safe": float(y_safe.mean()),
            "cube_in_jaws": float(y_jaws.mean()),
        },
        "thresholds": [],
    }
    bad = (y_prog < 0.5) | (y_safe < 0.5)
    for tau in (0.3, 0.5, 0.7, 0.9):
        hold = (p_prog < tau) | (p_safe < tau)
        out["thresholds"].append(
            {
                "tau": tau,
                "held": int(hold.sum()),
                "bad_steps": int(bad.sum()),
                "bad_held": int((hold & bad).sum()),
                "good_wrongly_held": int((hold & ~bad).sum()),
                "recall_bad": float((hold & bad).sum() / max(1, bad.sum())),
                "precision_hold": float((hold & bad).sum() / max(1, hold.sum())),
            }
        )
    return out


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos = p[y >= 0.5]
    neg = p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_arm(
    name: str,
    spec: str | None,
    *,
    n: int,
    seed0: int,
    tasks: tuple[str, ...] = TASKS,
    max_ticks: int = 120,
    device: str = "cuda",
    gate: float | None = None,
) -> dict[str, Any]:
    """Roll ``n`` episodes per task for one arm; returns per-task summaries and details."""
    from strands_robots import Robot

    brain = None
    if spec not in (None, "scripted", "random"):
        from .policy import S1VBrain

        path, _, flag = spec.partition(":")
        brain = S1VBrain(
            path, device=device, temperature_scaling=flag != "notemp", confidence_gate=gate, cuda_graph=True
        )
    robot = Robot("so101", mode="sim")
    report: dict[str, Any] = {"arm": name, "spec": spec, "gate": gate, "tasks": {}}
    try:
        scene = So101Scene(robot, seed=seed0)
        for task in tasks:
            results = []
            decisions: list[dict[str, Any]] = []
            labels: list[dict[str, float]] = []
            for i in range(n):
                scene.rng = np.random.default_rng(seed0 + i)
                if spec == "random":
                    actor = random_actor(seed0 + i)
                elif brain is None:
                    actor = None
                else:
                    actor = brain.actor(task)
                    start = len(brain.decisions)
                r = run_episode(scene, task, actor=actor, max_ticks=max_ticks, keep_images=False)
                if brain is not None:
                    decisions.extend(brain.decisions[start:])
                    labels.extend(rec["labels"] for rec in r["records"])
                r.pop("records")
                r["seed"] = seed0 + i
                results.append(r)
                print(
                    f"[eval] {name:12s} {task:5s} ep{i:02d} seed={seed0 + i} success={r['success']} "
                    f"ticks={r['ticks']} dist={100 * r['final_distance']:.1f}cm lift={100 * r['cube_lift']:.1f}cm",
                    flush=True,
                )
            summary = summarize(results)
            if brain is not None:
                summary["gate_analysis"] = gate_analysis(decisions, labels)
                summary["gated_ticks"] = int(sum(d["gated"] for d in decisions))
            report["tasks"][task] = {"summary": summary, "episodes": results}
        if brain is not None:
            ms = np.array(brain.tick_ms[5:] or brain.tick_ms)
            report["latency_ms"] = {
                "p50": float(np.percentile(ms, 50)),
                "p95": float(np.percentile(ms, 95)),
                "n": int(len(ms)),
            }
    finally:
        robot.destroy()
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry, see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", action="append", default=[], help="name[=spec]; spec = scripted | random | CKPT[:notemp]")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-ticks", type=int, default=120)
    ap.add_argument("--gate", type=float, default=None, help="confidence gate for learned arms (None = off)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    tasks = tuple(t for t in args.tasks.split(",") if t)
    reports = []
    t0 = time.time()
    for item in args.arm:
        name, _, spec = item.partition("=")
        spec = spec or name
        reports.append(
            run_arm(
                name,
                spec,
                n=args.n,
                seed0=args.seed,
                tasks=tasks,
                max_ticks=args.max_ticks,
                device=args.device,
                gate=args.gate,
            )
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"n": args.n, "seed": args.seed, "arms": reports, "seconds": time.time() - t0}, indent=1)
        )
    for rep in reports:
        for task, block in rep["tasks"].items():
            s = block["summary"]
            print(
                f"[eval] {rep['arm']:12s} {task:5s} success={s['success_rate']:.2f} ({s['successes']}/{s['n']}) "
                f"dist={s['mean_final_distance_cm']:.1f}cm steps={s['mean_steps_to_success']} lift={s['mean_cube_lift_cm']:.1f}cm"
            )


if __name__ == "__main__":
    main()
