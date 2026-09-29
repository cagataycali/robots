"""An agent trains a fleet: three arms, one prompt, one leaderboard.

The agent gets three tools and a prompt. ``train_policy`` (the stock strands
tool, provider ``rsl_rl`` -> mjlab PPO -> ONNX) trains a reach policy per arm;
``evaluate_policy`` plays an ONNX actor on the classic CPU MuJoCo backend for 20
seeded targets and returns success + median error; ``write_leaderboard`` saves
the markdown the agent writes. Nothing in the prompt names a file: the agent has
to carry ``exported_model`` from tool 1 into tool 2 and the numbers from tool 2
into tool 3.

Reach tasks for arms other than so101 do not exist in the trainer's registry, so
this example registers them on the fly with the recipe from
``02_every_arm_one_night.py`` (registry MJCF -> reachable cloud by FK -> mjlab
task, rewards scaled by the arm's reach) under ``Strands-Reach-<arm>``; the
trainer sees whatever the mjlab registry knows. The transcript (prompt, every
tool call with arguments and results, final answer) goes to ``--transcript``.

The agent runs on Bedrock through the usual environment (``AWS_BEARER_TOKEN_BEDROCK``).
``--no-llm`` runs the same three tools in the obvious order without a model and
says so in the transcript header; it exists so the pipeline can be verified
where no credentials are present.

Usage::

    python examples/mjlab/06_agent_trains_a_fleet.py --arms so101,koch,arx_l5 --iterations 200 --num-envs 1024 \
        --output-dir runs/fleet --transcript examples/mjlab/assets/transcript_fleet.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPT = """You run a robot training fleet in a MuJoCo-Warp simulator.
Train a reach policy for each of these arms: {arms}. For every arm call train_policy with
provider "rsl_rl", extra {{"task": "<task id from this table>"}}, steps={its}, batch_size={envs},
save_freq={its}, seed=0, embodiment "<arm>", output_dir "{out}/<arm>".
Task ids: {task_table}.
Then evaluate each exported_model with evaluate_policy on 20 episodes on the CPU backend.
Finally write a leaderboard markdown with write_leaderboard: one row per arm with DoF, training wall
time in minutes, the training run's final Metrics/reach/at_goal if the train result reports it,
evaluation success (x/20) and median final error in mm, sorted by success. Use only numbers the tools
returned; if a tool failed for an arm, keep the row and put the error in the status column."""


def trim_onnx_metadata_to_actuated(onnx_path: str, actuated_joints: list[str]) -> list[str]:
    """Rewrite ``joint_names`` / ``default_joint_pos`` in an mjlab ONNX file to the actuated joints.

    mjlab's ``get_base_metadata`` lists every joint of the entity while the actor has
    one output per ``joint_pos`` action target, so an arm with an unactuated joint
    (arx_l5: 8 joints, 7 actuators) exports 7 outputs against 8 names and the
    ``rsl_rl_onnx`` provider refuses the file (FINDINGS F15; needs upstream change in
    ``strands_robots/training/mjlab_tasks/export.py``). Filters in mjlab's natural
    joint order, which is the order the joint observation terms and the action term use.
    Returns the names kept.
    """
    import onnx

    model = onnx.load(onnx_path)
    props = {e.key: e.value for e in model.metadata_props}
    names = [n for n in props.get("joint_names", "").split(",") if n]
    keep = [i for i, n in enumerate(names) if n in set(actuated_joints)]
    if len(keep) == len(names):
        return names
    defaults = [d for d in props.get("default_joint_pos", "").split(",") if d]
    new = {"joint_names": ",".join(names[i] for i in keep)}
    if len(defaults) == len(names):
        new["default_joint_pos"] = ",".join(defaults[i] for i in keep)
    for entry in model.metadata_props:
        if entry.key in new:
            entry.value = new[entry.key]
    onnx.save(model, onnx_path)
    return [names[i] for i in keep]


def register_arm_tasks(arms: list[str], seed: int) -> dict[str, dict]:
    """Register ``Strands-Reach-<arm>`` for every arm but so101 (which the trainer already knows)."""
    from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task

    from strands_robots.training import mjlab_tasks

    every_arm = __import__("02_every_arm_one_night")
    mjlab_tasks.register_all()
    table: dict[str, dict] = {}
    for arm in arms:
        if arm == "so101":
            table[arm] = {"task": "Strands-Reach-SO101", "dof": 6, "ee": "site:gripper", "reach_m": None, "scale": 1.0}
            continue
        info = every_arm.inspect_arm(arm)
        fk = every_arm.ArmFK(info)
        cloud = fk.sample_reachable(4000, seed)
        reach_m = round(float((cloud**2).sum(1).max() ** 0.5), 3)
        scale = every_arm.reward_scale(reach_m)
        tid = f"Strands-Reach-{arm}"
        if tid not in list_tasks():
            register_mjlab_task(
                task_id=tid,
                env_cfg=every_arm.build_task(info, cloud, scale=scale),
                play_env_cfg=every_arm.build_task(info, cloud, play=True, scale=scale),
                rl_cfg=load_rl_cfg("Strands-Reach-SO101"),
            )
        table[arm] = {
            "task": tid,
            "dof": len(info.actuated_joints),
            "ee": f"{info.ee_kind}:{info.ee_name}",
            "reach_m": reach_m,
            "scale": round(scale, 3),
            "_info": info,
            "_fk": fk,
            "_cloud": cloud,
        }
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="so101,koch,arx_l5")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--transcript", required=True, help="markdown transcript path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-llm", action="store_true", help="run the three tools in order without a model")
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from strands.tools.decorator import tool

    from strands_robots.tools.train_policy import train_policy as core_train_policy

    every_arm = __import__("02_every_arm_one_night")
    table = register_arm_tasks(arms, a.seed)
    calls: list[dict[str, Any]] = []

    @tool
    def train_policy(
        provider: str,
        extra: dict[str, Any],
        steps: int,
        batch_size: int,
        save_freq: int,
        seed: int,
        embodiment: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """Train a policy with the stock strands-robots trainer and export its ONNX actor.

        Args:
            provider: Training provider, "rsl_rl" for the MuJoCo-Warp PPO trainer.
            extra: Provider extras; ``{"task": "<mjlab task id>"}`` picks the reach task.
            steps: PPO iterations.
            batch_size: Number of parallel simulated worlds.
            save_freq: Checkpoint cadence in iterations.
            seed: Random seed.
            embodiment: Registry name of the arm (e.g. "so101").
            output_dir: Where the run directory and the exported ONNX go.
        """
        tr = core_train_policy(
            provider=provider,
            extra=extra,
            steps=steps,
            batch_size=batch_size,
            save_freq=save_freq,
            seed=seed,
            embodiment=embodiment,
            output_dir=output_dir,
        )
        payload = next((c["json"] for c in tr.get("content", []) if isinstance(c, dict) and "json" in c), {})
        onnx = payload.get("exported_model")
        row = table.get(embodiment) or {}
        if onnx and row.get("_info") is not None:
            kept = trim_onnx_metadata_to_actuated(onnx, list(row["_info"].actuated_joints))
            calls.append({"tool": "trim_onnx_metadata", "args": {"onnx": onnx}, "result": {"joint_names": kept}})
        return tr

    @tool
    def evaluate_policy(onnx_path: str, robot: str, n_episodes: int = 20) -> dict[str, Any]:
        """Play a trained rsl_rl ONNX reach actor on the classic CPU MuJoCo backend for seeded targets.

        Args:
            onnx_path: The ``exported_model`` path returned by ``train_policy``.
            robot: Registry name of the arm the policy was trained on (e.g. "so101").
            n_episodes: Seeded targets to try (one episode each, 150 ticks at 50 Hz).
        """
        t0 = time.monotonic()
        try:
            row = table[robot]
            if robot == "so101":
                from sim2sim_reach import rollout, sample_targets

                from strands_robots import Robot
                from strands_robots.policies import create_policy
                from strands_robots.policies.rsl_rl_onnx.policy import _SiteFK
                from strands_robots.training.mjlab_tasks.so101_reach import SUCCESS_M

                fk = _SiteFK("so101", "gripper", ["1", "2", "3", "4", "5", "6"])
                policy = create_policy("rsl_rl_onnx", onnx_path=onnx_path, robot="so101")
                sim = Robot("so101", backend="mujoco")

                async def run():
                    return [
                        await rollout(sim, policy, fk, t, 150, 50.0) for t in sample_targets(n_episodes, a.seed + 1)
                    ]

                eps = asyncio.run(run())
                sim.cleanup()
                succ = sum(e["success"] for e in eps)
                res = {
                    "backend": "mujoco",
                    "n": n_episodes,
                    "success_m": SUCCESS_M,
                    "success": f"{succ}/{n_episodes}",
                    "success_rate": succ / n_episodes,
                    "final_err_median_m": float(np.median([e["final_err_m"] for e in eps])),
                }
            else:
                cloud = row["_cloud"]
                targets = cloud[np.random.default_rng(a.seed + 1).choice(len(cloud), n_episodes, replace=False)]
                res = asyncio.run(
                    every_arm.sim2sim(row["_info"], onnx_path, targets, success_m=every_arm.SUCCESS_M * row["scale"])
                )
                res.pop("episodes", None)
            res["wall_s"] = round(time.monotonic() - t0, 1)
        except Exception as exc:  # the agent must see the failure, not a crash
            res = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "wall_s": round(time.monotonic() - t0, 1)}
        calls.append(
            {
                "tool": "evaluate_policy",
                "args": {"onnx_path": onnx_path, "robot": robot, "n_episodes": n_episodes},
                "result": res,
            }
        )
        return res

    @tool
    def write_leaderboard(markdown: str) -> dict[str, Any]:
        """Save the leaderboard markdown table.

        Args:
            markdown: The full markdown text (heading + table).
        """
        path = out / "leaderboard.md"
        path.write_text(markdown, encoding="utf-8")
        calls.append({"tool": "write_leaderboard", "args": {"chars": len(markdown)}, "result": {"path": str(path)}})
        return {"path": str(path), "chars": len(markdown)}

    task_table = ", ".join(f"{arm}: {row['task']}" for arm, row in table.items())
    prompt = PROMPT.format(arms=", ".join(arms), its=a.iterations, envs=a.num_envs, out=str(out), task_table=task_table)
    t0 = time.monotonic()
    tool_uses: list[dict] = []
    if a.no_llm:
        header = "No model: the three tools were called in the obvious order by the script (--no-llm)."
        rows = []
        for arm, row in table.items():
            tr = train_policy(
                provider="rsl_rl",
                extra={"task": row["task"]},
                steps=a.iterations,
                batch_size=a.num_envs,
                save_freq=a.iterations,
                seed=0,
                embodiment=arm,
                output_dir=str(out / arm),
            )
            tool_uses.append({"name": "train_policy", "input": {"task": row["task"], "embodiment": arm}, "result": tr})
            # The stock tool returns a ToolResult: {"status", "content": [{"text"}, {"json": {...}}]}.
            payload = next((c["json"] for c in tr.get("content", []) if isinstance(c, dict) and "json" in c), {})
            tr = {**payload, **(payload.get("metrics") or {})}
            onnx = tr.get("exported_model")
            ev = evaluate_policy(onnx, arm) if onnx else {"error": "no exported_model"}
            tool_uses.append({"name": "evaluate_policy", "input": {"onnx_path": onnx, "robot": arm}, "result": ev})
            rows.append((arm, row["dof"], tr, ev))
        rows.sort(key=lambda r: -(r[3].get("success_rate") or 0))
        md = "# Fleet reach leaderboard\n\n| arm | DoF | train min | eval success | median err mm | status |\n|---|---|---|---|---|---|\n"
        for arm, dof, tr, ev in rows:
            mins = round((tr.get("wall_s") or 0) / 60, 1)
            err = round(ev["final_err_median_m"] * 1000) if "final_err_median_m" in ev else ""
            md += f"| {arm} | {dof} | {mins} | {ev.get('success', '')} | {err} | {ev.get('error', 'ok')} |\n"
        write_leaderboard(md)
        final = md
        model_desc = "none (--no-llm)"
    else:
        from strands import Agent
        from strands.tools.executors import SequentialToolExecutor

        # The default executor runs the model's tool calls concurrently. Two mjlab trainers in one
        # process both start a CUDA graph capture on the default stream and the second one dies with
        # "Graph capture already in progress on this stream", so this agent trains one arm at a time.
        agent = Agent(tools=[train_policy, evaluate_policy, write_leaderboard], tool_executor=SequentialToolExecutor())
        result = agent(prompt)
        final = str(result)
        model_desc = str(getattr(getattr(agent, "model", None), "config", None))
        for m in agent.messages:
            for c in m.get("content", []):
                if "toolUse" in c:
                    tool_uses.append(
                        {"name": c["toolUse"]["name"], "input": c["toolUse"]["input"], "id": c["toolUse"]["toolUseId"]}
                    )
                if "toolResult" in c:
                    tool_uses.append(
                        {"result_for": c["toolResult"]["toolUseId"], "content": c["toolResult"]["content"]}
                    )
        header = f"Model: {model_desc}. Every tool call below is copied from the agent's message history."
    wall = time.monotonic() - t0

    def short(x: Any, n: int = 1500) -> str:
        s = json.dumps(x, indent=1, default=str)
        return s if len(s) <= n else s[:n] + f"\n... ({len(s) - n} more chars)"

    lines = [
        "# Transcript: an agent trains a fleet",
        "",
        f"Generated by `examples/mjlab/06_agent_trains_a_fleet.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
        f"arms {', '.join(arms)}; {a.iterations} iterations at {a.num_envs} worlds each; wall {wall / 60:.1f} min.",
        "",
        header,
        "",
        "## Prompt",
        "",
        "```",
        prompt,
        "```",
        "",
        "## Tool calls",
        "",
    ]
    for i, tu in enumerate(tool_uses, 1):
        if "name" in tu:
            lines += [f"### {i}. `{tu['name']}`", "", "```json", short(tu["input"]), "```", ""]
            if "result" in tu:
                lines += ["result:", "", "```json", short(tu["result"]), "```", ""]
        else:
            lines += [f"result for `{tu['result_for']}`:", "", "```json", short(tu["content"]), "```", ""]
    lines += ["## Final answer", "", final, ""]
    lb = out / "leaderboard.md"
    if lb.exists():
        lines += ["## leaderboard.md as written by the agent", "", lb.read_text(encoding="utf-8"), ""]
    Path(a.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(a.transcript).write_text("\n".join(lines), encoding="utf-8")
    (out / "transcript.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "tool_uses": tool_uses,
                "final": final,
                "wall_s": round(wall, 1),
                "model": model_desc,
                "tasks": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in table.items()},
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wall_min": round(wall / 60, 1),
                "tool_calls": len([t for t in tool_uses if "name" in t]),
                "transcript": a.transcript,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
