"""Push an mjlab/rsl_rl run (ONNX + checkpoints + train log + eval JSON) to a private HF model repo.

Usage: python examples/mjlab/push_run_to_hf.py <run_dir> <onnx> <repo_id> --task Strands-Reach-SO101
       [--eval s2s.json ...] [--log train.log] [--note "..."]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

PROVENANCE = (
    "Trained with [mjlab](https://github.com/mujocolab/mjlab) on strands-robots' own MJCF, exported to ONNX "
    "with mjlab metadata, and played back through `strands_robots.policies.rsl_rl_onnx` on both the "
    "`mjlab` (MuJoCo-Warp) and `mujoco` (classic) engines."
)


def card(task: str, onnx: Path, evals: list[Path], note: str) -> str:
    import onnx as onnx_lib

    md = {p.key: p.value for p in onnx_lib.load(str(onnx)).metadata_props}
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: rsl_rl",
        "tags: [strands-robots, mjlab, rsl_rl, onnx, mujoco-warp]",
        "---",
        f"# {task} (mjlab + rsl_rl -> ONNX)",
        "",
        PROVENANCE,
        "",
        note,
        "",
        "## ONNX metadata",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for k in (
        "observation_names",
        "joint_names",
        "default_joint_pos",
        "action_scale",
        "joint_stiffness",
        "joint_damping",
        "command_names",
    ):
        if k in md:
            lines.append(f"| `{k}` | `{md[k][:120]}` |")
    for ev in evals:
        d = json.loads(ev.read_text())
        lines += [
            "",
            f"## Sim-to-sim eval `{ev.name}` (n={d['n']}, seed={d['seed']}, {d['ticks']} ticks @ {d['hz']} Hz)",
            "",
            "| backend | policy | success | final err median (m) | min err median (m) | ticks/s |",
            "|---|---|---|---|---|---|",
        ]
        for r in d["runs"]:
            lines.append(
                f"| {r['backend']} | {r['policy']} | {r['success']} | {r['final_err_median_m']:.3f} | {r['min_err_median_m']:.3f} | {r['ticks_per_s']} |"
            )
    lines += [
        "",
        "## Use",
        "",
        "```python",
        "from strands_robots import Robot",
        "from strands_robots.tools.run_policy import run_policy",
        'sim = Robot("so101", backend="mjlab")  # or backend="mujoco"',
        f'run_policy(sim, robot_name="so101", policy_provider="rsl_rl_onnx", policy_config={{"onnx_path": "hf://<repo>/{onnx.name}"}}, n_episodes=5)',
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("onnx")
    p.add_argument("repo_id")
    p.add_argument("--task", required=True)
    p.add_argument("--eval", nargs="*", default=[])
    p.add_argument("--log")
    p.add_argument("--note", default="")
    a = p.parse_args()
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=True, exist_ok=True)
    onnx = Path(a.onnx)
    api.upload_file(path_or_fileobj=str(onnx), path_in_repo=onnx.name, repo_id=a.repo_id)
    api.upload_folder(
        folder_path=a.run_dir,
        path_in_repo="run",
        repo_id=a.repo_id,
        allow_patterns=["*.pt", "*.onnx", "params/*", "*.yaml", "*.json"],
    )
    for ev in a.eval:
        api.upload_file(path_or_fileobj=ev, path_in_repo=f"eval/{Path(ev).name}", repo_id=a.repo_id)
    if a.log:
        api.upload_file(path_or_fileobj=a.log, path_in_repo="train.log", repo_id=a.repo_id)
    api.upload_file(
        path_or_fileobj=card(a.task, onnx, [Path(e) for e in a.eval], a.note).encode(),
        path_in_repo="README.md",
        repo_id=a.repo_id,
    )
    print(f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
