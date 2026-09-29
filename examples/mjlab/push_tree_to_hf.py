"""Push a whole results tree (one folder per arm) to ONE private Hub repo.

Usage: push_tree_to_hf.py <results_dir> <repo_id> "<title>"

Each arm folder holds ``<arm>_reach.onnx``, ``result.json`` and ``train/`` (rsl_rl checkpoints).
The card table copies the ``result.json`` rows verbatim; tfevents files are skipped.
"""

from __future__ import annotations

import json
import pathlib
import sys

from huggingface_hub import HfApi


def _row(d: pathlib.Path) -> str:
    rj = d / "result.json"
    if not rj.exists():
        return f"| {d.name} | (no result.json) | | |"
    r = json.loads(rj.read_text())
    ev = r.get("sim2sim") or {}
    tr = r.get("train") or {}
    ag = tr.get("at_goal_last10")
    ag_s = f"{ag:.2f}" if isinstance(ag, (int, float)) else str(r.get("stage") or "")
    err = ev.get("final_err_median_m")
    err_s = f"{err * 1000:.0f}" if isinstance(err, (int, float)) else ""
    return f"| {d.name} | {ag_s} | {ev.get('success', '')} | {err_s} |"


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1])
    repo = argv[2]
    title = argv[3]
    rows = [_row(d) for d in sorted(p for p in root.iterdir() if p.is_dir())]
    card = (
        "---\nlicense: apache-2.0\ntags: [mjlab, rsl_rl, strands-robots, reach]\n---\n"
        f"# {title}\n\n"
        "Research artefact of the strands-robots `examples/mjlab` lane (2026-09-29), sim only.\n"
        "One folder per arm: `<arm>_reach.onnx` (exported actor), `result.json` (train + classic MuJoCo "
        "sim-to-sim eval), `train/` (rsl_rl checkpoints).\n"
        "Numbers below are copied from each `result.json`; the README of the example holds the interpretation.\n\n"
        "| arm | at_goal (train) | classic MuJoCo success | median err mm |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )
    (root / "README.md").write_text(card)
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo,
        commit_message=f"{title} results tree",
        ignore_patterns=["*.tfevents*"],
    )
    info = api.repo_info(repo)
    print(repo, info.sha[:8], len(info.siblings), "private" if info.private else "PUBLIC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
