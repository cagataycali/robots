"""Push the Laya System-1 experiment artifacts to ONE private HF dataset repo.

Layout (one repo, research artifact, not a single LeRobot dataset):
  README.md                      dataset card (generated here)
  arms/<arm>_<task>/             LeRobot v3 dataset for that arm x task (meta/data/videos)
  results/<arm>_<task>.jsonl     per-episode metrics + per-tick Laya probabilities
  judge/                         H2-proper judge inputs/outputs when present
  REPORT.md, FINDINGS.md         copied from the lane dir when present

Idempotent: upload_folder only sends changed files. Re-run after each arm finishes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from huggingface_hub import HfApi

REPO = "cagataydev/laya-so101-mujoco-20260928"
LANE = pathlib.Path(__file__).resolve().parent.parent
EXP = LANE / "exp"

CARD = """---
license: apache-2.0
pretty_name: Laya System-1 on so101 (MuJoCo) - research artifact
tags: [robotics, lerobot, mujoco, so101, strands-robots, laya, research]
---
# Laya System-1 x strands-robots, so101 in MuJoCo (2026-09-28, private)

Research artifact for the LANE LAYA-SYSTEM1 experiment: can a text-only,
non-autoregressive typed-decision model (convaiinnovations/laya) act as a
System-1 controller for a MuJoCo so101 arm when fed a serialized robot state?

**No hardware. Sim only. Every arm sees the same seeds, cube positions and
primitive vocabulary.** 200 ticks per episode, 20 episodes per arm x task.

## Arms
| arm | what it is |
|---|---|
| scripted | greedy forward-kinematics primitive picker (upper reference; never grasps) |
| random | uniform random primitive (lower reference) |
| english | Laya checkpoint `english`, zero-shot, joint/direction/size question set |
| multilingual | Laya checkpoint `multilingual`, same |
| typed-decisions | Laya checkpoint `typed-decisions`, same |

## Tasks
* `reach`: end effector within 3 cm of the cube.
* `pick`: cube centre lifted >= 5 cm above the table with finger contact.

## Headline (H1, 20 episodes per cell, same seeds)
scripted reach 20/20 (median 45 ticks); random 0/20 (min distance 0.081 m); every zero-shot Laya arm 0/20 on both
tasks and none beats random on min distance (0.102-0.116 m = rest pose). Each Laya checkpoint emits one primitive per
(checkpoint, task text): english `none` 100 %, multilingual `gripper` 100 % (reach) / `shoulder_lift` 88 % (pick),
typed-decisions `elbow_flex` 84 % (reach) / `shoulder_lift` 99 % (pick). Nobody lifts the cube, the scripted grasp
closes on one finger only (pick rows are reach-and-descend references). Details and H2/H3 in `REPORT.md`.

## Layout
* `arms/<arm>_<task>/` LeRobot v3 datasets (`observation.images.scene`, `observation.images.wrist`,
  `observation.state`, `action`, timestamps). Load with `LeRobotDataset(REPO, root=..., ...)` after
  downloading the sub-folder, or `snapshot_download(REPO, allow_patterns="arms/english_reach/*")`.
* `results/<arm>_<task>.jsonl` one line per episode: success, steps, min/final distance, hold and
  gripper fractions, per-tick primitive wanted/applied and every Laya probability.
* `judge/` H2-proper inputs: baseline trajectories re-serialized to Laya's state text plus each
  checkpoint's `progress_ok` probability vs realized progress.
* `REPORT.md` H1/H2/H3 numbers, `FINDINGS.md` integration findings.

Produced with strands-robots branch `feat/laya-system1-policy` (fork cagataycali/robots),
provider `strands_robots/policies/laya/`. Owner: @cagataycali.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(REPO, repo_type="dataset")
    assert info.private, "repo must be private"

    (EXP / "README_hf.md").write_text(CARD)
    uploads: list[tuple[pathlib.Path, str]] = []
    for d in sorted((EXP / "lerobot").iterdir()):
        # complete datasets only: meta/info.json present with total_episodes >= 1
        info_json = d / "meta" / "info.json"
        if info_json.exists():
            import json

            n = json.loads(info_json.read_text()).get("total_episodes", 0)
            if n >= 20:
                uploads.append((d, f"arms/{d.name}"))
            else:
                print("skip (in progress,", n, "episodes)", d.name)
    uploads.append((EXP / "results", "results"))
    if (EXP / "judge").exists():
        uploads.append((EXP / "judge", "judge"))
    for f in uploads:
        print("folder", f[0].name, "->", f[1])
    if args.dry:
        return
    api.upload_file(
        path_or_fileobj=str(EXP / "README_hf.md"),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
        commit_message="dataset card",
    )
    for src, dst in uploads:
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=dst,
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"upload {dst}",
            ignore_patterns=["*.pyc", "__pycache__/*", "*.log"],
        )
    for name in ("REPORT.md", "FINDINGS.md"):
        p = LANE / name
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=name,
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"upload {name}",
            )
    print("pushed", REPO, "private=", api.repo_info(REPO, repo_type="dataset").private)


if __name__ == "__main__":
    sys.exit(main())
