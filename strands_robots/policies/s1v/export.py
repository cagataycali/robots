"""Export S1V episode shards as a LeRobot v3 dataset and push the checkpoint.

    python -m strands_robots.policies.s1v.export dataset ROOT... --repo cagataydev/s1v-so101-mujoco-20260928 --out DIR [--push]
    python -m strands_robots.policies.s1v.export model CKPT_DIR --repo cagataydev/s1v-so101-v1 --card CARD.md [--push]

The dataset keeps the two cameras as video, the 6-float state (deg x5 +
gripper %) and the executed joint setpoint (rad x5 + gripper ctrl) as the
action, and puts everything the decider is trained on into extra features:
expert / executed primitive index, the expert's typed factors, the three noul
labels, the expert phase and the privileged tcp-to-cube distance. Every shard
is one episode; the task string is the LeRobot ``task`` column.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import read_manifest
from .primitives import SO101_PRIMITIVES

TASK_NAMES = ("reach", "pick")
FPS = 10


def lerobot_features() -> dict[str, dict[str, Any]]:
    """The LeRobot v3 feature spec for one S1V frame."""
    return {
        "observation.images.scene": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        },
        "s1v.expert_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "s1v.executed_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "s1v.expert_factors": {"dtype": "int32", "shape": (3,), "names": ["joint", "direction", "size"]},
        "s1v.labels": {"dtype": "float32", "shape": (3,), "names": ["cube_in_jaws", "progress_if_executed", "safe"]},
        "s1v.phase": {"dtype": "int32", "shape": (1,), "names": None},
        "s1v.tcp_cube_distance": {"dtype": "float32", "shape": (1,), "names": None},
        "s1v.round": {"dtype": "int32", "shape": (1,), "names": None},
    }


def iter_shards(roots: list[Path]):
    """Yield ``(root, manifest_row, path)`` for every episode shard, sorted by name."""
    for root in roots:
        rows = read_manifest(root)
        for row in sorted(rows, key=lambda r: r["name"]):
            yield root, row, root / "episodes" / f"{row['name']}.npz"


def export_dataset(
    roots: list[Path],
    repo_id: str,
    out: Path,
    *,
    push: bool = False,
    limit: int | None = None,
    image_writer_threads: int = 4,
) -> dict:
    """Write the LeRobot v3 dataset under ``out`` (fresh) and optionally push it private.

    ``image_writer_threads`` feeds LeRobot's async image writer; the single-threaded default took ~5 s per
    120-frame episode on Thor (measured on the r0+r1 export), dominated by PNG writes before AV1 encoding.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id,
        FPS,
        root=out,
        features=lerobot_features(),
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )
    n_ep = 0
    n_frames = 0
    per_task = {t: 0 for t in TASK_NAMES}
    t0 = time.time()
    for _root, row, path in iter_shards(roots):
        if limit is not None and n_ep >= limit:
            break
        d = np.load(path, allow_pickle=False)
        task = TASK_NAMES[int(d["task"])]
        t = int(d["state"].shape[0])
        for i in range(t):
            ds.add_frame(
                {
                    "observation.images.scene": d["scene"][i],
                    "observation.images.wrist": d["wrist"][i],
                    "observation.state": d["state"][i].astype(np.float32),
                    "action": d["setpoint_qpos"][i].astype(np.float32),
                    "s1v.expert_idx": np.array([d["expert_idx"][i]], dtype=np.int32),
                    "s1v.executed_idx": np.array([d["executed_idx"][i]], dtype=np.int32),
                    "s1v.expert_factors": d["expert_factors"][i].astype(np.int32),
                    "s1v.labels": d["labels"][i].astype(np.float32),
                    "s1v.phase": np.array([d["phase"][i]], dtype=np.int32),
                    "s1v.tcp_cube_distance": np.array([d["tcp_cube_distance"][i]], dtype=np.float32),
                    "s1v.round": np.array([int(d["round"])], dtype=np.int32),
                    "task": task,
                }
            )
        ds.save_episode()
        n_ep += 1
        n_frames += t
        per_task[task] += 1
        if n_ep % 50 == 0:
            print(f"[export] {n_ep} episodes {n_frames} frames {time.time() - t0:.0f}s", flush=True)
    ds.finalize()
    primitives = [p.__dict__ if hasattr(p, "__dict__") else str(p) for p in SO101_PRIMITIVES]
    (out / "s1v_primitives.json").write_text(json.dumps(primitives, indent=1, default=str), encoding="utf-8")
    stats = {"episodes": n_ep, "frames": n_frames, "per_task": per_task, "seconds": time.time() - t0, "root": str(out)}
    if push:
        ds.push_to_hub(private=True, push_videos=True, upload_large_folder=True)
        stats["pushed"] = repo_id
    return stats


def push_model(ckpt: Path, repo_id: str, *, card: Path | None = None, push: bool = False) -> dict:
    """Upload a checkpoint directory (config.json, model.pt, metrics.json, README.md) as a private model repo."""
    from huggingface_hub import HfApi

    ckpt = Path(ckpt)
    if card is not None:
        shutil.copyfile(card, ckpt / "README.md")
    files = sorted(p.name for p in ckpt.iterdir() if p.is_file())
    if not push:
        return {"repo": repo_id, "files": files, "pushed": False}
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=repo_id, repo_type="model")
    return {"repo": repo_id, "files": files, "pushed": True}


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``dataset`` or ``model``."""
    import argparse

    ap = argparse.ArgumentParser(description="S1V export: LeRobot v3 dataset and model checkpoint to the Hub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("roots", nargs="+", type=Path)
    d.add_argument("--repo", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--push", action="store_true")
    d.add_argument("--image-writer-threads", type=int, default=4)
    m = sub.add_parser("model")
    m.add_argument("ckpt", type=Path)
    m.add_argument("--repo", required=True)
    m.add_argument("--card", type=Path, default=None)
    m.add_argument("--push", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "dataset":
        print(
            json.dumps(
                export_dataset(
                    args.roots,
                    args.repo,
                    args.out,
                    push=args.push,
                    limit=args.limit,
                    image_writer_threads=args.image_writer_threads,
                )
            )
        )
    else:
        print(json.dumps(push_model(args.ckpt, args.repo, card=args.card, push=args.push)))


if __name__ == "__main__":
    main()
