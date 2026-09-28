"""Episode shards and frozen-backbone feature cache for the S1V decider.

Layout under ``root``::

    episodes/ep_<worker>_<n>.npz   one episode: images, proprio, expert labels, noul labels
    features/ep_<worker>_<n>.npz   frozen DINOv2 tokens for the same ticks (float16)
    manifest.jsonl                 one line per episode (task, success, ticks, round, actor)

The shard writer is what ``run_episode`` streams into; the feature pass runs
once per shard so training never touches the backbone. Keeping the raw images
is what makes an optional round-2 unfreeze possible later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .expert import run_episode
from .primitives import SO101_PRIMITIVES, Primitive, factor_primitive, primitive_index
from .scene import So101Scene

TASKS = ("reach", "pick")
PHASES = ("open", "approach", "fly", "descend", "grasp", "lift", "done")
NOUL_KEYS = ("cube_in_jaws", "progress_if_executed", "safe")
BACKBONE = "facebook/dinov2-small"
GRID = 4


def encode_episode(result: dict[str, Any], *, round_id: int, actor: str) -> dict[str, Any]:
    """Turn a ``run_episode`` result into flat arrays (images stay uint8)."""
    recs = result["records"]
    t = len(recs)
    expert_idx = np.zeros(t, dtype=np.int16)
    executed_idx = np.zeros(t, dtype=np.int16)
    factors = np.zeros((t, 3), dtype=np.int16)
    labels = np.zeros((t, len(NOUL_KEYS)), dtype=np.float32)
    phase = np.zeros(t, dtype=np.int8)
    state = np.zeros((t, 6), dtype=np.float32)
    setpoint = np.zeros((t, 6), dtype=np.float32)
    dist = np.zeros(t, dtype=np.float32)
    for i, r in enumerate(recs):
        ex = Primitive(**r["expert"])
        exe = Primitive(**r["executed"])
        expert_idx[i] = primitive_index(ex, SO101_PRIMITIVES)
        executed_idx[i] = primitive_index(exe, SO101_PRIMITIVES)
        factors[i] = factor_primitive(ex)
        labels[i] = [float(r["labels"][k]) for k in NOUL_KEYS]
        phase[i] = PHASES.index(r["phase"])
        state[i] = r["state"]
        setpoint[i] = r["setpoint_qpos"]
        dist[i] = r["tcp_cube_distance"]
    out: dict[str, Any] = {
        "state": state,
        "setpoint_qpos": setpoint,
        "expert_idx": expert_idx,
        "executed_idx": executed_idx,
        "expert_factors": factors,
        "labels": labels,
        "phase": phase,
        "tcp_cube_distance": dist,
        "task": np.int8(TASKS.index(result["task"])),
        "success": np.bool_(result["success"]),
        "steps_to_success": np.int16(result["steps_to_success"] or -1),
        "final_distance": np.float32(result["final_distance"]),
        "cube_lift": np.float32(result["cube_lift"]),
        "round": np.int8(round_id),
        "actor": actor,
        "episode_json": json.dumps(result["episode"], default=float),
    }
    if "scene" in recs[0]:
        out["scene"] = np.stack([r["scene"] for r in recs]).astype(np.uint8)
        out["wrist"] = np.stack([r["wrist"] for r in recs]).astype(np.uint8)
    return out


def write_episode(root: Path, name: str, encoded: dict[str, Any]) -> Path:
    """Atomically write one encoded episode and append its manifest line."""
    ep_dir = root / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    path = ep_dir / f"{name}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **encoded)
    tmp.rename(path)
    with (root / "manifest.jsonl").open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "name": name,
                    "task": TASKS[int(encoded["task"])],
                    "success": bool(encoded["success"]),
                    "ticks": int(len(encoded["state"])),
                    "steps_to_success": int(encoded["steps_to_success"]),
                    "final_distance": float(encoded["final_distance"]),
                    "cube_lift": float(encoded["cube_lift"]),
                    "round": int(encoded["round"]),
                    "actor": str(encoded["actor"]),
                }
            )
            + "\n"
        )
    return path


def generate(
    root: Path,
    *,
    n_reach: int,
    n_pick: int,
    seed: int,
    worker: str = "w0",
    round_id: int = 0,
    actor_name: str = "expert",
    actor: Callable[..., Primitive] | None = None,
    keep_failures: bool = False,
    max_ticks: int = 120,
    log_every: int = 10,
) -> dict[str, Any]:
    """Roll ``n_reach + n_pick`` episodes and shard them under ``root``.

    Round 0 (``actor is None``) keeps successful expert episodes only unless
    ``keep_failures``; DAgger rounds keep everything (the expert label is what
    matters, not whether the learner got there).
    """
    from strands_robots import Robot

    root = Path(root)
    robot = Robot("so101", mode="sim")
    scene = So101Scene(robot, seed=seed)
    plan = ["reach"] * n_reach + ["pick"] * n_pick
    rng = np.random.default_rng(seed)
    rng.shuffle(plan)
    stats = {"attempted": 0, "kept": 0, "success": {"reach": 0, "pick": 0}, "n": {"reach": 0, "pick": 0}}
    t0 = time.time()
    try:
        for i, task in enumerate(plan):
            set_task = getattr(actor, "set_task", None)
            if set_task is not None:
                set_task(task)
            res = run_episode(scene, task, actor=actor, max_ticks=max_ticks, keep_images=True)
            stats["attempted"] += 1
            stats["n"][task] += 1
            stats["success"][task] += int(res["success"])
            if actor is None and not res["success"] and not keep_failures:
                continue
            write_episode(
                root, f"ep_r{round_id}_{worker}_{i:05d}", encode_episode(res, round_id=round_id, actor=actor_name)
            )
            stats["kept"] += 1
            if log_every and (i + 1) % log_every == 0:
                dt = time.time() - t0
                print(
                    f"[gen {worker}] {i + 1}/{len(plan)} kept={stats['kept']} "
                    f"reach={stats['success']['reach']}/{stats['n']['reach']} "
                    f"pick={stats['success']['pick']}/{stats['n']['pick']} {dt / (i + 1):.1f}s/ep",
                    flush=True,
                )
    finally:
        robot.destroy()
    stats["seconds"] = time.time() - t0
    return stats


def dagger_actor(
    checkpoint: str, *, beta: float = 0.0, seed: int = 0, device: str = "cuda"
) -> Callable[..., Primitive]:
    """The learner drives, the expert labels: ``(observation, privileged, expert) -> primitive``.

    With probability ``beta`` per tick the expert's primitive is executed instead
    (classic DAgger mixing); ``beta=0`` is pure learner roll-outs. The task is read
    from the privileged state so one actor serves both tasks.
    """
    from .expert import state_vector
    from .policy import S1VBrain

    brain = S1VBrain(checkpoint, device=device, cuda_graph=str(device).startswith("cuda"))
    rng = np.random.default_rng(seed + 7)

    class _Actor:
        task = "reach"

        def set_task(self, task: str) -> None:
            self.task = task
            brain.decisions.clear()
            brain.tick_ms.clear()

        def __call__(self, obs: dict[str, Any], priv: Any, expert: Primitive) -> Primitive:
            if beta > 0.0 and rng.random() < beta:
                return expert
            prim, _ = brain.decide(obs["scene"], obs["wrist"], state_vector(priv.qpos), self.task)
            return prim

    return _Actor()


def load_backbone(device: str = "cuda"):
    """Frozen DINOv2-small in eval mode plus its ImageNet normalisation tensors."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, mean, std


def featurize_images(model, mean, std, images: np.ndarray, *, grid: int = GRID, batch: int = 64) -> np.ndarray:
    """uint8 (N,224,224,3) -> float16 (N, 1 + grid*grid, 384): CLS then a pooled patch grid."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.inference_mode():
        for s in range(0, len(images), batch):
            x = torch.from_numpy(images[s : s + batch]).to(mean.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            h = model(pixel_values=x).last_hidden_state  # (B, 1+256, 384)
            cls = h[:, :1]
            patches = h[:, 1:]
            side = int(round(patches.shape[1] ** 0.5))
            pg = patches.reshape(-1, side, side, patches.shape[-1]).permute(0, 3, 1, 2)
            pg = F.adaptive_avg_pool2d(pg, grid).flatten(2).transpose(1, 2)
            out.append(torch.cat([cls, pg], dim=1).half().cpu().numpy())
    return np.concatenate(out, axis=0)


def featurize_root(root: Path, *, device: str = "cuda", grid: int = GRID, overwrite: bool = False) -> int:
    """Write ``features/<name>.npz`` for every episode shard lacking one. Returns count written."""
    root = Path(root)
    feat_dir = root / "features"
    feat_dir.mkdir(exist_ok=True)
    model, mean, std = load_backbone(device)
    done = 0
    for ep in sorted((root / "episodes").glob("ep_*.npz")):
        if ep.name.endswith(".tmp.npz"):
            continue
        target = feat_dir / ep.name
        if target.exists() and not overwrite:
            continue
        with np.load(ep) as z:
            scene = featurize_images(model, mean, std, z["scene"], grid=grid)
            wrist = featurize_images(model, mean, std, z["wrist"], grid=grid)
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, scene=scene, wrist=wrist, grid=np.int8(grid))
        tmp.rename(target)
        done += 1
    return done


def read_manifest(root: Path) -> list[dict[str, Any]]:
    """Manifest rows (one dict per episode) or ``[]`` when the root is empty."""
    path = Path(root) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    """CLI: ``generate`` expert episodes or ``featurize`` existing shards."""
    import argparse

    ap = argparse.ArgumentParser(description="S1V data: generate expert episodes or featurize shards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("root", type=Path)
    g.add_argument("--reach", type=int, default=100)
    g.add_argument("--pick", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--worker", default="w0")
    g.add_argument("--keep-failures", action="store_true")
    g.add_argument("--round", type=int, default=0, help="DAgger round id written into the shard names")
    g.add_argument(
        "--actor", default=None, help="DAgger: checkpoint dir or HF repo of the learner that drives; the expert labels"
    )
    g.add_argument(
        "--beta", type=float, default=0.0, help="DAgger mixing: probability per tick that the expert drives instead"
    )
    g.add_argument("--device", default="cuda")
    f = sub.add_parser("featurize")
    f.add_argument("root", type=Path)
    f.add_argument("--device", default="cuda")
    f.add_argument("--grid", type=int, default=GRID)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        actor = None
        actor_name = "expert"
        if args.actor is not None:
            actor = dagger_actor(args.actor, beta=args.beta, seed=args.seed, device=args.device)
            actor_name = f"dagger:{Path(args.actor).name}:beta={args.beta}"
        stats = generate(
            args.root,
            n_reach=args.reach,
            n_pick=args.pick,
            seed=args.seed,
            worker=args.worker,
            round_id=args.round,
            actor_name=actor_name,
            actor=actor,
            keep_failures=args.keep_failures,
        )
        print(json.dumps(stats))
    else:
        print(featurize_root(args.root, device=args.device, grid=args.grid))


if __name__ == "__main__":
    main()
