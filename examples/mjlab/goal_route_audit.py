"""Audit a LeRobot v3 dataset for GOAL ROUTES: can a policy trained on it know what to do?

RLENV it1 found 3,225,600 frames of so101 reach whose per-episode target was recorded
nowhere, and the model-side symptom (a policy collapsing to its action mean) had already
been measured by two other lanes. This tool makes that check one command on any shard.

It asks two INDEPENDENT questions, because they disagree exactly when the goal sits in a
column nobody reads:

STRUCTURAL  Is there any route by which the goal could reach the learner?
            - camera/video keys in meta/info.json      -> goal could live in pixels
            - a non-proprio state column (target/goal/object/ee_to_)
            - more than one distinct task string in meta/tasks.parquet -> goal in text

EMPIRICAL   At frame 0, does the INPUT move at all while the LABEL moves?
            A dataset whose frame-0 observation.state has zero spread across episodes while
            frame-0 action spreads is unlearnable at that frame by construction: identical
            input, different labels, so the BC optimum is the conditional mean. This is also
            the project's rung-1 law (D62/EVAL it20: every episode must differ at the first
            observation the policy sees).

Exit code 0 = at least one goal route AND frame-0 input spread > --min-jitter.
Exit code 1 = DO-NOT-TRAIN for a goal-conditioned policy; the reason is printed.

Usage::

    python examples/mjlab/goal_route_audit.py --root <dataset dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROPRIO_HINTS = ("joint", "qpos", "qvel", "pos", "vel", "gripper", "state")
GOAL_HINTS = ("target", "goal", "object", "cube", "ee_to", "command", "desired")


def _distinct_tasks(root: Path) -> list[str]:
    p = root / "meta" / "tasks.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        names = list(t.index.astype(str)) if t.index.name == "task" else [str(x) for x in t.iloc[:, 0]]
        return sorted(set(names))
    jl = root / "meta" / "tasks.jsonl"
    if jl.exists():
        return sorted({json.loads(ln)["task"] for ln in jl.read_text().splitlines() if ln.strip()})
    return []


def audit(root: Path, min_jitter: float) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    vid = [k for k in feats if k.startswith(("observation.image", "observation.images")) or "image" in k]
    vid += [k for k in (info.get("video_keys") or []) if k not in vid]
    state_names: list[str] = []
    for key in ("observation.state", "observation.environment_state"):
        if key in feats:
            state_names += [str(n) for n in (feats[key].get("names") or [])]
    goal_cols = [n for n in state_names if any(h in n.lower() for h in GOAL_HINTS)]
    goal_feats = [k for k in feats if any(h in k.lower() for h in GOAL_HINTS)]
    tasks = _distinct_tasks(root)

    out: dict = {
        "root": str(root),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "state_dim": (feats.get("observation.state") or {}).get("shape"),
        "camera_keys": vid,
        "goal_state_columns": goal_cols,
        "goal_features": goal_feats,
        "distinct_tasks": len(tasks),
        "task_sample": tasks[:4],
    }
    routes = {
        "pixels": bool(vid),
        "state": bool(goal_cols or goal_feats),
        "text": len(tasks) > 1,
    }
    out["goal_routes"] = routes
    out["n_goal_routes"] = sum(routes.values())

    # EMPIRICAL: frame 0 across episodes
    files = sorted((root / "data").rglob("*.parquet"))
    if files:
        df = pd.read_parquet(files[0], columns=["observation.state", "action", "frame_index", "episode_index"])
        f0 = df[df.frame_index == 0]
        S = np.stack(f0["observation.state"].to_numpy())
        A = np.stack(f0["action"].to_numpy())
        out["frame0"] = {
            "file": files[0].name,
            "episodes_sampled": int(len(f0)),
            "state_std": [round(float(x), 6) for x in S.std(0)],
            "state_range": [round(float(x), 6) for x in (S.max(0) - S.min(0))],
            "action_std": [round(float(x), 6) for x in A.std(0)],
            "action_range": [round(float(x), 6) for x in (A.max(0) - A.min(0))],
            "max_state_range": round(float((S.max(0) - S.min(0)).max()), 6),
            "max_action_range": round(float((A.max(0) - A.min(0)).max()), 6),
        }

    reasons = []
    if out["n_goal_routes"] == 0:
        reasons.append(
            "NO GOAL ROUTE: no camera keys, no goal/target state column, and "
            f"{out['distinct_tasks']} distinct task string(s) -> the goal is recorded nowhere, "
            "so the BC optimum is the conditional action mean."
        )
    f0d = out.get("frame0")
    if f0d and f0d["max_state_range"] <= min_jitter:
        reasons.append(
            f"RUNG-1 FAIL: frame-0 observation.state spread is {f0d['max_state_range']} "
            f"(<= --min-jitter {min_jitter}) across {f0d['episodes_sampled']} episodes while frame-0 action "
            f"spreads {f0d['max_action_range']} -> identical input, moving label."
        )
    out["verdict"] = "DO-NOT-TRAIN" if reasons else "OK"
    out["reasons"] = reasons
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--min-jitter", type=float, default=1e-9, help="frame-0 state spread at or below this fails rung-1")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    rec = audit(a.root, a.min_jitter)
    print(json.dumps(rec, indent=1))
    if a.json:
        a.json.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"\nVERDICT: {rec['verdict']}")
    for r in rec["reasons"]:
        print(f"  - {r}")
    return 1 if rec["verdict"] == "DO-NOT-TRAIN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
