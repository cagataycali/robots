#!/usr/bin/env python3
"""Prove a shard's EPISODES.jsonl goals are the goals that GENERATED it -- bit-exactly, not plausibly.

Why: the parquet carries no cube pose, so the per-episode goal lives in a sidecar. A sidecar rebuilt from
seeds is only as good as the rebuild: RLENV it11 shipped goals that were right-shaped and wrong-positioned
(wrong task sampler), and it12 nearly repeated it (band-scale transform missing). A rebuild cannot be checked
against itself, so this checks it against the RECORDED ACTIONS: re-solve the expert's keyframes from the
sidecar goal and compare the plan's first action to the episode's first recorded action. The expert plans
directly to the cube, so if the sidecar goal were wrong, that action would differ.
it13's lesson, applied to it11's own check, which was itself an uncommitted one-off.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root (the dir holding data/ meta/ EPISODES.jsonl)")
    p.add_argument("--report", default=None, help="GEN-REPORT.json of the shard (for band_scale)")
    p.add_argument("--episodes", type=int, default=10, help="how many episodes to check (spread evenly)")
    # NOT 0.0: the parquet stores `action` as float32, so a float64 replan can at best agree to float32
    # round-trip (~1.2e-07 observed on 10 so101 episodes). Demanding bit equality across that cast is a test
    # that fails for a reason that has nothing to do with the goal -- it13's "wrong instrument" lesson again.
    # it11 described its own inline check as "bit-exact"; this is the honest criterion.
    p.add_argument("--tol", type=float, default=1.5e-07,
                   help="max abs action delta allowed (default: float32 round-trip)")
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    sys.path.insert(0, a.svla)
    import pandas as pd
    from strands_vla import scene as S, expert as X
    from strands_vla.embodiments import EMBODIMENTS

    rows = [json.loads(l) for l in open(os.path.join(a.root, "EPISODES.jsonl"))]
    rep = json.load(open(a.report)) if a.report else {}
    arm = rows[0]["arm"]; task = rows[0]["task"]
    e = EMBODIMENTS[arm]
    k = float((rep or {}).get("band_scale") or 1.0)
    if k != 1.0:   # harvest.py:143-151 replaces cube_box; the goals were sampled from the SCALED box
        import dataclasses
        (xl, xh), (yl, yh) = e.cube_box
        cx, cy = (xl + xh) / 2.0, (yl + yh) / 2.0
        e = dataclasses.replace(e, cube_box=((cx + (xl - cx) * k, cx + (xh - cx) * k),
                                             (cy + (yl - cy) * k, cy + (yh - cy) * k)))
    # lerobot v3 layout: ONE parquet per chunk holding every episode, selected by episode_index --
    # not one file per episode (which is what the first version of this script assumed and did not find).
    import glob as _g
    pqs = sorted(_g.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True))
    if not pqs:
        print(json.dumps({"verdict": "NO PARQUET FOUND", "root": a.root})); return 1
    df = pd.concat([pd.read_parquet(f) for f in pqs], ignore_index=True)
    first = {int(ep): g.sort_values("frame_index").iloc[0] if "frame_index" in g else g.iloc[0]
             for ep, g in df.groupby("episode_index")}
    idx = np.linspace(0, len(rows) - 1, min(a.episodes, len(rows))).round().astype(int)
    robot, _ = S.build(arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
    checked, exact, worst, bad = [], 0, 0.0, []
    try:
        for i in idx:
            r = rows[int(i)]
            ei = int(r["episode_index"])
            if ei not in first:
                bad.append({"episode": ei, "why": "episode_index absent from parquet"}); continue
            rec = np.asarray(first[ei]["action"], dtype=float)
            robot.reset(); robot.move_object(S.CUBE, position=[r["cube_x"], r["cube_y"], e.cube_z])
            kf = X.solve_keyframes(robot, e, (r["cube_x"], r["cube_y"]), task)
            pol = X.ScriptedReplayPolicy(hz=e.control_hz)
            pol.set_robot_state_keys(list(e.action_keys)); pol.load_plan(kf, task)
            pred = np.asarray(pol.target_at(0), dtype=float)
            d = float(np.abs(pred - rec).max())
            worst = max(worst, d); checked.append(d)
            if d == 0.0: exact += 1
            elif d > a.tol: bad.append({"episode": int(r["episode_index"]), "max_abs_delta": d})
    finally:
        try: robot.close()
        except Exception: pass
    out = {"root": a.root, "arm": arm, "task": task, "band_scale": k, "n_checked": len(checked),
           "bit_exact": exact, "max_abs_delta": worst, "tol": a.tol, "failures": bad,
           "criterion": "max |replanned first action - recorded first action| <= tol (float32 round-trip)",
           "verdict": ("GOALS VERIFIED - the recorded first action is reproduced from the sidecar goal to float32"
                       if checked and not bad else
                       "GOALS NOT VERIFIED - the sidecar goal does not reproduce the recorded action")}
    print(json.dumps(out))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    return 0 if checked and not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
