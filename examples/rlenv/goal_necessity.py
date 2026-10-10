"""Is the TASK goal-necessary, or can a policy that never reads the goal succeed anyway?

This is the producer-side twin of a policy-side result. LONELY-ELECTRON (D186) ran a GOAL-BLIND
oracle -- one that aims at the centre of the target box and reads no goal at all -- and it scored
so101 28/80, which is close to the promoted head's 35/80. That is usually read as a fact about the
head. It is at least as much a fact about THE DATA: if aiming at the middle of the band succeeds
most of the time, then the dataset cannot teach goal-conditioning, however perfectly its goal is
labelled, because ignoring the label costs almost nothing.

So this measures, for a given scene config, the two numbers that bound what any head trained on
these shards could show:

    goal-aware   success of the real expert, which plans to the TRUE cube pose
    goal-blind   success of the same expert planning to the BOX CENTRE, with the cube still at its
                 true pose -- identical code path, one substitution

    headroom = goal-aware - goal-blind

Headroom is the only part of the score a learner can earn BY READING THE GOAL. A shard with a
small headroom is not broken, but no vision/goal ablation measured on it can be informative, and a
head trained on it should not be expected to be goal-sensitive. RLENV publishes this per shard next
to the two pixel numbers (frame-0 delta, goal-only contrast) so STARVLA can see what it is buying.

It also sweeps the BAND WIDTH, because that is the lever: scaling the sampling box about its centre
moves the goal further from the centre than the success tolerance, which is what makes reading the
goal necessary. The band that maximises headroom is reported as the recommended harvest band.

Usage: python examples/rlenv/goal_necessity.py --arm so101 --task touch --episodes 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def scaled(e, k: float):
    """The same embodiment with its cube_box scaled by k about its own centre."""
    (xlo, xhi), (ylo, yhi) = e.cube_box
    cx, cy = (xlo + xhi) / 2.0, (ylo + yhi) / 2.0
    box = ((cx + (xlo - cx) * k, cx + (xhi - cx) * k),
           (cy + (ylo - cy) * k, cy + (yhi - cy) * k))
    return dataclasses.replace(e, cube_box=box)


def run_cell(robot, e, S, X, task: str, n: int, blind: bool, seed0: int) -> dict:
    """n episodes; when blind, the expert plans to the box CENTRE but the cube stays where it is."""
    ok = 0
    gaps = []
    for i in range(n):
        rng = np.random.default_rng(90000 + seed0 + i)
        xy = S.sample_cube(e, rng, task)
        aim = np.asarray(S.center(e), dtype=float) if blind else xy

        def place():
            robot.reset()
            robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])

        place()
        kf = X.solve_keyframes(robot, e, aim, task)
        place()
        pol = X.ScriptedReplayPolicy(hz=e.control_hz)
        pol.set_robot_state_keys(list(e.action_keys))
        pol.load_plan(kf, task)
        st = {"gap": 1e9, "c0": None}

        def obs(ev, st=st):
            nm = type(ev).__name__
            if nm == "RunPolicyStarted":
                st["c0"] = S.cube_pos(robot)
            elif nm == "RunPolicyStep":
                st["gap"] = min(st["gap"], S.cube_surface_gap(robot, e))

        robot.run_policy(robot_name=e.name, policy_object=pol, instruction="probe",
                         duration=X.EPISODE_S, control_frequency=e.control_hz, n_episodes=1,
                         reset_between=True, seed=seed0 + i, observer=obs, fast_mode=True)
        c1 = S.cube_pos(robot)
        c0 = st["c0"] if st["c0"] is not None else c1
        disp = float(np.dot(c1 - c0, np.array([e.approach[0], e.approach[1], 0.0])))
        if X.success(task, st["gap"] <= X.TOUCH_GAP, disp):
            ok += 1
        gaps.append(float(st["gap"]))
    return {"n": n, "ok": ok, "rate": round(ok / n, 4), "median_gap": round(float(np.median(gaps)), 5)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="so101")
    p.add_argument("--task", default="touch", choices=("touch", "push"))
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--scales", default="1.0,1.5,2.0")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    base = EMBODIMENTS[a.arm]
    out = {"arm": a.arm, "task": a.task, "episodes_per_cell": a.episodes,
           "touch_gap_m": float(X.TOUCH_GAP), "base_cube_box": base.cube_box,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    for k in [float(x) for x in a.scales.split(",")]:
        e = scaled(base, k)
        robot, _ = S.build(a.arm, cam_size=(64, 64), with_wrist=bool(base.wrist_parent))
        try:
            # rebuild the scene per scale so the cube object is created inside the scaled band
            aware = run_cell(robot, e, S, X, a.task, a.episodes, False, a.seed)
            blind = run_cell(robot, e, S, X, a.task, a.episodes, True, a.seed)
        finally:
            robot.destroy()
        row = {"band_scale": k, "cube_box": e.cube_box, "goal_aware": aware, "goal_blind": blind,
               "headroom": round(aware["rate"] - blind["rate"], 4)}
        out["rows"].append(row)
        print(json.dumps({"scale": k, "aware": aware["rate"], "blind": blind["rate"],
                          "headroom": row["headroom"]}), flush=True)
    best = max(out["rows"], key=lambda r: r["headroom"])
    out["recommended_band_scale"] = best["band_scale"]
    out["recommended_headroom"] = best["headroom"]
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    print(json.dumps({"recommended_band_scale": best["band_scale"],
                      "headroom": best["headroom"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
