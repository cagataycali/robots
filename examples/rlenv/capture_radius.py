"""Is this lane's touch criterion void under the embodied bound?

D199 rules that ANY touch success rate is unshippable: "every touch cell we own is VOID under the embodied
bound: so101 needs a capture radius <= 0.040 m and measures 0.051, koch 0.063". Three of my six published
shards are touch, and their cards quote touch rates, so either the cards are wrong or my criterion is a
different quantity from the one that was priced.

It IS a different quantity -- TOUCH_GAP = 0.01 m of SURFACE-to-surface distance via mj_geomDistance, where
0 means touching, rather than a radius around the cube centre -- but that is an argument, and an argument is
not a measurement. So this measures the capture radius my criterion actually implies: at the moment the
criterion first fires, how far is the controlled point (the arm's IK reference site) from the cube CENTRE?
That distance is the radius within which my successes are declared, and it is directly comparable to the
0.040 m bound.

Reported per arm: the distribution of that radius over labelled successes, plus the surface gap and the
nearest-arm-point-to-centre distance, and a verdict against the bound. A criterion whose p95 radius exceeds
the bound is void and I correct the cards; one that stays under it is not, and the evidence is this file.

Usage: python examples/rlenv/capture_radius.py --arms so101,koch,so100 --episodes 40 --bound 0.040
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="so101,koch,so100")
    p.add_argument("--task", default="touch")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--bound", type=float, default=0.040, help="embodied capture-radius bound (m)")
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    out = {"bound_m": a.bound, "touch_gap_m": float(X.TOUCH_GAP), "episodes": a.episodes,
           "criterion": "surface gap <= TOUCH_GAP via mj_geomDistance (0 = contact)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}

    for arm in a.arms.split(","):
        e = EMBODIMENTS[arm]
        robot, _ = S.build(arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
        radii, gaps, near = [], [], []
        try:
            for i in range(a.episodes):
                rng = np.random.default_rng(90000 + a.seed + i)
                xy = S.sample_cube(e, rng, a.task)

                def place():
                    robot.reset()
                    robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])

                place()
                kf = X.solve_keyframes(robot, e, xy, a.task)
                place()
                pol = X.ScriptedReplayPolicy(hz=e.control_hz)
                pol.set_robot_state_keys(list(e.action_keys))
                pol.load_plan(kf, a.task)
                st = {"hit": None}

                def obs(ev, st=st):
                    if type(ev).__name__ != "RunPolicyStep" or st["hit"] is not None:
                        return
                    g, vec = S.cube_surface_gap(robot, e, want_vec=True)
                    if g <= X.TOUCH_GAP:  # the instant the criterion fires
                        c = S.cube_pos(robot)
                        st["hit"] = {
                            "gap": float(g),
                            # the controlled point: the arm's IK reference site/body
                            "radius_ee_to_centre": float(np.linalg.norm(S.ee_pos(robot, e) - c)),
                            # the nearest ARM SURFACE point to the cube centre
                            "nearest_arm_pt_to_centre": (
                                float(np.linalg.norm(vec)) if vec is None else
                                float(np.linalg.norm(c - (c - vec)))),
                        }

                robot.run_policy(robot_name=e.name, policy_object=pol, instruction="probe",
                                 duration=X.EPISODE_S, control_frequency=e.control_hz, n_episodes=1,
                                 reset_between=True, seed=a.seed + i, observer=obs, fast_mode=True)
                if st["hit"]:
                    radii.append(st["hit"]["radius_ee_to_centre"])
                    gaps.append(st["hit"]["gap"])
                    near.append(st["hit"]["nearest_arm_pt_to_centre"])
        finally:
            robot.destroy()

        if not radii:
            out["arms"][arm] = {"fired": 0, "verdict": "NO SUCCESSES — nothing to price"}
            print(json.dumps({arm: out["arms"][arm]}), flush=True)
            continue
        r = np.asarray(radii)
        row = {"fired": len(radii), "cube_half_m": round(float(e.cube_size) / 2, 4)
               if hasattr(e, "cube_size") else None,
               "radius_ee_to_centre_m": {"median": round(float(np.median(r)), 4),
                                         "p95": round(float(np.percentile(r, 95)), 4),
                                         "max": round(float(r.max()), 4)},
               "surface_gap_at_fire_m": {"median": round(float(np.median(gaps)), 4),
                                         "max": round(float(np.max(gaps)), 4)},
               "nearest_arm_pt_to_centre_m": round(float(np.median(near)), 4)}
        row["verdict"] = ("WITHIN BOUND" if float(np.percentile(r, 95)) <= a.bound
                          else "EXCEEDS BOUND — cells void, correct the cards")
        out["arms"][arm] = row
        print(json.dumps({arm: row}), flush=True)

    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
