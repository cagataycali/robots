"""Does my success LABEL mean what the shard's instruction says?

The expert's touch criterion is min-over-episode: `gap <= TOUCH_GAP` at ANY step. So a trajectory that
sweeps past the cube and grazes it for one control tick is recorded as a success and published with the
instruction "touch the red cube". A learner trained on that is being taught to graze.

LONELY-ELECTRON it35 raised this on the policy side (whether a goal-blind score is transient grazes that a
dwell criterion would remove). It is a sharper question for the PRODUCER, because I am the one writing the
label. So this re-runs the expert on the same seeds and records, per episode, the full gap trace rather
than its minimum:

    dwell      longest run of CONSECUTIVE steps within TOUCH_GAP
    terminal   gap at the final step (did it stay, or pass through?)

and reports what fraction of EXPERT-LABELLED SUCCESSES survive a stricter criterion. Run for both the
goal-aware expert (the shards' actual labels) and the goal-blind one (aims at the box centre), because if
the blind successes are grazes while the aware ones dwell, the headroom in goal_necessity.py is an
UNDERSTATEMENT and the stricter criterion is the one to publish.

Usage: python examples/rlenv/dwell_audit.py --arm so101 --episodes 40
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
    p.add_argument("--arm", default="so101")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    p.add_argument("--action-noise", type=float, default=0.0,
                   help="rad of Gaussian noise on the commanded joint targets. it23 measured that a "
                        "noisy expert FAILS LESS on this task (5.5%% -> 0%%), which is only possible "
                        "because success is a minimum over the episode. This flag exists to ask the "
                        "follow-up the shard producer owes: does the noise buy DWELL, or only grazes?")
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    e = EMBODIMENTS[a.arm]
    out = {"arm": a.arm, "task": "touch", "episodes": a.episodes, "touch_gap_m": float(X.TOUCH_GAP),
           "action_noise_rad": float(a.action_noise),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": {}}
    robot, _ = S.build(a.arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
    try:
        for blind in (False, True):
            eps = []
            for i in range(a.episodes):
                rng = np.random.default_rng(90000 + a.seed + i)
                xy = S.sample_cube(e, rng, "touch")
                aim = np.asarray(S.center(e), dtype=float) if blind else xy

                def place():
                    robot.reset()
                    robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])

                place()
                kf = X.solve_keyframes(robot, e, aim, "touch")
                place()
                pol = X.ScriptedReplayPolicy(hz=e.control_hz)
                pol.set_robot_state_keys(list(e.action_keys))
                pol.load_plan(kf, "touch")
                if a.action_noise > 0:
                    nrng = np.random.default_rng(700000 + a.seed + i)
                    _t = pol.target_at
                    pol.target_at = (lambda step, _t=_t, _n=nrng: np.asarray(_t(step), dtype=float)
                                     + _n.normal(0.0, a.action_noise, size=len(e.action_keys)))
                trace = []

                def obs(ev, trace=trace):
                    if type(ev).__name__ == "RunPolicyStep":
                        trace.append(S.cube_surface_gap(robot, e))

                robot.run_policy(robot_name=e.name, policy_object=pol, instruction="probe",
                                 duration=X.EPISODE_S, control_frequency=e.control_hz, n_episodes=1,
                                 reset_between=True, seed=a.seed + i, observer=obs, fast_mode=True)
                t = np.asarray(trace, dtype=float)
                inside = t <= X.TOUCH_GAP
                dwell, run = 0, 0
                for v in inside:
                    run = run + 1 if v else 0
                    dwell = max(dwell, run)
                eps.append({"min_gap": float(t.min()) if t.size else 1e9, "dwell": int(dwell),
                            "terminal": float(t[-1]) if t.size else 1e9, "steps": int(t.size)})
            lab = [x for x in eps if x["min_gap"] <= X.TOUCH_GAP]  # what the shard labels a success
            n = len(lab)
            out["cells"]["goal_blind" if blind else "goal_aware"] = {
                "labelled_success": n, "rate": round(n / a.episodes, 4),
                "median_dwell_steps": float(np.median([x["dwell"] for x in lab])) if n else None,
                "dwell_ge_3": sum(x["dwell"] >= 3 for x in lab),
                "dwell_ge_10": sum(x["dwell"] >= 10 for x in lab),
                "terminal_within_gap": sum(x["terminal"] <= X.TOUCH_GAP for x in lab),
                "graze_only_dwell_1": sum(x["dwell"] <= 1 for x in lab),
                "median_steps": float(np.median([x["steps"] for x in eps])),
                # the three criteria side by side, each over ALL episodes, so "which criterion does
                # noise win under" is answerable without dividing by a different denominator
                "rate_min_gap": round(n / a.episodes, 4),
                "rate_dwell_ge_3": round(sum(x["dwell"] >= 3 for x in eps) / a.episodes, 4),
                "rate_dwell_ge_10": round(sum(x["dwell"] >= 10 for x in eps) / a.episodes, 4),
                "rate_terminal": round(sum(x["terminal"] <= X.TOUCH_GAP for x in eps) / a.episodes, 4),
                "median_min_gap_mm": round(float(np.median([x["min_gap"] for x in eps])) * 1000, 3),
            }
            print(json.dumps({"blind": blind, **out["cells"]["goal_blind" if blind else "goal_aware"]}),
                  flush=True)
    finally:
        robot.destroy()
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
