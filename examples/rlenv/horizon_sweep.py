"""Does the venue's episode horizon change what the demonstrations should be?

LONELY-ELECTRON it37 established the project's first real venue: so101 PUSH at 18 s, where the pixel
oracle reaches 32/40 with contact in 40/40. Every push shard I have published is 6 s (60 frames at
10 Hz), because the expert's EPISODE_S is 6.0. So the training data and the venue disagree about how
long the task is, and a head trained on 6-s episodes is evaluated on 18-s ones.

Three things could be true and they have different consequences:
  (a) the expert's plan STRETCHES with the horizon -> my 6-s shards under-use the task and the longer
      episodes push further; re-harvest at 18 s
  (b) the plan finishes early and DWELLS -> 18-s shards are 2/3 padding, which is a real cost in frames
      and a distribution the head will see at eval; re-harvest anyway, but knowingly
  (c) the arm RETRACTS and disturbs the cube after succeeding -> the extra time can UNDO a success, and
      the label depends on when you stop looking, which is the producer's problem

So this runs the same seeds at several horizons and reports, per horizon: success under the shard's own
criterion, the displacement along the commanded direction, and -- the part that matters for (c) -- the
displacement measured at 6 s versus at the end, for the same episode.

Usage: python examples/rlenv/horizon_sweep.py --arm so101 --horizons 6,12,18 --episodes 30
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
    p.add_argument("--task", default="push")
    p.add_argument("--horizons", default="6,12,18")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    e = EMBODIMENTS[a.arm]
    hz = e.control_hz
    out = {"arm": a.arm, "task": a.task, "episodes": a.episodes, "control_hz": hz,
           "shard_episode_s": float(X.EPISODE_S), "touch_gap_m": float(X.TOUCH_GAP),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    robot, _ = S.build(a.arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
    try:
        for H in [float(x) for x in a.horizons.split(",")]:
            ok = 0
            disp_end, disp_at6, regressed, nsteps = [], [], 0, []
            for i in range(a.episodes):
                rng = np.random.default_rng(90000 + a.seed + i)
                xy = S.sample_cube(e, rng, a.task)

                def place():
                    robot.reset()
                    robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])

                place()
                kf = X.solve_keyframes(robot, e, xy, a.task)
                place()
                pol = X.ScriptedReplayPolicy(hz=hz)
                pol.set_robot_state_keys(list(e.action_keys))
                pol.load_plan(kf, a.task)
                ahat = np.array([e.approach[0], e.approach[1], 0.0])
                st = {"c0": None, "gap": 1e9, "d6": None, "n": 0}

                def obs(ev, st=st):
                    nm = type(ev).__name__
                    if nm == "RunPolicyStarted":
                        st["c0"] = S.cube_pos(robot)
                    elif nm == "RunPolicyStep":
                        st["n"] += 1
                        st["gap"] = min(st["gap"], S.cube_surface_gap(robot, e))
                        if st["d6"] is None and st["n"] >= int(X.EPISODE_S * hz):
                            st["d6"] = float(np.dot(S.cube_pos(robot) - st["c0"], ahat))

                robot.run_policy(robot_name=e.name, policy_object=pol, instruction="probe",
                                 duration=H, control_frequency=hz, n_episodes=1, reset_between=True,
                                 seed=a.seed + i, observer=obs, fast_mode=True)
                c1 = S.cube_pos(robot)
                d = float(np.dot(c1 - st["c0"], ahat))
                disp_end.append(d)
                nsteps.append(st["n"])
                if st["d6"] is not None:
                    disp_at6.append(st["d6"])
                    if st["d6"] - d > 0.005:  # the extra time UNDID 5 mm or more of the push
                        regressed += 1
                if X.success(a.task, st["gap"] <= X.TOUCH_GAP, d):
                    ok += 1
            # steps_measured, not H*hz: the first version of this sweep reported the REQUESTED step
            # count and three horizons came back byte-identical, which is either a real null or a knob
            # that did nothing. Only the measured count can tell those apart.
            row = {"horizon_s": H, "steps_requested": int(H * hz),
                   "steps_measured_median": float(np.median(nsteps)),
                   "steps_measured_max": int(max(nsteps)), "ok": ok, "rate": round(ok / a.episodes, 4),
                   "disp_end_median_mm": round(float(np.median(disp_end)) * 1000, 2),
                   "disp_at_6s_median_mm": (round(float(np.median(disp_at6)) * 1000, 2)
                                            if disp_at6 else None),
                   "episodes_regressed_after_6s": regressed}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        robot.destroy()
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
