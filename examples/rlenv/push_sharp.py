#!/usr/bin/env python3
"""The 'behind' negative control for the push criterion -- the only INFORMATIVE floor it has.

Why this file exists (RLENV it13, supervisor D212 order (a)): the push criterion's hold and random floors are
exactly 0.0 on every arm because an arm starting at rest never reaches the cube, so they cannot discriminate.
The control that can is an arm placed BEHIND the cube, adjacent to it, doing no directed motion: if proximity
alone cleared PUSH_MIN, this cell would show it. That control had been run ONCE, on so101, from an ad-hoc
command that was never committed -- so the lane's one informative control was not reproducible and the two
arms with the thinnest expert margin (so100, koch) had no control at all. This script is that control.

'Behind' is built from the expert's own plan rather than from a new pose guess: PHASES[task] is executed up to
the stage before the terminal stroke, then HELD for the remainder of the episode. So the gripper arrives at
its pre-contact pose next to the cube and stays there, and any displacement measured is contact/settle, not a
push. Re-running so101 reproduces runs/push-sharp-so101.json (median 0.0077, max 0.0225) or this file is wrong.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="so101,so100,koch")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    p.add_argument("--stop-after", default="behind",
                   help="last PHASES[task] stage to execute; the rest of the episode is HELD")
    a = p.parse_args(argv)
    sys.path.insert(0, a.svla)
    from strands_vla import scene as S, expert as X
    from strands_vla.embodiments import EMBODIMENTS

    STOP_AFTER = a.stop_after

    class BehindPolicy(X.ScriptedReplayPolicy):
        """The expert's plan minus its terminal stroke: arrive behind the cube, then hold."""
        def load_plan(self, kf, task):
            phases = list(X.PHASES[task])
            # NOT phases[:-1]: PHASES["push"] is behind_above, behind, mid, sweep, up, so dropping the last
            # stage drops the RETRACT and leaves the sweep -- i.e. the expert, not a control. it13 ran that by
            # mistake and it reproduced price_push's expert cell to 4 dp on all three arms. The control must
            # stop after the stage named by --stop-after ("behind"), which is adjacent-to-cube, pre-stroke.
            names = [n for n, _ in phases]
            if STOP_AFTER not in names:
                raise SystemExit(f"--stop-after {STOP_AFTER!r} not in PHASES[{task!r}] = {names}")
            kept = phases[:names.index(STOP_AFTER) + 1]
            held = sum(s for _, s in phases[len(kept):]) or 0.0
            seq, cur = [], np.asarray(kf["rest"], dtype=float)
            for name, secs in kept:
                n = max(1, int(round(secs * self.hz)))
                tgt = cur.copy() if name.startswith("hold") else np.asarray(kf[name], dtype=float).copy()
                seq.append((cur.copy(), tgt, n)); cur = tgt
            tail = max(1, int(round((held + max(0.0, X.EPISODE_S - sum(s for _, s in kept) - held)) * self.hz)))
            seq.append((cur.copy(), cur.copy(), tail))
            self.plan, self._step = seq, 0

    out = {"criterion": "cube displacement along the approach axis > PUSH_MIN",
           "threshold_m": float(X.PUSH_MIN), "cell": a.stop_after,
           "cell_meaning": f"expert plan stopped after stage {a.stop_after!r}, then held: adjacent to the cube, no stroke",
           "episodes": a.episodes, "seed": a.seed, "phases_push": [list(x) for x in X.PHASES["push"]],
           "arms": {}}
    for arm in a.arms.split(","):
        e = EMBODIMENTS[arm]
        robot, _ = S.build(arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
        along, planar, touched = [], [], 0
        try:
            for i in range(a.episodes):
                rng = np.random.default_rng(90000 + a.seed + i)   # same seed stream as price_push
                xy = S.sample_cube(e, rng, "push")
                def place():
                    robot.reset(); robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])
                place(); kf = X.solve_keyframes(robot, e, xy, "push"); place()
                c0 = S.cube_pos(robot).copy()
                axis = c0[:2] / (np.linalg.norm(c0[:2]) or 1.0)
                pol = BehindPolicy(hz=e.control_hz)
                pol.set_robot_state_keys(list(e.action_keys)); pol.load_plan(kf, "push")
                robot.run_policy(robot_name=e.name, policy_object=pol, instruction="push",
                                 duration=X.EPISODE_S, control_frequency=e.control_hz, n_episodes=1,
                                 reset_between=True, seed=a.seed + i, fast_mode=True)
                d = S.cube_pos(robot) - c0
                along.append(float(np.dot(d[:2], axis))); planar.append(float(np.linalg.norm(d[:2])))
                if np.linalg.norm(d[:2]) > 1e-4: touched += 1
        finally:
            try: robot.close()
            except Exception: pass
        al = np.asarray(along)
        cell = {"n": len(al), "start_pose_key": a.stop_after,
                "along_m": {"median": round(float(np.median(al)), 4), "p95": round(float(np.percentile(al, 95)), 4),
                            "max": round(float(al.max()), 4)},
                "false_successes": int((al > X.PUSH_MIN).sum()), "moved_at_all": touched}
        # The headline margin must be a FIELD, not prose (D212 finding 4).
        cell["margin_x"] = round(float(X.PUSH_MIN / al.max()), 3) if al.max() > 0 else None
        cell["verdict"] = ("CRITERION REQUIRES DIRECTED MOTION - proximity alone does not clear it"
                           if cell["false_successes"] == 0 else
                           "CRITERION UNSOUND - proximity without a stroke scores successes")
        out["arms"][arm] = cell
        print(json.dumps({"arm": arm, **cell}), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
