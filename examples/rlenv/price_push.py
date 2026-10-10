"""What does this lane's PUSH success criterion actually require? (floor-priced, per arm)

The touch criterion got priced against the project's embodied bound in it9 (capture_radius.py): one surface-gap
rule implied three different capture radii, and koch's cells turned out void. PUSH escapes that bound because
it is displacement-based, not contact-based -- and escaping a bound is not the same as having one. The
criterion is PUSH_MIN = 0.04 m of cube displacement along the approach axis, and a displacement threshold
means nothing until the displacement a NON-pushing arm produces is measured.

So this measures the floor with the harness's own negative controls, on the same scene, same episode length,
same seeds: HoldPolicy (arm frozen at rest -- pure settle/drift of the cube) and RandomWalkPolicy (the arm
flails without aiming -- displacement obtainable by accident). Both are compared to the expert's achieved
displacement on labelled successes. The margin between the floor and the threshold is what the criterion is
worth; if a flailing arm clears 0.04 m, every push rate we published is void the way koch's touch rates are.

Usage: python examples/rlenv/price_push.py --arms so101,so100,koch --episodes 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _scaled(e, k: float):
    """Scale an embodiment's cube_box about its centre, exactly as harvest.py:143-151 does.

    Without this, a variant shard harvested with --band-scale k could only ever INHERIT the canonical
    shard's criterion cells -- i.e. be judged on a band it was not generated from (it15's table made the
    inheritance visible, which is why this exists).
    """
    if k == 1.0:
        return e
    import dataclasses
    (xl, xh), (yl, yh) = e.cube_box
    cx, cy = (xl + xh) / 2.0, (yl + yh) / 2.0
    return dataclasses.replace(e, cube_box=((cx + (xl - cx) * k, cx + (xh - cx) * k),
                                            (cy + (yl - cy) * k, cy + (yh - cy) * k)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="so101,so100,koch")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    p.add_argument("--band-scale", type=float, default=1.0,
                   help="price the criterion on a VARIANT shard's scaled cube box")
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    out = {"threshold_m": float(X.PUSH_MIN), "episodes_per_cell": a.episodes,
           "criterion": "cube displacement along the approach axis > PUSH_MIN",
           "note": "planar displacement reported too, so the verdict does not rest on the axis convention",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}

    for arm in a.arms.split(","):
        e = _scaled(EMBODIMENTS[arm], float(a.band_scale))
        robot, _ = S.build(arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
        cells: dict[str, dict] = {}
        try:
            for cell in ("hold", "random", "expert"):
                along, planar = [], []
                for i in range(a.episodes):
                    rng = np.random.default_rng(90000 + a.seed + i)
                    xy = S.sample_cube(e, rng, "push")

                    def place():
                        robot.reset()
                        robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])

                    place()
                    kf = X.solve_keyframes(robot, e, xy, "push")
                    place()
                    c0 = S.cube_pos(robot).copy()
                    axis = c0[:2] / (np.linalg.norm(c0[:2]) or 1.0)  # outward approach direction
                    if cell == "hold":
                        pol = X.HoldPolicy(hz=e.control_hz)
                    elif cell == "random":
                        pol = X.RandomWalkPolicy(hz=e.control_hz)
                    else:
                        pol = X.ScriptedReplayPolicy(hz=e.control_hz)
                    pol.set_robot_state_keys(list(e.action_keys))
                    pol.load_plan(kf, "push")
                    robot.run_policy(robot_name=e.name, policy_object=pol, instruction="push",
                                     duration=X.EPISODE_S, control_frequency=e.control_hz, n_episodes=1,
                                     reset_between=True, seed=a.seed + i, fast_mode=True)
                    d = S.cube_pos(robot) - c0
                    along.append(float(np.dot(d[:2], axis)))
                    planar.append(float(np.linalg.norm(d[:2])))
                al, pl = np.asarray(along), np.asarray(planar)
                cells[cell] = {
                    "along_m": {"median": round(float(np.median(al)), 4),
                                "p95": round(float(np.percentile(al, 95)), 4),
                                "max": round(float(al.max()), 4)},
                    "planar_m": {"median": round(float(np.median(pl)), 4),
                                 "max": round(float(pl.max()), 4)},
                    "would_be_scored_success": int((al > X.PUSH_MIN).sum()),
                    "n": int(al.size)}
                print(json.dumps({arm: {cell: cells[cell]}}), flush=True)
        finally:
            robot.destroy()

        fl = max(cells["hold"]["along_m"]["max"], cells["random"]["along_m"]["max"])
        fp = cells["hold"]["would_be_scored_success"] + cells["random"]["would_be_scored_success"]
        cells["floor_max_along_m"] = round(float(fl), 4)
        cells["false_successes_on_floors"] = fp
        cells["margin_x"] = round(float(X.PUSH_MIN / fl), 2) if fl > 1e-6 else None
        cells["verdict"] = ("VOID — a non-pushing arm clears the threshold" if fp
                            else f"SOUND — threshold is {cells['margin_x']}x the worst floor displacement"
                            if cells["margin_x"] else "SOUND — floors produce no measurable displacement")
        out["arms"][arm] = cells
        print(json.dumps({arm: {k: cells[k] for k in
                                ("floor_max_along_m", "false_successes_on_floors", "margin_x", "verdict")}}),
              flush=True)

    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
