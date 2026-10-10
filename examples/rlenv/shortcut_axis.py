#!/usr/bin/env python3
"""Is the surviving shortcut AXIS predictable from the expert's own success landscape?

it26 claimed the AMOUNT of goal-decoupling is bought with expert competence; it27 falsified that on a
third arm (PREREG-it27.md). What the three arms did show is that the residual shortcut lands on a
DIFFERENT FEATURE per arm: so100 lateral (cube_y 0.69), koch radial (radius 0.7194), so101 nothing.

This script tests that as a PREDICTION rather than a story, and does it WITHOUT touching the recorded
pairs, so the test cannot be circular. For each arm it drives the goal-aware expert at the SAME action
noise the shards were generated with, over a GRID of cube positions spanning the band, and measures the
success rate at each grid point. The marginal spread of that rate along x (forward/radial) and along y
(lateral) is a property of the arm and its band alone. The prediction is that the axis with the larger
marginal spread is the axis whose goal feature carries the shortcut in that arm's success/failure pair.

A grid is not a sample of the band: every cell gets the same number of episodes, so the landscape is
measured where the band is sparse too. Positions come from the band actually used for harvesting
(band_scale 1.0 = the embodiment's own cube_box).

Usage: python examples/rlenv/shortcut_axis.py --arm koch --task push --bins 4 --episodes 8
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="koch")
    p.add_argument("--task", default="push", choices=("touch", "push"))
    p.add_argument("--bins", type=int, default=4, help="grid bins per axis")
    p.add_argument("--episodes", type=int, default=8, help="episodes per grid cell")
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    e = EMBODIMENTS[a.arm]
    (xlo, xhi), (ylo, yhi) = e.cube_box
    # cell CENTRES, so no grid point sits exactly on the band edge
    xs = [xlo + (xhi - xlo) * (i + 0.5) / a.bins for i in range(a.bins)]
    ys = [ylo + (yhi - ylo) * (j + 0.5) / a.bins for j in range(a.bins)]

    robot, _ = S.build(a.arm, cam_size=(64, 64), with_wrist=bool(e.wrist_parent))
    cells = []
    try:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                ok = 0
                for k in range(a.episodes):
                    seed = a.seed + 1000 * i + 100 * j + k

                    def place():
                        robot.reset()
                        robot.move_object(S.CUBE, position=[x, y, e.cube_z])

                    place()
                    kf = X.solve_keyframes(robot, e, np.array([x, y]), a.task)
                    place()
                    pol = X.ScriptedReplayPolicy(hz=e.control_hz)
                    pol.set_robot_state_keys(list(e.action_keys))
                    pol.load_plan(kf, a.task)
                    if a.action_noise > 0:
                        rng = np.random.default_rng(700000 + seed)
                        _t = pol.target_at
                        pol.target_at = (lambda step, _t=_t, rng=rng:
                                         np.asarray(_t(step), dtype=float)
                                         + rng.normal(0.0, a.action_noise, size=len(e.action_keys)))
                    st = {"gap": 1e9, "c0": None}

                    def obs(ev, st=st):
                        nm = type(ev).__name__
                        if nm == "RunPolicyStarted":
                            st["c0"] = S.cube_pos(robot)
                        elif nm == "RunPolicyStep":
                            st["gap"] = min(st["gap"], S.cube_surface_gap(robot, e))

                    robot.run_policy(robot_name=e.name, policy_object=pol, instruction="probe",
                                     duration=X.EPISODE_S, control_frequency=e.control_hz,
                                     n_episodes=1, reset_between=True, seed=seed, observer=obs,
                                     fast_mode=True)
                    c1 = S.cube_pos(robot)
                    c0 = st["c0"] if st["c0"] is not None else c1
                    disp = float(np.dot(c1 - c0, np.array([e.approach[0], e.approach[1], 0.0])))
                    if X.success(a.task, st["gap"] <= X.TOUCH_GAP, disp):
                        ok += 1
                cells.append({"ix": i, "iy": j, "x": round(float(x), 4), "y": round(float(y), 4),
                              "radius": round(float(np.hypot(x, y)), 4),
                              "n": a.episodes, "ok": ok, "rate": round(ok / a.episodes, 4)})
                print(json.dumps(cells[-1]), flush=True)
    finally:
        robot.destroy()

    def marg(key):
        out = {}
        for c in cells:
            out.setdefault(c[key], []).append(c["rate"])
        return {k: round(float(np.mean(v)), 4) for k, v in sorted(out.items())}

    mx, my = marg("ix"), marg("iy")
    sx = round(max(mx.values()) - min(mx.values()), 4)
    sy = round(max(my.values()) - min(my.values()), 4)
    res = {"arm": a.arm, "task": a.task, "action_noise_rad": a.action_noise, "bins": a.bins,
           "episodes_per_cell": a.episodes, "cube_box": e.cube_box,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "grid_mean_rate": round(float(np.mean([c["rate"] for c in cells])), 4),
           "marginal_x_forward": mx, "marginal_y_lateral": my,
           "spread_x_forward": sx, "spread_y_lateral": sy,
           "predicted_shortcut_axis": ("x_forward_radius" if sx > sy else
                                       "y_lateral" if sy > sx else "tie"),
           "predicted_flat": bool(max(sx, sy) < 0.25),
           "flat_threshold_note": ("predicted_flat uses a 0.25 marginal-spread threshold, fixed in "
                                   "PREREG-it28.md before any grid was run"),
           "cells": cells}
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("arm", "task", "grid_mean_rate", "spread_x_forward",
                                          "spread_y_lateral", "predicted_shortcut_axis",
                                          "predicted_flat")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
