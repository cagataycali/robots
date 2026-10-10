"""Can a negative be made INDEPENDENT of the goal by degrading the expert? Measured on the label alone.

it22: a seed-matched FAIL shard's label is 85-94% readable from the cube pose at frame 0.
it23: post-hoc goal matching cannot repair that (no goal-matched success exists at the scale that
decides the label, and matching with reuse just moves the shortcut into the arm's start pose).

The remaining fix is to make failure a property of the POLICY: run the same seeds (hence the same
goal distribution) with a degraded expert, so which episodes fail is decided by noise rather than by
where the cube is. That is a claim about labels, not about pixels, so it needs no recorded shard:
pass-A alone gives (cube_xy, ok) per seed, and the test is whether the goal still predicts ok.

Per arm/task/sigma: failure rate, goal AUC (single features + 5-fold logistic CV, held-out folds),
and a label-permuted null. Decoupling means the AUC sits inside its own permuted spread WHILE the
failure rate stays usable.

Usage: python examples/rlenv/noise_decouple.py --arm so100 --task push --sigmas 0,0.02,0.05 \
          --seeds 200 --out runs/noise-decouple-so100-push.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from start_separability import auc, cv_auc  # noqa: E402  (one definition of the measurement)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--task", default="push", choices=("touch", "push"))
    p.add_argument("--sigmas", default="0,0.02,0.05")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--jitter", type=float, default=0.10)
    p.add_argument("--cam", type=int, default=128)
    p.add_argument("--perms", type=int, default=20)
    p.add_argument("--min-class", type=int, default=20,
                   help="minimum minority-class size before a decoupling verdict is allowed")
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    import harvest as H  # the SAME episode function the shards were generated with
    from strands_vla import expert as X
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    e = EMBODIMENTS[a.arm]
    robot, emb = S.build(a.arm, cam_size=(a.cam, a.cam), with_wrist=bool(e.wrist_parent))
    cams = tuple(c for c in ("scene", "wrist") if c == "scene" or e.wrist_parent)
    # identical seed stream to harvest.py, so these labels are the shards' labels
    seeds = [int(x) for x in np.random.default_rng(a.seed).integers(0, 2**31 - 1, size=a.seeds)]

    res = {"arm": a.arm, "task": a.task, "seeds": a.seeds, "jitter": a.jitter,
           "perms": a.perms,
           "caveat": "sigma>0 pushes some joint targets outside the actuator ctrlrange and MuJoCo "
                     "clamps them (the sim says so on stderr), so the EFFECTIVE noise is smaller "
                     "and asymmetric near a joint limit -- sigma is the commanded sd, not the "
                     "executed one",
           "cells": []}
    for sig in [float(x) for x in a.sigmas.split(",")]:
        args = types.SimpleNamespace(arm=a.arm, task=a.task, jitter=a.jitter, cam=a.cam,
                                     action_noise=sig, duration_s=None, band_scale=1.0,
                                     keep="failures")
        t0 = time.perf_counter()
        xy, ok = [], []
        for sd in seeds:
            r = H._episode(robot, emb, S, X, args, sd, cams, record=False)
            xy.append(r["cube_xy"])
            ok.append(bool(r["ok"]))
        xy = np.asarray(xy, dtype=float)
        y = 1.0 - np.asarray(ok, dtype=float)  # 1 == FAILURE, as in start_separability
        G = np.column_stack([xy[:, 0], xy[:, 1], np.abs(xy[:, 1]), np.hypot(xy[:, 0], xy[:, 1])])
        cell = {"sigma_rad": sig, "n": len(y), "failures": int(y.sum()),
                "failure_rate": round(float(y.mean()), 4), "secs": round(time.perf_counter() - t0, 1)}
        if 0 < y.sum() < len(y):
            cell["goal_auc"] = round(cv_auc(G, y), 4)
            ps = []
            for i in range(a.perms):
                ys = y.copy()
                np.random.default_rng(1000 + i).shuffle(ys)
                ps.append(cv_auc(G, ys, rng=np.random.default_rng(i)))
            cell["goal_auc_permuted_mean"] = round(float(np.mean(ps)), 4)
            cell["goal_auc_permuted_max"] = round(float(np.max(ps)), 4)
            # POWER GATE: with 2 failures in 200 the permuted null is itself ~0.3-0.8 wide, so
            # "inside the null" means "no data", not "no shortcut". A decoupling claim needs a
            # minority class big enough for the null to be narrow.
            inside = bool(abs(cell["goal_auc"] - 0.5) <= max(ps) - 0.5 + 1e-9)
            minority = int(min(y.sum(), len(y) - y.sum()))
            cell["minority_class_n"] = minority
            if minority < a.min_class:
                cell["decoupled"] = None
                cell["decoupled_note"] = (f"underpowered: minority class {minority} < {a.min_class}; "
                                          f"permuted null spans {min(ps):.3f}-{max(ps):.3f}")
            else:
                cell["decoupled"] = inside
            cell["single"] = {k: round(auc(y, G[:, j]), 4)
                              for j, k in enumerate(("cube_x", "cube_y", "abs_cube_y", "radius"))}
        else:
            cell["note"] = "one class only -- no label to predict"
        res["cells"].append(cell)
        print(json.dumps(cell), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"out": a.out, "cells": len(res["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
