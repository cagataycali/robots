"""Is a failure already distinguishable at frame 0? (the question a negative split cannot be used without)

Every shard here is fixed-horizon and every episode starts from the same reset pose plus jitter, so a
success shard and its FAIL sibling differ only in (a) where the cube was sampled and (b) what the arm then
did. If the frame-0 information ALONE separates the two classes, then a head trained on positives+negatives
can learn "this starting configuration is a failure" without modelling behaviour at all, and any
failure-aware objective built on that split is measuring the sampler, not the policy.

What it measures, per arm/task that has both shards:
  * seed integrity -- the two shards are drawn from one seed stream, so a seed may not appear in both
    (if one does, the success criterion is nondeterministic and both shards are suspect).
  * AUC of a logistic regression on frame-0 features, 5-fold stratified CV, three feature sets:
      goal   = cube_x, cube_y, |cube_y|, radius      (where the cube was sampled)
      state  = observation.state at frame 0           (the arm's jittered reset pose)
      both
  * a LABEL-PERMUTED control for each (same code path, labels shuffled) -- the only thing that says
    whether an AUC of 0.6 on n=300 means anything.
  * single-feature AUCs so a number that is real is also readable.

AUC is the Mann-Whitney statistic (P(score_fail > score_success)), computed on held-out folds only.

Usage: python examples/rlenv/start_separability.py --data ~/svla-rlenv-data --out runs/start-sep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def sidecar(d):
    p = os.path.join(d, "ds", "EPISODES.jsonl")
    if not os.path.exists(p):
        return None
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])


def frame0(d):
    fs = sorted(glob.glob(os.path.join(d, "ds", "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"])
                    for f in fs], ignore_index=True)
    df = df[df.frame_index == 0].sort_values("episode_index")
    st = np.stack(df["observation.state"].to_numpy())
    return df["episode_index"].to_numpy(), st


def auc(y, s):
    # Mann-Whitney U / (n1*n0) with ties at 0.5; y==1 is the FAILURE class
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X, y, iters=4000, lr=0.5, l2=1e-3):
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return w


def cv_auc(X, y, folds=5, rng=None):
    """Stratified k-fold; scores come only from the fold the model did not see."""
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(y))
    out = np.zeros(len(y))
    for cls in (0, 1):
        c = idx[y == cls]
        rng.shuffle(c)
        for k, part in enumerate(np.array_split(c, folds)):
            out[part] = -(k + 1)  # negative marks fold id
    fold = -out
    scores = np.zeros(len(y))
    for k in range(1, folds + 1):
        te = fold == k
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        w = fit_logistic((X[tr] - mu) / sd, y[tr])
        Z = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))])
        scores[te] = Z @ w
    return auc(y, scores)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/svla-rlenv-data"))
    p.add_argument("--out", required=True)
    p.add_argument("--perms", type=int, default=20)
    a = p.parse_args(argv)

    pairs = []
    for d in sorted(glob.glob(os.path.join(a.data, "*-FAIL"))):
        pos = d[:-len("-FAIL")]
        if os.path.isdir(pos):
            pairs.append((os.path.basename(pos), pos, d))

    res = {"pairs": [], "perms": a.perms}
    for name, pos, neg in pairs:
        sp, sn = sidecar(pos), sidecar(neg)
        ep_p, st_p = frame0(pos)
        ep_n, st_n = frame0(neg)
        # align the sidecar to the parquet by episode_index -- never by row order
        sp = sp.set_index("episode_index").loc[ep_p]
        sn = sn.set_index("episode_index").loc[ep_n]
        seeds_p, seeds_n = set(sp.seed.tolist()), set(sn.seed.tolist())
        inter = sorted(seeds_p & seeds_n)

        def goalfeat(s):
            x, y = s.cube_x.to_numpy(), s.cube_y.to_numpy()
            return np.column_stack([x, y, np.abs(y), np.hypot(x, y)])

        G = np.vstack([goalfeat(sp), goalfeat(sn)])
        S = np.vstack([st_p, st_n])
        y = np.r_[np.zeros(len(sp)), np.ones(len(sn))]
        sets = {"goal": G, "state": S, "both": np.hstack([G, S])}
        row = {"shard": name, "n_success": int(len(sp)), "n_fail": int(len(sn)),
               "seed_overlap": len(inter), "seed_overlap_examples": inter[:5],
               "auc": {}, "auc_permuted_mean": {}, "auc_permuted_max": {}, "single": {}}
        for k, X in sets.items():
            row["auc"][k] = round(cv_auc(X, y), 4)
            ps = []
            for i in range(a.perms):
                rng = np.random.default_rng(1000 + i)
                ys = y.copy()
                rng.shuffle(ys)
                ps.append(cv_auc(X, ys, rng=np.random.default_rng(i)))
            row["auc_permuted_mean"][k] = round(float(np.mean(ps)), 4)
            row["auc_permuted_max"][k] = round(float(np.max(ps)), 4)
        for j, fn in enumerate(["cube_x", "cube_y", "abs_cube_y", "radius"]):
            row["single"][fn] = round(auc(y, G[:, j]), 4)
        for j in range(S.shape[1]):
            row["single"][f"state[{j}]"] = round(auc(y, S[:, j]), 4)
        # AUC on a raw feature is blind to a NON-MONOTONE failure region (so101 touch fails in a y BAND,
        # both tails succeed), so also report the narrowest interval holding every failure and how much of
        # the success set shares it -- that is the readable form of "the sampler decided the label".
        row["fail_window"] = {}
        for fn, vs, vf in (("cube_x", sp.cube_x.to_numpy(), sn.cube_x.to_numpy()),
                           ("cube_y", sp.cube_y.to_numpy(), sn.cube_y.to_numpy())):
            lo, hi = float(vf.min()), float(vf.max())
            band = float(np.mean((vs >= lo) & (vs <= hi)))
            full = float(max(vs.max(), hi) - min(vs.min(), lo))
            row["fail_window"][fn] = {"lo_m": round(lo, 4), "hi_m": round(hi, 4),
                                      "width_m": round(hi - lo, 4),
                                      "fraction_of_sampled_range": round((hi - lo) / full, 3) if full else None,
                                      "success_fraction_inside": round(band, 3)}
        bx, by = row["fail_window"]["cube_x"], row["fail_window"]["cube_y"]
        inside = ((sp.cube_x.to_numpy() >= bx["lo_m"]) & (sp.cube_x.to_numpy() <= bx["hi_m"])
                  & (sp.cube_y.to_numpy() >= by["lo_m"]) & (sp.cube_y.to_numpy() <= by["hi_m"]))
        row["fail_window"]["joint_box_success_fraction_inside"] = round(float(inside.mean()), 3)
        # the plain-language version of the same thing: how far apart the class medians are
        row["cube_y_median_success_m"] = round(float(np.median(sp.cube_y)), 4)
        row["cube_y_median_fail_m"] = round(float(np.median(sn.cube_y)), 4)
        row["cube_y_range_success_m"] = [round(float(sp.cube_y.min()), 4), round(float(sp.cube_y.max()), 4)]
        row["cube_y_range_fail_m"] = [round(float(sn.cube_y.min()), 4), round(float(sn.cube_y.max()), 4)]
        res["pairs"].append(row)
        print(json.dumps(row)[:400], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"pairs": len(res["pairs"]), "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
