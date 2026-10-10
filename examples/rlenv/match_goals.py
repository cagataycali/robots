"""Remove the frame-0 shortcut from a negative split by MATCHING goals, and measure that it is gone.

it22 measured that 85-94% of the success/failure discrimination in these shards exists before the
policy acts: the cube pose alone gives AUC 0.853-0.942, because the failures are concentrated where
the sampler happened to put a hard goal. A consumer who trains a failure-aware objective on
positives+negatives can therefore score by reading the goal.

The fix that needs no new simulation: pair each failure with a SUCCESS AT (nearly) THE SAME CUBE
POSE and hand consumers that index instead of the raw shards. Optimal 1:1 assignment on Euclidean
cube distance (scipy linear_sum_assignment), pairs beyond --caliper dropped. If the balance works,
the goal-only AUC on the matched subset falls to its own permuted control; what survives is what the
arm DID, which is the thing the split was supposed to be about.

Reported per pair, all on held-out folds (same cv_auc as it22, imported, not re-written):
  * matched n, and the distance distribution of the pairs actually kept
  * goal / state / both AUC on the matched subset, each with a label-permuted control
  * the same AUCs on the unmatched shards, so the two sit side by side
  * delta = how much of the shortcut the matching removed

Writes MATCHED.jsonl per FAIL shard (one row per kept pair, by episode_index AND seed, so a consumer
can join it to either shard without trusting row order) and a run JSON.

Usage:
  python examples/rlenv/match_goals.py --out runs/match-goals.json [--caliper 0.005] [--write-index]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from start_separability import cv_auc, frame0, sidecar  # one definition of the measurement


def goalfeat(s):
    x, y = s.cube_x.to_numpy(), s.cube_y.to_numpy()
    return np.column_stack([x, y, np.abs(y), np.hypot(x, y)])


def aucs(G, S, y, perms, tag):
    out = {}
    for k, X in {"goal": G, "state": S, "both": np.hstack([G, S])}.items():
        a = round(cv_auc(X, y), 4)
        ps = []
        for i in range(perms):
            rng = np.random.default_rng(1000 + i)
            ys = y.copy()
            rng.shuffle(ys)
            ps.append(cv_auc(X, ys, rng=np.random.default_rng(i)))
        out[k] = {"auc": a, "permuted_mean": round(float(np.mean(ps)), 4),
                  "permuted_max": round(float(np.max(ps)), 4),
                  # "shortcut" = how far above its OWN null the feature set sits
                  "above_null": round(a - float(np.mean(ps)), 4)}
    out["_tag"] = tag
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.expanduser("~/svla-rlenv-data"))
    p.add_argument("--out", required=True)
    p.add_argument("--caliper", type=float, default=0.005, help="max cube distance of a kept pair, metres")
    p.add_argument("--calipers", default="0.005,0.01,0.02,0.04",
                   help="caliper curve to report (metres) -- how many pairs survive at each scale")
    p.add_argument("--with-replacement", action="store_true",
                   help="nearest success within the caliper, reuse allowed (1:1 with replacement)")
    p.add_argument("--perms", type=int, default=20)
    p.add_argument("--write-index", action="store_true", help="write MATCHED.jsonl into each FAIL shard")
    a = p.parse_args(argv)

    res = {"caliper_m": a.caliper, "perms": a.perms, "pairs": []}
    for neg in sorted(glob.glob(os.path.join(a.data, "*-FAIL"))):
        pos = neg[: -len("-FAIL")]
        if not os.path.isdir(pos):
            continue
        name = os.path.basename(pos)
        sp, sn = sidecar(pos), sidecar(neg)
        ep_p, st_p = frame0(pos)
        ep_n, st_n = frame0(neg)
        sp = sp.set_index("episode_index").loc[ep_p]
        sn = sn.set_index("episode_index").loc[ep_n]

        P = np.column_stack([sp.cube_x.to_numpy(), sp.cube_y.to_numpy()])
        F = np.column_stack([sn.cube_x.to_numpy(), sn.cube_y.to_numpy()])
        D = np.linalg.norm(F[:, None, :] - P[None, :, :], axis=2)  # (fail, success)
        # optimal 1:1 assignment, then drop every pair wider than the caliper
        def pair_at(cal, replace):
            if replace:
                # each failure takes its nearest success; a success may serve several failures
                c = D.argmin(1)
                r = np.arange(len(F))
            else:
                r, c = linear_sum_assignment(D)
            k = D[r, c] <= cal
            return r[k], c[k]

        # the caliper CURVE: matching is only as good as the scale at which the poses agree
        curve = []
        for cal in [float(x) for x in a.calipers.split(",")]:
            for rep in (False, True):
                r_, c_ = pair_at(cal, rep)
                dd = D[r_, c_]
                curve.append({"caliper_mm": round(cal * 1000, 2), "with_replacement": rep,
                              "n_pairs": int(len(r_)),
                              "fraction_of_fail": round(float(len(r_) / len(F)), 3),
                              "distinct_successes_used": int(len(set(c_.tolist()))),
                              "median_mm": round(float(np.median(dd)) * 1000, 3) if len(dd) else None})
        ri, ci = pair_at(a.caliper, a.with_replacement)
        d = D[ri, ci]
        # how close ANY success is to each failure -- the density limit matching cannot beat
        nn_mm = D.min(1) * 1000

        row = {"shard": name, "n_success": int(len(sp)), "n_fail": int(len(sn)),
               "n_matched": int(len(ri)),
               "matched_fraction_of_fail": round(float(len(ri) / len(sn)), 3),
               "pair_distance_mm": {
                   "median": round(float(np.median(d)) * 1000, 3) if len(d) else None,
                   "p90": round(float(np.quantile(d, 0.9)) * 1000, 3) if len(d) else None,
                   "max": round(float(d.max()) * 1000, 3) if len(d) else None},
               "with_replacement": bool(a.with_replacement),
               "caliper_curve": curve,
               "nearest_success_mm": {"median": round(float(np.median(nn_mm)), 3),
                                      "p10": round(float(np.quantile(nn_mm, 0.1)), 3),
                                      "p90": round(float(np.quantile(nn_mm, 0.9)), 3)},
               "unmatched_min_distance_mm": round(float(D[~np.isin(np.arange(len(F)), ri)].min(1).min()) * 1000, 3)
               if len(ri) < len(F) else None}

        # before: every episode of both shards (it22's cells, recomputed here so the comparison is one run)
        y_all = np.r_[np.zeros(len(sp)), np.ones(len(sn))]
        row["unmatched"] = aucs(np.vstack([goalfeat(sp), goalfeat(sn)]),
                                np.vstack([st_p, st_n]), y_all, a.perms, "all episodes")
        # after: only the matched pairs
        if len(ri) >= 20:
            spm, snm = sp.iloc[ci], sn.iloc[ri]
            y_m = np.r_[np.zeros(len(spm)), np.ones(len(snm))]
            row["matched"] = aucs(np.vstack([goalfeat(spm), goalfeat(snm)]),
                                  np.vstack([st_p[ci], st_n[ri]]), y_m, a.perms, "matched pairs")
            row["goal_auc_delta"] = round(row["matched"]["goal"]["auc"] - row["unmatched"]["goal"]["auc"], 4)
            # A matched design is only admissible if it passes ITS OWN criteria. Writing an index that
            # fails them would hand a consumer a "balanced" split that is still readable at frame 0 --
            # worse than the raw shards, because the balance is a claim.
            m = row["matched"]
            crit = {
                "goal_auc_within_its_null": bool(m["goal"]["auc"] <= m["goal"]["permuted_max"]
                                                 and m["goal"]["auc"] >= 1 - m["goal"]["permuted_max"]),
                "no_other_frame0_leak": bool(m["state"]["auc"] <= m["state"]["permuted_max"]
                                             and m["state"]["auc"] >= 1 - m["state"]["permuted_max"]),
                "joint_within_its_null": bool(m["both"]["auc"] <= m["both"]["permuted_max"]
                                              and m["both"]["auc"] >= 1 - m["both"]["permuted_max"]),
                # reusing 11 controls for 100 cases is not a balanced design, whatever the AUC says
                "controls_not_degenerate": bool(len(set(ci.tolist())) >= 0.5 * len(ri)),
                "pairs_tight_enough": bool(np.median(d) <= 0.005),
            }
            row["admissible"] = all(crit.values())
            row["criteria"] = crit
            if a.write_index and row["admissible"]:
                idx = os.path.join(neg, "MATCHED.jsonl")
                with open(idx, "w") as fh:
                    for k in range(len(ri)):
                        fh.write(json.dumps({
                            "fail_episode_index": int(snm.index[k]),
                            "fail_seed": int(snm.seed.iloc[k]),
                            "success_shard": name,
                            "success_episode_index": int(spm.index[k]),
                            "success_seed": int(spm.seed.iloc[k]),
                            "cube_distance_mm": round(float(d[k]) * 1000, 4),
                            "fail_cube": [round(float(snm.cube_x.iloc[k]), 5), round(float(snm.cube_y.iloc[k]), 5)],
                            "success_cube": [round(float(spm.cube_x.iloc[k]), 5), round(float(spm.cube_y.iloc[k]), 5)],
                        }) + "\n")
                row["index"] = idx
            elif a.write_index:
                # leave the refusal where the next reader of the shard will find it
                stale = os.path.join(neg, "MATCHED.jsonl")
                if os.path.exists(stale):
                    os.remove(stale)
                    row["removed_stale_index"] = stale
                p_ = os.path.join(neg, "MATCHED-REJECTED.md")
                with open(p_, "w") as fh:
                    fh.write("# No goal-matched index for this shard\n\n"
                             "Goal matching was attempted (`examples/rlenv/match_goals.py`, caliper "
                             f"{a.caliper * 1000:.0f} mm, with_replacement={bool(a.with_replacement)}) and "
                             "REFUSED: a matched design that is still readable at frame 0 is worse than the raw\n"
                             "shards, because the balance would be a claim.\n\n"
                             f"- pairs kept: {len(ri)} of {len(sn)} failures, "
                             f"{len(set(ci.tolist()))} distinct successes used, median pair distance "
                             f"{np.median(d) * 1000:.2f} mm\n"
                             f"- frame-0 AUC after matching (held-out folds, permuted null in brackets): "
                             f"goal {m['goal']['auc']:.3f} [{m['goal']['permuted_mean']:.3f}], "
                             f"state {m['state']['auc']:.3f} [{m['state']['permuted_mean']:.3f}], "
                             f"both {m['both']['auc']:.3f} [{m['both']['permuted_mean']:.3f}]\n"
                             f"- criteria: " + json.dumps(crit) + "\n\n"
                             "Use this shard as negatives only with a goal-only baseline reported alongside "
                             "(see lanes/RLENV/NEGATIVES.md).\n")
                row["rejection_note"] = p_
        else:
            row["matched"] = None
            row["note"] = f"only {len(ri)} pairs inside the caliper -- too few to score"
        res["pairs"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("unmatched", "caliper_curve")})[:700], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"pairs": len(res["pairs"]), "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
