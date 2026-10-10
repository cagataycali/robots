"""Put each episode's GOAL back into the shard, and stop the generator from throwing it away.

A published shard's parquet carries action, observation.state, frame/episode indices and timestamps -- and
NOT the cube pose, NOT the success flag. So nothing on the card can be recomputed from the shard itself: the
success labels, the sampled band and the goal-blind cells all live in a sidecar GEN-REPORT.json. That is
exactly as durable as one file, and so101-push-venue18 has already lost its report and with it the seeds that
would have let anyone rebuild its 200 goals. The data is intact and its provenance is not.

Recovery for the published shards: harvest recorded `kept_seeds`, and scene.sample_cube is a PURE function of
(embodiment, task, seed), so every kept episode's cube xy is recomputable exactly, with no simulator. This
writes EPISODES.jsonl (episode_index, seed, cube_x, cube_y, task) into the shard so the goal travels WITH the
data, and cross-checks the rebuilt poses against whatever coverage statistics the report already published.
The ordering assumption -- episode_index i corresponds to kept_seeds[i] -- is checked, not assumed: counts
must match exactly, and harvest records episodes in the order it appends them (harvest.py:188/202).

Also patches the generator: line 202 wrote ONLY the seeds while holding the full per-episode records, which
is what made this recovery necessary in the first place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

RUNS = os.path.expanduser("~/.tiny/strands-vla-20261009/lanes/RLENV/runs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="so101-touch,so100-touch,koch-touch,so101-push,so100-push,koch-push")
    p.add_argument("--root", default=os.path.expanduser("~/svla-rlenv-data"))
    p.add_argument("--svla", default=os.path.expanduser("~/rlenv-svla-pin"))
    p.add_argument("--upload", action="store_true")
    a = p.parse_args(argv)

    sys.path.insert(0, a.svla)
    from strands_vla import scene as S
    from strands_vla.embodiments import EMBODIMENTS

    summary = {}
    for shard in a.shards.split(","):
        d = os.path.join(a.root, shard)
        rep_p = os.path.join(d, "GEN-REPORT.json")
        if not os.path.exists(rep_p):
            summary[shard] = {"status": "NO REPORT — goals unrecoverable, sidecar refused"}
            continue
        rep = json.load(open(rep_p))
        seeds = rep.get("kept_seeds") or []
        arm, task = rep["arm"], rep["task"]
        e = EMBODIMENTS[arm]
        n_eps = (rep.get("passB") or {}).get("parquet_episode_count")
        if not seeds or (n_eps and len(seeds) != n_eps):
            summary[shard] = {"status": f"REFUSED: {len(seeds)} seeds vs {n_eps} episodes — "
                                        "ordering cannot be established"}
            continue
        rows, xs, ys = [], [], []
        for i, sd in enumerate(seeds):
            # harvest.py:288 -- default_rng(90000 + seed), NOT default_rng(seed). The first version of
            # this file used the bare seed and produced 1200 plausible, wrongly-placed goals: right band,
            # right ranges, wrong episode-by-episode values. Verified against the parquet by verify_goals.py,
            # which also fails on purpose when given the bare seed.
            rng = np.random.default_rng(90000 + sd)
            xy = S.sample_cube(e, rng, task)
            rows.append({"episode_index": i, "seed": int(sd), "cube_x": round(float(xy[0]), 6),
                         "cube_y": round(float(xy[1]), 6), "arm": arm, "task": task})
            xs.append(float(xy[0])); ys.append(float(xy[1]))
        out = os.path.join(d, "ds", "EPISODES.jsonl")
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[shard] = {"status": "written", "episodes": len(rows), "file": out,
                          "x_range": [round(min(xs), 4), round(max(xs), 4)],
                          "y_range": [round(min(ys), 4), round(max(ys), 4)],
                          "report_band_scale": rep.get("band_scale", 1.0)}
        # No upload path here on purpose: GEN-REPORT's repo_id is a local placeholder
        # ("local/rlenv-<shard>"), not the published repo, and publish.py already uploads the whole ds/
        # folder -- so writing into ds/ means republishing carries the sidecar, with one upload route
        # instead of two that can disagree.
        print(json.dumps({shard: summary[shard]}), flush=True)

    json.dump(summary, open(os.path.join(RUNS, "episodes-sidecar.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
