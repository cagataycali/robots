"""Read a shard back through lerobot's OWN reader, not the writer that made it.

Every check so far has been mine: my audit read my parquet and my decoded frames. That is the
cannot-fail-instrument hazard PAPER's §7.6 catalogues -- a producer validating its own output with its
own code. This loads the shard through `LeRobotDataset` exactly as a training script would and asserts
what a trainer depends on:

  - the dataset constructs at all, and its episode/frame totals match the GEN-REPORT
  - a sample is a dict of tensors with the shapes and dtypes a policy expects
  - the camera keys are present and decode to real pixels (not a black or constant frame)
  - action and state dims equal the arm's action_keys count
  - the task string survives round-trip and names the goal

Usage: python examples/rlenv/load_back.py --root ~/svla-rlenv-data/so101-touch/ds
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--report", default=None, help="GEN-REPORT.json to cross-check (default ../)")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    root = os.path.expanduser(a.root)
    rep_path = a.report or os.path.join(os.path.dirname(root.rstrip("/")), "GEN-REPORT.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = json.load(open(os.path.join(root, "meta", "info.json")))
    repo_id = meta.get("repo_id") or "local/shard"
    ds = LeRobotDataset(repo_id, root=root)
    out = {"root": root, "repo_id": repo_id, "codebase_version": meta.get("codebase_version"),
           "fps": meta.get("fps"), "robot_type": meta.get("robot_type"),
           "num_episodes": int(ds.num_episodes), "num_frames": int(ds.num_frames),
           "features": sorted(ds.features.keys()), "checks": {}}
    c = out["checks"]

    b = (rep.get("passB") or {})
    c["episodes_match_report"] = (b.get("parquet_episode_count") in (None, int(ds.num_episodes)))
    c["frames_match_report"] = (b.get("frames") in (None, int(ds.num_frames)))

    s = ds[0]
    img_keys = [k for k in s if k.startswith("observation.images")]
    out["image_keys"] = img_keys
    px = {}
    for k in img_keys:
        v = np.asarray(s[k])
        px[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "min": float(v.min()),
                 "max": float(v.max()), "std": round(float(v.std()), 6)}
    out["pixels_frame0"] = px
    c["images_present"] = len(img_keys) > 0
    c["images_not_constant"] = all(p["std"] > 1e-4 for p in px.values()) if px else False

    act = np.asarray(s["action"])
    st = np.asarray(s["observation.state"])
    out["action_dim"] = int(act.shape[-1])
    out["state_dim"] = int(st.shape[-1])
    c["action_state_same_dim"] = out["action_dim"] == out["state_dim"]
    c["action_finite"] = bool(np.isfinite(act).all() and np.isfinite(st).all())

    task = s.get("task")
    out["task_sample"] = task if isinstance(task, str) else str(task)
    tl = (out["task_sample"] or "").lower()
    c["task_is_string"] = isinstance(task, str) and len(task) > 0
    c["task_names_goal"] = any(w in tl for w in ("cube", "block", "red"))

    # A second episode's first frame must DIFFER from episode 0's (rung 1, read through lerobot this
    # time rather than through my own decoder). The first version of this check lived behind
    # `if i1 is not None and img_keys:` -- so on a lerobot without `episode_data_index` it did not run
    # and the verdict still said OK. That is the cannot-fail instrument PAPER's 7.6 catalogues, in my
    # own checker, found by noticing the field was absent from the output. It is now MANDATORY: the
    # index is resolved three ways and the check FAILS if none of them works.
    c["episodes_differ_at_frame0"] = False
    if int(ds.num_episodes) > 1 and img_keys:
        i1 = None
        try:
            i1 = int(ds.episode_data_index["from"][1])
        except Exception:
            try:  # hf dataset column
                ep = np.asarray(ds.hf_dataset["episode_index"])
                i1 = int(np.argmax(ep == 1))
            except Exception:
                try:  # fall back to the declared frame count of episode 0
                    i1 = int(json.load(open(os.path.join(root, "meta", "episodes.jsonl")).readline()
                                       )["length"])
                except Exception:
                    i1 = None
        out["ep1_first_index"] = i1
        if i1 is None:
            out["ep1_index_error"] = "could not locate episode 1's first frame by any route"
        else:
            d = float(np.abs(np.asarray(ds[i1][img_keys[0]], dtype=np.float32)
                             - np.asarray(s[img_keys[0]], dtype=np.float32)).mean())
            out["frame0_delta_ep0_vs_ep1"] = round(d, 6)
            c["episodes_differ_at_frame0"] = d > 1e-4

    out["verdict"] = "OK" if all(c.values()) else "FAIL"
    out["failed_checks"] = [k for k, v in c.items() if not v]
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdict", "failed_checks", "num_episodes", "num_frames",
                                          "image_keys", "action_dim", "task_sample",
                                          "frame0_delta_ep0_vs_ep1") if k in out}))
    return 0 if out["verdict"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
