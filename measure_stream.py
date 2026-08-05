"""Stream a locally-cached dataset and record what each numeric knob yields.

Run against two trees; every row is the real outcome of the public call, and
each dump records the tree it measured so a before/after pair cannot be two
runs of the same code.
"""
from __future__ import annotations
import json, pathlib, sys, time
import strands_robots.streaming_dataset as sd

TREE = str(pathlib.Path(sd.__file__).parents[1])
ROOT = "/home/cagatay/.cache/huggingface/lerobot/lerobot/pusht"
FPS = 10                                            # pusht
ON_GRID = {"observation.state": [-0.2, -0.1, 0.0]}  # multiples of 1/10
OFF_GRID = {"observation.state": [0.0, -0.0167]}    # not a multiple of 1/10
LIMIT = 40


def one(label, knob, deltas, validate, **kw):
    rec = {"label": label, "knob": knob}
    t0 = time.time()
    try:
        r = sd.StreamingDatasetReader.open(
            "lerobot/pusht", root=ROOT, drop_videos=True,
            delta_timestamps=dict(deltas), validate_deltas=validate,
            shuffle=False, **kw)
        rec["open"] = "success"
        rec["open_s"] = round(time.time() - t0, 2)
        rec["num_shards"] = int(getattr(r.dataset, "num_shards", -1))
        n = 0
        try:
            for _ in r:
                n += 1
                if n >= LIMIT:
                    break
            rec["frames"], rec["iter"] = n, "ok"
        except BaseException as e:  # noqa: BLE001 - the failure IS the measurement
            rec["frames"], rec["iter"] = n, f"{type(e).__name__}: {str(e)[:80]}"
    except BaseException as e:  # noqa: BLE001
        rec["open"] = f"{type(e).__name__}: {str(e)[:150]}"
        rec["open_s"] = round(time.time() - t0, 2)
        rec["frames"], rec["iter"] = None, "not reached"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


CASES = [
    # (label, knob, deltas, validate_deltas, kwargs)
    ("defaults", "-", ON_GRID, True, {}),
    ("max_num_shards=0", "max_num_shards", ON_GRID, True, dict(max_num_shards=0)),
    ("max_num_shards=-5", "max_num_shards", ON_GRID, True, dict(max_num_shards=-5)),
    ("buffer_size=0", "buffer_size", ON_GRID, True, dict(buffer_size=0)),
    ("seed=-1", "seed", ON_GRID, True, dict(seed=-1)),
    ("tolerance_s=inf, OFF-grid deltas", "tolerance_s", OFF_GRID, True, dict(tolerance_s=float("inf"))),
    ("tolerance_s=nan, on-grid deltas", "tolerance_s", ON_GRID, True, dict(tolerance_s=float("nan"))),
    ("tolerance_s=1e-4, OFF-grid deltas", "control", OFF_GRID, True, dict(tolerance_s=1e-4)),
]
out = {"tree": TREE, "fps": FPS, "limit": LIMIT,
       "cases": [one(l, k, d, v, **kw) for l, k, d, v, kw in CASES]}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("TREE:", TREE)
print(f"{'case':<34} {'open':<46} {'shards':>6} {'frames':>7}  iter")
for c in out["cases"]:
    print(f"{c['label']:<34} {c['open'][:46]:<46} {str(c.get('num_shards','-')):>6} "
          f"{str(c['frames']):>7}  {c['iter'][:44]}")
