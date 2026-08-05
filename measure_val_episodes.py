"""Measure val_episodes on both trees. Prints its own tree so a pair is provable."""
import json, math, pathlib, sys, tempfile
import strands_robots.training.lerobot as _m
from strands_robots.tools.lerobot_train import build_train_command
from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

TREE = str(pathlib.Path(_m.__file__).parents[2])
TOTAL = 10
CASES = [3, 0, -5, True, 2.7, 0.5, float("nan"), "5", [5], None]
LABELS = ["3", "0", "-5", "True", "2.7", "0.5", "nan", "'5'", "[5]", "None"]
# The contract: a usable count (or None) is honored exactly; anything else is refused.
USABLE = {"3", "None"}

d = pathlib.Path(tempfile.mkdtemp()); ds = d / "ds"; (ds / "meta").mkdir(parents=True)
(ds / "meta" / "info.json").write_text(json.dumps({
    "codebase_version": "v3.0", "total_episodes": TOTAL, "total_tasks": 1,
    "total_frames": 1200, "fps": 30, "features": {}}))

def spec(v):
    return TrainSpec(dataset_root=str(ds), output_dir=str(d / "out"), base_model="lerobot/act",
                     embodiment="so101", steps=100, global_batch_size=8,
                     extra={"policy_type": "act"}, val_episodes=v)

rows = {}
for label, v in zip(LABELS, CASES):
    t, s = LerobotTrainer(), spec(v)
    r = {}
    try:
        probs = [p for p in t.validate(s) if ": val_episodes " in p]
        r["validate"] = "reports a problem" if probs else "no problem"
    except BaseException as e:  # an escape past the documented return IS the finding
        r["validate"] = f"raises {type(e).__name__}"
    # What the RUN does. train() fails closed on validate(), so model that
    # order rather than calling the pure argv-parity helper directly.
    if r["validate"] == "reports a problem":
        r["reserved"] = "refused before the run starts"
    else:
        try:
            flags = [c for c in t.build_command(s) if "eval_split" in c or "eval_steps" in c]
            if not flags:
                r["reserved"] = "no split, no eval pass"
            else:
                frac = float(next(f for f in flags if "eval_split" in f).split("=", 1)[1])
                n = "nan" if frac != frac else math.ceil(TOTAL * frac)
                r["reserved"] = f"{n} episode(s), split={frac:g}" if n != "nan" else "split=nan, evaluating"
        except BaseException as e:
            r["reserved"] = f"raises {type(e).__name__}"
    try:
        build_train_command(dataset_root=str(ds), policy_type="act", val_episodes=v)
        r["tool"] = "accepted"
    except ValueError as e:
        r["tool"] = "refused (names the field)" if "val_episodes must be" in str(e) else f"ValueError: {e}"[:44]
    except BaseException as e:
        r["tool"] = f"raises {type(e).__name__}"
    rows[label] = r

# One cell is "honoring the contract" when a usable value is honored exactly and
# an unusable one is refused with a message naming the field.
def ok(label, col, val):
    usable = label in USABLE
    if col == "validate":
        return val == ("no problem" if usable else "reports a problem")
    if col == "reserved":
        if label == "3":
            return val.startswith("3 episode")
        if label == "None":
            return val == "no split, no eval pass"
        return val == "refused before the run starts"
    return val == ("accepted" if usable else "refused (names the field)")

bad = sum(1 for lb, r in rows.items() for c, v in r.items() if not ok(lb, c, v))
out = {"tree": TREE, "total": TOTAL, "labels": LABELS, "usable": sorted(USABLE),
       "rows": rows, "cells": len(LABELS) * 3, "divergences": bad}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(f"TREE={TREE}  divergences={bad}/{len(LABELS)*3}")
