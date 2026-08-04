"""Measure how each backend answers an unusable process count. Run in two trees."""
import json, math, pathlib, sys
import strands_robots.training.base as _b
TREE = str(pathlib.Path(_b.__file__).parents[2])
from strands_robots.training.base import TrainSpec
from strands_robots.training import create_trainer

DS = pathlib.Path("/tmp/mt_ds"); (DS/"meta").mkdir(parents=True, exist_ok=True)
(DS/"meta"/"info.json").write_text(json.dumps({"total_episodes": 10}))

UNUSABLE = [("0",0),("-4",-4),("True",True),("nan",float("nan")),
            ("2.7",2.7),("inf",float("inf")),("'4'","4"),("None",None),("[2]",[2])]
CONTROLS = [("1",1),("8",8)]
FIELDS = ("num_gpus","num_nodes")
PROVS  = ("lerobot_local","groot","cosmos3")

def spec(**kw):
    d = dict(dataset_root=str(DS), output_dir="/tmp/mt_out", base_model="lerobot/act",
             embodiment="new_embodiment", steps=100, global_batch_size=8)
    d.update(kw); return TrainSpec(**d)

def classify(prov, field, val):
    t = create_trainer(prov); s = spec(**{field: val})
    try:
        problems = t.validate(s)
    except BaseException as e:
        return "raised", f"{type(e).__name__}: {e}"
    named = [p for p in problems if field in p]
    if not named:
        return "accepted", ""
    if any("must be a positive integer" in p for p in named):
        return "refused (domain)", named[0]
    return "refused (support)", named[0]

out = {"tree": TREE, "cells": {}, "controls": {}, "facts": {}}
for prov in PROVS:
    for field in FIELDS:
        for label, val in UNUSABLE:
            k, d = classify(prov, field, val)
            out["cells"][f"{prov}|{field}|{label}"] = {"verdict": k, "detail": d[:120]}
        for label, val in CONTROLS:
            k, d = classify(prov, field, val)
            out["controls"][f"{prov}|{field}|{label}"] = {"verdict": k, "detail": d[:120]}

# what the selector / clamp / launcher do with each value (tree-independent facts)
for label, val in UNUSABLE:
    f = {}
    try: f["gt1"] = bool(val > 1)
    except BaseException as e: f["gt1"] = f"raises {type(e).__name__}"
    try: f["max1"] = repr(max(1, val))
    except BaseException as e: f["max1"] = f"raises {type(e).__name__}"
    try:
        from torch.distributed.launcher.api import LaunchConfig
        LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=val)
        f["launchconfig"] = "accepts"
    except BaseException as e:
        f["launchconfig"] = f"{type(e).__name__}"
    out["facts"][label] = f

pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
n_acc = sum(1 for v in out["cells"].values() if v["verdict"] == "accepted")
n_raise = sum(1 for v in out["cells"].values() if v["verdict"] == "raised")
print(f"TREE={TREE}  cells={len(out['cells'])}  accepted={n_acc}  raised={n_raise}")
