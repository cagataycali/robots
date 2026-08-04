"""Measure what the training preflight does with each run-size value.

Run in two trees (upstream/main and this branch); each dump records its own
tree so the two halves cannot be confused.
"""
import json, pathlib, sys, tempfile
import strands_robots.training.base as _b
TREE = str(pathlib.Path(_b.__file__).parents[2])
from strands_robots.training.base import TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer

TRAINERS = {"mock": MockTrainer, "cosmos3": Cosmos3Trainer, "groot": Gr00tTrainer, "lerobot": LerobotTrainer}
CASES = [("steps", v) for v in (10000, 0, -5, True, 2.7, float("nan"), float("inf"), "1000", None)] + \
        [("global_batch_size", v) for v in (32, 0, -8, True, 2.7, float("nan"), "32")]

tmp = tempfile.mkdtemp(); ds = pathlib.Path(tmp)/"ds"; (ds/"meta").mkdir(parents=True)
(ds/"meta"/"info.json").write_text(json.dumps({"total_episodes": 10, "fps": 30}))
out = pathlib.Path(tmp)/"out"; out.mkdir()

def spec(field, value):
    kw = dict(dataset_root=str(ds), base_model="lerobot/act", output_dir=str(out), embodiment="so101")
    kw[field] = value
    return TrainSpec(**kw)

rows = []
for field, value in CASES:
    per = {}
    for tname, T in TRAINERS.items():
        try:
            problems = T().validate(spec(field, value))
            named = [p for p in problems if field in p]
            per[tname] = "refused" if named else "accepted"
        except BaseException as e:
            per[tname] = f"raised {type(e).__name__}"
    rows.append({"field": field, "value": repr(value), "per": per,
                 "uniform": per["mock"] if len(set(per.values())) == 1 else "SPLIT"})

# The four copies of the rule, and whether the batch factor is checked anywhere.
tdir = pathlib.Path(_b.__file__).parent
copies = sum(p.read_text().count("spec.steps <= 0") for p in tdir.glob("*.py"))
gate = sum(p.read_text().count("_run_size_problems(spec)") for p in tdir.glob("*.py") if p.name != "base.py")
batch_checked = any("global_batch_size" in p.read_text() and "positive" in p.read_text() for p in tdir.glob("*.py"))
print(json.dumps({"tree": TREE, "rows": rows, "local_copies": copies,
                  "gate_call_sites": gate, "batch_factor_checked": batch_checked}, indent=1))
