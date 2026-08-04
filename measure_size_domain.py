"""Measure how each candidate run size is treated, on whichever tree runs this."""
import json, math, sys
from pathlib import Path
import strands_robots.tools.lerobot_train as tool_mod

TREE = str(Path(tool_mod.__file__).parents[2])
BUILD = tool_mod.build_train_command
BASE = dict(dataset_root="/data/cubes", policy_type="act")

# What lerobot does with the token, measured separately and quoted here.
CONSEQUENCE = {
    "0":     "parses as int 0 -> range(0,0) empty: TRAINS NOTHING",
    "-5":    "parses as int -5 -> range(0,-5) empty: TRAINS NOTHING",
    "True":  "parses as int 1: a silent run of ONE",
    "2.7":   "DecodingError in the detached process, minutes later",
    "nan":   "DecodingError in the detached process, minutes later",
    "inf":   "DecodingError in the detached process, minutes later",
    "'20000'": "parses as 20000 (an accident of the string form)",
}
CONSEQUENCE_BS = {
    "0":    "DataLoader(batch_size=0): ValueError in the detached process",
    "-8":   "DataLoader(batch_size=-8): ValueError in the detached process",
    "True": "parses as int 1: a silent batch of ONE",
    "2.7":  "DecodingError in the detached process, minutes later",
}

def verdict(param, value):
    kw = dict(BASE); kw[param] = value
    try:
        cmd = BUILD(**kw)
    except ValueError as e:
        return {"v": "refused", "detail": str(e)}
    except BaseException as e:
        return {"v": "raised", "detail": f"{type(e).__name__}: {e}"}
    tok = [c for c in cmd if c.startswith(f"--{param}=")]
    return {"v": "accepted", "detail": tok[0] if tok else f"(--{param} omitted)"}

CASES = [
    ("steps", 0), ("steps", -5), ("steps", True), ("steps", 2.7),
    ("steps", float("nan")), ("steps", float("inf")),
    ("batch_size", 0), ("batch_size", -8), ("batch_size", True), ("batch_size", 2.7),
    ("steps", 20000), ("batch_size", 8), ("steps", None), ("save_freq", 0),
]
rows = []
for param, value in CASES:
    label = repr(value)
    src = CONSEQUENCE if param == "steps" else CONSEQUENCE_BS
    rows.append({
        "param": param, "label": label,
        "consequence": src.get(label, ""),
        **verdict(param, value),
    })

# No-regression: the argv a usable call builds, in full.
control_argv = BUILD(**BASE, steps=20000, batch_size=8, save_freq=5000, output_dir="/data/out")
out = {"tree": TREE, "rows": rows, "control_argv": control_argv}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1], "tree", TREE)
