"""Measure the set_eval_seed(None) contract. Run in each tree; dumps JSON."""
import json, math, random, sys
from pathlib import Path

import strands_robots.simulation.policy_runner as pr
from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

TREE = str(Path(pr.__file__).parents[2])
CASES = [("None", None), ("2.5", 2.5), ("7", 7), ("MAX+1", MAX_EVAL_SEED + 1)]


def verdict(seed, without_torch):
    saved = sys.modules.get("torch", "<absent>")
    if without_torch:
        sys.modules["torch"] = None
    random.seed(0xC0FFEE)
    before = [random.random() for _ in range(3)]
    try:
        import numpy as np
        np.random.seed(0xC0FFEE)
        nb = np.random.random(3).tolist()
        np.random.seed(0xC0FFEE)
    except ImportError:
        np, nb = None, None
    random.seed(0xC0FFEE)
    try:
        pr.set_eval_seed(seed)
        kind, msg = "accepted", ""
    except ValueError as e:
        kind, msg = "refused", str(e)
    except BaseException as e:
        kind, msg = "raised", f"{type(e).__name__}: {e}"
    after = [random.random() for _ in range(3)]
    na = np.random.random(3).tolist() if np is not None else None
    if without_torch:
        if saved == "<absent>":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    return {"kind": kind, "msg": msg, "random_moved": before != after,
            "numpy_moved": (nb != na) if nb is not None else None}


out = {"tree": TREE, "with_torch": {}, "without_torch": {}, "domain": {}}
for label, seed in CASES:
    out["with_torch"][label] = verdict(seed, False)
    out["without_torch"][label] = verdict(seed, True)
    out["domain"][label] = randomization_seed_error(seed, "set_eval_seed", max_seed=MAX_EVAL_SEED)

# No-regression band: default-path messages + facade None + valid seeds.
out["default_msgs"] = {
    r: randomization_seed_error(v, "randomize") for r, v in
    [("2.5", 2.5), ("-1", -1), ("'42'", "42"), ("True", True), ("nan", math.nan)]
}
out["default_ceiling"] = randomization_seed_error(MAX_EVAL_SEED + 1, "run_policy", max_seed=MAX_EVAL_SEED)
out["facade_none_accepted"] = randomization_seed_error(None, "run_policy", max_seed=MAX_EVAL_SEED) is None
out["randomize_none_accepted"] = randomization_seed_error(None, "randomize") is None
reps = {}
for s in (0, 7, MAX_EVAL_SEED):
    pr.set_eval_seed(s); a = [random.random() for _ in range(2)]
    pr.set_eval_seed(s); reps[str(s)] = (a == [random.random() for _ in range(2)])
out["valid_seeds_reproducible"] = reps
print(json.dumps(out, indent=2))
