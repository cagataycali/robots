"""Measure what a rollout gets from reseed_client_rngs, per tree.

Run inside the tree being measured so the editable install resolves to it.
"""
import json, math, pathlib, random, sys
import numpy as np
import strands_robots.policies._rng as rng_mod
from strands_robots.policies._rng import reseed_client_rngs

TREE = str(pathlib.Path(rng_mod.__file__).parents[2])

OUT_OF_DOMAIN = 2**32          # one past the legacy NumPy global RNG's range
IN_DOMAIN = 4242

def episode(seed):
    """Seed as a provider's reset() would, then draw from both streams."""
    status = "returned"
    try:
        reseed_client_rngs(seed)
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
    return status, [round(random.random(), 6) for _ in range(6)], [round(v, 6) for v in np.random.rand(6)]

def two_episodes(seed):
    """Two episodes seeded identically: a reproducible rollout matches."""
    random.seed(1); np.random.seed(1)
    s1, py1, np1 = episode(seed)
    random.seed(2); np.random.seed(2)   # a different inter-episode state
    s2, py2, np2 = episode(seed)
    return {"seed": repr(seed), "status": s1, "status2": s2,
            "py": [py1, py2], "np": [np1, np2],
            "py_reproducible": py1 == py2, "np_reproducible": np1 == np2}

facts = {"tree": TREE, "out_of_domain": two_episodes(OUT_OF_DOMAIN), "in_domain": two_episodes(IN_DOMAIN)}

# verdict sweep
sweep = {}
for s in [-1, 2.5, 3.0, True, "abc", math.nan, 2**32, 0, 7, 4294967295]:
    random.seed(999); np.random.seed(999)
    before = (random.getstate(), tuple(int(v) for v in np.random.get_state()[1][:6]))
    try:
        reseed_client_rngs(s); raised = None
    except Exception as e:
        raised = f"{type(e).__name__}"
    after = (random.getstate(), tuple(int(v) for v in np.random.get_state()[1][:6]))
    sweep[repr(s)] = {"raised": raised,
                      "py": "reseeded" if after[0] != before[0] else "untouched",
                      "np": "reseeded" if after[1] != before[1] else "untouched"}
facts["sweep"] = sweep
print(json.dumps(facts))
