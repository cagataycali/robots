"""Measure everything the figure claims. Dumps one JSON."""
import json, pathlib, re, subprocess, sys, types
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW = "tests/policies/cosmos3/test_native_stack_absent.py"
FACTS = {}
OUT = pathlib.Path(f"/tmp/art-{sys.argv[1]}.json")

def save():
    OUT.write_text(json.dumps(FACTS, indent=2))

import strands_robots.policies.cosmos3.policy_diffusers as pd
FACTS["tree"] = str(pathlib.Path(pd.__file__).parents[3])
save()

# ---- 1. per-line coverage, before/after, from two real subset runs --------
def cov_run(extra):
    cmd = [sys.executable, "-m", "pytest", "tests/policies/cosmos3", "-q", "--no-header",
           "-p", "no:randomly", "--cov=strands_robots", "--cov-fail-under=0",
           f"--cov-report=json:/tmp/c-{sys.argv[1]}.json", *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    d = json.load(open(f"/tmp/c-{sys.argv[1]}.json"))["files"]["strands_robots/policies/cosmos3/policy_diffusers.py"]
    p = re.search(r"(\d+) passed", r.stdout)
    return sorted(d["missing_lines"]), round(d["summary"]["percent_covered"], 1), int(p.group(1))

FACTS["before"] = dict(zip(("missing", "pct", "passed"), cov_run([f"--ignore={NEW}"])))
save()
FACTS["after"] = dict(zip(("missing", "pct", "passed"), cov_run([])))
save()

# ---- 2. the four lines, labelled by decision -----------------------------
FACTS["lines"] = [
    {"line": 255, "fn": "_load_pipeline",        "decision": "refuse"},
    {"line": 286, "fn": "_import_condition_cls", "decision": "refuse"},
    {"line": 343, "fn": "_as_action_tensor",     "decision": "refuse"},
    {"line": 403, "fn": "_to_numpy",             "decision": "degrade"},
]
save()

# ---- 3. behavioural outcomes on a torch-free install ---------------------
from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.policies.cosmos3.policy_diffusers import _install_hint

class FakeCondition:
    def __init__(self, **kw): self.kwargs = kw

class DetachChunk:
    def __init__(self, a): self._a = a
    def detach(self): return self
    def cpu(self): return self._a
    def __array__(self, dtype=None): return np.asarray(self._a, dtype=dtype)

class FakePipeline:
    def __init__(self, a): self._a = a
    def __call__(self, **kw): return types.SimpleNamespace(action=self._a, video="world", sound=None)

def no_torch():
    saved = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}
    for k in list(saved): del sys.modules[k]
    sys.modules["torch"] = None
    return saved

def restore(saved):
    if sys.modules.get("torch") is None: del sys.modules["torch"]
    sys.modules.update(saved)

chunk = np.arange(4 * 10, dtype=np.float32).reshape(4, 10)
obs = {"prompt": "pick the cube", "observation/wrist_image_left": np.zeros((8, 8, 3), np.uint8)}
def backend(**kw):
    return Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"),
                                   pipeline=FakePipeline([DetachChunk(chunk)]),
                                   condition_cls=FakeCondition, **kw)

beh = {}
saved = no_torch()
try:
    try:
        a = backend().infer(obs)["action"]
        beh["policy_mode"] = {"outcome": "completed", "shape": list(a.shape),
                              "dtype": str(a.dtype), "chunk_exact": bool(np.array_equal(a, chunk))}
    except Exception as e:
        beh["policy_mode"] = {"outcome": f"{type(e).__name__}", "msg": str(e)[:60]}
    try:
        backend(mode="forward_dynamics").infer(obs, raw_actions=chunk)
        beh["forward_dynamics"] = {"outcome": "completed"}
    except ImportError as e:
        beh["forward_dynamics"] = {"outcome": "refused", "is_shared_hint": str(e) == _install_hint()}
finally:
    restore(saved)
a = backend().infer(obs)["action"]
beh["policy_mode_torch_present"] = {"outcome": "completed", "shape": list(a.shape),
                                    "chunk_exact": bool(np.array_equal(a, chunk))}
FACTS["behaviour"] = beh
FACTS["hint_names_extra"] = "strands-robots[cosmos3-diffusers]" in _install_hint()
FACTS["hint_names_service"] = "backend='service'" in _install_hint()
save()

# ---- 4. mutation rows (re-read the measured dump) ------------------------
FACTS["mutations"] = json.loads(pathlib.Path(f"/tmp/mut-{sys.argv[1]}.json").read_text())
save()
print(json.dumps(FACTS, indent=2)[:1600])
