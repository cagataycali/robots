"""Capture the diagnostic verdicts + one real MuJoCo grounding render, per tree."""
import json, logging, math, os, pathlib, sys
import numpy as np, torch

import strands_robots.policies.lerobot_local.policy as pm
TREE = str(pathlib.Path(pm.__file__).parents[3])
print("TREE:", TREE)

from strands_robots.policies.lerobot_local.embodiment import ZeroActionMonitor, load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
NAN, INF = float("nan"), float("inf")

class Cap(logging.Handler):
    def __init__(self): super().__init__(); self.msgs = []
    def emit(self, r): self.msgs.append(r.getMessage())

def run_stream(vec, keys, steps=12):
    p = LerobotLocalPolicy(); p.set_robot_state_keys(keys)
    h = Cap(); lg = logging.getLogger("strands_robots.policies.lerobot_local.policy")
    lg.addHandler(h); prev = lg.level; lg.setLevel(logging.WARNING)
    dicts = []
    try:
        for _ in range(steps):
            dicts.append(p._tensor_to_action_dicts(torch.tensor(vec, dtype=torch.float32)))
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    return h.msgs, dicts[0][0]

def classify(msgs):
    if not msgs: return "silent"
    m = msgs[0]
    if "non-finite action" in m: return "non-finite"
    if "near-zero actions" in m: return "near-zero"
    return "other"

KEYS = ["1", "2", "3", "4", "5", "6"]
STREAMS = [
    ("dead policy: 0.0 x 6", [0.0] * 6, "near-zero"),
    ("poisoned: nan x 6", [NAN] * 6, "non-finite"),
    ("poisoned: inf x 6", [INF] * 6, "non-finite"),
    ("one nan, five real commands", [0.4, -0.3, NAN, 0.2, 0.1, -0.5], "non-finite"),
    ("healthy: 0.9 x 6", [0.9] * 6, "silent"),
]

facts = {"tree": TREE, "streams": [], "ctor": [], "rollout": {}}
for label, vec, truth in STREAMS:
    msgs, first = run_stream(vec, KEYS)
    facts["streams"].append({
        "label": label, "truth": truth, "reported": classify(msgs),
        "message": msgs[0] if msgs else "", "n_warnings": len(msgs),
    })

CTOR = [("threshold=nan", {"threshold": NAN}), ("threshold=inf", {"threshold": INF}),
        ("threshold=True", {"threshold": True}), ("patience=nan", {"patience": NAN}),
        ("patience=inf", {"patience": INF}), ("patience=2.7", {"patience": 2.7}),
        ("threshold=1e-3 (default)", {"threshold": 1e-3}), ("patience=10 (default)", {"patience": 10})]
for label, kw in CTOR:
    try:
        m = ZeroActionMonitor(**kw)
    except Exception as e:
        facts["ctor"].append({"label": label, "verdict": "refused", "detail": str(e)}); continue
    healthy = sum(1 for _ in range(30) if m.update(0.9)); m.reset()
    dead = sum(1 for _ in range(30) if m.update(0.0))
    facts["ctor"].append({"label": label, "verdict": "accepted",
                          "detail": f"warns on a MOVING arm: {healthy}   warns on a DEAD policy: {dead}"})

# ---- real MuJoCo rollout: the honored path, driven by the emitted action dict ----
from strands_robots import create_simulation
sim = create_simulation("mujoco")
try:
    sim.create_world()
    sim.add_robot("so101")
    sim.add_camera(name="look", position=[0.42, -0.44, 0.30], target=[0.0, 0.0, 0.12], fov=42)
    emb = load_embodiment("so101")
    joints = emb.action_keys

    def read():
        obs = sim.get_observation("so101", skip_images=True)
        return {k: float(np.ravel(obs[k])[0]) for k in joints}

    before = read()
    rad = emb.model_action_to_sim([30.0, 30.0, 30.0, 30.0, 30.0, 50.0])
    _, emitted = run_stream(rad, joints, steps=1)
    for _ in range(20):
        sim.send_action(emitted, robot_name="so101", n_substeps=10)
    after = read()
    r = sim.render(camera_name="look", width=760, height=620)
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / "rollout.png").write_bytes(png)
    facts["rollout"] = {
        "emitted_action": {k: float(v) for k, v in emitted.items()},
        "joints_after": {k: round(after[k], 12) for k in joints},
        "max_delta_deg": round(max(abs(after[k] - before[k]) for k in joints) * 180.0 / math.pi, 6),
        "png_bytes": len(png),
    }
finally:
    sim.cleanup()

(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print("WROTE", OUT)
print(json.dumps({k: v for k, v in facts.items() if k != "streams"}, indent=2)[:900])
