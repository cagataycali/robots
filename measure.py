"""Measure both gravity surfaces' verdicts + what Newton actually stored."""
import json, sys, threading
import numpy as np
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.models import SimWorld

CASES = [("-3.7", -3.7), ("[0,0,-3.7]", [0.0, 0.0, -3.7]),
         ("True", True), ("False", False), ("[0,0,True]", [0.0, 0.0, True]),
         ("np.bool_(True)", np.True_),
         ("np.float32(-3.7)", np.float32(-3.7)), ("np.float64(-3.7)", np.float64(-3.7)),
         ("np.array([0,0,-3.7])", np.array([0.0,0.0,-3.7])),
         ("nan", float("nan")), ("[0,0]", [0.0,0.0]), ("'heavy'", "heavy")]

def newton(g):
    e = NewtonSimEngine.__new__(NewtonSimEngine)
    e._lock = threading.RLock()
    e._world = SimWorld(timestep=0.002, gravity=[0.0, 0.0, -9.81])
    e._model = object()
    rebuilt = []
    e._rebuild = lambda: rebuilt.append(1)
    try:
        r = NewtonSimEngine.set_gravity(e, g)
        st = r.get("status")
    except BaseException as exc:  # noqa: BLE001
        st = f"raised {type(exc).__name__}"
    return {"status": st, "applied": [float(v) for v in e._world.gravity], "rebuilt": len(rebuilt)}

def isaac(g):
    s = IsaacSimulation.__new__(IsaacSimulation)
    s._lock = threading.RLock(); s._world_created = True; s._config = IsaacConfig()
    try:
        r = IsaacSimulation.create_world(s, gravity=g)
        txt = " ".join(c.get("text","") for c in r.get("content",[]))
    except BaseException as exc:  # noqa: BLE001
        return {"cleared": False, "text": f"raised {type(exc).__name__}"}
    return {"cleared": "World already created" in txt, "text": txt}

out = {}
for label, val in CASES:
    out[label] = {"newton": newton(val), "isaac": isaac(val)}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: (v["newton"]["status"], v["newton"]["applied"], v["isaac"]["cleared"]) for k,v in out.items()}, indent=1))
