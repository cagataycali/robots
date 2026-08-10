"""Measure add_camera's pose path on one tree: render, verdicts, neutering."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import strands_robots
from strands_robots.simulation.mujoco import simulation as sim_mod
from strands_robots.simulation import Simulation

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[2]

PROBES = {
    "3-short": [0.5, 0.3],
    "4-long": [0.5, 0.3, 0.2, 0.1],
    "empty": [],
    "string": "abc",
    "scalar": 3,
    "bool": True,
    "nan": [float("nan"), 0.0, 0.0],
    "inf": [float("inf"), 0.0, 0.0],
    "non-numeric": ["a", 1.0, 2.0],
    "mapping": {"x": 1},
    "beyond float64": [10**400, 0.0, 0.0],
    "0-d array": np.array(1.0),
    "list (usable)": [0.42, -0.38, 0.30],
    "NumPy (usable)": np.array([0.42, -0.38, 0.30]),
    "omitted": None,
}

def png_of(sim, cam):
    r = sim.render(camera_name=cam, width=680, height=560)
    if r.get("status") != "success":
        raise RuntimeError(f"render failed: {r}")
    return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

# ---- 1. verdicts for every probe, on both pose parameters ----------------
verdicts = {}
for key, val in PROBES.items():
    for param in ("position", "target"):
        s = Simulation(tool_name="art", mesh=False); s.create_world()
        kw = {"position": [0.42, -0.38, 0.30], "target": [0.0, 0.0, 0.12]}
        kw[param] = val
        if val is None:
            kw.pop(param)
        r = s.add_camera("cam", **kw)
        txt = next((c["text"] for c in r.get("content", []) if "text" in c), "")
        verdicts[f"{key}|{param}"] = [r.get("status"), txt[:150], "cam" in s._world.cameras]
        s.cleanup()

# ---- 2. is a second application present, and does it change anything? ----
src = pathlib.Path(sim_mod.__file__).read_text()
import ast
fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "add_camera")
rule_calls = [c.func.id for c in ast.walk(fn)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
              and c.func.id in ("coerce_pose_vector", "pose_vector_error")]
neuter = {"applications": len(rule_calls), "names": sorted(set(rule_calls))}
if "pose_vector_error" in rule_calls:
    calls = []
    orig = sim_mod.pose_vector_error
    sim_mod.pose_vector_error = lambda *a, **k: (calls.append(1), None)[1]
    after = {}
    for key, val in PROBES.items():
        for param in ("position", "target"):
            s = Simulation(tool_name="art", mesh=False); s.create_world()
            kw = {"position": [0.42, -0.38, 0.30], "target": [0.0, 0.0, 0.12]}
            kw[param] = val
            if val is None:
                kw.pop(param)
            r = s.add_camera("cam", **kw)
            txt = next((c["text"] for c in r.get("content", []) if "text" in c), "")
            after[f"{key}|{param}"] = [r.get("status"), txt[:150], "cam" in s._world.cameras]
            s.cleanup()
    sim_mod.pose_vector_error = orig
    neuter["invoked"] = len(calls)
    neuter["cases"] = len(after)
    neuter["changed"] = sum(1 for k in after if after[k] != verdicts[k])

# ---- 3. a real render through the surviving path -------------------------
s = Simulation(tool_name="art_render", mesh=False)
s.create_world()
s.add_robot(name="so100")
add = s.add_camera("look", position=np.array([0.42, -0.38, 0.30]), target=[0.0, 0.0, 0.12], fov=42)
assert add["status"] == "success", add
s.step(60)
png = png_of(s, "look")
(OUT / f"render_{TAG}.png").write_bytes(png)
import io
from PIL import Image
arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
np.save(OUT / f"render_{TAG}.npy", arr)
stored = s._world.cameras["look"].position
degenerate = s.add_camera("zero", position=[0.0, 0.0, 0.0])
s.cleanup()

json.dump({
    "tree": TREE, "tag": TAG, "verdicts": verdicts, "neuter": neuter,
    "add_status": add["status"],
    "stored_position": [float(v) for v in stored],
    "stored_types": sorted({type(v).__name__ for v in stored}),
    "degenerate_status": degenerate.get("status"),
    "degenerate_text": next((c["text"] for c in degenerate.get("content", []) if "text" in c), "")[:150],
    "render_shape": list(arr.shape),
}, open(OUT / f"facts_{TAG}.json", "w"), indent=2, default=str)
print("TREE:", TREE, "| tag:", TAG, "| neuter:", neuter)
