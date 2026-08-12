"""Capture the scene a caller asked for, and what each tree told them instead."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import strands_robots
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE, flush=True)
TAG = sys.argv[1]
OUT = pathlib.Path(f"/tmp/art-{TAG}")
OUT.mkdir(exist_ok=True)
facts: dict = {"tree": TREE}

CAM = dict(name="look", position=[0.85, -0.80, 0.55], target=[0.0, 0.0, 0.18], fov=38)
W, H = 720, 640

def text(result: dict) -> str:
    return " ".join(b["text"] for b in result.get("content", []) if "text" in b)

def render(sim) -> np.ndarray:
    r = sim.render(camera_name="look", width=W, height=H)
    assert r.get("status") == "success", r
    png = next(b["image"]["source"]["bytes"] for b in r["content"] if "image" in b)
    import imageio.v3 as iio
    return np.asarray(iio.imread(png))

def fresh():
    sim = MuJoCoSimEngine(tool_name=f"art_{TAG}", mesh=False)
    assert sim.create_world()["status"] == "success"
    assert sim.add_camera(**CAM)["status"] == "success"
    return sim

def save(name, arr):
    np.save(OUT / f"{name}.npy", arr)

# ---- A: the crate the caller asked for, spelled as a list ----
sim = fresh()
res = sim.add_object(name="crate", shape="box", size=[0.3, 0.3, 0.3], position=[0.0, 0.0, 0.15])
facts["reference"] = {"status": res["status"], "text": text(res)}
sim.step(120)
save("A_reference", render(sim))
sim.cleanup()

# ---- B: the same three edge lengths, produced lazily ----
sim = fresh()
unsized = (edge for edge in (0.3, 0.3, 0.3))
res = sim.add_object(name="crate", shape="box", size=unsized, position=[0.0, 0.0, 0.15])
facts["unsized_add_object"] = {"status": res["status"], "text": text(res)}
sim.step(120)
save("B_unsized", render(sim))
facts["objects_after"] = sim.list_objects()
sim.cleanup()

# ---- patch-op: the size field beside its sibling pos field ----
sim = fresh()
def op(field, value):
    base = {"op": "add_geom", "body": "world", "name": f"p_{field}", "type": "box",
            "size": [0.2, 0.2, 0.2], "pos": [1.0, 0.0, 0.3]}
    base[field] = value
    return base
for field in ("size", "pos"):
    r = sim.patch_scene_mjcf(ops=[op(field, (c for c in (0.25, 0.25, 0.25)))])
    facts[f"patch_{field}"] = {"status": r["status"], "text": text(r)}
sim.cleanup()

pathlib.Path(f"/tmp/artfacts-{TAG}.json").write_text(json.dumps(facts, indent=2))
for k, v in facts.items():
    if isinstance(v, dict) and "text" in v:
        print(f"  {k}: [{v['status']}] {v['text'][:150]}", flush=True)
print("WROTE", OUT, flush=True)
