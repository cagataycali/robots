"""Capture the fromto-resize outcome in whichever tree this runs in."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
from strands_robots.simulation import Simulation
import mujoco as mj

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = pathlib.Path(TREE).name

SCENE = """
<mujoco>
  <compiler angle="radian"/>
  <visual><global offwidth="1200" offheight="900"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.68 0.85" rgb2="0.2 0.28 0.42" width="64" height="64"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.75 0.75 0.78 1"/>
    <geom name="post_a" type="capsule" fromto="-0.30 0.16 0  -0.30 0.16 0.24" size="0.006" rgba="0.2 0.2 0.22 1"/>
    <geom name="post_b" type="capsule" fromto="0.30 0.16 0  0.30 0.16 0.24" size="0.006" rgba="0.2 0.2 0.22 1"/>
    <body name="link" pos="0 0 0.16">
      <geom name="cap" type="capsule" fromto="-0.15 0 0  0.15 0 0" size="0.05"
            density="800" rgba="0.95 0.45 0.12 1"/>
    </body>
    <camera name="look" pos="0 -1.05 0.42" xyaxes="1 0 0  0 0.38 0.92" fovy="34"/>
  </worldbody>
</mujoco>
"""

def render(sim, name):
    r = sim.render(camera_name="look", width=900, height=560)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / f"{name}.png").write_bytes(png)
    import imageio.v3 as iio
    return np.asarray(iio.imread(png))[:, :, :3]

def facts(sim, gid, bid):
    m = sim._world._model
    return {
        "geom_size": [round(float(x), 6) for x in m.geom_size[gid]],
        "mass": round(float(m.body_mass[bid]), 6),
        "inertia": [round(float(x), 6) for x in m.body_inertia[bid]],
    }

def mk():
    s = Simulation(tool_name="art", backend="mujoco", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(SCENE)["status"] == "success"
    return s

res = {"tree": TREE}
frames = {}

# --- the requested resize: 0.15 -> 0.30 half-length -------------------------
sim = mk(); m = sim._world._model
gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "cap"); bid = int(m.geom_bodyid[gid])
res["declared"] = facts(sim, gid, bid)
frames["declared"] = render(sim, f"declared-{TAG}")

r = sim.set_geom_properties(geom_name="cap", size=[0.05, 0.30])
res["resize"] = {"status": r["status"], "text": r["content"][0]["text"]}
res["after_set"] = facts(sim, gid, bid)
frames["after_set"] = render(sim, f"after_set-{TAG}")

assert sim.add_object(name="unrelated", shape="sphere", size=[0.03],
                      position=[0.9, 0, 0.4])["status"] == "success"
res["after_recompile"] = facts(sim, gid, bid)
frames["after_recompile"] = render(sim, f"after_recompile-{TAG}")
sim.cleanup()

# --- the honored path: thicken the same capsule (radius only) ---------------
sim = mk(); m = sim._world._model
gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "cap"); bid = int(m.geom_bodyid[gid])
r = sim.set_geom_properties(geom_name="cap", size=[0.09, 0.15])
res["thicken"] = {"status": r["status"], "text": r["content"][0]["text"]}
res["thicken_after_set"] = facts(sim, gid, bid)
assert sim.add_object(name="unrelated", shape="sphere", size=[0.03],
                      position=[0.9, 0, 0.4])["status"] == "success"
res["thicken_after_recompile"] = facts(sim, gid, bid)
frames["thicken"] = render(sim, f"thicken-{TAG}")
sim.cleanup()

def frac(a, b):
    return round(float((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 8).mean()) * 100.0, 2)

res["diff_pct"] = {
    "declared_vs_after_set": frac(frames["declared"], frames["after_set"]),
    "after_set_vs_after_recompile": frac(frames["after_set"], frames["after_recompile"]),
    "declared_vs_after_recompile": frac(frames["declared"], frames["after_recompile"]),
    "declared_vs_thicken": frac(frames["declared"], frames["thicken"]),
}
for k, f in frames.items():
    np.save(OUT / f"{k}-{TAG}.npy", f)
(OUT / f"facts-{TAG}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
