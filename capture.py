"""Capture the measured facts + one real get_frame render for the artifact."""
import json, pathlib, numpy as np, strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.rendering import HybridCompositor

OUT = pathlib.Path("_art"); OUT.mkdir(exist_ok=True)

def scene():
    s = Simulation(tool_name="artifact", mesh=False)
    s.create_world()
    s.add_object(name="crate", shape="box", size=[0.16, 0.16, 0.16],
                 position=[0.0, 0.0, 0.08], color=[0.95, 0.45, 0.10, 1.0], is_static=True)
    s.add_object(name="post", shape="cylinder", size=[0.05, 0.05, 0.30],
                 position=[0.26, 0.10, 0.15], color=[0.20, 0.55, 0.90, 1.0], is_static=True)
    s.add_camera(name="look", position=[0.62, -0.55, 0.42], target=[0.02, 0.0, 0.10], fov=38)
    return s

# --- 1. the real frame get_frame produces on a GL-capable host ---
s = scene()
rgb, depth = s.get_frame("look", width=640, height=520)
np.save(OUT / "frame.npy", rgb)
sat = float(((rgb.max(2).astype(int) - rgb.min(2).astype(int)) > 45).mean())
facts = {"tree": str(ROOT), "frame_shape": list(rgb.shape), "depth_shape": list(depth.shape),
         "saturated_frac": round(sat, 4),
         "depth_min": round(float(np.nanmin(depth)), 4), "depth_max": round(float(np.nanmax(depth)), 4)}
s.cleanup()

# --- 2. what each renderer consumer does when _get_renderer returns None ---
def outcome(fn):
    try:
        r = fn()
    except Exception as e:
        return {"kind": "raise", "type": type(e).__name__, "text": str(e)}
    if isinstance(r, dict) and "status" in r:
        t = next((b["text"] for b in r.get("content", []) if "text" in b), "")
        return {"kind": "envelope", "status": r["status"], "text": t}
    return {"kind": "value", "text": repr(r)[:80]}

s = scene()
s._get_renderer = lambda w, h: None
consumers = {
    "render": outcome(lambda: s.render(camera_name="look", width=8, height=6)),
    "render_depth": outcome(lambda: s.render_depth(camera_name="look", width=8, height=6)),
    "get_frame": outcome(lambda: s.get_frame("look", width=8, height=6)),
    "_get_sim_observation": {"kind": "skip", "text": "camera omitted, rest of the observation kept"},
}
consumers_extra = {
    "HybridCompositor.render": outcome(lambda: HybridCompositor(s).render("look")),
    "get_world_point": outcome(lambda: s.get_world_point(camera_name="look", pixels=[[2, 1]], width=8, height=6)),
}
s.cleanup()

# --- 3. the envelope-unpack hazard, measured ---
env = {"status": "error", "content": [{"text": "Rendering unavailable ..."}]}
a, b = env
facts["unpack_hazard"] = {"binds": [a, b], "asarray_shape": list(np.asarray(a).shape)}
facts["consumers"] = consumers
facts["extra"] = consumers_extra
(OUT / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if k != "consumers"}, indent=2)[:900])
for k, v in {**consumers, **consumers_extra}.items():
    print(f"  {k:26s} {v['kind']:9s} {v.get('text','')[:66]}")
