"""Capture what a caller sees when remove_camera's recompile is refused."""
import json, pathlib, sys
import mujoco
import strands_robots.simulation.mujoco.simulation as sim_mod
from strands_robots.simulation import Simulation

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[2]
TREE = str(pathlib.Path(sim_mod.__file__).parents[3])
print("TREE:", TREE)

W, H = 620, 500

def save(res, name):
    if res.get("status") != "success":
        return None
    b = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
    p = OUT / f"{TAG}_{name}.png"
    p.write_bytes(b)
    return str(p)

def cams(sim):
    m = sim._world._model
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]

def build():
    s = Simulation(backend="mujoco", mesh=False)
    s.create_world()
    s.add_object(name="plate", shape="box", size=[0.5, 0.5, 0.03], position=[0, 0, 0.015],
                 is_static=True, color=[0.82, 0.80, 0.74, 1.0])
    s.add_object(name="block_a", shape="box", size=[0.11, 0.11, 0.11], position=[-0.14, 0.02, 0.09],
                 color=[0.90, 0.42, 0.12, 1.0])
    s.add_object(name="post", shape="cylinder", size=[0.045, 0.045, 0.26], position=[0.16, -0.10, 0.13],
                 is_static=True, color=[0.24, 0.52, 0.86, 1.0])
    assert s.add_camera(name="watch", position=[0.62, -0.60, 0.44], target=[0.0, 0.0, 0.10],
                       fov=40)["status"] == "success"
    for _ in range(60):
        s.step(4)
    return s

facts = {"tree": TREE, "tag": TAG}
sim = build()
facts["reference_render"] = save(sim.render(camera_name="watch", width=W, height=H), "A_reference")
facts["reference_cams"] = cams(sim)

real = mujoco.MjSpec.recompile
def refuse(self, *a, **k):
    raise ValueError("scene is uncompilable (simulated)")
mujoco.MjSpec.recompile = refuse
try:
    r = sim.remove_camera("watch")
finally:
    mujoco.MjSpec.recompile = real
facts["remove_status"] = r["status"]
facts["remove_text"] = r["content"][0]["text"]
facts["after_remove_registry"] = list(sim._world.cameras)
facts["after_remove_model_cams"] = cams(sim)
facts["after_remove_spec_cams"] = [c.name for c in sim._world._backend_state["spec"].cameras]
res = sim.render(camera_name="watch", width=W, height=H)
facts["after_remove_render_status"] = res.get("status")
facts["after_remove_render"] = save(res, "B_after_remove")

add = sim.add_object(name="crate", shape="box", size=[0.13, 0.13, 0.13], position=[0.06, 0.24, 0.07],
                     color=[0.30, 0.68, 0.34, 1.0])
facts["later_add_status"] = add["status"]
for _ in range(30):
    sim.step(4)
facts["after_add_registry"] = list(sim._world.cameras)
facts["after_add_model_cams"] = cams(sim)
res2 = sim.render(camera_name="watch", width=W, height=H)
facts["after_add_render_status"] = res2.get("status")
facts["after_add_render_text"] = "" if res2.get("status") == "success" else res2["content"][0]["text"]
facts["after_add_render"] = save(res2, "C_after_later_add")
sim.cleanup()

(OUT / f"facts_{TAG}.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: v for k, v in facts.items() if "render" not in str(k)}, indent=2))
