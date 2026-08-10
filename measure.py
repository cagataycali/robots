"""Measure remove_camera's behaviour when the validating recompile is refused."""
import pathlib, json
import mujoco
import strands_robots.simulation.mujoco.simulation as sim_mod
from strands_robots.simulation import Simulation

print("TREE:", pathlib.Path(sim_mod.__file__).parents[3])

def build():
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_camera(name="watch", position=[0.7, -0.7, 0.5], target=[0, 0, 0.1])
    sim.add_camera(name="keep", position=[0.0, -1.0, 0.6], target=[0, 0, 0.1])
    return sim

def facts(sim, label):
    m = sim._world._model
    spec = sim._world._backend_state.get("spec")
    reg = list(sim._world.cameras)
    spec_cams = [c.name for c in spec.cameras] if spec is not None else None
    gp = None
    try:
        p = sim.get_camera_params(camera_name="watch")
        gp = f"resolved {p.width}x{p.height}"
    except Exception as e:
        gp = f"{type(e).__name__}: {e}"
    r = sim.render(camera_name="watch", width=80, height=60)
    return {
        "label": label,
        "registry": reg,
        "model_ncam": int(m.ncam),
        "model_cam_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)],
        "spec_cams": spec_cams,
        "get_camera_params": gp,
        "render_status": r.get("status"),
    }

out = {}

# --- 1. baseline: healthy remove
sim = build()
out["baseline_before"] = facts(sim, "before")
r = sim.remove_camera("watch")
out["healthy_remove"] = {"status": r["status"], "text": r["content"][0]["text"]}
out["baseline_after"] = facts(sim, "after healthy remove")
sim.cleanup()

# --- 2. refused recompile
sim = build()
real = mujoco.MjSpec.recompile
def refuse(self, *a, **k):
    raise ValueError("scene is uncompilable (simulated)")
mujoco.MjSpec.recompile = refuse
try:
    r = sim.remove_camera("watch")
finally:
    mujoco.MjSpec.recompile = real
out["refused_remove"] = {"status": r["status"], "text": r["content"][0]["text"]}
out["after_refused"] = facts(sim, "after refused remove")

# can the caller recover / re-add?
radd = sim.add_camera(name="watch", position=[0.7, -0.7, 0.5], target=[0, 0, 0.1])
out["readd_after_refused"] = {"status": radd["status"], "text": radd["content"][0]["text"]}
out["after_readd"] = facts(sim, "after re-add")
sim.cleanup()

# --- 3. delayed application: unrelated later mutation recompiles from the spec
sim = build()
mujoco.MjSpec.recompile = refuse
try:
    sim.remove_camera("watch")
finally:
    mujoco.MjSpec.recompile = real
mid = facts(sim, "right after refused remove")
add = sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.3, 0, 0.05])
out["later_mutation"] = {"status": add["status"]}
out["after_later_mutation"] = facts(sim, "after an UNRELATED add_object")
out["mid"] = mid
sim.cleanup()

print(json.dumps(out, indent=2))
