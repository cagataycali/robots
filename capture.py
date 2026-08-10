import io, json, math, pathlib
import numpy as np, strands_robots
from PIL import Image
TREE = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", TREE, flush=True)
from strands_robots.simulation import Simulation

ARM = pathlib.Path("tests/simulation/mujoco/test_motion_primitives.py").read_text().split('ARM_XML = """',1)[1].split('"""',1)[0]
p = pathlib.Path("/tmp/art_arm.xml"); p.write_text(ARM)
sim = Simulation(tool_name="art", mesh=False)
assert sim.create_world(gravity=[0,0,0])["status"] == "success"
assert sim.add_robot("arm", urdf_path=str(p))["status"] == "success"
assert sim.add_camera(name="look", position=[0.36,-0.38,0.34], target=[0.08,0.05,0.22], fov=30)["status"]=="success"

def shot():
    r = sim.render(camera_name="look", width=560, height=520)
    assert r["status"] == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)

def call(action, **kw):
    return sim._dispatch_action(action, {"action": action, **kw})
def text(r):
    return " ".join(c.get("text","") for c in r["content"] if isinstance(c, dict))

home = shot()
r = call("move_to", robot_name="arm", position=[0.2, 0.1, 0.2], tol=0.02, max_steps=400)
assert r["status"] == "success", text(r)
honored_msg = text(r)
after_honored = shot()
w = sim._world
q0 = np.array(w._data.qpos, copy=True); c0 = np.array(w._data.ctrl, copy=True); t0 = float(w._data.time)

REFUSALS = [
    ("rotate_wrist", {"target_yaw": math.nan}),
    ("rotate_wrist", {"target_yaw": 0.1, "tol": math.nan}),
    ("rotate_wrist", {"target_yaw": 0.1, "max_steps": 2.7}),
    ("move_to", {"position": [0.2,0.1,0.2], "tol": math.inf}),
    ("set_gripper", {"state": "open", "steps": True}),
]
rows = []
for action, kw in REFUSALS:
    res = call(action, robot_name="arm", **kw)
    rows.append({"call": f"{action}({', '.join(f'{k}={v!r}' for k,v in kw.items())})",
                 "status": res["status"], "message": text(res),
                 "qpos_same": bool(np.array_equal(w._data.qpos, q0)),
                 "ctrl_same": bool(np.array_equal(w._data.ctrl, c0)),
                 "clock_same": float(w._data.time) == t0})
after_refusals = shot()

facts = {
    "tree": str(TREE), "honored_msg": honored_msg, "rows": rows,
    "moved_frac": float(np.mean(np.any(home != after_honored, axis=2))),
    "refusal_max_delta": int(np.abs(after_honored.astype(int) - after_refusals.astype(int)).max()),
    "arm_frac": float(np.mean((after_honored.max(2).astype(int) - after_honored.min(2).astype(int)) > 45)),
}
np.save("/tmp/art_home.npy", home); np.save("/tmp/art_honored.npy", after_honored); np.save("/tmp/art_refused.npy", after_refusals)
json.dump(facts, open("/tmp/art_facts.json","w"), indent=1)
print(json.dumps({k:v for k,v in facts.items() if k!="rows"}, indent=1))
for r in rows: print(" ", r["status"], r["qpos_same"], r["ctrl_same"], r["clock_same"], r["message"][:78])
sim.cleanup()
