"""#1763: add_robot no longer rewinds the scene it is added to.

Builds a scene, drives an arm to a distinctive pose, lets a crate fall and settle,
then adds a SECOND robot - an ordinary incremental scene edit - and re-measures.
"""
import io, json, os, sys
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from PIL import Image
from strands_robots.simulation import Simulation

TREE = sys.argv[1]
OUT = f"/tmp/relnotes/assets/rewind_{TREE}"
W, H = 640, 480

def png(res):
    return next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
def jj(res):
    return next((c["json"] for c in res.get("content", []) if "json" in c), None)

def shot(sim):
    return np.array(Image.open(io.BytesIO(png(
        sim.render(camera_name="look", width=W, height=H)))).convert("RGB"))

def probe(sim):
    obs = sim.get_observation(robot_name="a")
    joints = {k: round(float(v), 4) for k, v in obs.items()
              if not k.endswith(".vel") and not hasattr(v, "shape")}
    crate = jj(sim.get_body_state(body_name="crate"))["position"]
    st = sim.get_state()
    txt = next(c["text"] for c in st["content"] if "text" in c)
    return {"joints": joints, "crate_z": round(float(crate[2]), 4),
            "crate_xy": [round(float(crate[0]), 3), round(float(crate[1]), 3)],
            "clock": txt.strip().splitlines()[0][:60]}

sim = Simulation(backend="mujoco", tool_name="rw", mesh=False)
try:
    sim.create_world()
    sim.add_robot(name="a", data_config="panda", position=[0.0, 0.30, 0.0])
    sim.add_object(name="pedestal", shape="box", size=[0.16, 0.16, 0.30],
                   position=[0.55, -0.30, 0.15], is_static=True, color=[0.55, 0.57, 0.62, 1])
    sim.add_object(name="crate", shape="box", size=[0.13, 0.13, 0.13],
                   position=[0.55, -0.30, 0.62], mass=0.5, color=[0.95, 0.42, 0.10, 1])
    sim.add_camera(name="look", position=[1.45, -1.32, 0.92], target=[0.34, -0.05, 0.30], fov=40)
    # Park the arm at a distinctive pose and let the crate land on the pedestal.
    pose = {"actuator1": 0.55, "actuator2": 0.45, "actuator4": -1.60, "actuator6": 1.30}
    for _ in range(500):
        sim.send_action(pose, robot_name="a", n_substeps=6)
    before = probe(sim)
    img_before = shot(sim)
    # An ordinary incremental edit: add a second robot to the running scene.
    add = sim.add_robot(name="b", data_config="panda", position=[0.0, -0.95, 0.0])
    after = probe(sim)
    img_after = shot(sim)
    facts = {"add_robot_status": add["status"], "before": before, "after": after}
    np.savez_compressed(f"{OUT}.npz", before=img_before, after=img_after)
    json.dump(facts, open(f"{OUT}.json", "w"), indent=1)
    print(TREE, "add_robot:", add["status"])
    print("  before:", before)
    print("  after :", after)
finally:
    sim.cleanup()
