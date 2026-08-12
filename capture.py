"""Render a Go2 added behind two arms, at spawn and after a PD hold."""
import json, pathlib, sys
import numpy as np, strands_robots, mujoco as mj
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE, flush=True)
from strands_robots.simulation import Simulation

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LABEL = sys.argv[2]
CAM = dict(position=[0.90, -0.75, 0.62], target=[0.10, 0.38, 0.14], fov=38)
W, H = 900, 700

def png(sim, name):
    r = sim.render(camera_name="look", width=W, height=H)
    assert r.get("status") == "success", r
    data = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / f"{name}.png").write_bytes(data)
    import imageio.v3 as iio
    return np.asarray(iio.imread(data))

def base_facts(sim):
    o = sim.get_observation(robot_name="base", skip_images=True)
    m, d = sim._world._model, sim._world._data
    fj = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE)][0]
    adr = int(m.jnt_qposadr[fj])
    q = [float(x) for x in d.qpos[adr : adr + 7]]
    return {
        "base_pos": [float(x) for x in o["base_pos"]],
        "quat_norm": float(np.linalg.norm(q[3:7])),
        "qpos": q,
        "thigh": float(o["FL_thigh_joint"]),
    }

def build(n_prior):
    sim = Simulation(mesh=False)
    assert sim.create_world()["status"] == "success"
    for i in range(n_prior):
        cfg = "so101" if i % 2 == 0 else "so100"
        assert sim.add_robot(name=f"arm_{i}", data_config=cfg, position=[0.6 * i, 0.0, 0.0])["status"] == "success"
        keys = sim.robot_action_keys(f"arm_{i}")
        sim.send_action(dict(zip(keys, [0.7, -0.9, 1.1, 0.0, 0.0, 0.4][: len(keys)])), robot_name=f"arm_{i}")
        sim.step(300)
    assert sim.add_robot(name="base", data_config="go2", position=[0.0, 0.8, 0.0], keyframe="home")["status"] == "success"
    assert sim.add_camera(name="look", **CAM)["status"] == "success"
    return sim

facts = {"tree": TREE, "label": LABEL}

# --- the reported scene: two arms already in the world ---
sim = build(2)
facts["spawn"] = base_facts(sim)
facts["arms_at_spawn"] = {
    r: {k: round(float(v), 6) for k, v in sim.get_observation(robot_name=r, skip_images=True).items()
        if not hasattr(v, "shape") and not isinstance(v, list) and not k.endswith(".vel")}
    for r in ("arm_0", "arm_1")
}
img_spawn = png(sim, f"{LABEL}_spawn")
# hold the spawn joint angles and let the legs carry the body
obs = sim.get_observation(robot_name="base", skip_images=True)
keys = sim.robot_action_keys("base")
hold = {k: float(obs[k]) for k in keys if k in obs}
for _ in range(8):
    sim.send_action(hold, robot_name="base", n_substeps=50)
facts["held"] = base_facts(sim)
img_held = png(sim, f"{LABEL}_held")
sim.destroy()

# --- control: the layout where main was accidentally right ---
sim = build(1)
facts["control_one_prior"] = base_facts(sim)
img_ctl = png(sim, f"{LABEL}_control")
sim.destroy()

for nm, im in (("spawn", img_spawn), ("held", img_held), ("control", img_ctl)):
    np.save(OUT / f"{LABEL}_{nm}.npy", im)
(OUT / f"facts_{LABEL}.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({k: facts[k] for k in ("spawn", "held", "control_one_prior")}, indent=1), flush=True)
