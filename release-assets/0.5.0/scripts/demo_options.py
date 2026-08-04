"""#1687: honor the physics options a robot model declares for itself.

panda.xml declares <option integrator="implicitfast"/>. <option> is model-global
and does not survive spec.attach(), so before the fix the scene silently kept its
own default (Euler) and the arm was integrated under a setting its own model
rejects. Measures position-servo tracking, residual jitter and settling.
"""
import io, json, os, sys
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, mujoco as mj
from PIL import Image
from strands_robots.simulation import Simulation

TREE = sys.argv[1]
OUT = f"/tmp/relnotes/assets/options_{TREE}"
W, H = 620, 470
TARGET = {"actuator1": 0.9, "actuator2": 0.55, "actuator4": -1.75, "actuator6": 1.55}

def png(r): return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

sim = Simulation(backend="mujoco", tool_name="opt", mesh=False)
try:
    sim.create_world()
    sim.add_robot(name="arm", data_config="panda")
    sim.add_camera(name="look", position=[1.30, -1.05, 0.95], target=[0.20, 0.0, 0.62], fov=42)
    m = sim._world._model
    integ = mj.mjtIntegrator(int(m.opt.integrator)).name
    declared = "mjINT_IMPLICITFAST"

    frames, trace = [], []
    N = 260
    for i in range(N):
        sim.send_action(TARGET, robot_name="arm", n_substeps=10)
        obs = sim.get_observation(robot_name="arm")
        err = max(abs(float(obs[k.replace("actuator", "joint")]) - v) for k, v in TARGET.items())
        vel = max(abs(float(obs[f"{k.replace('actuator','joint')}.vel"])) for k in TARGET)
        trace.append({"i": i, "max_track_err_rad": err, "max_joint_vel": vel})
        if i % 10 == 0 or i == N - 1:
            frames.append(np.array(Image.open(io.BytesIO(
                png(sim.render(camera_name="look", width=W, height=H)))).convert("RGB")))

    tail = trace[-60:]
    facts = {
        "tree": TREE,
        "integrator_compiled": integ,
        "integrator_declared_by_panda_xml": declared,
        "declaration_honored": integ == declared,
        "settled_track_err_rad": round(float(np.mean([t["max_track_err_rad"] for t in tail])), 6),
        "settled_jitter_rad_s": round(float(np.mean([t["max_joint_vel"] for t in tail])), 6),
        "peak_jitter_rad_s": round(float(max(t["max_joint_vel"] for t in trace)), 6),
    }
    np.savez_compressed(f"{OUT}.npz", frames=np.stack(frames), final=frames[-1])
    json.dump({"facts": facts, "trace": trace}, open(f"{OUT}.json", "w"))
    print(TREE, json.dumps(facts, indent=1), flush=True)
finally:
    sim.cleanup()
