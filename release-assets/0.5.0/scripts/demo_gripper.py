"""#1652: honor a gripper command identically for actuator-name and joint-name keys.

Drives the Panda gripper FULLY OPEN twice - once addressing the actuator by its
own name (what robot_action_keys returns, i.e. what every policy is fed) and once
by a joint the actuator drives - and measures the resulting finger gap.
Renders a close-up of the hand for each.
"""
import io, json, os, sys
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from PIL import Image
from strands_robots.simulation import Simulation

TREE = sys.argv[1]
OUT = f"/tmp/relnotes/assets/gripper_{TREE}"
W, H = 520, 420

def png(res):
    return next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)

def jj(res):
    return next((c["json"] for c in res.get("content", []) if "json" in c), None)

def gap(sim):
    """Finger gap = sum of the two prismatic finger joint positions."""
    obs = sim.get_observation(robot_name="arm")
    f = {k: v for k, v in obs.items() if "finger_joint" in k and not k.endswith(".vel")}
    return float(sum(float(v) for v in f.values())), f

def run(key_name):
    sim = Simulation(backend="mujoco", tool_name=f"g_{key_name}", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="arm", data_config="panda")
        # Head-on down +x at the jaw: the fingers separate in y, so the opening reads horizontally.
        sim.add_camera(name="hand", position=[0.40, -0.13, 0.975], target=[0.136, 0.0, 0.884], fov=22)
        sim.step(120)
        g0, _ = gap(sim)
        # Fully-open command. 1.0 is the normalized full-scale value a policy emits.
        res = sim.send_action({key_name: 1.0}, robot_name="arm", n_substeps=10)
        for _ in range(120):
            sim.send_action({key_name: 1.0}, robot_name="arm", n_substeps=10)
        g1, joints = gap(sim)
        img = np.array(Image.open(io.BytesIO(png(sim.render(camera_name="hand", width=W, height=H)))).convert("RGB"))
        return img, {"action_key": key_name, "status": res["status"],
                     "gap_before_m": round(g0, 6), "gap_after_m": round(g1, 6),
                     "joints": {k: round(float(v), 6) for k, v in joints.items()}}
    finally:
        sim.cleanup()

frames, facts = {}, []
for key in ("actuator8", "finger_joint1"):
    img, f = run(key)
    frames[key] = img
    facts.append(f)
    print(TREE, f, flush=True)

diff = float((np.abs(frames["actuator8"].astype(int) - frames["finger_joint1"].astype(int)).sum(2) > 30).mean())
gaps = [f["gap_after_m"] for f in facts]
print(f"FRAMING differing-pixel fraction between the two panels: {diff:.4f}  gaps={gaps}")
if abs(gaps[0] - gaps[1]) > 1e-4:
    assert diff > 0.004, f"gaps differ ({gaps}) but the renders do not ({diff}) - camera is not on the jaw"
else:
    assert diff < 0.004, f"gaps agree ({gaps}) but the renders differ ({diff})"

np.savez_compressed(f"{OUT}.npz", **frames)
json.dump(facts, open(f"{OUT}.json", "w"), indent=1)
print("SAVED", OUT)
