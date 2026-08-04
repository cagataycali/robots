"""#1649 get_world_point + #1654 analytic motion primitives.

An agent sees a frame, names a PIXEL, and the arm goes to that point in the world -
no joint targets, no hand-written IK.
"""
import io, json, os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from PIL import Image
from strands_robots.simulation import Simulation

A = "/tmp/relnotes/assets"
W, H = 660, 500
CUBE, TOP, X, Y = 0.030, 0.12, 0.50, 0.06
CZ = TOP + CUBE / 2
PAD_BELOW = 0.1214
DOWN = [0.0, 1.0, 0.0, 0.0]

def png(r): return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
def jj(r):  return next((c["json"] for c in r.get("content", []) if "json" in c), None)
def shot(sim, cam="eye"):
    return np.array(Image.open(io.BytesIO(png(sim.render(camera_name=cam, width=W, height=H)))).convert("RGB"))

sim = Simulation(backend="mujoco", tool_name="ground", mesh=False)
try:
    sim.create_world()
    sim.add_robot(name="arm", data_config="panda")
    sim.add_object(name="table", shape="box", size=[0.24, 0.34, TOP], position=[X, 0.0, TOP / 2],
                   is_static=True, color=[0.58, 0.60, 0.64, 1])
    sim.add_object(name="cube", shape="box", size=[CUBE] * 3, position=[X, Y, CZ],
                   mass=0.05, color=[0.95, 0.38, 0.08, 1])
    sim.add_camera(name="eye", position=[0.95, -0.52, 0.62], target=[0.46, 0.02, 0.16], fov=40)
    sim.step(250)

    frame = shot(sim)
    truth = jj(sim.get_body_state(body_name="cube"))["position"]

    # An agent picks the pixel by colour: the orange cube's centroid in the frame.
    f = frame.astype(int)
    mask = (f[:, :, 0] > 90) & (f[:, :, 0] - f[:, :, 2] > 45) & (f[:, :, 0] - f[:, :, 1] > 25)
    ys, xs = np.nonzero(mask)
    u, v = int(round(xs.mean())), int(round(ys.min() + 0.28 * (ys.max() - ys.min())))
    print(f"orange pixels={mask.sum()}  chosen pixel=({u},{v})")

    g = sim.get_world_point(camera_name="eye", pixels=[[u, v]], width=W, height=H)
    gj = jj(g)
    pt = gj["points"][0] if isinstance(gj, dict) and "points" in gj else gj
    print("get_world_point ->", g["status"], json.dumps(gj)[:300])
    world = pt["position"] if isinstance(pt, dict) and "position" in pt else pt
    world = [float(c) for c in world]
    err_mm = float(np.linalg.norm(np.array(world[:2]) - np.array(truth[:2]))) * 1000

    # Drive there with the analytic primitive - no joint targets anywhere.
    for _ in range(30): sim.send_action({"actuator8": 1.0}, robot_name="arm", n_substeps=8)
    hover = sim.move_to(robot_name="arm",
                        position=[world[0], world[1], world[2] + PAD_BELOW + 0.12],
                        orientation=DOWN, tol=0.02, max_steps=700)
    desc = sim.move_to(robot_name="arm",
                       position=[world[0], world[1], world[2] + PAD_BELOW],
                       orientation=DOWN, tol=0.015, max_steps=700)
    frame_at = shot(sim)
    dj = jj(desc) or {}
    ee = [float(c) for c in dj.get("ee_position", [0, 0, 0])]
    pad_xy_err_mm = float(np.linalg.norm(np.array(ee[:2]) - np.array(truth[:2]))) * 1000

    facts = {
        "pixel": [u, v], "world_point": [round(c, 4) for c in world],
        "cube_truth": [round(float(c), 4) for c in truth],
        "grounding_xy_error_mm": round(err_mm, 2),
        "hover": hover["status"], "descend": desc["status"],
        "reached": dj.get("reached"), "ik_residual_m": round(float(dj.get("ik_residual_m", 0)), 6),
        "ee_position": [round(c, 4) for c in ee],
        "pad_over_cube_xy_error_mm": round(pad_xy_err_mm, 2),
    }
    np.savez_compressed(f"{A}/grounding.npz", frame=frame, frame_at=frame_at, mask=mask)
    json.dump(facts, open(f"{A}/grounding.json", "w"), indent=1)
    print(json.dumps(facts, indent=1))
finally:
    sim.cleanup()
