import json, pathlib, numpy as np, strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
import mujoco as mj
from strands_robots.simulation.mujoco.simulation import Simulation
import sys
sys.path.insert(0, str(pathlib.Path("tests").resolve()))
from strands_robots.simulation.ik import MinkIKBridge

TEST = pathlib.Path("tests/simulation/mujoco/test_move_to_body_frame_end_effector.py").read_text()
import re
tpl = re.search(r'_BODY_ARM_TEMPLATE = """(.*?)"""', TEST, re.S).group(1)
REACHABLE = [0.2, 0.1, 0.2]
TOL, MAXS = 0.02, 400

def png(sim, cam):
    r = sim.render(camera_name=cam, width=820, height=700)
    assert r.get("status") == "success", r
    return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

def proj(cp, p):
    T, K = np.asarray(cp.T_world_cam, float), np.asarray(cp.K, float)
    R, t = T[:3, :3], T[:3, 3]
    pc = R.T @ (np.asarray(p, float) - t)
    d = -pc[2]
    return K[0, 0] * pc[0] / d + K[0, 2], K[1, 2] - K[1, 1] * pc[1] / d, d

facts = {"tree": TREE, "target": REACHABLE, "tol": TOL}
p = pathlib.Path("/tmp/art_hand_arm.xml"); p.write_text(tpl.format(tip="hand"))
sim = Simulation(backend="mujoco", mesh=False)
sim.create_world()
assert sim.add_robot(name="arm", urdf_path=str(p))["status"] == "success"
# camera BEFORE any rollout: add_camera recompiles the spec and drops ctrl
assert sim.add_camera(name="look", position=[0.46, -0.44, 0.38], target=[0.12, 0.045, 0.19], fov=38)["status"] == "success"
model, data = sim._world._model, sim._world._data
bid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm/hand"))
mj.mj_forward(model, data)
pathlib.Path("/tmp/art_home.png").write_bytes(png(sim, "look"))
facts["home_xpos"] = [float(v) for v in data.xpos[bid]]

res = sim.move_to(robot_name="arm", position=REACHABLE, tol=TOL, max_steps=MAXS)
js = next(c["json"] for c in res["content"] if "json" in c)
facts["status"] = res["status"]
facts["payload"] = {k: js[k] for k in ("reached","steps","position_error_m","ik_residual_m","frame","frame_type")}
facts["ee_position"] = [float(v) for v in js["ee_position"]]
facts["ee_quat"] = [float(v) for v in js["ee_orientation_wxyz"]]
pathlib.Path("/tmp/art_reached.png").write_bytes(png(sim, "look"))

xpos = np.asarray(data.xpos[bid], float); xipos = np.asarray(data.xipos[bid], float)
facts["xpos"] = xpos.tolist(); facts["xipos"] = xipos.tolist()
facts["frame_offset_m"] = float(np.linalg.norm(xipos - xpos))
facts["err_from_origin_m"] = float(np.linalg.norm(xpos - np.asarray(REACHABLE)))
facts["err_from_inertial_m"] = float(np.linalg.norm(xipos - np.asarray(REACHABLE)))
bridge = MinkIKBridge(model, js["frame"], js["frame_type"])
mink_p = bridge.ee_pose(np.array(data.qpos, float, copy=True))[:3, 3]
facts["mink_pose"] = [float(v) for v in mink_p]
facts["mink_vs_origin_m"] = float(np.abs(mink_p - xpos).max())
facts["mink_vs_inertial_m"] = float(np.abs(mink_p - xipos).max())
cp = sim.get_camera_params(camera_name="look", width=820, height=700)
facts["proj"] = {"target": proj(cp, REACHABLE)[:2], "xpos": proj(cp, xpos)[:2], "xipos": proj(cp, xipos)[:2]}
sim.cleanup()

# leaf-body route: same geometry, tip named link4
p2 = pathlib.Path("/tmp/art_leaf_arm.xml"); p2.write_text(tpl.format(tip="link4"))
s2 = Simulation(backend="mujoco", mesh=False); s2.create_world()
assert s2.add_robot(name="arm", urdf_path=str(p2))["status"] == "success"
r2 = s2.move_to(robot_name="arm", position=REACHABLE, tol=TOL, max_steps=MAXS)
j2 = next(c["json"] for c in r2["content"] if "json" in c)
facts["leaf"] = {"status": r2["status"], **{k: j2[k] for k in ("reached","steps","frame","frame_type")},
                 "ee_position": [float(v) for v in j2["ee_position"]]}
s2.cleanup()

json.dump(facts, open("/tmp/art_facts.json","w"), indent=2)
print(json.dumps({k: facts[k] for k in ("payload","frame_offset_m","err_from_origin_m","err_from_inertial_m",
                                        "mink_vs_origin_m","mink_vs_inertial_m","leaf")}, indent=2))
