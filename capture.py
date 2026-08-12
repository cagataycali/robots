"""Two sequential WBC balance rollouts on ONE world; render each end state.

Real GR00T-WholeBodyControl-Balance.onnx weights, headless MuJoCo (EGL).
"""
import glob, io, json, pathlib, sys
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
import numpy as np, mujoco
from PIL import Image
from strands_robots import Robot
from strands_robots.policies.wbc import WBCConfig, WBCPolicy
from strands_robots.policies.wbc import sim_control as sc

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[2]
SNAP = glob.glob("/home/cagatay/.cache/huggingface/hub/models--nepyope--GR00T-WholeBodyControl_g1/"
                 "snapshots/*/GR00T-WholeBodyControl-Balance.onnx")[0]
SERVO, TORQUE = int(mujoco.mjtBias.mjBIAS_AFFINE), int(mujoco.mjtBias.mjBIAS_NONE)

def policy():
    return WBCPolicy(config=WBCConfig(policy_path=SNAP), walk=False, allow_missing_models=False)

LOG: list[dict] = []
_orig = sc.WBCTorqueController.apply
def spy(self, action_dict, model, data, robot_name):
    LOG.append({"c": id(self), "b": int(model.actuator_biastype[self.leg_waist_actuator_ids[0]])})
    return _orig(self, action_dict, model, data, robot_name)
sc.WBCTorqueController.apply = spy

def render(sim, name):
    r = sim.render(camera_name="look", width=520, height=760)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    p = OUT / f"{TAG}_{name}.png"; p.write_bytes(png)
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    np.save(OUT / f"{TAG}_{name}.npy", a)
    return a

def pelvis_z(sim):
    m, d = sim._world._model, sim._world._data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(d.xpos[bid if bid >= 0 else 1][2])

sim = Robot("unitree_g1", mesh=False)
# camera BEFORE any rollout: add_camera recompiles the spec and drops ctrl
sim.add_camera(name="look", position=[2.2, -2.3, 1.25], target=[0.0, 0.0, 0.62], fov=32)
facts = {"tree": TREE, "servo": SERVO, "torque": TORQUE, "runs": []}
render(sim, "start")

for i in (1, 2):
    LOG.clear()
    res = sim.run_policy(robot_name="unitree_g1", policy_object=policy(),
                         policy_kwargs={"target_velocity": [0.0, 0.0, 0.0]},
                         duration=2.0, control_frequency=50.0, action_horizon=1)
    render(sim, f"run{i}")
    bts = sorted({e["b"] for e in LOG})
    facts["runs"].append({
        "run": i,
        "status": res.get("status"),
        "applies": len(LOG),
        "biastypes_during_apply": bts,
        "drove_torque_actuators": bts == [TORQUE],
        "controllers": len({e["c"] for e in LOG}),
        "registered_after": "action_controller" in sim._world._backend_state,
        "pelvis_z": round(pelvis_z(sim), 4),
    })
    print(f"[run{i}] {json.dumps(facts['runs'][-1])}")
sim.cleanup()
json.dump(facts, open(OUT / f"{TAG}_facts.json", "w"), indent=2)
print("wrote", OUT / f"{TAG}_facts.json")
