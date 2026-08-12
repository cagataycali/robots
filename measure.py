"""Two sequential WBC rollouts on one world: what drives the actuators each time."""
import glob, json, pathlib, sys
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
import mujoco
from strands_robots import Robot
from strands_robots.policies.wbc import WBCConfig, WBCPolicy
from strands_robots.policies.wbc import sim_control as sc

SNAP = glob.glob("/home/cagatay/.cache/huggingface/hub/models--nepyope--GR00T-WholeBodyControl_g1/"
                 "snapshots/*/GR00T-WholeBodyControl-Balance.onnx")[0]
AFFINE = int(mujoco.mjtBias.mjBIAS_AFFINE)   # position servo
NONE_ = int(mujoco.mjtBias.mjBIAS_NONE)      # torque motor

def policy():
    return WBCPolicy(config=WBCConfig(policy_path=SNAP), walk=False, allow_missing_models=False)

# class-level spy: every apply() records which controller ran and on what gains
LOG: list[dict] = []
_orig_apply = sc.WBCTorqueController.apply
def spy(self, action_dict, model, data, robot_name):
    LOG.append({"ctrl": id(self), "biastype": int(model.actuator_biastype[self.leg_waist_actuator_ids[0]])})
    return _orig_apply(self, action_dict, model, data, robot_name)
sc.WBCTorqueController.apply = spy

# the engine hook imports from the PACKAGE, so patch the package attribute
import strands_robots.policies.wbc as wbcpkg  # noqa: E402
_orig_install = wbcpkg.install_wbc_torque_control
INSTALLS: list[int] = []
def spy_install(sim, pol, name):
    c = _orig_install(sim, pol, name)
    INSTALLS.append(id(c))
    return c
wbcpkg.install_wbc_torque_control = spy_install

def pelvis_z(sim):
    m, d = sim._world._model, sim._world._data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(d.xpos[bid if bid >= 0 else 1][2])

sim = Robot("unitree_g1", mesh=False)
runs = []
for i in (1, 2):
    LOG.clear(); INSTALLS.clear()
    res = sim.run_policy(robot_name="unitree_g1", policy_object=policy(),
                         policy_kwargs={"target_velocity": [0.0, 0.0, 0.0]},
                         duration=2.0, control_frequency=50.0, action_horizon=1)
    biastypes = sorted({e["biastype"] for e in LOG})
    ctrls = sorted({e["ctrl"] for e in LOG})
    rec = {
        "run": i,
        "status": res.get("status"),
        "installs_this_run": len(INSTALLS),
        "applies": len(LOG),
        "distinct_controllers_dispatched": len(ctrls),
        "controller_is_fresh": bool(INSTALLS) and ctrls == INSTALLS,
        "biastype_during_apply": biastypes,
        "gains_were_torque": biastypes == [NONE_],
        "gains_were_servo": biastypes == [AFFINE],
        "registered_after_run": "action_controller" in sim._world._backend_state,
        "pelvis_z_after": round(pelvis_z(sim), 4),
    }
    runs.append(rec)
    print(f"[run{i}] {json.dumps(rec)}")
sim.cleanup()

out = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1]),
       "AFFINE_servo": AFFINE, "NONE_torque": NONE_, "runs": runs}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1])
