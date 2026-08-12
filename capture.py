"""Two consecutive real WBC balance rollouts on one sim, per tree."""
import glob, json, pathlib, sys
import strands_robots

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
tag = sys.argv[2]

from strands_robots import Robot
from strands_robots.policies.wbc import WBCConfig, WBCPolicy

CK = glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/models--nepyope--GR00T-WholeBodyControl_g1"
                                        "/snapshots/*/GR00T-WholeBodyControl-Balance.onnx"))[0]

sim = Robot("unitree_g1", mesh=False)
r = sim.add_camera(name="look", position=[2.1, -1.9, 1.15], target=[0.0, 0.0, 0.62], fov=34)
assert r["status"] == "success", r
policy = WBCPolicy(config=WBCConfig(policy_path=CK), walk=False)

real_obs = sim.get_observation
trace: list[float] = []
shim: list[bool] = []
def spy(*a, **k):
    out = real_obs(*a, **k)
    trace.append(float(sim._world._data.qpos[2]))
    shim.append(sim._world._backend_state.get("action_controller") is not None)
    return out

runs = []
for i in (1, 2):
    trace.clear(); shim.clear()
    sim.get_observation = spy            # type: ignore[method-assign]
    res = sim.run_policy(robot_name="unitree_g1", policy_object=policy,
                         policy_kwargs={"target_velocity": [0.0, 0.0, 0.0]},
                         duration=3.0, control_frequency=50.0, action_horizon=1)
    sim.get_observation = real_obs        # type: ignore[method-assign]
    img = sim.render(camera_name="look", width=700, height=640)
    assert img["status"] == "success", img
    png = next(c["image"]["source"]["bytes"] for c in img["content"] if "image" in c)
    (OUT / f"run{i}_{tag}.png").write_bytes(png)
    runs.append({
        "run": i, "status": res.get("status"), "ticks": len(trace),
        "shim_active_all_ticks": bool(shim and all(shim)),
        "shim_active_any_tick": bool(any(shim)),
        "pelvis_z_end": round(trace[-1], 4), "pelvis_z_min": round(min(trace), 4),
        "trace": [round(v, 5) for v in trace],
    })
    print(f"run {i}: status={res.get('status')} shim_all_ticks={runs[-1]['shim_active_all_ticks']} "
          f"pelvis_end={runs[-1]['pelvis_z_end']} pelvis_min={runs[-1]['pelvis_z_min']}")

(OUT / f"facts_{tag}.json").write_text(json.dumps(
    {"tree": TREE, "checkpoint": pathlib.Path(CK).name, "runs": runs}, indent=2))
sim.cleanup()
