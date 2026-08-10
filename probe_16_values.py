import json, pathlib, shutil, sys
import strands_robots.tools.run_policy as rp_mod
print("TREE:", pathlib.Path(rp_mod.__file__).parents[2], flush=True)
from strands_robots.tools.run_policy import run_policy
from strands_robots import Simulation

ARM = """<mujoco model="probe">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="2" range="-1.5 1.5" limited="true"/>
      <body name="link" pos="0 0 0.2">
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.018"/>
        <joint name="elbow" type="hinge" axis="0 1 0" damping="2" range="-1.5 1.5" limited="true"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30"/>
    <position name="a_elbow" joint="elbow" kp="30"/>
  </actuator>
</mujoco>"""

TMP = pathlib.Path("/tmp/probe_rp"); shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
xml = TMP / "arm.xml"; xml.write_text(ARM)

def truth(root):
    info = pathlib.Path(root) / "meta" / "info.json"
    if not info.exists():
        return {"present": False}
    d = json.loads(info.read_text())
    return {"present": True, "eps": d.get("total_episodes"), "frames": d.get("total_frames")}

def fresh_sim():
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(xml))
    return sim

CASES = [
    ("control_frequency", 0.0), ("control_frequency", -5.0), ("control_frequency", float("nan")),
    ("control_frequency", float("inf")), ("control_frequency", True), ("control_frequency", "30"),
    ("control_frequency", None), ("control_frequency", [30]),
    ("action_horizon", 0), ("action_horizon", -5), ("action_horizon", 2.7),
    ("action_horizon", float("nan")), ("action_horizon", True), ("action_horizon", "8"),
    ("action_horizon", None), ("action_horizon", [8]),
]

rows = []
for i, (param, val) in enumerate(CASES):
    root = TMP / f"ds{i}"
    # 1. seed a real 1-episode dataset with a fully usable config
    sim = fresh_sim()
    seed_res = run_policy(sim, robot_name="arm", n_episodes=1, n_steps=4,
                          control_frequency=30.0, dataset_fps=30,
                          dataset_root=str(root), dataset_repo_id="local/probe",
                          dataset_task="probe")
    sim.cleanup()
    before = truth(root)
    if not before.get("present") or before.get("eps") != 1:
        rows.append({"param": param, "value": repr(val), "SEED_FAILED": seed_res.get("status"),
                     "seed_text": (seed_res.get("content") or [{}])[0].get("text", "")[:200]})
        continue
    # 2. re-open the SAME dataset root with the bad knob
    sim = fresh_sim()
    kw = dict(robot_name="arm", n_episodes=1, n_steps=4,
              control_frequency=30.0, dataset_fps=30,
              dataset_root=str(root), dataset_repo_id="local/probe",
              dataset_task="probe")
    kw[param] = val
    res = run_policy(sim, **kw)
    sim.cleanup()
    after = truth(root)
    txt = (res.get("content") or [{}])[0].get("text", "")
    eps = next((b["json"] for b in res.get("content") or [] if "json" in b), {}).get("episodes", [])
    rows.append({
        "param": param, "value": repr(val), "status": res.get("status"),
        "summary": txt[:120],
        "before": before, "after": after,
        "destroyed": bool(before.get("eps") == 1 and after.get("eps") in (0, None)),
        "ep0_text": (eps[0].get("text", "")[:160] if eps else ""),
    })
    print(json.dumps(rows[-1]), flush=True)

pathlib.Path("/tmp/probe_rp_main.json").write_text(json.dumps(rows, indent=1))
print("WROTE /tmp/probe_rp_main.json")
