"""Measure the nine diagnoses, then record one real dataset end to end."""
import json, os, pathlib, shutil, sys, tempfile, threading
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import strands_robots, strands_robots.dataset_recorder as dr
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.simulation.mujoco.simulation import Simulation

J = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
REASON = ("lerobot is not installed (ModuleNotFoundError: No module named 'lerobot'). "
          "Install lerobot >= 0.6.0 with: pip install 'strands-robots[lerobot]'")
facts = {"tree": TREE}

def isaac():
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._config = IsaacConfig(render_mode="rtx_realtime"); e._lock = threading.RLock()
    e._world = None; e._world_created = True
    e._robots = {"so100": _RobotState(name="so100", prim_path="/World/Robots/so100",
                 joint_names=list(J), data_config="so100")}
    e._cameras = {}; e._objects = {}; e._prim_registry = []; e._cams_rec_state = None
    e._recording_state_dict = {}; e._action_controllers = {}; e._sim_time = 0.0
    e._step_count = 0; e._replicated = False; e._num_envs_active = 1
    e._pump_running = False; e._main_tid = threading.get_ident()
    return e, None

def newton():
    w = SimWorld()
    w.robots["so100"] = SimRobot(name="so100", urdf_path="so100.xml", data_config="so100", joint_names=list(J))
    e = NewtonSimEngine.__new__(NewtonSimEngine)
    e._world = w; e._model = object(); e.default_width = 64; e.default_height = 48
    return e, None

def mujoco():
    s = Simulation(tool_name="art_probe", mesh=False); s.create_world()
    return s, s.cleanup

orig_probe, orig_cls = dr.lerobot_dataset_import_error, dr.DatasetRecorder
CAUSES = ["absent-lerobot-extra", "module-did-not-import", "module-supplied-no-recorder"]

def diagnose(factory, cause, root):
    dr.lerobot_dataset_import_error, dr.DatasetRecorder = orig_probe, orig_cls
    if cause == "absent-lerobot-extra":
        dr.lerobot_dataset_import_error = lambda: REASON
    elif cause == "module-did-not-import":
        del dr.DatasetRecorder
    else:
        dr.lerobot_dataset_import_error = lambda: None; dr.DatasetRecorder = None
    eng, closer = factory()
    try:
        r = eng.start_recording(repo_id="local/art_probe", root=str(root))
        txt = r["content"][0]["text"]
        marker = next((l.strip() for l in txt.splitlines()
                       if l.strip().startswith(("lerobot is not", "strands_robots.dataset_recorder"))), "")
        fb = "start_cameras_recording" if "start_cameras_recording" in txt else "run_policy(video=...)"
        return {"status": r["status"], "marker": marker, "fallback": fb,
                "session_open": bool(eng._is_recording()), "root_created": root.exists()}
    finally:
        if closer: closer()
        dr.lerobot_dataset_import_error, dr.DatasetRecorder = orig_probe, orig_cls

tmp = pathlib.Path(tempfile.mkdtemp(prefix="art-diag-"))
facts["diagnoses"] = {}
for name, fac in [("mujoco", mujoco), ("newton", newton), ("isaac", isaac)]:
    facts["diagnoses"][name] = {c: diagnose(fac, c, tmp / f"{name}-{c}") for c in CAUSES}
shutil.rmtree(tmp, ignore_errors=True)

# ---- the honored path: one real dataset, recorded and read back -------------
ARM = """<mujoco model="arm">
  <compiler angle="radian"/>
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light pos="0.4 -0.4 0.8"/>
    <geom type="plane" size="2 2 0.05" rgba="0.55 0.57 0.6 1"/>
    <body name="base" pos="0 0 0.02">
      <geom type="cylinder" size="0.035 0.02" rgba="0.3 0.32 0.36 1"/>
      <body name="link1" pos="0 0 0.02">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" damping="3"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.026" rgba="0.22 0.62 0.86 1"/>
        <body name="link2" pos="0.14 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.6 1.6" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0.11 0 0" size="0.022" rgba="0.98 0.62 0.15 1"/>
          <site name="tip" pos="0.11 0 0" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="26"/>
    <position name="a_elbow" joint="elbow" kp="26"/>
  </actuator>
</mujoco>
"""
work = pathlib.Path(tempfile.mkdtemp(prefix="art-rec-"))
(work / "arm.xml").write_text(ARM)
sim = Simulation(tool_name="art_record", mesh=False)
sim.create_world()
sim.add_robot(name="arm", urdf_path=str(work / "arm.xml"))
sim.add_camera(name="look", position=[0.30, -0.28, 0.24], target=[0.10, 0.0, 0.05], fov=42)
root = work / "dataset"
FPS = 20
start = sim.start_recording(repo_id="local/art_honored", root=str(root), task="reach", fps=FPS)
rp = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=24,
                    control_frequency=float(FPS), action_horizon=1)
stop = sim.stop_recording()
r = sim.render(camera_name="look", width=680, height=560)
assert r.get("status") == "success", r
png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
(pathlib.Path("_art") / "scene.png").write_bytes(png)
sim.cleanup()

info = json.loads((root / "meta" / "info.json").read_text())
mp4s = sorted(p for p in root.rglob("*.mp4"))
frames = []
if mp4s:
    import imageio.v3 as iio
    frames = list(iio.imiter(mp4s[0]))
    np.save("_art/frame.npy", frames[len(frames) // 2])
facts["honored"] = {
    "start_status": start["status"], "rollout_status": rp["status"], "stop_status": stop["status"],
    "episodes": info.get("total_episodes"), "frames": info.get("total_frames"), "fps": info.get("fps"),
    "mp4_count": len(mp4s), "decoded_frames": len(frames),
}
shutil.rmtree(work, ignore_errors=True)
pathlib.Path(f"_art/facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts["honored"], indent=2))
for b, per in facts["diagnoses"].items():
    for c, d in per.items():
        print(f"{b:7s} {c:26s} {d['status']:6s} fb={d['fallback']:24s} | {d['marker'][:60]}")
