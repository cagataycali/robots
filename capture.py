import json, pathlib, shutil, sys
import numpy as np
import imageio.v3 as iio
import strands_robots.tools.run_policy as rp_mod
TREE = str(pathlib.Path(rp_mod.__file__).parents[2])
print("TREE:", TREE, flush=True)
from strands_robots import Simulation

ARM = """<mujoco model="probe">
  <compiler angle="radian"/>
  <visual><global offwidth="1280" offheight="960"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.5 0.65 0.85" rgb2="0.2 0.3 0.45" width="8" height="64"/></asset>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.024" rgba="0.85 0.32 0.10 1"/>
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="2" range="-1.5 1.5" limited="true"/>
      <body name="link" pos="0 0 0.2">
        <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.020" rgba="0.20 0.55 0.85 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" damping="2" range="-1.5 1.5" limited="true"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30"/>
    <position name="a_elbow" joint="elbow" kp="30"/>
  </actuator>
</mujoco>"""

tag = sys.argv[1]
TMP = pathlib.Path(f"/tmp/art_rp_{tag}"); shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
xml = TMP / "arm.xml"; xml.write_text(ARM)
root = TMP / "ds"

def truth():
    info = root / "meta" / "info.json"
    if not info.exists():
        return {"present": False, "eps": None, "frames": None}
    d = json.loads(info.read_text())
    return {"present": True, "eps": d.get("total_episodes"), "frames": d.get("total_frames")}

def mp4s():
    return sorted(str(p) for p in root.rglob("*.mp4"))

def first_frame():
    files = mp4s()
    if not files:
        return None
    for f in iio.imiter(files[0]):
        return np.asarray(f)
    return None

def sim():
    s = Simulation(backend="mujoco", mesh=False)
    s.create_world()
    s.add_robot(name="arm", urdf_path=str(xml))
    return s

facts = {"tree": TREE}

# 1. the caller records one episode
s = sim()
seed_res = rp_mod.run_policy(s, robot_name="arm", n_episodes=1, n_steps=8,
                             control_frequency=30.0, dataset_fps=30,
                             dataset_root=str(root), dataset_repo_id="local/art",
                             dataset_task="reach")
s.cleanup()
facts["seed_status"] = seed_res.get("status")
facts["before"] = truth()
facts["before_mp4s"] = len(mp4s())
f_before = first_frame()
assert f_before is not None, "no MP4 produced by the seed recording"
np.save(f"/tmp/art_frame_before_{tag}.npy", f_before)

# 2. the caller re-opens the same dataset with an unusable control_frequency
s = sim()
res = rp_mod.run_policy(s, robot_name="arm", n_episodes=1, n_steps=8,
                        control_frequency=0.0, dataset_fps=30,
                        dataset_root=str(root), dataset_repo_id="local/art",
                        dataset_task="reach")
s.cleanup()
facts["status"] = res.get("status")
facts["summary"] = (res.get("content") or [{}])[0].get("text", "")
eps = next((b["json"] for b in res.get("content") or [] if "json" in b), {}).get("episodes", [])
facts["ep0_text"] = eps[0].get("text", "") if eps else ""
facts["after"] = truth()
facts["after_mp4s"] = len(mp4s())
f_after = first_frame()
if f_after is not None:
    np.save(f"/tmp/art_frame_after_{tag}.npy", f_after)
facts["after_frame_decodable"] = f_after is not None

pathlib.Path(f"/tmp/art_facts_{tag}.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1))
