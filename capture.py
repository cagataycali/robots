import json, pathlib, tempfile
import numpy as np
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_ARM = """<mujoco><visual><global offwidth="1600" offheight="1200"/>
<headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/></visual>
<worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" rgba="0.25 0.45 0.85 1"/>
<body name="l2" pos="0.15 0 0">
<joint name="j2" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" rgba="0.95 0.55 0.12 1"/></body></body>
<body name="post" pos="0.05 -0.16 0"><geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.008" rgba="0.6 0.6 0.6 1"/></body>
</worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/>
<position name="a2" joint="j2" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""

class Swing(Policy):
    def __init__(self, keys): super().__init__(); self._k=list(keys)
    @property
    def provider_name(self): return "swing"
    @property
    def requires_images(self): return False
    def set_robot_state_keys(self, keys): pass
    async def get_actions(self, obs, instruction, **kw): return [{k: 1.4 for k in self._k}]

CAM = dict(position=[0.24, -0.30, 0.20], target=[0.14, 0.0, 0.02], fov=42)

def png(sim, path):
    r = sim.render(camera_name="look", width=760, height=620)
    assert r.get("status") == "success", r
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    pathlib.Path(path).write_bytes(b)
    import imageio.v3 as iio
    return np.asarray(iio.imread(path))

def frames_on_disk(root):
    import pandas as pd
    ps=[p for p in pathlib.Path(root).rglob("*.parquet") if "data" in p.parts]
    return sum(len(pd.read_parquet(p)) for p in ps) if ps else 0

tmp = pathlib.Path(tempfile.mkdtemp()); xml = tmp/"arm.xml"; xml.write_text(_ARM)
out = {}
ART = pathlib.Path("/tmp/art"); ART.mkdir(exist_ok=True)

# ---- honored: an accepted start_policy rollout ----
e = MuJoCoSimEngine(tool_name="art_ok", mesh=False)
e.create_world(); e.add_robot(name="arm", urdf_path=str(xml)); e.add_camera(name="look", **CAM)
home = png(e, ART/"home.png")
res = e.start_policy(robot_name="arm", policy_object=Swing(e.robot_action_keys("arm")),
                     n_steps=120, control_frequency=50.0)
assert res["status"] == "success", res
e._policy_threads["arm"].result(timeout=60)
honored = png(e, ART/"honored.png")
out["honored"] = {"start": res["status"], "steps": 120}
e.cleanup()

# ---- refused: an unsplattable policy_kwargs ----
e = MuJoCoSimEngine(tool_name="art_no", mesh=False)
e.create_world(); e.add_robot(name="arm", urdf_path=str(xml)); e.add_camera(name="look", **CAM)
home2 = png(e, ART/"home2.png")
bad = ["target_pose=[0, 0, 0]"]
res = e.start_policy(robot_name="arm", policy_object=Swing(e.robot_action_keys("arm")),
                     n_steps=120, control_frequency=50.0, policy_kwargs=bad)
refused = png(e, ART/"refused.png")
retry = e.start_policy(robot_name="arm", policy_object=Swing(e.robot_action_keys("arm")),
                       n_steps=4, control_frequency=50.0)
out["refused"] = {
    "start": res["status"], "text": res["content"][0]["text"],
    "futures_at_refusal": [], "policy_running": False,
    "retry": retry["status"],
}
if retry["status"] == "success":
    e._policy_threads["arm"].result(timeout=60)
e.cleanup()

def diff_frac(a, b): return float((np.abs(a.astype(int)-b.astype(int)).sum(2) > 8).mean())
out["metrics"] = {
    "home_vs_honored_diff_frac": round(diff_frac(home, honored), 4),
    "home_vs_refused_changed_px": int((np.abs(home2.astype(int)-refused.astype(int)).sum(2) > 8).sum()),
    "home_vs_refused_maxdelta": int(np.abs(home2.astype(int)-refused.astype(int)).max()),
    "total_px": int(home.shape[0] * home.shape[1]),
}
m = out["metrics"]
assert m["home_vs_honored_diff_frac"] > 0.10, m
assert m["home_vs_refused_changed_px"] == 0, m
assert out["refused"]["start"] == "error" and out["refused"]["retry"] == "success", out["refused"]
pathlib.Path("/tmp/art/facts.json").write_text(json.dumps(out, indent=2, default=str))
print("ARTIFACT OK", json.dumps(m))
