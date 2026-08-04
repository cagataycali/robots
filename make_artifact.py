"""Render what an Isaac joint-state write actually left, on both trees.

Isaac cannot render here, so the measured joint state is replayed onto a MuJoCo
arm declaring the same joint names: the picture is the measured write, not a
reconstruction.
"""
import json, math, pathlib, queue, sys, threading
import numpy as np

import strands_robots.simulation.base as _b
TREE = pathlib.Path(_b.__file__).parents[2]
print("TREE:", TREE)

from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation, _RobotState
from strands_robots import Simulation

JOINTS = ["shoulder", "elbow", "wrist"]
HOME = [0.10, 0.20, 0.30]
# The caller wants two joints posed; 'shouldre' is a typo for 'shoulder'.
REQUEST = {"shouldre": 1.15, "elbow": -1.30}
INTENDED = {"shoulder": 1.15, "elbow": -1.30, "wrist": 0.30}

ARM_XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.6 0.75" rgb2="0.85 0.9 0.95" width="64" height="64"/>
  </asset>
  <worldbody>
    <body name="post" pos="0 0 0.02">
      <geom type="cylinder" size="0.05 0.02" rgba="0.35 0.35 0.4 1"/>
    </body>
    <body name="link1" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-2.5 2.5" limited="true" damping="6"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.045" rgba="0.85 0.35 0.15 1"/>
      <body name="link2" pos="0 0 0.30">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-2.5 2.5" limited="true" damping="6"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.26" size="0.038" rgba="0.20 0.55 0.85 1"/>
        <body name="link3" pos="0 0 0.26">
          <joint name="wrist" type="hinge" axis="0 1 0" range="-2.5 2.5" limited="true" damping="6"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.030" rgba="0.25 0.70 0.35 1"/>
          <geom type="sphere" pos="0 0 0.17" size="0.05" rgba="0.95 0.85 0.20 1"/>
        </body>
      </body>
    </body>
    <camera name="look" pos="1.35 -0.05 0.55" mode="targetbody" target="link2" fovy="40"/>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="60"/>
    <position name="a_elbow" joint="elbow" kp="60"/>
    <position name="a_wrist" joint="wrist" kp="60"/>
  </actuator>
</mujoco>
"""


class FakeArticulation:
    def __init__(self):
        self.q = list(HOME)
        self.writes = []

    def get_joint_positions(self):
        return list(self.q)

    def set_joint_positions(self, arr):
        self.writes.append(list(np.asarray(arr, dtype=float).tolist()))
        self.q = list(np.asarray(arr, dtype=float).tolist())


def isaac_probe(request):
    e = IsaacSimulation.__new__(IsaacSimulation)
    e._lock = threading.RLock()
    e._world_created = True
    e._config = IsaacConfig()
    e._main_tid = threading.get_ident()
    e._action_q = queue.Queue()
    art = FakeArticulation()
    e._robots = {"arm": _RobotState(name="arm", prim_path="/World/Robots/arm",
                                    joint_names=list(JOINTS), articulation=art)}
    try:
        r = e.set_joint_positions(positions=request, robot_name="arm")
        verdict, text = r.get("status"), " ".join(c.get("text", "") for c in r.get("content", []) if "text" in c)
    except BaseException as exc:  # noqa: BLE001
        verdict, text = "raised", f"{type(exc).__name__}: {exc}"
    return {"verdict": verdict, "text": text, "joint_state": list(art.q), "writes": len(art.writes)}


def render(sim, names, pose, out):
    r = sim.set_joint_positions(positions={names[i]: pose[i] for i in range(3)})
    assert r["status"] == "success", r
    res = sim.render(camera_name="arm/look", width=640, height=480)
    assert res.get("status") == "success", res
    png = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
    pathlib.Path(out).write_bytes(png)
    import imageio.v3 as iio
    return np.asarray(iio.imread(out))


tag = sys.argv[1]
outdir = pathlib.Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
measured = isaac_probe(REQUEST)
print("isaac verdict:", measured["verdict"], "| state:", measured["joint_state"])
print("text:", measured["text"][:170])

xml = outdir / "arm.xml"; xml.write_text(ARM_XML)
sim = Simulation(backend="mujoco", mesh=False)
sim.create_world(gravity=[0, 0, 0])
sim.add_robot(name="arm", urdf_path=str(xml))
names = sim.robot_joint_names("arm")
names = names["content"][0]["json"]["joint_names"] if isinstance(names, dict) else names
print("model joint names:", names)

frames = {}
frames["home"] = render(sim, names, HOME, outdir / f"{tag}_home.png")
frames["intended"] = render(sim, names, [INTENDED[j] for j in JOINTS], outdir / f"{tag}_intended.png")
frames["result"] = render(sim, names, measured["joint_state"], outdir / f"{tag}_result.png")
sim.cleanup()

def dfrac(a, b):
    return float((np.abs(a.astype(int) - b.astype(int)).sum(2) > 12).mean())

facts = {
    "tree": str(TREE), "tag": tag, "request": REQUEST, "intended": INTENDED,
    "home": HOME, "measured": measured,
    "diff_intended_vs_result": dfrac(frames["intended"], frames["result"]),
    "diff_home_vs_result": dfrac(frames["home"], frames["result"]),
    "diff_home_vs_intended": dfrac(frames["home"], frames["intended"]),
}
(outdir / f"{tag}_facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps({k: v for k, v in facts.items() if k.startswith("diff")}, indent=1))
