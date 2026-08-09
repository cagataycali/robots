"""Capture the overwrite-posture outcome on whichever tree runs this script."""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import numpy as np

import strands_robots.simulation.recording as rec_mod
from strands_robots.simulation.mujoco.simulation import Simulation

TREE = str(Path(rec_mod.__file__).parents[2])
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

_ARM = """
<mujoco model="posture_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.65 0.65 0.65"/></visual>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.86 0.87 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="3"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.26 0 0" size="0.032"
            rgba="0.16 0.42 0.78 1"/>
      <body name="tip" pos="0.26 0 0">
        <geom name="tipball" type="sphere" size="0.05" rgba="0.95 0.55 0.12 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

model = OUT / "arm.xml"
model.write_text(_ARM)


def episodes(root: Path):
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    d = json.loads(info.read_text())
    return {"episodes": d["total_episodes"], "frames": d["total_frames"]}


def build(root: Path, n_steps: int = 8, **kw):
    sim = Simulation(tool_name="posture_art", mesh=False)
    sim.create_world()
    sim.add_robot("arm", urdf_path=str(model))
    sim.add_camera(name="view", position=[0.62, -0.62, 0.5], target=[0.1, 0.0, 0.16],
                   width=320, height=240)
    started = sim.start_recording(repo_id="local/posture_art", fps=30, root=str(root), **kw)
    res = {"start_status": started["status"],
           "start_text": " ".join(c["text"] for c in started.get("content", []) if "text" in c)}
    if started["status"] == "success":
        r = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=n_steps,
                           control_frequency=30.0)
        res["rollout"] = r["status"]
        sp = sim.stop_recording()
        res["stop_status"] = sp["status"]
    frame = sim.render(camera_name="view", width=560, height=420)
    png = next(c["image"]["source"]["bytes"] for c in frame["content"] if "image" in c)
    sim.cleanup()
    return res, np.asarray(iio.imread(png))


root = OUT / "dataset"
if root.exists():
    shutil.rmtree(root)

# 1. the caller records one episode.
seed, scene = build(root, overwrite=False)
before = episodes(root)

# 2. the caller re-opens to append a SECOND, longer episode and opts out of
#    overwrite with the spelling an operator reaches for.
second, _ = build(root, n_steps=12, overwrite="false")
after = episodes(root)

# 3. whatever the second call did, follow the advice the tree gives and append
#    with the documented spelling. On main that advice never appears.
remedy = None
if second["start_status"] == "error":
    remedy_res, _ = build(root, n_steps=12, overwrite=False)
    remedy = {"result": remedy_res, "state": episodes(root)}

# 4. decode frames back out of the dataset's own MP4 (proof the recording is real)
strip = []
mp4s = sorted(root.rglob("*.mp4"))
if mp4s:
    frames = list(iio.imiter(mp4s[0]))
    for i in (0, len(frames) // 2, len(frames) - 1):
        strip.append(np.asarray(frames[i]).tolist())

np.save(OUT / "scene.npy", scene)
facts = {
    "tree": TREE,
    "seed": seed,
    "before": before,
    "second": second,
    "after": after,
    "remedy": remedy,
    "mp4_count": len(mp4s),
    "strip_len": len(strip),
    "dataset_exists": root.exists(),
}
(OUT / "facts.json").write_text(json.dumps(facts, indent=1))
if strip:
    np.save(OUT / "strip.npy", np.asarray(strip, dtype=np.uint8))
print("TREE: " + TREE)
print("FACTS " + json.dumps(facts))
