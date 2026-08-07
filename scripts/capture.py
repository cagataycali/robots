"""Capture a real MuJoCo rollout per action_scale. Run on both trees."""
from __future__ import annotations
import json, math, pathlib, sys

import strands_robots.training.rl.env as envmod
TREE = str(pathlib.Path(envmod.__file__).parents[3])
print("TREE:", TREE, flush=True)

import numpy as np
import torch
from strands_robots.simulation import Simulation
from strands_robots.training.rl import SimEnv

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

ARM = """<mujoco model="probe">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65"/>
    <global offwidth="1600" offheight="1200"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.82 0.82 0.85 1"/>
    <body name="post" pos="0.42 0 0.16"><geom type="cylinder" size="0.012 0.16" rgba="0.45 0.45 0.5 1"/></body>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.055 0.1" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0.11">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.038" rgba="0.9 0.45 0.1 1"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
          <geom type="capsule" fromto="0 0 0 0.26 0 0" size="0.032" rgba="0.15 0.5 0.9 1"/>
          <body name="tip" pos="0.26 0 0"><geom type="sphere" size="0.045" rgba="0.15 0.75 0.35 1"/></body>
        </body>
      </body>
    </body>
    <camera name="look" pos="0.30 -1.05 0.72" mode="targetbody" target="post" fovy="38"/>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="60" ctrlrange="-2 2"/>
    <position name="a_elbow" joint="elbow" kp="60" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

def render(sim) -> np.ndarray:
    import io
    from PIL import Image
    r = sim.render(camera_name="arm/look", width=760, height=620)  # MJCF cameras are namespaced by add_robot
    b = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

def run(tag: str, scale) -> dict:
    tmp = pathlib.Path("/tmp/art_arm"); tmp.mkdir(exist_ok=True)
    (tmp / "arm.xml").write_text(ARM)
    sim = Simulation(backend="mujoco", tool_name=f"art_{tag}", mesh=False)
    sim.create_world(); sim.add_robot(name="arm", urdf_path=str(tmp / "arm.xml"))
    rec: dict = {"tag": tag, "scale": repr(scale), "tree": TREE}
    try:
        env = SimEnv(sim, actor_obs_keys=["shoulder", "elbow"],  # type: ignore[arg-type]
                     reward_terms=[lambda e: 1.0], robot_name="arm",
                     action_scale=scale, max_episode_steps=1000, n_substeps=10)  # type: ignore[arg-type]
    except ValueError as e:
        rec.update(ctor="REFUSED", message=str(e))
        np.save(OUT / f"{tag}.npy", render(sim))   # the untouched world
        sim.cleanup(); return rec
    rec["ctor"] = "accepted"
    statuses: list[str] = []
    real = sim.send_action
    def spy(a, robot_name=None, n_substeps=1):
        r = real(a, robot_name=robot_name, n_substeps=n_substeps)
        statuses.append(r.get("status", "?")); return r
    sim.send_action = spy  # type: ignore[method-assign]
    env.reset()
    total = 0.0
    for _ in range(60):
        _o, r, _d, _i = env.step(torch.tensor([0.9, -0.7], dtype=torch.float32))
        total += float(r.item())
    obs = sim.get_observation(robot_name="arm", skip_images=True)
    np.save(OUT / f"{tag}.npy", render(sim))
    rec.update(sends=len(statuses), ok=statuses.count("success"),
               err=sum(1 for s in statuses if s != "success"), reward=round(total, 1),
               shoulder=round(float(obs["shoulder"]), 4), elbow=round(float(obs["elbow"]), 4))
    sim.cleanup(); return rec

facts = [run("honored", 1.0), run("zero", 0), run("nan", float("nan"))]
(OUT / "facts.json").write_text(json.dumps(facts, indent=1))
for f in facts:
    print(f)
